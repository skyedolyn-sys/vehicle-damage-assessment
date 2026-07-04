"""DAMAGE_RECOGNITION_POLICY §4.3 — topology_comparator rule-trigger logging.

These tests pin the wiring of ``log_policy_conflict`` inside
``agents.topology_comparator``: every branch that flips a conclusion
(status / damage_level / confidence) must emit a structured entry that
contains ``part_id``, ``final_status``, ``conflict_sources`` and a
``rule=`` tag identifying the rule that fired.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

import pytest

from agents import _policy_logger, topology_comparator
from agents._policy_logger import log_policy_conflict
from models.part_state import DamageLevel, PartActualState, Status
from models.topology import TopologyNode, VehicleTopology


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def isolated_log_dir(monkeypatch, tmp_path):
    """Point the policy logger at ``tmp_path`` and reset its handlers."""
    monkeypatch.setattr(_policy_logger, "_LOG_DIR", tmp_path)
    _policy_logger._logger.handlers.clear()
    handler = logging.FileHandler(
        tmp_path / "policy_conflicts.log", mode="a", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _policy_logger._logger.addHandler(handler)
    _policy_logger._logger.setLevel(logging.INFO)
    yield tmp_path
    _policy_logger._logger.handlers.clear()


def _read_log(path: Path) -> str:
    return (path / "policy_conflicts.log").read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Direct logger import smoke test (from spec)
# ----------------------------------------------------------------------

class TestDirectLoggerHookup:
    """Spec smoke test: ensure ``_LOG_DIR`` is a Path-like and the helper works."""

    def test_log_dir_is_path(self):
        from agents._policy_logger import _LOG_DIR
        assert hasattr(_LOG_DIR, "exists") or hasattr(_LOG_DIR, "mkdir"), (
            "_LOG_DIR must be a Path"
        )

    def test_log_policy_conflict_writes_file(self, isolated_log_dir):
        log_policy_conflict(
            part_id="test_part",
            final_status="damaged",
            conflict_sources=["neighbor1", "neighbor2"],
            rule_applied="test_rule",
        )
        log_file = isolated_log_dir / "policy_conflicts.log"
        assert log_file.exists()
        text = log_file.read_text(encoding="utf-8")
        assert "test_part" in text
        assert "test_rule" in text


class TestTopologyComparatorImportsLogger:
    """Spec smoke test: topology_comparator must import + use the helper."""

    def test_module_imports_log_policy_conflict(self):
        src = inspect.getsource(topology_comparator)
        assert "log_policy_conflict" in src
        assert "from agents._policy_logger" in src


# ----------------------------------------------------------------------
# Topology builders per rule
# ----------------------------------------------------------------------

def _build_two_part_topology(part_a_id: str, part_b_id: str, side_a: str = "front",
                             side_b: str = "front") -> VehicleTopology:
    """Minimal two-node topology where A is adjacent to B."""
    nodes = {
        part_a_id: TopologyNode(
            node_id=part_a_id, part_id=part_a_id, node_name=part_a_id,
            node_type="panel", region="front", side=side_a,
            adjacent_nodes=[part_b_id],
        ),
        part_b_id: TopologyNode(
            node_id=part_b_id, part_id=part_b_id, node_name=part_b_id,
            node_type="panel", region="front", side=side_b,
            adjacent_nodes=[part_a_id],
        ),
    }
    return VehicleTopology(
        vehicle_id="t-policy",
        vehicle_name="TopoPolicyTest",
        nodes=nodes,
        regions={"front": [part_a_id, part_b_id]},
    )


def _make_state(
    part_id: str, status: Status = Status.INTACT,
    damage_level: DamageLevel = DamageLevel.NONE,
    confidence: str = "high", evidence: list | None = None,
    side: str = "center", part_name: str | None = None,
) -> PartActualState:
    return PartActualState(
        part_id=part_id,
        part_name=part_name or part_id,
        part_category="front",
        side=side,
        status=status,
        damage_level=damage_level,
        standard_exists=True,
        actual_visible=True,
        actual_present=True,
        confidence=confidence,
        evidence_sources=evidence or [{"view_id": "front", "status": status.value}],
    )


# ----------------------------------------------------------------------
# Rule-by-rule integration tests
# ----------------------------------------------------------------------

class TestRule1LogsConflict:
    """Rule 1: intact high-confidence part next to damaged neighbour -> cap to medium."""

    def test_intact_next_to_damaged_caps_confidence_and_logs(self, isolated_log_dir):
        topology = _build_two_part_topology("door_front_left", "fender_front_left")
        door = _make_state("door_front_left", confidence="high",
                           evidence=[{"view_id": "side_left", "status": "intact"}])
        fender = _make_state("fender_front_left", Status.DAMAGED,
                             DamageLevel.MODERATE, "medium",
                             evidence=[{"view_id": "front_left_45", "status": "damaged"}])
        topology_comparator.compare_topology(topology, [door, fender])

        text = _read_log(isolated_log_dir)
        assert "part_id=door_front_left" in text
        assert "rule=topology_rule_1" in text
        assert "fender_front_left" in text


class TestRule3LogsConflict:
    """Rule 3: intact door next to severely damaged neighbour -> cap to low."""

    def test_door_intact_next_to_severe_caps_confidence_and_logs(self, isolated_log_dir):
        topology = _build_two_part_topology("door_front_left", "fender_front_left")
        door = _make_state("door_front_left", confidence="high",
                           evidence=[{"view_id": "side_left", "status": "intact"}])
        fender = _make_state("fender_front_left", Status.DAMAGED,
                             DamageLevel.SEVERE, "high",
                             evidence=[{"view_id": "front_left_45", "status": "damaged"}])
        topology_comparator.compare_topology(topology, [door, fender])

        text = _read_log(isolated_log_dir)
        assert "part_id=door_front_left" in text
        assert "rule=topology_rule_3" in text
        assert "fender_front_left" in text


class TestRule10LogsConflict:
    """Rule 10: roof_front + severe front structural -> damaged severe."""

    def test_roof_front_intact_with_severe_pillar_flips_and_logs(self, isolated_log_dir):
        topology = _build_two_part_topology("roof_front", "pillar_a_left")
        roof = _make_state("roof_front", Status.INTACT, DamageLevel.NONE,
                           "medium",
                           evidence=[{"view_id": "front", "status": "intact"}])
        pillar = _make_state("pillar_a_left", Status.DAMAGED, DamageLevel.SEVERE,
                             "high",
                             evidence=[{"view_id": "front_left_45", "status": "damaged"}],
                             side="front_left")
        topology_comparator.compare_topology(topology, [roof, pillar])

        text = _read_log(isolated_log_dir)
        assert "part_id=roof_front" in text
        assert "rule=topology_rule_10" in text
        assert "pillar_a_left" in text


class TestRule11LogsConflict:
    """Rule 11: roof_middle/rear + severe neighbour -> damaged severe."""

    def test_roof_middle_uncertain_with_severe_roof_front_flips_and_logs(self, isolated_log_dir):
        topology = _build_two_part_topology("roof_middle", "roof_front")
        roof_middle = _make_state("roof_middle", Status.UNCERTAIN, DamageLevel.UNKNOWN,
                                  "low", evidence=[{"view_id": "side", "status": "uncertain"}])
        roof_front = _make_state("roof_front", Status.DAMAGED, DamageLevel.SEVERE,
                                 "high",
                                 evidence=[{"view_id": "front_left_45", "status": "damaged"}])
        topology_comparator.compare_topology(topology, [roof_middle, roof_front])

        text = _read_log(isolated_log_dir)
        assert "part_id=roof_middle" in text
        assert "rule=topology_rule_11" in text
        assert "roof_front" in text


class TestMissingRoofInferenceLogsConflict:
    """Missing-roof inference: uncertain roof + intact neighbours -> intact."""

    def test_roof_rack_uncertain_with_intact_neighbour_logs(self, isolated_log_dir):
        # Roof rack and roof_front are adjacent in the roof family. The
        # neighbour must be intact with medium+ confidence AND have a direct
        # top-view observation, otherwise the rule does not fire.
        topology = _build_two_part_topology("roof_rack", "roof_front")
        rack = _make_state("roof_rack", Status.UNCERTAIN, DamageLevel.UNKNOWN,
                           "low", evidence=[{"view_id": "side", "status": "uncertain"}])
        front = _make_state("roof_front", Status.INTACT, DamageLevel.NONE, "high",
                            evidence=[{"view_id": "top", "status": "intact"}])
        topology_comparator.compare_topology(topology, [rack, front])

        text = _read_log(isolated_log_dir)
        assert "part_id=roof_rack" in text
        assert "rule=topology_rule_missing_roof_inference" in text
        assert "roof_front" in text