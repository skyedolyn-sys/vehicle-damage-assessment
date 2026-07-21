"""Tests for DAMAGE_RECOGNITION_POLICY §4.3 policy conflict logging.

Covers:
1. ``log_policy_conflict`` writes to the configured log file.
2. Log line format includes part_id, final_status, conflict_sources, rule.
3. Integration: a synthesizer run that triggers Rule 6 / Rule 8 produces
   at least one entry in the policy conflict log.
4. Re-importing the module does not add duplicate file handlers.
"""
from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path

import pytest

from agents import _policy_logger
from agents.synthesizer import synthesizer_agent
from models.topology import TopologyNode, VehicleTopology


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def isolated_log_dir(monkeypatch, tmp_path):
    """Point the policy logger at a tmp dir and reset its handler set."""
    monkeypatch.setattr(_policy_logger, "_LOG_DIR", tmp_path)
    # Drop any existing handlers so the new path is used cleanly.
    _policy_logger._logger.handlers.clear()
    # Re-add a fresh handler at the new path.
    handler = logging.FileHandler(tmp_path / "policy_conflicts.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _policy_logger._logger.addHandler(handler)
    _policy_logger._logger.setLevel(logging.INFO)
    yield tmp_path
    _policy_logger._logger.handlers.clear()


def _read_log(path: Path) -> str:
    return path.joinpath("policy_conflicts.log").read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Direct logger tests
# ----------------------------------------------------------------------

class TestLogPolicyConflictDirect:
    def test_writes_to_configured_dir(self, isolated_log_dir):
        _policy_logger.log_policy_conflict(
            part_id="door_front_left",
            final_status="intact",
            conflict_sources=["fender_front_left"],
            rule_applied="adjacency_rule_6",
        )
        text = _read_log(isolated_log_dir)
        assert "part_id=door_front_left" in text
        assert "rule=adjacency_rule_6" in text

    def test_message_includes_all_required_fields(self, isolated_log_dir):
        _policy_logger.log_policy_conflict(
            part_id="fender_rear_right",
            final_status="damaged",
            conflict_sources=["bumper_rear", "trunk_lid"],
            rule_applied="adjacency_rule_8",
        )
        text = _read_log(isolated_log_dir)
        # All four required fields must be present.
        assert "part_id=fender_rear_right" in text
        assert "final_status=damaged" in text
        assert "conflict_sources=" in text
        assert "rule=adjacency_rule_8" in text
        # The conflict sources themselves must appear.
        assert "bumper_rear" in text
        assert "trunk_lid" in text

    def test_conflict_sources_rendered_as_list(self, isolated_log_dir):
        _policy_logger.log_policy_conflict(
            part_id="x",
            final_status="intact",
            conflict_sources=["a", "b", "c"],
            rule_applied="test_rule",
        )
        text = _read_log(isolated_log_dir)
        # Default %s formatting of a list wraps it in [...]
        assert re.search(r"conflict_sources=\['a', 'b', 'c'\]", text)


class TestHandlerDedup:
    def test_reimport_does_not_duplicate_handlers(self, tmp_path, monkeypatch):
        """Re-importing _policy_logger should not stack file handlers."""
        monkeypatch.setattr(_policy_logger, "_LOG_DIR", tmp_path)
        _policy_logger._logger.handlers.clear()

        importlib.reload(_policy_logger)
        handlers_after_first = [
            h for h in _policy_logger._logger.handlers
            if isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "").endswith("policy_conflicts.log")
        ]
        assert len(handlers_after_first) == 1

        # Re-import a second time: handler count must stay the same.
        importlib.reload(_policy_logger)
        handlers_after_second = [
            h for h in _policy_logger._logger.handlers
            if isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "").endswith("policy_conflicts.log")
        ]
        assert len(handlers_after_second) == 1


# ----------------------------------------------------------------------
# Integration: synthesizer triggers a log entry
# ----------------------------------------------------------------------

def _build_rear_right_topology() -> VehicleTopology:
    """Topology where Rule 8 (rear-side inference) can fire."""
    nodes = {
        "taillight_rear_right": TopologyNode(
            node_id="taillight_rear_right", part_id="taillight_rear_right",
            node_name="右后尾灯", node_type="light", region="rear", side="rear_right",
            adjacent_nodes=["bumper_rear", "fender_rear_right"],
        ),
        "bumper_rear": TopologyNode(
            node_id="bumper_rear", part_id="bumper_rear", node_name="后保险杠",
            node_type="panel", region="rear", side="center",
            adjacent_nodes=["taillight_rear_right", "fender_rear_right", "trunk_lid"],
        ),
        "trunk_lid": TopologyNode(
            node_id="trunk_lid", part_id="trunk_lid", node_name="后备箱盖",
            node_type="panel", region="rear", side="center",
            adjacent_nodes=["bumper_rear", "fender_rear_right", "taillight_rear_right"],
        ),
        "fender_rear_right": TopologyNode(
            node_id="fender_rear_right", part_id="fender_rear_right", node_name="右后翼子板",
            node_type="panel", region="right", side="rear_right",
            adjacent_nodes=["bumper_rear", "taillight_rear_right", "trunk_lid"],
        ),
    }
    return VehicleTopology(
        vehicle_id="sedan-001",
        vehicle_name="Test Sedan",
        nodes=nodes,
        regions={
            "rear": ["taillight_rear_right", "bumper_rear", "trunk_lid"],
            "right": ["fender_rear_right"],
        },
    )


class TestSynthesizerLogsConflict:
    def test_rule_8_fires_and_logs(self, isolated_log_dir, monkeypatch):
        """fender_rear_right uncertain + severe rear-core neighbors -> Rule 8 fires
        and writes a policy_conflicts.log entry."""
        topology = _build_rear_right_topology()
        region_results = [
            {
                "region": "rear_right",
                "parts": [
                    {
                        "part_id": "taillight_rear_right",
                        "part_name": "右后尾灯",
                        "part_category": "rear",
                        "side": "rear_right",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["missing"],
                        # high confidence: a credible severe neighbour.  Rule 8
                        # only propagates from credible (non-low-confidence)
                        # severe neighbours after the 172852 FP fix, so the
                        # fixture must use high to exercise the rule.
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "损毁",
                    },
                    {
                        "part_id": "bumper_rear",
                        "part_name": "后保险杠",
                        "part_category": "rear",
                        "side": "center",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["tear"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "撕裂",
                    },
                    {
                        "part_id": "fender_rear_right",
                        "part_name": "右后翼子板",
                        "part_category": "right",
                        "side": "rear_right",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": "遮挡",
                    },
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results, topology=topology)
        # Sanity: the rule actually flipped the part.
        fender = next(p for p in result["parts"] if p["part_id"] == "fender_rear_right")
        assert fender["status"] == "damaged"
        assert fender["damage_level"] == "severe"

        text = _read_log(isolated_log_dir)
        # At least one log entry was written.
        assert "part_id=fender_rear_right" in text
        assert "rule=adjacency_rule_8" in text
        # Conflict sources should include the severe neighbors.
        assert "bumper_rear" in text
        assert "taillight_rear_right" in text


# ----------------------------------------------------------------------
# Integration: topology comparator triggers a log entry
# ----------------------------------------------------------------------

def _build_pillar_topology() -> VehicleTopology:
    """Minimal topology for Rule 9 (pillar-to-roof propagation)."""
    nodes = {
        "windshield_front": TopologyNode(
            node_id="windshield_front", part_id="windshield_front",
            node_name="前挡风玻璃", node_type="glass", region="front", side="center",
            adjacent_nodes=["pillar_a_left"],
        ),
        "pillar_a_left": TopologyNode(
            node_id="pillar_a_left", part_id="pillar_a_left", node_name="左A柱",
            node_type="structural", region="left", side="front_left",
            adjacent_nodes=["windshield_front"],
        ),
    }
    return VehicleTopology(
        vehicle_id="test-001",
        vehicle_name="Test",
        nodes=nodes,
        regions={"front": ["windshield_front"], "left": ["pillar_a_left"]},
    )


class TestTopologyLogsConflict:
    def test_rule_9_fires_and_logs(self, isolated_log_dir):
        from agents.topology_comparator import compare_topology
        from models.part_state import DamageLevel, PartActualState, Status

        topology = _build_pillar_topology()
        actual = [
            PartActualState(
                part_id="windshield_front", part_name="前挡风玻璃",
                part_category="front", side="center",
                status=Status.DAMAGED, damage_level=DamageLevel.SEVERE,
                evidence_sources=[{"view_id": "front", "status": "damaged"}],
            ),
            PartActualState(
                part_id="pillar_a_left", part_name="左A柱",
                part_category="left", side="front_left",
                status=Status.UNCERTAIN, damage_level=DamageLevel.UNKNOWN,
                evidence_sources=[{"view_id": "front_left_45", "status": "uncertain"}],
            ),
        ]
        assessment = compare_topology(topology, actual)
        pillar = next(p for p in assessment.parts if p.part_id == "pillar_a_left")
        assert pillar.status == Status.DAMAGED
        assert pillar.damage_level == DamageLevel.SEVERE

        text = _read_log(isolated_log_dir)
        assert "part_id=pillar_a_left" in text
        assert "rule=topology_rule_9" in text
        assert "windshield_front" in text
