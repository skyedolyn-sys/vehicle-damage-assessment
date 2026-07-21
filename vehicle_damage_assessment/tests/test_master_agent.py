import pytest
from unittest.mock import AsyncMock, patch

from agents.master_agent import (
    _extract_photo_classifications,
    _aggregate_part_evidence,
    _apply_facing_consensus,
    _part_side,
    _boost_confidence,
    _build_region_results,
    master_assessment_agent,
)
from models.part_state import PartActualState, Status


def _prior(pid, facing="front", camera_side=None, usable=True):
    """Minimal face_prior for facing-consensus tests."""
    return {
        "photo_id": pid,
        "facing": facing,
        "camera_side": camera_side,
        "usable": usable,
    }


def _view_result(pid, damaged_part_ids):
    """Minimal view_result with the given damaged part ids."""
    return {
        "photo_id": pid,
        "parts": [
            {"part_id": p, "status": "damaged", "confidence": "high",
             "model_confidence_score": 0.85}
            for p in damaged_part_ids
        ],
    }


def test_part_side():
    assert _part_side("door_front_left") == "left"
    assert _part_side("pillar_a_right") == "right"
    assert _part_side("hood") == "center"
    assert _part_side("windshield_front") == "center"


def test_facing_consensus_downgrades_contradictory_low_conf_damage():
    # High-conf photos put their damage on the RIGHT side.
    priors = {
        "a": _prior("a", camera_side="right", usable=True),
        "b": _prior("b", camera_side="right", usable=True),
        # low-conf close-up: its (mis-faced) damage landed on the LEFT side.
        "c": _prior("c", facing="unclear", camera_side=None, usable=False),
    }
    results = [
        _view_result("a", ["pillar_a_right", "door_front_right"]),
        _view_result("b", ["fender_front_right"]),
        _view_result("c", ["pillar_a_left"]),  # contradicts consensus side
    ]
    _apply_facing_consensus(priors, results)
    c_obs = results[2]["parts"][0]
    assert c_obs["confidence"] == "low"
    assert c_obs["_consensus_downgraded"] is True


def test_facing_consensus_keeps_consistent_low_conf_damage():
    # Consensus side is right; the low-conf photo's right-side damage is
    # self-consistent, so it is rescued (NOT downgraded).
    priors = {
        "a": _prior("a", camera_side="right", usable=True),
        "c": _prior("c", facing="unclear", camera_side=None, usable=False),
    }
    results = [
        _view_result("a", ["pillar_a_right"]),
        _view_result("c", ["door_rear_right"]),  # consistent with consensus
    ]
    _apply_facing_consensus(priors, results)
    c_obs = results[1]["parts"][0]
    assert c_obs["confidence"] == "high"
    assert "_consensus_downgraded" not in c_obs


def test_facing_consensus_ignores_center_parts():
    priors = {
        "a": _prior("a", camera_side="right", usable=True),
        "c": _prior("c", facing="unclear", camera_side=None, usable=False),
    }
    results = [
        _view_result("a", ["pillar_a_right"]),
        _view_result("c", ["hood", "windshield_front"]),  # center: no side signal
    ]
    _apply_facing_consensus(priors, results)
    for obs in results[1]["parts"]:
        assert obs["confidence"] == "high"
        assert "_consensus_downgraded" not in obs


def test_facing_consensus_leaves_usable_photos_alone():
    priors = {
        "a": _prior("a", camera_side="right", usable=True),
        # usable photo even though its damage is on the other side — trusted.
        "b": _prior("b", camera_side="left", usable=True),
    }
    results = [
        _view_result("a", ["pillar_a_right"]),
        _view_result("b", ["door_front_left"]),
    ]
    _apply_facing_consensus(priors, results)
    assert results[1]["parts"][0]["confidence"] == "high"


def test_extract_photo_classifications_new_schema():
    plan = {
        "photo_classifications": [
            {"photo_id": "a", "category": "exterior"},
            {"photo_id": "b", "category": "interior"},
        ]
    }
    result = _extract_photo_classifications(plan)
    assert result == {"a": "exterior", "b": "interior"}


def test_extract_photo_classifications_legacy_view_groups():
    plan = {
        "view_groups": {
            "front": [{"id": "a"}],
            "interior": [{"id": "b"}],
            "auxiliary": [{"id": "c"}],
        }
    }
    result = _extract_photo_classifications(plan)
    assert result["a"] == "exterior"
    assert result["b"] == "interior"
    assert result["c"] == "vehicle_info"


def test_aggregate_part_evidence_consensus_boosts_confidence():
    view_results = [
        {
            "photo_id": "p1",
            "primary_view": "front",
            "parts": [
                {
                    "part_id": "hood",
                    "status": "damaged",
                    "damage_level": "moderate",
                    "damage_types": ["dent"],
                    "confidence": "medium",
                    "description": "凹陷",
                },
            ],
        },
        {
            "photo_id": "p2",
            "primary_view": "front_left",
            "parts": [
                {
                    "part_id": "hood",
                    "status": "damaged",
                    "damage_level": "moderate",
                    "damage_types": ["dent"],
                    "confidence": "medium",
                    "description": "凹陷",
                },
            ],
        },
        {
            "photo_id": "p3",
            "primary_view": "front_right",
            "parts": [
                {
                    "part_id": "hood",
                    "status": "damaged",
                    "damage_level": "moderate",
                    "damage_types": ["dent"],
                    "confidence": "low",
                    "description": "凹陷",
                },
            ],
        },
    ]
    evidence = _aggregate_part_evidence(view_results)
    hood = evidence["hood"]
    assert hood["aggregated_status"] == "damaged"
    assert hood["aggregated_confidence"] == "high"
    assert hood["conflicting"] is False


def test_aggregate_part_evidence_conflict_detected():
    view_results = [
        {
            "photo_id": "p1",
            "primary_view": "front",
            "parts": [
                {"part_id": "hood", "status": "intact", "damage_level": "none", "damage_types": ["none"], "confidence": "high", "description": "平整"},
            ],
        },
        {
            "photo_id": "p2",
            "primary_view": "front_left",
            "parts": [
                {"part_id": "hood", "status": "damaged", "damage_level": "moderate", "damage_types": ["dent"], "confidence": "high", "description": "凹陷"},
            ],
        },
    ]
    evidence = _aggregate_part_evidence(view_results)
    assert evidence["hood"]["conflicting"] is True
    assert evidence["hood"]["aggregated_status"] == "damaged"


def test_boost_confidence_schema():
    assert _boost_confidence("low", 3) == "high"
    assert _boost_confidence("medium", 2) == "medium"
    assert _boost_confidence("low", 1) == "low"


def test_build_region_results_groups_by_primary_view():
    view_results = [
        {
            "photo_id": "p1",
            "primary_view": "front",
            "parts": [
                {"part_id": "hood", "part_name": "引擎盖", "status": "intact", "damage_level": "none", "damage_types": ["none"], "confidence": "high", "description": "平整"},
            ],
        },
    ]
    evidence = _aggregate_part_evidence(view_results)
    region_results = _build_region_results(view_results, evidence)
    assert len(region_results) == 1
    assert region_results[0]["region"] == "front"
    assert region_results[0]["parts"][0]["part_id"] == "hood"


def test_aggregate_part_evidence_keeps_backfilled_absent_observations_in_evidence():
    """Backfill ``absent`` observations are KEPT in aggregator evidence but
    tagged with ``_backfill=True`` so the synthesizer / topology layer can
    recognize them as "no information".

    Why we don't drop them at the aggregator layer: the synth + topology
    layers already have well-tested filters for backfilled rows
    (observed=False on evidence_sources, ``_is_backfilled_uncertain`` on
    candidates).  Dropping them here would let a single-source
    high-confidence damaged observation on a candidate part convict
    without any cross-photo counterweight, surfacing new FPs (172852 E2E:
    damaged 11 → 16 after the drop).  The aggregator must keep the row
    but the row carries the backfill marker.
    """
    view_results = [
        {
            "photo_id": "p1",
            "primary_view": "front",
            "parts": [
                {
                    "part_id": "roof_middle",
                    "status": "damaged",
                    "damage_level": "severe",
                    "damage_types": ["deformation"],
                    "confidence": "high",
                    "description": "车顶中部塌陷",
                },
            ],
        },
        {
            "photo_id": "p2",
            "primary_view": "rear",
            "parts": [
                {
                    "part_id": "roof_middle",
                    "status": "absent",          # sentinel (was "uncertain")
                    "damage_level": "unknown",
                    "damage_types": ["none"],
                    "model_confidence_score": 0.0,
                    "confidence": "low",
                    "_backfill": True,
                    "description": "backfill placeholder",
                },
            ],
        },
    ]

    evidence = _aggregate_part_evidence(view_results)
    assert "roof_middle" in evidence
    entry = evidence["roof_middle"]
    # Both observations present; backfill is kept but tagged.
    assert len(entry["observations"]) == 2
    backfill = next(o for o in entry["observations"] if o.get("_backfill"))
    assert backfill["photo_id"] == "p2"
    real = next(o for o in entry["observations"] if o.get("status") == "damaged")
    assert real["photo_id"] == "p1"
    # Real high-confidence damaged observation must drive aggregator verdict.
    assert entry["aggregated_status"] == "damaged"


def test_synthesizer_treats_absent_backfill_as_no_information():
    """The synthesizer (``synthesizer._is_backfilled_uncertain``) must
    recognise ``status="absent" + _backfill=True`` and exclude it from
    consensus voting, just like the legacy ``status="uncertain" + score=0``
    backfill it replaces.  This is the **only** layer that should actually
    filter backfilled rows out of the verdict.
    """
    from agents.synthesizer import _is_backfilled_uncertain

    # New sentinel — view_agent writes these.
    new_backfill = {
        "status": "absent",
        "_backfill": True,
        "model_confidence_score": 0.0,
        "confidence": "low",
    }
    assert _is_backfilled_uncertain(new_backfill) is True

    # Legacy backfill signature — still recognised for backward compat.
    legacy_backfill = {
        "status": "uncertain",
        "model_confidence_score": 0.0,
        "description": "该部件在照片中未被识别到，按视角清单补齐为uncertain",
    }
    assert _is_backfilled_uncertain(legacy_backfill) is True

    # Real observations are NOT backfills.
    real_damaged = {
        "status": "damaged",
        "damage_level": "severe",
        "model_confidence_score": 0.85,
        "confidence": "high",
    }
    assert _is_backfilled_uncertain(real_damaged) is False

    real_uncertain = {
        "status": "uncertain",
        "damage_level": "unknown",
        "model_confidence_score": 0.45,    # non-zero, real uncertain
        "confidence": "low",
    }
    assert _is_backfilled_uncertain(real_uncertain) is False


@pytest.mark.asyncio
async def test_master_agent_dispatches_view_agent_per_exterior_photo():
    fake_prior = {
        "vehicle": "Test Car",
        "vehicle_specs": {},
        "topology": {},
        "key_anchors": {},
    }
    fake_plan = {
        "photo_classifications": [
            {"photo_id": "a", "category": "exterior"},
            {"photo_id": "b", "category": "exterior"},
            {"photo_id": "c", "category": "vehicle_info"},
        ],
    }

    async def fake_view_agent(photo, vehicle_prior):
        return {
            "photo_id": photo["id"],
            "primary_view": "front",
            "view_detections": [],
            "parts": [
                {"part_id": "hood", "part_name": "引擎盖", "status": "intact", "damage_level": "none", "damage_types": ["none"], "confidence": "high", "description": "", "evidence_photo": photo["id"]},
            ],
        }

    with patch("agents.master_agent.vehicle_prior_agent", new=AsyncMock(return_value=fake_prior)):
        with patch("agents.master_agent.build_vehicle_topology") as mock_topology:
            from models.topology import TopologyNode, VehicleTopology
            topology = VehicleTopology(
                vehicle_id="test",
                vehicle_name="Test",
                nodes={
                    "hood": TopologyNode(
                        node_id="hood",
                        part_id="hood",
                        node_name="引擎盖",
                        node_type="panel",
                        region="front",
                        side="center",
                        visibility_from=["front"],
                    ),
                },
                regions={"front": ["hood"]},
            )
            mock_topology.return_value = topology
            # face_profiler is now the default entry point under
            # use_face_path=True.  Return empty so master_agent falls
            # through to view_agent for each exterior photo.
            with patch("agents.master_agent.face_profiler_agent",
                       new=AsyncMock(return_value={})):
                with patch("agents.master_agent.planner_agent", new=AsyncMock(return_value=fake_plan)):
                    with patch("agents.master_agent.view_agent", new=AsyncMock(side_effect=fake_view_agent)):
                        with patch("agents.master_agent.reviewer_subagent", new=AsyncMock(return_value={
                            "reviewed_parts": [],
                            "reviewed_part_actual_states": [],
                        })):
                            result = await master_assessment_agent(
                                [{"id": "a", "path": "/a.png"}, {"id": "b", "path": "/b.png"}, {"id": "c", "path": "/c.png"}],
                                {"vehicle_name": "Test"},
                            )

    assert result is not None
    # Two exterior photos should produce two view_agent calls.
    view_calls = [c for c in result._plan.get("photo_classifications", []) if c["category"] == "exterior"]
    assert len(view_calls) == 2
