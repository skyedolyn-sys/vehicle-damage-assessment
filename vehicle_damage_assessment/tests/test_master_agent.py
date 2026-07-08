import pytest
from unittest.mock import AsyncMock, patch

from agents.master_agent import (
    _extract_photo_classifications,
    _aggregate_part_evidence,
    _boost_confidence,
    _build_region_results,
    master_assessment_agent,
)
from models.part_state import PartActualState, Status


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
            "primary_view": "front_left",
            "parts": [
                {"part_id": "hood", "status": "intact", "damage_level": "none", "damage_types": ["none"], "confidence": "high", "description": "平整"},
            ],
        },
        {
            "photo_id": "p2",
            "primary_view": "rear",
            "parts": [
                {"part_id": "hood", "status": "damaged", "damage_level": "moderate", "damage_types": ["dent"], "confidence": "high", "description": "凹陷"},
            ],
        },
    ]
    evidence = _aggregate_part_evidence(view_results)
    # Neither primary strong intact nor primary strong damaged; consensus
    # is uncertain and conflicting is True.
    assert evidence["hood"]["conflicting"] is True
    assert evidence["hood"]["aggregated_status"] == "uncertain"


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
