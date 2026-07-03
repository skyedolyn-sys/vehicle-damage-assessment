import pytest
from unittest.mock import AsyncMock, patch

from agents.assessment_orchestrator import _merge_two_states
from models.part_state import DamageLevel, PartActualState, Status


def test_merge_two_states_takes_worst_status():
    a = PartActualState(
        part_id="hood",
        part_name="引擎盖",
        part_category="front",
        side="center",
        status=Status.INTACT,
        damage_level=DamageLevel.NONE,
        confidence="high",
    )
    b = PartActualState(
        part_id="hood",
        part_name="引擎盖",
        part_category="front",
        side="center",
        status=Status.DAMAGED,
        damage_level=DamageLevel.MODERATE,
        confidence="medium",
    )
    merged = _merge_two_states(a, b)
    assert merged.status == Status.DAMAGED
    assert merged.damage_level == DamageLevel.MODERATE
    assert merged.confidence == "medium"


def test_merge_two_states_missing_wins():
    a = PartActualState(
        part_id="mirror_left",
        part_name="左后视镜",
        part_category="left",
        side="front_left",
        status=Status.DAMAGED,
        damage_level=DamageLevel.SEVERE,
        confidence="high",
    )
    b = PartActualState(
        part_id="mirror_left",
        part_name="左后视镜",
        part_category="left",
        side="front_left",
        status=Status.MISSING,
        damage_level=DamageLevel.SEVERE,
        confidence="medium",
    )
    merged = _merge_two_states(a, b)
    assert merged.status == Status.MISSING


@pytest.mark.asyncio
async def test_assessment_orchestrator_integration():
    from agents.assessment_orchestrator import assessment_orchestrator

    files = [
        {"id": "a.png", "path": "/a.png"},
        {"id": "b.png", "path": "/b.png"},
        {"id": "c.png", "path": "/c.png"},
    ]
    vehicle_info = {"brand": "", "model": "", "year": ""}

    fake_prior = {
        "vehicle": "Test Car",
        "vehicle_specs": {},
        "topology": {},
        "key_anchors": {},
    }
    fake_plan = {
        "photo_views": [
            {"photo_id": "a.png", "view_id": "front", "confidence": "high", "reason": ""},
            {"photo_id": "b.png", "view_id": "rear", "confidence": "high", "reason": ""},
            {"photo_id": "c.png", "view_id": "interior", "confidence": "high", "reason": ""},
        ],
        "view_groups": {
            "front": [{"id": "a.png", "path": "/a.png"}],
            "rear": [{"id": "b.png", "path": "/b.png"}],
            "interior": [{"id": "c.png", "path": "/c.png"}],
        },
        "coverage_gaps": [],
        "workflow_plan": {},
    }
    fake_vision_front = {
        "view_id": "front",
        "regions": ["front"],
        "parts": [],
        "part_actual_states": [
            PartActualState(
                part_id="hood",
                part_name="引擎盖",
                part_category="front",
                side="center",
                status=Status.INTACT,
                damage_level=DamageLevel.NONE,
                confidence="high",
                evidence_photos=["a.png"],
            )
        ],
        "uncertain_items": [],
    }
    fake_vision_rear = {
        "view_id": "rear",
        "regions": ["rear"],
        "parts": [],
        "part_actual_states": [
            PartActualState(
                part_id="trunk_lid",
                part_name="后备箱盖",
                part_category="rear",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.SEVERE,
                confidence="high",
                evidence_photos=["b.png"],
            )
        ],
        "uncertain_items": [],
    }
    fake_review = {
        "reviewed_parts": [],
        "reviewed_part_actual_states": [],
        "added_uncertain_items": [],
        "needs_rephotography": [],
        "summary": "OK",
    }

    with patch("agents.assessment_orchestrator.vehicle_prior_agent", new=AsyncMock(return_value=fake_prior)):
        with patch("agents.assessment_orchestrator.build_vehicle_topology") as mock_topology:
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
                    "trunk_lid": TopologyNode(
                        node_id="trunk_lid",
                        part_id="trunk_lid",
                        node_name="后备箱盖",
                        node_type="panel",
                        region="rear",
                        side="center",
                        visibility_from=["rear"],
                    ),
                },
                regions={"front": ["hood"], "rear": ["trunk_lid"]},
            )
            mock_topology.return_value = topology
            with patch("agents.assessment_orchestrator.planner_agent", new=AsyncMock(return_value=fake_plan)):
                with patch("agents.assessment_orchestrator.vision_subagent", new=AsyncMock(side_effect=[fake_vision_front, fake_vision_rear])):
                    with patch("agents.assessment_orchestrator.reviewer_subagent", new=AsyncMock(return_value=fake_review)):
                        result = await assessment_orchestrator(files, vehicle_info)

    assert "parts" in result
    part_ids = {p["part_id"] for p in result["parts"]}
    assert "hood" in part_ids
    assert "trunk_lid" in part_ids
    excluded_ids = {p["id"] for p in result.get("excluded_photos", [])}
    assert "c.png" in excluded_ids
    assert "a.png" not in excluded_ids
    assert "b.png" not in excluded_ids
