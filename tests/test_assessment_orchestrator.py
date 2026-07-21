import pytest
from unittest.mock import AsyncMock, patch

from agents.master_agent import _apply_review_overrides
from models.part_state import DamageLevel, PartActualState, Status


def test_apply_review_overrides_takes_worst_status():
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
    merged = _apply_review_overrides([a], [b])
    assert len(merged) == 1
    assert merged[0].status == Status.DAMAGED
    assert merged[0].damage_level == DamageLevel.MODERATE
    assert merged[0].confidence == "medium"


def test_apply_review_overrides_missing_wins():
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
    merged = _apply_review_overrides([a], [b])
    assert merged[0].status == Status.MISSING


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
        "photo_classifications": [
            {"photo_id": "a.png", "category": "exterior", "confidence": "high"},
            {"photo_id": "b.png", "category": "exterior", "confidence": "high"},
            {"photo_id": "c.png", "category": "interior", "confidence": "high"},
        ],
    }
    fake_view_a = {
        "photo_id": "a.png",
        "primary_view": "front",
        "view_detections": [{"view_id": "front", "confidence_score": 0.9, "is_primary": True}],
        "parts": [
            {
                "part_id": "hood",
                "part_name": "引擎盖",
                "status": "intact",
                "damage_level": "none",
                "damage_types": ["none"],
                "model_confidence_score": 0.9,
                "confidence": "high",
                "description": "引擎盖平整无损伤",
                "evidence_photo": "a.png",
            },
        ],
    }
    fake_view_b = {
        "photo_id": "b.png",
        "primary_view": "rear",
        "view_detections": [{"view_id": "rear", "confidence_score": 0.9, "is_primary": True}],
        "parts": [
            {
                "part_id": "trunk_lid",
                "part_name": "后备箱盖",
                "status": "damaged",
                "damage_level": "severe",
                "damage_types": ["deformation"],
                "model_confidence_score": 0.9,
                "confidence": "high",
                "description": "后备箱盖严重变形",
                "evidence_photo": "b.png",
            },
        ],
    }
    fake_review = {
        "reviewed_parts": [],
        "reviewed_part_actual_states": [],
        "added_uncertain_items": [],
        "needs_rephotography": [],
        "summary": "OK",
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
            with patch("agents.master_agent.planner_agent", new=AsyncMock(return_value=fake_plan)):
                with patch("agents.master_agent.view_agent", new=AsyncMock(side_effect=[fake_view_a, fake_view_b])):
                    with patch("agents.master_agent.reviewer_subagent", new=AsyncMock(return_value=fake_review)):
                        # face_profiler is the default entry point under
                        # use_face_path=True; return empty so view_agent runs.
                        with patch("agents.master_agent.face_profiler_agent",
                                   new=AsyncMock(return_value={})):
                            result = await assessment_orchestrator(files, vehicle_info)

    assert "parts" in result
    part_ids = {p["part_id"] for p in result["parts"]}
    assert "hood" in part_ids
    assert "trunk_lid" in part_ids
    excluded_ids = {p["id"] for p in result.get("excluded_photos", [])}
    assert "c.png" in excluded_ids
    assert "a.png" not in excluded_ids
    assert "b.png" not in excluded_ids
