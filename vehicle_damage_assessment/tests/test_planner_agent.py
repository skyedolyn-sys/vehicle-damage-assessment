import pytest
from unittest.mock import AsyncMock, patch

from agents.planner_agent import (
    _classify_photo_by_filename,
    _stabilize_plan,
    _group_photos_by_view,
    get_coverage_summary,
    get_photos_for_region,
    normalize_view_id,
    plan_to_location_map,
    planner_agent,
)
from agents.view_mapping import STANDARD_VIEWS


def test_normalize_view_id_variants():
    assert normalize_view_id("front_left_45") == "front_left_45"
    assert normalize_view_id("车头左前45度") == "front_left_45"
    assert normalize_view_id("右侧") == "right_90"
    assert normalize_view_id("车顶俯视") == "top"


def test_classify_photo_by_filename():
    assert _classify_photo_by_filename("行驶证.png") == "auxiliary"
    assert _classify_photo_by_filename("vin.png") == "auxiliary"
    assert _classify_photo_by_filename("车内座椅.png") == "interior"
    assert _classify_photo_by_filename("车头.png") == ""


def test_stabilize_plan_filters_non_exterior():
    photos = [
        {"id": "a.png", "path": "/a.png"},
        {"id": "b.png", "path": "/b.png"},
        {"id": "c.png", "path": "/c.png"},
    ]
    photo_views = [
        {"photo_id": "a.png", "view_id": "front", "confidence": "high", "reason": ""},
        {"photo_id": "b.png", "view_id": "interior", "confidence": "high", "reason": ""},
        {"photo_id": "c.png", "view_id": "unknown", "confidence": "low", "reason": ""},
    ]
    photo_types = {"a.png": "exterior", "b.png": "interior", "c.png": "unknown"}
    plan = _stabilize_plan(photo_views, photos, photo_types)

    assert len(plan["view_groups"]["front"]) == 1
    assert plan["view_groups"]["front"][0]["id"] == "a.png"
    assert len(plan["view_groups"]["interior"]) == 1
    assert plan["view_groups"]["interior"][0]["id"] == "b.png"
    assert len(plan["view_groups"]["unknown"]) == 1
    assert "right_90" in [g["missing_view"] for g in plan["coverage_gaps"]]
    assert plan["workflow_plan"]["priority_views"] == ["front"]


def test_stabilize_plan_sorts_and_deduplicates_by_confidence():
    photos = [
        {"id": "low.png", "path": "/low.png"},
        {"id": "high.png", "path": "/high.png"},
    ]
    photo_views = [
        {"photo_id": "low.png", "view_id": "front", "confidence": "low", "reason": ""},
        {"photo_id": "high.png", "view_id": "front", "confidence": "high", "reason": ""},
    ]
    photo_types = {"low.png": "exterior", "high.png": "exterior"}
    plan = _stabilize_plan(photo_views, photos, photo_types)

    assert len(plan["view_groups"]["front"]) == 1
    assert plan["view_groups"]["front"][0]["id"] == "high.png"


def test_group_photos_by_view():
    photos = [
        {"id": "a.png", "path": "/a.png"},
        {"id": "b.png", "path": "/b.png"},
    ]
    photo_views = [
        {"photo_id": "a.png", "view_id": "front", "confidence": "high", "reason": ""},
        {"photo_id": "b.png", "view_id": "left_90", "confidence": "high", "reason": ""},
    ]
    groups = _group_photos_by_view(photo_views, photos)
    assert len(groups["front"]) == 1
    assert len(groups["left_90"]) == 1
    assert groups["front"][0]["_planner_view"] == "front"


def test_plan_to_location_map():
    plan = {
        "photo_views": [
            {"photo_id": "a.png", "view_id": "front_left_45", "confidence": "high", "reason": ""},
            {"photo_id": "b.png", "view_id": "rear", "confidence": "high", "reason": ""},
        ]
    }
    location_map = plan_to_location_map(plan)
    assert location_map["a.png"]["location"] == "front"
    assert location_map["a.png"]["secondary_locations"] == ["left"]
    assert location_map["b.png"]["location"] == "rear"


def test_get_photos_for_region():
    plan = {
        "view_groups": {
            "front_left_45": [
                {"id": "a.png", "path": "/a.png", "_planner_view": "front_left_45"},
            ],
            "left_90": [
                {"id": "b.png", "path": "/b.png", "_planner_view": "left_90"},
            ],
            "rear": [
                {"id": "c.png", "path": "/c.png", "_planner_view": "rear"},
            ],
        }
    }
    left_photos = get_photos_for_region(plan, "left")
    assert len(left_photos) == 2
    assert {p["id"] for p in left_photos} == {"a.png", "b.png"}


def test_get_coverage_summary():
    plan = {
        "view_groups": {
            "front": [{"id": "a.png"}],
            "rear": [{"id": "b.png"}],
            "interior": [{"id": "c.png"}],
            "unknown": [],
        },
        "coverage_gaps": [
            {"missing_view": "left_90", "suggested_action": "补拍左侧"}
        ],
    }
    summary = get_coverage_summary(plan)
    assert summary["covered_views"] == ["front", "rear"]
    assert summary["exterior_photo_count"] == 2
    assert summary["ignored_photo_count"] == 1


@pytest.mark.asyncio
async def test_planner_agent_empty_photos():
    result = await planner_agent([], {})
    assert result["photo_views"] == []
    assert result["coverage_gaps"] == []
    for view in STANDARD_VIEWS:
        assert result["view_groups"][view] == []


@pytest.mark.asyncio
async def test_planner_agent_adds_missing_entries():
    photos = [{"id": "a.png", "path": "/a.png"}]
    fake_json = {
        "photo_views": [],
        "coverage_gaps": [],
        "workflow_plan": {},
    }
    with patch("agents.planner_agent._build_image_content", return_value={"type": "image_url", "image_url": {"url": "data:fake"}}):
        with patch("agents.planner_agent.call_minimax", new=AsyncMock(return_value="{}")):
            with patch("agents.planner_agent.extract_json", return_value=fake_json):
                result = await planner_agent(photos, {"vehicle": "test"})
    assert len(result["photo_views"]) == 1
    assert result["photo_views"][0]["view_id"] == "unknown"
