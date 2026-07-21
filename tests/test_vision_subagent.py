import pytest
from unittest.mock import AsyncMock, patch

from agents.vision_subagent import (
    vision_subagent,
    _partition_checklist,
    _merge_batch_results,
)


@pytest.mark.asyncio
async def test_vision_subagent_empty_photos():
    result = await vision_subagent("front_left_45", [], {}, None)
    assert result["view_id"] == "front_left_45"
    assert result["regions"] == ["front", "left", "roof_front"]
    assert result["parts"] == []
    assert result["part_actual_states"] == []
    assert result["uncertain_items"] == []
    assert result["cross_view_candidates"] == []
    assert result["additional_findings"] == []


@pytest.mark.asyncio
async def test_vision_subagent_returns_parts():
    photos = [{"id": "a.png", "path": "/a.png"}]
    fake_json = {
        "view_id": "front_left_45",
        "parts": [
            {
                "part_id": "hood",
                "part_name": "引擎盖",
                "region": "front",
                "side": "center",
                "status": "intact",
                "damage_level": "none",
                "damage_type": ["none"],
                "standard_exists": True,
                "actual_visible": True,
                "actual_present": True,
                "confidence": "high",
                "evidence_photo": ["a.png"],
                "notes": "完好",
            }
        ],
        "uncertain_items": [],
    }
    with patch("agents.vision_subagent.build_image_content", return_value={"type": "image_url", "image_url": {"url": "data:fake"}}):
        with patch("agents.vision_subagent.call_minimax", new=AsyncMock(return_value="{}")):
            with patch("agents.vision_subagent.extract_json", return_value=fake_json):
                result = await vision_subagent("front_left_45", photos, {"vehicle": "test"}, None)

    assert len(result["part_actual_states"]) >= 1
    assert any(s.part_id == "hood" for s in result["part_actual_states"])


def test_partition_checklist_groups_by_region_and_splits_large_groups():
    checklist = [
        {"part_id": f"front_{i}", "part_name": f"Front {i}", "region": "front"}
        for i in range(13)
    ]
    batches = _partition_checklist(checklist, max_size=6)

    assert len(batches) == 3
    assert [len(b) for b in batches] == [6, 6, 1]
    assert all(item["region"] == "front" for batch in batches for item in batch)
    # Order is preserved.
    flat = [item["part_id"] for batch in batches for item in batch]
    assert flat == [f"front_{i}" for i in range(13)]


def test_partition_checklist_separates_regions():
    checklist = [
        {"part_id": "f1", "part_name": "F1", "region": "front"},
        {"part_id": "l1", "part_name": "L1", "region": "left"},
        {"part_id": "f2", "part_name": "F2", "region": "front"},
        {"part_id": "l2", "part_name": "L2", "region": "left"},
    ]
    batches = _partition_checklist(checklist, max_size=6)

    assert len(batches) == 2
    region_ids = {tuple(item["part_id"] for item in batch) for batch in batches}
    assert region_ids == {("f1", "f2"), ("l1", "l2")}


def test_merge_batch_results_prefers_damaged_and_higher_confidence():
    results = [
        {
            "parts": [
                {"part_id": "hood", "status": "intact", "confidence": "high", "notes": ""},
                {"part_id": "bumper_front", "status": "uncertain", "confidence": "medium", "notes": ""},
            ],
            "uncertain_items": [{"part_id": "hood", "reason": "glare"}],
            "cross_view_candidates": [{"part_id": "door_front_left", "view": "left"}],
            "additional_findings": [{"finding": "scratch on glass"}],
        },
        {
            "parts": [
                {"part_id": "hood", "status": "damaged", "confidence": "low", "notes": ""},
                {"part_id": "bumper_front", "status": "uncertain", "confidence": "high", "notes": ""},
            ],
            "uncertain_items": [{"part_id": "hood", "reason": "glare"}],
            "cross_view_candidates": [{"part_id": "door_front_left", "view": "left"}],
            "additional_findings": [{"finding": "scratch on glass"}],
        },
    ]

    merged = _merge_batch_results(results)
    parts_by_id = {p["part_id"]: p for p in merged["parts"]}

    assert parts_by_id["hood"]["status"] == "damaged"
    assert parts_by_id["bumper_front"]["confidence"] == "high"
    # Dedup keeps one copy of each identical item.
    assert len(merged["uncertain_items"]) == 1
    assert len(merged["cross_view_candidates"]) == 1
    assert len(merged["additional_findings"]) == 1


@pytest.mark.asyncio
async def test_vision_subagent_long_checklist_batches_and_returns_all_parts():
    """A 13-item front-region checklist should be split and all parts merged."""
    photos = [{"id": "a.png", "path": "/a.png"}]
    checklist = [
        {"part_id": f"front_{i}", "part_name": f"Front {i}", "region": "front"}
        for i in range(13)
    ]

    call_count = 0

    def fake_extract_json(raw):
        nonlocal call_count
        call_count += 1
        # Each batch returns exactly the parts it was asked about.
        return {
            "parts": [
                {
                    "part_id": item["part_id"],
                    "part_name": item["part_name"],
                    "region": item["region"],
                    "side": "center",
                    "status": "intact",
                    "damage_level": "none",
                    "damage_type": ["none"],
                    "standard_exists": True,
                    "actual_visible": True,
                    "actual_present": True,
                    "confidence": "high",
                    "evidence_photo": ["a.png"],
                    "notes": "完好",
                }
                for item in checklist[(call_count - 1) * 6 : call_count * 6]
            ],
            "uncertain_items": [],
        }

    with patch("agents.vision_subagent.build_image_content", return_value={"type": "image_url", "image_url": {"url": "data:fake"}}):
        with patch("agents.vision_subagent.call_minimax", new=AsyncMock(return_value="{}")):
            with patch("agents.vision_subagent.extract_json", side_effect=fake_extract_json):
                with patch("agents.vision_subagent._build_checklist", return_value=checklist):
                    result = await vision_subagent("front_left_45", photos, {"vehicle": "test"}, None)

    assert call_count == 3
    returned_ids = {s.part_id for s in result["part_actual_states"]}
    expected_ids = {item["part_id"] for item in checklist}
    assert returned_ids == expected_ids
    assert result["view_id"] == "front_left_45"
    assert "cross_view_candidates" in result
    assert "additional_findings" in result
