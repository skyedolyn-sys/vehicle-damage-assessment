import pytest
from unittest.mock import AsyncMock, patch

from agents.vision_subagent import vision_subagent


@pytest.mark.asyncio
async def test_vision_subagent_empty_photos():
    result = await vision_subagent("front_left_45", [], {}, None)
    assert result["view_id"] == "front_left_45"
    assert result["regions"] == ["front", "left", "roof_front"]
    assert result["parts"] == []
    assert result["part_actual_states"] == []


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

    assert len(result["part_actual_states"]) == 1
    assert result["part_actual_states"][0].part_id == "hood"
    assert result["part_actual_states"][0].status.value == "intact"
