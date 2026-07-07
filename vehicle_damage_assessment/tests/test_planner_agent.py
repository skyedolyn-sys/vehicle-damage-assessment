from typing import Dict

import pytest
from unittest.mock import patch
import asyncio

from agents.planner_agent import (
    _classify_by_filename,
    _classify_by_signals,
    _classify_photo_types,
    planner_agent,
)
from agents.view_mapping import EXTERIOR_VIEWS, PHOTO_TYPE_CATEGORIES


def test_classify_by_filename_vehicle_info():
    assert _classify_by_filename("行驶证.png") == "vehicle_info"
    assert _classify_by_filename("vin_plate.png") == "vehicle_info"
    assert _classify_by_filename("车牌.png") == "vehicle_info"
    assert _classify_by_filename("保单.png") == "vehicle_info"


def test_classify_by_filename_interior():
    assert _classify_by_filename("内饰.png") == "interior"
    assert _classify_by_filename("驾驶舱.png") == "interior"
    assert _classify_by_filename("座椅.png") == "interior"


def test_classify_by_filename_exterior():
    assert _classify_by_filename("车头.png") == "exterior"
    assert _classify_by_filename("左侧.png") == "exterior"
    assert _classify_by_filename("车顶.png") == "exterior"


def test_classify_by_filename_scene_intake():
    assert _classify_by_filename("现场.png") == "scene_intake"
    assert _classify_by_filename("全景.png") == "scene_intake"


def test_classify_by_signals_uses_filename():
    assert _classify_by_signals({"id": "行驶证.png"}) == "vehicle_info"
    assert _classify_by_signals({"id": "车头.png"}) == "exterior"


def test_classify_by_signals_extreme_aspect_ratio():
    # Portrait close-up should still route to exterior for ViewAgent inspection.
    assert _classify_by_signals({"id": "a.png", "_decoded_width": 400, "_decoded_height": 800}) == "exterior"
    # Landscape close-up.
    assert _classify_by_signals({"id": "a.png", "_decoded_width": 1600, "_decoded_height": 800}) == "exterior"


def test_classify_by_signals_standard_aspect_ratio():
    assert _classify_by_signals({"id": "a.png", "_decoded_width": 1024, "_decoded_height": 768}) == "exterior"


@pytest.mark.asyncio
async def test_classify_photo_types_no_llm():
    """Planner classification is deterministic and must not call any LLM."""
    from agents import minimax_client

    original_call_minimax = minimax_client.call_minimax
    minimax_client.call_minimax = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("LLM was called - planner must not call LLM")
    )
    try:
        result = await _classify_photo_types(
            [
                {"id": "172852-行驶证.png"},
                {"id": "172852-车头.png"},
                {"id": "172852-内饰.png"},
            ],
            {"vehicle": "Test"},
        )
    finally:
        minimax_client.call_minimax = original_call_minimax
    assert result["172852-行驶证.png"] == "vehicle_info"
    assert result["172852-车头.png"] == "exterior"
    assert result["172852-内饰.png"] == "interior"


@pytest.mark.asyncio
async def test_planner_agent_empty_photos():
    result = await planner_agent([], {})
    assert result == {"photo_classifications": []}


@pytest.mark.asyncio
async def test_planner_agent_classifies_all_photos():
    photos = [
        {"id": "172852-行驶证.png", "path": "/tmp/a.png"},
        {"id": "172852-车头.png", "path": "/tmp/b.png"},
        {"id": "172852-内饰.png", "path": "/tmp/c.png"},
        {"id": "172852-现场.png", "path": "/tmp/d.png"},
    ]
    result = await planner_agent(photos, {"vehicle": "Test Car"})
    classifications = result["photo_classifications"]
    assert len(classifications) == 4

    by_id = {c["photo_id"]: c for c in classifications}
    assert by_id["172852-行驶证.png"]["category"] == "vehicle_info"
    assert by_id["172852-车头.png"]["category"] == "exterior"
    assert by_id["172852-内饰.png"]["category"] == "interior"
    assert by_id["172852-现场.png"]["category"] == "scene_intake"

    for c in classifications:
        assert c["category"] in PHOTO_TYPE_CATEGORIES
        assert c["confidence"] in {"high", "medium", "low"}
        assert 0.0 <= c["confidence_score"] <= 1.0
        assert isinstance(c["reason"], str)


def test_all_exterior_keywords_map_to_exterior():
    """Coverage guard: every keyword listed for exterior still routes to exterior."""
    from agents.planner_agent import _EXTERIOR_KEYWORDS
    for kw in _EXTERIOR_KEYWORDS:
        assert _classify_by_filename(f"{kw}.png") == "exterior", kw
