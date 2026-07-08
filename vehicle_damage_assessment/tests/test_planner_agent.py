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
async def test_classify_photo_types_falls_back_when_llm_fails():
    """When the vision call errors on every photo, planner falls back to
    the deterministic filename+aspect-ratio classifier so the pipeline
    never deadlocks.

    The key behavioural change in this round: the planner DOES call the
    vision model by default (4-class photo classification cannot work
    from filenames alone, e.g. ``172852-01.png`` has no signal).  When
    the vision endpoint is unreachable, we fall back so a network blip
    does not break the whole assessment.
    """
    from agents import mcp_client
    from agents.planner_agent import _normalize_category

    async def fake_understand_image(image_path, prompt, **kwargs):
        raise RuntimeError("simulated vision outage")

    original = mcp_client.understand_image
    mcp_client.understand_image = fake_understand_image
    try:
        result = await _classify_photo_types(
            [
                {"id": "172852-行驶证.png", "path": "/tmp/a.png"},
                {"id": "172852-车头.png", "path": "/tmp/b.png"},
                {"id": "172852-内饰.png", "path": "/tmp/c.png"},
            ],
            {"vehicle": "Test"},
        )
    finally:
        mcp_client.understand_image = original
    # The fallback filename-based classifier should still route the
    # keyword-bearing filenames to the right categories.
    assert result["172852-行驶证.png"] == "vehicle_info"
    assert result["172852-车头.png"] == "exterior"
    assert result["172852-内饰.png"] == "interior"


@pytest.mark.asyncio
async def test_classify_photo_types_uses_vision_when_available():
    """When the vision model succeeds, planner uses its verdict — filename is ignored.

    We simulate the ``understand_image`` call returning a JSON object.
    A photo whose filename would route to ``vehicle_info`` by the keyword
    heuristic (e.g. ``行驶证.png``) should still be classified as the
    vision verdict says, even when they disagree.
    """
    from agents import mcp_client

    responses = {
        "/tmp/fake-01.png": '{"photo_id": "172852-01.png", "category": "exterior", "reason": "车头外观照"}',
        "/tmp/fake-09.png": '{"photo_id": "172852-09.png", "category": "interior", "reason": "驾驶舱照片"}',
        "/tmp/fake-13.png": '{"photo_id": "172852-13.png", "category": "vehicle_info", "reason": "行驶证特写"}',
    }

    async def fake_understand_image(image_path, prompt, **kwargs):
        # The planner dispatches per-photo, so each call should carry
        # exactly one image path.
        return responses[image_path]

    original = mcp_client.understand_image
    mcp_client.understand_image = fake_understand_image
    try:
        result = await _classify_photo_types(
            [
                {"id": "172852-01.png", "path": "/tmp/fake-01.png"},
                {"id": "172852-09.png", "path": "/tmp/fake-09.png"},
                {"id": "172852-13.png", "path": "/tmp/fake-13.png"},
            ],
            {"vehicle": "Test"},
        )
    finally:
        mcp_client.understand_image = original
    # Filenames like ``172852-XX.png`` carry no signal — only the
    # vision verdict should drive the result.
    assert result["172852-01.png"] == "exterior"
    assert result["172852-09.png"] == "interior"
    assert result["172852-13.png"] == "vehicle_info"


@pytest.mark.asyncio
async def test_classify_photo_types_empty_when_no_photos():
    """No photos → empty type_map (no vision call needed)."""
    from agents import mcp_client

    called = {"n": 0}

    async def tracking(image_path, prompt, **kwargs):
        called["n"] += 1
        return "{}"

    original = mcp_client.understand_image
    mcp_client.understand_image = tracking
    try:
        result = await _classify_photo_types([], {"vehicle": "Test"})
    finally:
        mcp_client.understand_image = original
    assert result == {}
    assert called["n"] == 0  # short-circuit before any vision call


def test_normalize_category_aliases():
    """The vision model may emit ``auxiliary`` but our enum is ``vehicle_info``."""
    from agents.planner_agent import _normalize_category
    assert _normalize_category("exterior") == "exterior"
    assert _normalize_category("interior") == "interior"
    assert _normalize_category("auxiliary") == "vehicle_info"
    assert _normalize_category("license") == "vehicle_info"
    assert _normalize_category("scene") == "scene_intake"
    assert _normalize_category("bogus") == ""
    assert _normalize_category("") == ""
    assert _normalize_category(None) == ""


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

    from agents import mcp_client

    responses = {
        "/tmp/a.png": '{"photo_id": "172852-行驶证.png", "category": "auxiliary", "reason": "证件照"}',
        "/tmp/b.png": '{"photo_id": "172852-车头.png", "category": "exterior", "reason": "车头照"}',
        "/tmp/c.png": '{"photo_id": "172852-内饰.png", "category": "interior", "reason": "驾驶舱"}',
        "/tmp/d.png": '{"photo_id": "172852-现场.png", "category": "scene_intake", "reason": "现场环境"}',
    }

    async def fake_understand_image(image_path, prompt, **kwargs):
        return responses[image_path]

    original = mcp_client.understand_image
    mcp_client.understand_image = fake_understand_image
    try:
        result = await planner_agent(photos, {"vehicle": "Test Car"})
    finally:
        mcp_client.understand_image = original

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
