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
    """When the LLM call errors, planner falls back to the deterministic
    filename+aspect-ratio classifier so the pipeline never deadlocks.

    The key behavioural change in this round: the planner DOES call the
    LLM by default (4-class photo classification cannot work from
    filenames alone, e.g. ``172852-01.png`` has no signal).  When the
    LLM is unreachable, we fall back so a network blip does not break
    the whole assessment.
    """
    from agents import minimax_client
    from agents.planner_agent import _normalize_category

    original_call_minimax = minimax_client.call_minimax
    minimax_client.call_minimax = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("simulated LLM outage")
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
    # The fallback filename-based classifier should still route the
    # keyword-bearing filenames to the right categories.
    assert result["172852-行驶证.png"] == "vehicle_info"
    assert result["172852-车头.png"] == "exterior"
    assert result["172852-内饰.png"] == "interior"


@pytest.mark.asyncio
async def test_classify_photo_types_uses_llm_when_available():
    """When the LLM succeeds, planner uses the LLM verdict — filename is ignored.

    Simulate the LLM by returning a JSON payload that maps each photo
    to a category.  A photo whose filename would route to ``vehicle_info``
    by the keyword heuristic (e.g. ``行驶证.png``) should still be
    classified as the LLM says, even when they disagree.
    """
    from agents import minimax_client

    fake_response = (
        '{"classifications": ['
        '{"photo_id": "172852-01.png", "photo_type": "exterior", "reason": "车头外观照"},'
        '{"photo_id": "172852-09.png", "photo_type": "interior", "reason": "驾驶舱照片"},'
        '{"photo_id": "172852-13.png", "photo_type": "vehicle_info", "reason": "行驶证特写"}'
        ']}'
    )

    original_call_minimax = minimax_client.call_minimax
    original_build_image = minimax_client.build_image_content

    async def fake_call_minimax(messages, **kwargs):
        return fake_response

    def fake_build_image(path, max_width=None):
        # Avoid reading the file; the test never actually invokes the
        # network.  Return a stub content block.
        return {"type": "image_url", "image_url": {"url": f"stub://{path}"}}

    minimax_client.call_minimax = fake_call_minimax
    minimax_client.build_image_content = fake_build_image
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
        minimax_client.call_minimax = original_call_minimax
        minimax_client.build_image_content = original_build_image
    # Filenames like ``172852-XX.png`` carry no signal — only the LLM
    # verdict should drive the result.
    assert result["172852-01.png"] == "exterior"
    assert result["172852-09.png"] == "interior"
    assert result["172852-13.png"] == "vehicle_info"


@pytest.mark.asyncio
async def test_classify_photo_types_empty_when_no_photos():
    """No photos → empty type_map (no LLM call needed)."""
    from agents import minimax_client

    called = {"n": 0}

    async def tracking_call(messages, **kwargs):
        called["n"] += 1
        return "{}"

    original = minimax_client.call_minimax
    minimax_client.call_minimax = tracking_call
    try:
        result = await _classify_photo_types([], {"vehicle": "Test"})
    finally:
        minimax_client.call_minimax = original
    assert result == {}
    assert called["n"] == 0  # short-circuit before any LLM call


def test_normalize_category_aliases():
    """The LLM may emit ``auxiliary`` but our enum is ``vehicle_info``."""
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

    from agents import minimax_client

    fake_response = (
        '{"classifications": ['
        '{"photo_id": "172852-行驶证.png", "photo_type": "auxiliary", "reason": "证件照"},'
        '{"photo_id": "172852-车头.png", "photo_type": "exterior", "reason": "车头照"},'
        '{"photo_id": "172852-内饰.png", "photo_type": "interior", "reason": "驾驶舱"},'
        '{"photo_id": "172852-现场.png", "photo_type": "scene_intake", "reason": "现场环境"}'
        ']}'
    )

    original_call_minimax = minimax_client.call_minimax
    original_build_image = minimax_client.build_image_content

    async def fake_call_minimax(messages, **kwargs):
        return fake_response

    def fake_build_image(path, max_width=None):
        return {"type": "image_url", "image_url": {"url": f"stub://{path}"}}

    minimax_client.call_minimax = fake_call_minimax
    minimax_client.build_image_content = fake_build_image
    try:
        result = await planner_agent(photos, {"vehicle": "Test Car"})
    finally:
        minimax_client.call_minimax = original_call_minimax
        minimax_client.build_image_content = original_build_image

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
