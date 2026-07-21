"""Tests for the vision-driven planner classification.

The planner now classifies photos by looking at them with the vision model
(MiniMax M3), not by filename keyword or aspect ratio.  All tests mock
``call_minimax`` so no real API call is made.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from agents.planner_agent import (
    _apply_exterior_gate,
    _normalize_result,
    _realign_to_input,
    _sanitize_item,
    planner_agent,
)
from agents.view_mapping import PHOTO_TYPE_CATEGORIES


def _vision_response(items):
    """Wrap a list of per-photo vision dicts as the model's raw JSON string."""
    return json.dumps(items, ensure_ascii=False)


def _photo(pid):
    return {"id": pid, "path": f"/tmp/{pid}"}


def _mock_vision(items_or_raw):
    """Context manager: mock call_minimax + build_image_content (no real file I/O).

    ``build_image_content`` would otherwise try to open the photo path on disk;
    the tests use synthetic paths, so it must be stubbed alongside the LLM call.
    """
    raw = items_or_raw if isinstance(items_or_raw, str) else _vision_response(items_or_raw)
    return (
        patch("agents.planner_agent.call_minimax", new=AsyncMock(return_value=raw)),
        patch(
            "agents.planner_agent.build_image_content",
            new=lambda path, max_width=0: {"type": "image_url", "image_url": {"url": f"stub:{path}"}},
        ),
    )


# ---------------------------------------------------------------------------
# _apply_exterior_gate — the outline/position gate is the core of BUG1 fix.
# ---------------------------------------------------------------------------

def test_gate_passes_qualified_exterior():
    item = {"category": "exterior", "position": "front", "has_vehicle_outline": True}
    assert _apply_exterior_gate(item) == "exterior"


def test_gate_strips_exterior_without_outline():
    # Close-up with no discernible vehicle outline → stripped from exterior pool.
    item = {"category": "exterior", "position": "front", "has_vehicle_outline": False}
    assert _apply_exterior_gate(item) == "exclude"


def test_gate_strips_exterior_without_position():
    # Exterior-looking but position unclear → stripped.
    item = {"category": "exterior", "position": "unclear", "has_vehicle_outline": True}
    assert _apply_exterior_gate(item) == "exclude"


def test_gate_passes_interior():
    assert _apply_exterior_gate({"category": "interior"}) == "interior"


def test_gate_passes_vehicle_info():
    assert _apply_exterior_gate({"category": "vehicle_info"}) == "vehicle_info"


def test_gate_strips_unknown_category():
    assert _apply_exterior_gate({"category": "garbage"}) == "exclude"
    assert _apply_exterior_gate({}) == "exclude"


# ---------------------------------------------------------------------------
# _normalize_result / _sanitize_item / _realign_to_input
# ---------------------------------------------------------------------------

def test_normalize_accepts_bare_list():
    items = [{"photo_id": "a.png", "category": "exterior"}]
    assert _normalize_result(items) == items


def test_normalize_accepts_wrapped_results():
    items = [{"photo_id": "a.png", "category": "exterior"}]
    assert _normalize_result({"results": items}) == items
    assert _normalize_result({"classifications": items}) == items


def test_normalize_wraps_single_dict():
    item = {"photo_id": "a.png", "category": "exterior"}
    assert _normalize_result(item) == [item]


def test_normalize_rejects_unusable():
    assert _normalize_result("not a list") is None
    assert _normalize_result(42) is None


def test_sanitize_defaults_unknown_category_to_exclude():
    out = _sanitize_item({"photo_id": "a.png", "category": "weird"}, "a.png")
    assert out["category"] == "exclude"


def test_sanitize_clamps_position_and_confidence():
    out = _sanitize_item(
        {"photo_id": "a.png", "category": "exterior", "position": "sideways", "confidence": "very"},
        "a.png",
    )
    assert out["position"] == "unclear"
    assert out["confidence"] == "low"


def test_realign_falls_back_positionally():
    # Model renamed ids; positional fallback should still classify.
    items = [{"photo_id": "WRONG", "category": "interior", "position": "unclear",
              "has_vehicle_outline": False, "cabin_evidence": "气囊", "confidence": "high", "reason": "车内"}]
    photos = [_photo("172852-13.png")]
    aligned = _realign_to_input(items, photos)
    assert aligned[0]["category"] == "interior"
    assert aligned[0]["photo_id"] == "172852-13.png"


def test_realign_excludes_missing_photo():
    # No usable output for a photo → exclude (no deterministic guess).
    aligned = _realign_to_input([], [_photo("172852-13.png")])
    assert aligned[0]["category"] == "exclude"
    assert "剥离" in aligned[0]["reason"]


# ---------------------------------------------------------------------------
# planner_agent end-to-end (mocked vision call)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_planner_classifies_interior_exterior_vehicleinfo():
    photos = [_photo("172852-13.png"), _photo("172852-01.png"), _photo("172852-99.png")]
    vision_items = [
        # 13: interior — airbag/dashboard shot (the BUG1 offender).
        {"photo_id": "172852-13.png", "category": "interior", "position": "unclear",
         "has_vehicle_outline": False, "cabin_evidence": "弹开的气囊+撕裂的仪表台",
         "confidence": "high", "reason": "车内"},
        # 01: qualified exterior — front-left 3/4 with clear outline.
        {"photo_id": "172852-01.png", "category": "exterior", "position": "front",
         "has_vehicle_outline": True, "cabin_evidence": "无",
         "confidence": "high", "reason": "车头左前45度"},
        # 99: vehicle info — license plate close-up.
        {"photo_id": "172852-99.png", "category": "vehicle_info", "position": "unclear",
         "has_vehicle_outline": False, "cabin_evidence": "无",
         "confidence": "high", "reason": "行驶证"},
    ]
    cm_minimax, cm_image = _mock_vision(vision_items)
    with cm_minimax, cm_image:
        result = await planner_agent(photos, {"vehicle": "Test Car"})

    by_id = {c["photo_id"]: c for c in result["photo_classifications"]}
    assert by_id["172852-13.png"]["category"] == "interior"
    assert by_id["172852-01.png"]["category"] == "exterior"
    assert by_id["172852-99.png"]["category"] == "vehicle_info"

    for c in result["photo_classifications"]:
        assert c["category"] in PHOTO_TYPE_CATEGORIES
        assert c["confidence"] in {"high", "medium", "low"}
        assert 0.0 <= c["confidence_score"] <= 1.0
        assert isinstance(c["reason"], str)


@pytest.mark.asyncio
async def test_planner_strips_exterior_closeup_without_outline():
    """A碎玻璃特写 with no discernible outline must NOT enter the exterior pool."""
    photos = [_photo("172852-22.png")]
    vision_items = [
        {"photo_id": "172852-22.png", "category": "exterior", "position": "unclear",
         "has_vehicle_outline": False, "cabin_evidence": "无",
         "confidence": "medium", "reason": "纯特写，看不出车身轮廓"},
    ]
    cm_minimax, cm_image = _mock_vision(vision_items)
    with cm_minimax, cm_image:
        result = await planner_agent(photos, {"vehicle": "Test Car"})

    c = result["photo_classifications"][0]
    assert c["category"] == "exclude"
    assert "剥离" in c["reason"]


@pytest.mark.asyncio
async def test_planner_excludes_on_unparseable_vision_output():
    """No deterministic fallback: unparseable vision output → exclude."""
    photos = [_photo("172852-13.png")]
    cm_minimax, cm_image = _mock_vision("this is not json at all")
    with cm_minimax, cm_image:
        result = await planner_agent(photos, {"vehicle": "Test Car"})

    assert result["photo_classifications"][0]["category"] == "exclude"


@pytest.mark.asyncio
async def test_planner_excludes_on_non_list_vision_output():
    photos = [_photo("172852-13.png")]
    cm_minimax, cm_image = _mock_vision(json.dumps({"unexpected": 42}))
    with cm_minimax, cm_image:
        result = await planner_agent(photos, {"vehicle": "Test Car"})

    # Single dict gets wrapped; its category is missing → exclude.
    assert result["photo_classifications"][0]["category"] == "exclude"


@pytest.mark.asyncio
async def test_planner_batches_large_photo_sets():
    """More than _BATCH_SIZE photos must trigger multiple vision calls."""
    from agents.planner_agent import _BATCH_SIZE

    photos = [_photo(f"p{i:02d}.png") for i in range(_BATCH_SIZE + 2)]
    vision_items = [
        {"photo_id": p["id"], "category": "exterior", "position": "front",
         "has_vehicle_outline": True, "cabin_evidence": "无",
         "confidence": "high", "reason": "车头"}
        for p in photos
    ]

    call_count = {"n": 0}

    async def _fake_call(*args, **kwargs):
        call_count["n"] += 1
        # Return only the batch's photos; the realign step maps by id.
        return _vision_response(vision_items)

    cm_minimax = patch("agents.planner_agent.call_minimax", new=AsyncMock(side_effect=_fake_call))
    cm_image = patch(
        "agents.planner_agent.build_image_content",
        new=lambda path, max_width=0: {"type": "image_url", "image_url": {"url": f"stub:{path}"}},
    )
    with cm_minimax, cm_image:
        result = await planner_agent(photos, {"vehicle": "Test Car"})

    assert call_count["n"] == 2  # ceil((BATCH+2)/BATCH)
    assert all(c["category"] == "exterior" for c in result["photo_classifications"])


@pytest.mark.asyncio
async def test_planner_agent_empty_photos():
    result = await planner_agent([], {})
    assert result == {"photo_classifications": []}
