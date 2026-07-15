import pytest
from unittest.mock import AsyncMock, patch

from agents.view_agent import (
    _normalize_view_agent_result,
    _normalize_part_observation,
    _calibrate_damage_confidence,
    _score_to_level,
    view_agent,
)


def test_normalize_view_detections_selects_primary():
    raw = {
        "view_detections": [
            {"view_id": "front_left", "confidence_score": 0.9, "is_primary": True},
            {"view_id": "left", "confidence_score": 0.3, "is_primary": False},
        ],
        "parts": [],
    }
    result = _normalize_view_agent_result("p1", raw)
    assert result["primary_view"] == "front_left"
    assert len(result["view_detections"]) == 2


def test_normalize_view_detections_below_threshold():
    raw = {
        "view_detections": [
            {"view_id": "front_left", "confidence_score": 0.3, "is_primary": True},
        ],
        "parts": [],
    }
    result = _normalize_view_agent_result("p1", raw)
    assert result["primary_view"] is None


def test_normalize_part_observation_intact():
    part = {
        "part_id": "hood",
        "part_name": "引擎盖",
        "status": "intact",
        "damage_level": "none",
        "damage_types": ["none"],
        "model_confidence_score": 0.9,
        "confidence": "high",
        "description": "引擎盖平整",
    }
    normalized = _normalize_part_observation(part, "p1")
    assert normalized["status"] == "intact"
    assert normalized["damage_level"] == "none"
    assert normalized["damage_types"] == ["none"]


def test_normalize_part_observation_missing():
    part = {
        "part_id": "hood",
        "status": "missing",
        "damage_level": "unknown",
        "damage_types": ["missing"],
        "model_confidence_score": 0.9,
        "confidence": "high",
        "description": "引擎盖缺失",
    }
    normalized = _normalize_part_observation(part, "p1")
    assert normalized["status"] == "missing"
    assert normalized["damage_level"] == "severe"
    assert "breakage" in normalized["damage_types"]


def test_normalize_part_observation_damaged_none_level_fixed():
    part = {
        "part_id": "bumper_front",
        "status": "damaged",
        "damage_level": "none",
        "damage_types": ["scratch"],
        "model_confidence_score": 0.8,
        "confidence": "high",
        "description": "保险杠有划痕",
    }
    normalized = _normalize_part_observation(part, "p1")
    assert normalized["damage_level"] == "light"


def test_normalize_part_observation_unknown_part_dropped():
    part = {
        "part_id": "not_a_part",
        "status": "damaged",
        "damage_level": "severe",
        "damage_types": ["deformation"],
    }
    assert _normalize_part_observation(part, "p1") is None


def test_normalize_result_alias_maps_free_form_names():
    """Free-form LLM names resolve to canonical ids via the alias table."""
    raw = {
        "view_detections": [],
        "primary_view": "rear",
        "parts": [
            {"part_id": "sunroof", "status": "damaged", "damage_level": "severe",
             "damage_types": ["shatter"], "confidence": "high"},
            {"part_id": "panoramic_sunroof_glass", "status": "damaged",
             "damage_level": "severe", "damage_types": ["shatter"], "confidence": "high"},
            {"part_id": "hood_front", "status": "intact", "damage_level": "none",
             "damage_types": ["none"], "confidence": "high"},
        ],
    }
    result = _normalize_view_agent_result("p1", raw)
    kept = {p["part_id"] for p in result["parts"]}
    assert "sunroof_glass" in kept
    assert "hood" in kept
    assert result["unmapped_parts"] == []


def test_normalize_result_collects_unmapped_unknown_and_out_of_candidate():
    """Unknown and out-of-candidate parts are traced, not silently dropped."""
    raw = {
        "view_detections": [],
        "primary_view": "rear",
        "parts": [
            {"part_id": "pillar_d_right", "status": "damaged", "damage_level": "moderate"},
            {"part_id": "door_rear_right", "status": "damaged", "damage_level": "severe"},
            {"part_id": "roof_rear", "status": "damaged", "damage_level": "moderate"},
        ],
    }
    result = _normalize_view_agent_result(
        "p1", raw, candidate_parts={"door_rear_right"}
    )
    kept = {p["part_id"] for p in result["parts"]}
    assert kept == {"door_rear_right"}

    by_reason = {u["drop_reason"]: u for u in result["unmapped_parts"]}
    assert by_reason["unknown_part_id"]["raw_name"] == "pillar_d_right"
    assert by_reason["out_of_candidate"]["raw_name"] == "roof_rear"
    assert by_reason["out_of_candidate"]["resolved_part_id"] == "roof_rear"


def test_normalize_result_part_name_fallback_maps_chinese_free_text():
    """An observation keyed only by part_name still goes through alias mapping."""
    raw = {
        "view_detections": [],
        "primary_view": "front",
        "parts": [
            {"part_name": "hood_front", "status": "intact", "damage_level": "none",
             "damage_types": ["none"], "confidence": "high"},
        ],
    }
    result = _normalize_view_agent_result("p1", raw)
    assert {p["part_id"] for p in result["parts"]} == {"hood"}


@pytest.mark.asyncio
async def test_backfill_missing_parts_for_primary_view():
    fake_json = {
        "view_detections": [{"view_id": "front", "confidence_score": 0.9, "is_primary": True}],
        "parts": [
            {"part_id": "hood", "status": "intact", "damage_level": "none", "damage_types": ["none"],
             "model_confidence_score": 0.9, "confidence": "high", "description": ""},
        ],
    }
    with patch("agents.view_agent.build_image_content", return_value={"type": "image_url"}):
        with patch("agents.view_agent.call_minimax", new=AsyncMock(return_value="{}")):
            with patch("agents.view_agent.extract_json", return_value=fake_json):
                result = await view_agent({"id": "p1", "path": "/tmp/p1.png"}, {"vehicle": "Test"})

    part_ids = {p["part_id"] for p in result["parts"]}
    assert "bumper_front" in part_ids
    assert "windshield_front" in part_ids
    backfilled = next(p for p in result["parts"] if p["part_id"] == "windshield_front")
    assert backfilled["status"] == "uncertain"


def test_calibrate_damage_confidence_description_signals():
    raw = {
        "model_confidence_score": 0.8,
        "description": "引擎盖有明显凹陷变形",
        "damage_types": ["dent", "deformation"],
        "status": "damaged",
    }
    score = _calibrate_damage_confidence(raw)
    assert score >= 0.75
    assert _score_to_level(score) in ("high", "medium")


def test_calibrate_damage_confidence_intact_with_positive_evidence():
    raw = {
        "model_confidence_score": 0.8,
        "description": "引擎盖平整无凹陷",
        "damage_types": ["none"],
        "status": "intact",
    }
    score = _calibrate_damage_confidence(raw)
    assert score >= 0.75
    assert _score_to_level(score) == "high"


@pytest.mark.asyncio
async def test_view_agent_calls_minimax_and_parses():
    fake_json = {
        "photo_id": "p1",
        "primary_view": "front_left",
        "view_detections": [
            {"view_id": "front_left", "confidence_score": 0.92, "is_primary": True},
        ],
        "parts": [
            {
                "part_id": "hood",
                "status": "damaged",
                "damage_level": "moderate",
                "damage_types": ["dent"],
                "model_confidence_score": 0.85,
                "confidence": "high",
                "description": "引擎盖有明显凹陷",
            },
        ],
    }
    with patch("agents.view_agent.build_image_content", return_value={"type": "image_url"}):
        with patch("agents.view_agent.call_minimax", new=AsyncMock(return_value="{}")):
            with patch("agents.view_agent.extract_json", return_value=fake_json):
                result = await view_agent({"id": "p1", "path": "/tmp/p1.png"}, {"vehicle": "Test"})

    assert result["photo_id"] == "p1"
    assert result["primary_view"] == "front_left"
    hood = next(p for p in result["parts"] if p["part_id"] == "hood")
    assert hood["status"] == "damaged"
    assert hood["damage_level"] == "moderate"


@pytest.mark.asyncio
async def test_view_agent_result_to_part_actual_states():
    from agents.view_agent import view_agent_result_to_part_actual_states
    result = {
        "photo_id": "p1",
        "primary_view": "front",
        "view_detections": [],
        "parts": [
            {
                "part_id": "hood",
                "part_name": "引擎盖",
                "status": "intact",
                "damage_level": "none",
                "damage_types": ["none"],
                "model_confidence_score": 0.9,
                "confidence": "high",
                "description": "平整",
                "evidence_photo": "p1",
            },
        ],
    }
    states = view_agent_result_to_part_actual_states(result)
    assert len(states) == 1
    assert states[0].part_id == "hood"
    assert states[0].photo_type == "front"
    assert states[0].evidence_photos == ["p1"]
