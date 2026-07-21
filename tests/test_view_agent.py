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


@pytest.mark.asyncio
async def test_symmetric_glass_closeup_yields_no_side_view():
    """BUG2b+BUG2c 修订版：a symmetric glass/roof close-up that still shows the
    vehicle's front structural features (visible_faces has front) keeps its
    facing=front, but drops camera_side — so view_agent gets primary_view=front
    and can convict roof/sunroof/windshield (center parts), but CANNOT convict
    any left/right-suffixed part (pillar_a_left, mirror_right, etc).

    172852 photo 20: symmetric front view with collapsed roof + sunroof shatter.
    The car's hood/windshield frame is visible, so facing=front is reliable.
    But it's NOT a 3/4 view, so we cannot tell left from right — central damage
    must not be attributed to a specific side pillar.
    """
    from agents.face_mapping import build_face_prior

    profile = {
        "facing": "front",
        "side_panel_pos": "left",  # model guessed a side on a symmetric shot
        "visible_faces": [
            {"face": "front", "coverage": "dominant"},
            {"face": "roof", "coverage": "partial"},
        ],  # no side face → only camera_side drops, facing stays front
        "anchor": "前挡风玻璃+车顶",
        "confidence": "high",
    }
    prior = build_face_prior("photo_20", profile)
    assert prior["facing"] == "front", (
        "visible front structural features (windshield frame, hood edge) make facing reliable"
    )
    assert prior["camera_side"] is None
    assert prior["usable"] is True
    # 候选部件里不能有任何"车身侧面"的 left/right（door/mirror/pillar_*_left/right 等）
    # ——camera_side=None 已剥离侧面归属
    side_body_left_right = [
        p for p in prior["candidate_parts"]
        if (p.endswith("_left") or p.endswith("_right"))
        and any(p.startswith(prefix) for prefix in
                ("door_", "mirror_", "pillar_", "fender_rear_", "quarter_", "taillight_"))
    ]
    assert side_body_left_right == [], (
        f"camera_side=None must strip side-body left/right candidates, got: {side_body_left_right}"
    )
    # 但仍能定罪 center 部件（roof/sunroof/windshield_front 等）
    assert "roof_front" in prior["candidate_parts"]
    assert "windshield_front" in prior["candidate_parts"]
    assert "sunroof_glass" in prior["candidate_parts"]
    # 前部的左右大灯/翼子板保留（它们是 front 视角的两个独立可见部件，不属于挂错侧问题）
    assert "headlight_front_left" in prior["candidate_parts"]
    assert "headlight_front_right" in prior["candidate_parts"]

    fake_json = {"parts": []}
    with patch("agents.view_agent.build_image_content", return_value={"type": "image_url"}):
        with patch("agents.view_agent.call_minimax", new=AsyncMock(return_value="{}")):
            with patch("agents.view_agent.extract_json", return_value=fake_json):
                result = await view_agent(
                    {"id": "photo_20", "path": "/tmp/photo_20.png"},
                    {"vehicle": "Test"},
                    face_prior=prior,
                )

    # primary_view 应该是 front（不是 None）——该照片仍能贡献中心部件观察
    assert result["primary_view"] == "front", (
        "a symmetric front view with visible front features must keep primary_view=front"
    )

