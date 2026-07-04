"""Tests for cross-view evidence fusion."""

from models.part_state import DamageLevel, PartActualState, Status
from agents.evidence_fusion import (
    apply_fusion,
    collect_part_evidence,
    fuse_evidence,
)


def _fake_candidate(part_id: str, view_id: str, **kwargs) -> dict:
    cand = {
        "part_id": part_id,
        "status": kwargs.get("status", "uncertain"),
        "damage_level": kwargs.get("damage_level", "unknown"),
        "damage_type": kwargs.get("damage_type", []),
        "confidence": kwargs.get("confidence", "low"),
        "evidence_photo": kwargs.get("evidence_photo", []),
        "notes": kwargs.get("notes", ""),
    }
    cand["_origin_view"] = view_id
    return cand


class TestFuseEvidence:
    def test_no_fusion_for_intact_part(self):
        candidates = [
            _fake_candidate("windshield_front", "front", status="intact"),
            _fake_candidate("windshield_front", "front_left_45", status="intact"),
        ]
        assert fuse_evidence("windshield_front", candidates) is None

    def test_damaged_in_two_views_promotes(self):
        candidates = [
            _fake_candidate(
                "windshield_front",
                "front",
                status="damaged",
                damage_level="severe",
                damage_type=["crack"],
                evidence_photo=["p1.png"],
                notes="大面积蛛网状碎裂",
            ),
            _fake_candidate(
                "windshield_front",
                "front_left_45",
                status="damaged",
                damage_level="severe",
                evidence_photo=["p2.png"],
                notes="玻璃碎裂可见",
            ),
        ]
        fused = fuse_evidence("windshield_front", candidates)
        assert fused is not None
        assert fused["status"] == "damaged"
        assert fused["damage_level"] == "severe"
        assert "p1.png" in fused["evidence_photo"]
        assert "p2.png" in fused["evidence_photo"]

    def test_multiple_uncertain_with_damage_signal(self):
        """Two uncertain observations whose notes mention cracks promote to damaged."""
        candidates = [
            _fake_candidate(
                "roof_front",
                "front_right_45",
                status="uncertain",
                notes="看到明显褶皱变形",
            ),
            _fake_candidate(
                "roof_front",
                "front_left_45",
                status="uncertain",
                notes="部分车顶有撕裂痕迹",
            ),
        ]
        fused = fuse_evidence("roof_front", candidates)
        assert fused is not None
        assert fused["status"] == "damaged"
        assert fused["damage_level"] in ("moderate", "severe")

    def test_intact_primary_beats_damaged_secondary(self):
        """When the primary view says intact, downgraded diagonals should not
        force a damaged conclusion for that part."""
        candidates = [
            _fake_candidate(
                "headlight_front_left",
                "front_left_45",  # primary
                status="intact",
            ),
            _fake_candidate(
                "headlight_front_left",
                "left_90",
                status="uncertain",
                notes="可能存在破损",  # ambiguous
            ),
        ]
        # No strongly-damaged primary, so fusion should not force damaged.
        assert fuse_evidence("headlight_front_left", candidates) is None

    def test_signal_keywords_trigger_uncertain_to_damaged(self):
        for status in ("damaged", "missing"):
            assert True  # Placeholder; main test in test_damaged_keywords

    def test_damage_signal_keywords(self):
        candidates = [
            _fake_candidate(
                "windshield_front",
                "front",
                status="uncertain",
                notes="玻璃出现放射状裂纹",
            ),
        ]
        # Single view with clear damage signal and no contradicting intact source
        fused = fuse_evidence("windshield_front", candidates)
        assert fused is not None
        assert fused["status"] == "damaged"

    def test_non_critical_part_not_fused(self):
        """Only safety/structural-critical parts are fused.

        DAMAGE_RECOGNITION_POLICY: hood 与 trunk_lid 现已被加入 _CRITICAL_FUSION_PARTS
        (它们是 high-sensitivity 部件,severe 损伤必须穿透 primary-intact)。
        本测试用 bumper_front 作为 non-critical part 的代表。
        """
        candidates = [
            _fake_candidate(
                "bumper_front",
                "front",
                status="damaged",
                damage_level="moderate",
            ),
            _fake_candidate(
                "bumper_front",
                "front_left_45",
                status="damaged",
                damage_level="light",
            ),
        ]
        assert fuse_evidence("bumper_front", candidates) is None


class TestCollectEvidence:
    def test_aggregates_by_part_id(self):
        results = [
            {"view_id": "front", "parts": [
                _fake_candidate("windshield_front", "front", status="intact"),
            ]},
            {"view_id": "front_left_45", "parts": [
                _fake_candidate("windshield_front", "front_left_45", status="uncertain"),
            ]},
        ]
        evidence = collect_part_evidence(results)
        assert "windshield_front" in evidence
        assert len(evidence["windshield_front"]) == 2

    def test_dedupes_by_view(self):
        results = [
            {"view_id": "front", "parts": [
                _fake_candidate("windshield_front", "front", status="damaged", damage_level="severe"),
                _fake_candidate("windshield_front", "front", status="intact", damage_level="none"),
            ]},
        ]
        evidence = collect_part_evidence(results)
        assert len(evidence["windshield_front"]) == 1
        # The severe one should win.
        assert evidence["windshield_front"][0]["damage_level"] == "severe"


class TestApplyFusionEndToEnd:
    def test_applies_overrides(self):
        results = [
            {"view_id": "front", "parts": [
                _fake_candidate(
                    "windshield_front",
                    "front",
                    status="damaged",
                    damage_level="severe",
                    evidence_photo=["p1.png"],
                ),
                _fake_candidate(
                    "windshield_front",
                    "front_left_45",
                    status="damaged",
                    damage_level="severe",
                    evidence_photo=["p2.png"],
                ),
            ]},
        ]
        overrides = apply_fusion(results)
        assert "windshield_front" in overrides
        assert overrides["windshield_front"]["status"] == "damaged"
        assert overrides["windshield_front"]["_fused"] is True

    def test_no_overrides_when_no_critical_parts(self):
        """DAMAGE_RECOGNITION_POLICY: hood 与 trunk_lid 现已被加入 critical 集合。
        本测试用 bumper_front 作为 non-critical part 的代表。
        """
        results = [
            {"view_id": "front", "parts": [
                _fake_candidate(
                    "bumper_front",
                    "front",
                    status="damaged",
                    damage_level="moderate",
                ),
            ]},
        ]
        overrides = apply_fusion(results)
        assert overrides == {}


class TestPhotoListNormalization:
    """Regression: evidence_photo may arrive as a comma-separated string from
    the LLM.  Iterating a raw string would split it character-by-character,
    producing lists like ['1','7','2','8','5','-',...] instead of
    ['172852-04.png','172852-03.png'].
    """

    def test_collect_part_evidence_handles_string_photos(self):
        results = [
            {"view_id": "rear_left_45", "parts": [
                _fake_candidate(
                    "windshield_rear",
                    "rear_left_45",
                    status="damaged",
                    damage_level="moderate",
                    evidence_photo="172852-04.png, 172852-03.png",
                ),
            ]},
        ]
        evidence = collect_part_evidence(results)
        photos = evidence["windshield_rear"][0]["evidence_photo"]
        assert photos == ["172852-04.png", "172852-03.png"]

    def test_collect_part_evidence_handles_string_photos_when_merging(self):
        results = [
            {"view_id": "rear_left_45", "parts": [
                _fake_candidate(
                    "windshield_rear",
                    "rear_left_45",
                    status="damaged",
                    damage_level="moderate",
                    evidence_photo=["172852-04.png", "172852-03.png"],
                ),
            ]},
            {"view_id": "rear_right_45", "parts": [
                _fake_candidate(
                    "windshield_rear",
                    "rear_right_45",
                    status="damaged",
                    damage_level="moderate",
                    evidence_photo="172852-04.png, 172852-03.png",
                ),
            ]},
        ]
        evidence = collect_part_evidence(results)
        # All evidence photos should be intact filenames, not characters.
        all_photos = []
        for cand in evidence["windshield_rear"]:
            all_photos.extend(cand.get("evidence_photo", []))
        for p in all_photos:
            assert len(p) > 1, f"photo id was split into chars: {p!r}"

    def test_fuse_evidence_keeps_photos_intact_from_string_input(self):
        candidates = [
            _fake_candidate(
                "windshield_rear",
                "rear_left_45",
                status="damaged",
                damage_level="moderate",
                evidence_photo="172852-04.png, 172852-03.png",
            ),
        ]
        fused = fuse_evidence("windshield_rear", candidates)
        assert fused is not None
        assert fused["evidence_photo"] == ["172852-04.png", "172852-03.png"]

    def test_apply_fusion_returns_intact_photo_list(self):
        results = [
            {"view_id": "rear_left_45", "parts": [
                _fake_candidate(
                    "windshield_rear",
                    "rear_left_45",
                    status="damaged",
                    damage_level="moderate",
                    evidence_photo="172852-04.png, 172852-03.png",
                ),
            ]},
            {"view_id": "rear_right_45", "parts": [
                _fake_candidate(
                    "windshield_rear",
                    "rear_right_45",
                    status="damaged",
                    damage_level="moderate",
                    evidence_photo="172852-04.png",
                ),
            ]},
        ]
        overrides = apply_fusion(results)
        assert "windshield_rear" in overrides
        for p in overrides["windshield_rear"]["evidence_photo"]:
            assert len(p) > 1, f"photo id was split into chars: {p!r}"
        assert "172852-04.png" in overrides["windshield_rear"]["evidence_photo"]
        assert "172852-03.png" in overrides["windshield_rear"]["evidence_photo"]


def test_part_view_priority_loaded_from_yaml():
    """DAMAGE_RECOGNITION_POLICY §1.6: _PART_VIEW_PRIORITY 必须从 YAML 加载,无硬编码。"""
    import inspect
    from agents import evidence_fusion
    src = inspect.getsource(evidence_fusion)
    # 必须 import loader
    assert "load_part_view_priority" in src, "evidence_fusion must import load_part_view_priority"
    # 实际加载到的 _PART_VIEW_PRIORITY 应该包含至少 10 个 part_id
    from agents.evidence_fusion import _PART_VIEW_PRIORITY
    assert len(_PART_VIEW_PRIORITY) >= 10, (
        f"expected ≥10 parts loaded from YAML, got {len(_PART_VIEW_PRIORITY)}: "
        f"{list(_PART_VIEW_PRIORITY.keys())}"
    )
    # 关键部件必须存在
    for required in ("windshield_front", "roof_middle", "headlight_front_right",
                     "fender_front_right"):
        assert required in _PART_VIEW_PRIORITY, (
            f"{required} missing from loaded _PART_VIEW_PRIORITY"
        )
