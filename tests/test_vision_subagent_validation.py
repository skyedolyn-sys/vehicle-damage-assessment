"""Tests for vision_subagent Python validation layer.

DAMAGE_RECOGNITION_POLICY §2.1 — high-sensitivity parts marked ``intact``
must include a positive evidence phrase in notes, otherwise the Python layer
downgrades them to ``uncertain``.

DAMAGE_RECOGNITION_POLICY §2.2 — high-sensitivity parts seen only at the
edge (≤1 evidence photo) with low/medium confidence must be downgraded to
``uncertain``.
"""

from agents.vision_subagent import (
    _check_positive_anchor,
    _enforce_positive_anchor,
    _enforce_edge_visible,
    _should_downgrade_edge_visible,
    _positive_anchor_category,
    _HIGH_SENSITIVITY_PARTS,
)


class TestHighSensitivityParts:
    def test_includes_safety_critical_glass_and_roof(self):
        expected = {
            "windshield_front", "windshield_rear", "sunroof_glass",
            "roof_front", "roof_middle", "roof_rear",
            "pillar_a_left", "pillar_a_right",
            "pillar_b_left", "pillar_b_right",
            "pillar_c_left", "pillar_c_right",
            "hood", "trunk_lid",
        }
        assert expected.issubset(_HIGH_SENSITIVITY_PARTS)

    def test_part_category_mapping(self):
        assert _positive_anchor_category("windshield_front") == "windshield"
        assert _positive_anchor_category("sunroof_glass") == "windshield"
        assert _positive_anchor_category("roof_middle") == "roof"
        assert _positive_anchor_category("pillar_a_left") == "pillar"
        assert _positive_anchor_category("hood") == "hood"
        assert _positive_anchor_category("trunk_lid") == "trunk"
        # Non-high-sensitivity returns empty:
        assert _positive_anchor_category("door_front_left") == ""


class TestPositiveAnchorCheck:
    """DAMAGE_RECOGNITION_POLICY §2.1: intact 高敏感部件必须含正向证据短语。"""

    def test_non_intact_always_passes(self):
        # damaged/uncertain/missing 都放过,不需正向证据
        for status in ("damaged", "missing", "uncertain"):
            assert _check_positive_anchor("roof_front", status, "任何 notes")
            assert _check_positive_anchor("pillar_a_left", status, "")

    def test_non_high_sensitivity_always_passes(self):
        # 普通部件即使 intact 且无正向证据,也不需降级(规则不适用)
        assert _check_positive_anchor("door_front_left", "intact", "")
        assert _check_positive_anchor("bumper_front", "intact", "random notes")

    def test_high_sensitivity_intact_with_anchor_passes(self):
        # 含正向证据短语,不降级
        assert _check_positive_anchor("roof_front", "intact", "钣金平整")
        assert _check_positive_anchor("pillar_a_left", "intact", "立柱直线无弯折")
        assert _check_positive_anchor("windshield_front", "intact", "玻璃面平整无裂纹")
        assert _check_positive_anchor("hood", "intact", "钣金弧线连续")

    def test_high_sensitivity_intact_without_anchor_downgrades(self):
        # 缺正向证据短语的 intact 高敏感部件 → 应该被拒绝(返回 False)
        assert not _check_positive_anchor("roof_front", "intact", "未见异常")
        assert not _check_positive_anchor("pillar_a_left", "intact", "看起来没事")
        assert not _check_positive_anchor("windshield_front", "intact", "")
        assert not _check_positive_anchor("hood", "intact", "主体可见")

    def test_partial_phrase_still_passes(self):
        # 子串匹配生效("钣金平整" 出现在 "车顶钣金平整且无变形" 中)
        assert _check_positive_anchor("roof_front", "intact", "车顶钣金平整且无变形")


class TestEnforcePositiveAnchor:
    """_enforce_positive_anchor 是 mutation-style 的 in-place 降级函数。"""

    def test_downgrades_intact_high_sensitivity_without_anchor(self):
        part = {
            "part_id": "roof_front",
            "status": "intact",
            "damage_level": "none",
            "notes": "未见异常",
        }
        _enforce_positive_anchor(part)
        assert part["status"] == "uncertain"
        assert part["damage_level"] == "unknown"
        assert "已按政策 §2.1 降级" in part["notes"]
        assert part["_positive_anchor_downgraded"] is True

    def test_preserves_damaged_high_sensitivity(self):
        part = {
            "part_id": "roof_front",
            "status": "damaged",
            "damage_level": "severe",
            "notes": "车顶凹陷",
        }
        _enforce_positive_anchor(part)
        assert part["status"] == "damaged"
        assert part.get("_positive_anchor_downgraded") is not True

    def test_preserves_intact_non_high_sensitivity(self):
        part = {
            "part_id": "door_front_left",
            "status": "intact",
            "damage_level": "none",
            "notes": "车门外板",
        }
        _enforce_positive_anchor(part)
        assert part["status"] == "intact"
        assert "_positive_anchor_downgraded" not in part

    def test_preserves_intact_high_sensitivity_with_anchor(self):
        part = {
            "part_id": "pillar_a_left",
            "status": "intact",
            "damage_level": "none",
            "notes": "立柱直线无弯折",
        }
        _enforce_positive_anchor(part)
        assert part["status"] == "intact"
        assert "_positive_anchor_downgraded" not in part

    def test_appends_note_when_existing(self):
        part = {
            "part_id": "trunk_lid",
            "status": "intact",
            "damage_level": "none",
            "notes": "后备箱盖无变形",
        }
        _enforce_positive_anchor(part)
        assert "后备箱盖无变形" in part["notes"]
        assert part["notes"].startswith("后备箱盖无变形")


class TestEdgeVisibleDowngrade:
    """DAMAGE_RECOGNITION_POLICY §2.2: 边缘可见(≤1 张证据照片)+ high sensitivity → 降级。"""

    def test_non_high_sensitivity_never_downgrades(self):
        assert not _should_downgrade_edge_visible("bumper_front", {"status": "intact"})

    def test_high_sensitivity_damaged_never_downgrades(self):
        # damaged/missing status 不受 §2.2 影响
        assert not _should_downgrade_edge_visible("roof_front", {"status": "damaged"})
        assert not _should_downgrade_edge_visible("roof_front", {"status": "missing"})

    def test_close_up_damage_with_intact_passes(self):
        # close_up_damage 表明聚焦,允许 intact
        assert not _should_downgrade_edge_visible(
            "roof_front", {"status": "intact", "confidence": "low", "photo_type": "close_up_damage"}
        )

    def test_high_confidence_intact_passes(self):
        # 高置信度 + wide_shot + intact 不降级
        assert not _should_downgrade_edge_visible(
            "roof_front", {"status": "intact", "confidence": "high", "photo_type": "wide_shot"}
        )

    def test_wide_shot_low_confidence_intact_downgrades(self):
        # wide_shot 但 low confidence + 高敏感 → 降级
        assert _should_downgrade_edge_visible(
            "roof_front", {"status": "intact", "confidence": "low", "photo_type": "wide_shot"}
        )

    def test_default_photo_type_low_confidence_intact_downgrades(self):
        # 未知 photo_type 但 low confidence 边缘可见 → 降级
        assert _should_downgrade_edge_visible(
            "pillar_a_left", {"status": "intact", "confidence": "low"}
        )


class TestEnforceEdgeVisible:
    """_enforce_edge_visible 是 mutation-style 的 in-place 降级函数。"""

    def test_downgrades_edge_visible_high_sensitivity(self):
        part = {
            "part_id": "roof_front",
            "status": "intact",
            "damage_level": "none",
            "notes": "钣金平整",
            "evidence_photo": ["photo_01"],
            "confidence": "low",
            "photo_type": "wide_shot",
        }
        _enforce_edge_visible(part)
        assert part["status"] == "uncertain"
        assert part["damage_level"] == "unknown"
        assert "已按政策 §2.2 降级" in part["notes"]
        assert part["_edge_visible_downgraded"] is True

    def test_skips_if_already_downgraded_by_section_2_1(self):
        part = {
            "part_id": "roof_front",
            "status": "uncertain",
            "damage_level": "unknown",
            "notes": "缺正向证据",
            "_positive_anchor_downgraded": True,
        }
        _enforce_edge_visible(part)
        # 不应该二次降级,也不应该重复追加 §2.2 note
        assert "_edge_visible_downgraded" not in part
        assert "已按政策 §2.2 降级" not in part["notes"]

    def test_close_up_damage_passes_through(self):
        part = {
            "part_id": "roof_front",
            "status": "intact",
            "damage_level": "none",
            "notes": "钣金平整",
            "evidence_photo": ["photo_01"],
            "confidence": "low",
            "photo_type": "close_up_damage",
        }
        _enforce_edge_visible(part)
        assert part["status"] == "intact"

    def test_high_confidence_intact_passes_through(self):
        part = {
            "part_id": "roof_front",
            "status": "intact",
            "damage_level": "none",
            "notes": "钣金平整",
            "evidence_photo": ["photo_01", "photo_02"],
            "confidence": "high",
            "photo_type": "wide_shot",
        }
        _enforce_edge_visible(part)
        assert part["status"] == "intact"
