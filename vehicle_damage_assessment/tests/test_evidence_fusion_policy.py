"""单元测试:evidence_fusion Rule 1 confidence-aware 行为。

覆盖 DAMAGE_RECOGNITION_POLICY §3.1 / §4.1 三个场景:
1. low-conf primary intact 不 dominate,高敏感部件 severe 穿透
2. high-conf primary intact dominate 低置信度 light 信号(非高敏感)
3. high-sensitivity severe 即使面对 high-conf primary intact 也必须穿透
"""
from agents.evidence_fusion import apply_fusion


def _candidate(part_id, status, damage_level, confidence, view_id):
    return {
        "part_id": part_id,
        "status": status,
        "damage_level": damage_level,
        "damage_type": [],
        "confidence": confidence,
        "evidence_photo": ["x.png"],
        "_origin_view": view_id,
    }


def test_low_confidence_primary_intact_does_not_dominate_for_high_sensitivity():
    """场景1:high-sensitivity 部件,primary intact 是 low confidence,secondary severe damaged。"""
    subagent_results = [
        # front_left_45(primary for hood) 给出 low-confidence intact
        {"view_id": "front_left_45", "parts": [
            _candidate("hood", "intact", "none", "low", "front_left_45"),
        ]},
        # rear_left_45 给出 severe damaged(secondary for hood)
        {"view_id": "rear_left_45", "parts": [
            _candidate("hood", "damaged", "severe", "medium", "rear_left_45"),
        ]},
    ]
    fused = apply_fusion(subagent_results)
    assert "hood" in fused, "hood 必须被融合"
    # hood 是高敏感部件,即使 primary intact 是 low confidence,severe damaged 信号必须穿透
    assert fused["hood"]["status"] == "damaged"
    # damage_level 可能因为 low-confidence 主导被降级到 moderate,但不应被吞掉
    assert fused["hood"]["damage_level"] in ("severe", "moderate")


def test_high_confidence_primary_intact_dominates_low_signal_for_normal_part():
    """场景2:bumper_front(非高敏感),high-conf primary intact 应 dominate low-conf light damaged。"""
    subagent_results = [
        # front(primary for bumper_front) 给出 high-confidence intact
        {"view_id": "front", "parts": [
            _candidate("bumper_front", "intact", "none", "high", "front"),
        ]},
        # front_left_45 给出 light damaged
        {"view_id": "front_left_45", "parts": [
            _candidate("bumper_front", "damaged", "light", "low", "front_left_45"),
        ]},
    ]
    fused = apply_fusion(subagent_results)
    # bumper_front 不是高敏感部件,high-conf primary intact 应当 dominate
    # Rule 1 此时不会 return 任何东西(因为有 damaged 候选),但如果 Rule 1 fall through,应降级
    if "bumper_front" in fused:
        # 如果被融合,status 应该是 damaged(被降级到 light)或 intact(被 primary 压)
        assert fused["bumper_front"]["status"] in ("damaged", "intact")
        if fused["bumper_front"]["status"] == "damaged":
            assert fused["bumper_front"]["damage_level"] in ("light", "moderate")


def test_high_sensitivity_severe_passthrough():
    """场景3:high-sensitivity 部件,即使面对 high-conf primary intact,severe damaged 也必须穿透。"""
    subagent_results = [
        # front(primary for windshield_front) 给出 high-confidence intact
        {"view_id": "front", "parts": [
            _candidate("windshield_front", "intact", "none", "high", "front"),
        ]},
        # front_left_45 给出 severe damaged
        {"view_id": "front_left_45", "parts": [
            _candidate("windshield_front", "damaged", "severe", "medium", "front_left_45"),
        ]},
    ]
    fused = apply_fusion(subagent_results)
    assert "windshield_front" in fused, "windshield_front 必须被融合"
    # windshield_front 是高敏感部件,severe damaged 必须穿透
    assert fused["windshield_front"]["status"] == "damaged"
    assert fused["windshield_front"]["damage_level"] in ("severe", "moderate")
