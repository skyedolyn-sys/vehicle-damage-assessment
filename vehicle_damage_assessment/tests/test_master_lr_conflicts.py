"""Unit tests for master_agent._resolve_left_right_conflicts."""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
try:
    import django

    django.setup()
except Exception:
    pass

from agents.master_agent import _resolve_left_right_conflicts


def _obs(pid, status, level="moderate", conf="high", photo_id="p1"):
    return {
        "part_id": pid,
        "status": status,
        "damage_level": level,
        "confidence": conf,
        "photo_id": photo_id,
    }


def _entry(pid, observations):
    return {
        "part_id": pid,
        "observations": observations,
        "aggregated_status": "uncertain",
        "aggregated_level": "unknown",
        "aggregated_confidence": "low",
        "conflicting": False,
    }


def _fp(usable=True, camera_side="right"):
    return {"usable": usable, "camera_side": camera_side}


def test_bilateral_damage_skips_mutual_exclusion():
    """≥2 对左右都 damaged → 尊重原始识别，不加互斥。"""
    evidence = {
        # pair 1: door_front both damaged
        "door_front_left": _entry("door_front_left", [_obs("door_front_left", "damaged")]),
        "door_front_right": _entry("door_front_right", [_obs("door_front_right", "damaged")]),
        # pair 2: fender both damaged
        "fender_rear_left": _entry("fender_rear_left", [_obs("fender_rear_left", "damaged")]),
        "fender_rear_right": _entry("fender_rear_right", [_obs("fender_rear_right", "damaged")]),
    }
    face_priors = {}
    _resolve_left_right_conflicts(evidence, face_priors)
    # 所有 observations 保持 damaged（不被降级）
    for entry in evidence.values():
        for obs in entry["observations"]:
            assert obs["status"] == "damaged"


def test_unilateral_damage_triggers_mutual_exclusion():
    """只有一侧 damaged → 另一侧降级 uncertain。"""
    evidence = {
        "door_front_left": _entry("door_front_left", [_obs("door_front_left", "intact")]),
        "door_front_right": _entry("door_front_right", [_obs("door_front_right", "damaged", conf="high")]),
    }
    face_priors = {"p1": _fp(camera_side="right")}
    _resolve_left_right_conflicts(evidence, face_priors)
    # door_front_left 的 intact observation 不受影响（只降级 damaged）
    assert evidence["door_front_left"]["observations"][0]["status"] == "intact"
    # door_front_right 保持 damaged
    assert evidence["door_front_right"]["observations"][0]["status"] == "damaged"


def test_unilateral_damage_downgrades_opposite_damaged():
    """左侧 damaged，右侧 intact → 右侧的 damaged observation（如果有）被降级。

    真实场景：左侧 damaged，右侧 view_agent 也报了 damaged（幻觉），但右侧
    同时有 intact observation。Layer 2 检测到"只有左侧确定 damaged"→ 右侧
    所有 damaged observation 降级 uncertain。
    """
    evidence = {
        "mirror_left": _entry("mirror_left", [_obs("mirror_left", "damaged", conf="high")]),
        "mirror_right": _entry("mirror_right", [
            _obs("mirror_right", "intact", conf="high"),  # 主 observation
            _obs("mirror_right", "damaged", conf="low"),   # 幻觉 observation
        ]),
    }
    face_priors = {"p1": _fp(camera_side="left")}
    _resolve_left_right_conflicts(evidence, face_priors)
    # mirror_left 保持 damaged
    assert evidence["mirror_left"]["observations"][0]["status"] == "damaged"
    # mirror_right 的 damaged observation 被降级
    right_obs = evidence["mirror_right"]["observations"]
    assert right_obs[0]["status"] == "intact"  # intact 不受影响
    assert right_obs[1]["status"] == "uncertain"  # damaged → uncertain
    assert right_obs[1]["confidence"] == "low"
    assert "_lr_excluded" in right_obs[1]


def test_suspect_photo_light_damage_downgraded():
    """camera_side 与共识矛盾的照片，light damage 降级 uncertain。"""
    evidence = {
        "pillar_a_left": _entry("pillar_a_left", [
            _obs("pillar_a_left", "damaged", level="light", photo_id="suspect1"),
        ]),
    }
    face_priors = {
        "p1": _fp(camera_side="right"),
        "p2": _fp(camera_side="right"),
        "p3": _fp(camera_side="right"),
        "suspect1": _fp(camera_side="left"),  # 与共识矛盾
    }
    _resolve_left_right_conflicts(evidence, face_priors)
    obs = evidence["pillar_a_left"]["observations"][0]
    assert obs["status"] == "uncertain"
    assert obs["confidence"] == "low"
    assert obs.get("_consensus_penalty") is True


def test_suspect_photo_severe_damage_survives():
    """camera_side 矛盾照片但 damage_level=severe → 保留。"""
    evidence = {
        "door_rear_left": _entry("door_rear_left", [
            _obs("door_rear_left", "damaged", level="severe", photo_id="suspect1"),
        ]),
        # 需要一个对立面让 Layer 2 不触发（两侧都有 damaged 时不互斥单个）
        "door_rear_right": _entry("door_rear_right", [
            _obs("door_rear_right", "damaged", level="severe", photo_id="p1"),
        ]),
    }
    face_priors = {
        "p1": _fp(camera_side="right"),
        "p2": _fp(camera_side="right"),
        "suspect1": _fp(camera_side="left"),
    }
    _resolve_left_right_conflicts(evidence, face_priors)
    # severe damage 保留（不被 Layer 3 降级）
    obs = evidence["door_rear_left"]["observations"][0]
    assert obs["status"] == "damaged"


def test_no_consensus_no_penalty():
    """camera_side 平票时无共识，不做 Layer 3 降级。"""
    evidence = {
        "pillar_a_left": _entry("pillar_a_left", [
            _obs("pillar_a_left", "damaged", level="light", photo_id="p1"),
        ]),
    }
    face_priors = {
        "p1": _fp(camera_side="left"),
        "p2": _fp(camera_side="right"),
    }
    _resolve_left_right_conflicts(evidence, face_priors)
    # light damage 保留（无共识 → 无 suspect photos）
    obs = evidence["pillar_a_left"]["observations"][0]
    assert obs["status"] == "damaged"
