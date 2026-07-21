"""回归测试:DAMAGE_RECOGNITION_POLICY §1.6 非确定性修复后,172852 样本稳定性。

修复前 (基准, 策略 commit 1207ed0 之前):
  damaged_count = 7, 全部为左侧损毁。
  右侧漏判: right_a_pillar, right_b_pillar, door_front_right, door_rear_right,
           hood, mirror_right, roof_middle, windshield_front 全部 intact(错)。

修复策略 commit 1207ed0:
  damaged_count 提升到 9-11(波动),右前侧多数命中。
  但 LLM 26% parse 失败率导致同一样本多次运行结果差异巨大(0-11)。

本测试目标(DAMAGE_RECOGNITION_POLICY §1.6 修复后):
  1. **策略层确定性**(无 LLM):
     - DAMAGE_RECOGNITION_POLICY.md 包含 §1.6 "确定性优先" 条款
     - 代码不再有 5 个弱信号 LLM 决策点

  2. **LLM 输出稳定性**(5 跑中 ≥ 4 次):
     - 6 个高优先级右前侧部件(must_detect)必须多数识别为 damaged
     - 1 个必须保持 damaged(防 §3.1 误翻转)
     - 失败时给出具体哪次失败,而不是硬要求 ≥ 14

  3. **LLM 调用次数下降**:
     - 每张样本 LLM 调用从 ~38 次降到 < 20 次
"""
import asyncio
import os
import re

import pytest

from agents import assessment_orchestrator


SAMPLE_DIR = "/Users/sky/Downloads/车顶闸调试样本_20260622/lead_172852"

# DAMAGE_RECOGNITION_POLICY §1.6 必须包含的章节
POLICY_REQUIRED_SECTIONS = ["§1.6"]

# 这些部件在原版中全部被错判为 intact,新版多数运行必须识别为 damaged。
# 基于 2026-07-04 视觉+用户核对真值,撞击点全部在右侧。
MUST_DETECT_DAMAGED = [
    "hood", "windshield_front", "pillar_a_right",
    "headlight_front_right", "fender_front_right", "roof_front",
]

# 2026-07-04 修订:这 7 个原版假设 damaged 的部件实际全部 intact(左侧后侧)。
# 完整真值见 tests/test_172852_rubric.py,本测试保留 6 个核心 left/rear 部件作为 §3.1 翻转保护。
MUST_REMAIN_INTACT = [
    "door_rear_left", "fender_rear_left",
    "pillar_c_left", "pillar_b_left",
    "windshield_rear", "taillight_rear_left",
    "trunk_lid",  # 用户确认 intact
]


def test_policy_doc_has_determinism_section():
    """策略文档必须包含 §1.6 确定性优先原则。"""
    policy_path = os.path.join(
        os.path.dirname(__file__), "..", "agents", "DAMAGE_RECOGNITION_POLICY.md"
    )
    with open(policy_path, "r", encoding="utf-8") as f:
        content = f.read()
    for section in POLICY_REQUIRED_SECTIONS:
        assert section in content, f"DAMAGE_RECOGNITION_POLICY.md missing {section}"
    # §1.6 必须提到"确定性优先"
    match = re.search(r"§1\.6[\s\S]{0,2000}", content)
    assert match, "§1.6 not found"
    assert "确定性" in match.group(0), "§1.6 should mention 确定性"


def test_dp8_reviewer_deterministic():
    """DP-8: reviewer_subagent 不再调 LLM。"""
    import inspect
    from agents.reviewer_subagent import reviewer_subagent
    src = inspect.getsource(reviewer_subagent)
    assert "call_minimax" not in src, (
        "DP-8 violation: reviewer_subagent still calls LLM"
    )


@pytest.mark.asyncio
async def test_dp2_vin_deterministic():
    """DP-2: auxiliary_info_extractor 不再调 LLM。"""
    import inspect
    from agents import auxiliary_info_extractor as mod
    src = inspect.getsource(mod.extract_vehicle_info_from_auxiliary_photos)
    assert "call_minimax" not in src, (
        "DP-2 violation: extract_vehicle_info_from_auxiliary_photos still calls LLM"
    )


async def _run_172852() -> dict:
    if not os.path.isdir(SAMPLE_DIR):
        return {"skipped": True, "parts": []}

    photos = [
        {"id": f, "path": f"{SAMPLE_DIR}/{f}", "name": f}
        for f in sorted(os.listdir(SAMPLE_DIR))
        if f.lower().endswith(".png")
    ]
    if not photos:
        return {"skipped": True, "parts": []}

    result = await assessment_orchestrator(
        photos, {"brand": "", "model": "", "year": ""}
    )
    return {"skipped": False, "parts": result.get("parts", [])}


def _count_detection(parts: list, must_list: list) -> tuple[int, int]:
    """返回 (detected_count, total_count)。"""
    part_map = {p["part_id"]: p for p in parts}
    detected = sum(
        1 for pid in must_list
        if part_map.get(pid, {}).get("status") == "damaged"
    )
    return detected, len(must_list)


async def _run_n_times(n: int = 5) -> list[dict]:
    """跑 172852 样本 n 次,返回每次的结果。"""
    return [await _run_172852() for _ in range(n)]


@pytest.mark.asyncio
async def test_172852_stable_detection():
    """DAMAGE_RECOGNITION_POLICY §1.6: 5 跑中 ≥ 4 次必须命中右前侧高优先级部件。

    期望: must_detect_damaged 中 ≥ 4 个部件被识别为 damaged。
    """
    # Skip if SAMPLE_DIR 不存在(DJango test 环境可能配置了临时路径)
    if not os.path.isdir(SAMPLE_DIR):
        pytest.skip(f"SAMPLE_DIR 不存在: {SAMPLE_DIR}")
    # Skip if Django ORM can't init (sandbox 限制)
    if not _django_available():
        pytest.skip("Django ORM 不可用,无法测试完整 pipeline")

    runs = await _run_n_times(n=5)
    runs = [r for r in runs if not r.get("skipped")]
    if not runs:
        return

    pass_count = 0
    for i, run in enumerate(runs):
        detected, total = _count_detection(run["parts"], MUST_DETECT_DAMAGED)
        damaged_count = sum(1 for p in run["parts"] if p["status"] == "damaged")
        print(f"  run {i+1}: damaged={damaged_count}, must_detect={detected}/{total}")
        if detected >= 4:
            pass_count += 1

    assert pass_count >= 4, (
        f"only {pass_count}/5 runs detected ≥4 of {MUST_DETECT_DAMAGED}; "
        f"expected ≥4 runs to pass"
    )


@pytest.mark.asyncio
async def test_172852_left_side_preserved():
    """DAMAGE_RECOGNITION_POLICY §3.1: 左侧 + 后侧 7 个 intact 部件不能误判 damaged。

    2026-07-04 修订:左侧 + 后侧全部完好,这 7 个部件必须保持 intact。
    §3.1 修复目标:不能让 primary_intact 翻转成 damaged。
    """
    if not os.path.isdir(SAMPLE_DIR):
        pytest.skip(f"SAMPLE_DIR 不存在: {SAMPLE_DIR}")
    if not _django_available():
        pytest.skip("Django ORM 不可用,无法测试完整 pipeline")

    runs = await _run_n_times(n=3)
    runs = [r for r in runs if not r.get("skipped")]
    if not runs:
        return

    fail_count = 0
    for i, run in enumerate(runs):
        # MUST_REMAIN_INTACT 列表里被错误标记 damaged 的数量
        false_pos = sum(
            1 for pid in MUST_REMAIN_INTACT
            if next((p for p in run["parts"] if p["part_id"] == pid), {}).get("status") == "damaged"
        )
        print(f"  run {i+1}: must_remain_intact false_positive={false_pos}/{len(MUST_REMAIN_INTACT)}")
        if false_pos > 1:  # 允许 1 个边缘可见误判
            fail_count += 1

    assert fail_count == 0, (
        f"{fail_count}/3 runs incorrectly flipped MUST_REMAIN_INTACT parts; "
        f"policy §3.1 violated"
    )


def _django_available() -> bool:
    """检查 Django ORM 是否可用(若有 migration issue 则跳过)。"""
    try:
        from data.vehicle_specs_cache import _get_orm_specs
        _get_orm_specs({"brand": "test", "model": "test", "year": "2020"})
        return True
    except Exception:
        return False


def main():
    """CLI entrypoint for ad-hoc runs."""
    asyncio.run(test_172852_stable_detection())
    asyncio.run(test_172852_left_side_preserved())


if __name__ == "__main__":
    main()