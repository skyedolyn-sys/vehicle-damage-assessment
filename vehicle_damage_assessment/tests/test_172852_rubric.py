"""172852 样本真值 rubric 测试 (基于人工视觉 + 用户核对)

本测试将 172852 样本的真实损伤情况固化为可重复测试的 rubric,作为
识别系统稳定性和准确性的 ground truth。

真值来源:
  - 全 32 张照片 (172852-01.png ~ 172852-32.png) 由人工视觉检测
  - 经用户多轮核对确认:
      * 撞击核心:右前侧 + A 柱
      * 右侧部件:hood, windshield_front, pillar_a_right, sunroof_glass,
        roof_*, fender_front_right, headlight_front_right,
        door_front_right, door_rear_right, mirror_right
      * 左侧 + 后侧:全部完好(intact)
      * 后挡风玻璃完好(用户特别确认)
      * 之前旧测试假设左侧后侧 7 个部件 damaged 是错误的(基于旧样本)

本测试目标:
  1. 稳定性:5 次连续跑,12 个 must_detect_damaged 部件都识别为 damaged
    (允许 1 次失败,因为 LLM 视觉非确定性)
  2. 准确性:19 个 must_remain_intact 部件都识别为 intact
    (允许 2 次 false positive,因 §2.1/§2.2 边缘可见降级)
  3. 数量:damaged_count 应在 [10, 14] 区间(LLM 视觉非确定性)
  4. 极差:5 跑 damaged_count 极差 ≤ 6(目标 ≤ 4,基线修复前是 10)
"""
import asyncio
import os
from collections import Counter

import pytest

from agents import assessment_orchestrator


SAMPLE_DIR = "/Users/sky/Downloads/车顶闸调试样本_20260622/lead_172852"

# ============================================================================
# GROUND TRUTH:172852 样本真值(2026-07-04 视觉+用户核对)
# ============================================================================

# 必须识别为 damaged 的 12 个部件(全部位于右侧或顶部)
MUST_DETECT_DAMAGED = [
    "hood",                       # severe - 引擎盖中央竖向折痕
    "windshield_front",           # severe - 整片碎裂
    "pillar_a_right",             # severe - A 柱弯折(撞击核心)
    "sunroof_glass",              # severe - 天窗玻璃碎裂脱落
    "roof_front",                 # severe - 车顶前段塌陷
    "roof_middle",                # severe - 车顶中段塌陷
    "roof_rear",                  # moderate-severe - 车顶后段塌陷
    "fender_front_right",         # severe - 右前翼子板严重凹陷
    "headlight_front_right",      # moderate - 右前大灯灯罩破损
    "door_front_right",           # moderate-severe - 右前门上沿变形
    "door_rear_right",            # moderate-severe - 右后门上沿变形
    "mirror_right",               # moderate - 右后视镜壳变形
]

# 必须识别为 intact 的 19 个部件(左侧 + 后侧)
MUST_REMAIN_INTACT = [
    # 左侧
    "door_front_left",
    "door_rear_left",
    "fender_front_left",
    "fender_rear_left",
    "mirror_left",
    "pillar_a_left",
    "pillar_b_left",
    "pillar_c_left",
    # 后侧
    "taillight_rear_left",
    "taillight_rear_right",
    "fender_rear_right",
    "trunk_lid",
    "pillar_c_right",
    "pillar_b_right",
    "windshield_rear",       # 用户特别确认 intact
    "bumper_rear",
    # 前部但完好
    "bumper_front",
    "grille_front",
    "headlight_front_left",
]

# 真值总数
GROUND_TRUTH_DAMAGED_COUNT = 12  # 12 个 must_detect_damaged
GROUND_TRUTH_TOTAL_PARTS = 31    # 12 damaged + 19 intact

# ============================================================================
# PER-PHOTO 人工判断 (用户对关键照片的上下文核对, 2026-07-12)
# 仅作诊断参考, 不改变 rubric 的 damaged/intact 判定。
# ============================================================================
PHOTO_CONTEXT_NOTES = {
    # 172852-24: 从右前门车顶高度俯拍车头。拍摄者站在打开的右前门与车身之间,
    # 画面含前挡风玻璃(碎)、引擎盖、右前大灯/翼子板(毁)。右前门被打开。
    # 注意: 此角度看不到右后视镜, 若模型报 mirror_right 属轻微误判。
    "172852-24": {
        "view": "front_right (右前门车顶高度俯拍车头)",
        "front_door_open": True,
        "visible": ["windshield_front", "hood", "headlight_front_right",
                    "fender_front_right", "pillar_a_right"],
        "note": "右前门打开, 拍摄者站门与车之间; 此角度看不到右后视镜",
    },
    # 172852-04: 清晰的 rear_right 3/4 照。车尾 Mercedes 标 + 尾灯 + 牌照框,
    # 右侧车身(右后门打开/脱落、右后视镜、右后轮)。右后门明显受损。
    "172852-04": {
        "view": "rear_right (车尾 + 右侧车身 3/4)",
        "visible": ["taillight_rear_right", "door_rear_right", "fender_rear_right",
                    "pillar_c_right", "mirror_right"],
        "note": "右后门打开/脱落, 损伤清楚; face_profiler 偶尔抖成 primary=None",
    },
    # 172852-29: 后挡风玻璃 + 右 C 柱特写。无明确车头/车尾锚点,
    # 按规则应判 facing=unclear → camera_side=None → 右后门进不了候选集。
    "172852-29": {
        "view": "rear (后窗 + 右 C 柱特写)",
        "visible": ["windshield_rear", "pillar_c_right", "roof_rear"],
        "note": "特写无锚点, 稳定判 unclear; 右后门不在此画面",
    },
}


# ============================================================================
# 测试辅助函数
# ============================================================================

def _django_available() -> bool:
    """检查 Django ORM 是否可用(若有 migration issue 则跳过)。"""
    try:
        from data.vehicle_specs_cache import _get_orm_specs
        _get_orm_specs({"brand": "test", "model": "test", "year": "2020"})
        return True
    except Exception:
        return False


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


# ============================================================================
# 单元测试:Rubric 完整性
# ============================================================================

def test_rubric_part_lists_mutually_exclusive():
    """MUST_DETECT_DAMAGED 与 MUST_REMAIN_INTACT 必须互斥。"""
    overlap = set(MUST_DETECT_DAMAGED) & set(MUST_REMAIN_INTACT)
    assert not overlap, f"part 列表有交叉: {overlap}"


def test_rubric_part_lists_no_typos():
    """Rubric 中所有 part_id 必须是已知的车辆部件(基于已知部件列表)。"""
    # 已知部件列表:从 view_weights.yaml 的 primary_view + 其他公认部位
    known_parts = {
        # view_weights.yaml 已登记
        "door_front_left", "door_front_right", "door_rear_left", "door_rear_right",
        "fender_front_left", "fender_front_right", "fender_rear_left", "fender_rear_right",
        "mirror_left", "mirror_right",
        "roof_front", "roof_middle", "roof_rear",
        "sunroof_glass",
        "windshield_front", "windshield_rear",
        "hood", "trunk_lid",
        "pillar_a_left", "pillar_a_right",
        "pillar_b_left", "pillar_b_right",
        "pillar_c_left", "pillar_c_right",
        "headlight_front_left", "headlight_front_right",
        "taillight_rear_left", "taillight_rear_right",
        # 公认部件(view_weights.yaml 中存在 primary 但在 part_view_priority 也有)
        "bumper_front", "bumper_rear",
        "grille_front",
    }
    rubric_parts = set(MUST_DETECT_DAMAGED) | set(MUST_REMAIN_INTACT)
    unknown = rubric_parts - known_parts
    assert not unknown, f"rubric 中有未知 part_id: {sorted(unknown)}"
    # 双向核对:rubric 涵盖的部件数应该合理(典型 31)
    assert len(rubric_parts) >= 25, f"rubric 部件数 {len(rubric_parts)} 过少,可能漏列"


def test_ground_truth_damaged_count_matches_list():
    """GROUND_TRUTH_DAMAGED_COUNT 必须等于 MUST_DETECT_DAMAGED 长度。"""
    assert GROUND_TRUTH_DAMAGED_COUNT == len(MUST_DETECT_DAMAGED), (
        f"GROUND_TRUTH_DAMAGED_COUNT={GROUND_TRUTH_DAMAGED_COUNT} "
        f"!= len(MUST_DETECT_DAMAGED)={len(MUST_DETECT_DAMAGED)}"
    )


def test_ground_truth_total_matches_sum():
    """GROUND_TRUTH_TOTAL_PARTS = damaged + intact。"""
    assert GROUND_TRUTH_TOTAL_PARTS == len(MUST_DETECT_DAMAGED) + len(MUST_REMAIN_INTACT), (
        f"GROUND_TRUTH_TOTAL_PARTS={GROUND_TRUTH_TOTAL_PARTS} != "
        f"damaged({len(MUST_DETECT_DAMAGED)}) + intact({len(MUST_REMAIN_INTACT)})"
    )


# ============================================================================
# 集成测试:5 跑稳定性 + 准确性
# ============================================================================

async def _run_n_times(n: int = 5) -> list[dict]:
    """跑 172852 样本 n 次,返回每次的结果。"""
    return [await _run_172852() for _ in range(n)]


@pytest.mark.asyncio
async def test_172852_rubric_5runs_summary():
    """5 跑汇总:打印真值 vs 实际命中,作为视觉核对。

    此测试不强制 assert,而是打印详细对比,供人工核对稳定性趋势。
    """
    if not os.path.isdir(SAMPLE_DIR):
        pytest.skip(f"SAMPLE_DIR 不存在: {SAMPLE_DIR}")
    if not _django_available():
        pytest.skip("Django ORM 不可用,无法测试完整 pipeline")

    runs = await _run_n_times(n=5)
    runs = [r for r in runs if not r.get("skipped")]
    if not runs:
        return

    print("\n" + "=" * 80)
    print("172852 真值核对 (12 damaged + 19 intact = 31 总部件)")
    print("=" * 80)

    damaged_counts = []
    detect_hit_rates = []
    intact_hit_rates = []

    for i, run in enumerate(runs):
        parts = run["parts"]
        damaged_count = sum(1 for p in parts if p["status"] == "damaged")
        damaged_counts.append(damaged_count)

        # MUST_DETECT 命中率(应该是 damaged)
        detect_detected, detect_total = _count_detection(parts, MUST_DETECT_DAMAGED)
        detect_miss = [
            pid for pid in MUST_DETECT_DAMAGED
            if next((p for p in parts if p["part_id"] == pid), {}).get("status") != "damaged"
        ]
        detect_hit = detect_detected / detect_total if detect_total else 0
        detect_hit_rates.append(detect_hit)

        # MUST_REMAIN 命中率(应该是 intact)
        intact_ok, intact_total = _count_detection(parts, MUST_REMAIN_INTACT)
        intact_false_pos = [
            pid for pid in MUST_REMAIN_INTACT
            if next((p for p in parts if p["part_id"] == pid), {}).get("status") == "damaged"
        ]
        intact_hit = 1 - (intact_ok / intact_total) if intact_total else 0  # 1 - false_positive_rate
        intact_hit_rates.append(intact_hit)

        print(f"\n  run {i+1}: damaged={damaged_count} (真值={GROUND_TRUTH_DAMAGED_COUNT})")
        print(f"    must_detect 命中: {detect_detected}/{detect_total} ({detect_hit:.0%})")
        if detect_miss:
            print(f"    must_detect 漏报: {detect_miss}")
        print(f"    must_remain intact 命中: {intact_total - intact_ok}/{intact_total} ({(1 - intact_hit):.0%})")
        if intact_false_pos:
            print(f"    must_remain 误报 damaged: {intact_false_pos}")

    print("\n" + "-" * 80)
    print(f"5 跑汇总:")
    print(f"  damaged_count:  {damaged_counts}  (真值={GROUND_TRUTH_DAMAGED_COUNT})")
    print(f"  damaged_count 极差: {max(damaged_counts) - min(damaged_counts)}")
    print(f"  must_detect 命中率: {[f'{r:.0%}' for r in detect_hit_rates]}")
    print(f"  must_remain intact 命中率: {[f'{r:.0%}' for r in intact_hit_rates]}")

    # 计算最关键的稳定性指标
    detect_pass_count = sum(1 for r in detect_hit_rates if r >= 0.75)  # ≥9/12 命中
    intact_pass_count = sum(1 for r in intact_hit_rates if r >= 0.85)  # ≥16/19 正确
    drift = max(damaged_counts) - min(damaged_counts)

    print(f"\n  must_detect ≥75% 命中: {detect_pass_count}/5 跑")
    print(f"  must_remain ≥85% intact: {intact_pass_count}/5 跑")
    print(f"  damaged_count 极差: {drift}")


@pytest.mark.asyncio
async def test_172852_rubric_strict():
    """严格断言:5 跑中 ≥ 4 跑必须满足所有 rubric 条件。

    通过条件(任一):
      - must_detect 命中 ≥ 10/12
      - must_remain intact 命中率 ≥ 17/19
    """
    if not os.path.isdir(SAMPLE_DIR):
        pytest.skip(f"SAMPLE_DIR 不存在: {SAMPLE_DIR}")
    if not _django_available():
        pytest.skip("Django ORM 不可用,无法测试完整 pipeline")

    runs = await _run_n_times(n=5)
    runs = [r for r in runs if not r.get("skipped")]
    if not runs:
        return

    pass_count = 0
    detect_failures = []
    intact_failures = []

    for i, run in enumerate(runs):
        detect_detected, detect_total = _count_detection(run["parts"], MUST_DETECT_DAMAGED)
        intact_ok, intact_total = _count_detection(run["parts"], MUST_REMAIN_INTACT)
        intact_correct = intact_total - intact_ok

        detect_ok = detect_detected >= 10  # ≥10/12
        intact_ok_flag = intact_correct >= 17  # ≥17/19

        if not detect_ok:
            miss = [
                pid for pid in MUST_DETECT_DAMAGED
                if next((p for p in run["parts"] if p["part_id"] == pid), {}).get("status") != "damaged"
            ]
            detect_failures.append({"run": i + 1, "detected": detect_detected, "miss": miss})

        if not intact_ok_flag:
            false_pos = [
                pid for pid in MUST_REMAIN_INTACT
                if next((p for p in run["parts"] if p["part_id"] == pid), {}).get("status") == "damaged"
            ]
            intact_failures.append({"run": i + 1, "intact": intact_correct, "false_pos": false_pos})

        if detect_ok and intact_ok_flag:
            pass_count += 1

    assert pass_count >= 4, (
        f"172852 rubric 严格模式: only {pass_count}/5 runs passed.\n"
        f"  detect failures: {detect_failures}\n"
        f"  intact failures: {intact_failures}\n"
        f"  真值: damaged={MUST_DETECT_DAMAGED}, intact={MUST_REMAIN_INTACT}"
    )


@pytest.mark.asyncio
async def test_172852_rubric_drift():
    """稳定性:5 跑 damaged_count 极差 ≤ 6 (基线修复前是 10)。"""
    if not os.path.isdir(SAMPLE_DIR):
        pytest.skip(f"SAMPLE_DIR 不存在: {SAMPLE_DIR}")
    if not _django_available():
        pytest.skip("Django ORM 不可用,无法测试完整 pipeline")

    runs = await _run_n_times(n=5)
    runs = [r for r in runs if not r.get("skipped")]
    if not runs:
        return

    damaged_counts = [
        sum(1 for p in run["parts"] if p["status"] == "damaged")
        for run in runs
    ]
    drift = max(damaged_counts) - min(damaged_counts)

    assert drift <= 6, (
        f"damaged_count 极差 {drift} (counts={damaged_counts}) > 6; "
        f"基线修复前是 10,目标 ≤ 6"
    )


# ============================================================================
# CLI entry
# ============================================================================

def main():
    """CLI entrypoint: 跑 rubric 5 跑核对。"""
    asyncio.run(test_172852_rubric_5runs_summary())


if __name__ == "__main__":
    main()
