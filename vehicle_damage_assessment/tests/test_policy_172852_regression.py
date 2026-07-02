"""回归测试:DAMAGE_RECOGNITION_POLICY 修复后,172852 样本的右侧损毁必须被识别。

修复前 (基准):
  damaged_count = 7, 全部为左侧损毁。
  右侧漏判: right_a_pillar, right_b_pillar, door_front_right, door_rear_right,
           hood, mirror_right, roof_middle, windshield_front 全部 intact(错)。

修复后 (本测试期望):
  damaged_count >= 14(原 7 + 右侧 7+)
  MUST_NOT_BE_INTACT 列表中的部件: status != "intact"
  MUST_REMAIN_DAMAGED 列表中的部件: status == "damaged"(防止 §3.1 误翻转)
"""
import asyncio
import os
import sys

sys.path.insert(0, "/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment")

from agents import assessment_orchestrator


SAMPLE_DIR = "/Users/sky/Downloads/车顶闸调试样本_20260622/lead_172852"

# 这些部件在原版中全部被错判为 intact,新版必须为 damaged 或 uncertain(不能是 intact)。
MUST_NOT_BE_INTACT = [
    "right_a_pillar", "right_b_pillar",
    "door_front_right", "door_rear_right",
    "hood", "mirror_right", "roof_middle", "windshield_front",
]

# 这 7 个原判正确的受损部件必须保持 damaged(防止 §3.1 误翻转)。
MUST_REMAIN_DAMAGED = [
    "door_rear_left", "fender_rear_left",
    "pillar_c_left", "pillar_b_left",
    "windshield_rear", "taillight_rear_left", "trunk_lid",
]


async def main():
    if not os.path.isdir(SAMPLE_DIR):
        print(f"SKIP: 样本目录不存在 {SAMPLE_DIR}")
        return

    photos = [
        {"id": f, "path": f"{SAMPLE_DIR}/{f}", "name": f}
        for f in sorted(os.listdir(SAMPLE_DIR))
        if f.lower().endswith(".png")
    ]
    if not photos:
        print(f"SKIP: 样本目录无照片 {SAMPLE_DIR}")
        return

    result = await assessment_orchestrator(
        photos, {"brand": "", "model": "", "year": ""}
    )
    parts = {p["part_id"]: p for p in result.get("parts", [])}

    failed_not_intact = [
        p for p in MUST_NOT_BE_INTACT if parts.get(p, {}).get("status") == "intact"
    ]
    failed_remain_damaged = [
        p for p in MUST_REMAIN_DAMAGED
        if parts.get(p, {}).get("status") != "damaged"
    ]
    damaged_count = sum(1 for p in result["parts"] if p["status"] == "damaged")

    print(f"damaged_count = {damaged_count}")
    print(f"MUST_NOT_BE_INTACT failures ({len(failed_not_intact)}): {failed_not_intact}")
    print(f"MUST_REMAIN_DAMAGED failures ({len(failed_remain_damaged)}): {failed_remain_damaged}")

    assert not failed_not_intact, f"右侧漏判未修复: {failed_not_intact}"
    assert not failed_remain_damaged, f"原判正确的部件被翻转: {failed_remain_damaged}"
    assert damaged_count >= 14, f"damaged_count={damaged_count} < 14(原 7,目标 ≥14)"
    print("OK: 172852 样本回归测试通过")


if __name__ == "__main__":
    asyncio.run(main())
