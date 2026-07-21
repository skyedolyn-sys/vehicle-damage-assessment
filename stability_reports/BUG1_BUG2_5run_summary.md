# BUG1 + BUG2 修复后 172852 五轮分布 (2026-07-14)

修复内容：
- **BUG1（内饰污染）**: planner_agent 由"文件名+宽高比盲分类"改为视觉三分类
  (exterior/interior/vehicle_info)，外观关卡要求 has_vehicle_outline + position。
  无兜底——识别不出直接剥离出数据流。13/21/22（气囊/仪表台/车顶内衬特写）稳定
  归为 interior，pillar_a_left 证据链 [01,13,21,22] → [01]。
- **BUG2a（photo_id 格式）**: face_mapping 新增 `align_profile_ids`，模型回显
  `172852-07`（丢扩展名）时按"精确→去扩展名→位置"配对权威 id。07 号稳定获得
  face_prior、facing=unclear、view=None，不再落入无约束 legacy 路径自猜 front_left。

## 五轮结果

| run | detect | intact | flip | pillar_a_left | 误报部件 |
|---|---|---|---|---|---|
| bug2-1 | 10/12 | 13/19 | 1 | intact ✅ | windshield_rear |
| bug2-2 |  9/12 | 12/19 | 2 | damaged ❌ | — |
| bug2-3 | 11/12 | 17/19 | 1 | intact ✅ | windshield_rear, bumper_front |
| bug2-4 |  8/12 | 17/19 | 0 ✓ | intact ✅ | bumper_front, grille_front |
| bug2-5 |  7/12 | 13/19 | 2 | damaged ❌ | — |

- intact 均值 **13.8/19**（修复前 ~12.4），run3/4 达 17/19。
- pillar_a_left FP 从 6/6 → **2/5**（run2/run5）。07 号 id 问题已根治。
- detect 波动 7–11/12，漏报集中在车顶件（sunroof_glass/roof_middle/roof_rear）
  与 door_rear_right——模型从未真正观察（backfill FN）。

## 残留两条正交线

1. **BUG2b（FP 残留）**: face_profiler 对碎玻璃特写（photo 20，从车尾拍的碎后挡风+
   方向盘）前/后误判为 `front + left`，usable=True → view=front_left → pillar_a_left
   primary_view → FP。治"误判"。修复点：prompt 已有的"碎玻璃无法区分 front/rear 必须
   unclear"未生效且给了 high confidence，需强化或对"高 confidence 但损伤位置与 facing
   矛盾"降权。
2. **根因2 / Task#47（FN）**: roof_middle/sunroof 等车顶件模型从未真正观察 → 漏报。
   治"没看见"。修复点：flickering 视角双采样取并集降方差。

两者正交：BUG2b 影响 intact（误报），Task#47 影响 detect（漏报）。
