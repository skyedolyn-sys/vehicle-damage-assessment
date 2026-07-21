# Multi-Agent 损伤评估流程

## 完整调用图（含 exterior / interior / 决策分支）

```
┌─────────────────────────────────────────────────────────────────────┐
│ HTTP POST /api/assess/<task_id>                                    │
│   ├─ 解析 task → 读取 UploadedPhoto 列表                              │
│   └─ 调用 assessment_orchestrator_stream                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MasterAgent.master_assessment_agent(files, vehicle_info)            │
│ 行为规范：master_agent_system.j2（不调 LLM，纯 Python 调度）          │
│                                                                       │
│ Step 1: 加载车辆先验                                                 │
│   vehicle_prior = await vehicle_prior_agent(vehicle_info)             │
│     └─ vehicle_prior_agent: 调 LLM 加载车型默认外观（缓存友好）       │
│                                                                       │
│ Step 2: 构建拓扑（确定性，33 part × 9 view）                         │
│   topology = build_vehicle_topology(vehicle_info, vehicle_prior)      │
│     └─ topology_builder: 合并 PARTS_CATALOG + 车辆规格              │
│                                                                       │
│ Step 3: Planner 4-class 分类（确定性，不调 LLM）                     │
│   plan = await planner_agent(files, vehicle_prior)                   │
│     ├─ _classify_photo_types: 按文件名+长宽比（确定性）             │
│     │   ├─ "行驶证" / "vin" / "车牌" → vehicle_info                  │
│     │   ├─ "内饰" / "驾驶舱" / "座椅" → interior                     │
│     │   ├─ 竖图 / 横图极端比例 → 可能是 close_up                      │
│     │   └─ 默认 → exterior                                           │
│     └─ 输出 {photo_classifications: [{photo_id, category, ...}]}    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4-class 决策分支（master_agent.master_assessment_agent line 83-89）  │
│                                                                       │
│   for each file in files:                                            │
│     category = classifications[file.id]                             │
│     ├─ category == "exterior" ────→ [A] ViewAgent Team 并行处理      │
│     ├─ category == "interior" ────→ [B] Interior Agent（**待实现**）  │
│     ├─ category == "vehicle_info"→ [C] VIN / 车牌提取（**待实现**）│
│     └─ category == "exclude" ────→ [D] 跳过，归入 excluded_photos     │
└─────────────────────────────────────────────────────────────────────┘

[A] Exterior 路径（已实现）
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ view_agent(photo, vehicle_prior)   ← ViewAgent Team (并行)         │
│ 行为规范：view_agent_system.j2 + view_agent_task.j2（双 role）        │
│                                                                       │
│ Step 1: 加载两个 .j2 模板                                            │
│   system_prompt = render_prompt_template("view_agent_system")        │
│     - 身份：车辆外观损伤识别专家                                      │
│     - 方法论：camera_position → 翻坐标系 → 部件判定                  │
│     - 输出协议：camera_analysis + primary_view + view_detections     │
│       + parts（强制两步）                                            │
│                                                                       │
│   task_prompt = render_prompt_template(                              │
│       "view_agent_task", photo_id=..., vehicle_name=...)             │
│     - 9 视角清单 + 屏幕方向映射表                                    │
│     - 33 部件清单                                                    │
│     - 字段约束                                                       │
│                                                                       │
│ Step 2: 构造 messages                                                │
│   [                                                                   │
│     {"role": "system", "content": system_prompt},                   │
│     {"role": "user", "content": [                                    │
│       {"type": "text", "text": task_prompt},                        │
│       image_content,            ← base64 image                      │
│     ]},                                                              │
│   ]                                                                  │
│                                                                       │
│ Step 3: await call_minimax(                                         │
│     messages, temperature=0.2, max_tokens=5000,                      │
│     response_format={"type": "json_object"}                          │
│   )                                                                  │
│                                                                       │
│ Step 4: extract_json → ViewAgentResult                              │
│   {photo_id, camera_analysis, primary_view, view_detections, parts}  │
│                                                                       │
│ Step 5: _dump_per_photo_verdict + _dump_minimax_raw                  │
│   → 写入 ~/vehicle_damage_assessment_view.log 和 _trace.log          │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MasterAgent 内部：PartEvidence 聚合                                │
│                                                                       │
│ for each ViewAgentResult:                                            │
│   for each observed part in result.parts:                           │
│     PartEvidence[part_id].observations.append({                     │
│       photo_id, view_id=primary_view, status, level, types, ...      │
│     })                                                               │
│                                                                       │
│ Aggregation:                                                        │
│   - missing > damaged > uncertain > intact                          │
│   - 强 primary（priority=0）damaged ≥1 + strong intact < 2 → damaged  │
│   - 强 primary intact ≥2 + strong damaged = 0 → intact               │
│   - 其他 → uncertain                                                  │
│   - confidence boost: ≥3 张一致 → high                              │
│   - conflict detection: status 不一致则 conflicting=True            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ reviewer_subagent + synthesizer_agent                              │
│   - reviewer: 确定性跨照片交叉验证 (evidence_fusion.apply_fusion)   │
│   - synthesizer: 屋顶规则 + 邻接规则 + 不存在 part 过滤            │
│     输出 PartActualState[]                                          │
│                                                                       │
│ topology_comparator                                                  │
│   - compare(topo, PartActualState[])                                 │
│   - 输出 DamageAssessment（parts, damaged_parts, intact_parts,        │
│     structural_patterns, overall_severity, primary_damage_zone）     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
                       DamageAssessment.to_legacy_result()
                             │
                             ▼
                   SSE event: result → 客户端
```

[B] Interior 路径（**待实现，任务 #4**）
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ interior_agent(photo, vehicle_prior)   ← 拟新增                    │
│ 行为规范：interior_agent_system.j2（**待创建**）                      │
│                                                                       │
│ 输入：interior 类别照片（已通过 planner 分类）                        │
│ 输出：{interior_components: [...], damage_summary: {...}}            │
│                                                                       │
│ 拟实现方案：                                                          │
│   - 创建 InteriorAgent（不在本轮做）                                 │
│   - 替代：interior_agent_system.j2 模板                              │
│   - 双 role 同样模式：system=interior_agent_system, user=task+image  │
│   - 视图清单：座椅/仪表台/方向盘/中控/门内饰/车顶灯                │
│   - 部件清单：interior-specific（不与 exterior 重叠）                 │
│   - 与 exterior parts 关系：interior 不影响 exterior 的 DamageAssessment│
│   - 输出合并：interior 字段独立追加到 DamageAssessment                │
│                                                                       │
│ 目前：interior 照片被 master_agent 过滤掉（line 87），不参与评估      │
└─────────────────────────────────────────────────────────────────────┘

[C] Vehicle Info 路径（**待实现，任务 #4**）
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ auxiliary_info_extractor.extract_vehicle_info_from_auxiliary_photos  │
│ 已有 stub，实现可以解析行驶证/VIN/铭牌                              │
│                                                                       │
│ 拟实现：                                                              │
│   - 调 LLM 解析图中文本（行驶证 / 车牌）                             │
│   - 提取：VIN、车主姓名、注册日期                                   │
│   - 附加到 DamageAssessment.vehicle_info 字段                       │
└─────────────────────────────────────────────────────────────────────┘

[D] Exclude 路径（已实现）
═══════════════════════════════════════════════════════════════════════

Excluded photos 列表返回给前端（不参与 DamageAssessment 计算）。
```

## 当前实现状态

| 路径 | 状态 | 任务 |
|---|---|---|
| [A] Exterior | ✅ 已实现（C/B 阶段已 commit） | 视角识别修对（02/08/09/19/21/28） |
| [B] Interior | ❌ 丢弃 | 任务 #4 |
| [C] Vehicle Info | ⚠️ 有 stub 未接入 | 任务 #4 |
| [D] Exclude | ✅ 已实现 | - |

## 每个 Agent 的 SP 文件对应

| Agent | 是否调 LLM | system role 模板 | task role 模板 | 状态 |
|---|---|---|---|---|
| MasterAgent | ❌（纯调度） | `master_agent_system.j2` | (无) | ✅ |
| PlannerAgent | ❌（确定性） | (无) | (无) | ✅ |
| ViewAgent | ✅ | `view_agent_system.j2` | `view_agent_task.j2` | ✅ |
| ReviewerAgent | ❌（确定性） | (无) | (无) | ✅ |
| SynthesizerAgent | ❌（确定性） | (无) | (无) | ✅ |
| VehiclePriorAgent | ✅ | 隐式 inline（`_SYSTEM_PROMPT`） | inline | ⚠️ 旧模式 |
| InteriorAgent | ❌（未实现） | (未创建) | (未创建) | ❌ |
| AuxiliaryExtractor | ⚠️ stub | (未创建) | (未创建) | ❌ |

## 建议下一步

1. **任务 #4**：实现 InteriorAgent + 接入 master_agent 分支
2. **任务 #2**：重试策略 + anomaly 检测
3. **vehicle_prior_agent 重构**：把 inline SP 拆成独立 .j2 模板（与 ViewAgent 一致）
4. **如果需要 mermaid/plantuml 图**：告诉我格式，我生成 .png 或 .svg
