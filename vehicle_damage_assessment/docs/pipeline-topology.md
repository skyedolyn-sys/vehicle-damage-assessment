# 损伤识别流程拓扑图（v1.0-stable 权威版）

> 单源真相：本文档基于 2026-07-20 的 `damage` 分支 (HEAD = `ce411b4`,
> tag `v1.0-stable`) 的 172852 E2E 6 轮均值 **detect=9.0 / intact=14.33 / flip=1.0**。
>
> 代码位置会随演进漂移，引用具体行号时请 `grep -n` 复核。
>
> 维护原则：每次改动核心管线（新增 agent / 改并发 / 改 SSE 事件 / 改 silent 节点）后，同步更新本文档。

---

## 0. 关键术语 + 双管线分流

| 标记 | 含义 |
|---|---|
| **(主版本)** | v1.0-stable 是当前默认 + 唯一经 6 轮 E2E 验证的版本 |
| **(legacy)** | 仅 `?legacy=true` flag 时启用，已不再做主动维护 |
| **(silent)** | 调用存在但 **不写日志**，无法从 master.log 直接审计，需查返回值或加日志 |
| **(NEW)** | v1.0-stable 相对主仓基线 (`4b140aa`) 新增的节点 |
| **🔌** | 涉及 LLM 调用 (`call_minimax`) 的节点 |

### 双管线对比

| 步骤 | **Legacy** (`_run_assess_workflow`) | **新管线 (主版本)** (`master_assessment_agent`) |
|---|---|---|
| 入口 | `?legacy=true` | 默认（无 flag） |
| 车型先验 + 拓扑 | ✅ | ✅ |
| 4 类视角预筛 | ✅ | ✅（同路径，但 `face_path=False` 也用） |
| **face_profiler (NEW)** | ❌ 不调用 | ✅ `use_face_path=True` 时启用 |
| **face_mapping (NEW)** | ❌ view_agent 自推 facing | ✅ view_agent 只在 candidate 内评估 |
| view_agent 自判 view + Step B 翻转 | ✅ | ❌（由 face_prior 提供） |
| side_consistency (mirror flip, 旧) | ⚠️ 旧实现 | ❌（被 LR 替代） |
| side_consistency (view-out-of-primary, 新) | n/a | ✅ |
| LR 互斥 (NEW) | n/a | ✅ |
| Output validator (`validate_and_enrich`) | ✅ | ❌ 不调用（绕过） |
| 稳定性 | 6 轮未验 | 6 轮 E2E 验证 |

---

## 1. 完整链路（新管线 / 主版本 = `use_face_path=True`）

```
┌──────────────────────────────────────────────────────────────────────┐
│ ① 客户端 (浏览器)                                                   │
│   Form: brand, model, year + files[]                                 │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                │ POST /api/upload
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ② 落库  api/views.py upload_files()                                  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                │ GET /api/assess/<task_id>...&legacy=false (默认)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ③ SSE 入口  api/views.py assess_stream()                             │
│    feature flag → use_orchestrator（默认）/ legacy (legacy=true)      │
│    返回 StreamingHttpResponse(text/event-stream)                      │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ④ 主循环  master_assessment_agent (agents/master_agent.py:52)        │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │ 1. vehicle_prior_agent + build_vehicle_topology       (A1/A2)│  │
│    │ 2. planner_agent (4 类预筛)                              (B1) │  │
│    │ 3. face_profiler_agent (NEW, 双采样 N=2) + face_mapping      │  │
│    │     (派生 camera_side + candidate_parts)                (NEW) │  │
│    │ 4. ViewAgent Team (per-photo, on candidate_parts only)  (C1) │  │
│    │ 5. C2 子阶段（见 §2）：                                       │  │
│    │    ├─ _aggregate_part_evidence                              │  │
│    │    ├─ _check_side_consistency (view-out-of-primary)          │  │
│    │    ├─ _apply_facing_consensus (NEW, camera_side 跨照片共识) │  │
│    │    ├─ _resolve_left_right_conflicts (NEW, 3 层防御)         │  │
│    │    └─ _build_region_results                                  │  │
│    │ 6. reviewer_subagent + _apply_review_overrides         (C3) │  │
│    │ 7. synthesizer_agent (邻接规则 1-11)                  (D1)  │  │
│    │ 8. compare_topology                                       (D3) │  │
│    │ 9. DamageAssessment → API 返回                                │  │
│    └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. C2 阶段详解（v1.0-stable 的核心稳定层）

> C2 是 v1.0-stable 唯一**集中优化过**的阶段。拓扑图中其余阶段（除 face_profiler/face_mapping）的逻辑保持原状。

### C2 调用顺序（`agents/master_agent.py:192-219`）

```
view_results (N 张图 × per-photo parts[])
  │
  ▼ _aggregate_part_evidence  (line 292)
  │  - DAMAGE_RECOGNITION_POLICY §3.1/§4.1 (primary-strong gating)
  │  - Rule 1~7：缺损/双 strong-primary intact/单 strong-primary damaged/
  │             primary view damaged/secondary consensus/全 intact/默认
  │  - _boost_confidence (consistency × count)
  │  - conflicting flag (status 矛盾)
  ▼ part_evidence (dict[part_id] → {observations, aggregated_*})
  │
  ▼ _check_side_consistency  (line 495)
  │  - 仅 _left/_right 后缀部件
  │  - view_id ∉ primary view set → confidence 降 low（不改 suffix / status）
  │  - 触发率：~73 轮/many，最大 2 obs/次
  │
  ▼ _apply_facing_consensus  (NEW, line 420, commit 72557ce)
  │  - 统计 face_priors 中 camera_side 的多数 → 共识侧
  │  - 共识侧矛盾的"low-confidence" observation 降级
  │  - 触发率：~78 轮，1-2 obs/次
  │
  ▼ _resolve_left_right_conflicts  (NEW, line 578, commit ce411b4)
  │  3 层防御:
  │   L1 bilateral: ≥2 pair 双侧 damaged → 跳过互斥（保留 raw）
  │   L2 unilateral: 单侧 damaged → 另一侧 damaged observation 降级 uncertain
  │   L3 camera_side: 照片 camera_side 与共识矛盾 → light damage 降级
  │  - 触发率：~28 轮，5.6 obs/轮
  │
  ▼ _build_region_results  (line 815)
     - 按区域组织 → list[RegionResult] → 传给 C3 reviewer + D1 synthesizer
```

### 各子节点的激活度（最近 28 轮 E2E）

| 子节点 | 实际激活 | obs/触发 | 必要 |
|---|---|---|---|
| `_aggregate_part_evidence` | 100% 必调用 | 全量处理 | ✅ |
| `_check_side_consistency` | 100% 调用 / 73 触发 | 1-2 | ⚠️ 部分被 LR 覆盖 |
| `_apply_facing_consensus` | 100% / 78 触发 | 1-2 | ✅ |
| `_resolve_left_right_conflicts` | 100% / 28 触发 | 5.6 | ✅ |
| `_build_region_results` | 100% | 全量 | ✅ |

---

## 3. 已发现的拓扑图与代码不同步问题

| 问题 | 现象 | 行动 |
|---|---|---|
| **D4 sunroof_propagation** | 拓扑图有但代码已删（abfe365 提交后被某次清理移除） | ✅ **本版本删** |
| **D5 output_validator** | 拓扑图说新管线也调用，实际**只在 legacy** (`api/views_legacy.py`) 路径调用 | ✅ **本版本标 "仅 legacy"** |
| C3 / D1 / D3 silent | 确定性规则 silent（无 logger），无法从 master.log 审计 | ⚠️ 建议加 `logger.info(summary)` |
| C2 `_check_side_consistency` 与 LR 部分重叠 | side_consistency 降 low + LR 互斥降级双层保护 | ⚠️ 当前双层冗余，无害但需审计 |

---

## 4. LLM 调用总数

| 阶段 | LLM 调用 | 并发 | 备注 |
|---|---|---|---|
| A1 `vehicle_prior_agent` 🔌 | 1× | — | temp=0.1 |
| B1 `planner_agent` 🔌 | 0~1× | 批量 | 启发式能解就跳过 LLM |
| **NEW** face_profiler 🔌 | **2× (双采样 N=2)** | 串行 | temp=0.1 + temp=0.4 |
| C1 view_agent 🔌 | N× | Semaphore(6~10) | 主要瓶颈 |
| C3 reviewer | 0× | — | silent |
| D1 synthesizer | 0× | — | silent |
| D3 compare_topology | 0× | — | silent |

**总计**：**N + 3~4 次 LLM 调用**（N 外饰 + 双采样 2 次 + prior 1 次 + planner 0~1 次）

---

## 5. SSE 事件时间线

```
t=0       step(0) 开始识别
t≈1s      step(1) 车型先验
          vehicle_prior
          topology
t≈3s      step(2) 视角规划 + face_profiler（NEW，可选 2× LLM）
          locations / coverage_summary
t≈3-30s   step(3) 视觉识别
          subagent_partial(view_id) × N
t≈end     step(4) 复核审查
          review_partial / review_summary
t≈end+1s  step(5) 生成报告
          result                                ← 终态
          complete
```

---

## 6. 关键模块对照表（v1.0-stable 当前 commit）

| 阶段 | 模块 | 文件 | LLM? | silent? | 角色 |
|---|---|---|---|---|---|
| ① | `upload_files` | api/views.py | ❌ | ❌ | 落库 |
| ② | `assess_stream` | api/views.py | ❌ | ❌ | SSE 入口 + legacy flag |
| ④ | `master_assessment_agent` | agents/master_agent.py:52 | ❌ | ❌ | 主循环编排 |
| A1 | `vehicle_prior_agent` | agents/vehicle_prior.py | 🔌 1× | ❌ | 提取车型规格 |
| A2 | `build_vehicle_topology` | agents/topology_builder.py | ❌ | ❌ | 部件集合 + 邻接表 |
| B1 | `planner_agent` | agents/planner_agent.py | 🔌 0~1× | ❌ | 4 类视角预筛 |
| **NEW** | `face_profiler_agent` | agents/face_profiler.py | 🔌 2× (N=2 双采样) | ❌ | facing + face coverage |
| **NEW** | `build_face_prior` | agents/face_mapping.py | ❌ | ❌ | camera_side 派生 + candidate_parts |
| **NEW** | `_apply_facing_consensus` | master_agent.py:420 | ❌ | ❌ | 跨照片 camera_side 共识 |
| C1 | `view_agent` | agents/view_agent.py | 🔌 N× | ❌ | per-photo 损伤评估 |
| C2 | `_aggregate_part_evidence` | master_agent.py:292 | ❌ | ❌ | 部件级聚合（primary-strong gating） |
| C2 | `_check_side_consistency` | master_agent.py:495 | ❌ | ❌ | view-out-of-primary 降权 |
| **NEW** | `_resolve_left_right_conflicts` | master_agent.py:578 | ❌ | ❌ | 3 层 LR 防御 |
| C2 | `_build_region_results` | master_agent.py:815 | ❌ | ❌ | 区域组织 |
| C3 | `reviewer_subagent` | agents/reviewer_subagent.py:31 | ❌ | **silent** | fusion + advisory（summary 在返回值里） |
| D1 | `synthesizer_agent` | agents/synthesizer.py:815 | ❌ | **silent** | 邻接规则 1-11 + status resolution |
| D3 | `compare_topology` | agents/topology_comparator.py | ❌ | **silent** | 拓扑一致性执行 |
| ~~D4~~ | `~~_apply_sunroof_roof_propagation~~` | n/a | ❌ | n/a | **已删除**（abfe365 被清理） |
| D5 | `~~output_validator~~` (legacy only) | agents/output_validator.py | ❌ | ❌ | **新管线不调用** |

---

## 7. silent 节点的审计建议

> 这三个节点当前 **silent**（无日志），无法从 master.log 直接审计实际触发了哪些规则、覆盖了多少部件。建议至少给 `reviewer_subagent` 加一行 `logger.info(summary)` 把 summary 写进日志。

```python
# 建议在 reviewer_subagent.py summary 后加：
logger.info("[reviewer] %s", summary)
logger.info("[reviewer] needs_rephotography=%d", len(needs_rephotography))
```

synthesizer + topology 类似。不修复不影响功能，但**无法验证 v1.0-stable 的 silent 节点正在做正确的事**。

---

## 8. 配置项

- `MAX_CONCURRENT_API_CALLS`：ViewAgent 并发上限（config.py）
- `IMAGE_MAX_WIDTH`：ViewAgent 图片压缩宽度（view_agent.py 顶部，v1.0-stable 用 512px）
- `REQUEST_TIMEOUT`：MiniMax 单次请求超时（config.py）
- `face_profiler N` (硬编码 2)：双采样次数（face_profiler.py）
- `face_profiler temperature` (硬编码 0.1 + 0.4)：双采样温度（face_profiler.py）

## 9. 日志位置

- `~/vehicle_damage_assessment_master.log`：MasterAgent 主循环 + LR/face 前置节点
- `~/vehicle_damage_assessment_faceprofiler.log`：face_profiler LLM 调用
- `~/vehicle_damage_assessment_minimax.log`：所有 MiniMax 调用 + 重试
- `~/vehicle_damage_assessment_orchestrator.log`：编排层
- `~/minimax_diagnostic_<call_id>.txt`：解析失败的原始输出

## 10. 维护提示

- **新增 silent 节点**：先加 `logger.info` 再上线，避免成为 hidden 行为
- **修改 C2 子节点**：调用顺序敏感（aggregate → side_consistency → facing_consensus → LR → build_region），不能随意插入
- **改 silent 节点（C3/D1/D3）**：先用返回值序列化 + 临时加日志验证 3 轮后再合入
- **改邻接规则 1-11**：同步 D1 描述；`synthesizer.py` 里 `DEFENSIVE_ADJACENCY_RULES` 块
- **改 master_agent.py**：C2 改动要保持"确定性 / 无 LLM / 不改 suffix" 三原则

---

## 版本历史

- **v1.0-stable** (HEAD `ce411b4`, 2026-07-19) — 当前主版本
  - 4 个 clean commits over 4b140aa baseline
  - 6 轮 E2E: detect=9.0 / intact=14.33 / flip=1.0
- **4b140aa** (baseline) — 6 轮 E2E: detect=9.17 / intact=14.67 / flip=1.33
- **b392555** — 6 轮 E2E: detect=8.7 / intact=11.7 / flip=2.0

主版本相对基线的核心收益：**flip -25%**（1.33 → 1.0），主碰撞侧误报率大幅降低。
