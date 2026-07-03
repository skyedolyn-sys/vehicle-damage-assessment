# DAMAGE_RECOGNITION_POLICY

> 车辆损伤识别的策略约定。所有识别类智能体(Planner / Vision Subagent / Evidence Fusion / Synthesizer / Topology Comparator)必须遵守本文档。
> 文档版本: v1.1   最后更新: 2026-07-03

## 如何被引用

| 智能体 | 引用方式 | 强制级别 |
| --- | --- | --- |
| Planner | 系统 prompt 顶部插入「政策引用」段;`planner_agent.py` 的 `_stabilize_plan` 把 §1.2、§1.3 当作硬约束 | 硬约束(代码 enforce) |
| Vision Subagent | 系统 prompt 顶部插入「政策引用」段;输出的每个部件 JSON 必须包含 §2.1 规定的正向证据短语 | 硬约束(prompt + JSON 后处理校验) |
| Evidence Fusion | `evidence_fusion.py` 的 Rule 1 必须实现 §3.1 的 confidence-aware 逻辑 | 硬约束(代码 enforce) |
| Synthesizer | `_resolve_status_roof` 实现 §3.3;adjacency Rule 6 实现 §3.4 | 硬约束(代码 enforce) |
| Topology Comparator | `_infer_missing_roof_parts` 实现 §3.2 | 硬约束(代码 enforce) |
| Reviewer | 步骤 4 起 reviewer 改为**纯确定性**,不再调 LLM;复用 `evidence_fusion.apply_fusion` | 硬约束(代码 enforce) |
| Auxiliary Info | `auxiliary_info_extractor.py` 步骤 5 起改为纯确定性(VIN 正则 + WMI 字典) | 硬约束(代码 enforce) |

LLM prompt 失效时由 Python 层执行 §4 兜底。

## §1 Planner 阶段规则

### §1.1 禁止静默丢弃照片
任何 photo 不得因 planner 无法识别视角就被丢弃。每张必须落到 `wide_shot` / `close_up_damage` / `close_up_detail` / `interior` / `auxiliary` / `scene_intake` 之一。

### §1.2 新增 `scene_intake` 桶
当 LLM 对照片视角判断 confidence ≤ medium、或无法对应任一标准外观视角时,**不允许** 把 `view_id` 设为 `"unknown"`,应改为:
- `photo_type = "scene_intake"`
- `view_id = "scene_intake"`
- `confidence = "low"`
- `reason` 中说明 "未能确定视角,需进入次级识别队列"

`scene_intake` 桶照片**仍然必须 dispatch**(见 §1.3),不允许被 orchestrator 跳过。

### §1.3 close_up_damage 与 scene_intake 强制 dispatch
- `close_up_damage`:必须 dispatch 到 LLM 推断出的部件所在视角的 vision subagent,`_planner_reason` 注明「局部损伤特写」。
- `scene_intake`:必须 dispatch 到 intake subagent,输入是该张照片 + 所有已 dispatch 视角的 part 结论快照;输出对该照片所含部件的 damaged/uncertain 候选项,注入 fusion 阶段 damaged 候选池。

### §1.4 close-up 中的损伤证据不允许被否决
close_up_damage 照片明显聚焦在某个部件的破损处时,planner 不得因为该特写"不能证明其他区域完好"就将其归为 unknown 或 scene_intake;它必须立即进入对应视角的 subagent 输入。

### §1.5 vision 分类禁误伤(后续 PR 处理,本次未实现)
`_classify_photo_types` 的 LLM 调用存在非确定性:事故车辆侧面照片偶尔被错分成 auxiliary(VIN/铭牌),导致关键证据照片丢失。本次发现该问题,但 prompt 层修复不稳定。

**未来 PR 应基于确定性信号**(图像尺寸阈值、OCR 文本检测、文件名启发式)而不是 LLM prompt 来防止 auxiliary 误伤。当前已部分缓解:`_stabilize_plan` 把所有 non-exterior 桶照片通过 scene_intake 转给 intake subagent,即使有少量错分,scene_intake 仍然能让 intake subagent 对这些照片做视觉识别(虽然 view 信息丢失,但仍能产出部件级 evidence)。

### §1.6 确定性优先(DAMAGE_RECOGNITION_POLICY v1.1,2026-07-03)

**核心原则**:对每个 LLM 决策点,先问"能否用确定性信号替代?"——能替代则替代,不能替代再考虑 LLM。

#### §1.6.1 决策点矩阵

| ID | 决策点 | 关键度 | 是否替代 | 替代方式 |
|---|---|---|---|---|
| DP-1 | `vehicle_prior` 车型推断 | 重要 | 部分替代 | VIN WMI + PARTS_CATALOG 默认 sedan |
| DP-2 | `auxiliary_info_extractor` VIN/行驶证 | 弱 | **完全替代** | VIN 正则 + WMI 字典(8 主流品牌) |
| DP-3 | `planner._classify_photo_types` photo_type | 弱 | **完全替代** | 文件名关键词 + EXIF + 长宽比 |
| DP-4 | `planner._planner_agent_single_batch` view_id | 关键 | 部分替代 | filename hint 前置到 LLM 之前 |
| DP-5 | `planner._fallback_replan` 同 DP-4 | 重要 | 部分替代 | 同 DP-4 |
| DP-6 | `vision_subagent` checklist 部件 status | **关键(最高方差)** | **不可替代** | 视觉识别必须 LLM;但输入输出约束(prompt + checklist backfill) |
| DP-7 | `regional_worker` legacy | 弱 | N/A (legacy) | N/A |
| DP-8 | `reviewer_subagent` cross-validate | 弱 | **完全替代** | 复用 `evidence_fusion.apply_fusion` |
| DP-9 | `photo_locator` legacy | 弱 | N/A (legacy) | N/A |

#### §1.6.2 客户端 LLM 输出清洗(§5 兜底前的最后一道关)

`agents/minimax_client.py::clean_minimax_output` 必须按以下优先级提取 JSON:
1. 整段作为 JSON(无 think 标签时)
2. `<think>` 标签外的 JSON
3. `<think>` 标签内嵌的 JSON
4. narrative 中嵌入的 JSON 块(栈匹配找所有 `{ }` 平衡块)
5. 全部失败,返回 cleaned narrative(走调用方 fallback)

LLM 输出 17-33KB narrative + 0 个有效 JSON 的情况占 26% 失败率的 60%——这是 LLM 经常**完全不输出 JSON**(只输出思考),而非 JSON 格式错误。`_find_first_valid_json` 用栈匹配从 narrative 里挖 JSON 是必要的安全网。

#### §1.6.3 不再追加 LLM 调用

后续 PR 禁止引入"看起来更准"的 LLM 调用而不用确定性信号。任何新 LLM 决策点必须先通过下列审查:
- 能否用 filename / EXIF / 拓扑 / 规则 替代?
- 替代方案的 5 跑稳定性是否 ≥ LLM 方案?
- LLM 调用失败回退路径是什么?

## §2 Vision Subagent 阶段规则

### §2.1 正向证据要求(高敏感部件)
对以下高敏感部件,输出 `intact` 时 `notes` 必须包含至少一条正向证据短语,否则系统层降级为 `uncertain`:

| 部件类别 | 允许的正向证据短语 |
|---|---|
| windshield_front / rear, sunroof_glass | "玻璃面平整无裂纹", "无破碎", "玻璃与车架密封条完整" |
| roof_front / middle / rear | "钣金平整", "行李架对齐", "天窗框架无错位", "车顶线条连续" |
| pillar_a/b/c_left/right | "立柱直线无弯折", "与车顶/车门接缝齐平", "漆面连续" |
| hood, trunk_lid | "钣金弧线连续", "与两侧翼子板接缝齐平", "漆面无裂纹" |

正面证据**不得**写成"局部可见,未见异常"——这是 §2.2 明确禁止的边缘可见默认 intact 模式。

### §2.2 禁止 edge-only → intact
明确废除"如仅边缘可见且无异常,优先尝试标 intact"。改为:
> 当部件仅边缘可见(占比 < 30%)且无明显损伤痕迹时,**必须标 `uncertain`** 并在 notes 说明"仅边缘可见,需要更直接视角覆盖",**不得标 intact**。

### §2.3 cross_view_candidates 输出字段
每个 subagent 输出 JSON 顶层增加 `cross_view_candidates`:
```json
"cross_view_candidates": [
  {"part_id": "right_a_pillar", "reason": "本视角仅见 A 柱下段,相邻结构有塌陷信号,疑似损伤延伸", "target_view": "top"}
]
```
仅当本视角看到某部件疑似损伤但 confidence ≤ medium 时填写,目的是让 fusion 知道该部件需其他视角交叉验证。

### §2.4 高敏感部件 intact 系统降级
若 subagent 输出某高敏感部件 intact 但 positive_anchor 为空或不在 §2.1 允许的短语集合中,**Python 校验层**(在 `vision_subagent.py` 的 JSON 后处理)强制改为 `uncertain`,notes 追加"缺少正向证据,已按政策降级"。

## §3 跨视角一致性规则(Fusion / Synthesizer / Topology)

### §3.1 primary-view-intact 必须配合 confidence 检查
- 仅当 `primary_intact` 至少一条候选 `confidence == "high"` 时才允许 dominate(压制 damaged 候选);
- 若 primary intact 的 confidence 全部为 medium/low,fall through 到 damaged/missing 候选,但 `damage_level` 降一档(severe→moderate),标注 `_downgraded_for_low_confidence: true`;
- 高敏感部件即使 primary intact 是 high confidence,若存在任何 secondary view 的 damaged 候选 `damage_level ≥ moderate`,仍**保留 damaged 结论**(高敏感部件不允许被单视角 intact 完全屏蔽)。

### §3.2 missing-roof inference 收紧
- 相邻 roof 部件必须 `confidence == "medium" or "high"` 才允许作为推断 intact 的依据;low confidence 不再触发推断;
- 至少有一个相邻 roof 部件曾被 `top` 或 `close_up_damage` 视角直接观察过,否则不允许推断 intact;
- 推断 intact 的 `notes` 显式写"无直接视角覆盖,依据相邻 intact 推断;不覆盖高敏感部件 secondary damaged 信号"。

### §3.3 车顶 intact-bias 反转
原规则"secondary 冲突或不完整时,因后部结构损伤常溢出到车顶边缘,倾向 intact"——**作废**。
新规则:
- `roof_*` 任一候选为 damaged(不论来源 view)→ 不允许默认 intact,保留 damaged 并把 damage_level 至少设为 moderate;
- 所有候选为 uncertain 但存在至少一条 secondary 视图报告 damaged(即使 confidence=low)→ 改为 uncertain(不再默认 intact);
- 仅当所有候选都 intact 且至少一条 confidence ≥ medium 才输出 intact。

### §3.4 车门 adjacency Rule 6 收紧
原规则:door 仅靠 diagonal 视角报告 damaged 且相邻翼子板 severe → 翻 intact。新规则:
- 仅当该 door **完全没有 primary 视角证据**(front_left_45 / front_right_45 / left_90 / right_90 中无任何输入)时才允许 Rule 6 生效;
- 若有任一 primary 视角报告该门 damaged(即使 confidence=medium),**禁止** Rule 6 把它翻成 intact;
- 翻 intact 时必须附 `_adjacency_override: true` 标记,方便 reviewer 审计。

## §4 兜底防线(Python 层,LMM prompt 失效时执行)

### §4.1 高敏感部件严重损伤穿透
对 §2.1 列表中的高敏感部件,任意 view 输出 `damage_level = "severe"`,fusion 最终结论必须至少 `damaged moderate`,不允许被任意 primary-intact 信号覆盖。

### §4.2 视角完整性检查
每个 part 的 evidence_sources 中若**完全没有** primary view 贡献且没有任何 wide_shot 照片,status 强制 `uncertain`,damage_level=`unknown`,防止 topology inference 仅凭低质量相邻部件推出错误 intact。

### §4.3 矛盾对日志
fusion 最终结论与 N ≥ 2 个 secondary view 的 damaged 信号冲突时,向 `logs/policy_conflicts.log` 输出一行包含 `part_id`, `final_status`, `conflict_sources`, `rule_applied` 的结构化日志,便于事后审查。

### §4.4 Reviewer subagent 必须尊重
reviewer_subagent 系统 prompt 顶部增加一行:"任何被标记 `damaged` 的结论若被 §4.1 高敏感部件列表覆盖,保留;任何被标记 `intact` 但 evidence_sources 全部 confidence ≤ medium 的结论,标记为需人工复核。"

## §5 LLM 客户端清理规则(v1.1 新增)

### §5.1 为什么需要客户端清理
MiniMax-M3 经常输出 `<think>...</think>` 包裹的长 narrative(17-33KB),真正的 JSON 答案可能嵌入其中。`_extract_json_like_snippet` 仅尝试最外层 `{...}` 的 fallback 在内嵌嵌套时失效,导致 26% LLM 调用 JSON 解析失败。

### §5.2 清理优先级(已实现于 `agents/minimax_client.py`)
1. 整段作为 JSON(`{...}` 或 `[...]` 边界)
2. `<think>` 标签外的内容,栈匹配找有效 JSON 块
3. `<think>` 标签内嵌的内容,栈匹配找有效 JSON 块
4. 整段 narrative,栈匹配找所有平衡 `{ }` 块,返回第一个 `json.loads` 成功的
5. 全部失败 → 返回空字符串,调用方走 fallback

`_find_first_valid_json` 必须:
- 用栈匹配找所有 `{...}` / `[...]` 平衡块(不只是最外层)
- 跳过字符串内的 `{`(通过 in_string/escape 状态机)
- 只接受 `json.loads` 成功的 dict 或 list
- 按文本出现顺序,返回第一个有效的

### §5.3 单元测试覆盖
`tests/test_minimax_cleaner.py` 必须包含至少 10 个测试用例:
1. 纯 JSON 输入
2. JSON 在 `<think>` 外
3. JSON 在 `<think>` 内
4. narrative 中嵌入 JSON(无 think 标签)
5. narrative 里的 fake `{}` 必须被忽略
6. narrative + 多个 JSON 块(返回第一个有效的)
7. 嵌套 JSON `{a: {b: 1}}`
8. 列表 JSON `[{...}, {...}]`
9. 字符串内的 `{}` 不应被误抽(如 `"hello {world}"`)
10. 完全无 JSON(返回 cleaned narrative)