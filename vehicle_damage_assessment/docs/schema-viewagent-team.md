# ViewAgent Team 架构 — Schema 与接口规范

> 版本：v1.0  
> 状态：设计定稿，待实现  
> 适用范围：车辆损伤评估多 Agent 流水线

## 1. 目标与原则

当前 PlannerAgent 同时承担“大类分类 + 视角判定 + 损伤语义”，职责过重。本文档定义新的 Agent 边界与数据接口。

| Agent | 职责 | 输入 | 输出 |
|---|---|---|---|
| **PlannerAgent** | 4-class 初筛 | 全部照片 + 车辆先验 | 每张照片的 category |
| **ViewAgent Team** | 单张外观照片的视觉判定 | category == exterior 的照片 | 视角 + 可见 part + 损伤 |
| **MasterAgent** | 任务分发、聚合、校准、串联 | 照片集 + 车辆信息 | DamageAssessment |

核心原则：

1. **PlannerAgent 不区分视角**：只判断照片是否属于外观，不输出 view_id。
2. **ViewAgent 只做外观判定**：只接收 exterior 照片，不处理 interior/vehicle_info/exclude。
3. **Part 清单固定**：不根据车型动态过滤，保证输出稳定、prompt 可固化。
4. **View 清单固定**：9 个外观视角，命名去掉数字，仅保留方位字母。
5. **置信度分层校准**：ViewAgent 输出原始分数，MasterAgent 结合证据数量二次校准。

## 2. 固定 Part 清单

总计 **33** 个 part，按区域分组。该清单在 ViewAgent prompt、拓扑比较、结构一致性规则中全部固定使用。

### front（8 个）

| part_id | part_name |
|---|---|
| hood | 引擎盖 |
| bumper_front | 前保险杠 |
| headlight_front_left | 左前大灯 |
| headlight_front_right | 右前大灯 |
| grille_front | 前格栅 |
| fender_front_left | 左前翼子板 |
| fender_front_right | 右前翼子板 |
| windshield_front | 前挡风玻璃 |

### rear（7 个）

| part_id | part_name |
|---|---|
| trunk_lid | 后备箱盖 |
| tailgate | 尾门 |
| bumper_rear | 后保险杠 |
| taillight_rear_left | 左后尾灯 |
| taillight_rear_right | 右后尾灯 |
| windshield_rear | 后挡风玻璃 |

### left（7 个）

| part_id | part_name |
|---|---|
| door_front_left | 左前门 |
| door_rear_left | 左后门 |
| mirror_left | 左后视镜 |
| fender_rear_left | 左后翼子板 |
| pillar_a_left | 左A柱 |
| pillar_b_left | 左B柱 |
| pillar_c_left | 左C柱 |

### right（7 个）

| part_id | part_name |
|---|---|
| door_front_right | 右前门 |
| door_rear_right | 右后门 |
| mirror_right | 右后视镜 |
| fender_rear_right | 右后翼子板 |
| pillar_a_right | 右A柱 |
| pillar_b_right | 右B柱 |
| pillar_c_right | 右C柱 |

### roof（5 个）

| part_id | part_name |
|---|---|
| roof_front | 车顶前部 |
| roof_middle | 车顶中部 |
| roof_rear | 车顶后部 |
| sunroof_glass | 天窗玻璃 |
| roof_rack | 车顶行李架 |

## 3. 固定 View 清单与视角 → Part 映射

### 3.1 View 清单（9 个外观视角）

| view_id | view_name |
|---|---|
| front | 车头正前 |
| front_left | 车头左前 |
| front_right | 车头右前 |
| rear | 车尾正后 |
| rear_left | 车尾左后 |
| rear_right | 车尾右后 |
| left | 车辆左侧 |
| right | 车辆右侧 |
| top | 车顶俯视 |

### 3.2 视角 → Part 映射

ViewAgent 判定一张照片时，只从对应视角的 part 清单中选择可见 part。

| view_id | 可见 part 清单 |
|---|---|
| front | hood, bumper_front, headlight_front_left, headlight_front_right, grille_front, fender_front_left, fender_front_right, windshield_front |
| front_left | hood, bumper_front, headlight_front_left, fender_front_left, mirror_left, door_front_left, pillar_a_left, roof_front |
| front_right | hood, bumper_front, headlight_front_right, fender_front_right, mirror_right, door_front_right, pillar_a_right, roof_front |
| rear | trunk_lid, tailgate, bumper_rear, taillight_rear_left, taillight_rear_right, windshield_rear |
| rear_left | trunk_lid, tailgate, bumper_rear, taillight_rear_left, fender_rear_left, door_rear_left, mirror_left, pillar_c_left, roof_rear |
| rear_right | trunk_lid, tailgate, bumper_rear, taillight_rear_right, fender_rear_right, door_rear_right, mirror_right, pillar_c_right, roof_rear |
| left | door_front_left, door_rear_left, mirror_left, fender_rear_left, pillar_a_left, pillar_b_left, pillar_c_left |
| right | door_front_right, door_rear_right, mirror_right, fender_rear_right, pillar_a_right, pillar_b_right, pillar_c_right |
| top | roof_front, roof_middle, roof_rear, sunroof_glass, roof_rack |

## 4. 置信度模型（方案 3 + 4）

### 4.1 输出字段

ViewAgent 输出两层置信度：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model_confidence_score` | float [0,1] | 视觉模型自评的原始分数 |
| `confidence` | enum | 校准后的离散值：`high` / `medium` / `low` |

### 4.2 ViewAgent 内校准（方案 3）

对损伤判定，结合以下信号加权平均：

```python
def calibrate_damage_confidence(raw: dict) -> float:
    signals = []

    # 1. 模型原始分数
    signals.append(raw["model_confidence_score"])

    # 2. 描述具体程度
    desc = raw.get("description", "")
    if any(kw in desc for kw in ["明显", "严重", "大面积", "变形", "凹陷", "断裂"]):
        signals.append(0.9)
    elif any(kw in desc for kw in ["轻微", "小", "局部", "细微"]):
        signals.append(0.6)
    else:
        signals.append(0.4)

    # 3. 损伤类型数量
    types = [t for t in raw.get("damage_types", []) if t != "none"]
    if len(types) >= 2:
        signals.append(0.85)
    elif len(types) == 1:
        signals.append(0.7)
    else:
        signals.append(0.5)

    # 4. 状态一致性
    if raw["status"] == "intact" and "无" in desc:
        signals.append(0.9)
    elif raw["status"] == "damaged" and types:
        signals.append(0.8)
    else:
        signals.append(0.4)

    score = sum(signals) / len(signals)
    return score


def score_to_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"
```

### 4.3 主视角判定阈值

```python
def pick_primary_view(view_detections: list) -> str | None:
    if not view_detections:
        return None
    best = max(view_detections, key=lambda x: x["confidence_score"])
    if best["confidence_score"] < 0.5:
        return None  # 没有可靠主视角，标记为 uncertain
    return best["view_id"]
```

### 4.4 MasterAgent 证据增强（方案 4）

对同一个 `part_id`，当多张照片结论一致时提升置信度：

```python
def boost_confidence(base: str, evidence_count: int) -> str:
    if evidence_count >= 3:
        return "high"
    if evidence_count == 2:
        return base  # 保持原级
    return base
```

冲突处理：如果多张照片对同一 part 给出不同 `status`，不 boost，交给 ReviewerAgent 解决。

## 5. Agent 接口

### 5.1 PlannerAgent

```python
async def planner_agent(
    photos: list[dict[str, Any]],
    vehicle_prior: dict[str, Any],
) -> PlannerResult:
    ...
```

输出 `PlannerResult`：

```json
{
  "photo_classifications": [
    {
      "photo_id": "172852-01",
      "category": "exterior",
      "confidence_score": 0.95,
      "confidence": "high",
      "reason": "车身外观照片，可见车头左前区域"
    },
    {
      "photo_id": "172852-13",
      "category": "vehicle_info",
      "confidence_score": 0.98,
      "confidence": "high",
      "reason": "行驶证照片"
    }
  ]
}
```

`category` 枚举：`exterior`, `interior`, `vehicle_info`, `exclude`。

### 5.2 ViewAgent

```python
async def view_agent(
    photo: dict[str, Any],
    vehicle_prior: dict[str, Any],
) -> ViewAgentResult:
    ...
```

`photo` 字段：

```json
{
  "id": "172852-01",
  "path": "/path/to/172852-01.png",
  "category": "exterior"
}
```

输出 `ViewAgentResult`：

```json
{
  "photo_id": "172852-01",
  "primary_view": "front_left",
  "view_detections": [
    {
      "view_id": "front_left",
      "confidence_score": 0.92,
      "is_primary": true
    },
    {
      "view_id": "left",
      "confidence_score": 0.34,
      "is_primary": false
    }
  ],
  "parts": [
    {
      "part_id": "hood",
      "part_name": "引擎盖",
      "status": "damaged",
      "damage_level": "moderate",
      "damage_types": ["deformation", "dent"],
      "model_confidence_score": 0.85,
      "confidence": "high",
      "description": "引擎盖中部有明显凹陷变形"
    },
    {
      "part_id": "headlight_front_left",
      "part_name": "左前大灯",
      "status": "intact",
      "damage_level": "none",
      "damage_types": ["none"],
      "model_confidence_score": 0.9,
      "confidence": "high",
      "description": "左前大灯完整无破损"
    }
  ]
}
```

字段约束：

- `status`: `intact`, `damaged`, `missing`, `uncertain`
- `damage_level`: `none`, `light`, `moderate`, `severe`, `unknown`
- `damage_types`: 从固定枚举中选择，可多选
  - `none`
  - `scratch`（划痕）
  - `dent`（凹陷）
  - `deformation`（变形）
  - `crack`（裂纹）
  - `breakage`（碎裂/断裂）
  - `paint_loss`（掉漆）
  - `corrosion`（锈蚀）
- `confidence`: `high`, `medium`, `low`

### 5.3 MasterAgent

```python
async def master_assessment_agent(
    files: list[dict[str, Any]],
    vehicle_info: dict[str, str],
) -> DamageAssessment:
    ...
```

内部流程：

1. 调用 `vehicle_prior_agent` 获取先验
2. 调用 `build_vehicle_topology` 构建拓扑
3. 调用 `planner_agent` 做 4-class 初筛
4. 对 `category == exterior` 的照片并行调用 `view_agent`
5. 聚合：按 `part_id` 合并多张照片证据，置信度增强
6. 调用 `reviewer_subagent` 处理冲突
7. 调用 `topology_comparator` 做结构一致性
8. 调用 `synthesizer` 输出最终 part states
9. 包装为 `DamageAssessment`

## 6. 数据 Schema

### 6.1 ViewAgentResult

```python
class ViewDetection(TypedDict):
    view_id: str
    confidence_score: float
    is_primary: bool


class PartObservation(TypedDict):
    part_id: str
    part_name: str
    status: Literal["intact", "damaged", "missing", "uncertain"]
    damage_level: Literal["none", "light", "moderate", "severe", "unknown"]
    damage_types: list[Literal[
        "none", "scratch", "dent", "deformation",
        "crack", "breakage", "paint_loss", "corrosion"
    ]]
    model_confidence_score: float
    confidence: Literal["high", "medium", "low"]
    description: str


class ViewAgentResult(TypedDict):
    photo_id: str
    primary_view: str | None
    view_detections: list[ViewDetection]
    parts: list[PartObservation]
```

### 6.2 MasterAgent 中间聚合结构

```python
class PartEvidence(TypedDict):
    part_id: str
    observations: list[PartObservation]  # 来自不同照片
    aggregated_status: str
    aggregated_level: str
    aggregated_confidence: str
    conflicting: bool
```

## 7. 迁移说明

| 现有文件 | 处理方式 |
|---|---|
| `agents/planner_agent.py` | **大幅删减**：移除视角分配逻辑，只保留 4-class 分类 |
| `agents/vision_subagent.py` | **重构为 `agents/view_agent.py`**：输入单张照片，输出视角 + part + 损伤 |
| `agents/assessment_orchestrator.py` | **保留为兼容层**：内部调用 `master_assessment_agent`，对外接口不变 |
| `agents/view_mapping.py` | **更新 view 命名**：`front_left_45` → `front_left`，`left_90` → `left`，其余类推 |
| `agents/topology_comparator.py` | **基本不变**：输入格式仍为 part states |
| `agents/synthesizer.py` | **调整输入**：从按 view 聚合改为按 photo/part 聚合 |
| `agents/reviewer_subagent.py` | **输入调整**：从 view-level 冲突改为 photo-level/part-level 冲突 |

## 8. 待实现事项

1. 更新 `view_mapping.py` 中的 view 命名与映射。
2. 重写 `planner_agent.py`，移除 view assignment，只输出 4-class 分类。
3. 新建 `agents/view_agent.py`，实现单张照片视角 + 损伤识别。
4. 新建 `agents/master_agent.py`，实现任务分发与聚合。
5. 调整 `synthesizer.py` 与 `reviewer_subagent.py` 的输入格式。
6. 编写 ViewAgent prompt，固化 33 part 清单与 9 view 清单。
7. 添加单元测试与集成测试。
