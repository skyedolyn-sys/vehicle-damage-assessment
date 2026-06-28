# 车辆损伤评估系统 — 标况拓扑建模与对比架构升级计划

**计划名称**：车辆损伤评估系统 — 标况拓扑建模与对比架构升级计划  
**简称**：拓扑对比升级计划  
**文档路径**：`/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/docs/plan-topology-comparison.md`  
**制定日期**：2026/06/22

---

## 1. 现状问题分析

### 1.1 当前架构的核心问题

| 问题 | 描述 | 影响 |
|------|------|------|
| **拓扑模型是文本描述** | `vehicle_prior` 输出 `topology.front` 等字段为自然语言字符串，无结构化部件关系 | 无法与部件清单 `PARTS_CATALOG` 建立精确映射，无法做"标况 vs 实际"的对比 |
| **拓扑与部件状态分离** | 车型先验和部件评估是两条独立流水线，没有统一的拓扑模型贯穿全程 | 无法回答"该车型应该有哪些部件、这些部件在标况下是什么关系" |
| **部件状态是扁平列表** | 最终输出是 `parts[]` 数组，缺少部件之间的空间/结构关系 | 无法判断"相邻部件同时损伤"是否构成结构性事故模式 |
| **损伤判断缺少标况参照** | `regional_worker` 直接判定 `status=intact/damaged/uncertain`，没有与标况模型对比的过程 | 无法区分"部件缺失"（标况有、实际无）vs"部件完好"（标况有、实际有且正常） |
| **structural flag 规则硬编码** | `output_validator.py` 中规则 A/B/C 是人工编写的启发式规则，未利用拓扑关系 | 规则覆盖不全，无法扩展为"基于拓扑的结构性损伤模式识别" |

### 1.2 数据流问题

```
当前流程：
  vehicle_info → vehicle_prior(文本描述) → photo_locator(视角分类)
                                              ↓
  photos → regional_worker(按位置分组) → synthesizer(保守合并) → output_validator(补全+规则)

问题：没有"标况拓扑模型"这个核心数据结构，各 Agent 之间传递的是文本和扁平列表
```

### 1.3 目标流程

```
目标流程：
  vehicle_info → vehicle_prior(结构化标况拓扑模型)
                        ↓
  photos → photo_locator(视角分类) → 将照片映射到拓扑节点
                        ↓
  regional_worker(按拓扑区域分组) → 视觉识别部件状况
                        ↓
  synthesizer(汇总部件状况) → 生成实际状态拓扑
                        ↓
  damage_comparator(标况拓扑 vs 实际拓扑) → 得出车损结论
                        ↓
  output_validator(校验、summary、structural flag)
```

---

## 2. 目标架构设计

### 2.1 核心数据结构

#### 2.1.1 标况拓扑模型 `VehicleTopology`

位置：`models/topology.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any


@dataclass(frozen=True)
class TopologyNode:
    """拓扑节点：代表一个部件或部件组"""
    node_id: str           # 唯一标识，如 "hood", "bumper_front"
    part_id: str           # 映射到 PARTS_CATALOG 的 part_id
    node_name: str         # 中文名称
    node_type: str         # "panel" | "glass" | "light" | "structural" | "trim"
    region: str            # "front" | "rear" | "left" | "right" | "roof" | "undercarriage"
    side: str              # "center" | "front_left" | "front_right" | "rear_left" | "rear_right"

    # 拓扑关系
    adjacent_nodes: List[str] = field(default_factory=list)
    parent_node: Optional[str] = None
    child_nodes: List[str] = field(default_factory=list)

    # 标况特征
    standard_features: List[str] = field(default_factory=list)
    key_anchors: List[str] = field(default_factory=list)
    visibility_from: List[str] = field(default_factory=list)
```

```python
class VehicleTopology:
    """整车标况拓扑模型"""
    vehicle_id: str
    vehicle_name: str
    nodes: Dict[str, TopologyNode]
    regions: Dict[str, List[str]]
```

#### 2.1.2 部件状态表示 `PartActualState`

位置：`models/part_state.py`

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict


class Status(str, Enum):
    INTACT = "intact"       # 完好：标况有，实际有且正常
    DAMAGED = "damaged"     # 损伤：标况有，实际有但异常
    MISSING = "missing"     # 缺失：标况有，实际无
    UNCERTAIN = "uncertain" # 不确定：无法判断
    NOT_APPLICABLE = "na"   # 不适用：标况无


class DamageLevel(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


@dataclass
class PartActualState:
    """部件实际状态（与标况对比后的结论）"""
    part_id: str
    part_name: str
    region: str
    side: str

    # 与标况对比后的状态
    status: Status
    damage_level: DamageLevel
    damage_types: List[str] = field(default_factory=list)

    # 对比信息
    standard_exists: bool = True
    actual_visible: bool = False
    actual_present: bool = True

    confidence: str = "low"
    evidence_photos: List[str] = field(default_factory=list)
    notes: str = ""
    adjacent_status: Dict[str, str] = field(default_factory=dict)
```

#### 2.1.3 损伤对比结论 `DamageAssessment`

位置：`models/assessment.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class StructuralDamagePattern:
    """结构性损伤模式"""
    pattern_id: str
    pattern_name: str
    description: str
    matched_nodes: List[str]
    severity: str
    confidence: str = "medium"


@dataclass
class DamageAssessment:
    """最终车损评估报告"""
    vehicle_info: Dict[str, Any]
    topology_model: Dict[str, Any]
    parts: List[PartActualState]
    missing_parts: List[str]
    damaged_parts: List[str]
    intact_parts: List[str]
    uncertain_parts: List[str]
    structural_patterns: List[StructuralDamagePattern]
    structural_damage_flag: bool
    overall_severity: str
    primary_damage_zone: str
    summary: Dict[str, Any]
```

### 2.2 模块架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              标况拓扑建模层                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐   │
│  │ vehicle_prior   │───→│ TopologyBuilder │───→│ VehicleTopology (标况)   │   │
│  │ (LLM: 车型特征)  │    │ (结构化拓扑生成)  │    │ 结构化节点 + 关系 + 特征   │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              视觉识别层                                       │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────────┐  │
│  │ photo_locator   │───→│ PhotoTopologyMapper (照片→拓扑节点映射)          │  │
│  │ (视角分类)       │    │ 输出：每个节点被哪些照片覆盖                       │  │
│  └─────────────────┘    └─────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ regional_worker (改造)                                                  │ │
│  │ 输入：区域 + 照片 + 标况拓扑节点特征                                      │ │
│  │ 输出：各部件实际状态（与标况对比：完好/损伤/缺失/不确定）                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              拓扑对比与结论层                                 │
│  ┌─────────────────┐    ┌─────────────────────────┐    ┌─────────────────┐  │
│  │ synthesizer     │───→│ TopologyComparator      │───→│ output_validator│  │
│  │ (汇总部件状态)   │    │ (标况拓扑 vs 实际状态)    │    │ (校验+summary)  │  │
│  │ 按 part_id 合并  │    │ 识别缺失、对比相邻关系     │    │ 拓扑模式识别      │  │
│  └─────────────────┘    └─────────────────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 数据结构变化详表

### 3.1 vehicle_prior 输出格式变化

| 字段 | 当前 | 目标 | 说明 |
|------|------|------|------|
| `topology` | `{"front": "自然语言描述", ...}` | 保留 | 作为 LLM 原始输出供调试 |
| `key_anchors` | `{"front": ["锚点1"], ...}` | 保留 | 锚点与部件绑定 |
| `topology_model` | 无 | `VehicleTopology` 字典 | 新增结构化拓扑模型 |

### 3.2 新增数据结构

| 文件 | 内容 |
|------|------|
| `models/topology.py` | `VehicleTopology`, `TopologyNode` |
| `models/part_state.py` | `PartActualState`, `Status`, `DamageLevel` |
| `models/assessment.py` | `DamageAssessment`, `StructuralDamagePattern` |
| `models/__init__.py` | 导出 |

### 3.3 改造数据结构

| 文件 | 改造点 |
|------|--------|
| `config.py` | 增加 `PARTS_TOPOLOGY` 拓扑配置 |
| `agents/vehicle_prior.py` | 输出增加 `topology_model` |
| `agents/regional_worker.py` | 输入增加标况拓扑节点特征，输出改为 `PartActualState` |
| `agents/synthesizer.py` | 合并逻辑增加拓扑相邻关系传播 |
| `agents/output_validator.py` | structural flag 改为基于拓扑模式识别 |

---

## 4. 各模块改造点

### 4.1 新增模块

#### 4.1.1 `models/topology.py`

- `TopologyNode`：不可变拓扑节点
- `VehicleTopology`：整车拓扑模型，支持区域查询、邻居查询、区域覆盖度计算

#### 4.1.2 `models/part_state.py`

- `Status` / `DamageLevel`：枚举类型
- `PartActualState`：部件实际状态，含 `to_legacy_dict()` 兼容旧格式

#### 4.1.3 `models/assessment.py`

- `StructuralDamagePattern`：结构性损伤模式
- `DamageAssessment`：最终评估报告

#### 4.1.4 `agents/topology_builder.py`

从 `vehicle_prior` 的 LLM 输出和 `PARTS_CATALOG` 构建 `VehicleTopology`：

1. 从 `PARTS_CATALOG` 获取基础部件列表
2. 从 `PARTS_TOPOLOGY` 获取相邻关系、部件类型、视角可见性
3. 将 LLM 文本特征注入对应节点
4. 返回 `VehicleTopology`

#### 4.1.5 `agents/topology_comparator.py`

核心比较器，实现标况 vs 实际对比：

1. 遍历拓扑中所有节点
2. 查找实际状态，分类为 intact / damaged / missing / uncertain
3. BFS 查找连通损伤簇
4. 识别 5 种结构性损伤模式
5. 返回 `DamageAssessment` 基础数据

### 4.2 改造模块

#### 4.2.1 `config.py`

新增 `PARTS_TOPOLOGY`：

```python
PARTS_TOPOLOGY = {
    "adjacency": {
        "hood": ["grille_front", "fender_front_left", "fender_front_right", "windshield_front"],
        # ...
    },
    "node_types": {
        "hood": "panel",
        # ...
    },
    "visibility": {
        "hood": ["front", "front_left_45", "front_right_45"],
        # ...
    }
}
```

#### 4.2.2 `agents/vehicle_prior.py`

- 保留现有 LLM prompt
- 输出中新增 `topology_model` 字段
- 调用 `topology_builder.build_vehicle_topology()` 生成结构化拓扑

#### 4.2.3 `agents/regional_worker.py`

- 输入增加 `VehicleTopology`
- prompt 中增加标况拓扑节点特征
- 输出增加字段：`standard_exists`, `actual_visible`, `actual_present`
- 新增状态 `missing`
- 判定规则增加：
  - 安装位置可见但部件不在 → `missing` + `severe`
  - 照片未覆盖 → `uncertain`

#### 4.2.4 `agents/synthesizer.py`

- 按 `part_id` 合并
- 增加拓扑相邻关系传播：相邻部件缺失时，本部件若为 `uncertain` 降低置信度

#### 4.2.5 `agents/output_validator.py`

- 保留字段补全逻辑
- structural flag 改为基于 `TopologyComparator` 的模式识别
- 输出新增 `structural_patterns` 数组

#### 4.2.6 `backend.py`

- workflow 中注入拓扑模型
- `damage_assessor_agent` 和 `validate_and_enrich` 增加 `topology` 参数

---

## 5. 关键算法/判断逻辑

### 5.1 标况拓扑构建算法

```
输入: vehicle_info + LLM 输出的文本特征
输出: VehicleTopology

1. 从 PARTS_CATALOG 获取基础部件列表
2. 从 PARTS_TOPOLOGY 获取相邻关系、类型、可见性
3. 将 LLM 文本特征注入对应节点
4. 构建 VehicleTopology 对象
```

### 5.2 照片-拓扑映射算法

```
输入: 照片列表 + location_map + VehicleTopology
输出: node_id -> [photo_id, ...]

1. 对每张照片获取其 location
2. location 映射到视角标识
3. 查询 visibility_from 包含该视角的节点
4. 结合 photo_locator 的 visible_parts 精筛
5. 输出节点覆盖映射
```

### 5.3 标况 vs 实际对比算法

```
输入: VehicleTopology + List[PartActualState]
输出: DamageAssessment 基础数据

1. 遍历拓扑中所有节点
2. 查找实际状态并分类
3. 对 damaged + missing 节点做拓扑相邻分析
4. 识别结构性损伤模式
5. 生成评估结果
```

### 5.4 结构性损伤模式识别规则

| 模式 ID | 名称 | 触发条件 | 严重度 |
|---------|------|----------|--------|
| `regional_mass_damage` | 区域大面积损伤 | 单区域 >=3 个部件受损/缺失 | severe |
| `cross_region_penetration` | 跨区域穿透损伤 | 连通损伤簇跨越 >=2 个区域 | severe |
| `structural_component_damage` | 结构件受损 | 节点类型为 structural 的部件受损 | severe |
| `symmetric_damage` | 对称损伤 | >=2 对对称部件同时受损 | moderate |
| `safety_critical_missing` | 安全关键部件缺失 | 大灯/尾灯/后视镜缺失 | severe |

---

## 6. 开发阶段和优先级

### Phase 1: 基础设施（第 1-2 周）

| 优先级 | 任务 | 文件 | 工作量 |
|--------|------|------|--------|
| P0 | 创建数据模型包 | `models/*.py` | 2d |
| P0 | 创建拓扑构建器 | `agents/topology_builder.py` | 1.5d |
| P1 | 扩展 `config.py` 拓扑配置 | `config.py` | 0.5d |
| P1 | 改造 `vehicle_prior.py` 输出拓扑 | `agents/vehicle_prior.py` | 1d |
| P1 | 单元测试：拓扑构建 | `tests/test_topology*.py` | 1d |

**验收标准**：
- `build_vehicle_topology()` 能为任意车型输出正确的 `VehicleTopology`
- 所有节点都有正确的相邻关系
- 序列化/反序列化正确

### Phase 2: 核心对比引擎（第 2-3 周）

| 优先级 | 任务 | 文件 | 工作量 |
|--------|------|------|--------|
| P0 | 创建拓扑比较器 | `agents/topology_comparator.py` | 2d |
| P0 | 改造 `regional_worker.py` 输出 `PartActualState` | `agents/regional_worker.py` | 1.5d |
| P1 | 改造 `synthesizer.py` 增加拓扑传播 | `agents/synthesizer.py` | 1d |
| P1 | 单元测试：对比逻辑 | `tests/test_comparator.py` | 1d |

**验收标准**：
- 5 种结构性损伤模式都能正确识别
- `missing` 状态能正确区分"没拍到"和"拍到但缺失"
- 相邻关系传播逻辑正确

### Phase 3: 集成与替换（第 3-4 周）

| 优先级 | 任务 | 文件 | 工作量 |
|--------|------|------|--------|
| P0 | 改造 `damage_assessor.py` 传入拓扑 | `agents/damage_assessor.py` | 1d |
| P0 | 改造 `output_validator.py` 使用拓扑比较器 | `agents/output_validator.py` | 1.5d |
| P0 | 改造 `backend.py` workflow | `backend.py` | 0.5d |
| P1 | 增加照片-拓扑映射 | `agents/photo_locator.py` | 1d |
| P1 | 端到端测试 | `tests/test_e2e*.py` | 1d |

**验收标准**：
- 完整 workflow 能跑通
- 输出包含 `topology_model` 和 `structural_patterns`
- 旧格式输出仍兼容

### Phase 4: 验证与优化（第 4-5 周）

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 收集真实案例测试 | 对比新旧架构输出差异 |
| P1 | 优化 `vehicle_prior` prompt | 让 LLM 输出更精确的部件特征 |
| P1 | 调整拓扑相邻关系 | 根据测试结果修正 `PARTS_TOPOLOGY` |
| P2 | 增加更多结构性损伤模式 | 根据业务需求扩展 |
| P2 | 性能优化 | 拓扑构建缓存（同车型复用） |

---

## 7. 测试验证方案

### 7.1 单元测试

| 测试文件 | 覆盖内容 | 用例数 |
|----------|----------|--------|
| `tests/test_topology.py` | TopologyNode, VehicleTopology 构建、查询、序列化 | 15 |
| `tests/test_topology_builder.py` | build_vehicle_topology 正确性 | 10 |
| `tests/test_part_state.py` | PartActualState 状态转换、序列化 | 10 |
| `tests/test_comparator.py` | 对比逻辑、5种模式识别 | 20 |
| `tests/test_synthesizer.py` | 合并逻辑、拓扑传播 | 10 |

### 7.2 集成测试

| 测试文件 | 场景 |
|----------|------|
| `tests/test_integration_flow.py` | 完整 workflow，模拟 API 调用 |
| `tests/test_integration_missing.py` | 部件缺失场景 |
| `tests/test_integration_structural.py` | 结构性损伤场景 |

### 7.3 端到端测试

| 测试文件 | 输入 | 验证点 |
|----------|------|--------|
| `tests/test_e2e_front_collision.py` | 正面碰撞照片集 | 前部部件 damaged，触发 symmetric_damage |
| `tests/test_e2e_side_scrape.py` | 左侧刮擦照片集 | 左侧部件 damaged，触发 regional_mass_damage |
| `tests/test_e2e_missing_parts.py` | 大灯缺失照片 | 触发 safety_critical_missing |
| `tests/test_e2e_intact.py` | 完好车辆照片 | 所有部件 intact，无 structural flag |

### 7.4 回归测试

```python
def test_backward_compatibility():
    result = run_workflow(photos, vehicle_info)
    assert "parts" in result
    assert "assessment_summary" in result
    assert "topology_model" in result
    assert "structural_patterns" in result
```

---

## 8. 风险和回退方案

### 8.1 风险清单

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| LLM 输出不稳定导致拓扑构建失败 | 中 | 高 | 增加 JSON schema 校验，失败时回退到默认拓扑 |
| 拓扑相邻关系定义不准确 | 中 | 中 | 根据测试案例持续调整 `PARTS_TOPOLOGY` |
| 新架构输出格式破坏前端 | 低 | 高 | 保留 `to_legacy_dict()` 兼容层 |
| 性能下降 | 低 | 低 | 拓扑构建结果缓存，同车型复用 |
| 5种模式覆盖不全 | 中 | 中 | 与业务专家 review，逐步扩展 |

### 8.2 回退方案

**方案 A：特性开关**

```python
# config.py
ENABLE_TOPOLOGY_MODEL = os.getenv("ENABLE_TOPOLOGY_MODEL", "false").lower() == "true"

# backend.py
if ENABLE_TOPOLOGY_MODEL:
    topology = build_vehicle_topology(vehicle_info, vehicle_prior)
    final_result = validate_and_enrich(damage_result, topology)
else:
    final_result = validate_and_enrich(damage_result)
```

**方案 B：蓝绿部署**

- 新架构部署到 `/api/assess-v2`
- 旧 `/api/assess` 保持不变
- 验证通过后切换

**方案 C：数据兼容**

- `output_validator` 始终输出旧格式字段
- 新格式字段作为扩展字段附加

### 8.3 关键决策点

| 决策 | 建议 |
|------|------|
| 是否保留旧 `topology` 文本字段 | **保留**，作为 LLM 原始输出供调试 |
| `TopologyNode` 是否用 dataclass | **dataclass**，类型安全 |
| 拓扑关系是否支持车型定制 | **先通用**，后续支持车型覆盖 |
| 是否持久化拓扑模型 | **内存 + 缓存**，同车型复用 |

---

## 9. 文件变更汇总

### 新建文件

| 文件 | 行数估计 | 说明 |
|------|----------|------|
| `models/__init__.py` | 10 | 模型包导出 |
| `models/topology.py` | 150 | 拓扑数据结构 |
| `models/part_state.py` | 80 | 部件状态枚举和数据类 |
| `models/assessment.py` | 60 | 评估报告数据类 |
| `agents/topology_builder.py` | 200 | 拓扑构建器 |
| `agents/topology_comparator.py` | 250 | 标况 vs 实际比较器 |
| `tests/test_topology.py` | 200 | 拓扑测试 |
| `tests/test_comparator.py` | 250 | 比较器测试 |
| `tests/test_integration_*.py` | 300 | 集成测试 |

### 改造文件

| 文件 | 改动范围 | 说明 |
|------|----------|------|
| `config.py` | 增加 `PARTS_TOPOLOGY` | 拓扑配置 |
| `agents/vehicle_prior.py` | 输出增加 `topology_model` | 结构化拓扑 |
| `agents/regional_worker.py` | prompt + 输出格式 | 标况对比 |
| `agents/synthesizer.py` | 增加拓扑传播 | 相邻影响 |
| `agents/damage_assessor.py` | 传入 topology | 参数扩展 |
| `agents/output_validator.py` | 替换 structural flag | 模式识别 |
| `agents/__init__.py` | 导出新模块 | 包接口 |
| `backend.py` | workflow 注入拓扑 | 主流程 |

---

## 10. 成功标准

- [ ] `vehicle_prior` 输出包含结构化 `VehicleTopology`
- [ ] `regional_worker` 输出 `PartActualState`，包含 `standard_exists` / `actual_visible` / `actual_present`
- [ ] `TopologyComparator` 能正确识别 5 种结构性损伤模式
- [ ] 完整 workflow 端到端通过，输出包含 `structural_patterns`
- [ ] 旧格式输出兼容，`to_legacy_dict()` 测试通过
- [ ] 单元测试覆盖率 >= 80%
- [ ] 真实案例验证，结构性损伤识别准确率 >= 90%
- [ ] 同车型拓扑构建可复用（缓存），性能无退化
