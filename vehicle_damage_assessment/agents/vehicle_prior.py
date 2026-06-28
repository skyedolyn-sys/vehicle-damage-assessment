"""Vehicle prior agent — generates standard exterior features for a vehicle."""

from typing import Dict, Any
from agents.minimax_client import call_minimax, extract_json
from models.vehicle_specs import VehicleSpecs
from asgiref.sync import sync_to_async
from data.vehicle_specs_cache import get_cached_specs, save_cached_specs


_SYSTEM_PROMPT = """你是汽车车型外观专家。请根据给定的车型，输出该车型的标准外观特征描述，用于帮助视觉模型识别照片拍摄角度和损伤部位。

输出必须是 JSON，格式如下：
{
  "vehicle": "车型完整名称",
  "vehicle_specs": {
    "body_style": "sedan | hatchback | suv | mpv | van | pickup | coupe | convertible | wagon",
    "doors": 4,
    "has_sunroof": false,
    "has_roof_rack": false,
    "headlight_layout": "separate",
    "rear_door_type": "trunk_lid | tailgate | sliding | none",
    "notes": "其他外观特征备注"
  },
  "topology": {
    "front": "车头典型特征：如 X-bar 前脸、分体式大灯、品牌 logo 位置等",
    "rear": "车尾典型特征：如贯穿尾灯、后备箱造型等",
    "left": "左侧典型特征：如车门数量、后视镜位置、腰线等",
    "right": "右侧典型特征",
    "roof": "车顶典型特征：如全景天窗、行李架、鲨鱼鳍天线等"
  },
  "key_anchors": {
    "front": ["车头锚点1", "车头锚点2"],
    "rear": ["车尾锚点1", "车尾锚点2"],
    "left": ["左侧锚点1"],
    "right": ["右侧锚点1"],
    "roof": ["车顶锚点1"]
  },
  "common_photo_angles": ["车头正前", "车头左前45度", "车尾正后", "左侧90度", "右侧90度", "车顶俯视"]
}

【body_style 说明】请从以下选项中选择最准确的一个：
- sedan: 三厢轿车，如奥迪A6L、宝马5系、比亚迪秦PLUS
- hatchback: 两厢掀背车，如大众高尔夫、比亚迪海豚
- suv: 运动型多用途车，如蔚来ES8、特斯拉Model Y、丰田RAV4
- mpv: 多用途汽车，如别克GL8、本田奥德赛
- van: 厢式货车/面包车，如五菱宏光
- pickup: 皮卡，如长城炮、福特F-150
- coupe: 双门轿跑，如宝马4系双门
- convertible: 敞篷车，如宝马Z4
- wagon: 旅行车，如奥迪A4 Avant

【doors 计数规则】
- 2-door: 双门跑车/轿跑（如宝马4系双门、保时捷911）
- 3-door: 两厢三门车（如部分大众高尔夫三门版）
- 4-door: 四门轿车（如奥迪A6L、丰田凯美瑞）
- 5-door: 五门SUV/两厢/MPV/旅行车（尾门算一扇门，如特斯拉Model Y、大众高尔夫五门版、别克GL8）

【rear_door_type 说明】
- trunk_lid: 独立后备箱盖，通常用于 sedan/coupe/convertible
- tailgate: 整体掀背尾门，用于 hatchback/suv/mpv/wagon/pickup
- sliding: 侧滑门，用于 van/部分 mpv（如五菱宏光、本田奥德赛）
- none: 无后舱门（极少，如部分单排皮卡或特殊改装车）

【notes 字段指导】
请在 notes 中注明以下特殊配置（如有）：
- 是否有天窗/天幕（全景天窗、分段式天窗等）
- 是否有行李架
- 大灯类型（贯穿式大灯、分体式大灯、矩阵式大灯等）
- 其他显著外观特征（如隐藏式门把手、无框车门、空气动力学套件等）

【完整示例】
以 "2024 蔚来 ES6" 为例：
{
  "vehicle": "2024 蔚来 ES6",
  "vehicle_specs": {
    "body_style": "suv",
    "doors": 5,
    "has_sunroof": true,
    "has_roof_rack": false,
    "headlight_layout": "split",
    "rear_door_type": "tailgate",
    "notes": "纯电动中型SUV，分体式大灯，车顶全景天幕，无行李架"
  },
  "topology": {
    "front": "X-Bar前脸设计，分体式大灯，顶部瞭望塔式激光雷达，封闭式格栅",
    "rear": "贯穿式尾灯，尾门开启方式为整体掀背，后保险杠集成充电口",
    "left": "左侧双门+尾门，隐藏式门把手，车顶鲨鱼鳍天线，腰线从前翼子板延伸至尾灯",
    "right": "右侧双门+尾门，与左侧对称",
    "roof": "全景天幕，无行李架，车顶瞭望塔式激光雷达模块"
  },
  "key_anchors": {
    "front": ["分体式大灯上部", "分体式大灯下部", "品牌logo", "前保险杠"],
    "rear": ["贯穿尾灯", "尾门把手", "后保险杠", "品牌logo"],
    "left": ["左后视镜", "左前车门", "左后车门", "左后翼子板"],
    "right": ["右后视镜", "右前车门", "右后车门", "右后翼子板"],
    "roof": ["全景天幕", "激光雷达模块", "车顶鲨鱼鳍天线"]
  },
  "common_photo_angles": ["车头正前", "车头左前45度", "车尾正后", "车尾左后45度", "左侧90度", "右侧90度", "车顶俯视"]
}

【重要提示】
1. 车辆规格必须基于真实车型参数，不要猜测。如果不确定某个字段，使用最可能的值并在 notes 中说明。
2. 如果用户未提供品牌/型号/年份，或提供的信息不完整，请尝试根据车型典型特征推断；若仍无法确定，使用最符合照片特征的默认值（如不确定车身形态可默认 sedan）并在 notes 中注明"用户未提供车型信息，按默认 sedan 处理"。
3. 即使车型信息缺失，也必须输出完整的 JSON 结构，不允许返回空值。"""


async def vehicle_prior_agent(
    vehicle_info: Dict[str, str],
    use_cache: bool = True,
) -> Dict[str, Any]:
    """第一层：车型先验

    输入车型信息（可为空），输出标准外观特征和拓扑关系。
    当车型信息缺失时，使用 LLM 推断或默认值继续流程。
    """
    brand = vehicle_info.get("brand") or "未知"
    model = vehicle_info.get("model") or "未知"
    year = vehicle_info.get("year") or "未知"

    # Check cache first — only for vehicle_specs, NOT for topology/key_anchors
    cached_specs = None
    if use_cache:
        cached_specs = await sync_to_async(get_cached_specs)(vehicle_info)

    user_prompt = f"请输出 {year} {brand} {model} 的标准外观特征 JSON。"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw = await call_minimax(messages, temperature=0.1, max_tokens=2000)
    result = extract_json(raw)
    if not isinstance(result, dict):
        result = {}

    # Normalize LLM vehicle_specs
    llm_specs = extract_vehicle_specs(result)
    result["vehicle_specs"] = llm_specs.to_dict()

    # If cached specs exist, use them instead of LLM specs (more reliable)
    if cached_specs is not None:
        result["vehicle_specs"] = cached_specs.to_dict()
        # Only save back to cache if LLM specs differ (unlikely, but keeps cache fresh)
        if llm_specs.to_dict() != cached_specs.to_dict():
            await sync_to_async(save_cached_specs)(vehicle_info, cached_specs)
    elif use_cache:
        # No cache hit — save LLM specs
        await sync_to_async(save_cached_specs)(vehicle_info, llm_specs)

    return result


def extract_vehicle_specs(result: Dict[str, Any]) -> VehicleSpecs:
    """Extract and normalize vehicle_specs from a vehicle_prior result dict.

    If the result is None or has no ``vehicle_specs`` key, returns a default
    sedan-like VehicleSpecs for backward compatibility.
    """
    if result is None:
        return VehicleSpecs.from_dict({})
    raw_specs = result.get("vehicle_specs")
    if not isinstance(raw_specs, dict):
        return VehicleSpecs.from_dict({})
    return VehicleSpecs.from_dict(raw_specs)
