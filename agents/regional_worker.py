import json
from typing import List, Dict, Any, Optional

from agents.minimax_client import call_minimax, build_image_content, extract_json
from agents.view_mapping import canonicalize_view_id
from config import PARTS_CATALOG, PARTS_BY_ID
from models import VehicleTopology, PartActualState, Status, DamageLevel


# 位置到相关部件的映射
LOCATION_TO_PARTS = {
    "车头": ["hood", "bumper_front", "headlight_front_left", "headlight_front_right",
            "grille_front", "fender_front_left", "fender_front_right", "windshield_front"],
    "车尾": ["trunk_lid", "bumper_rear", "taillight_rear_left", "taillight_rear_right",
            "windshield_rear"],
    "左侧": ["door_front_left", "door_rear_left", "mirror_left", "fender_rear_left"],
    "右侧": ["door_front_right", "door_rear_right", "mirror_right", "fender_rear_right"],
    "车顶": ["roof_front", "roof_middle", "roof_rear", "sunroof_glass", "roof_rack"],
    "内饰": [],
}

# 中文位置描述到视角标识的映射（输出标准视角 ID）
LOCATION_DETAIL_TO_VIEW = {
    "车头": "front",
    "正面": "front",
    "前": "front",
    "车尾": "rear",
    "后面": "rear",
    "后": "rear",
    "左侧": "left_90",
    "左": "left_90",
    "右侧": "right_90",
    "右": "right_90",
    "车顶": "top",
    "顶部": "top",
    "车头左侧": "front_left_45",
    "左前": "front_left_45",
    "车头左前": "front_left_45",
    "车头右侧": "front_right_45",
    "右前": "front_right_45",
    "车头右前": "front_right_45",
    "车尾左侧": "rear_left_45",
    "左后": "rear_left_45",
    "车尾左后": "rear_left_45",
    "车尾右侧": "rear_right_45",
    "右后": "rear_right_45",
    "车尾右后": "rear_right_45",
    "车头正前": "front",
    "车尾正后": "rear",
    "左侧正侧": "left_90",
    "左侧面": "left_90",
    "右侧正侧": "right_90",
    "右侧面": "right_90",
}

# 位置到区域标识的映射
LOCATION_TO_REGION = {
    "车头": "front",
    "车尾": "rear",
    "左侧": "left",
    "右侧": "right",
    "车顶": "roof",
    "内饰": "interior",
}


def _get_location_detail(vehicle_prior: Dict[str, Any], location: str) -> str:
    """从车型先验中提取某个位置的描述"""
    topology = vehicle_prior.get("topology", {})
    key = None
    mapping = {
        "车头": "front",
        "车尾": "rear",
        "左侧": "left",
        "右侧": "right",
        "车顶": "roof",
    }
    key = mapping.get(location, "")
    return topology.get(key, "")


def _location_detail_to_view(location_detail: str) -> str:
    """将照片位置描述字符串映射到视角标识。"""
    if not location_detail:
        return ""
    # 精确匹配
    view = LOCATION_DETAIL_TO_VIEW.get(location_detail.strip())
    if view:
        return view
    # 子串匹配
    for cn, view_id in LOCATION_DETAIL_TO_VIEW.items():
        if cn in location_detail or location_detail in cn:
            return view_id
    return ""


def map_photos_to_topology_nodes(
    photos: List[Dict[str, Any]],
    location: str,
    topology: VehicleTopology,
) -> Dict[str, List[str]]:
    """将照片映射到拓扑节点。

    返回 Dict[node_id, List[photo_id]]，表示每个节点被哪些照片覆盖。
    """
    region = LOCATION_TO_REGION.get(location, "")
    node_ids = topology.regions.get(region, [])

    coverage: Dict[str, List[str]] = {nid: [] for nid in node_ids}

    for photo in photos:
        photo_id = photo.get("id", "")
        detail = photo.get("detail", "")
        view = _location_detail_to_view(detail)

        if not view:
            # fallback: 用 region 匹配
            view = region

        for nid in node_ids:
            node = topology.get_node(nid)
            if node is None:
                continue
            canonical_view = canonicalize_view_id(view)
            node_views = {canonicalize_view_id(v) for v in node.visibility_from}
            if canonical_view in node_views or region in node.visibility_from:
                coverage[nid].append(photo_id)

    return coverage


def _build_topology_prompt(
    location: str,
    topology: VehicleTopology,
) -> str:
    """为给定位置构建包含拓扑节点标况特征的 prompt 片段。"""
    region = LOCATION_TO_REGION.get(location, "")
    nodes = topology.get_nodes_by_region(region)

    lines: List[str] = []
    lines.append(f"\n标况拓扑节点信息（{location} 区域）：")
    lines.append("-" * 40)

    for node in nodes:
        lines.append(f"\n部件: {node.node_name} (ID: {node.part_id})")
        lines.append(f"  类型: {node.node_type}, 位置: {node.side}")
        if node.standard_features:
            lines.append(f"  标况特征: {', '.join(node.standard_features)}")
        if node.key_anchors:
            lines.append(f"  关键锚点: {', '.join(node.key_anchors)}")
        if node.visibility_from:
            lines.append(f"  可见视角: {', '.join(node.visibility_from)}")
        if node.adjacent_nodes:
            lines.append(f"  相邻部件: {', '.join(node.adjacent_nodes)}")

    lines.append("-" * 40)
    return "\n".join(lines)


def _llm_dict_to_part_actual_state(
    data: Dict[str, Any],
    region: str,
) -> PartActualState:
    """将 LLM 返回的 dict 转换为 PartActualState。"""
    status_str = data.get("status", "uncertain")
    damage_level_str = data.get("damage_level", "unknown")

    # 处理 missing 状态
    if status_str == "missing":
        status = Status.MISSING
        damage_level = DamageLevel.SEVERE
    else:
        try:
            status = Status(status_str)
        except ValueError:
            status = Status.UNCERTAIN
        try:
            damage_level = DamageLevel(damage_level_str)
        except ValueError:
            damage_level = DamageLevel.UNKNOWN

    # 处理 damage_type(s)
    damage_types: List[str] = []
    raw_types = data.get("damage_type", data.get("damage_types", []))
    if isinstance(raw_types, str):
        if raw_types and raw_types != "none":
            damage_types = [t.strip() for t in raw_types.split(",") if t.strip()]
    elif isinstance(raw_types, list):
        damage_types = [str(t) for t in raw_types if t]

    # 处理 evidence_photo(s)
    from agents.evidence_photo import to_photo_list
    evidence_photos: List[str] = to_photo_list(
        data.get("evidence_photo", data.get("evidence_photos", []))
    )

    # 推断 actual_visible / actual_present
    actual_visible = data.get("actual_visible", len(evidence_photos) > 0)
    actual_present = data.get("actual_present", status != Status.MISSING)

    return PartActualState(
        part_id=data.get("part_id", ""),
        part_name=data.get("part_name", ""),
        part_category=region,
        side=data.get("side", ""),
        status=status,
        damage_level=damage_level,
        damage_types=damage_types,
        standard_exists=data.get("standard_exists", True),
        actual_visible=actual_visible,
        actual_present=actual_present,
        confidence=data.get("confidence", "low"),
        evidence_photos=evidence_photos,
        notes=data.get("notes", ""),
    )


async def regional_damage_worker(
    location: str,
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
    topology: Optional[VehicleTopology] = None,
) -> Dict[str, Any]:
    """
    区域级损伤识别 Worker。
    输入一个位置（车头/车尾/左侧/右侧/车顶）和该位置的多张照片，
    输出该区域相关部件的损伤评估。

    当提供 topology 时，使用标况拓扑节点特征进行结构化评估，
    输出兼容 PartActualState 的新格式；同时保留 legacy 格式供旧消费者使用。
    当 topology 为 None 时，保持原有行为不变。
    """
    vehicle_name = vehicle_prior.get("vehicle", "该车")
    location_detail = _get_location_detail(vehicle_prior, location)

    # 最多选 2 张最高置信度的照片（MiniMax-M3 单次最多稳定处理 2 张）
    sorted_photos = sorted(
        photos,
        key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("confidence", "low"), 3),
    )
    selected_photos = sorted_photos[:2]

    # 该区域相关的部件
    if topology is not None:
        region = LOCATION_TO_REGION.get(location, "")
        related_part_ids = topology.regions.get(region, [])
    else:
        related_part_ids = LOCATION_TO_PARTS.get(location, [])
    related_parts = [PARTS_BY_ID[pid] for pid in related_part_ids if pid in PARTS_BY_ID]
    parts_list = json.dumps(related_parts, ensure_ascii=False, indent=2)

    # 构建系统 prompt
    system_prompt_parts = [
        f"你是汽车{location}损伤评估专家。给定 {vehicle_name} 的{location}先验信息和 {len(selected_photos)} 张{location}照片，请对该区域相关部件进行损伤识别，输出结构化 JSON。",
    ]

    if topology is not None:
        system_prompt_parts.append(_build_topology_prompt(location, topology))

    system_prompt_parts.extend([
        f"\n{location}先验：",
        location_detail,
        "\n必须检查的部件清单（每个部件都必须有结论）：",
        parts_list,
    ])

    # 输出格式说明
    if topology is not None:
        system_prompt_parts.append("""
输出 JSON 格式：
{
  "region": "LOCATION",
  "parts": [
    {
      "part_id": "hood",
      "part_name": "引擎盖",
      "region": "front",
      "side": "center",
      "status": "intact|damaged|missing|uncertain",
      "damage_level": "none|light|moderate|severe|unknown",
      "damage_type": ["scratch", "dent", "crack", "tear", "deformation", "missing", "other"],
      "standard_exists": true,
      "actual_visible": true,
      "actual_present": true,
      "confidence": "high|medium|low",
      "evidence_photo": ["167111-02.png", "167111-03.png"],
      "notes": "补充说明"
    }
  ],
  "uncertain_items": [
    {
      "item": "前挡风玻璃状态",
      "reason": "被气囊遮挡",
      "suggested_action": "补拍前挡风玻璃正面照片"
    }
  ]
}

判定规则：
1. status 为 damaged 时，damage_level 必须是 light/moderate/severe，不能是 none
2. status 为 intact 时，damage_level 必须是 none
3. status 为 uncertain 时，damage_level 可以是 none 或 unknown
4. status 为 missing 时，damage_level 必须是 severe，damage_type 必须包含 "missing"
5. standard_exists 表示该车型标况下是否应有此部件（通常为 true）
6. actual_visible 表示照片中是否能看到该部件或其安装位置
7. actual_present 表示该部件实际是否存在（missing 时为 false）
8. 结构件（A/B/C柱、车顶框架）只要变形就至少 moderate
9. 气囊弹出区域的相关部件至少 moderate
10. 不要把泥点、水渍、阴影、反光、拍摄畸变误判为损伤
11. **evidence_photo 必须使用本次输入的照片真实编号**（如 167111-02.png），严禁使用 photo_01/photo_02 等占位符
12. **侧后方视角判定**：车尾左后45度/右后45度照片虽然主要展示车尾，但也覆盖了同侧后翼子板、后车门、尾灯安装区域。只要照片中能看到某一部件或其安装锚点，就要给出明确结论（damaged/intact/missing），不要保守标记为 uncertain
13. **关键区分**：
    - 如果照片拍到了该部件的**安装位置/锚点区域**，但该部件本身已经**缺失、脱落、完全撞毁不见**，必须判定为 `missing`，`damage_level=severe`，`damage_type` 必须包含 `missing`，`actual_present=false`，`actual_visible=true`。
    - 如果照片完全**没有覆盖**到该部件及其安装位置，才标记为 `uncertain`，`actual_visible=false`。
    - 例如：左前大灯安装座可见但灯总成缺失 → missing + severe + missing + actual_visible=true + actual_present=false；左前大灯安装座在画面外 → uncertain + actual_visible=false。
12. 同区域多张照片冲突时，取最保守结论（损伤程度往高报，置信度往低调）

只输出 JSON，不要额外文字。
""".replace("LOCATION", location))
    else:
        system_prompt_parts.append("""
输出 JSON 格式：
{
  "region": "LOCATION",
  "parts": [
    {
      "part_id": "hood",
      "part_name": "引擎盖",
      "status": "intact|damaged|uncertain",
      "damage_level": "none|light|moderate|severe",
      "damage_type": ["scratch", "dent", "crack", "tear", "deformation", "missing", "other"],
      "confidence": "high|medium|low",
      "evidence_photo": ["167111-02.png", "167111-03.png"],
      "notes": "补充说明"
    }
  ],
  "uncertain_items": [
    {
      "item": "前挡风玻璃状态",
      "reason": "被气囊遮挡",
      "suggested_action": "补拍前挡风玻璃正面照片"
    }
  ]
}

判定规则：
1. status 为 damaged 时，damage_level 必须是 light/moderate/severe，不能是 none
2. status 为 intact 时，damage_level 必须是 none
3. status 为 uncertain 时，damage_level 可以是 none 或 unknown
4. 结构件（A/B/C柱、车顶框架）只要变形就至少 moderate
5. 气囊弹出区域的相关部件至少 moderate
6. 不要把泥点、水渍、阴影、反光、拍摄畸变误判为损伤
7. **关键区分**：
   - 如果照片拍到了该部件的**安装位置/锚点区域**，但该部件本身已经**缺失、脱落、完全撞毁不见**，必须判定为 `damaged`，`damage_level=severe`，`damage_type` 必须包含 `missing`。
   - 如果照片完全**没有覆盖**到该部件及其安装位置，才标记为 `uncertain`。
   - 例如：左前大灯安装座可见但灯总成缺失 → damaged + severe + missing；左前大灯安装座在画面外 → uncertain。
8. 同区域多张照片冲突时，取最保守结论（损伤程度往高报，置信度往低调）

只输出 JSON，不要额外文字。
""".replace("LOCATION", location))

    system_prompt = "\n".join(system_prompt_parts)

    content = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": f"以下是{location}的 {len(selected_photos)} 张照片，请联合分析："},
    ]
    for photo in selected_photos:
        content.append({"type": "text", "text": f"照片 {photo['id']} - {photo.get('detail', '')} (置信度: {photo.get('confidence', 'low')})"})
        content.append(build_image_content(photo["path"]))

    messages = [{"role": "user", "content": content}]

    raw = await call_minimax(messages, temperature=0.1, max_tokens=3000)
    result = extract_json(raw)
    if result is None:
        if topology is not None:
            return _build_empty_topology_result(location, topology)
        return {"region": location, "parts": [], "uncertain_items": []}
    if isinstance(result, list):
        if topology is not None:
            return _build_empty_topology_result(location, topology)
        return {"region": location, "parts": result, "uncertain_items": []}

    # 给每个 part 补齐 part_category 和 side
    for part in result.get("parts", []):
        if isinstance(part, dict):
            part_id = part.get("part_id")
            if part_id and part_id in PARTS_BY_ID:
                part.setdefault("part_category", PARTS_BY_ID[part_id]["part_category"])
                part.setdefault("side", PARTS_BY_ID[part_id]["side"])

    result.setdefault("region", location)
    result.setdefault("parts", [])
    result.setdefault("uncertain_items", [])

    # 当提供了 topology 时，转换为新格式
    if topology is not None:
        return _convert_to_topology_result(result, location, topology)

    return result


def _build_empty_topology_result(
    location: str,
    topology: VehicleTopology,
) -> Dict[str, Any]:
    """当 LLM 返回空结果时，构建默认的拓扑兼容输出。"""
    region = LOCATION_TO_REGION.get(location, "")
    nodes = topology.get_nodes_by_region(region)

    states: List[PartActualState] = []
    for node in nodes:
        states.append(PartActualState.from_region_part(
            part_id=node.part_id,
            part_name=node.node_name,
            part_category=region,
            side=node.side,
            status=Status.UNCERTAIN,
            damage_level=DamageLevel.UNKNOWN,
        ))

    return {
        "region": location,
        "parts": [s.to_legacy_dict() for s in states],
        "part_actual_states": states,
        "uncertain_items": [],
    }


def _convert_to_topology_result(
    result: Dict[str, Any],
    location: str,
    topology: VehicleTopology,
) -> Dict[str, Any]:
    """将 LLM 结果转换为拓扑兼容输出（包含 PartActualState 和 legacy dict）。"""
    region = LOCATION_TO_REGION.get(location, "")
    nodes = topology.get_nodes_by_region(region)

    # 建立 part_id -> node 映射
    node_map = {node.part_id: node for node in nodes}

    # 解析 LLM 返回的 parts
    states: List[PartActualState] = []
    seen_part_ids: set = set()

    for part_dict in result.get("parts", []):
        if not isinstance(part_dict, dict):
            continue
        part_id = part_dict.get("part_id", "")
        if not part_id:
            continue
        seen_part_ids.add(part_id)

        # 补充 region / side 如果缺失
        if not part_dict.get("region"):
            part_dict["region"] = region
        if not part_dict.get("side") and part_id in node_map:
            part_dict["side"] = node_map[part_id].side

        state = _llm_dict_to_part_actual_state(part_dict, region)
        states.append(state)

    # 为拓扑中但未在 LLM 结果中出现的部件生成 UNCERTAIN 状态
    for node in nodes:
        if node.part_id not in seen_part_ids:
            states.append(PartActualState.from_region_part(
                part_id=node.part_id,
                part_name=node.node_name,
                part_category=region,
                side=node.side,
                status=Status.UNCERTAIN,
                damage_level=DamageLevel.UNKNOWN,
            ))

    return {
        "region": location,
        "parts": [s.to_legacy_dict() for s in states],
        "part_actual_states": states,
        "uncertain_items": result.get("uncertain_items", []),
    }
