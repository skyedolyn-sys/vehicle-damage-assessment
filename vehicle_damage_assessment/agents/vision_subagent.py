"""Vision subagent — per-view damage assessment.

Each vision subagent is an independent LLM call responsible for a single
standard view (e.g. ``front_left``).  It receives all photos assigned to
that view and evaluates every exterior region that the view covers.

The subagent does not decide which regions are covered — that is provided by
the orchestrator via ``view_mapping.VIEW_TO_REGIONS``.  This keeps the
subagent focused on visual recognition.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from agents.minimax_client import call_minimax, build_image_content, extract_json
from agents.view_mapping import get_display_name, get_regions_for_view
from config import IMAGE_MAX_WIDTH, PARTS_BY_ID
from models import PartActualState, Status, DamageLevel


#: Mapping from common LLM-output part id aliases back to canonical ids.
_PART_ID_ALIASES: Dict[str, str] = {
    # Front
    "front_bumper": "bumper_front",
    "front_bumber": "bumper_front",
    "bumper_front": "bumper_front",
    "front_hood": "hood",
    "engine_hood": "hood",
    "hood": "hood",
    "front_grille": "grille_front",
    "grille": "grille_front",
    "left_front_headlight": "headlight_front_left",
    "left_headlight": "headlight_front_left",
    "headlight_left": "headlight_front_left",
    "front_left_headlight": "headlight_front_left",
    "headlight_front_left": "headlight_front_left",
    "right_front_headlight": "headlight_front_right",
    "right_headlight": "headlight_front_right",
    "headlight_right": "headlight_front_right",
    "front_right_headlight": "headlight_front_right",
    "headlight_front_right": "headlight_front_right",
    "left_front_fender": "fender_front_left",
    "left_fender_front": "fender_front_left",
    "left_fender": "fender_front_left",
    "fender_front_left": "fender_front_left",
    "right_front_fender": "fender_front_right",
    "right_fender_front": "fender_front_right",
    "right_fender": "fender_front_right",
    "fender_front_right": "fender_front_right",
    "front_windshield": "windshield_front",
    "windshield_front": "windshield_front",
    # Rear
    "rear_bumper": "bumper_rear",
    "bumper_rear": "bumper_rear",
    "trunk_lid": "trunk_lid",
    "trunk": "trunk_lid",
    "tailgate": "tailgate",
    "left_rear_taillight": "taillight_rear_left",
    "left_rear_tail_light": "taillight_rear_left",
    "left_taillight": "taillight_rear_left",
    "left_tail_light": "taillight_rear_left",
    "taillight_left": "taillight_rear_left",
    "tail_light_left": "taillight_rear_left",
    "taillight_rear_left": "taillight_rear_left",
    "right_rear_taillight": "taillight_rear_right",
    "right_rear_tail_light": "taillight_rear_right",
    "right_taillight": "taillight_rear_right",
    "right_tail_light": "taillight_rear_right",
    "taillight_right": "taillight_rear_right",
    "tail_light_right": "taillight_rear_right",
    "taillight_rear_right": "taillight_rear_right",
    "left_rear_light": "taillight_rear_left",
    "left_tail_light_rear": "taillight_rear_left",
    "right_rear_light": "taillight_rear_right",
    "right_tail_light_rear": "taillight_rear_right",
    "rear_windshield": "windshield_rear",
    "windshield_rear": "windshield_rear",
    # Side / doors
    "left_front_door": "door_front_left",
    "door_front_left": "door_front_left",
    "left_rear_door": "door_rear_left",
    "left_back_door": "door_rear_left",
    "door_rear_left": "door_rear_left",
    "right_front_door": "door_front_right",
    "door_front_right": "door_front_right",
    "right_rear_door": "door_rear_right",
    "right_back_door": "door_rear_right",
    "door_rear_right": "door_rear_right",
    # Mirrors
    "left_side_mirror": "mirror_left",
    "left_mirror": "mirror_left",
    "left_rearview_mirror": "mirror_left",
    "mirror_left": "mirror_left",
    "right_side_mirror": "mirror_right",
    "right_mirror": "mirror_right",
    "right_rearview_mirror": "mirror_right",
    "mirror_right": "mirror_right",
    # Rear fenders / quarter panels
    "left_rear_fender": "fender_rear_left",
    "left_rear_quarter_panel": "fender_rear_left",
    "left_quarter_panel": "fender_rear_left",
    "left_quarter": "fender_rear_left",
    "fender_rear_left": "fender_rear_left",
    "right_rear_fender": "fender_rear_right",
    "right_rear_quarter_panel": "fender_rear_right",
    "right_quarter_panel": "fender_rear_right",
    "right_quarter": "fender_rear_right",
    "fender_rear_right": "fender_rear_right",
    # Roof
    "roof_front": "roof_front",
    "roof_middle": "roof_middle",
    "roof_rear": "roof_rear",
    "sunroof_glass": "sunroof_glass",
    "roof_rack": "roof_rack",
}


def _normalize_part_id(raw_id: str) -> str:
    """Map a free-form part id returned by the LLM to a canonical part id."""
    if not raw_id:
        return ""
    normalized = raw_id.strip().lower().replace(" ", "_")
    return _PART_ID_ALIASES.get(normalized, normalized)



_SYSTEM_PROMPT_TEMPLATE = """你是车辆外部损伤识别专家。本次任务只针对 **{view_display_name}** 视角下的照片进行识别。

## 必须检查的部件清单（checklist）
以下部件属于本次视角的覆盖范围，必须逐项给出结论，禁止省略：
{checklist_text}

## 其他可见异常描述
除了 checklist 内的标准部件外，如果照片中还出现以下情况，请在 additional_findings 中客观描述：
- 不属于标准清单的额外损伤或异常（例如大面积凹陷、玻璃碎裂、液体泄漏、轮胎异常、标识缺失等）
- 无法对应到具体部件但整体可见的损伤区域
- 拍摄质量问题导致无法判断某些区域的说明

车型信息：{vehicle_name}

输出 JSON 格式：
{{
  "view_id": "{view_id}",
  "view_display_name": "{view_display_name}",
  "parts": [
    {{
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
      "evidence_photo": ["167111-02.png"],
      "notes": "补充说明"
    }}
  ],
  "uncertain_items": [
    {{
      "item": "左前翼子板完整性",
      "reason": "视角遮挡",
      "suggested_action": "补拍..."
    }}
  ],
  "additional_findings": [
    {{
      "description": "保险杠下方有轻微刮擦，不在标准部件清单中",
      "location_hint": "前保险杠下沿",
      "severity": "light",
      "evidence_photo": ["167111-02.png"]
    }}
  ]
}}

判定规则：
1. **必须逐项输出 checklist 中的每一个部件，禁止省略任何部件。**
2. 可见性判定（按优先级）：
   - 完全可见或主要部分可见（占比 >= 30% 或可见关键轮廓）：给出明确结论，优先标 intact 或 damaged。
   - 远端部件在透视中占比小但无变形/划痕/裂纹等异常：可判定 intact，confidence=low，notes 中说明"远端可见，无明显变形"。
   - 完全不可见（占比 < 10% 或被严重遮挡）：status=uncertain，damage_level=unknown，confidence=low。
   - 局部可见但无法确认完整状态：可标 uncertain，但如仅边缘可见且无异常，优先尝试标 intact 并说明"局部可见，未见异常"。
3. 车顶区域部件（roof_front, roof_middle, roof_rear, sunroof_glass, roof_rack）在带俯角的视角中只要能看到相应车顶边缘，也必须输出结论；只要车顶边缘可见且未看到凹陷、变形、裂纹、玻璃碎裂等明显异常，优先标 intact；只有当确实看到上述异常时才标 damaged；完全看不到车顶才标 uncertain。
4. status 为 damaged 时，damage_level 不能是 none。
5. status 为 intact 时，damage_level 必须是 none，damage_type 为空或 ["none"]。
6. status 为 missing 时，damage_level 必须是 severe，damage_type 必须包含 "missing"，actual_present=false。
7. status 为 uncertain 时，说明该部件属于 checklist 但照片无法看清；完全不在 checklist 中的部件如果要描述，请放到 additional_findings，不要输出在 parts 中。
8. 同视角多张照片冲突时，取最保守结论（损伤程度往高报，置信度往低调）。
9. 不要把泥点、水渍、阴影、反光、拍摄畸变误判为损伤。
10. evidence_photo 必须使用本次输入照片的真实编号；如果多张照片都看到同一部件，列出所有相关照片编号。
11. additional_findings 只描述客观可见事实，不要推断原因或给出维修建议。
12. damage_level 量化标准（针对 damaged 状态）：
    - light：轻微划痕、小面积漆面损伤、浅凹陷直径 < 5cm、无结构变形。
    - moderate：明显凹陷/变形 5-20cm、裂纹但未断裂、局部漆面脱落、可修复钣金损伤。
    - severe：大面积变形 > 20cm、断裂/撕裂、部件脱落或缺失、需要更换的损伤。
13. 只输出 JSON，不要额外文字。
"""


def _build_covered_regions_text(view_id: str, topology: Any) -> str:
    """Build a human-readable description of regions covered by this view."""
    from agents.view_mapping import ROOF_SUB_REGION_TO_PARTS

    regions = get_regions_for_view(view_id)
    if not regions:
        return "（本次视角不覆盖外部区域）"

    lines: List[str] = []
    for region in regions:
        # Roof sub-regions map to specific standard parts.
        if region.startswith("roof_"):
            part_ids = ROOF_SUB_REGION_TO_PARTS.get(region, [])
            part_names = []
            seen = set()
            for pid in part_ids:
                name = topology.nodes[pid].node_name if topology and pid in topology.nodes else pid
                if name not in seen:
                    part_names.append(name)
                    seen.add(name)
            lines.append(f"- {region} 区域（车顶）：{', '.join(part_names) if part_names else '无具体部件'}")
            continue

        nodes = topology.get_nodes_by_region(region) if topology else []
        part_names = [n.node_name for n in nodes]
        lines.append(f"- {region} 区域：{', '.join(part_names) if part_names else '无具体部件'}")
    return "\n".join(lines)


def _build_checklist(view_id: str, topology: Any) -> List[Dict[str, str]]:
    """Return the focused checklist of parts this view should clearly evaluate.

    Unlike the old region-based checklist, this version filters parts by the
    visibility map in ``PARTS_TOPOLOGY``.  Only parts whose canonical visibility
    angles include the current ``view_id`` are required.  Distant or opposite-side
    parts that merely fall inside the broad region are moved to
    ``additional_findings`` instead of being forced into ``parts``.

    The list is sorted by region and then by catalog order for stable prompts.
    """
    from agents.view_mapping import ROOF_SUB_REGION_TO_PARTS
    from config import PARTS_TOPOLOGY

    visibility = PARTS_TOPOLOGY.get("visibility", {})
    regions = get_regions_for_view(view_id)
    checklist: List[Dict[str, str]] = []
    seen_ids: set = set()

    for region in regions:
        if region.startswith("roof_"):
            part_ids = ROOF_SUB_REGION_TO_PARTS.get(region, [])
        else:
            nodes = topology.get_nodes_by_region(region) if topology else []
            part_ids = [n.part_id for n in nodes]

        for part_id in part_ids:
            if part_id in seen_ids:
                continue
            # Only require the part if this view is one of its canonical angles.
            if view_id not in visibility.get(part_id, []):
                continue
            seen_ids.add(part_id)
            part_name = PARTS_BY_ID.get(part_id, {}).get("part_name", part_id)
            checklist.append({"part_id": part_id, "part_name": part_name, "region": region})

    return checklist


def _build_checklist_text(checklist: List[Dict[str, str]], view_id: str) -> str:
    """Format the checklist as numbered lines with visibility expectations."""
    if not checklist:
        return "（本次视角无强制检查的外部部件清单）"

    lines = []
    for i, item in enumerate(checklist, start=1):
        part_id = item["part_id"]
        part_name = item["part_name"]
        # Visibility hint for the LLM.
        if part_id.startswith("roof_") or part_id == "sunroof_glass" or part_id == "roof_rack":
            hint = (
                "车顶/天窗在本视角中通常只能看到边缘轮廓；若仅见边缘且无凹陷/变形/裂纹/玻璃碎裂等明显异常，应标 intact；"
                "只有当主体面板明确可见变形、裂纹、碎裂或明显缺失时才标 damaged"
            )
        elif part_id in ("headlight_front_left", "headlight_front_right", "taillight_rear_left", "taillight_rear_right"):
            hint = "灯具本体可见即可评估；无裂纹/破损/进水痕迹则标 intact；远端仅见灯壳边缘时可标 uncertain"
        elif part_id in ("mirror_left", "mirror_right"):
            hint = (
                "后视镜本体可见即可评估；必须看到镜壳外侧，外壳完整无裂纹则标 intact；"
                "仅看到镜面反光、镜壳内侧或边缘时标 uncertain"
            )
        elif part_id.startswith("door_"):
            hint = "门板主体可见（≥30% 面积或门把手/腰线轮廓清晰）才给出明确结论；仅看到车门边缘/窗框/门缝时请标 uncertain；有凹陷/划痕/变形/漆面脱落才标 damaged"
        elif part_id.startswith("fender_") or part_id in ("bumper_front", "bumper_rear"):
            hint = "主体面板可见即可评估；无凹陷/划痕/变形/漆面脱落则标 intact；有损伤时按凹陷/划痕/变形面积选择 damage_level（light/moderate/severe）"
        elif "windshield" in part_id:
            hint = "玻璃区域可见即可评估；无裂纹/碎裂则标 intact"
        elif part_id in ("hood", "grille_front"):
            hint = "主体可见即可评估；前保险杠若被白条、车牌、强光反光遮挡，遮挡区域不要判为损伤"
        else:
            hint = "该部件在本视角下应主体可见；无异常则标 intact，有异常按实际损伤程度选择 level"
        lines.append(f"{i}. {part_id}（{part_name}）— {hint}")
    if len(lines) == 0:
        return "（本次视角无强制检查的外部部件清单）"
    return "\n".join(lines)


def _make_uncertain_part(
    item: Dict[str, str],
    photo_ids: List[str],
    context_states: List[PartActualState] | None = None,
) -> PartActualState:
    """Create a PartActualState for a checklist item the LLM omitted.

    When the rest of the view is unanimously intact, it is more likely that the
    LLM simply forgot the item than that it is genuinely unrecognisable.  In
    that case we mark it intact with low confidence so the synthesizer can use
    the concrete status and fall back to other views if they disagree.
    """
    if context_states:
        concrete = [s for s in context_states if s.status != Status.UNCERTAIN]
        if concrete and all(s.status == Status.INTACT for s in concrete):
            return PartActualState(
                part_id=item["part_id"],
                part_name=item["part_name"],
                region=item["region"],
                side="",
                status=Status.INTACT,
                damage_level=DamageLevel.NONE,
                damage_types=[],
                standard_exists=True,
                actual_visible=True,
                actual_present=True,
                confidence="low",
                evidence_photos=list(photo_ids),
                notes="checklist 要求评估该部件，子 agent 未明确输出；同视角其他部件均 intact，按可见性推断为 intact",
                adjacent_status={},
            )

    return PartActualState(
        part_id=item["part_id"],
        part_name=item["part_name"],
        region=item["region"],
        side="",
        status=Status.UNCERTAIN,
        damage_level=DamageLevel.UNKNOWN,
        damage_types=[],
        standard_exists=True,
        actual_visible=False,
        actual_present=True,
        confidence="low",
        evidence_photos=list(photo_ids),
        notes="checklist 要求评估该部件，但子 agent 输出中未包含，自动标记为 uncertain",
        adjacent_status={},
    )


def _llm_dict_to_part_actual_state(data: Dict[str, Any], region: str) -> PartActualState:
    """Convert an LLM dict to PartActualState."""
    status_str = data.get("status", "uncertain")
    damage_level_str = data.get("damage_level", "unknown")

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

    damage_types: List[str] = []
    raw_types = data.get("damage_type", data.get("damage_types", []))
    if isinstance(raw_types, str):
        if raw_types and raw_types != "none":
            damage_types = [t.strip() for t in raw_types.split(",") if t.strip()]
    elif isinstance(raw_types, list):
        damage_types = [str(t) for t in raw_types if t]

    evidence_photos: List[str] = []
    raw_photos = data.get("evidence_photo", data.get("evidence_photos", []))
    if isinstance(raw_photos, str):
        if raw_photos:
            evidence_photos = [p.strip() for p in raw_photos.split(",") if p.strip()]
    elif isinstance(raw_photos, list):
        evidence_photos = [str(p) for p in raw_photos if p]

    actual_visible = data.get("actual_visible", len(evidence_photos) > 0)
    actual_present = data.get("actual_present", status != Status.MISSING)

    return PartActualState(
        part_id=data.get("part_id", ""),
        part_name=data.get("part_name", ""),
        region=region,
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
        adjacent_status={},
    )


async def vision_subagent(
    view_id: str,
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
    topology: Any,
) -> Dict[str, Any]:
    """Run a vision subagent for a single view.

    Parameters
    ----------
    view_id:
        Canonical view identifier (e.g. ``front_left``).
    photos:
        Photos assigned to this view by the planner.
    vehicle_prior:
        Output from ``vehicle_prior_agent``.
    topology:
        ``VehicleTopology`` for the vehicle.

    Returns
    -------
    dict
        ``{"view_id": ..., "regions": [...], "parts": [...], "part_actual_states": [...], "uncertain_items": [...]}``
    """
    if not photos:
        return {
            "view_id": view_id,
            "regions": get_regions_for_view(view_id),
            "parts": [],
            "part_actual_states": [],
            "uncertain_items": [],
        }

    vehicle_name = vehicle_prior.get("vehicle", "该车")
    view_display_name = get_display_name(view_id)
    checklist = _build_checklist(view_id, topology)
    checklist_text = _build_checklist_text(checklist, view_id)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        view_id=view_id,
        view_display_name=view_display_name,
        checklist_text=checklist_text,
        vehicle_name=vehicle_name,
    )

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": f"以下是 {len(photos)} 张{view_display_name}照片，请联合分析："},
    ]

    for photo in photos:
        content.append({"type": "text", "text": f"照片编号: {photo.get('id', '')}"})
        content.append(build_image_content(photo.get("path") or photo.get("url", ""), max_width=IMAGE_MAX_WIDTH))

    messages = [{"role": "user", "content": content}]
    raw = await call_minimax(messages, temperature=0.0, max_tokens=3000)
    result = extract_json(raw) or {}

    if not isinstance(result, dict):
        result = {}

    parts: List[Dict[str, Any]] = []
    part_actual_states: List[PartActualState] = []
    seen_ids: set = set()

    for part_dict in result.get("parts", []):
        if not isinstance(part_dict, dict):
            continue
        raw_part_id = part_dict.get("part_id", "")
        part_id = _normalize_part_id(raw_part_id)
        if not part_id or part_id in seen_ids:
            continue
        seen_ids.add(part_id)
        part_dict["part_id"] = part_id

        # Enrich region/side/part_name from topology/PARTS_BY_ID
        if part_id in PARTS_BY_ID:
            part_dict.setdefault("part_name", PARTS_BY_ID[part_id]["part_name"])
        if topology:
            node = topology.get_node(part_id)
            if node:
                part_dict.setdefault("region", node.region)
                part_dict.setdefault("side", node.side)

        state = _llm_dict_to_part_actual_state(part_dict, part_dict.get("region", ""))
        part_actual_states.append(state)
        parts.append(state.to_legacy_dict())

    # Backfill any checklist items the LLM omitted so callers always get a
    # complete, predictable set of parts for this view.  Use the concrete states
    # produced by the LLM as context when deciding whether an omission is likely
    # a genuine visibility problem or just a forgotten checklist item.
    photo_ids = [p.get("id", "") for p in photos if p.get("id")]
    checklist_ids = {item["part_id"] for item in checklist}
    missing_checklist_ids = checklist_ids - seen_ids
    concrete_states = [s for s in part_actual_states if s.status != Status.UNCERTAIN]
    for item in checklist:
        if item["part_id"] not in missing_checklist_ids:
            continue
        state = _make_uncertain_part(item, photo_ids, concrete_states)
        part_actual_states.append(state)
        parts.append(state.to_legacy_dict())
        seen_ids.add(item["part_id"])

    uncertain_items = [
        item for item in result.get("uncertain_items", [])
        if isinstance(item, dict)
    ]

    additional_findings = [
        item for item in result.get("additional_findings", [])
        if isinstance(item, dict)
    ]

    # If the subagent produced no parts at all, something went wrong with JSON
    # parsing or the model ignored the schema.  Treat that as an anomaly.
    if not parts:
        raise RuntimeError(
            f"Vision subagent for {view_id} returned no parts; treating as anomaly"
        )

    return {
        "view_id": view_id,
        "regions": get_regions_for_view(view_id),
        "parts": parts,
        "part_actual_states": part_actual_states,
        "uncertain_items": uncertain_items,
        "additional_findings": additional_findings,
    }
