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
import logging
import os
import re
from typing import Any, Dict, List, Tuple

from agents.minimax_client import call_minimax, build_image_content, extract_json
from agents.rules import render_prompt_template
from agents.view_mapping import get_display_name, get_regions_for_view, ROOF_SUB_REGION_TO_PARTS
from config import IMAGE_MAX_WIDTH, PARTS_BY_ID
from models import PartActualState, Status, DamageLevel

logger = logging.getLogger(__name__)
# Dedicated file log so Django console log level does not swallow subagent diagnostics.
_vision_file_handler = logging.FileHandler(
    os.path.expanduser("~/vehicle_damage_assessment_vision.log"), mode="a", encoding="utf-8"
)
_vision_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_vision_file_handler.setLevel(logging.INFO)
logger.addHandler(_vision_file_handler)
logger.setLevel(logging.INFO)


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



def _build_system_prompt(view_id: str, view_display_name: str, checklist_text: str, vehicle_name: str) -> str:
    """Render the vision subagent system prompt from the rules package template."""
    return render_prompt_template(
        "vision_system_prompt",
        view_id=view_id,
        view_display_name=view_display_name,
        checklist_text=checklist_text,
        vehicle_name=vehicle_name,
    )


def _build_covered_regions_text(view_id: str, topology: Any) -> str:
    """Build a human-readable description of regions covered by this view."""
    regions = get_regions_for_view(view_id)
    if not regions:
        return "（本次视角不覆盖外部区域）"

    lines: List[str] = []
    for region in regions:
        # Roof sub-regions map to specific standard parts.
        if region.startswith("roof_"):
            part_ids = ROOF_SUB_REGION_TO_PARTS.get(region, [])
            seen = set()
            part_names = [
                topology.nodes[pid].node_name if topology and pid in topology.nodes else pid
                for pid in part_ids
                if not (pid in seen or seen.add(pid))
            ]
            lines.append(f"- {region} 区域（车顶）：{', '.join(part_names) if part_names else '无具体部件'}")
            continue

        nodes = topology.get_nodes_by_region(region) if topology else []
        part_names = [n.node_name for n in nodes]
        lines.append(f"- {region} 区域：{', '.join(part_names) if part_names else '无具体部件'}")
    return "\n".join(lines)


def _build_checklist(view_id: str, topology: Any) -> List[Dict[str, str]]:
    """Return the focused checklist of parts this view should clearly evaluate."""
    from config import PARTS_TOPOLOGY

    visibility = PARTS_TOPOLOGY.get("visibility", {})
    regions = get_regions_for_view(view_id)
    checklist: List[Dict[str, str]] = []
    seen_ids: set = set()

    for region in regions:
        if region.startswith("roof_"):
            part_ids = ROOF_SUB_REGION_TO_PARTS.get(region, [])
        elif topology is not None:
            part_ids = [n.part_id for n in topology.get_nodes_by_region(region)]
        else:
            part_ids = []

        for part_id in part_ids:
            if part_id in seen_ids or view_id not in visibility.get(part_id, []):
                continue
            seen_ids.add(part_id)
            part_name = PARTS_BY_ID.get(part_id, {}).get("part_name", part_id)
            checklist.append({"part_id": part_id, "part_name": part_name, "region": region})

    return checklist


_CHECKLIST_HINTS: List[tuple[str, str]] = [
    ("roof_", "车顶/天窗在本视角中通常只能看到边缘轮廓；若仅见边缘且无凹陷/变形/裂纹/玻璃碎裂等明显异常，应标 intact；只有当主体面板明确可见变形、裂纹、碎裂或明显缺失时才标 damaged"),
    ("sunroof_glass", "车顶/天窗在本视角中通常只能看到边缘轮廓；若仅见边缘且无凹陷/变形/裂纹/玻璃碎裂等明显异常，应标 intact；只有当主体面板明确可见变形、裂纹、碎裂或明显缺失时才标 damaged"),
    ("roof_rack", "车顶/天窗在本视角中通常只能看到边缘轮廓；若仅见边缘且无凹陷/变形/裂纹/玻璃碎裂等明显异常，应标 intact；只有当主体面板明确可见变形、裂纹、碎裂或明显缺失时才标 damaged"),
    ("headlight_", "灯具本体可见即可评估；无裂纹/破损/进水痕迹则标 intact；远端仅见灯壳边缘时可标 uncertain"),
    ("taillight_", "灯具本体可见即可评估；若灯壳完整、无裂纹/破损/进水痕迹则标 intact；若尾灯区域被严重遮挡、仅残留边缘、灯壳不可辨识或周围钣金严重撕裂变形，应优先标 damaged severe 或 missing，不要仅因'未见裂纹'就判 intact；远端仅见灯壳边缘且无明显异常时可标 uncertain"),
    ("mirror_", "后视镜：只要镜壳外侧任何部分可见，且该可见部分无裂纹/破损/变形/脱落，即标 intact（confidence=low），并在 notes 中说明可见程度；只有当后视镜完全不可见，或可见部分存在明确损伤时，才标 damaged/uncertain"),
    ("door_", "严格区分门板主体与相邻翼子板/C柱/轮拱：仅评估车门面板本身（含门把手、腰线、窗下沿以下板面）。若凹陷/变形实际位于翼子板、C柱或后翼子板轮拱区域，即使靠近车门边缘，也不要判为车门损伤；车门板主体无异常则标 intact；门板主体明确可见凹陷/划痕/变形/漆面脱落才标 damaged；仅看到车门边缘/窗框/门缝时请标 uncertain"),
    ("pillar_", "立柱属于结构性安全件：只要画面中能看到该立柱且存在变形、断裂、褶皱、撕裂、钣金错位等任何结构异常，即标 damaged severe；立柱轻微刮擦可标 light；完全看不到该立柱时才标 uncertain。特别注意：左前45度/右前45度视角必须明确评估A柱，左侧/右侧正视角必须明确评估B柱，左后45度/右后45度视角必须明确评估C柱，不要遗漏"),
    ("fender_", "主体面板可见即可评估；无凹陷/划痕/变形/漆面脱落则标 intact；有损伤时按凹陷/划痕/变形面积选择 damage_level（light/moderate/severe）"),
    ("bumper_", "主体面板可见即可评估；无凹陷/划痕/变形/漆面脱落则标 intact；有损伤时按凹陷/划痕/变形面积选择 damage_level（light/moderate/severe）"),
    ("windshield", "玻璃区域可见即可评估；无裂纹/碎裂则标 intact"),
    ("hood", "主体可见即可评估；前保险杠若被白条、车牌、强光反光遮挡，遮挡区域不要判为损伤"),
    ("grille_front", "主体可见即可评估；前保险杠若被白条、车牌、强光反光遮挡，遮挡区域不要判为损伤"),
]


def _hint_for_part(part_id: str) -> str:
    """Return the visibility hint for a part id."""
    for prefix, hint in _CHECKLIST_HINTS:
        if part_id.startswith(prefix) or part_id == prefix:
            return hint
    return "该部件在本视角下应主体可见；无异常则标 intact，有异常按实际损伤程度选择 level"


def _build_checklist_text(checklist: List[Dict[str, str]], view_id: str) -> str:
    """Format the checklist as numbered lines with visibility expectations."""
    if not checklist:
        return "（本次视角无强制检查的外部部件清单）"

    lines = [f"{i}. {item['part_id']}（{item['part_name']}）— {_hint_for_part(item['part_id'])}" for i, item in enumerate(checklist, start=1)]
    return "\n".join(lines) if lines else "（本次视角无强制检查的外部部件清单）"


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
                photo_type="unknown",
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
        photo_type="unknown",
    )


def _parse_string_or_list(raw: Any, sep: str = ",") -> List[str]:
    """Normalize a string-or-list field from the LLM into a list of strings."""
    if isinstance(raw, str):
        if not raw or raw == "none":
            return []
        return [x.strip() for x in raw.split(sep) if x.strip()]
    return [str(x) for x in raw if x]


def _resolve_photo_type(evidence_photos: List[str], photo_type_lookup: Dict[str, str]) -> str:
    """Map evidence photo ids to a single representative photo_type.

    Priority: wide_shot > close_up_damage > close_up_detail > unknown.
    """
    types = {photo_type_lookup.get(pid, "unknown") for pid in evidence_photos}
    for preferred in ("wide_shot", "close_up_damage", "close_up_detail"):
        if preferred in types:
            return preferred
    return "unknown"


def _llm_dict_to_part_actual_state(
    data: Dict[str, Any], region: str, photo_type: str = "unknown"
) -> PartActualState:
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

    damage_types = _parse_string_or_list(data.get("damage_type", data.get("damage_types", [])))
    evidence_photos = _parse_string_or_list(data.get("evidence_photo", data.get("evidence_photos", [])))

    actual_visible = data.get("actual_visible", len(evidence_photos) > 0)
    actual_present = data.get("actual_present", status != Status.MISSING)

    evidence_source = {
        "region": region,
        "status": status.value,
        "damage_level": damage_level.value,
        "confidence": data.get("confidence", "low"),
        "evidence_photo": evidence_photos,
        "notes": data.get("notes", ""),
    }

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
        photo_type=photo_type,
        evidence_sources=[evidence_source],
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

    system_prompt = _build_system_prompt(
        view_id=view_id,
        view_display_name=view_display_name,
        checklist_text=checklist_text,
        vehicle_name=vehicle_name,
    )

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": f"以下是 {len(photos)} 张{view_display_name}照片，请联合分析："},
    ]

    photo_type_lookup: Dict[str, str] = {}
    for photo in photos:
        photo_id = photo.get("id", "")
        if photo_id:
            photo_type_lookup[photo_id] = photo.get("_planner_photo_type", "unknown")

    for photo in photos:
        content.append({"type": "text", "text": f"照片编号: {photo.get('id', '')}"})
        content.append(build_image_content(photo.get("path") or photo.get("url", ""), max_width=IMAGE_MAX_WIDTH))

    messages = [{"role": "user", "content": content}]
    photo_ids = [p.get("id", "") for p in photos if p.get("id")]
    logger.info("[vision:%s] start checklist=%s photo_ids=%s", view_id, [c["part_id"] for c in checklist], photo_ids)
    raw = await call_minimax(messages, temperature=0.0, max_tokens=3000)
    result = extract_json(raw) or {}
    logger.info("[vision:%s] result type=%s has_parts=%s has_uncertain=%s", view_id, type(result).__name__, bool(result.get("parts")), bool(result.get("uncertain_items")))

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

        evidence_photos = _parse_string_or_list(part_dict.get("evidence_photo", part_dict.get("evidence_photos", [])))

        photo_type = _resolve_photo_type(evidence_photos, photo_type_lookup)
        part_dict["photo_type"] = photo_type

        state = _llm_dict_to_part_actual_state(
            part_dict, part_dict.get("region", ""), photo_type=photo_type
        )
        part_actual_states.append(state)
        parts.append(state.to_legacy_dict())

    # Backfill any checklist items the LLM omitted so callers always get a
    # complete, predictable set of parts for this view.  Use the concrete states
    # produced by the LLM as context when deciding whether an omission is likely
    # a genuine visibility problem or just a forgotten checklist item.
    checklist_ids = {item["part_id"] for item in checklist}
    missing_checklist_ids = checklist_ids - seen_ids
    concrete_states = [s for s in part_actual_states if s.status != Status.UNCERTAIN]
    logger.info("[vision:%s] seen=%s missing=%s concrete=%d", view_id, sorted(seen_ids), sorted(missing_checklist_ids), len(concrete_states))
    for item in checklist:
        if item["part_id"] not in missing_checklist_ids:
            continue
        state = _make_uncertain_part(item, photo_ids, concrete_states)
        part_actual_states.append(state)
        parts.append(state.to_legacy_dict())
        seen_ids.add(item["part_id"])
        logger.info("[vision:%s] backfilled %s as %s", view_id, item["part_id"], state.status.value)

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
        logger.error("[vision:%s] no parts produced; raising anomaly", view_id)
        raise RuntimeError(
            f"Vision subagent for {view_id} returned no parts; treating as anomaly"
        )

    logger.info("[vision:%s] done parts=%d uncertain_items=%d additional=%d", view_id, len(parts), len(uncertain_items), len(additional_findings))
    return {
        "view_id": view_id,
        "regions": get_regions_for_view(view_id),
        "parts": parts,
        "part_actual_states": part_actual_states,
        "uncertain_items": uncertain_items,
        "additional_findings": additional_findings,
    }
