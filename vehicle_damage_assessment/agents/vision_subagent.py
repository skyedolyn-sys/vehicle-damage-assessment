"""Vision subagent — per-view damage assessment.

Each vision subagent is an independent LLM call responsible for a single
standard view (e.g. ``front_left``).  It receives all photos assigned to
that view and evaluates every exterior region that the view covers.

The subagent does not decide which regions are covered — that is provided by
the orchestrator via ``view_mapping.VIEW_TO_REGIONS``.  This keeps the
subagent focused on visual recognition.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Set, Tuple

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



def _build_system_prompt(view_id: str, view_display_name: str, checklist_text: str, vehicle_name: str, minimal: bool = False) -> str:
    """Render the vision subagent system prompt from the rules package template."""
    template_name = "vision_system_prompt_minimal" if minimal else "vision_system_prompt"
    return render_prompt_template(
        template_name,
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


#: High-sensitivity parts (mirrors evidence_fusion._HIGH_SENSITIVITY_PARTS).
#: Per DAMAGE_RECOGNITION_POLICY §2.1, marking any of these ``intact`` requires
#: a positive evidence phrase in the notes field.
_HIGH_SENSITIVITY_PARTS: Set[str] = {
    "windshield_front", "windshield_rear", "sunroof_glass",
    "roof_front", "roof_middle", "roof_rear",
    "pillar_a_left", "pillar_a_right",
    "pillar_b_left", "pillar_b_right",
    "pillar_c_left", "pillar_c_right",
    "hood", "trunk_lid",
}

#: Accepted positive evidence phrases per category (from policy §2.1).
_POSITIVE_ANCHORS: Dict[str, List[str]] = {
    "windshield": ["玻璃面平整无裂纹", "无破碎", "玻璃与车架密封条完整", "玻璃完整"],
    "roof": ["钣金平整", "行李架对齐", "天窗框架无错位", "车顶线条连续", "车顶钣金平整", "面板完整"],
    "pillar": ["立柱直线无弯折", "与车顶/车门接缝齐平", "漆面连续", "立柱无变形", "立柱笔直"],
    "hood": ["钣金弧线连续", "与两侧翼子板接缝齐平", "漆面无裂纹", "弧线连续"],
    "trunk": ["钣金弧线连续", "与两侧翼子板接缝齐平", "漆面无裂纹", "弧线连续"],
}


def _positive_anchor_category(part_id: str) -> str:
    """Map a high-sensitivity part_id to its anchor category, or '' if not high-sensitivity."""
    if part_id.startswith("windshield") or part_id == "sunroof_glass":
        return "windshield"
    if part_id.startswith("roof"):
        return "roof"
    if part_id.startswith("pillar"):
        return "pillar"
    if part_id == "hood":
        return "hood"
    if part_id == "trunk_lid":
        return "trunk"
    return ""


def _check_positive_anchor(part_id: str, status: str, notes: str) -> bool:
    """Return True if intact high-sensitivity parts have a positive evidence phrase.

    Implements DAMAGE_RECOGNITION_POLICY §2.1 in the Python validation layer.
    """
    if status != "intact":
        return True
    if part_id not in _HIGH_SENSITIVITY_PARTS:
        return True
    category = _positive_anchor_category(part_id)
    if not category:
        return True
    notes_low = (notes or "").lower()
    return any(anchor.lower() in notes_low for anchor in _POSITIVE_ANCHORS[category])


def _enforce_positive_anchor(part_dict: Dict[str, Any]) -> None:
    """Downgrade intact high-sensitivity parts lacking positive anchor to uncertain.

    Per DAMAGE_RECOGNITION_POLICY §2.4, the Python validation layer is the last
    line of defense when the LLM fails to provide a positive evidence phrase.
    Mutates ``part_dict`` in place so downstream code picks up the downgraded state.
    """
    part_id = part_dict.get("part_id", "")
    status = part_dict.get("status", "uncertain")
    notes = part_dict.get("notes", "")
    if not _check_positive_anchor(part_id, status, notes):
        logger.info(
            "[vision] positive-anchor downgrade: %s status=%s → uncertain (notes=%r)",
            part_id, status, (notes or "")[:60],
        )
        part_dict["status"] = "uncertain"
        part_dict["damage_level"] = "unknown"
        prefix = "；缺少正向证据,已按政策 §2.1 降级为 uncertain"
        if notes:
            part_dict["notes"] = f"{notes}{prefix}"
        else:
            part_dict["notes"] = prefix.lstrip("；")
        part_dict["_positive_anchor_downgraded"] = True


def _should_downgrade_edge_visible(part_id: str, part_dict: Dict[str, Any]) -> bool:
    """Return True if a high-sensitivity intact part is only edge-visible.

    DAMAGE_RECOGNITION_POLICY §2.2: 仅边缘可见(占比 < 30%)且无明显损伤痕迹时,
    必须标 uncertain。Python 层启发式:
    - 仅 high-sensitivity 部件受影响
    - 仅 intact 状态受影响
    - close_up_damage 表明聚焦,允许 intact(主体可见)
    - high confidence 表明 LLM 已观察足够多,不降级
    - 其他情况(wide_shot/close_up_detail/unknown + low/medium confidence)→ 降级
    """
    if part_id not in _HIGH_SENSITIVITY_PARTS:
        return False
    if part_dict.get("status") != "intact":
        return False
    photo_type = part_dict.get("photo_type", "unknown")
    if photo_type == "close_up_damage":
        return False  # 特写聚焦,允许 intact
    confidence = part_dict.get("confidence", "low")
    if confidence == "high":
        return False  # high confidence + intact 通常有足够证据
    return True


def _enforce_edge_visible(part_dict: Dict[str, Any]) -> None:
    """Downgrade intact high-sensitivity parts only seen at edge to uncertain.

    Implements DAMAGE_RECOGNITION_POLICY §2.2 in the Python validation layer.
    Should run after ``_enforce_positive_anchor`` since §2.1 already downgrades
    some intact → uncertain, eliminating the need for §2.2 to consider them.
    """
    part_id = part_dict.get("part_id", "")
    # §2.1 已降级的部件无需再走 §2.2
    if part_dict.get("_positive_anchor_downgraded"):
        return
    if not _should_downgrade_edge_visible(part_id, part_dict):
        return
    evidence_photos = _parse_string_or_list(part_dict.get("evidence_photo", []))
    logger.info(
        "[vision] edge-visible downgrade: %s photos=%d photo_type=%s confidence=%s → uncertain",
        part_id, len(evidence_photos),
        part_dict.get("photo_type", "unknown"), part_dict.get("confidence", "low"),
    )
    part_dict["status"] = "uncertain"
    part_dict["damage_level"] = "unknown"
    prefix = "；仅边缘可见(≤1 张证据照片),已按政策 §2.2 降级为 uncertain"
    existing = part_dict.get("notes", "")
    part_dict["notes"] = f"{existing}{prefix}" if existing else prefix.lstrip("；")
    part_dict["_edge_visible_downgraded"] = True


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
                part_category=item["region"],
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
        part_category=item["region"],
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
    """Normalize a string-or-list field from the LLM into a list of strings.

    Thin wrapper that delegates the shape-conversion work to the canonical
    :func:`agents.evidence_photo.to_photo_list`. The ``sep`` parameter is
    retained for backwards compatibility; only ``","`` is exercised today.
    """
    from agents.evidence_photo import to_photo_list
    if sep != ",":
        # Fallback for any future non-comma callers: replicate the legacy
        # split-by-arbitrary-separator behaviour without going through the
        # canonical helper (which is comma-specific by contract).
        if isinstance(raw, str):
            if not raw or raw.strip().lower() == "none":
                return []
            return [x.strip() for x in raw.split(sep) if x.strip()]
        return [str(x).strip() for x in raw if x is not None and str(x).strip()]
    return to_photo_list(raw)


def _normalise_damage_types(raw: Any) -> List[str]:
    """Return a canonical damage_type list, validating against the central allow-list.

    Accepts:
      - None / "" / "none" / [] → ["none"]
      - str (single value, may be whitespace, may be csv "a, b, c")
      - list/tuple of strings (may contain aliases or unknown values)

    Applies alias mapping first, then allow-list filter. Unknown values
    fall back to the configured default. Empty result becomes ["none"].
    """
    from agents.rules import load_damage_type_allowlist

    allowlist = load_damage_type_allowlist()
    allowed = set(allowlist["allowed"])
    default = allowlist["default"]
    aliases = allowlist["aliases"]

    if raw is None:
        return [default]
    if isinstance(raw, str):
        raw = [t.strip() for t in raw.split(",") if t.strip()]
    if not isinstance(raw, (list, tuple)):
        return [default]
    out: List[str] = []
    for item in raw:
        if item is None:
            continue
        s = str(item).strip().lower()
        if not s:
            continue
        canonical = aliases.get(s, s)
        if canonical in allowed:
            if canonical not in out:
                out.append(canonical)
        else:
            if default not in out:
                out.append(default)
    if not out:
        return [default]
    return out


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
    data: Dict[str, Any],
    region: str,
    photo_type: str = "unknown",
    view_id: str = "",
) -> PartActualState:
    """Convert an LLM dict to PartActualState.

    The ``region`` argument is the topology part_category (e.g. "front").
    The optional ``view_id`` argument records the camera angle that produced
    this observation (e.g. "front_left_45").  Both are written into the
    evidence_source dict using their semantic names so downstream consumers
    can compare against the right reference set (view_id vs part_category).
    """
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

    damage_types = _normalise_damage_types(data.get("damage_type", data.get("damage_types", [])))
    evidence_photos = _parse_string_or_list(data.get("evidence_photo", data.get("evidence_photos", [])))

    actual_visible = data.get("actual_visible", len(evidence_photos) > 0)
    actual_present = data.get("actual_present", status != Status.MISSING)

    evidence_source = {
        "view_id": view_id,
        "part_category": region,
        "status": status.value,
        "damage_level": damage_level.value,
        "confidence": data.get("confidence", "low"),
        "evidence_photo": evidence_photos,
        "notes": data.get("notes", ""),
        "photo_type": photo_type,
    }

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
        adjacent_status={},
        photo_type=photo_type,
        evidence_sources=[evidence_source],
    )


async def _run_single_vision_attempt(
    view_id: str,
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
    topology: Any,
    minimal: bool = False,
) -> Tuple[Dict[str, Any], List[str], bool]:
    """Run one vision subagent attempt.

    Returns ``(parsed_result, photo_ids, llm_returned_empty)``.
    ``llm_returned_empty`` is True when the LLM output contained no usable
    ``parts`` array, which is the primary stability failure mode we want to
    retry.
    """
    vehicle_name = vehicle_prior.get("vehicle", "该车")
    view_display_name = get_display_name(view_id)
    checklist = _build_checklist(view_id, topology)
    checklist_text = _build_checklist_text(checklist, view_id)

    system_prompt = _build_system_prompt(
        view_id=view_id,
        view_display_name=view_display_name,
        checklist_text=checklist_text,
        vehicle_name=vehicle_name,
        minimal=minimal,
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
    label = "minimal" if minimal else "primary"
    logger.info("[vision:%s] %s start checklist=%s photo_ids=%s", view_id, label, [c["part_id"] for c in checklist], photo_ids)

    max_tokens = 3000 if minimal else 5000
    temperature = 0.1 if minimal else 0.0
    raw = await call_minimax(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    extracted = extract_json(raw)
    if not isinstance(extracted, dict):
        logger.warning("[vision:%s] %s call returned non-object JSON (%s)", view_id, label, type(extracted).__name__)
        return {}, photo_ids, True

    logger.info("[vision:%s] %s result type=%s has_parts=%s has_uncertain=%s", view_id, label, type(extracted).__name__, bool(extracted.get("parts")), bool(extracted.get("uncertain_items")))
    return extracted, photo_ids, not bool(extracted.get("parts"))


async def _run_vision_with_fallback(
    view_id: str,
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
    topology: Any,
) -> Tuple[Dict[str, Any], List[str], bool]:
    """Run vision subagent, falling back to a minimal prompt if LLM omits JSON."""
    result, photo_ids, empty = await _run_single_vision_attempt(view_id, photos, vehicle_prior, topology, minimal=False)
    if empty:
        result, photo_ids, empty = await _run_single_vision_attempt(view_id, photos, vehicle_prior, topology, minimal=True)
    return result, photo_ids, empty


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

    checklist = _build_checklist(view_id, topology)

    photo_type_lookup: Dict[str, str] = {}
    for photo in photos:
        photo_id = photo.get("id", "")
        if photo_id:
            photo_type_lookup[photo_id] = photo.get("_planner_photo_type", "unknown")

    logger.info("[vision:%s] start checklist=%s photo_ids=%s", view_id, [c["part_id"] for c in checklist], [p.get("id", "") for p in photos if p.get("id")])
    result, photo_ids, llm_returned_empty = await _run_vision_with_fallback(
        view_id, photos, vehicle_prior, topology
    )

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

        # Enrich part_category/side/part_name from topology/PARTS_BY_ID
        if part_id in PARTS_BY_ID:
            part_dict.setdefault("part_name", PARTS_BY_ID[part_id]["part_name"])
        if topology:
            node = topology.get_node(part_id)
            if node:
                part_dict.setdefault("part_category", node.region)
                part_dict.setdefault("side", node.side)

        evidence_photos = _parse_string_or_list(part_dict.get("evidence_photo", part_dict.get("evidence_photos", [])))

        photo_type = _resolve_photo_type(evidence_photos, photo_type_lookup)
        part_dict["photo_type"] = photo_type

        # DAMAGE_RECOGNITION_POLICY §2.4: Python 校验层强制降级缺正向证据短语的高敏感部件。
        # 必须在 _llm_dict_to_part_actual_state 之前调用,使 PartActualState 继承降级后结论。
        _enforce_positive_anchor(part_dict)
        # DAMAGE_RECOGNITION_POLICY §2.2: 仅边缘可见的高敏感部件也强制降级。
        _enforce_edge_visible(part_dict)

        state = _llm_dict_to_part_actual_state(
            part_dict,
            part_dict.get("part_category", ""),
            photo_type=photo_type,
            view_id=view_id,
        )
        part_actual_states.append(state)
        parts.append(state.to_dict())

    # Backfill any checklist items the LLM omitted so callers always get a
    # complete, predictable set of parts for this view.  Use the concrete states
    # produced by the LLM as context when deciding whether an omission is likely
    # a genuine visibility problem or just a forgotten checklist item.
    checklist_ids = {item["part_id"] for item in checklist}
    missing_checklist_ids = checklist_ids - seen_ids
    concrete_states = [s for s in part_actual_states if s.status != Status.UNCERTAIN]
    logger.info("[vision:%s] seen=%s missing=%s concrete=%d llm_empty=%s", view_id, sorted(seen_ids), sorted(missing_checklist_ids), len(concrete_states), llm_returned_empty)
    for item in checklist:
        if item["part_id"] not in missing_checklist_ids:
            continue
        state = _make_uncertain_part(item, photo_ids, concrete_states)
        part_actual_states.append(state)
        parts.append(state.to_dict())
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
        "_llm_returned_empty": llm_returned_empty,
    }
