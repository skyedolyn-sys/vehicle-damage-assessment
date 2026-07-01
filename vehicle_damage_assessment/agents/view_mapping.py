"""Vehicle photo view mapping — standard views and their covered regions.

This module centralises the relationship between camera angles (views) and
the vehicle regions / parts that are visible from each angle.  It is used by:

- ``planner_agent`` to assign a standard view label to every photo.
- ``assessment_orchestrator`` to route photos to the correct vision subagents.
- ``vision_subagent`` to know which regions it should evaluate.
"""

from __future__ import annotations

from typing import Dict, List, Set


#: Canonical view identifiers used across the assessment pipeline.
#: Nine exterior buckets (cardinal sides, four diagonal corners, plus top-down)
#: plus non-exterior categories.
STANDARD_VIEWS = [
    "front",
    "front_left_45",
    "front_right_45",
    "rear",
    "rear_left_45",
    "rear_right_45",
    "left_90",
    "right_90",
    "top",
    "interior",
    "auxiliary",
    "unknown",
]

#: Human-readable Chinese names for each standard view.
VIEW_DISPLAY_NAMES: Dict[str, str] = {
    "front": "车头正前",
    "front_left_45": "车头左前45度",
    "front_right_45": "车头右前45度",
    "rear": "车尾正后",
    "rear_left_45": "车尾左后45度",
    "rear_right_45": "车尾右后45度",
    "left_90": "车辆左侧",
    "right_90": "车辆右侧",
    "top": "车顶俯视",
    "interior": "内饰",
    "auxiliary": "辅助信息",
    "unknown": "无法定位",
}

#: Mapping from view to the primary body region(s) evaluated by a subagent.
#: Regions are part_category values from config.PARTS_CATALOG.
VIEW_TO_REGIONS: Dict[str, List[str]] = {
    "front": ["front"],
    "front_left_45": ["front", "left", "roof_front"],
    "front_right_45": ["front", "right", "roof_front"],
    "rear": ["rear"],
    "rear_left_45": ["rear", "left", "roof_rear"],
    "rear_right_45": ["rear", "right", "roof_rear"],
    "left_90": ["left"],
    "right_90": ["right"],
    "top": ["roof", "roof_front", "roof_middle", "roof_rear"],
    "interior": [],
    "auxiliary": [],
    "unknown": [],
}

#: Mapping from roof sub-region to the standard part ids that belong there.
#: Kept for compatibility with vision_subagent / topology_builder.
ROOF_SUB_REGION_TO_PARTS: Dict[str, List[str]] = {
    "roof_front": ["roof_front", "sunroof_glass"],
    "roof_middle": ["roof_middle", "sunroof_glass", "roof_rack"],
    "roof_rear": ["roof_rear"],
    "roof": ["roof_front", "roof_middle", "roof_rear", "sunroof_glass", "roof_rack"],
}

VIEW_COVERAGE_DESCRIPTION: Dict[str, str] = {
    "front": "车头正前",
    "front_left_45": "车头前部 + 车辆左侧 + A柱/左前门",
    "front_right_45": "车头前部 + 车辆右侧 + A柱/右前门",
    "rear": "车尾正后",
    "rear_left_45": "车尾后部 + 车辆左侧 + C柱/左后门",
    "rear_right_45": "车尾后部 + 车辆右侧 + C柱/右后门",
    "left_90": "车辆左侧（含A/B/C柱、左前后门、左后视镜、左后翼子板）",
    "right_90": "车辆右侧（含A/B/C柱、右前后门、右后视镜、右后翼子板）",
    "top": "车顶俯视（含车顶前/中/后部、天窗、车顶行李架）",
}

#: Set of views that are useful for exterior damage assessment.
EXTERIOR_VIEWS: Set[str] = {
    "front",
    "front_left_45",
    "front_right_45",
    "rear",
    "rear_left_45",
    "rear_right_45",
    "left_90",
    "right_90",
    "top",
}

#: Views that should be ignored by the exterior assessment pipeline.
NON_EXTERIOR_VIEWS: Set[str] = {"interior", "auxiliary", "unknown"}

#: Photo type categories used by the pre-filter before view planning.
PHOTO_TYPE_CATEGORIES: Set[str] = {"exterior", "interior", "auxiliary", "unknown"}


def get_regions_for_view(view_id: str) -> List[str]:
    """Return the regions covered by a standard view.

    Returns an empty list for interior / auxiliary / unknown views.
    """
    normalized = _normalize_view_id(view_id)
    return list(VIEW_TO_REGIONS.get(normalized, []))


def get_display_name(view_id: str) -> str:
    """Return the human-readable Chinese name for a view."""
    normalized = _normalize_view_id(view_id)
    return VIEW_DISPLAY_NAMES.get(normalized, view_id)


def is_exterior_view(view_id: str) -> bool:
    """Return True if the view contributes to exterior damage assessment."""
    return _normalize_view_id(view_id) in EXTERIOR_VIEWS


def _normalize_view_id(view_id: str) -> str:
    """Map common view aliases to canonical view ids."""
    aliases = {
        "front_left": "front_left_45",
        "front_right": "front_right_45",
        "rear_left": "rear_left_45",
        "rear_right": "rear_right_45",
        "left": "left_90",
        "right": "right_90",
        "top": "top",
    }
    return aliases.get(view_id, view_id)


def get_all_exterior_views() -> List[str]:
    """Return all exterior view identifiers."""
    return sorted(EXTERIOR_VIEWS)


def get_parts_for_roof_sub_region(sub_region: str) -> List[str]:
    """Return the standard part ids that belong to a roof sub-region."""
    return list(ROOF_SUB_REGION_TO_PARTS.get(sub_region, []))


def get_view_selection_prompt() -> str:
    """Return the prompt snippet that constrains planner view selection."""
    lines = ["标准视角选项（必须严格从中选择）："]
    for view_id in STANDARD_VIEWS:
        description = VIEW_COVERAGE_DESCRIPTION.get(view_id)
        if description:
            lines.append(f"  - {view_id}（{VIEW_DISPLAY_NAMES[view_id]}）→ {description}")
        else:
            lines.append(f"  - {view_id}（{VIEW_DISPLAY_NAMES[view_id]}）")
    return "\n".join(lines)


def normalize_view_id(raw: str) -> str:
    """Normalise a free-form view string to a canonical view id.

    The planner may output a Chinese description or an id; this function maps
    common variants back to ``STANDARD_VIEWS``.
    """
    if not raw:
        return "unknown"

    raw = raw.strip().lower()

    # Direct id match
    if raw in STANDARD_VIEWS:
        return raw

    # Common Chinese aliases and legacy ids
    aliases: Dict[str, str] = {
        "车头": "front",
        "车头正前": "front",
        "正前": "front",
        "正面": "front",
        "车头左侧": "front_left_45",
        "左前": "front_left_45",
        "车头左前": "front_left_45",
        "车头左前45度": "front_left_45",
        "车头右侧": "front_right_45",
        "右前": "front_right_45",
        "车头右前": "front_right_45",
        "车头右前45度": "front_right_45",
        "车尾": "rear",
        "车尾正后": "rear",
        "正后": "rear",
        "后面": "rear",
        "车尾左侧": "rear_left_45",
        "左后": "rear_left_45",
        "车尾左后": "rear_left_45",
        "车尾左后45度": "rear_left_45",
        "车尾右侧": "rear_right_45",
        "右后": "rear_right_45",
        "车尾右后": "rear_right_45",
        "车尾右后45度": "rear_right_45",
        "左侧": "left_90",
        "左侧90度": "left_90",
        "左侧正侧": "left_90",
        "左侧面": "left_90",
        "右侧": "right_90",
        "右侧90度": "right_90",
        "右侧正侧": "right_90",
        "右侧面": "right_90",
        "车顶": "top",
        "车顶俯视": "top",
        "俯视": "top",
        "顶": "top",
        "内饰": "interior",
        "车内": "interior",
        "辅助信息": "auxiliary",
        "证件": "auxiliary",
        "行驶证": "auxiliary",
        "vin": "auxiliary",
        "铭牌": "auxiliary",
        "无法定位": "unknown",
        "未知": "unknown",
        # Legacy normalizations
        "front_left": "front_left_45",
        "front_right": "front_right_45",
        "rear_left": "rear_left_45",
        "rear_right": "rear_right_45",
        "left": "left_90",
        "right": "right_90",
    }

    return aliases.get(raw, "unknown")
