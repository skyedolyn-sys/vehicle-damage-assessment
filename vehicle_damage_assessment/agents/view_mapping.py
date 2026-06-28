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
#: Eight directional buckets: four cardinal sides plus four diagonal corners.
STANDARD_VIEWS = [
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
    "left",
    "right",
    "interior",
    "auxiliary",
    "unknown",
]

#: Human-readable Chinese names for each standard view.
VIEW_DISPLAY_NAMES: Dict[str, str] = {
    "front": "车头正前",
    "front_left": "车头左侧",
    "front_right": "车头右侧",
    "rear": "车尾正后",
    "rear_left": "车尾左侧",
    "rear_right": "车尾右侧",
    "left": "车辆左侧",
    "right": "车辆右侧",
    "interior": "内饰",
    "auxiliary": "辅助信息",
    "unknown": "无法定位",
}

#: Mapping from view to the primary region(s) evaluated by a subagent for that view.
#: A single view may cover parts from more than one region (e.g. front_left
#: covers both the front and the left side).  Each view is limited to at most
#: three regions: two side/body faces plus a roof edge.
VIEW_TO_REGIONS: Dict[str, List[str]] = {
    "front": ["front", "roof_front"],
    "front_left": ["front", "left", "roof_front"],
    "front_right": ["front", "right", "roof_front"],
    "rear": ["rear", "roof_rear"],
    "rear_left": ["rear", "left", "roof_rear"],
    "rear_right": ["rear", "right", "roof_rear"],
    "left": ["left", "roof_left"],
    "right": ["right", "roof_right"],
    "interior": [],
    "auxiliary": [],
    "unknown": [],
}

#: Mapping from roof sub-region to the standard part ids that belong there.
ROOF_SUB_REGION_TO_PARTS: Dict[str, List[str]] = {
    "roof_front": ["roof_front", "sunroof_glass"],
    "roof_middle": ["roof_middle", "sunroof_glass", "roof_rack"],
    "roof_rear": ["roof_rear"],
    "roof_left": ["roof_front", "roof_middle", "sunroof_glass"],
    "roof_right": ["roof_front", "roof_middle", "sunroof_glass"],
    "roof": ["roof_front", "roof_middle", "roof_rear", "sunroof_glass", "roof_rack"],
}

#: Mapping from view to the human-readable description of what it covers.
VIEW_COVERAGE_DESCRIPTION: Dict[str, str] = {
    "front": "车头正前 + 车顶前缘",
    "front_left": "车头前部 + 车辆左侧 + 车顶前左区域",
    "front_right": "车头前部 + 车辆右侧 + 车顶前右区域",
    "rear": "车尾正后 + 车顶后缘",
    "rear_left": "车尾后部 + 车辆左侧 + 车顶后左区域",
    "rear_right": "车尾后部 + 车辆右侧 + 车顶后右区域",
    "left": "车辆左侧 + 车顶左缘",
    "right": "车辆右侧 + 车顶右缘",
}

#: Set of views that are useful for exterior damage assessment.
EXTERIOR_VIEWS: Set[str] = {
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
    "left",
    "right",
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
        "front_left_45": "front_left",
        "front_right_45": "front_right",
        "rear_left_45": "rear_left",
        "rear_right_45": "rear_right",
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

    # Common Chinese aliases
    aliases: Dict[str, str] = {
        "车头": "front",
        "车头正前": "front",
        "正前": "front",
        "正面": "front",
        "车头左侧": "front_left",
        "左前": "front_left",
        "车头左前": "front_left",
        "车头右侧": "front_right",
        "右前": "front_right",
        "车头右前": "front_right",
        "车尾": "rear",
        "车尾正后": "rear",
        "正后": "rear",
        "后面": "rear",
        "车尾左侧": "rear_left",
        "左后": "rear_left",
        "车尾左后": "rear_left",
        "车尾右侧": "rear_right",
        "右后": "rear_right",
        "车尾右后": "rear_right",
        "左侧": "left",
        "左侧90度": "left",
        "左侧正侧": "left",
        "左侧面": "left",
        "右侧": "right",
        "右侧90度": "right",
        "右侧正侧": "right",
        "右侧面": "right",
        "内饰": "interior",
        "车内": "interior",
        "辅助信息": "auxiliary",
        "证件": "auxiliary",
        "行驶证": "auxiliary",
        "vin": "auxiliary",
        "铭牌": "auxiliary",
        "无法定位": "unknown",
        "未知": "unknown",
    }

    return aliases.get(raw, "unknown")
