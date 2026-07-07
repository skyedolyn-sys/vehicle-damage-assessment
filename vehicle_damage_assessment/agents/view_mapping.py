"""Vehicle photo view mapping — standard views and their covered parts.

This module centralises the relationship between camera angles (views) and
the vehicle parts that are visible from each angle.  It is used by:

- ``planner_agent`` to classify photos into exterior / interior / etc.
- ``master_agent`` to route exterior photos to the ViewAgent Team.
- ``view_agent`` to know which parts it should evaluate for a given photo.

All view/part lists are fixed and match ``docs/schema-viewagent-team.md``.
"""

from __future__ import annotations

from typing import Dict, List, Set


#: View id used when a photo cannot be classified as a standard exterior view.
SCENE_INTAKE_VIEW = "scene_intake"

#: Canonical exterior view identifiers (9 fixed views, no numeric suffixes).
EXTERIOR_VIEWS: Set[str] = {
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
    "left",
    "right",
    "top",
}

#: Human-readable Chinese names for each exterior view.
VIEW_DISPLAY_NAMES: Dict[str, str] = {
    "front": "车头正前",
    "front_left": "车头左前",
    "front_right": "车头右前",
    "rear": "车尾正后",
    "rear_left": "车尾左后",
    "rear_right": "车尾右后",
    "left": "车辆左侧",
    "right": "车辆右侧",
    "top": "车顶俯视",
}

#: Mapping from view to the primary body region(s) evaluated by the view.
#: Regions are part_category values from config.PARTS_CATALOG.
VIEW_TO_REGIONS: Dict[str, List[str]] = {
    "front": ["front"],
    "front_left": ["front", "left", "roof_front"],
    "front_right": ["front", "right", "roof_front"],
    "rear": ["rear"],
    "rear_left": ["rear", "left", "roof_rear"],
    "rear_right": ["rear", "right", "roof_rear"],
    "left": ["left"],
    "right": ["right"],
    "top": ["roof", "roof_front", "roof_middle", "roof_rear"],
}

#: Mapping from view to the fixed list of visible part ids.
#: Part ids must match the 33-part catalog in config.PARTS_CATALOG.
VIEW_TO_PARTS: Dict[str, List[str]] = {
    "front": [
        "hood",
        "bumper_front",
        "headlight_front_left",
        "headlight_front_right",
        "grille_front",
        "fender_front_left",
        "fender_front_right",
        "windshield_front",
    ],
    "front_left": [
        "hood",
        "bumper_front",
        "headlight_front_left",
        "fender_front_left",
        "mirror_left",
        "door_front_left",
        "pillar_a_left",
        "roof_front",
    ],
    "front_right": [
        "hood",
        "bumper_front",
        "headlight_front_right",
        "fender_front_right",
        "mirror_right",
        "door_front_right",
        "pillar_a_right",
        "roof_front",
    ],
    "rear": [
        "trunk_lid",
        "tailgate",
        "bumper_rear",
        "taillight_rear_left",
        "taillight_rear_right",
        "windshield_rear",
    ],
    "rear_left": [
        "trunk_lid",
        "tailgate",
        "bumper_rear",
        "taillight_rear_left",
        "fender_rear_left",
        "door_rear_left",
        "mirror_left",
        "pillar_c_left",
        "roof_rear",
    ],
    "rear_right": [
        "trunk_lid",
        "tailgate",
        "bumper_rear",
        "taillight_rear_right",
        "fender_rear_right",
        "door_rear_right",
        "mirror_right",
        "pillar_c_right",
        "roof_rear",
    ],
    "left": [
        "door_front_left",
        "door_rear_left",
        "mirror_left",
        "fender_rear_left",
        "pillar_a_left",
        "pillar_b_left",
        "pillar_c_left",
    ],
    "right": [
        "door_front_right",
        "door_rear_right",
        "mirror_right",
        "fender_rear_right",
        "pillar_a_right",
        "pillar_b_right",
        "pillar_c_right",
    ],
    "top": [
        "roof_front",
        "roof_middle",
        "roof_rear",
        "sunroof_glass",
        "roof_rack",
    ],
}

#: Human-readable coverage descriptions for exterior views.
VIEW_COVERAGE_DESCRIPTION: Dict[str, str] = {
    "front": "车头正前",
    "front_left": "车头前部 + 车辆左侧 + A柱/左前门",
    "front_right": "车头前部 + 车辆右侧 + A柱/右前门",
    "rear": "车尾正后",
    "rear_left": "车尾后部 + 车辆左侧 + C柱/左后门",
    "rear_right": "车尾后部 + 车辆右侧 + C柱/右后门",
    "left": "车辆左侧（含A/B/C柱、左前后门、左后视镜、左后翼子板）",
    "right": "车辆右侧（含A/B/C柱、右前后门、右后视镜、右后翼子板）",
    "top": "车顶俯视（含车顶前/中/后部、天窗、车顶行李架）",
}

#: Non-exterior view ids used for routing / filtering.
#: ``auxiliary`` is kept for backward compatibility with legacy plans even
#: though the new planner emits ``vehicle_info`` instead.
NON_EXTERIOR_VIEWS: Set[str] = {
    "interior",
    "vehicle_info",
    "auxiliary",
    "exclude",
    "unknown",
    SCENE_INTAKE_VIEW,
}

#: Photo category labels produced by PlannerAgent and consumed downstream.
PHOTO_TYPE_CATEGORIES: Set[str] = {
    "exterior",
    "interior",
    "vehicle_info",
    "exclude",
    "scene_intake",
}

#: Mapping from roof sub-region to the standard part ids that belong there.
ROOF_SUB_REGION_TO_PARTS: Dict[str, List[str]] = {
    "roof_front": ["roof_front", "sunroof_glass"],
    "roof_middle": ["roof_middle", "sunroof_glass", "roof_rack"],
    "roof_rear": ["roof_rear"],
    "roof": ["roof_front", "roof_middle", "roof_rear", "sunroof_glass", "roof_rack"],
}


def get_regions_for_view(view_id: str) -> List[str]:
    """Return the regions covered by a standard view.

    Returns an empty list for interior / vehicle_info / exclude / unknown views.
    """
    normalized = _normalize_view_id(view_id)
    return list(VIEW_TO_REGIONS.get(normalized, []))


def get_parts_for_view(view_id: str) -> List[str]:
    """Return the fixed visible-part list for a standard exterior view."""
    normalized = _normalize_view_id(view_id)
    return list(VIEW_TO_PARTS.get(normalized, []))


def get_display_name(view_id: str) -> str:
    """Return the human-readable Chinese name for a view."""
    normalized = _normalize_view_id(view_id)
    return VIEW_DISPLAY_NAMES.get(normalized, view_id)


def is_exterior_view(view_id: str) -> bool:
    """Return True if the view contributes to exterior damage assessment."""
    return _normalize_view_id(view_id) in EXTERIOR_VIEWS


def is_valid_photo_category(category: str) -> bool:
    """Return True if category is a valid PlannerAgent output label."""
    return category in PHOTO_TYPE_CATEGORIES


def _normalize_view_id(view_id: str) -> str:
    """Map legacy view aliases and short names to canonical view ids.

    Supports both the new short ids (``front_left``) and the old numeric
    suffix ids (``front_left_45``, ``left_90``) so that downstream code and
    YAML configs can migrate incrementally.
    """
    if not isinstance(view_id, str):
        return str(view_id)
    aliases = {
        # Legacy numeric suffixes
        "front_left_45": "front_left",
        "front_right_45": "front_right",
        "rear_left_45": "rear_left",
        "rear_right_45": "rear_right",
        "left_90": "left",
        "right_90": "right",
        # Short aliases (already canonical)
        "front_left": "front_left",
        "front_right": "front_right",
        "rear_left": "rear_left",
        "rear_right": "rear_right",
        "left": "left",
        "right": "right",
        "front": "front",
        "rear": "rear",
        "top": "top",
    }
    return aliases.get(view_id, view_id)


def get_all_exterior_views() -> List[str]:
    """Return all exterior view identifiers in deterministic order."""
    return sorted(EXTERIOR_VIEWS)


def get_parts_for_roof_sub_region(sub_region: str) -> List[str]:
    """Return the standard part ids that belong to a roof sub-region."""
    return list(ROOF_SUB_REGION_TO_PARTS.get(sub_region, []))


def get_view_selection_prompt() -> str:
    """Return the prompt snippet that constrains view selection."""
    lines = ["标准视角选项（必须严格从中选择）："]
    for view_id in get_all_exterior_views():
        description = VIEW_COVERAGE_DESCRIPTION.get(view_id)
        if description:
            lines.append(f"  - {view_id}：{description}")
    return "\n".join(lines)


def canonicalize_view_id(view_id: str) -> str:
    """Alias for ``_normalize_view_id``; kept for backward compatibility."""
    return _normalize_view_id(view_id)


# Backward-compatible alias used by tests and legacy code.
normalize_view_id = canonicalize_view_id
