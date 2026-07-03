"""Public loader API for the rules package.

All external code should import from ``agents.rules`` rather than reaching
into submodules.  The loader reads YAML configs from ``agents/rules/config/``,
merges vehicle-type overrides, validates against the canonical part catalog,
and caches results per process.

If a config file is missing or a key is absent, the loader falls back to a
built-in legacy constant copy.  This preserves backward compatibility while
allowing incremental migration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.rules.engine.cache import load_with_cache
from agents.rules.engine.merge import merge_vehicle_type_config
from agents.rules.engine.validate import ConfigValidationError, validate_config

logger = logging.getLogger(__name__)


class RuleLoadError(Exception):
    """Raised when a rule config file cannot be loaded or validated."""


#: Directory containing YAML config files.
_CONFIG_DIR = Path(__file__).resolve().parent / "config"

#: Canonical vehicle type slugs known to the system.
_KNOWN_VEHICLE_TYPES = {"sedan", "suv", "truck", "motorcycle", "van", "ev"}

#: Lazy Jinja2 environment for rendering prompt templates.
_jinja_env = None


# ---------------------------------------------------------------------------
# Legacy fallback constants (copied from existing agent modules)
# ---------------------------------------------------------------------------

_LEGACY_STATUS_PRIORITY = {"damaged": 3, "uncertain": 2, "intact": 1, "missing": 4}
_LEGACY_LEVEL_PRIORITY = {"severe": 4, "moderate": 3, "light": 2, "none": 1, "unknown": 0}
_LEGACY_CONFIDENCE_PRIORITY = {"low": 0, "medium": 1, "high": 2}
_LEGACY_UNCERTAIN_STATUS_PRIORITY = {"missing": 4, "damaged": 3, "intact": 1, "uncertain": 0}

_LEGACY_PART_PROFILES: Dict[str, Set[str]] = {
    "conservative": {
        "door_front_left", "door_rear_left",
        "door_front_right", "door_rear_right",
        "mirror_left", "mirror_right",
        "roof_middle", "roof_rear", "sunroof_glass",
    },
    "roof": {"roof_front", "roof_middle", "roof_rear", "sunroof_glass", "roof_rack"},
    "front_false_damage": {
        "bumper_front",
        "door_front_left", "door_front_right",
        "fender_front_left", "fender_front_right",
        "headlight_front_left", "headlight_front_right",
    },
    "spill_over_prone": {
        "door_front_left", "door_front_right",
        "door_rear_left", "door_rear_right",
        "fender_rear_right", "taillight_rear_right",
    },
    "rear_core": {
        "trunk_lid", "tailgate", "bumper_rear",
        "taillight_rear_left", "taillight_rear_right", "windshield_rear",
    },
    "rear_core_structural": {"trunk_lid", "tailgate", "bumper_rear"},
}

_LEGACY_VIEW_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "primary_view": {
        "door_front_left": ["left_90"],
        "door_rear_left": ["left_90"],
        "door_front_right": ["right_90"],
        "door_rear_right": ["right_90"],
        "mirror_left": ["left_90"],
        "mirror_right": ["right_90"],
        "roof_front": ["top"],
        "roof_middle": ["top"],
        "roof_rear": ["top"],
        "sunroof_glass": ["top"],
        "pillar_a_left": ["front_left_45"],
        "pillar_a_right": ["front_right_45"],
        "pillar_b_left": ["left_90"],
        "pillar_b_right": ["right_90"],
        "pillar_c_left": ["rear_left_45"],
        "pillar_c_right": ["rear_right_45"],
    },
    "view_weights": {
        "door_front_left": {"primary": {"left_90"}, "secondary": {"front_left_45", "rear_left_45"}},
        "door_rear_left": {"primary": {"left_90"}, "secondary": {"front_left_45", "rear_left_45"}},
        "door_front_right": {"primary": {"right_90"}, "secondary": {"front_right_45", "rear_right_45"}},
        "door_rear_right": {"primary": {"right_90"}, "secondary": {"front_right_45", "rear_right_45"}},
        "mirror_left": {"primary": {"left_90"}, "secondary": {"front_left_45", "rear_left_45"}},
        "mirror_right": {"primary": {"right_90"}, "secondary": {"front_right_45", "rear_right_45"}},
        "roof_front": {"primary": {"top"}, "secondary": {"front", "front_left_45", "front_right_45"}},
        "roof_middle": {"primary": {"top"}, "secondary": {"left_90", "right_90"}},
        "roof_rear": {"primary": {"top"}, "secondary": {"rear", "rear_left_45", "rear_right_45"}},
        "sunroof_glass": {"primary": {"top"}, "secondary": {"front", "front_left_45", "front_right_45", "rear", "rear_left_45", "rear_right_45", "left_90", "right_90"}},
        "pillar_a_left": {"primary": {"front_left_45"}, "secondary": {"left_90", "top"}},
        "pillar_a_right": {"primary": {"front_right_45"}, "secondary": {"right_90", "top"}},
        "pillar_b_left": {"primary": {"left_90"}, "secondary": {"front_left_45", "rear_left_45", "top"}},
        "pillar_b_right": {"primary": {"right_90"}, "secondary": {"front_right_45", "rear_right_45", "top"}},
        "pillar_c_left": {"primary": {"rear_left_45"}, "secondary": {"left_90", "top"}},
        "pillar_c_right": {"primary": {"rear_right_45"}, "secondary": {"right_90", "top"}},
    },
    "roof_primary_regions": {"top"},
    "roof_secondary_regions": {
        "left_90", "right_90", "front_left_45", "front_right_45", "rear", "rear_left_45", "rear_right_45"
    },
}

_LEGACY_THRESHOLDS: Dict[str, float] = {
    "visibility_definite_ratio": 0.30,
    "visibility_invisible_ratio": 0.10,
    "damage_light_max_diameter": 0.05,
    "damage_moderate_max_diameter": 0.20,
    "wide_shot_body_ratio": 0.60,
    "close_up_area_cm": 30,
    "planner_llm_classification_cap": 12,
    "fill_fallback_min_exterior_views": 6,
    "fill_fallback_remaining_ratio": 0.25,
    "fill_fallback_min_remaining": 2,
    "synthesizer_agreement_ratio": 0.75,
}

_LEGACY_FILENAME_HEURISTICS = [
    {"name": "auxiliary_license", "patterns": [
        "行驶证", "证件", "vin", "铭牌", "license", "plate", "证", "车牌",
        "车架号", "登记证书", "保单", "发票",
    ], "priority": 100},
    {"name": "interior", "patterns": [
        "车内", "内饰", "驾驶舱", "座椅", "方向盘", "仪表盘", "中控", "后排",
    ], "priority": 90},
    {"name": "auxiliary_01", "patterns": ["-01."], "priority": 80},
    {"name": "auxiliary_09", "patterns": ["-09."], "priority": 80},
    {"name": "interior_07", "patterns": ["-07."], "priority": 80},
    {"name": "interior_08", "patterns": ["-08."], "priority": 80},
    {"name": "interior_11", "patterns": ["-11."], "priority": 80},
]

_LEGACY_PART_ALIASES: Dict[str, List[str]] = {
    "bumper_front": ["front_bumper", "front_bumber"],
    "hood": ["front_hood", "engine_hood"],
    "grille_front": ["front_grille", "grille"],
    "headlight_front_left": ["left_front_headlight", "left_headlight", "headlight_left", "front_left_headlight"],
    "headlight_front_right": ["right_front_headlight", "right_headlight", "headlight_right", "front_right_headlight"],
    "fender_front_left": ["left_front_fender", "left_fender_front", "left_fender"],
    "fender_front_right": ["right_front_fender", "right_fender_front", "right_fender"],
    "windshield_front": ["front_windshield"],
    "pillar_a_left": ["left_a_pillar", "a_pillar_left"],
    "pillar_a_right": ["right_a_pillar", "a_pillar_right"],
    "pillar_b_left": ["left_b_pillar", "b_pillar_left"],
    "pillar_b_right": ["right_b_pillar", "b_pillar_right"],
    "pillar_c_left": ["left_c_pillar", "c_pillar_left"],
    "pillar_c_right": ["right_c_pillar", "c_pillar_right"],
    "bumper_rear": ["rear_bumper"],
    "trunk_lid": ["trunk"],
    "taillight_rear_left": [
        "left_rear_taillight", "left_rear_tail_light", "left_taillight", "left_tail_light",
        "taillight_left", "tail_light_left", "left_rear_light", "left_tail_light_rear",
    ],
    "taillight_rear_right": [
        "right_rear_taillight", "right_rear_tail_light", "right_taillight", "right_tail_light",
        "taillight_right", "tail_light_right", "right_rear_light", "right_tail_light_rear",
    ],
    "windshield_rear": ["rear_windshield", "back_windshield"],
    "door_front_left": ["left_front_door"],
    "door_rear_left": ["left_rear_door", "left_back_door"],
    "door_front_right": ["right_front_door"],
    "door_rear_right": ["right_rear_door", "right_back_door"],
    "mirror_left": ["left_side_mirror", "left_mirror", "left_rearview_mirror"],
    "mirror_right": ["right_side_mirror", "right_mirror", "right_rearview_mirror"],
    "fender_rear_left": ["left_rear_fender", "left_rear_quarter_panel", "left_quarter_panel", "left_quarter"],
    "fender_rear_right": ["right_rear_fender", "right_rear_quarter_panel", "right_quarter_panel", "right_quarter"],
}

_LEGACY_CHECKLIST_HINTS: List[Dict[str, str]] = [
    {"match": "roof_", "hint": "车顶/天窗在本视角中通常只能看到边缘轮廓；若仅见边缘且无凹陷/变形/裂纹/玻璃碎裂等明显异常，应标 intact；只有当主体面板明确可见变形、裂纹、碎裂或明显缺失时才标 damaged"},
    {"match": "sunroof_glass", "hint": "车顶/天窗在本视角中通常只能看到边缘轮廓；若仅见边缘且无凹陷/变形/裂纹/玻璃碎裂等明显异常，应标 intact；只有当主体面板明确可见变形、裂纹、碎裂或明显缺失时才标 damaged"},
    {"match": "roof_rack", "hint": "车顶/天窗在本视角中通常只能看到边缘轮廓；若仅见边缘且无凹陷/变形/裂纹/玻璃碎裂等明显异常，应标 intact；只有当主体面板明确可见变形、裂纹、碎裂或明显缺失时才标 damaged"},
    {"match": "headlight_", "hint": "灯具本体可见即可评估；无裂纹/破损/进水痕迹则标 intact；远端仅见灯壳边缘时可标 uncertain"},
    {"match": "taillight_", "hint": "灯具本体可见即可评估；若灯壳完整、无裂纹/破损/进水痕迹则标 intact；若尾灯区域被严重遮挡、仅残留边缘、灯壳不可辨识或周围钣金严重撕裂变形，应优先标 damaged severe 或 missing，不要仅因'未见裂纹'就判 intact；远端仅见灯壳边缘且无明显异常时可标 uncertain"},
    {"match": "mirror_", "hint": "后视镜：只要镜壳外侧任何部分可见，且该可见部分无裂纹/破损/变形/脱落，即标 intact（confidence=low），并在 notes 中说明可见程度；只有当后视镜完全不可见，或可见部分存在明确损伤时，才标 damaged/uncertain"},
    {"match": "door_", "hint": "严格区分门板主体与相邻翼子板/C柱/轮拱：仅评估车门面板本身（含门把手、腰线、窗下沿以下板面）。若凹陷/变形实际位于翼子板、C柱或后翼子板轮拱区域，即使靠近车门边缘，也不要判为车门损伤；车门板主体无异常则标 intact；门板主体明确可见凹陷/划痕/变形/漆面脱落才标 damaged；仅看到车门边缘/窗框/门缝时请标 uncertain"},
    {"match": "pillar_", "hint": "立柱属于结构性安全件：只要画面中能看到该立柱且存在变形、断裂、褶皱、撕裂、钣金错位等任何结构异常，即标 damaged severe；立柱轻微刮擦可标 light；完全看不到该立柱时才标 uncertain"},
    {"match": "fender_", "hint": "主体面板可见即可评估；无凹陷/划痕/变形/漆面脱落则标 intact；有损伤时按凹陷/划痕/变形面积选择 damage_level（light/moderate/severe）"},
    {"match": "bumper_", "hint": "主体面板可见即可评估；无凹陷/划痕/变形/漆面脱落则标 intact；有损伤时按凹陷/划痕/变形面积选择 damage_level（light/moderate/severe）"},
    {"match": "windshield", "hint": "玻璃区域可见即可评估；无裂纹/碎裂则标 intact"},
    {"match": "hood", "hint": "主体可见即可评估；前保险杠若被白条、车牌、强光反光遮挡，遮挡区域不要判为损伤"},
    {"match": "grille_front", "hint": "主体可见即可评估；前保险杠若被白条、车牌、强光反光遮挡，遮挡区域不要判为损伤"},
]

_LEGACY_FILENAME_VIEW_HINTS: List[Tuple[str, str]] = [
    ("行驶证", "auxiliary"),
    ("vin", "auxiliary"),
    ("铭牌", "auxiliary"),
    ("证件", "auxiliary"),
    ("车牌", "auxiliary"),
    ("内饰", "interior"),
    ("车内", "interior"),
    ("座椅", "interior"),
    ("-01.", "auxiliary"),
    ("-09.", "auxiliary"),
    ("-07.", "interior"),
    ("-08.", "interior"),
    ("-11.", "interior"),
]

_LEGACY_REGION_UNITS: Dict[str, Set[str]] = {
    "rear_unit": {"tailgate", "windshield_rear"},
}

_LEGACY_TRIGGER_SETS: Dict[str, Set[str]] = {
    "front_false_damage_parts": {
        "bumper_front", "door_front_left", "door_front_right",
        "fender_front_left", "fender_front_right",
        "headlight_front_left", "headlight_front_right",
    },
    "spill_over_prone_parts": {
        "door_front_left", "door_front_right", "door_rear_left", "door_rear_right",
        "fender_rear_right", "taillight_rear_right",
    },
    "rear_core_parts": {
        "trunk_lid", "tailgate", "bumper_rear",
        "taillight_rear_left", "taillight_rear_right", "windshield_rear",
    },
    "rear_core_structural_parts": {"trunk_lid", "tailgate", "bumper_rear"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _known_parts() -> Set[str]:
    """Return canonical part ids from the config module, avoiding early import cycles."""
    try:
        from config import PARTS_BY_ID

        return set(PARTS_BY_ID.keys())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not load PARTS_BY_ID for validation: %s", exc)
        return set()


def _load_config(name: str, vehicle_type: Optional[str] = None) -> Dict[str, Any]:
    """Load and merge a YAML config, falling back to legacy constants if needed."""
    path = _CONFIG_DIR / f"{name}.yaml"
    raw = load_with_cache(path)
    fallback = not raw

    if fallback:
        logger.info("[rules] Falling back to legacy constants for %s", name)
        raw = _legacy_config(name)

    known_parts = _known_parts()
    errors = validate_config(name, raw, known_parts=known_parts, known_vehicle_types=_KNOWN_VEHICLE_TYPES)
    if errors:
        if fallback:
            # Even legacy constants can drift from the catalog; raise loudly.
            raise RuleLoadError(f"Legacy constants for {name!r} failed validation: {errors}")
        logger.info("[rules] YAML config %s invalid (%s); falling back to legacy constants", name, "; ".join(errors))
        raw = _legacy_config(name)
        errors = validate_config(name, raw, known_parts=known_parts, known_vehicle_types=_KNOWN_VEHICLE_TYPES)
        if errors:
            raise RuleLoadError(f"Legacy constants for {name!r} failed validation: {errors}")

    if "default" in raw:
        merged = merge_vehicle_type_config(raw, vehicle_type or "")
    else:
        merged = raw

    if name == "view_weights":
        merged = _normalize_view_weights_sets(merged)

    return merged


def _normalize_view_weights_sets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure view weight primary/secondary buckets are sets for downstream code."""
    config = dict(config)
    view_weights = config.get("view_weights", {})
    if isinstance(view_weights, dict):
        normalized: Dict[str, Any] = {}
        for part_id, weights in view_weights.items():
            if not isinstance(weights, dict):
                normalized[part_id] = weights
                continue
            normalized[part_id] = {
                "primary": set(weights.get("primary", [])),
                "secondary": set(weights.get("secondary", [])),
            }
        config["view_weights"] = normalized
    return config


def _legacy_config(name: str) -> Dict[str, Any]:
    """Return a config document shaped like the YAML files from legacy constants."""
    if name == "priorities":
        return {
            "default": {
                "status": _LEGACY_STATUS_PRIORITY,
                "uncertain_status": _LEGACY_UNCERTAIN_STATUS_PRIORITY,
                "level": _LEGACY_LEVEL_PRIORITY,
                "confidence": _LEGACY_CONFIDENCE_PRIORITY,
            }
        }
    if name == "part_profiles":
        return {"default": {k: sorted(v) for k, v in _LEGACY_PART_PROFILES.items()}}
    if name == "view_weights":
        return {
            "default": {
                "primary_view": _LEGACY_VIEW_WEIGHTS["primary_view"],
                "view_weights": {
                    k: {"primary": set(v["primary"]), "secondary": set(v["secondary"])}
                    for k, v in _LEGACY_VIEW_WEIGHTS["view_weights"].items()
                },
                "roof_primary_regions": sorted(_LEGACY_VIEW_WEIGHTS["roof_primary_regions"]),
                "roof_secondary_regions": sorted(_LEGACY_VIEW_WEIGHTS["roof_secondary_regions"]),
            }
        }
    if name == "thresholds":
        return {"default": _LEGACY_THRESHOLDS}
    if name == "filename_heuristics":
        return {"rules": _LEGACY_FILENAME_HEURISTICS}
    if name == "part_aliases":
        return {"aliases": {k: {"canonical": k, "synonyms": v} for k, v in _LEGACY_PART_ALIASES.items()}}
    if name == "checklist_hints":
        return {"default": [{"condition": h["match"], "hint": h["hint"]} for h in _LEGACY_CHECKLIST_HINTS]}
    if name == "trigger_sets":
        return {"default": {k: sorted(v) for k, v in _LEGACY_TRIGGER_SETS.items()}}
    if name == "filename_view_hints":
        return {"rules": [{"pattern": p, "view_id": v, "priority": 0} for p, v in _LEGACY_FILENAME_VIEW_HINTS]}
    if name == "region_units":
        return {"default": {k: sorted(v) for k, v in _LEGACY_REGION_UNITS.items()}}
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_priority_map(vehicle_type: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """Load priority rankings for status, level, confidence, and uncertain status."""
    return _load_config("priorities", vehicle_type)


def load_part_profile(profile_name: str, vehicle_type: Optional[str] = None) -> Set[str]:
    """Return the set of part ids belonging to a named profile."""
    config = _load_config("part_profiles", vehicle_type)
    return set(config.get(profile_name, []))


def load_view_weights(vehicle_type: Optional[str] = None) -> Dict[str, Any]:
    """Load view authority configuration (primary_view, view_weights, roof regions)."""
    return _load_config("view_weights", vehicle_type)


def load_part_view_priority(vehicle_type: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """Return per-part view priority tables from the rules config.

    Priority 0 = most authoritative canonical view for this part; higher
    values are less authoritative; 99 = effectively invisible from that
    angle.  Used by evidence_fusion to decide whether a primary-view
    intact signal dominates secondary-view damaged signals.

    Sourced from ``view_weights.yaml#part_view_priority`` (default block).
    """
    config = _load_config("view_weights", vehicle_type)
    return {
        part_id: dict(priorities)
        for part_id, priorities in config.get("part_view_priority", {}).items()
    }


def load_damage_type_allowlist(vehicle_type: Optional[str] = None) -> Dict[str, Any]:
    """Return the central damage_type allow-list.

    Returns a dict with three keys:
      - allowed: list of canonical damage_type strings
      - default: fallback string when input cannot be normalised
      - aliases: dict mapping raw LLM-emitted variants to canonical strings
    """
    config = _load_config("damage_types", vehicle_type)
    return {
        "allowed": list(config.get("allowed", [])),
        "default": str(config.get("default", "none")),
        "aliases": dict(config.get("aliases", {})),
    }


def load_threshold(key: str, vehicle_type: Optional[str] = None) -> float:
    """Return a single numeric threshold by key."""
    config = _load_config("thresholds", vehicle_type)
    if key not in config:
        raise RuleLoadError(f"Unknown threshold {key!r}")
    return float(config[key])


def load_all_thresholds(vehicle_type: Optional[str] = None) -> Dict[str, float]:
    """Return all thresholds as a dict."""
    return _load_config("thresholds", vehicle_type)


def load_filename_heuristics() -> List[Dict[str, Any]]:
    """Return filename heuristic rules for planner fallback."""
    config = _load_config("filename_heuristics")
    return list(config.get("rules", []))


def resolve_part_alias(raw_name: str) -> str:
    """Map a raw part name/alias to its canonical part id.

    Matching is case-insensitive and strips whitespace/underscores.  Unknown
    names pass through unchanged.
    """
    config = _load_config("part_aliases")
    aliases = config.get("aliases", {})

    index: Dict[str, str] = {}
    for canonical, entry in aliases.items():
        index[_normalize_alias_key(canonical)] = canonical
        for synonym in entry.get("synonyms", []):
            index[_normalize_alias_key(synonym)] = canonical

    normalized = _normalize_alias_key(raw_name)
    return index.get(normalized, raw_name)


def _normalize_alias_key(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def load_trigger_set(set_name: str, vehicle_type: Optional[str] = None) -> Set[str]:
    """Return a named trigger set used by adjacency/spill-over logic."""
    config = _load_config("trigger_sets", vehicle_type)
    return set(config.get(set_name, []))


def load_region_units(vehicle_type: Optional[str] = None) -> Dict[str, Set[str]]:
    """Return region unit definitions: groups of physically connected parts."""
    config = _load_config("region_units", vehicle_type)
    return {unit: set(parts) for unit, parts in config.items()}


def load_filename_view_hints() -> List[Tuple[str, str]]:
    """Return ordered (pattern, view_id) tuples for planner filename fallback.

    Rules are sorted by descending priority; original list order is used as a
    tiebreaker to preserve deterministic behavior.
    """
    config = _load_config("filename_view_hints")
    rules = list(config.get("rules", []))
    sorted_rules = sorted(
        enumerate(rules),
        key=lambda item: (-int(item[1].get("priority", 0)), item[0]),
    )
    return [(str(r["pattern"]), str(r["view_id"])) for _, r in sorted_rules]


def get_prompt_template(name: str) -> Any:
    """Return a Jinja2 template from ``agents/rules/templates/`` by base name."""
    env = _get_jinja_env()
    return env.get_template(f"{name}.j2")


def render_prompt_template(name: str, **kwargs: Any) -> str:
    """Render a named Jinja2 template with the supplied variables."""
    return get_prompt_template(name).render(**kwargs)


def _get_jinja_env() -> Any:
    """Lazy-initialize the Jinja2 environment used by prompt templates."""
    global _jinja_env
    if _jinja_env is None:
        from jinja2 import Environment, FileSystemLoader

        template_dir = Path(__file__).resolve().parent / "templates"
        _jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            variable_start_string="[[",
            variable_end_string="]]",
        )
    return _jinja_env


def get_checklist_hints(
    damaged_parts: Optional[Set[str]] = None,
    vehicle_type: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return checklist hints whose conditions match the given damaged parts.

    The minimal DSL supports:
    - "part_id": true if part_id is in damaged_parts
    - "any_of:[p1,p2,...]": true if any listed part is damaged
    - "all_of:[p1,p2,...]": true if all listed parts are damaged
    """
    import re

    config = _load_config("checklist_hints", vehicle_type)
    hints = config if isinstance(config, list) else config.get("default", [])
    damaged_parts = damaged_parts or set()
    result: List[Dict[str, str]] = []

    for hint in hints:
        condition = str(hint.get("condition", ""))
        if _eval_condition(condition, damaged_parts):
            result.append({
                "condition": condition,
                "hint": str(hint.get("hint", "")),
                "severity": str(hint.get("severity", "info")),
            })

    return result


def _eval_condition(condition: str, damaged_parts: Set[str]) -> bool:
    import re

    any_match = re.match(r"any_of:\[(.*)\]", condition)
    if any_match:
        parts = {p.strip() for p in any_match.group(1).split(",") if p.strip()}
        return bool(parts & damaged_parts)

    all_match = re.match(r"all_of:\[(.*)\]", condition)
    if all_match:
        parts = {p.strip() for p in all_match.group(1).split(",") if p.strip()}
        return parts.issubset(damaged_parts)

    return condition in damaged_parts
