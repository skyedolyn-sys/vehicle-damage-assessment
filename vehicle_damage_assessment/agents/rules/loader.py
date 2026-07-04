"""Public loader API for the rules package.

All external code should import from ``agents.rules`` rather than reaching
into submodules.  The loader reads YAML configs from ``agents/rules/config/``,
merges vehicle-type overrides, validates against the canonical part catalog,
and caches results per process.

A missing YAML file is a hard failure (``RuleLoadError``); there is no
in-code fallback so the deployment must keep the YAML in sync with code.
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
    """Load a YAML config; raise loudly if it is missing or invalid.

    Fail-loud contract: a missing or invalid YAML is a deployment error and
    must never be silently masked by an in-code fallback.  Operators are
    expected to keep ``agents/rules/config/<name>.yaml`` in sync with code.
    """
    path = _CONFIG_DIR / f"{name}.yaml"
    raw = load_with_cache(path)
    if not raw:
        raise RuleLoadError(f"Required config file missing or empty: {path}")

    known_parts = _known_parts()
    errors = validate_config(name, raw, known_parts=known_parts, known_vehicle_types=_KNOWN_VEHICLE_TYPES)
    if errors:
        raise RuleLoadError(
            f"Config {name!r} failed validation: {'; '.join(errors)}"
        )

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
