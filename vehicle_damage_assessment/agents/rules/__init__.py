"""Rules package — centralized configuration for vehicle damage assessment.

This package exposes a single public API surface for loading rules, thresholds,
view weights, part aliases, and trigger sets from YAML configuration files.

Example:
    from agents.rules import load_threshold, load_part_profile

    min_conf = load_threshold("min_confidence_for_damage")
    roof_parts = load_part_profile("roof_parts")
"""

from agents.rules.loader import (
    RuleLoadError,
    get_checklist_hints,
    get_prompt_template,
    load_all_thresholds,
    load_filename_heuristics,
    load_filename_view_hints,
    load_part_profile,
    load_priority_map,
    load_region_units,
    load_threshold,
    load_trigger_set,
    load_view_weights,
    render_prompt_template,
    resolve_part_alias,
)

__all__ = [
    "RuleLoadError",
    "get_checklist_hints",
    "get_prompt_template",
    "load_all_thresholds",
    "load_filename_heuristics",
    "load_filename_view_hints",
    "load_part_profile",
    "load_priority_map",
    "load_region_units",
    "load_threshold",
    "load_trigger_set",
    "load_view_weights",
    "render_prompt_template",
    "resolve_part_alias",
]
