"""Lightweight validation for YAML rule configs.

Validation is intentionally minimal: it checks structure, required keys, and
that referenced part names exist in the canonical catalog.  It does not try to
validate semantic correctness of the rules themselves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


class ConfigValidationError(Exception):
    """Raised when a config file fails validation."""


REQUIRED_TOP_KEYS = {"default"}


def _error(path: str, message: str) -> str:
    return f"{path}: {message}"


def validate_scalar_map(
    data: Dict[str, Any],
    config_name: str,
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate a config with ``default`` and optional ``by_vehicle_type`` scalar maps."""
    errors: List[str] = []

    if not isinstance(data, dict):
        errors.append(_error(config_name, "config must be a mapping"))
        return errors

    missing = REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        errors.append(_error(config_name, f"missing required keys: {sorted(missing)}"))

    default = data.get("default")
    if not isinstance(default, dict):
        errors.append(_error(f"{config_name}.default", "must be a mapping"))

    by_vehicle_type = data.get("by_vehicle_type", {})
    if not isinstance(by_vehicle_type, dict):
        errors.append(_error(f"{config_name}.by_vehicle_type", "must be a mapping"))
        return errors

    for vt, vt_config in by_vehicle_type.items():
        if known_vehicle_types and vt not in known_vehicle_types:
            errors.append(_error(f"{config_name}.by_vehicle_type.{vt}", "unknown vehicle type"))
        if not isinstance(vt_config, dict):
            errors.append(_error(f"{config_name}.by_vehicle_type.{vt}", "must be a mapping"))

    return errors


def validate_part_list_config(
    data: Dict[str, Any],
    config_name: str,
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate a config where default maps profile->part list and by_vehicle_type maps vt->profile->part list."""
    errors = validate_scalar_map(data, config_name, known_parts, known_vehicle_types)

    default = data.get("default", {})
    if isinstance(default, dict):
        errors.extend(_validate_profile_map(default, f"{config_name}.default", known_parts))

    by_vehicle_type = data.get("by_vehicle_type", {})
    if isinstance(by_vehicle_type, dict):
        for vt, vt_config in by_vehicle_type.items():
            if known_vehicle_types and vt not in known_vehicle_types:
                errors.append(_error(f"{config_name}.by_vehicle_type.{vt}", "unknown vehicle type"))
            if isinstance(vt_config, dict):
                errors.extend(_validate_profile_map(vt_config, f"{config_name}.by_vehicle_type.{vt}", known_parts))

    return errors


def _validate_profile_map(profiles: Dict[str, Any], path: str, known_parts: Optional[Set[str]]) -> List[str]:
    errors: List[str] = []
    if not isinstance(profiles, dict):
        errors.append(_error(path, "must be a mapping of profile -> part list"))
        return errors
    for profile_name, part_list in profiles.items():
        if not isinstance(part_list, list):
            errors.append(_error(f"{path}.{profile_name}", "must be a list of part ids"))
            continue
        for idx, part_id in enumerate(part_list):
            if not isinstance(part_id, str):
                errors.append(_error(f"{path}.{profile_name}[{idx}]", "part id must be a string"))
            elif known_parts and part_id not in known_parts:
                errors.append(_error(f"{path}.{profile_name}[{idx}]", f"unknown part id {part_id!r}"))
    return errors


def validate_priorities(
    data: Dict[str, Any],
    config_name: str = "priorities",
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate priorities.yaml.

    Priorities describe status/level/confidence orderings, not part ids, so
    we do not validate keys against the part catalog.
    """
    errors = validate_scalar_map(data, "priorities", known_parts=None, known_vehicle_types=known_vehicle_types)

    for section in ("default", "by_vehicle_type"):
        section_data = data.get(section, {})
        if not isinstance(section_data, dict):
            continue
        for scope, priorities in section_data.items():
            if section == "by_vehicle_type" and not isinstance(priorities, dict):
                continue
            if not isinstance(priorities, dict):
                errors.append(_error(f"priorities.{section}.{scope}", "must be a mapping of name -> integer"))
                continue
            for name, priority in priorities.items():
                if not isinstance(priority, int):
                    errors.append(_error(f"priorities.{section}.{scope}.{name}", "priority must be an integer"))

    return errors


def validate_thresholds(
    data: Dict[str, Any],
    config_name: str = "thresholds",
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate thresholds.yaml (default is a flat key -> number mapping)."""
    errors = validate_scalar_map(data, "thresholds", known_parts=None, known_vehicle_types=known_vehicle_types)

    default = data.get("default", {})
    if isinstance(default, dict):
        for key, value in default.items():
            if not isinstance(value, (int, float)):
                errors.append(_error(f"thresholds.default.{key}", "threshold must be a number"))

    by_vehicle_type = data.get("by_vehicle_type", {})
    if isinstance(by_vehicle_type, dict):
        for vt, vt_config in by_vehicle_type.items():
            if known_vehicle_types and vt not in known_vehicle_types:
                errors.append(_error(f"thresholds.by_vehicle_type.{vt}", "unknown vehicle type"))
            if isinstance(vt_config, dict):
                for key, value in vt_config.items():
                    if not isinstance(value, (int, float)):
                        errors.append(_error(f"thresholds.by_vehicle_type.{vt}.{key}", "threshold must be a number"))

    return errors


def validate_view_weights(
    data: Dict[str, Any],
    config_name: str = "view_weights",
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate view_weights.yaml.

    Expected structure:
    default:
      primary_view: {part_id: [view_id, ...]}
      view_weights: {part_id: {primary: [...], secondary: [...]}}
      roof_primary_regions: [view_id, ...]
      roof_secondary_regions: [view_id, ...]
    """
    errors = validate_scalar_map(data, "view_weights", known_parts=None, known_vehicle_types=known_vehicle_types)

    default = data.get("default", {})
    if not isinstance(default, dict):
        return errors

    primary_view = default.get("primary_view", {})
    if isinstance(primary_view, dict):
        for part_id, views in primary_view.items():
            if known_parts and part_id not in known_parts:
                errors.append(_error(f"view_weights.default.primary_view.{part_id}", f"unknown part id {part_id!r}"))
            if not isinstance(views, list):
                errors.append(_error(f"view_weights.default.primary_view.{part_id}", "must be a list of view ids"))

    view_weights = default.get("view_weights", {})
    if isinstance(view_weights, dict):
        for part_id, weights in view_weights.items():
            if known_parts and part_id not in known_parts:
                errors.append(_error(f"view_weights.default.view_weights.{part_id}", f"unknown part id {part_id!r}"))
            if not isinstance(weights, dict):
                errors.append(_error(f"view_weights.default.view_weights.{part_id}", "must be a mapping with primary/secondary"))
                continue
            for bucket, view_set in weights.items():
                if bucket not in ("primary", "secondary"):
                    errors.append(_error(f"view_weights.default.view_weights.{part_id}.{bucket}", "must be primary or secondary"))
                if not isinstance(view_set, list):
                    errors.append(_error(f"view_weights.default.view_weights.{part_id}.{bucket}", "must be a list of view ids"))

    for key in ("roof_primary_regions", "roof_secondary_regions"):
        regions = default.get(key, [])
        if not isinstance(regions, list):
            errors.append(_error(f"view_weights.default.{key}", "must be a list of view ids"))

    by_vehicle_type = data.get("by_vehicle_type", {})
    if isinstance(by_vehicle_type, dict):
        for vt, vt_config in by_vehicle_type.items():
            if known_vehicle_types and vt not in known_vehicle_types:
                errors.append(_error(f"view_weights.by_vehicle_type.{vt}", "unknown vehicle type"))
            if isinstance(vt_config, dict):
                # Same structure as default; validate recursively by building a fake doc.
                errors.extend(validate_view_weights(
                    {"default": vt_config},
                    config_name=f"view_weights.by_vehicle_type.{vt}",
                    known_parts=known_parts,
                    known_vehicle_types=known_vehicle_types,
                ))

    return errors


def validate_part_aliases(
    data: Dict[str, Any],
    config_name: str = "part_aliases",
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate part_aliases.yaml."""
    errors: List[str] = []

    if not isinstance(data, dict):
        errors.append(_error("part_aliases", "config must be a mapping"))
        return errors

    aliases = data.get("aliases", {})
    if not isinstance(aliases, dict):
        errors.append(_error("part_aliases.aliases", "must be a mapping"))
        return errors

    for canonical, entry in aliases.items():
        if known_parts is not None and canonical not in known_parts:
            errors.append(_error(f"part_aliases.aliases.{canonical}", f"unknown canonical part id {canonical!r}"))
        if not isinstance(entry, dict):
            errors.append(_error(f"part_aliases.aliases.{canonical}", "must be a mapping with 'synonyms'"))
            continue
        synonyms = entry.get("synonyms", [])
        if not isinstance(synonyms, list):
            errors.append(_error(f"part_aliases.aliases.{canonical}.synonyms", "must be a list of strings"))
            continue
        for idx, synonym in enumerate(synonyms):
            if not isinstance(synonym, str):
                errors.append(_error(f"part_aliases.aliases.{canonical}.synonyms[{idx}]", "synonym must be a string"))

    return errors


def validate_region_units(
    data: Dict[str, Any],
    config_name: str = "region_units",
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate region_units.yaml (default is unit_name -> list of part ids)."""
    errors = validate_scalar_map(data, "region_units", known_parts=None, known_vehicle_types=known_vehicle_types)

    default = data.get("default", {})
    if isinstance(default, dict):
        for unit_name, part_list in default.items():
            if not isinstance(part_list, list):
                errors.append(_error(f"region_units.default.{unit_name}", "must be a list of part ids"))
                continue
            for idx, part_id in enumerate(part_list):
                if not isinstance(part_id, str):
                    errors.append(_error(f"region_units.default.{unit_name}[{idx}]", "part id must be a string"))
                elif known_parts is not None and part_id not in known_parts:
                    errors.append(_error(f"region_units.default.{unit_name}[{idx}]", f"unknown part id {part_id!r}"))

    by_vehicle_type = data.get("by_vehicle_type", {})
    if isinstance(by_vehicle_type, dict):
        for vt, vt_config in by_vehicle_type.items():
            if known_vehicle_types and vt not in known_vehicle_types:
                errors.append(_error(f"region_units.by_vehicle_type.{vt}", "unknown vehicle type"))
            if not isinstance(vt_config, dict):
                continue
            for unit_name, part_list in vt_config.items():
                if not isinstance(part_list, list):
                    errors.append(_error(f"region_units.by_vehicle_type.{vt}.{unit_name}", "must be a list of part ids"))
                    continue
                for idx, part_id in enumerate(part_list):
                    if not isinstance(part_id, str):
                        errors.append(_error(f"region_units.by_vehicle_type.{vt}.{unit_name}[{idx}]", "part id must be a string"))
                    elif known_parts is not None and part_id not in known_parts:
                        errors.append(_error(f"region_units.by_vehicle_type.{vt}.{unit_name}[{idx}]", f"unknown part id {part_id!r}"))

    return errors


def validate_filename_view_hints(
    data: Dict[str, Any],
    config_name: str = "filename_view_hints",
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate filename_view_hints.yaml."""
    errors: List[str] = []

    if not isinstance(data, dict):
        errors.append(_error("filename_view_hints", "config must be a mapping"))
        return errors

    rules = data.get("rules", [])
    if not isinstance(rules, list):
        errors.append(_error("filename_view_hints.rules", "must be a list"))
        return errors

    for idx, rule in enumerate(rules):
        prefix = f"filename_view_hints.rules[{idx}]"
        if not isinstance(rule, dict):
            errors.append(_error(prefix, "rule must be a mapping"))
            continue
        for required in ("pattern", "view_id", "priority"):
            if required not in rule:
                errors.append(_error(prefix, f"missing required field {required!r}"))
        if "priority" in rule and not isinstance(rule["priority"], int):
            errors.append(_error(f"{prefix}.priority", "must be an integer"))

    return errors


def validate_filename_heuristics(
    data: Dict[str, Any],
    config_name: str = "filename_heuristics",
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate filename_heuristics.yaml."""
    errors: List[str] = []

    if not isinstance(data, dict):
        errors.append(_error("filename_heuristics", "config must be a mapping"))
        return errors

    rules = data.get("rules", [])
    if not isinstance(rules, list):
        errors.append(_error("filename_heuristics.rules", "must be a list"))
        return errors

    for idx, rule in enumerate(rules):
        prefix = f"filename_heuristics.rules[{idx}]"
        if not isinstance(rule, dict):
            errors.append(_error(prefix, "rule must be a mapping"))
            continue
        for required in ("name", "patterns", "priority"):
            if required not in rule:
                errors.append(_error(prefix, f"missing required field {required!r}"))
        if "patterns" in rule and not isinstance(rule["patterns"], list):
            errors.append(_error(f"{prefix}.patterns", "must be a list of strings"))
        if "priority" in rule and not isinstance(rule["priority"], int):
            errors.append(_error(f"{prefix}.priority", "must be an integer"))

    return errors


def validate_config(
    config_name: str,
    data: Dict[str, Any],
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate a loaded config document and return a list of error messages.

    Parameters
    ----------
    config_name:
        Logical name of the config (e.g. ``"priorities"``).
    data:
        Parsed YAML data.
    known_parts:
        Optional set of valid canonical part ids.
    known_vehicle_types:
        Optional set of valid vehicle type slugs.

    Returns
    -------
    list[str]
        Empty list if valid; otherwise human-readable error messages.
    """
    validators = {
        "priorities": validate_priorities,
        "part_profiles": validate_part_list_config,
        "view_weights": validate_view_weights,
        "thresholds": validate_thresholds,
        "filename_heuristics": validate_filename_heuristics,
        "part_aliases": validate_part_aliases,
        "trigger_sets": validate_part_list_config,
        "checklist_hints": _validate_checklist_hints,
        "region_units": validate_region_units,
        "filename_view_hints": validate_filename_view_hints,
    }

    validator = validators.get(config_name)
    if validator is None:
        return [_error(config_name, f"no validator registered for {config_name!r}")]

    return validator(data, config_name=config_name, known_parts=known_parts, known_vehicle_types=known_vehicle_types)


def _validate_checklist_hints(
    data: Dict[str, Any],
    config_name: str = "checklist_hints",
    known_parts: Optional[Set[str]] = None,
    known_vehicle_types: Optional[Set[str]] = None,
) -> List[str]:
    """Validate checklist_hints.yaml."""
    errors: List[str] = []

    if not isinstance(data, dict):
        errors.append(_error("checklist_hints", "config must be a mapping"))
        return errors

    default = data.get("default", [])
    if not isinstance(default, list):
        errors.append(_error("checklist_hints.default", "must be a list"))
        return errors

    for idx, hint in enumerate(default):
        errors.extend(_validate_hint(hint, f"checklist_hints.default[{idx}]", known_parts))

    by_vehicle_type = data.get("by_vehicle_type", {})
    if not isinstance(by_vehicle_type, dict):
        errors.append(_error("checklist_hints.by_vehicle_type", "must be a mapping"))
        return errors

    for vt, hints in by_vehicle_type.items():
        if known_vehicle_types and vt not in known_vehicle_types:
            errors.append(_error(f"checklist_hints.by_vehicle_type.{vt}", "unknown vehicle type"))
        if not isinstance(hints, list):
            errors.append(_error(f"checklist_hints.by_vehicle_type.{vt}", "must be a list"))
            continue
        for idx, hint in enumerate(hints):
            errors.extend(_validate_hint(hint, f"checklist_hints.by_vehicle_type.{vt}[{idx}]", known_parts))

    return errors


def _validate_hint(hint: Any, path: str, known_parts: Optional[Set[str]]) -> List[str]:
    errors: List[str] = []
    if not isinstance(hint, dict):
        errors.append(_error(path, "hint must be a mapping"))
        return errors
    for required in ("condition", "hint"):
        if required not in hint:
            errors.append(_error(path, f"missing required field {required!r}"))
    if "severity" in hint and hint["severity"] not in ("info", "warning", "critical"):
        errors.append(_error(f"{path}.severity", "must be one of info/warning/critical"))
    return errors
