"""Deep-merge utilities for combining default config with vehicle-type overrides."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Union


def deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
    list_mode: str = "replace",
) -> Dict[str, Any]:
    """Return a new dict that merges ``override`` into ``base``.

    Scalar dicts are merged shallowly (override key replaces base key).
    Lists are handled according to ``list_mode``:

    - ``"replace"`` (default): the override list replaces the base list.
    - ``"additive"``: the override list is appended to the base list.

    Nested dicts are recursively merged.

    Parameters
    ----------
    base:
        Default mapping.
    override:
        Mapping applied on top of ``base``.
    list_mode:
        Either ``"replace"`` or ``"additive"``.

    Returns
    -------
    dict
        A new deep-copied dict representing the merged result.
    """
    if list_mode not in ("replace", "additive"):
        raise ValueError(f"list_mode must be 'replace' or 'additive', got {list_mode!r}")

    merged = deepcopy(base)
    for key, value in override.items():
        if key not in merged:
            merged[key] = deepcopy(value)
            continue

        base_value = merged[key]
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge(base_value, value, list_mode=list_mode)
        elif isinstance(base_value, list) and isinstance(value, list):
            if list_mode == "replace":
                merged[key] = deepcopy(value)
            else:
                merged[key] = deepcopy(base_value) + deepcopy(value)
        else:
            merged[key] = deepcopy(value)

    return merged


def merge_vehicle_type_config(
    config: Dict[str, Any],
    vehicle_type: str,
) -> Dict[str, Any]:
    """Merge ``by_vehicle_type[vehicle_type]`` overrides onto ``default``.

    Parameters
    ----------
    config:
        Config document containing ``default`` and optional ``by_vehicle_type``.
    vehicle_type:
        Vehicle type slug to apply.

    Returns
    -------
    dict
        Merged config dict. If ``by_vehicle_type`` has no entry for the slug,
        returns a copy of ``default``.
    """
    default = config.get("default", {})
    overrides = config.get("by_vehicle_type", {}).get(vehicle_type, {})
    if not overrides:
        return deepcopy(default)
    return deep_merge(default, overrides, list_mode="replace")
