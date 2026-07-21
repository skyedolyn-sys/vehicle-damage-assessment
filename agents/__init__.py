"""Public ``agents`` package surface.

This module re-exports the public agent entry points so external code can
use the flat ``from agents import planner_agent`` form, while still allowing
``from agents.planner_agent import X`` to access submodule internals.

Python's import system has a well-known trap: ``from .submodule import name``
where ``name`` is also the attribute being accessed via
``from package.submodule import name`` causes the submodule attribute to be
shadowed by the imported function.  We avoid that by binding both the
submodule and the function attribute to different names internally and
exposing them via a module-level ``__getattr__`` hook.
"""

from __future__ import annotations

from . import (
    auxiliary_info_extractor,
    damage_assessor,
    master_agent,
    minimax_client,
    output_validator,
    photo_locator,
    planner_agent,
    regional_worker,
    reviewer_subagent,
    assessment_orchestrator,
    synthesizer,
    topology_builder,
    topology_comparator,
    vehicle_prior,
    view_agent,
    view_mapping,
    vision_subagent,
)

# Module-level __getattr__ dispatches attribute lookups to the matching
# submodule.  Public entry points are kept in sync with __all__ below.
_PUBLIC_NAMES = {
    "extract_vehicle_info_from_auxiliary_photos":
        ("auxiliary_info_extractor", "extract_vehicle_info_from_auxiliary_photos"),
    "vehicle_prior_agent": ("vehicle_prior", "vehicle_prior_agent"),
    "extract_vehicle_specs": ("vehicle_prior", "extract_vehicle_specs"),
    "photo_locator_agent": ("photo_locator", "photo_locator_agent"),
    "damage_assessor_agent": ("damage_assessor", "damage_assessor_agent"),
    "validate_and_enrich": ("output_validator", "validate_and_enrich"),
    "call_minimax": ("minimax_client", "call_minimax"),
    "build_image_content": ("minimax_client", "build_image_content"),
    "extract_json": ("minimax_client", "extract_json"),
    "regional_damage_worker": ("regional_worker", "regional_damage_worker"),
    "synthesizer_agent": ("synthesizer", "synthesizer_agent"),
    "build_vehicle_topology": ("topology_builder", "build_vehicle_topology"),
    "topology_to_dict": ("topology_builder", "topology_to_dict"),
    "TopologyComparator": ("topology_comparator", "TopologyComparator"),
    "compare_topology": ("topology_comparator", "compare_topology"),
    "EXTERIOR_VIEWS": ("view_mapping", "EXTERIOR_VIEWS"),
    "NON_EXTERIOR_VIEWS": ("view_mapping", "NON_EXTERIOR_VIEWS"),
    "VIEW_TO_REGIONS": ("view_mapping", "VIEW_TO_REGIONS"),
    "get_all_exterior_views": ("view_mapping", "get_all_exterior_views"),
    "get_display_name": ("view_mapping", "get_display_name"),
    "get_parts_for_view": ("view_mapping", "get_parts_for_view"),
    "get_regions_for_view": ("view_mapping", "get_regions_for_view"),
    "is_exterior_view": ("view_mapping", "is_exterior_view"),
    "normalize_view_id": ("view_mapping", "normalize_view_id"),
    "planner_agent": ("planner_agent", "planner_agent"),
    "view_agent": ("view_agent", "view_agent"),
    "view_agent_result_to_part_actual_states":
        ("view_agent", "view_agent_result_to_part_actual_states"),
    "master_assessment_agent": ("master_agent", "master_assessment_agent"),
    "reviewer_subagent": ("reviewer_subagent", "reviewer_subagent"),
    "assessment_orchestrator":
        ("assessment_orchestrator", "assessment_orchestrator"),
    "assessment_orchestrator_stream":
        ("assessment_orchestrator", "assessment_orchestrator_stream"),
}


def __getattr__(name: str):
    """Lazily proxy attribute access to the matching submodule.

    This keeps both ``from agents import planner_agent`` and
    ``from agents.planner_agent import _classify_photo_types`` working
    without one shadowing the other.
    """
    if name in _PUBLIC_NAMES:
        module_name, attr = _PUBLIC_NAMES[name]
        import importlib

        module = importlib.import_module(f"agents.{module_name}")
        value = getattr(module, attr)
        # Cache the resolved attribute so subsequent lookups skip the hook.
        globals()[name] = value
        return value
    raise AttributeError(f"module 'agents' has no attribute {name!r}")


__all__ = list(_PUBLIC_NAMES.keys())
