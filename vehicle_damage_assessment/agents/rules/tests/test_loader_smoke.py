"""Smoke tests for the rules loader."""

import pytest

from agents.rules import (
    get_checklist_hints,
    load_all_thresholds,
    load_filename_heuristics,
    load_filename_view_hints,
    load_part_profile,
    load_priority_map,
    load_region_units,
    load_threshold,
    load_trigger_set,
    load_view_weights,
    resolve_part_alias,
)
from agents.rules.loader import RuleLoadError


def test_load_priority_map_returns_all_priority_types():
    priorities = load_priority_map()
    assert set(priorities.keys()) == {"status", "uncertain_status", "level", "confidence"}
    assert priorities["level"]["severe"] > priorities["level"]["moderate"]


def test_load_part_profile_conservative():
    conservative = load_part_profile("conservative")
    assert "door_front_left" in conservative
    assert "sunroof_glass" in conservative


def test_load_view_weights_has_primary_and_secondary():
    weights = load_view_weights()
    assert "primary_view" in weights
    assert "view_weights" in weights
    assert "roof_primary_regions" in weights
    assert set(weights["roof_primary_regions"]) == {"top"}


def test_load_threshold_known_key():
    assert load_threshold("visibility_definite_ratio") == pytest.approx(0.30)


def test_load_threshold_unknown_key_raises():
    with pytest.raises(RuleLoadError):
        load_threshold("not_a_real_threshold")


def test_load_all_thresholds():
    thresholds = load_all_thresholds()
    assert "visibility_definite_ratio" in thresholds
    assert "planner_llm_classification_cap" in thresholds


def test_load_filename_heuristics():
    heuristics = load_filename_heuristics()
    assert len(heuristics) >= 7
    names = {h["name"] for h in heuristics}
    assert "auxiliary_license" in names
    assert "interior" in names


def test_resolve_part_alias():
    assert resolve_part_alias("left_headlight") == "headlight_front_left"
    assert resolve_part_alias("LEFT_HEADLIGHT") == "headlight_front_left"
    assert resolve_part_alias("unknown_part_xyz") == "unknown_part_xyz"


def test_load_trigger_set():
    front_false = load_trigger_set("front_false_damage_parts")
    assert "bumper_front" in front_false
    assert "headlight_front_left" in front_false


def test_get_checklist_hints_condition_matching():
    hints = get_checklist_hints({"hood"})
    assert any(h["condition"] == "hood" for h in hints)


def test_get_checklist_hints_any_of():
    hints = get_checklist_hints({"hood"})
    # No any_of conditions in the current hints; just ensure loading works.
    assert isinstance(hints, list)


def test_load_region_units():
    units = load_region_units()
    assert "rear_unit" in units
    assert {"tailgate", "windshield_rear"}.issubset(units["rear_unit"])


def test_load_filename_view_hints():
    hints = load_filename_view_hints()
    assert isinstance(hints, list)
    assert any(pattern == "行驶证" and view_id == "auxiliary" for pattern, view_id in hints)
