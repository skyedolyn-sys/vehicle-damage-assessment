"""Backward-compatibility tests for the rules package.

These tests verify that rules-loaded data remains consistent with the
consuming agent modules.  Some legacy planner constants have been removed
because the planner no longer assigns views; only the deterministic
classification logic remains.
"""

from agents.rules import (
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


def test_legacy_synthesizer_imports_still_work():
    """Existing agent modules must continue to expose their constants."""
    from agents.synthesizer import (
        CONSERVATIVE_PARTS,
        FRONT_FALSE_DAMAGE_PARTS,
        LEVEL_PRIORITY,
        ROOF_PARTS,
        SPILL_OVER_PRONE_PARTS,
        STATUS_PRIORITY,
        UNCERTAIN_STATUS_PRIORITY,
        VIEW_WEIGHTS,
    )
    assert "door_front_left" in CONSERVATIVE_PARTS
    assert "bumper_front" in FRONT_FALSE_DAMAGE_PARTS
    assert "roof_front" in ROOF_PARTS
    assert "severe" in LEVEL_PRIORITY


def test_legacy_vision_imports_still_work():
    from agents.vision_subagent import _PART_ID_ALIASES
    assert _PART_ID_ALIASES["left_headlight"] == "headlight_front_left"


def test_yaml_defaults_match_legacy_constants():
    """Rules loaded from YAML must equal the legacy Python defaults."""
    from agents.synthesizer import (
        CONSERVATIVE_PARTS,
        FRONT_FALSE_DAMAGE_PARTS,
        ROOF_PARTS,
        SPILL_OVER_PRONE_PARTS,
        VIEW_WEIGHTS,
    )

    assert load_part_profile("conservative") == CONSERVATIVE_PARTS
    assert load_part_profile("roof") == ROOF_PARTS
    assert load_part_profile("front_false_damage") == FRONT_FALSE_DAMAGE_PARTS
    assert load_part_profile("spill_over_prone") == SPILL_OVER_PRONE_PARTS

    weights = load_view_weights()
    assert weights["view_weights"] == VIEW_WEIGHTS


def test_priority_map_matches_legacy():
    from agents.synthesizer import (
        CONFIDENCE_PRIORITY,
        LEVEL_PRIORITY,
        STATUS_PRIORITY,
        UNCERTAIN_STATUS_PRIORITY,
    )

    priorities = load_priority_map()
    assert priorities["status"] == STATUS_PRIORITY
    assert priorities["uncertain_status"] == UNCERTAIN_STATUS_PRIORITY
    assert priorities["level"] == LEVEL_PRIORITY
    assert priorities["confidence"] == CONFIDENCE_PRIORITY


def test_threshold_matches_legacy_value():
    assert load_threshold("visibility_definite_ratio") == 0.30


def test_part_alias_matches_legacy():
    assert resolve_part_alias("left_headlight") == "headlight_front_left"


def test_trigger_sets_match_legacy():
    from agents.synthesizer import FRONT_FALSE_DAMAGE_PARTS, SPILL_OVER_PRONE_PARTS
    assert load_trigger_set("front_false_damage_parts") == FRONT_FALSE_DAMAGE_PARTS
    assert load_trigger_set("spill_over_prone_parts") == SPILL_OVER_PRONE_PARTS


def test_filename_heuristics_match_legacy():
    heuristics = load_filename_heuristics()
    rule_map = {h["name"]: h for h in heuristics}
    assert "auxiliary_license" in rule_map
    assert "行驶证" in rule_map["auxiliary_license"]["patterns"]


def test_region_units_match_legacy():
    from agents.topology_comparator import _REGION_UNITS
    assert load_region_units() == _REGION_UNITS


def test_rendered_templates_are_non_empty_and_contain_expected_content():
    """Rendered Jinja2 templates produce non-empty output with expected placeholders filled."""
    from agents.view_mapping import get_view_selection_prompt
    from agents.rules import render_prompt_template

    rendered = render_prompt_template(
        "view_agent_prompt",
        photo_id="p1",
        vehicle_name="该车",
    )
    assert rendered
    assert "输出 JSON Schema" in rendered
    assert "front_left" in rendered
    assert "hood" in rendered
