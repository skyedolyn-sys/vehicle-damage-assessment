"""Tests for the central damage_type allow-list and the vision_subagent normaliser.

Covers:
- Loader returns the expected shape and contents from damage_types.yaml
- The normaliser handles None / "" / "none" / single values / csv strings /
  lists of strings (including aliases, unknown values, duplicates)
- The allow-list is the single source of truth for canonical strings

The ``_normalise_damage_types`` helper is the public entry point used by
``_llm_dict_to_part_actual_state`` so every ``PartActualState`` emitted by
vision_subagent has a canonical damage_types list.
"""

from __future__ import annotations

import pytest

from agents.rules import load_damage_type_allowlist
from agents.vision_subagent import _normalise_damage_types


# ---------------------------------------------------------------------------
# Loader smoke
# ---------------------------------------------------------------------------

class TestDamageTypeAllowlistLoader:
    def test_loader_returns_three_keys(self):
        allowlist = load_damage_type_allowlist()
        assert set(allowlist.keys()) == {"allowed", "default", "aliases"}

    def test_allowed_contains_known_canonical_types(self):
        allowlist = load_damage_type_allowlist()
        for canonical in ("crack", "deformation", "tear", "missing", "none"):
            assert canonical in allowlist["allowed"], (
                f"{canonical!r} missing from allowed list"
            )

    def test_default_is_a_string(self):
        allowlist = load_damage_type_allowlist()
        assert isinstance(allowlist["default"], str)
        assert allowlist["default"]

    def test_aliases_are_str_to_str(self):
        allowlist = load_damage_type_allowlist()
        assert isinstance(allowlist["aliases"], dict)
        for raw, canonical in allowlist["aliases"].items():
            assert isinstance(raw, str)
            assert isinstance(canonical, str)
            assert canonical in allowlist["allowed"], (
                f"alias {raw!r} points at non-canonical {canonical!r}"
            )


# ---------------------------------------------------------------------------
# _normalise_damage_types — table-driven simple cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # None / empty inputs → ["none"]
        (None, ["none"]),
        ("", ["none"]),
        ("none", ["none"]),
        ([], ["none"]),
        # Single canonical string
        ("crack", ["crack"]),
        # Alias → canonical
        ("broken", ["glass_breakage"]),
        ("shattered", ["glass_breakage"]),
        ("cracked", ["crack"]),
        # CSV string with whitespace + alias + canonical + canonical
        (
            "broken, cracked, paint_damage",
            ["glass_breakage", "crack", "paint_damage"],
        ),
        # List with mixed canonical + unknown → unknown falls back to default
        (["crack", "tear", "made_up_type"], ["crack", "tear", "none"]),
        # Dedupe
        (["crack", "crack"], ["crack"]),
        # Alias + canonical mix
        (["shattered", "crack"], ["glass_breakage", "crack"]),
        # Already-canonical strings are passed through
        (["dent", "scratch"], ["dent", "scratch"]),
        # All-unknown list → ["default"]
        (["made_up_type", "another_unknown"], ["none"]),
        # Whitespace and case are normalised
        ("  Crack  ", ["crack"]),
        ("DENT", ["dent"]),
    ],
)
def test_normalise_damage_types_table(raw, expected):
    assert _normalise_damage_types(raw) == expected


# ---------------------------------------------------------------------------
# _normalise_damage_types — structural / type-coercion cases
# ---------------------------------------------------------------------------


class TestNormaliseDamageTypesShape:
    def test_returns_list_of_strings(self):
        out = _normalise_damage_types(["crack", "deformation"])
        assert isinstance(out, list)
        assert all(isinstance(x, str) for x in out)

    def test_returns_only_canonical_strings(self):
        allowlist = load_damage_type_allowlist()
        canonical = set(allowlist["allowed"])
        # Feed a wide mix of valid, alias, and unknown values.
        out = _normalise_damage_types(
            ["crack", "broken", "made_up_type", "tear", "shattered"]
        )
        # Each returned token must be in the allow-list.
        for token in out:
            assert token in canonical, f"{token!r} is not in allow-list"

    def test_preserves_input_order_for_canonical(self):
        out = _normalise_damage_types(["tear", "crack", "dent"])
        assert out == ["tear", "crack", "dent"]

    def test_csv_with_extra_whitespace(self):
        out = _normalise_damage_types("crack  ,   dent ,scratch")
        assert out == ["crack", "dent", "scratch"]

    def test_none_yields_default(self):
        out = _normalise_damage_types(None)
        assert out == ["none"]

    def test_unknown_type_yields_default(self):
        out = _normalise_damage_types(["something_brand_new"])
        assert out == ["none"]

    def test_already_canonical_passes_through_unchanged(self):
        out = _normalise_damage_types(["crack"])
        assert out == ["crack"]

    def test_alias_chains_resolved_to_canonical(self):
        # broken → glass_breakage, then glass_breakage is canonical.
        out = _normalise_damage_types(["broken"])
        assert out == ["glass_breakage"]

    def test_mixed_list_with_unknown_falls_back_per_item(self):
        out = _normalise_damage_types(["crack", "bogus", "tear"])
        # crack and tear stay; bogus falls back to default ("none")
        assert "crack" in out
        assert "tear" in out
        assert "none" in out
        assert len(out) == 3