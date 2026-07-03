"""Tests for the canonical evidence_photo normaliser.

Covers the three input shapes the field takes across the codebase:
- JSON list (LLM JSON output, ``PartActualState.to_dict()``)
- CSV string (some LLM prompts, ``PartActualState.to_legacy_dict()``)
- The :class:`PartActualState` ``evidence_photos`` (plural) attribute

Also covers ``to_csv`` (rendering normalised list back to CSV).

This test file replaces the per-module ad-hoc split logic with a single
canonical normaliser. If you find yourself adding a new ``split(',')`` to
normalise evidence_photo data anywhere in the codebase, you should:
1. Add a case here that pins the behaviour.
2. Route the new call site through :func:`to_photo_list` instead.
"""

from __future__ import annotations

import pytest

from agents.evidence_photo import to_csv, to_photo_list


# ---------------------------------------------------------------------------
# to_photo_list — None / empty / "none" sentinels
# ---------------------------------------------------------------------------

class TestToPhotoListSentinels:
    @pytest.mark.parametrize(
        "raw",
        [None, "", "none", "NONE", "None", "  ", "\t\n", "  none  "],
        ids=["none", "empty", "none_lower", "none_upper", "none_title",
             "whitespace_only", "tabs_newlines", "padded_none"],
    )
    def test_returns_empty_for_sentinels(self, raw):
        assert to_photo_list(raw) == []

    def test_none_string_with_outer_whitespace(self):
        # The spec calls for case-insensitive match; whitespace around "none"
        # is stripped by the comparison.
        assert to_photo_list(" none ") == []


# ---------------------------------------------------------------------------
# to_photo_list — string inputs
# ---------------------------------------------------------------------------

class TestToPhotoListStrings:
    def test_single_photo(self):
        assert to_photo_list("a.png") == ["a.png"]

    def test_two_photos_csv(self):
        assert to_photo_list("a.png, b.png") == ["a.png", "b.png"]

    def test_two_photos_csv_no_space(self):
        # No space after comma — still works, leading/trailing spaces are
        # stripped per fragment.
        assert to_photo_list("a.png,b.png") == ["a.png", "b.png"]

    def test_trailing_comma_drops_empty_fragment(self):
        assert to_photo_list("a.png, b.png, ") == ["a.png", "b.png"]

    def test_leading_comma_drops_empty_fragment(self):
        assert to_photo_list(", a.png, b.png") == ["a.png", "b.png"]

    def test_multiple_photos_with_padding(self):
        # Spaces around fragments should be stripped.
        assert to_photo_list("  a.png ,  b.png  ,  c.png  ") == [
            "a.png", "b.png", "c.png"
        ]

    def test_preserves_duplicates(self):
        # to_photo_list does not dedup — callers wrap if they need it.
        assert to_photo_list("a.png, a.png, b.png") == ["a.png", "a.png", "b.png"]


# ---------------------------------------------------------------------------
# to_photo_list — list / tuple inputs
# ---------------------------------------------------------------------------

class TestToPhotoListLists:
    def test_single_element_list(self):
        assert to_photo_list(["a.png"]) == ["a.png"]

    def test_two_element_list(self):
        assert to_photo_list(["a.png", "b.png"]) == ["a.png", "b.png"]

    def test_none_in_list_filtered(self):
        assert to_photo_list(["a.png", None, "b.png"]) == ["a.png", "b.png"]

    def test_empty_string_in_list_filtered(self):
        assert to_photo_list(["", "a.png"]) == ["a.png"]

    def test_whitespace_only_string_in_list_filtered(self):
        # str("  ").strip() == "" so the fragment is dropped.
        assert to_photo_list(["  ", "a.png"]) == ["a.png"]

    def test_empty_list(self):
        assert to_photo_list([]) == []

    def test_tuple_input(self):
        assert to_photo_list(("a.png", "b.png")) == ["a.png", "b.png"]

    def test_tuple_with_none_filtered(self):
        assert to_photo_list(("a.png", None, "b.png")) == ["a.png", "b.png"]

    def test_list_items_stripped(self):
        # str(p).strip() is applied per element.
        assert to_photo_list(["  a.png  ", "  b.png  "]) == ["a.png", "b.png"]


# ---------------------------------------------------------------------------
# to_photo_list — non-iterable / unsupported inputs
# ---------------------------------------------------------------------------

class TestToPhotoListOther:
    def test_int_returns_empty(self):
        # Unsupported scalar types return []; only str / list / tuple
        # are real input shapes.
        assert to_photo_list(42) == []

    def test_dict_returns_empty(self):
        assert to_photo_list({"k": "v"}) == []

    def test_bool_returns_empty(self):
        # bool is a Python int subclass; we don't accept it as photo data.
        assert to_photo_list(True) == []


# ---------------------------------------------------------------------------
# to_csv — render a normalised list back to a CSV string
# ---------------------------------------------------------------------------

class TestToCsv:
    def test_empty_list_renders_as_empty_string(self):
        assert to_csv([]) == ""

    def test_single_photo(self):
        assert to_csv(["a.png"]) == "a.png"

    def test_two_photos(self):
        assert to_csv(["a.png", "b.png"]) == "a.png, b.png"

    def test_three_photos(self):
        assert to_csv(["a.png", "b.png", "c.png"]) == "a.png, b.png, c.png"

    def test_none_in_list_filtered(self):
        assert to_csv([None, "a.png"]) == "a.png"

    def test_empty_string_in_list_filtered(self):
        # `if p` skips the empty string.
        assert to_csv(["", "a.png"]) == "a.png"

    def test_round_trip_csv(self):
        # to_csv then to_photo_list should preserve identity for clean inputs.
        original = ["172852-04.png", "172852-03.png"]
        assert to_photo_list(to_csv(original)) == original
