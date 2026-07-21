"""Tests for filename_view_hints loader."""

from agents.rules import load_filename_view_hints


def test_load_filename_view_hints_returns_ordered_tuples():
    hints = load_filename_view_hints()
    assert isinstance(hints, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in hints)


def test_load_filename_view_hints_has_known_patterns():
    hints = load_filename_view_hints()
    by_pattern = {p: v for p, v in hints}
    assert by_pattern.get("行驶证") == "auxiliary"
    assert by_pattern.get("车内") == "interior"
    assert by_pattern.get("-01.") == "auxiliary"
    assert by_pattern.get("-07.") == "interior"


def test_load_filename_view_hints_priority_ordering():
    hints = load_filename_view_hints()
    # Higher priority patterns should appear before lower priority ones.
    priorities = {
        "行驶证": 100,
        "vin": 100,
        "内饰": 90,
        "-01.": 80,
    }
    positions = {pattern: idx for idx, (pattern, _) in enumerate(hints)}
    for hi_pat, hi_pri in priorities.items():
        for lo_pat, lo_pri in priorities.items():
            if hi_pri > lo_pri:
                assert positions[hi_pat] < positions[lo_pat], f"{hi_pat} should precede {lo_pat}"
