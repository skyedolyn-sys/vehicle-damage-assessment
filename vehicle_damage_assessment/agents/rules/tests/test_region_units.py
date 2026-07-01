"""Tests for region_units loader."""

from agents.rules import load_region_units


def test_load_region_units_returns_sets():
    units = load_region_units()
    assert isinstance(units, dict)
    assert "rear_unit" in units
    assert isinstance(units["rear_unit"], set)


def test_load_region_units_rear_unit_membership():
    units = load_region_units()
    assert {"tailgate", "windshield_rear"}.issubset(units["rear_unit"])


def test_load_region_units_unknown_vehicle_type_uses_default():
    units = load_region_units("not_a_real_type")
    assert "rear_unit" in units
