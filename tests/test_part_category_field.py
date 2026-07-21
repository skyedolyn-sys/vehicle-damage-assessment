"""Tests for PartActualState.part_category field rename.

The dataclass field is ``part_category`` (not ``region``) so it matches the
on-wire JSON key emitted by ``to_dict`` / ``to_legacy_dict`` and the
``part_category`` key in :data:`config.PARTS_CATALOG`.  This file pins that
contract.
"""

import dataclasses

import pytest

from models.part_state import DamageLevel, PartActualState, Status


class TestPartCategoryFieldContract:
    """Pin the Python field name, the wire JSON key, and the absence of ``region``."""

    def test_to_dict_emits_part_category_not_region(self):
        """to_dict() must emit ``part_category`` and must not emit ``region``."""
        state = PartActualState(
            part_id="hood",
            part_name="引擎盖",
            part_category="front",
            side="center",
            status=Status.INTACT,
            damage_level=DamageLevel.NONE,
        )
        out = state.to_dict()
        assert "part_category" in out
        assert out["part_category"] == "front"
        assert "region" not in out

    def test_to_legacy_dict_emits_part_category_not_region(self):
        """to_legacy_dict() must emit ``part_category`` and must not emit ``region``."""
        state = PartActualState(
            part_id="bumper_front",
            part_name="前保险杠",
            part_category="front",
            side="center",
            status=Status.DAMAGED,
            damage_level=DamageLevel.MODERATE,
            damage_types=["scratch"],
            confidence="high",
            evidence_photos=["p1.jpg"],
            notes="Light scratch",
        )
        out = state.to_legacy_dict()
        assert "part_category" in out
        assert out["part_category"] == "front"
        assert "region" not in out

    def test_from_legacy_dict_roundtrip(self):
        """from_legacy_dict reads ``part_category`` from the wire and roundtrips."""
        legacy = {
            "part_id": "trunk_lid",
            "part_name": "后备箱盖",
            "part_category": "rear",
            "side": "center",
            "status": "damaged",
            "damage_level": "severe",
            "damage_type": "deformation",
            "confidence": "high",
            "evidence_photo": "p1.jpg",
            "notes": "Crushed rear lid",
        }
        state = PartActualState.from_legacy_dict(legacy)
        assert state.part_category == "rear"

        out = state.to_dict()
        assert out["part_category"] == "rear"
        # to_dict() must not re-emit it under the old ``region`` key.
        assert "region" not in out

    def test_dataclass_attribute_is_part_category(self):
        """The dataclass field is named ``part_category`` and there is no ``region`` field."""
        field_names = {f.name for f in dataclasses.fields(PartActualState)}
        assert "part_category" in field_names
        assert "region" not in field_names

        state = PartActualState(
            part_id="roof_middle",
            part_name="车顶中部",
            part_category="roof",
            side="center",
            status=Status.INTACT,
            damage_level=DamageLevel.NONE,
        )
        # The attribute exists and is accessible under the new name.
        assert state.part_category == "roof"
        # The old attribute name is gone — accessing it must raise AttributeError.
        with pytest.raises(AttributeError):
            _ = state.region

    def test_from_region_part_uses_part_category_kwarg(self):
        """from_region_part factory exposes ``part_category`` as the kwarg name."""
        state = PartActualState.from_region_part(
            part_id="pillar_a_left",
            part_name="左A柱",
            part_category="left",
            side="front_left",
            status=Status.DAMAGED,
            damage_level=DamageLevel.LIGHT,
        )
        assert state.part_category == "left"
