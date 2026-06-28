"""Tests for part_state models — Status, DamageLevel, PartActualState."""

import pytest

from models.part_state import DamageLevel, PartActualState, Status


class TestStatusEnum:
    """Test Status enum values and behaviour."""

    def test_status_values(self):
        """Each status has the expected string value."""
        assert Status.INTACT.value == "intact"
        assert Status.DAMAGED.value == "damaged"
        assert Status.MISSING.value == "missing"
        assert Status.UNCERTAIN.value == "uncertain"
        assert Status.NOT_APPLICABLE.value == "na"

    def test_status_from_string(self):
        """Status can be constructed from its string value."""
        assert Status("intact") == Status.INTACT
        assert Status("damaged") == Status.DAMAGED
        assert Status("missing") == Status.MISSING

    def test_invalid_status_raises(self):
        """An invalid string raises ValueError."""
        with pytest.raises(ValueError):
            Status("broken")


class TestDamageLevelEnum:
    """Test DamageLevel enum values and behaviour."""

    def test_damage_level_values(self):
        """Each damage level has the expected string value."""
        assert DamageLevel.NONE.value == "none"
        assert DamageLevel.LIGHT.value == "light"
        assert DamageLevel.MODERATE.value == "moderate"
        assert DamageLevel.SEVERE.value == "severe"
        assert DamageLevel.UNKNOWN.value == "unknown"

    def test_damage_level_from_string(self):
        """DamageLevel can be constructed from its string value."""
        assert DamageLevel("severe") == DamageLevel.SEVERE
        assert DamageLevel("light") == DamageLevel.LIGHT

    def test_invalid_damage_level_raises(self):
        """An invalid string raises ValueError."""
        with pytest.raises(ValueError):
            DamageLevel("critical")


class TestPartActualStateCreation:
    """Test PartActualState construction and field defaults."""

    def test_create_minimal_state(self):
        """A minimal state has sensible defaults for optional fields."""
        state = PartActualState(
            part_id="hood",
            part_name="引擎盖",
            region="front",
            side="center",
            status=Status.INTACT,
            damage_level=DamageLevel.NONE,
        )
        assert state.part_id == "hood"
        assert state.status == Status.INTACT
        assert state.damage_level == DamageLevel.NONE
        assert state.damage_types == []
        assert state.standard_exists is True
        assert state.actual_visible is False
        assert state.actual_present is True
        assert state.confidence == "low"
        assert state.evidence_photos == []
        assert state.notes == ""
        assert state.adjacent_status == {}

    def test_create_full_state(self):
        """All fields can be provided explicitly."""
        state = PartActualState(
            part_id="bumper_front",
            part_name="前保险杠",
            region="front",
            side="center",
            status=Status.DAMAGED,
            damage_level=DamageLevel.MODERATE,
            damage_types=["scratch", "dent"],
            standard_exists=True,
            actual_visible=True,
            actual_present=True,
            confidence="high",
            evidence_photos=["photo_001.jpg", "photo_002.jpg"],
            notes="Visible scratches on lower edge",
            adjacent_status={"grille_front": "intact"},
        )
        assert state.damage_types == ["scratch", "dent"]
        assert state.confidence == "high"
        assert state.evidence_photos == ["photo_001.jpg", "photo_002.jpg"]
        assert state.adjacent_status == {"grille_front": "intact"}


class TestPartActualStateLegacyRoundTrip:
    """Test to_legacy_dict / from_legacy_dict round-trip conversion."""

    def test_to_legacy_dict_structure(self):
        """to_legacy_dict produces the expected flat keys."""
        state = PartActualState(
            part_id="hood",
            part_name="引擎盖",
            region="front",
            side="center",
            status=Status.DAMAGED,
            damage_level=DamageLevel.SEVERE,
            damage_types=["dent", "scratch"],
            confidence="high",
            evidence_photos=["p1.jpg"],
            notes="Deep dent",
        )
        legacy = state.to_legacy_dict()
        assert legacy["part_id"] == "hood"
        assert legacy["part_name"] == "引擎盖"
        assert legacy["part_category"] == "front"
        assert legacy["side"] == "center"
        assert legacy["status"] == "damaged"
        assert legacy["damage_level"] == "severe"
        assert legacy["damage_type"] == "dent, scratch"
        assert legacy["confidence"] == "high"
        assert legacy["evidence_photo"] == "p1.jpg"
        assert legacy["notes"] == "Deep dent"

    def test_to_legacy_dict_empty_lists(self):
        """Empty damage_types and evidence_photos become 'none' and ''."""
        state = PartActualState(
            part_id="grille_front",
            part_name="前格栅",
            region="front",
            side="center",
            status=Status.INTACT,
            damage_level=DamageLevel.NONE,
        )
        legacy = state.to_legacy_dict()
        assert legacy["damage_type"] == "none"
        assert legacy["evidence_photo"] == ""

    def test_from_legacy_dict_round_trip(self):
        """from_legacy_dict(to_legacy_dict()) reconstructs an equivalent state."""
        original = PartActualState(
            part_id="headlight_front_left",
            part_name="左前大灯",
            region="front",
            side="front_left",
            status=Status.MISSING,
            damage_level=DamageLevel.SEVERE,
            damage_types=["shattered"],
            confidence="medium",
            evidence_photos=["img_01.jpg", "img_02.jpg"],
            notes="Completely missing",
        )
        legacy = original.to_legacy_dict()
        restored = PartActualState.from_legacy_dict(legacy)

        assert restored.part_id == original.part_id
        assert restored.part_name == original.part_name
        assert restored.region == original.region
        assert restored.side == original.side
        assert restored.status == original.status
        assert restored.damage_level == original.damage_level
        assert restored.damage_types == original.damage_types
        assert restored.confidence == original.confidence
        assert restored.evidence_photos == original.evidence_photos
        assert restored.notes == original.notes

    def test_from_legacy_dict_with_none_damage_type(self):
        """Legacy dict with damage_type 'none' yields empty damage_types list."""
        legacy = {
            "part_id": "hood",
            "part_name": "引擎盖",
            "part_category": "front",
            "side": "center",
            "status": "intact",
            "damage_level": "none",
            "damage_type": "none",
            "confidence": "low",
            "evidence_photo": "",
            "notes": "",
        }
        state = PartActualState.from_legacy_dict(legacy)
        assert state.damage_types == []
        assert state.status == Status.INTACT
        assert state.damage_level == DamageLevel.NONE

    def test_from_legacy_dict_defaults_on_missing_keys(self):
        """Missing optional keys in legacy dict use sensible defaults."""
        legacy = {
            "part_id": "roof_front",
            "part_name": "车顶前部",
            "part_category": "roof",
            "side": "center",
        }
        state = PartActualState.from_legacy_dict(legacy)
        assert state.status == Status.UNCERTAIN
        assert state.damage_level == DamageLevel.UNKNOWN
        assert state.damage_types == []
        assert state.confidence == "low"
        assert state.evidence_photos == []
        assert state.notes == ""

    def test_from_region_part_factory(self):
        """from_region_part factory creates a state with correct region/side."""
        state = PartActualState.from_region_part(
            part_id="mirror_left",
            part_name="左后视镜",
            region="left",
            side="front_left",
            status=Status.DAMAGED,
            damage_level=DamageLevel.LIGHT,
        )
        assert state.part_id == "mirror_left"
        assert state.region == "left"
        assert state.side == "front_left"
        assert state.status == Status.DAMAGED
        assert state.damage_level == DamageLevel.LIGHT
