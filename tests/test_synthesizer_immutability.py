"""Tests verifying immutability of synthesizer helpers.

These tests protect the contract that ``_append_note`` and
``_set_damaged_severe`` do not mutate their input dict — they return a
new dict instead.  This matters because the synthesizer frequently holds
references to parts that downstream callers (orchestrator, merger) also
inspect; an in-place mutation here can silently corrupt the upstream
state.
"""

from agents.synthesizer import _append_note, _set_damaged_severe


class TestAppendNoteImmutability:
    def test_returns_new_dict_not_mutating_input(self):
        part = {"part_id": "hood", "notes": "original"}
        new_part = _append_note(part, "appended")
        # Original dict unchanged.
        assert part["notes"] == "original"
        # New dict has the appended note.
        assert new_part["notes"] == "original；appended"
        # Caller must rebind to the new dict; old reference is untouched.
        assert part is not new_part

    def test_no_existing_note(self):
        part = {"part_id": "fender_front_left"}
        new_part = _append_note(part, "first note")
        assert part == {"part_id": "fender_front_left"}
        assert new_part["notes"] == "first note"
        assert part is not new_part

    def test_appending_twice_does_not_alias(self):
        part = {"part_id": "door_front_left", "notes": "base"}
        new_part = _append_note(part, "first")
        # Caller forgot to use the return value — original is intact, that
        # is the whole point of immutability.
        new_part_2 = _append_note(new_part, "second")
        assert part["notes"] == "base"
        assert new_part["notes"] == "base；first"
        assert new_part_2["notes"] == "base；first；second"
        assert part is not new_part
        assert new_part is not new_part_2


class TestSetDamagedSevereImmutability:
    def test_returns_new_dict_not_mutating_input(self):
        part = {
            "part_id": "hood",
            "status": "uncertain",
            "damage_level": "unknown",
            "damage_type": [],
            "confidence": "high",
            "notes": "",
        }
        new_part = _set_damaged_severe(part, "deformation", "test note")
        # Original unchanged.
        assert part["status"] == "uncertain"
        assert part["damage_level"] == "unknown"
        assert part["damage_type"] == []
        assert part["confidence"] == "high"
        # New part has the damaged-severe conclusion and the appended note.
        assert new_part["status"] == "damaged"
        assert new_part["damage_level"] == "severe"
        assert new_part["damage_type"] == ["deformation"]
        assert new_part["confidence"] == "low"
        assert new_part["notes"] == "test note"
        assert part is not new_part

    def test_preserves_other_keys(self):
        part = {
            "part_id": "taillight_rear_left",
            "status": "intact",
            "_adjacency_override": False,
            "evidence_sources": [{"view_id": "side_left"}],
        }
        new_part = _set_damaged_severe(part, "missing", "neighbor-driven")
        assert new_part["evidence_sources"] == [{"view_id": "side_left"}]
        assert new_part["_adjacency_override"] is False
        assert new_part["status"] == "damaged"
        assert new_part["notes"] == "neighbor-driven"
        # Original preserved.
        assert part["status"] == "intact"
        assert part.get("notes") is None
