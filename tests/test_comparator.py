"""Tests for topology_comparator — TopologyComparator and compare_topology."""

import pytest

from agents.topology_builder import build_vehicle_topology
from agents.topology_comparator import TopologyComparator, compare_topology
from models.part_state import DamageLevel, PartActualState, Status
from models.topology import VehicleTopology


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_topology():
    """Return a full 33-part VehicleTopology built from dummy inputs."""
    vehicle_info = {"vehicle_id": "v-001", "vehicle_name": "Test Sedan"}
    vehicle_prior = {
        "topology": {
            "front": "前部部件",
            "rear": "后部部件",
            "left": "左侧部件",
            "right": "右侧部件",
            "roof": "车顶部件",
        },
        "key_anchors": {
            "front": ["anchor_front"],
            "rear": ["anchor_rear"],
            "left": ["anchor_left"],
            "right": ["anchor_right"],
            "roof": ["anchor_roof"],
        },
    }
    return build_vehicle_topology(vehicle_info, vehicle_prior)


def _make_state(part_id, region, side, status, damage_level=DamageLevel.NONE, **kwargs):
    """Helper to create a PartActualState with minimal boilerplate.

    The ``region`` parameter is the part_category string (e.g. "front") and is
    forwarded to the PartActualState field of the same name.
    """
    return PartActualState(
        part_id=part_id,
        part_name=part_id.replace("_", " "),
        part_category=region,
        side=side,
        status=status,
        damage_level=damage_level,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Basic comparison scenarios
# ---------------------------------------------------------------------------

class TestCompareAllIntact:
    """Scenario: every part is intact — no patterns, no structural flag."""

    def test_all_intact_no_patterns(self, base_topology):
        """When all parts are intact, no structural patterns are detected."""
        states = [
            _make_state(node.node_id, node.region, node.side, Status.INTACT, DamageLevel.NONE)
            for node in base_topology.nodes.values()
        ]
        result = compare_topology(base_topology, states)

        assert result.structural_damage_flag is False
        assert result.structural_patterns == []
        assert result.overall_severity == "none"
        assert len(result.intact_parts) == 33
        assert result.damaged_parts == []
        assert result.missing_parts == []

    def test_all_intact_summary_counts(self, base_topology):
        """Summary counts reflect 33 intact, 0 damaged/missing/uncertain."""
        states = [
            _make_state(node.node_id, node.region, node.side, Status.INTACT, DamageLevel.NONE)
            for node in base_topology.nodes.values()
        ]
        result = compare_topology(base_topology, states)

        assert result.summary["total_parts"] == 33
        assert result.summary["intact_count"] == 33
        assert result.summary["damaged_count"] == 0
        assert result.summary["missing_count"] == 0
        assert result.summary["uncertain_count"] == 0


class TestCompareOneDamaged:
    """Scenario: a single part is damaged — no structural patterns."""

    def test_single_damaged_no_structural_flag(self, base_topology):
        """One damaged part does not trigger any structural pattern."""
        states = [
            _make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.MODERATE),
        ]
        result = compare_topology(base_topology, states)

        assert result.structural_damage_flag is False
        assert result.damaged_parts == ["hood"]
        assert len(result.intact_parts) == 0  # only 1 state provided, rest are uncertain
        assert result.overall_severity == "none"

    def test_uncertain_parts_counted_when_not_provided(self, base_topology):
        """Nodes without an actual state are synthesised as UNCERTAIN."""
        states = [_make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.MODERATE)]
        result = compare_topology(base_topology, states)

        assert result.summary["uncertain_count"] == 32
        expected_uncertain = [nid for nid in base_topology.nodes if nid != "hood"]
        assert sorted(result.uncertain_parts) == sorted(expected_uncertain)


class TestSafetyCriticalMissing:
    """Scenario: missing headlight triggers safety_critical_missing pattern."""

    def test_missing_headlight_triggers_safety_critical(self, base_topology):
        """A missing headlight is safety-critical and sets structural_flag=True."""
        states = [
            _make_state("headlight_front_left", "front", "front_left", Status.MISSING, DamageLevel.SEVERE),
        ]
        result = compare_topology(base_topology, states)

        assert result.structural_damage_flag is True
        pattern_ids = [p.pattern_id for p in result.structural_patterns]
        assert "safety_critical_missing" in pattern_ids

    def test_missing_taillight_and_mirror_triggers_safety_critical(self, base_topology):
        """Multiple safety-critical missing parts are captured in one pattern."""
        states = [
            _make_state("taillight_rear_left", "rear", "rear_left", Status.MISSING, DamageLevel.SEVERE),
            _make_state("mirror_left", "left", "front_left", Status.MISSING, DamageLevel.SEVERE),
        ]
        result = compare_topology(base_topology, states)

        safety_pat = [p for p in result.structural_patterns if p.pattern_id == "safety_critical_missing"]
        assert len(safety_pat) == 1
        assert set(safety_pat[0].matched_nodes) == {"taillight_rear_left", "mirror_left"}
        assert safety_pat[0].severity == "severe"


class TestFrontCollision:
    """Scenario: front collision with multiple parts — triggers regional_mass_damage and symmetric_damage."""

    def test_front_collision_regional_mass_damage(self, base_topology):
        """>=3 damaged parts in the front region triggers regional_mass_damage."""
        states = [
            _make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("bumper_front", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("grille_front", "front", "center", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("headlight_front_left", "front", "front_left", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("headlight_front_right", "front", "front_right", Status.DAMAGED, DamageLevel.MODERATE),
        ]
        result = compare_topology(base_topology, states)

        pattern_ids = [p.pattern_id for p in result.structural_patterns]
        assert "regional_mass_damage" in pattern_ids
        # front has 5 damaged parts >= 3 threshold
        regional = [p for p in result.structural_patterns if p.pattern_id == "regional_mass_damage"][0]
        assert regional.severity == "severe"
        assert len(regional.matched_nodes) == 5

    def test_front_collision_symmetric_damage(self, base_topology):
        """Both left and right headlights + both fenders triggers symmetric_damage."""
        states = [
            _make_state("headlight_front_left", "front", "front_left", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("headlight_front_right", "front", "front_right", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("fender_front_left", "front", "front_left", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("fender_front_right", "front", "front_right", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("door_front_left", "left", "front_left", Status.DAMAGED, DamageLevel.LIGHT),
            _make_state("door_front_right", "right", "front_right", Status.DAMAGED, DamageLevel.LIGHT),
        ]
        result = compare_topology(base_topology, states)

        pattern_ids = [p.pattern_id for p in result.structural_patterns]
        assert "symmetric_damage" in pattern_ids
        sym = [p for p in result.structural_patterns if p.pattern_id == "symmetric_damage"][0]
        assert sym.severity == "moderate"
        # At least 2 symmetric pairs: headlights + fenders + doors = 3 pairs
        assert len(sym.matched_nodes) >= 4

    def test_front_collision_overall_severity_moderate(self, base_topology):
        """Front collision with regional_mass + symmetric yields moderate overall."""
        states = [
            _make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("bumper_front", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("grille_front", "front", "center", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("headlight_front_left", "front", "front_left", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("headlight_front_right", "front", "front_right", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("fender_front_left", "front", "front_left", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("fender_front_right", "front", "front_right", Status.DAMAGED, DamageLevel.MODERATE),
        ]
        result = compare_topology(base_topology, states)

        # regional_mass (severe) + symmetric (moderate, headlights+fenders) => at least 1 severe => moderate overall
        assert result.overall_severity == "moderate"
        assert result.structural_damage_flag is True

    def test_front_collision_primary_zone_is_front(self, base_topology):
        """Primary damage zone should be 'front' when front parts dominate."""
        states = [
            _make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("bumper_front", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("grille_front", "front", "center", Status.DAMAGED, DamageLevel.MODERATE),
        ]
        result = compare_topology(base_topology, states)

        assert result.primary_damage_zone == "front"


class TestSideScrape:
    """Scenario: left-side scrape triggers regional_mass_damage on the left region."""

    def test_left_side_scrape_regional_mass(self, base_topology):
        """3+ damaged parts on the left side triggers regional_mass_damage for 'left'."""
        states = [
            _make_state("door_front_left", "left", "front_left", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("door_rear_left", "left", "rear_left", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("fender_rear_left", "left", "rear_left", Status.DAMAGED, DamageLevel.LIGHT),
            _make_state("mirror_left", "left", "front_left", Status.DAMAGED, DamageLevel.LIGHT),
        ]
        result = compare_topology(base_topology, states)

        pattern_ids = [p.pattern_id for p in result.structural_patterns]
        assert "regional_mass_damage" in pattern_ids
        regional = [p for p in result.structural_patterns if p.pattern_id == "regional_mass_damage"][0]
        assert "left" in regional.description
        assert regional.severity == "severe"

    def test_right_side_scrape_regional_mass(self, base_topology):
        """3+ damaged parts on the right side triggers regional_mass_damage for 'right'."""
        states = [
            _make_state("door_front_right", "right", "front_right", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("door_rear_right", "right", "rear_right", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("fender_rear_right", "right", "rear_right", Status.DAMAGED, DamageLevel.LIGHT),
            _make_state("mirror_right", "right", "front_right", Status.DAMAGED, DamageLevel.LIGHT),
        ]
        result = compare_topology(base_topology, states)

        pattern_ids = [p.pattern_id for p in result.structural_patterns]
        assert "regional_mass_damage" in pattern_ids
        regional = [p for p in result.structural_patterns if p.pattern_id == "regional_mass_damage"][0]
        assert "right" in regional.description


class TestStructuralComponentDamage:
    """Scenario: structural roof damage triggers structural_component_damage."""

    def test_roof_front_structural_damage(self, base_topology):
        """Damage to a structural-type node triggers structural_component_damage."""
        states = [
            _make_state("roof_front", "roof", "center", Status.DAMAGED, DamageLevel.SEVERE),
        ]
        result = compare_topology(base_topology, states)

        pattern_ids = [p.pattern_id for p in result.structural_patterns]
        assert "structural_component_damage" in pattern_ids
        struct = [p for p in result.structural_patterns if p.pattern_id == "structural_component_damage"][0]
        assert "roof_front" in struct.matched_nodes
        assert struct.severity == "severe"

    def test_multiple_structural_parts(self, base_topology):
        """Multiple structural parts damaged are all in one pattern."""
        states = [
            _make_state("roof_front", "roof", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("roof_middle", "roof", "center", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("roof_rear", "roof", "center", Status.DAMAGED, DamageLevel.MODERATE),
        ]
        result = compare_topology(base_topology, states)

        struct = [p for p in result.structural_patterns if p.pattern_id == "structural_component_damage"][0]
        assert set(struct.matched_nodes) == {"roof_front", "roof_middle", "roof_rear"}
        assert result.structural_damage_flag is True


class TestCrossRegionPenetration:
    """Scenario: damage cluster spans multiple regions via adjacency."""

    def test_cross_region_via_adjacency(self, base_topology):
        """A connected cluster of damaged nodes spanning front+left triggers cross_region_penetration."""
        # fender_front_left connects front and left regions
        states = [
            _make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("fender_front_left", "front", "front_left", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("door_front_left", "left", "front_left", Status.DAMAGED, DamageLevel.MODERATE),
        ]
        result = compare_topology(base_topology, states)

        pattern_ids = [p.pattern_id for p in result.structural_patterns]
        assert "cross_region_penetration" in pattern_ids
        cross = [p for p in result.structural_patterns if p.pattern_id == "cross_region_penetration"][0]
        assert cross.severity == "severe"
        regions_spanned = {"front", "left"}
        assert regions_spanned.issubset(set(cross.description.split())) or "front" in cross.description

    def test_cross_region_front_to_roof(self, base_topology):
        """Damage from front through windshield to roof spans front+roof."""
        states = [
            _make_state("windshield_front", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("roof_front", "roof", "center", Status.DAMAGED, DamageLevel.SEVERE),
        ]
        result = compare_topology(base_topology, states)

        # windshield_front is adjacent to roof_front
        pattern_ids = [p.pattern_id for p in result.structural_patterns]
        assert "cross_region_penetration" in pattern_ids

    def test_no_cross_region_when_regions_not_connected(self, base_topology):
        """Disjoint damaged nodes in different regions do NOT trigger cross_region_penetration."""
        # hood (front) and trunk_lid (rear) are not adjacent
        states = [
            _make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("trunk_lid", "rear", "center", Status.DAMAGED, DamageLevel.SEVERE),
        ]
        result = compare_topology(base_topology, states)

        pattern_ids = [p.pattern_id for p in result.structural_patterns]
        assert "cross_region_penetration" not in pattern_ids


class TestMissingVsUncertain:
    """Test that missing parts are distinguished from uncertain parts."""

    def test_missing_part_in_missing_list(self, base_topology):
        """A part with status MISSING appears in missing_parts list."""
        states = [
            _make_state("headlight_front_left", "front", "front_left", Status.MISSING, DamageLevel.SEVERE),
        ]
        result = compare_topology(base_topology, states)
        assert "headlight_front_left" in result.missing_parts
        assert "headlight_front_left" not in result.damaged_parts
        assert "headlight_front_left" not in result.intact_parts

    def test_uncertain_part_in_uncertain_list(self, base_topology):
        """A part with status UNCERTAIN appears in uncertain_parts list."""
        states = [
            _make_state("hood", "front", "center", Status.UNCERTAIN, DamageLevel.UNKNOWN),
        ]
        result = compare_topology(base_topology, states)
        assert "hood" in result.uncertain_parts

    def test_unprovided_node_becomes_uncertain(self, base_topology):
        """A topology node with no corresponding actual state is synthesised as UNCERTAIN."""
        states = []
        result = compare_topology(base_topology, states)
        assert len(result.uncertain_parts) == 33
        # Verify all parts in the result have UNCERTAIN status
        for part in result.parts:
            assert part.status == Status.UNCERTAIN


class TestOverallSeverity:
    """Test overall severity computation."""

    def test_severe_none(self, base_topology):
        """No patterns => overall severity 'none'."""
        states = [_make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.LIGHT)]
        result = compare_topology(base_topology, states)
        assert result.overall_severity == "none"

    def test_severe_light(self, base_topology):
        """Only moderate patterns (1 moderate) => overall severity 'light'."""
        # Use 2 symmetric pairs (mirrors + doors) to get symmetric_damage (moderate)
        # without triggering regional_mass_damage (need 3+ in same region)
        states = [
            _make_state("mirror_left", "left", "front_left", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("mirror_right", "right", "front_right", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("door_front_left", "left", "front_left", Status.DAMAGED, DamageLevel.MODERATE),
            _make_state("door_front_right", "right", "front_right", Status.DAMAGED, DamageLevel.MODERATE),
        ]
        result = compare_topology(base_topology, states)
        # 1 moderate pattern (symmetric, 2 pairs) => light overall
        assert result.overall_severity == "light"

    def test_severe_moderate(self, base_topology):
        """At least one severe pattern => overall severity 'moderate'."""
        states = [
            _make_state("headlight_front_left", "front", "front_left", Status.MISSING, DamageLevel.SEVERE),
        ]
        result = compare_topology(base_topology, states)
        # safety_critical_missing is severe => moderate overall
        assert result.overall_severity == "moderate"

    def test_severe_severe(self, base_topology):
        """>=3 severe patterns => overall severity 'severe'."""
        # Create 3 severe patterns: safety_critical + regional_mass + structural_component
        states = [
            # safety_critical_missing
            _make_state("headlight_front_left", "front", "front_left", Status.MISSING, DamageLevel.SEVERE),
            # regional_mass_damage in front (need 3+ in front)
            _make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("bumper_front", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            _make_state("grille_front", "front", "center", Status.DAMAGED, DamageLevel.SEVERE),
            # structural_component_damage
            _make_state("roof_front", "roof", "center", Status.DAMAGED, DamageLevel.SEVERE),
        ]
        result = compare_topology(base_topology, states)
        severe_count = sum(1 for p in result.structural_patterns if p.severity == "severe")
        assert severe_count >= 3
        assert result.overall_severity == "severe"


class TestPrimaryDamageZone:
    """Test primary damage zone computation."""

    def test_primary_zone_front(self, base_topology):
        """Highest weighted damage in front region => primary_zone 'front'."""
        # Provide all 25 parts: 3 damaged in front (high weight), 1 light in rear
        states = []
        for node in base_topology.nodes.values():
            if node.node_id == "hood":
                states.append(_make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.SEVERE))
            elif node.node_id == "bumper_front":
                states.append(_make_state("bumper_front", "front", "center", Status.DAMAGED, DamageLevel.SEVERE))
            elif node.node_id == "grille_front":
                states.append(_make_state("grille_front", "front", "center", Status.DAMAGED, DamageLevel.MODERATE))
            elif node.node_id == "trunk_lid":
                states.append(_make_state("trunk_lid", "rear", "center", Status.DAMAGED, DamageLevel.LIGHT))
            else:
                states.append(_make_state(node.node_id, node.region, node.side, Status.INTACT, DamageLevel.NONE))
        result = compare_topology(base_topology, states)
        assert result.primary_damage_zone == "front"

    def test_primary_zone_multiple_when_tied(self, base_topology):
        """Tied scores across regions => primary_zone 'multiple'."""
        # Provide all 25 parts: 1 MODERATE in front, 1 MODERATE in rear, rest INTACT
        states = []
        for node in base_topology.nodes.values():
            if node.node_id == "hood":
                states.append(_make_state("hood", "front", "center", Status.DAMAGED, DamageLevel.MODERATE))
            elif node.node_id == "trunk_lid":
                states.append(_make_state("trunk_lid", "rear", "center", Status.DAMAGED, DamageLevel.MODERATE))
            else:
                states.append(_make_state(node.node_id, node.region, node.side, Status.INTACT, DamageLevel.NONE))
        result = compare_topology(base_topology, states)
        assert result.primary_damage_zone == "multiple"

    def test_primary_zone_none_when_no_damage(self, base_topology):
        """No damaged parts => primary_zone 'multiple'."""
        # All 25 parts intact
        states = [
            _make_state(node.node_id, node.region, node.side, Status.INTACT, DamageLevel.NONE)
            for node in base_topology.nodes.values()
        ]
        result = compare_topology(base_topology, states)
        assert result.primary_damage_zone == "multiple"
