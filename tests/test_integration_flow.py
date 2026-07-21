"""Integration tests — full workflow: build topology, create states, compare, validate output."""

import pytest

from agents.topology_builder import build_vehicle_topology
from agents.topology_comparator import compare_topology
from models.assessment import DamageAssessment
from models.part_state import DamageLevel, PartActualState, Status
from models.topology import VehicleTopology


@pytest.fixture
def sample_vehicle_info():
    return {"vehicle_id": "v-int-001", "vehicle_name": "Integration Test Sedan"}


@pytest.fixture
def sample_vehicle_prior():
    return {
        "topology": {
            "front": "前部部件群",
            "rear": "后部部件群",
            "left": "左侧部件群",
            "right": "右侧部件群",
            "roof": "车顶部件群",
        },
        "key_anchors": {
            "front": ["front_anchor"],
            "rear": ["rear_anchor"],
            "left": ["left_anchor"],
            "right": ["right_anchor"],
            "roof": ["roof_anchor"],
        },
    }


class TestIntegrationBuildAndCompare:
    """End-to-end: build topology from vehicle info, compare with actual states."""

    def test_build_topology_has_all_nodes(self, sample_vehicle_info, sample_vehicle_prior):
        """Building a topology yields 33 nodes and 5 canonical regions."""
        topo = build_vehicle_topology(sample_vehicle_info, sample_vehicle_prior)

        assert isinstance(topo, VehicleTopology)
        assert topo.vehicle_id == "v-int-001"
        assert topo.vehicle_name == "Integration Test Sedan"
        assert len(topo.nodes) == 33
        assert len(topo.regions) == 5

    def test_compare_output_has_required_keys(self, sample_vehicle_info, sample_vehicle_prior):
        """The DamageAssessment result contains all required top-level keys."""
        topo = build_vehicle_topology(sample_vehicle_info, sample_vehicle_prior)

        states = [
            PartActualState(
                part_id="hood",
                part_name="引擎盖",
                part_category="front",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.MODERATE,
                damage_types=["dent"],
                confidence="high",
                evidence_photos=["photo_001.jpg"],
                notes="Visible dent on hood center",
            ),
            PartActualState(
                part_id="bumper_front",
                part_name="前保险杠",
                part_category="front",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.LIGHT,
                damage_types=["scratch"],
                confidence="medium",
                evidence_photos=["photo_002.jpg"],
            ),
        ]

        result = compare_topology(topo, states)

        assert isinstance(result, DamageAssessment)
        assert "vehicle_id" in result.vehicle_info
        assert result.vehicle_info["vehicle_id"] == "v-int-001"

        # Topology model embedded
        assert "topology_model" in result.topology_model or result.topology_model
        assert result.topology_model is not None

        # Parts list
        assert len(result.parts) == 33
        hood_state = next(p for p in result.parts if p.part_id == "hood")
        assert hood_state.status == Status.DAMAGED
        assert hood_state.damage_level == DamageLevel.MODERATE

        # Classification lists
        assert "hood" in result.damaged_parts
        assert "bumper_front" in result.damaged_parts
        assert len(result.missing_parts) == 0

        # Structural patterns
        assert isinstance(result.structural_patterns, list)
        # With only 2 damaged parts, no patterns should trigger
        assert result.structural_patterns == []
        assert result.structural_damage_flag is False

        # Severity and zone
        assert result.overall_severity == "none"
        assert result.primary_damage_zone == "front"

        # Summary
        assert result.summary["total_parts"] == 33
        assert result.summary["damaged_count"] == 2
        assert result.summary["missing_count"] == 0

    def test_compare_with_missing_and_damaged(self, sample_vehicle_info, sample_vehicle_prior):
        """A mix of missing and damaged parts produces correct classification."""
        topo = build_vehicle_topology(sample_vehicle_info, sample_vehicle_prior)

        states = [
            PartActualState(
                part_id="headlight_front_left",
                part_name="左前大灯",
                part_category="front",
                side="front_left",
                status=Status.MISSING,
                damage_level=DamageLevel.SEVERE,
                confidence="high",
            ),
            PartActualState(
                part_id="hood",
                part_name="引擎盖",
                part_category="front",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.SEVERE,
                damage_types=["crumple"],
                confidence="high",
            ),
            PartActualState(
                part_id="bumper_front",
                part_name="前保险杠",
                part_category="front",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.MODERATE,
                confidence="high",
            ),
            PartActualState(
                part_id="grille_front",
                part_name="前格栅",
                part_category="front",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.MODERATE,
                confidence="medium",
            ),
        ]

        result = compare_topology(topo, states)

        assert "headlight_front_left" in result.missing_parts
        assert "hood" in result.damaged_parts
        assert "bumper_front" in result.damaged_parts
        assert "grille_front" in result.damaged_parts

        # Should trigger safety_critical_missing + regional_mass_damage (front has 4 damaged/missing)
        pattern_ids = [p.pattern_id for p in result.structural_patterns]
        assert "safety_critical_missing" in pattern_ids
        assert "regional_mass_damage" in pattern_ids
        assert result.structural_damage_flag is True
        assert result.overall_severity == "moderate"

    def test_legacy_output_compatibility(self, sample_vehicle_info, sample_vehicle_prior):
        """The DamageAssessment can produce a legacy-compatible dict."""
        topo = build_vehicle_topology(sample_vehicle_info, sample_vehicle_prior)

        states = [
            PartActualState(
                part_id="hood",
                part_name="引擎盖",
                part_category="front",
                side="center",
                status=Status.INTACT,
                damage_level=DamageLevel.NONE,
                confidence="high",
            ),
        ]

        result = compare_topology(topo, states)
        legacy = result.to_legacy_result()

        # Legacy keys must be present
        assert "parts" in legacy
        assert "assessment_summary" in legacy
        assert "structural_damage_flag" in legacy
        assert "structural_damage_reasoning" in legacy

        # New extension keys must also be present
        assert "topology_model" in legacy
        assert "structural_patterns" in legacy
        assert "missing_parts" in legacy
        assert "damaged_parts" in legacy
        assert "intact_parts" in legacy
        assert "uncertain_parts" in legacy
        assert "overall_severity" in legacy
        assert "primary_damage_zone" in legacy

        # parts array should contain frontend-friendly dicts (arrays for
        # multi-value fields so the existing HTML UI can render them).
        assert len(legacy["parts"]) == 33
        hood_legacy = next(p for p in legacy["parts"] if p["part_id"] == "hood")
        assert hood_legacy["status"] == "intact"
        assert hood_legacy["damage_level"] == "none"
        assert hood_legacy["damage_type"] == ["none"]
        assert hood_legacy["evidence_photo"] == []

    def test_empty_actual_states_all_uncertain(self, sample_vehicle_info, sample_vehicle_prior):
        """When no actual states are provided, all 33 nodes are UNCERTAIN."""
        topo = build_vehicle_topology(sample_vehicle_info, sample_vehicle_prior)
        result = compare_topology(topo, [])

        assert len(result.parts) == 33
        assert len(result.uncertain_parts) == 33
        assert result.damaged_parts == []
        assert result.missing_parts == []
        assert result.intact_parts == []
        assert result.structural_damage_flag is False
        assert result.overall_severity == "none"
