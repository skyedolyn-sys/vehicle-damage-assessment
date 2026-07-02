"""Tests for cross-view pillar/roof propagation rules (Rules 9-11).

Rules 9-11 were moved from the synthesizer to TopologyConsistencyEnforcer in
Phase 3.  These tests exercise the full pipeline: synthesizer merges per-view
evidence, then TopologyComparator applies the topology-driven structural rules.
"""

import pytest
from agents.synthesizer import synthesizer_agent
from agents.topology_comparator import compare_topology
from models.topology import TopologyNode, VehicleTopology


def _build_full_topology():
    """Topology with all parts for pillar/roof propagation tests."""
    adjacency = {
        # front
        "hood": ["grille_front", "fender_front_left", "fender_front_right", "windshield_front"],
        "bumper_front": ["grille_front", "fender_front_left", "fender_front_right", "headlight_front_left", "headlight_front_right"],
        "headlight_front_left": ["bumper_front", "fender_front_left", "grille_front"],
        "headlight_front_right": ["bumper_front", "fender_front_right", "grille_front"],
        "grille_front": ["hood", "bumper_front", "headlight_front_left", "headlight_front_right", "windshield_front"],
        "fender_front_left": ["hood", "bumper_front", "headlight_front_left", "door_front_left", "mirror_left"],
        "fender_front_right": ["hood", "bumper_front", "headlight_front_right", "door_front_right", "mirror_right"],
        "windshield_front": ["hood", "grille_front", "roof_front", "pillar_a_left", "pillar_a_right"],
        # rear
        "trunk_lid": ["bumper_rear", "taillight_rear_left", "taillight_rear_right", "windshield_rear", "roof_rear"],
        "tailgate": ["bumper_rear", "windshield_rear", "roof_rear"],
        "bumper_rear": ["trunk_lid", "tailgate", "taillight_rear_left", "taillight_rear_right", "fender_rear_left", "fender_rear_right"],
        "taillight_rear_left": ["bumper_rear", "trunk_lid", "tailgate", "fender_rear_left"],
        "taillight_rear_right": ["bumper_rear", "trunk_lid", "tailgate", "fender_rear_right"],
        "windshield_rear": ["trunk_lid", "tailgate", "roof_rear", "pillar_c_left", "pillar_c_right"],
        # left
        "door_front_left": ["fender_front_left", "door_rear_left", "mirror_left", "pillar_a_left"],
        "door_rear_left": ["door_front_left", "fender_rear_left", "pillar_b_left", "pillar_c_left"],
        "mirror_left": ["fender_front_left", "door_front_left", "pillar_a_left"],
        "fender_rear_left": ["door_rear_left", "bumper_rear", "taillight_rear_left", "pillar_c_left"],
        "pillar_a_left": ["fender_front_left", "door_front_left", "mirror_left", "roof_front", "windshield_front"],
        "pillar_b_left": ["door_front_left", "door_rear_left", "roof_middle", "pillar_a_left", "pillar_c_left"],
        "pillar_c_left": ["door_rear_left", "fender_rear_left", "roof_rear", "windshield_rear", "pillar_b_left"],
        # right
        "door_front_right": ["fender_front_right", "door_rear_right", "mirror_right", "pillar_a_right"],
        "door_rear_right": ["door_front_right", "fender_rear_right", "pillar_b_right", "pillar_c_right"],
        "mirror_right": ["fender_front_right", "door_front_right", "pillar_a_right"],
        "fender_rear_right": ["door_rear_right", "bumper_rear", "taillight_rear_right", "pillar_c_right"],
        "pillar_a_right": ["fender_front_right", "door_front_right", "mirror_right", "roof_front", "windshield_front"],
        "pillar_b_right": ["door_front_right", "door_rear_right", "roof_middle", "pillar_a_right", "pillar_c_right"],
        "pillar_c_right": ["door_rear_right", "fender_rear_right", "roof_rear", "windshield_rear", "pillar_b_right"],
        # roof
        "roof_front": ["windshield_front", "roof_middle", "sunroof_glass", "pillar_a_left", "pillar_a_right"],
        "roof_middle": ["roof_front", "roof_rear", "sunroof_glass", "roof_rack", "pillar_b_left", "pillar_b_right"],
        "roof_rear": ["roof_middle", "windshield_rear", "trunk_lid", "tailgate", "pillar_c_left", "pillar_c_right"],
        "sunroof_glass": ["roof_front", "roof_middle"],
        "roof_rack": ["roof_middle", "roof_front", "roof_rear"],
    }
    nodes = {}
    for part_id, adj in adjacency.items():
        nodes[part_id] = TopologyNode(
            node_id=part_id,
            part_id=part_id,
            node_name=part_id,
            node_type="structural" if part_id.startswith(("pillar_", "roof_")) else "panel",
            region="front" if part_id in ["hood", "bumper_front", "headlight_front_left", "headlight_front_right", "grille_front", "fender_front_left", "fender_front_right", "windshield_front"] else
                   "rear" if part_id in ["trunk_lid", "tailgate", "bumper_rear", "taillight_rear_left", "taillight_rear_right", "windshield_rear"] else
                   "left" if "_left" in part_id else
                   "right" if "_right" in part_id else
                   "roof",
            side="center",
            adjacent_nodes=adj,
        )
    return VehicleTopology(
        vehicle_id="test-001",
        vehicle_name="Test Vehicle",
        nodes=nodes,
        regions={
            "front": ["hood", "bumper_front", "headlight_front_left", "headlight_front_right", "grille_front", "fender_front_left", "fender_front_right", "windshield_front"],
            "rear": ["trunk_lid", "tailgate", "bumper_rear", "taillight_rear_left", "taillight_rear_right", "windshield_rear"],
            "left": ["door_front_left", "door_rear_left", "mirror_left", "fender_rear_left", "pillar_a_left", "pillar_b_left", "pillar_c_left"],
            "right": ["door_front_right", "door_rear_right", "mirror_right", "fender_rear_right", "pillar_a_right", "pillar_b_right", "pillar_c_right"],
            "roof": ["roof_front", "roof_middle", "roof_rear", "sunroof_glass", "roof_rack"],
        },
    )


def _run_pipeline(region_results, topology):
    """Run synthesizer then topology comparison and return legacy parts list."""
    synth = synthesizer_agent(region_results, topology=topology)
    assessment = compare_topology(topology, synth["part_actual_states"])
    return assessment.to_legacy_result()["parts"]


class TestRule9PillarPropagation:
    """Rule 9: Pillars should be inferred damaged when adjacent structural parts are severe."""

    def test_pillar_a_left_uncertain_with_severe_windshield(self):
        """pillar_a_left uncertain + windshield_front severe -> damaged severe."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "front",
                "parts": [
                    {
                        "part_id": "windshield_front",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["crack"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "碎裂",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "front_left",
                "parts": [
                    {
                        "part_id": "pillar_a_left",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": "遮挡",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        pillar = parts["pillar_a_left"]
        assert pillar["status"] == "damaged"
        assert pillar["damage_level"] == "severe"
        assert "推断" in pillar["notes"]

    def test_pillar_b_left_uncertain_with_severe_roof_middle(self):
        """pillar_b_left uncertain + roof_middle severe -> damaged severe."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "top",
                "parts": [
                    {
                        "part_id": "roof_middle",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "塌陷",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "left",
                "parts": [
                    {
                        "part_id": "pillar_b_left",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": "遮挡",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        pillar = parts["pillar_b_left"]
        assert pillar["status"] == "damaged"
        assert pillar["damage_level"] == "severe"

    def test_pillar_c_right_uncertain_with_severe_pillar_b(self):
        """pillar_c_right uncertain + pillar_b_right severe -> damaged severe."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "right",
                "parts": [
                    {
                        "part_id": "pillar_b_right",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "变形",
                    },
                    {
                        "part_id": "pillar_c_right",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": "遮挡",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        pillar = parts["pillar_c_right"]
        assert pillar["status"] == "damaged"
        assert pillar["damage_level"] == "severe"

    def test_pillar_intact_primary_overrides_propagation(self):
        """Primary view says pillar intact -> keep intact despite severe neighbors."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "front_left",
                "parts": [
                    {
                        "part_id": "pillar_a_left",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "清晰可见",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "front",
                "parts": [
                    {
                        "part_id": "windshield_front",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["crack"],
                        "confidence": "high",
                        "evidence_photo": ["photo_02"],
                        "notes": "碎裂",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        pillar = parts["pillar_a_left"]
        assert pillar["status"] == "intact"

    def test_pillar_no_severe_neighbor_unchanged(self):
        """Pillar uncertain with no severe structural neighbors stays uncertain."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "left",
                "parts": [
                    {
                        "part_id": "pillar_b_left",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": "遮挡",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        pillar = parts["pillar_b_left"]
        assert pillar["status"] == "uncertain"


class TestRule10RoofFrontPropagation:
    """Rule 10: Roof-front should be inferred damaged when front structural parts are severe."""

    def test_roof_front_intact_with_severe_pillar_a(self):
        """roof_front intact (from non-top view) + pillar_a_left severe -> damaged severe."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "front_left",
                "parts": [
                    {
                        "part_id": "pillar_a_left",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "变形",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "front",
                "parts": [
                    {
                        "part_id": "roof_front",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "medium",
                        "evidence_photo": ["photo_02"],
                        "notes": "边缘可见",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        roof = parts["roof_front"]
        assert roof["status"] == "damaged"
        assert roof["damage_level"] == "severe"
        assert "前部结构件" in roof["notes"]

    def test_roof_front_top_intact_overrides_propagation(self):
        """Top view clearly shows roof_front intact -> keep intact."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "front_left",
                "parts": [
                    {
                        "part_id": "pillar_a_left",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "变形",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "top",
                "parts": [
                    {
                        "part_id": "roof_front",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_02"],
                        "notes": "俯视清晰可见",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        roof = parts["roof_front"]
        # Top view intact should override the propagation
        assert roof["status"] == "intact"

    def test_roof_front_no_severe_neighbor_unchanged(self):
        """roof_front intact with no severe front neighbors stays intact."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "top",
                "parts": [
                    {
                        "part_id": "roof_front",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "完好",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        roof = parts["roof_front"]
        assert roof["status"] == "intact"

    def test_roof_front_with_severe_windshield(self):
        """roof_front uncertain + windshield_front severe -> damaged severe."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "front",
                "parts": [
                    {
                        "part_id": "windshield_front",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["crack"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "碎裂",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "top",
                "parts": [
                    {
                        "part_id": "roof_front",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": " unclear",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        roof = parts["roof_front"]
        assert roof["status"] == "damaged"
        assert roof["damage_level"] == "severe"


class TestRule11RoofMiddleRearPropagation:
    """Rule 11: Roof-middle/rear should be inferred damaged when adjacent roof/pillar parts are severe."""

    def test_roof_middle_uncertain_with_severe_roof_front(self):
        """roof_middle uncertain + roof_front severe -> damaged severe."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "top",
                "parts": [
                    {
                        "part_id": "roof_front",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "塌陷",
                    },
                    {
                        "part_id": "roof_middle",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": " unclear",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        roof = parts["roof_middle"]
        assert roof["status"] == "damaged"
        assert roof["damage_level"] == "severe"

    def test_roof_rear_uncertain_with_severe_trunk_lid(self):
        """roof_rear uncertain + trunk_lid severe -> damaged severe."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "rear",
                "parts": [
                    {
                        "part_id": "trunk_lid",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "塌陷",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "top",
                "parts": [
                    {
                        "part_id": "roof_rear",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": " unclear",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        roof = parts["roof_rear"]
        assert roof["status"] == "damaged"
        assert roof["damage_level"] == "severe"

    def test_roof_rear_uncertain_with_severe_pillar_c(self):
        """roof_rear uncertain + pillar_c_left severe (from its primary rear-left view) -> damaged severe."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "rear_left_45",
                "parts": [
                    {
                        "part_id": "pillar_c_left",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "变形",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "top",
                "parts": [
                    {
                        "part_id": "roof_rear",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": " unclear",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        roof = parts["roof_rear"]
        assert roof["status"] == "damaged"
        assert roof["damage_level"] == "severe"

    def test_roof_middle_top_intact_overrides_propagation(self):
        """Top view clearly shows roof_middle intact -> keep intact."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "top",
                "parts": [
                    {
                        "part_id": "roof_middle",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "俯视完好",
                    },
                    {
                        "part_id": "roof_front",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_02"],
                        "notes": "塌陷",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        roof = parts["roof_middle"]
        # Top view intact should override propagation
        assert roof["status"] == "intact"

    def test_roof_rear_intact_no_propagation(self):
        """roof_rear intact (not uncertain) should not trigger propagation even with severe neighbors."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "top",
                "parts": [
                    {
                        "part_id": "roof_rear",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "完好",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "rear",
                "parts": [
                    {
                        "part_id": "trunk_lid",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_02"],
                        "notes": "塌陷",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        roof = parts["roof_rear"]
        # roof_rear is intact, Rule 11 only triggers on "uncertain"
        assert roof["status"] == "intact"

    def test_roof_middle_no_severe_neighbor_unchanged(self):
        """roof_middle uncertain with no severe neighbors stays uncertain."""
        topology = _build_full_topology()
        region_results = [
            {
                "region": "top",
                "parts": [
                    {
                        "part_id": "roof_middle",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": " unclear",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        parts = {p["part_id"]: p for p in _run_pipeline(region_results, topology)}
        roof = parts["roof_middle"]
        assert roof["status"] == "uncertain"
