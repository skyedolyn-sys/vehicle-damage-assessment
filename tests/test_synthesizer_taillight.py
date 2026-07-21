"""Tests for taillight rear adjacency rule (Rule 7)."""

import pytest
from agents.synthesizer import synthesizer_agent
from models.topology import VehicleTopology, TopologyNode


def _build_rear_collision_topology():
    """Topology for a sedan rear-end collision scene."""
    nodes = {
        "trunk_lid": TopologyNode(
            node_id="trunk_lid", part_id="trunk_lid", node_name="后备箱盖",
            node_type="panel", region="rear", side="center",
            adjacent_nodes=["bumper_rear", "taillight_rear_left", "taillight_rear_right", "windshield_rear"],
        ),
        "bumper_rear": TopologyNode(
            node_id="bumper_rear", part_id="bumper_rear", node_name="后保险杠",
            node_type="panel", region="rear", side="center",
            adjacent_nodes=["trunk_lid", "taillight_rear_left", "taillight_rear_right", "fender_rear_left", "fender_rear_right"],
        ),
        "taillight_rear_left": TopologyNode(
            node_id="taillight_rear_left", part_id="taillight_rear_left", node_name="左后尾灯",
            node_type="light", region="rear", side="rear_left",
            adjacent_nodes=["bumper_rear", "trunk_lid", "fender_rear_left"],
        ),
        "taillight_rear_right": TopologyNode(
            node_id="taillight_rear_right", part_id="taillight_rear_right", node_name="右后尾灯",
            node_type="light", region="rear", side="rear_right",
            adjacent_nodes=["bumper_rear", "trunk_lid", "fender_rear_right"],
        ),
        "fender_rear_right": TopologyNode(
            node_id="fender_rear_right", part_id="fender_rear_right", node_name="右后翼子板",
            node_type="panel", region="right", side="rear_right",
            adjacent_nodes=["door_rear_right", "bumper_rear", "taillight_rear_right"],
        ),
        "fender_rear_left": TopologyNode(
            node_id="fender_rear_left", part_id="fender_rear_left", node_name="左后翼子板",
            node_type="panel", region="left", side="rear_left",
            adjacent_nodes=["door_rear_left", "bumper_rear", "taillight_rear_left"],
        ),
        "windshield_rear": TopologyNode(
            node_id="windshield_rear", part_id="windshield_rear", node_name="后挡风玻璃",
            node_type="glass", region="rear", side="center",
            adjacent_nodes=["trunk_lid", "roof_rear"],
        ),
    }
    return VehicleTopology(
        vehicle_id="sedan-001",
        vehicle_name="Test Sedan",
        nodes=nodes,
        regions={
            "rear": ["trunk_lid", "bumper_rear", "taillight_rear_left", "taillight_rear_right", "windshield_rear"],
            "right": ["fender_rear_right"],
            "left": ["fender_rear_left"],
        },
    )


class TestSynthesizerTaillightAdjacency:
    """Rear taillights adjacent to severe rear damage should not stay intact."""

    def test_taillight_right_intact_adjacent_severe_fender_bumper(self):
        """Single rear_right view says taillight intact, but adjacent fender/bumper are severe -> damaged."""
        topology = _build_rear_collision_topology()
        region_results = [
            {
                "region": "rear_right",
                "parts": [
                    {
                        "part_id": "taillight_rear_right",
                        "part_name": "右后尾灯",
                        "part_category": "rear",
                        "side": "rear_right",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "medium",
                        "evidence_photo": ["photo_01"],
                        "notes": "远端可见",
                    },
                    {
                        "part_id": "fender_rear_right",
                        "part_name": "右后翼子板",
                        "part_category": "right",
                        "side": "rear_right",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "严重变形",
                    },
                    {
                        "part_id": "bumper_rear",
                        "part_name": "后保险杠",
                        "part_category": "rear",
                        "side": "center",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["tear"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "撕裂",
                    },
                    {
                        "part_id": "trunk_lid",
                        "part_name": "后备箱盖",
                        "part_category": "rear",
                        "side": "center",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "塌陷",
                    },
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results, topology=topology)
        light = next(p for p in result["parts"] if p["part_id"] == "taillight_rear_right")
        assert light["status"] == "damaged"
        assert light["damage_level"] == "severe"
        assert light["confidence"] == "low"

    def test_taillight_left_intact_adjacent_severe_fender(self):
        """Left taillight intact but adjacent left rear fender is severe -> damaged."""
        topology = _build_rear_collision_topology()
        region_results = [
            {
                "region": "rear_left",
                "parts": [
                    {
                        "part_id": "taillight_rear_left",
                        "part_name": "左后尾灯",
                        "part_category": "rear",
                        "side": "rear_left",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "medium",
                        "evidence_photo": ["photo_01"],
                        "notes": "远端可见",
                    },
                    {
                        "part_id": "fender_rear_left",
                        "part_name": "左后翼子板",
                        "part_category": "left",
                        "side": "rear_left",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "严重变形",
                    },
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results, topology=topology)
        light = next(p for p in result["parts"] if p["part_id"] == "taillight_rear_left")
        assert light["status"] == "damaged"
        assert light["damage_level"] == "severe"

    def test_taillight_intact_no_severe_neighbor_unchanged(self):
        """Taillight intact with no severe adjacent damage should stay intact."""
        topology = _build_rear_collision_topology()
        region_results = [
            {
                "region": "rear_left",
                "parts": [
                    {
                        "part_id": "taillight_rear_left",
                        "part_name": "左后尾灯",
                        "part_category": "rear",
                        "side": "rear_left",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "清晰可见",
                    },
                    {
                        "part_id": "fender_rear_left",
                        "part_name": "左后翼子板",
                        "part_category": "left",
                        "side": "rear_left",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "完好",
                    },
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results, topology=topology)
        light = next(p for p in result["parts"] if p["part_id"] == "taillight_rear_left")
        assert light["status"] == "intact"
        assert light["damage_level"] == "none"
