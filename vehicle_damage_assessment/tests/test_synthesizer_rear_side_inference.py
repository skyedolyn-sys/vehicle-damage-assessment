"""Tests for rear-side inference rules (Rule 8)."""

import pytest
from agents.synthesizer import synthesizer_agent
from models.topology import VehicleTopology, TopologyNode


def _build_right_rear_topology():
    """Topology for right-rear collision inference tests."""
    nodes = {
        "taillight_rear_right": TopologyNode(
            node_id="taillight_rear_right", part_id="taillight_rear_right", node_name="右后尾灯",
            node_type="light", region="rear", side="rear_right",
            adjacent_nodes=["bumper_rear", "trunk_lid", "fender_rear_right"],
        ),
        "bumper_rear": TopologyNode(
            node_id="bumper_rear", part_id="bumper_rear", node_name="后保险杠",
            node_type="panel", region="rear", side="center",
            adjacent_nodes=["trunk_lid", "taillight_rear_right", "fender_rear_right"],
        ),
        "trunk_lid": TopologyNode(
            node_id="trunk_lid", part_id="trunk_lid", node_name="后备箱盖",
            node_type="panel", region="rear", side="center",
            adjacent_nodes=["bumper_rear", "taillight_rear_right", "windshield_rear", "roof_rear"],
        ),
        "fender_rear_right": TopologyNode(
            node_id="fender_rear_right", part_id="fender_rear_right", node_name="右后翼子板",
            node_type="panel", region="right", side="rear_right",
            adjacent_nodes=["door_rear_right", "bumper_rear", "taillight_rear_right"],
        ),
        "door_rear_right": TopologyNode(
            node_id="door_rear_right", part_id="door_rear_right", node_name="右后门",
            node_type="panel", region="right", side="rear_right",
            adjacent_nodes=["door_front_right", "fender_rear_right"],
        ),
        "door_front_right": TopologyNode(
            node_id="door_front_right", part_id="door_front_right", node_name="右前门",
            node_type="panel", region="right", side="front_right",
            adjacent_nodes=["fender_front_right", "door_rear_right", "mirror_right"],
        ),
        "mirror_right": TopologyNode(
            node_id="mirror_right", part_id="mirror_right", node_name="右后视镜",
            node_type="glass", region="right", side="front_right",
            adjacent_nodes=["fender_front_right", "door_front_right"],
        ),
        "roof_rear": TopologyNode(
            node_id="roof_rear", part_id="roof_rear", node_name="车顶后部",
            node_type="structural", region="roof", side="center",
            adjacent_nodes=["trunk_lid", "roof_middle"],
        ),
    }
    return VehicleTopology(
        vehicle_id="sedan-001",
        vehicle_name="Test Sedan",
        nodes=nodes,
        regions={
            "rear": ["taillight_rear_right", "bumper_rear", "trunk_lid"],
            "right": ["fender_rear_right", "door_rear_right", "door_front_right", "mirror_right"],
            "roof": ["roof_rear"],
        },
    )


class TestSynthesizerRearSideInference:
    """Rear fenders/doors adjacent to severe rear-core damage should be inferred as damaged."""

    def test_fender_rear_right_uncertain_with_severe_neighbors(self):
        """fender_rear_right uncertain + taillight/bumper severe -> damaged severe."""
        topology = _build_right_rear_topology()
        region_results = [
            {
                "region": "rear_right",
                "parts": [
                    {
                        "part_id": "taillight_rear_right",
                        "part_name": "右后尾灯",
                        "part_category": "rear",
                        "side": "rear_right",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["missing"],
                        "confidence": "low",
                        "evidence_photo": ["photo_01"],
                        "notes": "损毁",
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
                        "part_id": "fender_rear_right",
                        "part_name": "右后翼子板",
                        "part_category": "right",
                        "side": "rear_right",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": "遮挡",
                    },
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results, topology=topology)
        part = next(p for p in result["parts"] if p["part_id"] == "fender_rear_right")
        assert part["status"] == "damaged"
        assert part["damage_level"] == "severe"
        assert part["confidence"] == "low"

    def test_door_rear_right_uncertain_with_severe_neighbors(self):
        """door_rear_right uncertain + fender severe -> damaged severe."""
        topology = _build_right_rear_topology()
        region_results = [
            {
                "region": "right",
                "parts": [
                    {
                        "part_id": "door_rear_right",
                        "part_name": "右后门",
                        "part_category": "right",
                        "side": "rear_right",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": "遮挡",
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
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results, topology=topology)
        part = next(p for p in result["parts"] if p["part_id"] == "door_rear_right")
        assert part["status"] == "damaged"
        assert part["damage_level"] == "severe"

    def test_fender_rear_right_intact_primary_overrides_inference(self):
        """Primary right view says fender intact -> keep intact despite severe neighbors."""
        topology = _build_right_rear_topology()
        region_results = [
            {
                "region": "right",
                "parts": [
                    {
                        "part_id": "fender_rear_right",
                        "part_name": "右后翼子板",
                        "part_category": "right",
                        "side": "rear_right",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "右侧正视角清晰可见",
                    },
                ],
                "uncertain_items": [],
            },
            {
                "region": "rear_right",
                "parts": [
                    {
                        "part_id": "taillight_rear_right",
                        "part_name": "右后尾灯",
                        "part_category": "rear",
                        "side": "rear_right",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["missing"],
                        "confidence": "high",
                        "evidence_photo": ["photo_02"],
                        "notes": "损毁",
                    },
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results, topology=topology)
        part = next(p for p in result["parts"] if p["part_id"] == "fender_rear_right")
        assert part["status"] == "intact"

    def test_intact_status_clears_damage_type(self):
        """When final status is intact, damage_type should be empty."""
        topology = _build_right_rear_topology()
        region_results = [
            {
                "region": "rear_left",
                "parts": [
                    {
                        "part_id": "roof_rear",
                        "part_name": "车顶后部",
                        "part_category": "roof",
                        "side": "center",
                        "status": "damaged",
                        "damage_level": "moderate",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "变形",
                    },
                ],
                "uncertain_items": [],
            },
            {
                "region": "rear",
                "parts": [
                    {
                        "part_id": "roof_rear",
                        "part_name": "车顶后部",
                        "part_category": "roof",
                        "side": "center",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "medium",
                        "evidence_photo": ["photo_02"],
                        "notes": "边缘可见无异常",
                    },
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results, topology=topology)
        part = next(p for p in result["parts"] if p["part_id"] == "roof_rear")
        assert part["status"] == "intact"
        assert part["damage_type"] == []
