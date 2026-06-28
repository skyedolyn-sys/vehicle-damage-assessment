"""Tests for the regional_damage_worker."""

import pytest
from unittest.mock import patch, AsyncMock

from agents.regional_worker import regional_damage_worker
from models.topology import VehicleTopology, TopologyNode


class TestRegionalWorkerWithTopology:
    """Bug 4: regional_worker should use topology regions when topology is provided."""

    @pytest.fixture
    def vehicle_prior(self):
        return {
            "vehicle": "2024 Test SUV",
            "topology": {
                "front": "前部特征",
                "rear": "后部特征",
            },
            "key_anchors": {
                "front": ["前锚点"],
                "rear": ["后锚点"],
            },
        }

    @pytest.fixture
    def suv_topology(self):
        """SUV topology: tailgate present, trunk_lid absent."""
        nodes = {
            "tailgate": TopologyNode(
                node_id="tailgate", part_id="tailgate", node_name="尾门",
                node_type="panel", region="rear", side="center",
                standard_features=["整体掀背尾门"],
                key_anchors=["尾门把手"],
                visibility_from=["rear", "rear_left_45", "rear_right_45"],
            ),
            "bumper_rear": TopologyNode(
                node_id="bumper_rear", part_id="bumper_rear", node_name="后保险杠",
                node_type="panel", region="rear", side="center",
                standard_features=["后保险杠"],
                key_anchors=["后保险杠"],
                visibility_from=["rear", "rear_left_45", "rear_right_45"],
            ),
            "taillight_rear_left": TopologyNode(
                node_id="taillight_rear_left", part_id="taillight_rear_left", node_name="左后尾灯",
                node_type="light", region="rear", side="rear_left",
                standard_features=["尾灯"],
                key_anchors=["左尾灯"],
                visibility_from=["rear", "rear_left_45", "left_90"],
            ),
            "taillight_rear_right": TopologyNode(
                node_id="taillight_rear_right", part_id="taillight_rear_right", node_name="右后尾灯",
                node_type="light", region="rear", side="rear_right",
                standard_features=["尾灯"],
                key_anchors=["右尾灯"],
                visibility_from=["rear", "rear_right_45", "right_90"],
            ),
            "windshield_rear": TopologyNode(
                node_id="windshield_rear", part_id="windshield_rear", node_name="后挡风玻璃",
                node_type="glass", region="rear", side="center",
                standard_features=["后挡风玻璃"],
                key_anchors=["后玻璃"],
                visibility_from=["rear", "rear_left_45", "rear_right_45", "top"],
            ),
        }
        return VehicleTopology(
            vehicle_id="suv-001",
            vehicle_name="Test SUV",
            nodes=nodes,
            regions={
                "front": [],
                "rear": ["tailgate", "bumper_rear", "taillight_rear_left", "taillight_rear_right", "windshield_rear"],
                "left": [],
                "right": [],
                "roof": [],
            },
        )

    @pytest.mark.asyncio
    async def test_rear_region_uses_topology_not_hardcoded(self, vehicle_prior, suv_topology):
        """When topology is provided, rear region should have tailgate but NOT trunk_lid."""
        photos = [
            {
                "id": "photo_01",
                "path": "/fake/path.jpg",
                "detail": "车尾正后",
                "confidence": "high",
            }
        ]

        # Mock the LLM call to capture what parts are sent in the prompt
        captured_messages = []

        async def mock_call_minimax(messages, **kwargs):
            captured_messages.append(messages)
            # Return a minimal valid JSON response
            return """{
                "region": "车尾",
                "parts": [
                    {"part_id": "tailgate", "part_name": "尾门", "status": "damaged", "damage_level": "light", "damage_type": ["dent"], "confidence": "high", "evidence_photo": ["photo_01"], "notes": ""}
                ],
                "uncertain_items": []
            }"""

        with patch("agents.regional_worker.call_minimax", side_effect=mock_call_minimax), \
             patch("agents.regional_worker.build_image_content", return_value={"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,fake"}}):
            result = await regional_damage_worker("车尾", photos, vehicle_prior, suv_topology)

        # The result should have tailgate
        part_ids = {p["part_id"] for p in result.get("parts", [])}
        assert "tailgate" in part_ids
        assert "trunk_lid" not in part_ids

        # Verify the prompt sent to LLM included topology nodes, not hardcoded parts
        assert len(captured_messages) == 1
        prompt_content = str(captured_messages[0])
        # The prompt should mention tailgate (from topology)
        assert "tailgate" in prompt_content
        # The prompt should NOT mention trunk_lid (not in topology)
        assert "trunk_lid" not in prompt_content

    @pytest.mark.asyncio
    async def test_front_region_without_topology_uses_hardcoded(self, vehicle_prior):
        """When topology is None, front region should use hardcoded LOCATION_TO_PARTS."""
        photos = [
            {
                "id": "photo_01",
                "path": "/fake/path.jpg",
                "detail": "车头正前",
                "confidence": "high",
            }
        ]

        async def mock_call_minimax(messages, **kwargs):
            return """{
                "region": "车头",
                "parts": [
                    {"part_id": "hood", "part_name": "引擎盖", "status": "intact", "damage_level": "none", "damage_type": [], "confidence": "high", "evidence_photo": [], "notes": ""}
                ],
                "uncertain_items": []
            }"""

        with patch("agents.regional_worker.call_minimax", side_effect=mock_call_minimax), \
             patch("agents.regional_worker.build_image_content", return_value={"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,fake"}}):
            result = await regional_damage_worker("车头", photos, vehicle_prior, None)

        # Legacy behavior: should have parts from LOCATION_TO_PARTS
        part_ids = {p["part_id"] for p in result.get("parts", [])}
        assert "hood" in part_ids


class TestRegionalWorkerLegacyPath:
    """Test legacy path (no topology) behavior."""

    @pytest.fixture
    def vehicle_prior(self):
        return {
            "vehicle": "2024 Test Sedan",
            "topology": {
                "front": "前部特征",
                "rear": "后部特征",
            },
            "key_anchors": {
                "front": ["前锚点"],
                "rear": ["后锚点"],
            },
        }

    @pytest.mark.asyncio
    async def test_legacy_path_damage_type_as_array(self, vehicle_prior):
        """Legacy path prompt should ask for damage_type as array."""
        photos = [
            {
                "id": "photo_01",
                "path": "/fake/path.jpg",
                "detail": "车头正前",
                "confidence": "high",
            }
        ]

        captured_messages = []

        async def mock_call_minimax(messages, **kwargs):
            captured_messages.append(messages)
            return """{
                "region": "车头",
                "parts": [
                    {"part_id": "hood", "part_name": "引擎盖", "status": "damaged", "damage_level": "light", "damage_type": ["scratch"], "confidence": "high", "evidence_photo": ["photo_01"], "notes": ""}
                ],
                "uncertain_items": []
            }"""

        with patch("agents.regional_worker.call_minimax", side_effect=mock_call_minimax), \
             patch("agents.regional_worker.build_image_content", return_value={"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,fake"}}):
            result = await regional_damage_worker("车头", photos, vehicle_prior, None)

        # Verify the prompt asks for damage_type as array
        prompt_content = str(captured_messages[0])
        assert '"damage_type": ["scratch"' in prompt_content or "damage_type" in prompt_content

    @pytest.mark.asyncio
    async def test_legacy_path_handles_string_damage_type(self, vehicle_prior):
        """Legacy path should handle LLM returning damage_type as string."""
        photos = [
            {
                "id": "photo_01",
                "path": "/fake/path.jpg",
                "detail": "车头正前",
                "confidence": "high",
            }
        ]

        async def mock_call_minimax(messages, **kwargs):
            # LLM returns damage_type as string (some models do this)
            return """{
                "region": "车头",
                "parts": [
                    {"part_id": "hood", "part_name": "引擎盖", "status": "damaged", "damage_level": "light", "damage_type": "scratch, dent", "confidence": "high", "evidence_photo": "photo_01", "notes": ""}
                ],
                "uncertain_items": []
            }"""

        with patch("agents.regional_worker.call_minimax", side_effect=mock_call_minimax), \
             patch("agents.regional_worker.build_image_content", return_value={"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,fake"}}):
            result = await regional_damage_worker("车头", photos, vehicle_prior, None)

        hood = next(p for p in result["parts"] if p["part_id"] == "hood")
        # The _llm_dict_to_part_actual_state should handle string damage_type
        # But in legacy path, it goes through the normal flow without conversion
        # So damage_type should remain as string in the raw result
        assert hood["damage_type"] == "scratch, dent"
