"""Tests for the synthesizer agent."""

import pytest
from agents.synthesizer import synthesizer_agent
from models.topology import VehicleTopology, TopologyNode


class TestSynthesizerDamageTypeParsing:
    """Bug 2: damage_type string should be parsed, not iterated character by character."""

    def test_damage_type_string_comma_separated(self):
        """damage_type='scratch, dent' should produce ['dent', 'scratch'], not ['a', 'c', 'd', 'e', 'h', 'n', 'r', 's', 't']."""
        region_results = [
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "part_name": "引擎盖",
                        "part_category": "front",
                        "side": "center",
                        "status": "damaged",
                        "damage_level": "light",
                        "damage_type": "scratch, dent",
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "轻微划痕和凹陷",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results)
        hood = next(p for p in result["parts"] if p["part_id"] == "hood")
        assert sorted(hood["damage_type"]) == ["dent", "scratch"]

    def test_damage_type_string_none_yields_empty(self):
        """damage_type='none' should produce empty list, not ['e', 'n', 'o']."""
        region_results = [
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "part_name": "引擎盖",
                        "part_category": "front",
                        "side": "center",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": "none",
                        "confidence": "high",
                        "evidence_photo": [],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results)
        hood = next(p for p in result["parts"] if p["part_id"] == "hood")
        assert hood["damage_type"] == []

    def test_damage_type_list_still_works(self):
        """damage_type=['scratch', 'dent'] should continue to work."""
        region_results = [
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "part_name": "引擎盖",
                        "part_category": "front",
                        "side": "center",
                        "status": "damaged",
                        "damage_level": "light",
                        "damage_type": ["scratch", "dent"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results)
        hood = next(p for p in result["parts"] if p["part_id"] == "hood")
        assert sorted(hood["damage_type"]) == ["dent", "scratch"]

    def test_evidence_photo_string_comma_separated(self):
        """evidence_photo='photo_01, photo_02' should produce ['photo_01', 'photo_02']."""
        region_results = [
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "part_name": "引擎盖",
                        "part_category": "front",
                        "side": "center",
                        "status": "damaged",
                        "damage_level": "light",
                        "damage_type": ["scratch"],
                        "confidence": "high",
                        "evidence_photo": "photo_01, photo_02",
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results)
        hood = next(p for p in result["parts"] if p["part_id"] == "hood")
        assert hood["evidence_photo"] == ["photo_01", "photo_02"]


class TestSynthesizerTopologyAware:
    """Bug 3: synthesizer should only output topology nodes when topology is provided."""

    @pytest.fixture
    def sample_topology(self):
        """A topology with only 3 nodes (front region only, no trunk_lid)."""
        nodes = {
            "hood": TopologyNode(
                node_id="hood", part_id="hood", node_name="引擎盖",
                node_type="panel", region="front", side="center",
            ),
            "bumper_front": TopologyNode(
                node_id="bumper_front", part_id="bumper_front", node_name="前保险杠",
                node_type="panel", region="front", side="center",
            ),
            "headlight_front_left": TopologyNode(
                node_id="headlight_front_left", part_id="headlight_front_left", node_name="左前大灯",
                node_type="light", region="front", side="front_left",
            ),
        }
        return VehicleTopology(
            vehicle_id="test-001",
            vehicle_name="Test Vehicle",
            nodes=nodes,
            regions={"front": ["hood", "bumper_front", "headlight_front_left"]},
        )

    def test_with_topology_only_outputs_topology_nodes(self, sample_topology):
        """When topology is provided, only output nodes in topology."""
        region_results = [
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "part_name": "引擎盖",
                        "part_category": "front",
                        "side": "center",
                        "status": "damaged",
                        "damage_level": "light",
                        "damage_type": ["scratch"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results, topology=sample_topology)
        part_ids = {p["part_id"] for p in result["parts"]}
        # Should only have the 3 topology nodes
        assert part_ids == {"hood", "bumper_front", "headlight_front_left"}
        # hood should be damaged (from input)
        hood = next(p for p in result["parts"] if p["part_id"] == "hood")
        assert hood["status"] == "damaged"
        # bumper_front and headlight should be uncertain (no input)
        bumper = next(p for p in result["parts"] if p["part_id"] == "bumper_front")
        assert bumper["status"] == "uncertain"

    def test_without_topology_outputs_all_parts(self):
        """When topology is NOT provided, output all PARTS_BY_ID parts (legacy behavior)."""
        region_results = [
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "part_name": "引擎盖",
                        "part_category": "front",
                        "side": "center",
                        "status": "damaged",
                        "damage_level": "light",
                        "damage_type": ["scratch"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results)
        # Should have all parts from PARTS_BY_ID
        from config import PARTS_BY_ID
        assert len(result["parts"]) == len(PARTS_BY_ID)

    def test_topology_with_tailgate_no_trunk_lid(self):
        """SUV topology should have tailgate but not trunk_lid."""
        nodes = {
            "tailgate": TopologyNode(
                node_id="tailgate", part_id="tailgate", node_name="尾门",
                node_type="panel", region="rear", side="center",
            ),
            "bumper_rear": TopologyNode(
                node_id="bumper_rear", part_id="bumper_rear", node_name="后保险杠",
                node_type="panel", region="rear", side="center",
            ),
        }
        topology = VehicleTopology(
            vehicle_id="suv-001",
            vehicle_name="Test SUV",
            nodes=nodes,
            regions={"rear": ["tailgate", "bumper_rear"]},
        )
        region_results = [
            {
                "region": "车尾",
                "parts": [
                    {
                        "part_id": "tailgate",
                        "part_name": "尾门",
                        "part_category": "rear",
                        "side": "center",
                        "status": "damaged",
                        "damage_level": "moderate",
                        "damage_type": ["dent"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results, topology=topology)
        part_ids = {p["part_id"] for p in result["parts"]}
        assert "tailgate" in part_ids
        assert "bumper_rear" in part_ids
        assert "trunk_lid" not in part_ids


class TestSynthesizerMergeLogic:
    """Test the conservative merge logic (status, level, confidence)."""

    def test_status_takes_most_severe(self):
        """When multiple regions report on same part, status takes worst."""
        region_results = [
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": [],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "status": "damaged",
                        "damage_level": "light",
                        "damage_type": ["scratch"],
                        "confidence": "medium",
                        "evidence_photo": ["photo_02"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results)
        hood = next(p for p in result["parts"] if p["part_id"] == "hood")
        assert hood["status"] == "damaged"  # damaged > intact
        assert hood["damage_level"] == "light"  # light > none
        assert hood["confidence"] == "medium"  # min(high, medium) = medium

    def test_damage_types_merged_and_deduplicated(self):
        """damage_types from multiple candidates should be merged."""
        region_results = [
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "status": "damaged",
                        "damage_level": "light",
                        "damage_type": ["scratch"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "status": "damaged",
                        "damage_level": "moderate",
                        "damage_type": ["scratch", "dent"],  # scratch duplicated
                        "confidence": "medium",
                        "evidence_photo": ["photo_02"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results)
        hood = next(p for p in result["parts"] if p["part_id"] == "hood")
        assert sorted(hood["damage_type"]) == ["dent", "scratch"]

    def test_evidence_photos_merged_and_deduplicated(self):
        """evidence_photos from multiple candidates should be merged without duplicates."""
        region_results = [
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "status": "damaged",
                        "damage_level": "light",
                        "damage_type": ["scratch"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01", "photo_02"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "车头",
                "parts": [
                    {
                        "part_id": "hood",
                        "status": "damaged",
                        "damage_level": "moderate",
                        "damage_type": ["dent"],
                        "confidence": "medium",
                        "evidence_photo": ["photo_02", "photo_03"],  # photo_02 duplicated
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results)
        hood = next(p for p in result["parts"] if p["part_id"] == "hood")
        assert hood["evidence_photo"] == ["photo_01", "photo_02", "photo_03"]


class TestSynthesizerViewWeights:
    """View-weighted conflict resolution for conservative parts."""

    def test_primary_intact_overrides_secondary_damage(self):
        """A primary side view saying intact should override rear-diagonal damage spill-over."""
        region_results = [
            {
                "region": "left",
                "parts": [
                    {
                        "part_id": "door_rear_left",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "rear_left",
                "parts": [
                    {
                        "part_id": "door_rear_left",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["dent"],
                        "confidence": "high",
                        "evidence_photo": ["photo_02"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results)
        door = next(p for p in result["parts"] if p["part_id"] == "door_rear_left")
        assert door["status"] == "intact"
        assert door["damage_level"] == "none"

    def test_primary_damage_kept(self):
        """A primary side view saying damaged should be preserved."""
        region_results = [
            {
                "region": "left",
                "parts": [
                    {
                        "part_id": "door_rear_left",
                        "status": "damaged",
                        "damage_level": "moderate",
                        "damage_type": ["scratch"],
                        "confidence": "medium",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "rear_left",
                "parts": [
                    {
                        "part_id": "door_rear_left",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_02"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results)
        door = next(p for p in result["parts"] if p["part_id"] == "door_rear_left")
        assert door["status"] == "damaged"
        assert door["damage_level"] == "moderate"

    def test_secondary_only_damaged_downgraded(self):
        """Damage reported only from secondary edge views is downgraded one level."""
        region_results = [
            {
                "region": "rear_left",
                "parts": [
                    {
                        "part_id": "door_rear_left",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["dent"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results)
        door = next(p for p in result["parts"] if p["part_id"] == "door_rear_left")
        assert door["status"] == "damaged"
        assert door["damage_level"] == "moderate"
        assert door["confidence"] == "low"

    def test_no_primary_multiple_secondary_agree(self):
        """Without primary coverage, multiple secondary agreeing sources keep severe."""
        region_results = [
            {
                "region": "rear_left",
                "parts": [
                    {
                        "part_id": "door_rear_left",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["dent"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "front_left",
                "parts": [
                    {
                        "part_id": "door_rear_left",
                        "status": "damaged",
                        "damage_level": "moderate",
                        "damage_type": ["scratch"],
                        "confidence": "medium",
                        "evidence_photo": ["photo_02"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results)
        door = next(p for p in result["parts"] if p["part_id"] == "door_rear_left")
        assert door["status"] == "damaged"
        assert door["damage_level"] == "severe"


class TestSynthesizerRoofRules:
    """Roof parts should not inherit severe damage from rear/side views."""

    def test_sunroof_single_primary_damaged_not_severe(self):
        region_results = [
            {
                "region": "left",
                "parts": [
                    {
                        "part_id": "sunroof_glass",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["crack"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results)
        part = next(p for p in result["parts"] if p["part_id"] == "sunroof_glass")
        assert part["status"] == "damaged"
        assert part["damage_level"] == "moderate"

    def test_sunroof_secondary_only_uncertain(self):
        region_results = [
            {
                "region": "rear_left",
                "parts": [
                    {
                        "part_id": "sunroof_glass",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["crack"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results)
        part = next(p for p in result["parts"] if p["part_id"] == "sunroof_glass")
        assert part["status"] == "uncertain"

    def test_roof_rear_from_secondary_only_capped(self):
        region_results = [
            {
                "region": "rear_left",
                "parts": [
                    {
                        "part_id": "roof_rear",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results)
        part = next(p for p in result["parts"] if p["part_id"] == "roof_rear")
        assert part["status"] == "damaged"
        assert part["damage_level"] == "moderate"

    def test_roof_multiple_primary_damaged_trusted(self):
        region_results = [
            {
                "region": "left",
                "parts": [
                    {
                        "part_id": "roof_middle",
                        "status": "damaged",
                        "damage_level": "moderate",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "right",
                "parts": [
                    {
                        "part_id": "roof_middle",
                        "status": "damaged",
                        "damage_level": "moderate",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_02"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results)
        part = next(p for p in result["parts"] if p["part_id"] == "roof_middle")
        assert part["status"] == "damaged"
        assert part["damage_level"] == "moderate"


class TestSynthesizerMirrorRules:
    """Mirror visibility should be resolved conservatively."""

    def test_mirror_single_primary_uncertain_stays_uncertain(self):
        region_results = [
            {
                "region": "left",
                "parts": [
                    {
                        "part_id": "mirror_left",
                        "status": "uncertain",
                        "damage_level": "unknown",
                        "damage_type": [],
                        "confidence": "low",
                        "evidence_photo": [],
                        "notes": "仅远端可见",
                    }
                ],
                "uncertain_items": [],
            }
        ]
        result = synthesizer_agent(region_results)
        part = next(p for p in result["parts"] if p["part_id"] == "mirror_left")
        assert part["status"] == "uncertain"

    def test_mirror_primary_intact_overrides_secondary_damage(self):
        region_results = [
            {
                "region": "left",
                "parts": [
                    {
                        "part_id": "mirror_left",
                        "status": "intact",
                        "damage_level": "none",
                        "damage_type": [],
                        "confidence": "high",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "front_left",
                "parts": [
                    {
                        "part_id": "mirror_left",
                        "status": "damaged",
                        "damage_level": "light",
                        "damage_type": ["scratch"],
                        "confidence": "medium",
                        "evidence_photo": ["photo_02"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results)
        part = next(p for p in result["parts"] if p["part_id"] == "mirror_left")
        assert part["status"] == "intact"


class TestSynthesizerFrontPartRules:
    """Front parts damaged from single corner view with adjacent severe rear damage should be downgraded."""

    def test_bumper_front_false_damage_downgraded(self):
        from models.topology import VehicleTopology, TopologyNode

        nodes = {
            "bumper_front": TopologyNode(
                node_id="bumper_front", part_id="bumper_front", node_name="前保险杠",
                node_type="panel", region="front", side="center",
                adjacent_nodes=["fender_rear_left"],
            ),
            "fender_rear_left": TopologyNode(
                node_id="fender_rear_left", part_id="fender_rear_left", node_name="左后翼子板",
                node_type="panel", region="left", side="rear_left",
                adjacent_nodes=["bumper_front"],
            ),
        }
        topology = VehicleTopology(
            vehicle_id="test-001",
            vehicle_name="Test Vehicle",
            nodes=nodes,
            regions={"front": ["bumper_front"], "left": ["fender_rear_left"]},
        )
        region_results = [
            {
                "region": "front_left",
                "parts": [
                    {
                        "part_id": "bumper_front",
                        "status": "damaged",
                        "damage_level": "moderate",
                        "damage_type": ["scratch"],
                        "confidence": "low",
                        "evidence_photo": ["photo_01"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
            {
                "region": "left",
                "parts": [
                    {
                        "part_id": "fender_rear_left",
                        "status": "damaged",
                        "damage_level": "severe",
                        "damage_type": ["deformation"],
                        "confidence": "high",
                        "evidence_photo": ["photo_02"],
                        "notes": "",
                    }
                ],
                "uncertain_items": [],
            },
        ]
        result = synthesizer_agent(region_results, topology=topology)
        part = next(p for p in result["parts"] if p["part_id"] == "bumper_front")
        assert part["status"] == "damaged"
        assert part["damage_level"] == "light"
        assert part["confidence"] == "low"
