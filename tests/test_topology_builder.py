"""Tests for topology_builder — build_vehicle_topology and helpers."""

import pytest

from agents.topology_builder import build_vehicle_topology, topology_to_dict, compute_present_part_ids
from config import PARTS_CATALOG, PARTS_TOPOLOGY
from models.topology import VehicleTopology
from models.vehicle_specs import VehicleSpecs


class TestBuildVehicleTopology:
    """Test build_vehicle_topology correctness and completeness."""

    @pytest.fixture
    def dummy_vehicle_info(self):
        return {
            "vehicle_id": "v-test-001",
            "vehicle_name": "Test Sedan",
        }

    @pytest.fixture
    def dummy_vehicle_prior(self):
        return {
            "topology": {
                "front": "前部包含引擎盖、保险杠、大灯、格栅、翼子板、挡风玻璃",
                "rear": "后部包含后备箱盖、后保险杠、尾灯、后挡风玻璃",
                "left": "左侧包含前后门、后视镜、后翼子板",
                "right": "右侧包含前后门、后视镜、后翼子板",
                "roof": "车顶包含前中后三部分、天窗玻璃、行李架",
            },
            "key_anchors": {
                "front": ["hood latch", "headlight mounting", "grille emblem"],
                "rear": ["trunk hinge", "taillight socket"],
                "left": ["door handle", "mirror base"],
                "right": ["door handle", "mirror base"],
                "roof": ["sunroof rail", "roof rack mount"],
            },
        }

    def test_returns_vehicle_topology_instance(self, dummy_vehicle_info, dummy_vehicle_prior):
        """The return value is a VehicleTopology dataclass instance."""
        topo = build_vehicle_topology(dummy_vehicle_info, dummy_vehicle_prior)
        assert isinstance(topo, VehicleTopology)

    def test_all_parts_present_when_no_specs(self, dummy_vehicle_info, dummy_vehicle_prior):
        """Backward-compatible: no vehicle_specs means all parts present."""
        topo = build_vehicle_topology(dummy_vehicle_info, dummy_vehicle_prior)
        expected_ids = {p["part_id"] for p in PARTS_CATALOG}
        actual_ids = set(topo.nodes.keys())
        assert actual_ids == expected_ids
        assert len(topo.nodes) == len(PARTS_CATALOG)

    def test_vehicle_info_copied(self, dummy_vehicle_info, dummy_vehicle_prior):
        """vehicle_id and vehicle_name are taken from vehicle_info."""
        topo = build_vehicle_topology(dummy_vehicle_info, dummy_vehicle_prior)
        assert topo.vehicle_id == "v-test-001"
        assert topo.vehicle_name == "Test Sedan"

    def test_regions_grouped_correctly(self, dummy_vehicle_info, dummy_vehicle_prior):
        """Regions dict groups node_ids by their part_category."""
        topo = build_vehicle_topology(dummy_vehicle_info, dummy_vehicle_prior)
        assert "front" in topo.regions
        assert "rear" in topo.regions
        assert "left" in topo.regions
        assert "right" in topo.regions
        assert "roof" in topo.regions

        front_ids = set(topo.regions["front"])
        assert "hood" in front_ids
        assert "bumper_front" in front_ids
        assert "headlight_front_left" in front_ids
        assert len(topo.regions["front"]) == 8

    def test_adjacency_populated_from_parts_topology(self, dummy_vehicle_info, dummy_vehicle_prior):
        """Each node's adjacent_nodes list comes from PARTS_TOPOLOGY adjacency."""
        topo = build_vehicle_topology(dummy_vehicle_info, dummy_vehicle_prior)
        hood = topo.get_node("hood")
        assert hood.adjacent_nodes == PARTS_TOPOLOGY["adjacency"]["hood"]

        bumper = topo.get_node("bumper_front")
        assert bumper.adjacent_nodes == PARTS_TOPOLOGY["adjacency"]["bumper_front"]

    def test_node_types_populated_from_parts_topology(self, dummy_vehicle_info, dummy_vehicle_prior):
        """Each node's node_type comes from PARTS_TOPOLOGY node_types."""
        topo = build_vehicle_topology(dummy_vehicle_info, dummy_vehicle_prior)
        assert topo.get_node("hood").node_type == "panel"
        assert topo.get_node("headlight_front_left").node_type == "light"
        assert topo.get_node("windshield_front").node_type == "glass"
        assert topo.get_node("grille_front").node_type == "trim"
        assert topo.get_node("roof_front").node_type == "structural"

    def test_visibility_populated_from_parts_topology(self, dummy_vehicle_info, dummy_vehicle_prior):
        """Each node's visibility_from comes from PARTS_TOPOLOGY visibility."""
        topo = build_vehicle_topology(dummy_vehicle_info, dummy_vehicle_prior)
        hood_vis = topo.get_node("hood").visibility_from
        assert "front" in hood_vis
        assert "front_left" in hood_vis or "front_left_45" in hood_vis

        roof_vis = topo.get_node("roof_rack").visibility_from
        assert "front" in roof_vis or "top" in roof_vis

    def test_standard_features_injected_from_prior_topology(self, dummy_vehicle_info, dummy_vehicle_prior):
        """standard_features is injected from vehicle_prior['topology'] text per region."""
        topo = build_vehicle_topology(dummy_vehicle_info, dummy_vehicle_prior)
        hood = topo.get_node("hood")
        assert hood.standard_features == ["前部包含引擎盖、保险杠、大灯、格栅、翼子板、挡风玻璃"]

        trunk = topo.get_node("trunk_lid")
        assert trunk.standard_features == ["后部包含后备箱盖、后保险杠、尾灯、后挡风玻璃"]

    def test_key_anchors_injected_from_prior(self, dummy_vehicle_info, dummy_vehicle_prior):
        """key_anchors is injected from vehicle_prior['key_anchors'] per region."""
        topo = build_vehicle_topology(dummy_vehicle_info, dummy_vehicle_prior)
        hood = topo.get_node("hood")
        assert hood.key_anchors == ["hood latch", "headlight mounting", "grille emblem"]

        taillight = topo.get_node("taillight_rear_left")
        assert taillight.key_anchors == ["trunk hinge", "taillight socket"]

    def test_empty_prior_defaults_to_empty_features(self, dummy_vehicle_info):
        """When vehicle_prior has no topology/anchors, features default to empty lists."""
        topo = build_vehicle_topology(dummy_vehicle_info, {})
        hood = topo.get_node("hood")
        assert hood.standard_features == []
        assert hood.key_anchors == []
        # Still has all catalog parts (backward-compatible)
        assert len(topo.nodes) == len(PARTS_CATALOG)

    def test_topology_to_dict_wrapper(self, dummy_vehicle_info, dummy_vehicle_prior):
        """topology_to_dict is a thin wrapper that serializes correctly."""
        topo = build_vehicle_topology(dummy_vehicle_info, dummy_vehicle_prior)
        d = topology_to_dict(topo)
        assert d["vehicle_id"] == "v-test-001"
        assert len(d["nodes"]) == len(PARTS_CATALOG)
        assert "regions" in d


class TestComputePresentPartIds:
    """Test compute_present_part_ids with various vehicle specs."""

    def test_sedan_all_parts(self):
        """Sedan: trunk_lid present, tailgate absent, all doors present."""
        specs = VehicleSpecs(body_style="sedan", doors=4, has_sunroof=True, has_roof_rack=True)
        present = compute_present_part_ids(specs)
        assert "trunk_lid" in present
        assert "tailgate" not in present
        assert "door_rear_left" in present
        assert "door_rear_right" in present
        assert "sunroof_glass" in present
        assert "roof_rack" in present

    def test_suv_tailgate_present(self):
        """SUV: tailgate present, trunk_lid absent."""
        specs = VehicleSpecs(body_style="suv", doors=5, has_sunroof=False, has_roof_rack=False, rear_door_type="tailgate")
        present = compute_present_part_ids(specs)
        assert "tailgate" in present
        assert "trunk_lid" not in present
        assert "door_rear_left" in present
        assert "door_rear_right" in present

    def test_hatchback_tailgate_present(self):
        """Hatchback: tailgate present, trunk_lid absent."""
        specs = VehicleSpecs(body_style="hatchback", doors=5, has_sunroof=False, has_roof_rack=False, rear_door_type="tailgate")
        present = compute_present_part_ids(specs)
        assert "tailgate" in present
        assert "trunk_lid" not in present

    def test_pickup_tailgate_present(self):
        """Pickup: tailgate present, trunk_lid absent."""
        specs = VehicleSpecs(body_style="pickup", doors=2, has_sunroof=False, has_roof_rack=False, rear_door_type="tailgate")
        present = compute_present_part_ids(specs)
        assert "tailgate" in present
        assert "trunk_lid" not in present

    def test_wagon_tailgate_present(self):
        """Wagon: tailgate present, trunk_lid absent."""
        specs = VehicleSpecs(body_style="wagon", doors=5, has_sunroof=False, has_roof_rack=False, rear_door_type="tailgate")
        present = compute_present_part_ids(specs)
        assert "tailgate" in present
        assert "trunk_lid" not in present

    def test_mpv_tailgate_present(self):
        """MPV: tailgate present, trunk_lid absent."""
        specs = VehicleSpecs(body_style="mpv", doors=5, has_sunroof=False, has_roof_rack=False, rear_door_type="tailgate")
        present = compute_present_part_ids(specs)
        assert "tailgate" in present
        assert "trunk_lid" not in present

    def test_van_tailgate_present(self):
        """Van: tailgate present, trunk_lid absent."""
        specs = VehicleSpecs(body_style="van", doors=4, has_sunroof=False, has_roof_rack=False, rear_door_type="tailgate")
        present = compute_present_part_ids(specs)
        assert "tailgate" in present
        assert "trunk_lid" not in present

    def test_coupe_no_rear_doors(self):
        """Coupe: rear doors absent regardless of door count."""
        specs = VehicleSpecs(body_style="coupe", doors=2, has_sunroof=False, has_roof_rack=False)
        present = compute_present_part_ids(specs)
        assert "door_rear_left" not in present
        assert "door_rear_right" not in present
        assert "tailgate" not in present  # coupe -> trunk_lid
        assert "trunk_lid" in present

    def test_2_door_no_rear_doors(self):
        """2-door sedan: rear doors absent."""
        specs = VehicleSpecs(body_style="sedan", doors=2, has_sunroof=False, has_roof_rack=False)
        present = compute_present_part_ids(specs)
        assert "door_rear_left" not in present
        assert "door_rear_right" not in present
        assert "trunk_lid" in present

    def test_3_door_no_rear_doors(self):
        """3-door hatchback: rear doors absent."""
        specs = VehicleSpecs(body_style="hatchback", doors=3, has_sunroof=False, has_roof_rack=False, rear_door_type="tailgate")
        present = compute_present_part_ids(specs)
        assert "door_rear_left" not in present
        assert "door_rear_right" not in present
        assert "tailgate" in present
        assert "trunk_lid" not in present

    def test_4_door_has_rear_doors(self):
        """4-door sedan: rear doors present."""
        specs = VehicleSpecs(body_style="sedan", doors=4, has_sunroof=False, has_roof_rack=False)
        present = compute_present_part_ids(specs)
        assert "door_rear_left" in present
        assert "door_rear_right" in present

    def test_no_sunroof_no_sunroof_glass(self):
        """No sunroof: sunroof_glass absent."""
        specs = VehicleSpecs(body_style="sedan", doors=4, has_sunroof=False, has_roof_rack=True)
        present = compute_present_part_ids(specs)
        assert "sunroof_glass" not in present
        assert "roof_rack" in present

    def test_has_sunroof_has_sunroof_glass(self):
        """Has sunroof: sunroof_glass present."""
        specs = VehicleSpecs(body_style="sedan", doors=4, has_sunroof=True, has_roof_rack=False)
        present = compute_present_part_ids(specs)
        assert "sunroof_glass" in present
        assert "roof_rack" not in present

    def test_no_roof_rack_no_roof_rack(self):
        """No roof rack: roof_rack absent."""
        specs = VehicleSpecs(body_style="sedan", doors=4, has_sunroof=True, has_roof_rack=False)
        present = compute_present_part_ids(specs)
        assert "roof_rack" not in present

    def test_has_roof_rack_has_roof_rack(self):
        """Has roof rack: roof_rack present."""
        specs = VehicleSpecs(body_style="sedan", doors=4, has_sunroof=False, has_roof_rack=True)
        present = compute_present_part_ids(specs)
        assert "roof_rack" in present

    def test_convertible_trunk_lid(self):
        """Convertible: trunk_lid present, tailgate absent."""
        specs = VehicleSpecs(body_style="convertible", doors=2, has_sunroof=False, has_roof_rack=False)
        present = compute_present_part_ids(specs)
        assert "trunk_lid" in present
        assert "tailgate" not in present
        assert "door_rear_left" not in present
        assert "door_rear_right" not in present
        assert "sunroof_glass" not in present
        assert "roof_rack" not in present

    def test_rear_door_type_tailgate_override(self):
        """Explicit rear_door_type='tailgate' overrides sedan default."""
        specs = VehicleSpecs(body_style="sedan", doors=4, rear_door_type="tailgate")
        present = compute_present_part_ids(specs)
        assert "tailgate" in present
        assert "trunk_lid" not in present

    def test_rear_door_type_trunk_lid_override(self):
        """Explicit rear_door_type='trunk_lid' overrides suv default."""
        specs = VehicleSpecs(body_style="suv", doors=5, rear_door_type="trunk_lid")
        present = compute_present_part_ids(specs)
        assert "trunk_lid" in present
        assert "tailgate" not in present

    def test_order_preserved(self):
        """Sedan with all features: 32 parts (all except tailgate), order preserved."""
        specs = VehicleSpecs(body_style="sedan", doors=4, has_sunroof=True, has_roof_rack=True)
        present = compute_present_part_ids(specs)
        catalog_order = [p["part_id"] for p in PARTS_CATALOG]
        # Sedan has trunk_lid, not tailgate
        expected = [p for p in catalog_order if p != "tailgate"]
        assert present == expected
        assert len(present) == 32

    def test_sedan_filtered_order(self):
        """Sedan (trunk_lid, no tailgate) preserves order among remaining parts."""
        specs = VehicleSpecs(body_style="sedan", doors=4, has_sunroof=True, has_roof_rack=True)
        present = compute_present_part_ids(specs)
        catalog_order = [p["part_id"] for p in PARTS_CATALOG]
        # trunk_lid is in, tailgate is out
        assert present == [p for p in catalog_order if p != "tailgate"]


class TestBuildVehicleTopologyWithSpecs:
    """Test topology building with vehicle_specs filtering."""

    def _build_with_specs(self, specs_dict):
        vehicle_info = {"vehicle_id": "v-specs", "vehicle_name": "Specs Test"}
        vehicle_prior = {
            "vehicle_specs": specs_dict,
            "topology": {
                "front": "front features",
                "rear": "rear features",
                "left": "left features",
                "right": "right features",
                "roof": "roof features",
            },
            "key_anchors": {},
        }
        return build_vehicle_topology(vehicle_info, vehicle_prior)

    def test_sedan_topology_has_trunk_lid_no_tailgate(self):
        """Sedan topology: trunk_lid standard_exists, tailgate standard_exists=False."""
        topo = self._build_with_specs({
            "body_style": "sedan", "doors": 4,
            "has_sunroof": False, "has_roof_rack": False,
        })
        assert topo.get_node("trunk_lid").standard_exists is True
        assert topo.get_node("tailgate").standard_exists is False
        # All 33 canonical parts are retained.
        assert len(topo.nodes) == len(PARTS_CATALOG)

    def test_suv_topology_has_tailgate_no_trunk_lid(self):
        """SUV topology: tailgate standard_exists, trunk_lid standard_exists=False."""
        topo = self._build_with_specs({
            "body_style": "suv", "doors": 5,
            "has_sunroof": False, "has_roof_rack": False,
        })
        assert topo.get_node("tailgate").standard_exists is True
        assert topo.get_node("trunk_lid").standard_exists is False
        assert len(topo.nodes) == len(PARTS_CATALOG)

    def test_coupe_topology_no_rear_doors(self):
        """Coupe topology: rear doors standard_exists=False, trunk_lid standard_exists."""
        topo = self._build_with_specs({
            "body_style": "coupe", "doors": 2,
            "has_sunroof": False, "has_roof_rack": False,
        })
        assert topo.get_node("door_rear_left").standard_exists is False
        assert topo.get_node("door_rear_right").standard_exists is False
        assert topo.get_node("trunk_lid").standard_exists is True
        assert topo.get_node("tailgate").standard_exists is False
        assert len(topo.nodes) == len(PARTS_CATALOG)

    def test_2_door_sedan_no_rear_doors(self):
        """2-door sedan: rear doors standard_exists=False."""
        topo = self._build_with_specs({
            "body_style": "sedan", "doors": 2,
            "has_sunroof": False, "has_roof_rack": False,
        })
        assert topo.get_node("door_rear_left").standard_exists is False
        assert topo.get_node("door_rear_right").standard_exists is False
        assert len(topo.nodes) == len(PARTS_CATALOG)

    def test_no_sunroof_no_sunroof_glass(self):
        """No sunroof: sunroof_glass standard_exists=False."""
        topo = self._build_with_specs({
            "body_style": "sedan", "doors": 4,
            "has_sunroof": False, "has_roof_rack": False,
        })
        assert topo.get_node("sunroof_glass").standard_exists is False
        assert len(topo.nodes) == len(PARTS_CATALOG)

    def test_has_sunroof_has_sunroof_glass(self):
        """Has sunroof: sunroof_glass standard_exists=True."""
        topo = self._build_with_specs({
            "body_style": "sedan", "doors": 4,
            "has_sunroof": True, "has_roof_rack": False,
        })
        assert topo.get_node("sunroof_glass").standard_exists is True
        assert len(topo.nodes) == len(PARTS_CATALOG)

    def test_no_roof_rack_no_roof_rack(self):
        """No roof rack: roof_rack standard_exists=False."""
        topo = self._build_with_specs({
            "body_style": "sedan", "doors": 4,
            "has_sunroof": False, "has_roof_rack": False,
        })
        assert topo.get_node("roof_rack").standard_exists is False

    def test_has_roof_rack_has_roof_rack(self):
        """Has roof rack: roof_rack standard_exists=True."""
        topo = self._build_with_specs({
            "body_style": "sedan", "doors": 4,
            "has_sunroof": False, "has_roof_rack": True,
        })
        assert topo.get_node("roof_rack").standard_exists is True

    def test_adjacency_kept_complete(self):
        """Adjacency lists are always complete (new schema: 33 fixed nodes)."""
        topo = self._build_with_specs({
            "body_style": "sedan", "doors": 4,
            "has_sunroof": False, "has_roof_rack": False,
        })
        bumper_rear = topo.get_node("bumper_rear")
        # tailgate is still adjacent even on sedan; it just doesn't exist on this vehicle.
        assert "tailgate" in bumper_rear.adjacent_nodes
        assert "trunk_lid" in bumper_rear.adjacent_nodes

    def test_regions_contain_all_parts(self):
        """Regions contain all catalog parts regardless of specs."""
        topo = self._build_with_specs({
            "body_style": "coupe", "doors": 2,
            "has_sunroof": False, "has_roof_rack": False,
        })
        left_ids = set(topo.regions["left"])
        assert "door_rear_left" in left_ids
        assert "door_front_left" in left_ids
        assert "mirror_left" in left_ids
        assert "fender_rear_left" in left_ids

    def test_rear_region_has_both_trunk_lid_and_tailgate(self):
        """Rear region contains both trunk_lid and tailgate for stability."""
        topo_sedan = self._build_with_specs({
            "body_style": "sedan", "doors": 4,
            "has_sunroof": False, "has_roof_rack": False,
        })
        rear_ids = set(topo_sedan.regions["rear"])
        assert "trunk_lid" in rear_ids
        assert "tailgate" in rear_ids
        assert len(topo_sedan.regions["rear"]) == 6

        topo_suv = self._build_with_specs({
            "body_style": "suv", "doors": 5,
            "has_sunroof": False, "has_roof_rack": False,
        })
        rear_ids_suv = set(topo_suv.regions["rear"])
        assert "tailgate" in rear_ids_suv
        assert "trunk_lid" in rear_ids_suv
        assert len(topo_suv.regions["rear"]) == 6

    def test_roof_region_complete(self):
        """Roof region always contains all roof parts."""
        topo = self._build_with_specs({
            "body_style": "sedan", "doors": 4,
            "has_sunroof": False, "has_roof_rack": False,
        })
        roof_ids = set(topo.regions["roof"])
        assert "sunroof_glass" in roof_ids
        assert "roof_rack" in roof_ids
        assert "roof_front" in roof_ids
        assert "roof_middle" in roof_ids
        assert "roof_rear" in roof_ids
        assert len(topo.regions["roof"]) == 5

    def test_topology_to_dict_with_specs(self):
        """topology_to_dict works with the fixed 33-node topology."""
        topo = self._build_with_specs({
            "body_style": "coupe", "doors": 2,
            "has_sunroof": False, "has_roof_rack": False,
        })
        d = topology_to_dict(topo)
        assert d["vehicle_id"] == "v-specs"
        assert "tailgate" in d["nodes"]
        assert "trunk_lid" in d["nodes"]
        assert "door_rear_left" in d["nodes"]
        assert "regions" in d

    def test_backward_compatible_no_specs(self):
        """Without vehicle_specs, all parts are present (backward compatible)."""
        vehicle_info = {"vehicle_id": "v-compat", "vehicle_name": "Compat"}
        vehicle_prior = {
            "topology": {"front": "f", "rear": "r", "left": "l", "right": "r", "roof": "ro"},
            "key_anchors": {},
        }
        topo = build_vehicle_topology(vehicle_info, vehicle_prior)
        assert len(topo.nodes) == len(PARTS_CATALOG)
        assert "trunk_lid" in topo.nodes
        assert "tailgate" in topo.nodes
        assert "door_rear_left" in topo.nodes
        assert "door_rear_right" in topo.nodes
        assert "sunroof_glass" in topo.nodes
        assert "roof_rack" in topo.nodes
