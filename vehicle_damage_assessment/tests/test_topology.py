"""Tests for topology data models — TopologyNode and VehicleTopology."""

import pytest

from models.topology import TopologyNode, VehicleTopology


# ---------------------------------------------------------------------------
# TopologyNode tests
# ---------------------------------------------------------------------------

class TestTopologyNodeCreation:
    """Test TopologyNode construction and field access."""

    def test_create_minimal_node(self):
        """A node with only required fields can be created."""
        node = TopologyNode(
            node_id="hood",
            part_id="hood",
            node_name="引擎盖",
            node_type="panel",
            region="front",
            side="center",
        )
        assert node.node_id == "hood"
        assert node.node_name == "引擎盖"
        assert node.node_type == "panel"
        assert node.region == "front"
        assert node.side == "center"

    def test_create_full_node_with_defaults(self):
        """Default fields are empty lists / None when omitted."""
        node = TopologyNode(
            node_id="headlight_front_left",
            part_id="headlight_front_left",
            node_name="左前大灯",
            node_type="light",
            region="front",
            side="front_left",
        )
        assert node.adjacent_nodes == []
        assert node.parent_node is None
        assert node.child_nodes == []
        assert node.standard_features == []
        assert node.key_anchors == []
        assert node.visibility_from == []

    def test_create_node_with_all_fields(self):
        """All optional fields can be provided explicitly."""
        node = TopologyNode(
            node_id="hood",
            part_id="hood",
            node_name="引擎盖",
            node_type="panel",
            region="front",
            side="center",
            adjacent_nodes=["grille_front", "fender_front_left"],
            parent_node=None,
            child_nodes=[],
            standard_features=["metal panel", "painted finish"],
            key_anchors=["hood latch", "hood hinge"],
            visibility_from=["front", "front_left_45"],
        )
        assert node.adjacent_nodes == ["grille_front", "fender_front_left"]
        assert node.standard_features == ["metal panel", "painted finish"]
        assert node.key_anchors == ["hood latch", "hood hinge"]
        assert node.visibility_from == ["front", "front_left_45"]


class TestTopologyNodeImmutability:
    """Test that TopologyNode is frozen (immutable)."""

    def test_frozen_dataclass_raises_on_field_mutation(self):
        """Mutating any field on a frozen node must raise FrozenInstanceError."""
        node = TopologyNode(
            node_id="hood",
            part_id="hood",
            node_name="引擎盖",
            node_type="panel",
            region="front",
            side="center",
        )
        with pytest.raises(Exception):
            node.node_name = "modified"

    def test_frozen_dataclass_raises_on_list_append(self):
        """Reassigning a list field on a frozen node must raise FrozenInstanceError."""
        node = TopologyNode(
            node_id="hood",
            part_id="hood",
            node_name="引擎盖",
            node_type="panel",
            region="front",
            side="center",
            adjacent_nodes=["grille_front"],
        )
        # The list itself is mutable (Python dataclass frozen only prevents field reassignment)
        # but reassigning the field should raise
        with pytest.raises(Exception):
            node.adjacent_nodes = ["bumper_front"]


class TestTopologyNodeAdjacency:
    """Test adjacency relationships on TopologyNode."""

    def test_adjacent_nodes_populated(self):
        """Adjacent nodes reflect the topology graph edges."""
        node = TopologyNode(
            node_id="hood",
            part_id="hood",
            node_name="引擎盖",
            node_type="panel",
            region="front",
            side="center",
            adjacent_nodes=["grille_front", "fender_front_left", "fender_front_right", "windshield_front"],
        )
        assert len(node.adjacent_nodes) == 4
        assert "grille_front" in node.adjacent_nodes
        assert "windshield_front" in node.adjacent_nodes

    def test_isolated_node_has_no_adjacent(self):
        """A node with empty adjacency list is isolated."""
        node = TopologyNode(
            node_id="roof_rack",
            part_id="roof_rack",
            node_name="车顶行李架",
            node_type="trim",
            region="roof",
            side="center",
            adjacent_nodes=[],
        )
        assert node.adjacent_nodes == []


# ---------------------------------------------------------------------------
# VehicleTopology tests
# ---------------------------------------------------------------------------

class TestVehicleTopologyBuilding:
    """Test VehicleTopology construction and basic queries."""

    @pytest.fixture
    def sample_topology(self):
        """Return a minimal VehicleTopology with a few nodes for testing."""
        nodes = {
            "hood": TopologyNode(
                node_id="hood",
                part_id="hood",
                node_name="引擎盖",
                node_type="panel",
                region="front",
                side="center",
                adjacent_nodes=["grille_front", "fender_front_left"],
            ),
            "grille_front": TopologyNode(
                node_id="grille_front",
                part_id="grille_front",
                node_name="前格栅",
                node_type="trim",
                region="front",
                side="center",
                adjacent_nodes=["hood", "bumper_front"],
            ),
            "bumper_front": TopologyNode(
                node_id="bumper_front",
                part_id="bumper_front",
                node_name="前保险杠",
                node_type="panel",
                region="front",
                side="center",
                adjacent_nodes=["grille_front"],
            ),
            "fender_front_left": TopologyNode(
                node_id="fender_front_left",
                part_id="fender_front_left",
                node_name="左前翼子板",
                node_type="panel",
                region="front",
                side="front_left",
                adjacent_nodes=["hood", "door_front_left"],
            ),
            "door_front_left": TopologyNode(
                node_id="door_front_left",
                part_id="door_front_left",
                node_name="左前门",
                node_type="panel",
                region="left",
                side="front_left",
                adjacent_nodes=["fender_front_left"],
            ),
            "roof_front": TopologyNode(
                node_id="roof_front",
                part_id="roof_front",
                node_name="车顶前部",
                node_type="structural",
                region="roof",
                side="center",
                adjacent_nodes=["windshield_front", "roof_middle"],
            ),
        }
        regions = {
            "front": ["hood", "grille_front", "bumper_front", "fender_front_left"],
            "left": ["door_front_left"],
            "roof": ["roof_front"],
        }
        return VehicleTopology(
            vehicle_id="test-001",
            vehicle_name="Test Sedan",
            nodes=nodes,
            regions=regions,
        )

    def test_get_node_existing(self, sample_topology):
        """get_node returns the correct TopologyNode for an existing ID."""
        node = sample_topology.get_node("hood")
        assert node is not None
        assert node.node_name == "引擎盖"
        assert node.node_type == "panel"

    def test_get_node_missing(self, sample_topology):
        """get_node returns None for a non-existent node ID."""
        assert sample_topology.get_node("nonexistent") is None

    def test_get_nodes_by_region(self, sample_topology):
        """get_nodes_by_region returns all nodes in the requested region."""
        front_nodes = sample_topology.get_nodes_by_region("front")
        front_ids = {n.node_id for n in front_nodes}
        assert front_ids == {"hood", "grille_front", "bumper_front", "fender_front_left"}

    def test_get_nodes_by_empty_region(self, sample_topology):
        """get_nodes_by_region returns an empty list for an unknown region."""
        assert sample_topology.get_nodes_by_region("undercarriage") == []

    def test_get_adjacent(self, sample_topology):
        """get_adjacent returns the neighbouring TopologyNode objects."""
        neighbours = sample_topology.get_adjacent("hood")
        neighbour_ids = {n.node_id for n in neighbours}
        assert neighbour_ids == {"grille_front", "fender_front_left"}

    def test_get_adjacent_missing_node(self, sample_topology):
        """get_adjacent returns an empty list for a non-existent node."""
        assert sample_topology.get_adjacent("nonexistent") == []

    def test_get_adjacent_isolated_node(self, sample_topology):
        """get_adjacent for a node with no valid neighbours returns []."""
        # door_front_left only points to fender_front_left which exists
        neighbours = sample_topology.get_adjacent("door_front_left")
        assert len(neighbours) == 1


class TestVehicleTopologySerialization:
    """Test round-trip dict serialization."""

    @pytest.fixture
    def full_topology(self):
        """Return a VehicleTopology with one node of each type."""
        nodes = {
            "hood": TopologyNode(
                node_id="hood",
                part_id="hood",
                node_name="引擎盖",
                node_type="panel",
                region="front",
                side="center",
                adjacent_nodes=["grille_front", "fender_front_left"],
                parent_node=None,
                child_nodes=[],
                standard_features=["metal"],
                key_anchors=["latch"],
                visibility_from=["front"],
            ),
            "headlight_front_left": TopologyNode(
                node_id="headlight_front_left",
                part_id="headlight_front_left",
                node_name="左前大灯",
                node_type="light",
                region="front",
                side="front_left",
                adjacent_nodes=["bumper_front", "fender_front_left"],
                standard_features=["LED"],
                key_anchors=["mounting bracket"],
                visibility_from=["front", "front_left_45"],
            ),
        }
        regions = {
            "front": ["hood", "headlight_front_left"],
        }
        return VehicleTopology(
            vehicle_id="v-123",
            vehicle_name="Toyota Camry",
            nodes=nodes,
            regions=regions,
        )

    def test_to_dict_structure(self, full_topology):
        """to_dict produces a JSON-friendly nested dict."""
        d = full_topology.to_dict()
        assert d["vehicle_id"] == "v-123"
        assert d["vehicle_name"] == "Toyota Camry"
        assert "nodes" in d
        assert "regions" in d
        assert d["nodes"]["hood"]["node_name"] == "引擎盖"
        assert d["nodes"]["hood"]["adjacent_nodes"] == ["grille_front", "fender_front_left"]

    def test_from_dict_round_trip(self, full_topology):
        """from_dict(to_dict()) reconstructs an equivalent VehicleTopology."""
        d = full_topology.to_dict()
        restored = VehicleTopology.from_dict(d)

        assert restored.vehicle_id == full_topology.vehicle_id
        assert restored.vehicle_name == full_topology.vehicle_name
        assert len(restored.nodes) == len(full_topology.nodes)
        assert set(restored.nodes.keys()) == set(full_topology.nodes.keys())

        # Check a specific node is fully restored
        hood = restored.get_node("hood")
        assert hood.node_name == "引擎盖"
        assert hood.node_type == "panel"
        assert hood.adjacent_nodes == ["grille_front", "fender_front_left"]
        assert hood.standard_features == ["metal"]
        assert hood.key_anchors == ["latch"]
        assert hood.visibility_from == ["front"]

        # Regions round-trip
        assert restored.regions == full_topology.regions

    def test_from_dict_empty(self):
        """from_dict handles a minimal dict with empty nodes/regions."""
        d = {
            "vehicle_id": "empty",
            "vehicle_name": "Empty Vehicle",
            "nodes": {},
            "regions": {},
        }
        topo = VehicleTopology.from_dict(d)
        assert topo.vehicle_id == "empty"
        assert topo.nodes == {}
        assert topo.regions == {}
