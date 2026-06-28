"""Vehicle topology models — standard topology representation for damage assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TopologyNode:
    """A single node in the vehicle topology graph.

    Represents a part or part group, with spatial relationships,
    standard features, and visibility information.
    """

    node_id: str
    part_id: str
    node_name: str
    node_type: str
    region: str
    side: str

    # Topology relations
    adjacent_nodes: List[str] = field(default_factory=list)
    parent_node: Optional[str] = None
    child_nodes: List[str] = field(default_factory=list)

    # Standard-condition features
    standard_features: List[str] = field(default_factory=list)
    key_anchors: List[str] = field(default_factory=list)
    visibility_from: List[str] = field(default_factory=list)


@dataclass
class VehicleTopology:
    """Complete standard-condition topology model for a vehicle."""

    vehicle_id: str
    vehicle_name: str
    nodes: Dict[str, TopologyNode] = field(default_factory=dict)
    regions: Dict[str, List[str]] = field(default_factory=dict)

    def get_node(self, node_id: str) -> Optional[TopologyNode]:
        """Return the node with the given *node_id*, or None."""
        return self.nodes.get(node_id)

    def get_nodes_by_region(self, region: str) -> List[TopologyNode]:
        """Return all nodes that belong to *region*."""
        node_ids = self.regions.get(region, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_adjacent(self, node_id: str) -> List[TopologyNode]:
        """Return the adjacent nodes of *node_id*."""
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [
            self.nodes[adj]
            for adj in node.adjacent_nodes
            if adj in self.nodes
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-friendly)."""
        return {
            "vehicle_id": self.vehicle_id,
            "vehicle_name": self.vehicle_name,
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "part_id": n.part_id,
                    "node_name": n.node_name,
                    "node_type": n.node_type,
                    "region": n.region,
                    "side": n.side,
                    "adjacent_nodes": list(n.adjacent_nodes),
                    "parent_node": n.parent_node,
                    "child_nodes": list(n.child_nodes),
                    "standard_features": list(n.standard_features),
                    "key_anchors": list(n.key_anchors),
                    "visibility_from": list(n.visibility_from),
                }
                for nid, n in self.nodes.items()
            },
            "regions": {
                r: list(node_ids) for r, node_ids in self.regions.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VehicleTopology:
        """Deserialize from a plain dict."""
        nodes = {
            nid: TopologyNode(
                node_id=n["node_id"],
                part_id=n["part_id"],
                node_name=n["node_name"],
                node_type=n["node_type"],
                region=n["region"],
                side=n["side"],
                adjacent_nodes=list(n.get("adjacent_nodes", [])),
                parent_node=n.get("parent_node"),
                child_nodes=list(n.get("child_nodes", [])),
                standard_features=list(n.get("standard_features", [])),
                key_anchors=list(n.get("key_anchors", [])),
                visibility_from=list(n.get("visibility_from", [])),
            )
            for nid, n in data.get("nodes", {}).items()
        }
        regions = {
            r: list(node_ids)
            for r, node_ids in data.get("regions", {}).items()
        }
        return cls(
            vehicle_id=data["vehicle_id"],
            vehicle_name=data["vehicle_name"],
            nodes=nodes,
            regions=regions,
        )
