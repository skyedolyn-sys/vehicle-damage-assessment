"""Topology builder — constructs VehicleTopology from vehicle info and prior.

The graph always contains all 33 canonical parts so that prompts and
comparison rules remain stable.  Vehicle specs only influence whether a
part is ``standard_exists`` for this specific vehicle.
"""

from typing import Any, Dict, List

from config import PARTS_CATALOG, PARTS_TOPOLOGY
from models.topology import TopologyNode, VehicleTopology
from models.vehicle_specs import VehicleSpecs


# Parts that are always present regardless of specs
_ALL_PART_IDS = {p["part_id"] for p in PARTS_CATALOG}


def _infer_rear_door_type(body_style: str) -> str:
    """Infer rear_door_type from body_style (fallback when specs unavailable)."""
    if body_style in ("sedan", "coupe", "convertible"):
        return "trunk_lid"
    return "tailgate"


def compute_present_part_ids(specs: VehicleSpecs) -> List[str]:
    """Return the list of part IDs that should be present for this vehicle.

    Rules
    -----
    - Start with all parts in PARTS_CATALOG
    - If rear_door_type == "tailgate": remove "trunk_lid"
    - If rear_door_type == "trunk_lid": remove "tailgate"
    - If rear_door_type is something else (e.g. "sliding", "none"):
        remove both "trunk_lid" and "tailgate"
    - If doors <= 3 OR body_style == "coupe":
        remove "door_rear_left", "door_rear_right"
    - If has_sunroof is False: remove "sunroof_glass"
    - If has_roof_rack is False: remove "roof_rack"
    """
    present = set(_ALL_PART_IDS)

    # Rear door type — explicit rear_door_type takes precedence over body_style
    if specs.rear_door_type == "tailgate":
        present.discard("trunk_lid")
    elif specs.rear_door_type == "trunk_lid":
        present.discard("tailgate")
    elif specs.rear_door_type in ("sliding", "none"):
        present.discard("trunk_lid")
        present.discard("tailgate")
    else:
        # Fallback: infer from body_style
        tailgate_styles = {"hatchback", "suv", "mpv", "van", "pickup", "wagon"}
        if specs.body_style in tailgate_styles:
            present.discard("trunk_lid")
        else:
            present.discard("tailgate")

    # Door count / coupe
    if specs.doors <= 3 or specs.body_style == "coupe":
        present.discard("door_rear_left")
        present.discard("door_rear_right")

    # Sunroof
    if not specs.has_sunroof:
        present.discard("sunroof_glass")

    # Roof rack
    if not specs.has_roof_rack:
        present.discard("roof_rack")

    # Preserve original catalog order
    return [p["part_id"] for p in PARTS_CATALOG if p["part_id"] in present]


def build_vehicle_topology(
    vehicle_info: Dict[str, Any],
    vehicle_prior: Dict[str, Any],
) -> VehicleTopology:
    """Build a standard-condition VehicleTopology from catalog + prior data.

    Parameters
    ----------
    vehicle_info:
        Basic vehicle info (must contain at least ``vehicle_id`` and
        ``vehicle_name``).
    vehicle_prior:
        Output from the vehicle-prior agent. Expected keys:
        ``topology`` (dict region -> text), ``key_anchors``
        (dict region -> list), and optionally ``vehicle_specs`` (dict).

    Returns
    -------
    VehicleTopology
        Topology graph containing all 33 canonical parts.  Vehicle specs
        determine ``standard_exists`` on each node; when no specs are
        available all nodes default to existing.
    """

    adjacency = PARTS_TOPOLOGY["adjacency"]
    node_types = PARTS_TOPOLOGY["node_types"]
    visibility = PARTS_TOPOLOGY["visibility"]

    prior_topology = vehicle_prior.get("topology", {})
    prior_anchors = vehicle_prior.get("key_anchors", {})

    # Extract vehicle specs — backward-compatible: if no vehicle_specs present,
    # assume all parts exist.
    raw_specs = vehicle_prior.get("vehicle_specs", {})
    has_specs = isinstance(raw_specs, dict) and raw_specs
    if has_specs:
        specs = VehicleSpecs.from_dict(raw_specs)
        present_set = set(compute_present_part_ids(specs))
    else:
        present_set = _ALL_PART_IDS

    nodes: Dict[str, TopologyNode] = {}
    regions: Dict[str, List[str]] = {}

    for part in PARTS_CATALOG:
        part_id = part["part_id"]
        standard_exists = part_id in present_set

        region = part["part_category"]
        side = part["side"]

        # Inject standard_features from prior topology text for this region
        standard_features: List[str] = []
        region_text = prior_topology.get(region)
        if isinstance(region_text, str):
            standard_features = [region_text.strip()]
        elif isinstance(region_text, list):
            standard_features = [str(t).strip() for t in region_text]

        # Inject key_anchors from prior for this region
        key_anchors: List[str] = []
        region_anchors = prior_anchors.get(region)
        if isinstance(region_anchors, list):
            key_anchors = [str(a).strip() for a in region_anchors]

        # Adjacency is always built from the full catalog so rules stay stable.
        filtered_adjacent = adjacency.get(part_id, [])

        node = TopologyNode(
            node_id=part_id,
            part_id=part_id,
            node_name=part["part_name"],
            node_type=node_types.get(part_id, "panel"),
            region=region,
            side=side,
            adjacent_nodes=list(filtered_adjacent),
            standard_features=standard_features,
            key_anchors=key_anchors,
            visibility_from=list(visibility.get(part_id, [])),
            standard_exists=standard_exists,
        )
        nodes[part_id] = node

        regions.setdefault(region, []).append(part_id)

    return VehicleTopology(
        vehicle_id=vehicle_info.get("vehicle_id", "unknown"),
        vehicle_name=vehicle_info.get("vehicle_name", "Unknown Vehicle"),
        nodes=nodes,
        regions=regions,
    )


def topology_to_dict(topology: VehicleTopology) -> Dict[str, Any]:
    """Serialize a VehicleTopology to a plain dict (JSON-friendly)."""
    return topology.to_dict()
