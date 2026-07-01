"""Topology builder — constructs VehicleTopology from vehicle info and prior."""

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
        Topology graph with only the parts relevant to this vehicle's
        body style and features.  Backward-compatible: if no
        ``vehicle_specs`` is present, defaults to all parts.
    """

    adjacency = PARTS_TOPOLOGY["adjacency"]
    node_types = PARTS_TOPOLOGY["node_types"]
    visibility = PARTS_TOPOLOGY["visibility"]

    prior_topology = vehicle_prior.get("topology", {})
    prior_anchors = vehicle_prior.get("key_anchors", {})

    # Extract vehicle specs — backward-compatible: if no vehicle_specs present,
    # include ALL parts (no filtering)
    raw_specs = vehicle_prior.get("vehicle_specs", {})
    has_specs = isinstance(raw_specs, dict) and raw_specs
    if has_specs:
        specs = VehicleSpecs.from_dict(raw_specs)
        present_part_ids = compute_present_part_ids(specs)
        present_set = set(present_part_ids)
    else:
        # Backward-compatible: no specs means all parts present
        present_set = _ALL_PART_IDS

    nodes: Dict[str, TopologyNode] = {}
    regions: Dict[str, List[str]] = {}

    # Roof sub-region mapping for views that see only an edge of the roof.
    roof_sub_regions: Dict[str, List[str]] = {
        "roof_front": ["roof_front", "sunroof_glass"],
        "roof_middle": ["roof_middle", "sunroof_glass", "roof_rack"],
        "roof_rear": ["roof_rear"],
    }

    for part in PARTS_CATALOG:
        part_id = part["part_id"]
        if part_id not in present_set:
            continue

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

        # Filter adjacency to only present parts
        raw_adjacent = adjacency.get(part_id, [])
        filtered_adjacent = [a for a in raw_adjacent if a in present_set]

        # Filter visibility to only present parts (visibility is independent,
        # but we keep it as-is since it describes camera angles, not parts)
        node = TopologyNode(
            node_id=part_id,
            part_id=part_id,
            node_name=part["part_name"],
            node_type=node_types.get(part_id, "panel"),
            region=region,
            side=side,
            adjacent_nodes=filtered_adjacent,
            standard_features=standard_features,
            key_anchors=key_anchors,
            visibility_from=list(visibility.get(part_id, [])),
        )
        nodes[part_id] = node

        regions.setdefault(region, []).append(part_id)

    # Register roof sub-regions so views can map to them without duplicating nodes.
    for sub_region, part_ids in roof_sub_regions.items():
        filtered_ids = [pid for pid in part_ids if pid in present_set]
        if filtered_ids:
            regions.setdefault(sub_region, []).extend(filtered_ids)
            # Deduplicate while preserving order
            regions[sub_region] = list(dict.fromkeys(regions[sub_region]))

    return VehicleTopology(
        vehicle_id=vehicle_info.get("vehicle_id", "unknown"),
        vehicle_name=vehicle_info.get("vehicle_name", "Unknown Vehicle"),
        nodes=nodes,
        regions=regions,
    )


def topology_to_dict(topology: VehicleTopology) -> Dict[str, Any]:
    """Serialize a VehicleTopology to a plain dict (JSON-friendly)."""
    return topology.to_dict()
