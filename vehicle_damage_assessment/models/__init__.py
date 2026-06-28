"""Models package — topology-based damage assessment data structures."""

from .topology import TopologyNode, VehicleTopology
from .part_state import PartActualState, DamageLevel, Status
from .assessment import DamageAssessment, StructuralDamagePattern
from .vehicle_specs import VehicleSpecs

__all__ = [
    "TopologyNode",
    "VehicleTopology",
    "PartActualState",
    "DamageLevel",
    "Status",
    "DamageAssessment",
    "StructuralDamagePattern",
    "VehicleSpecs",
]
