"""Assessment models — final damage assessment report with topology comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .part_state import PartActualState


@dataclass
class StructuralDamagePattern:
    """A recognised structural damage pattern based on topology analysis."""

    pattern_id: str
    pattern_name: str
    description: str
    matched_nodes: List[str] = field(default_factory=list)
    severity: str = ""
    confidence: str = "medium"


@dataclass
class DamageAssessment:
    """Final vehicle damage assessment report."""

    vehicle_info: Dict[str, Any] = field(default_factory=dict)
    topology_model: Dict[str, Any] = field(default_factory=dict)
    parts: List[PartActualState] = field(default_factory=list)

    missing_parts: List[str] = field(default_factory=list)
    damaged_parts: List[str] = field(default_factory=list)
    intact_parts: List[str] = field(default_factory=list)
    uncertain_parts: List[str] = field(default_factory=list)

    structural_patterns: List[StructuralDamagePattern] = field(default_factory=list)
    structural_damage_flag: bool = False
    overall_severity: str = ""
    primary_damage_zone: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_result(self) -> Dict[str, Any]:
        """Return a dict compatible with the old output format.

        Includes old keys (parts, assessment_summary, structural_damage_flag,
        structural_damage_reasoning) plus new keys as extensions.

        The ``parts`` list uses :meth:`PartActualState.to_dict` so that
        the existing HTML frontend can render ``damage_type`` and
        ``evidence_photo`` with ``.join(', ')``.
        """
        legacy_parts = [p.to_dict() for p in self.parts]

        structural_reasoning = ""
        if self.structural_patterns:
            reasoning_parts = []
            for pat in self.structural_patterns:
                reasoning_parts.append(
                    f"{pat.pattern_name}: {pat.description} "
                    f"(severity={pat.severity}, confidence={pat.confidence})"
                )
            structural_reasoning = "; ".join(reasoning_parts)

        return {
            # Legacy keys
            "parts": legacy_parts,
            "assessment_summary": {
                **self.summary,
                "overall_severity": self.overall_severity,
                "primary_damage_zone": self.primary_damage_zone,
                "structural_damage_flag": self.structural_damage_flag,
                "total_parts": len(self.parts),
                "damaged_parts_count": len(self.damaged_parts),
                "intact_parts_count": len(self.intact_parts),
                "uncertain_parts_count": len(self.uncertain_parts),
                "missing_parts_count": len(self.missing_parts),
            },
            "structural_damage_flag": self.structural_damage_flag,
            "structural_damage_reasoning": structural_reasoning,
            # New extension keys
            "topology_model": self.topology_model,
            "structural_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_name": p.pattern_name,
                    "description": p.description,
                    "matched_nodes": list(p.matched_nodes),
                    "severity": p.severity,
                    "confidence": p.confidence,
                }
                for p in self.structural_patterns
            ],
            "missing_parts": list(self.missing_parts),
            "damaged_parts": list(self.damaged_parts),
            "intact_parts": list(self.intact_parts),
            "uncertain_parts": list(self.uncertain_parts),
            "overall_severity": self.overall_severity,
            "primary_damage_zone": self.primary_damage_zone,
        }
