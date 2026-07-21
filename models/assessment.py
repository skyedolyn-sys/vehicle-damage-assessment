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
    # Items the synthesizer or reviewer flagged for human follow-up.  These
    # are surfaced on the UI's "不确定项 / 需人工复核" panel and were
    # previously dropped on the default orchestrator path because the
    # legacy view_agent path was the only consumer.
    uncertain_items: List[Dict[str, Any]] = field(default_factory=list)

    def to_legacy_result(self) -> Dict[str, Any]:
        """Return a dict compatible with the old output format.

        Includes old keys (parts, assessment_summary, structural_damage_flag,
        structural_damage_reasoning) plus new keys as extensions.

        The ``parts`` list uses :meth:`PartActualState.to_dict` so that
        the existing HTML frontend can render ``damage_type`` and
        ``evidence_photo`` with ``.join(', ')``.

        ``structural_damage_reasoning`` is emitted as the dict shape used by
        ``output_validator.validate_and_enrich`` (``triggered`` /
        ``rules_matched`` / ``description``) so the orchestrator path is
        shape-compatible with the legacy path.  Earlier versions emitted a
        plain string, which silently broke any consumer expecting the dict.
        """
        legacy_parts = [p.to_dict() for p in self.parts]

        rules_matched: List[str] = []
        for pat in self.structural_patterns:
            rules_matched.append(
                f"{pat.pattern_name}: {pat.description} "
                f"(severity={pat.severity}, confidence={pat.confidence})"
            )
        structural_reasoning = {
            "triggered": self.structural_damage_flag,
            "rules_matched": rules_matched,
            "description": (
                "触发整车结构性事故标记。基于拓扑模式识别检测到结构性损伤模式。"
                if self.structural_damage_flag
                else "未触发整车结构性事故标记。"
            ),
        }

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
            "uncertain_items": list(self.uncertain_items),
            "overall_severity": self.overall_severity,
            "primary_damage_zone": self.primary_damage_zone,
        }
