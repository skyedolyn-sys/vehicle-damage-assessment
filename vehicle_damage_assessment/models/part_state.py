"""Part state models — actual condition of a part compared to standard topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class Status(str, Enum):
    """Comparison status of a part against the standard topology."""

    INTACT = "intact"          # Standard exists, actual exists and normal
    DAMAGED = "damaged"        # Standard exists, actual exists but abnormal
    MISSING = "missing"        # Standard exists, actual does not exist
    UNCERTAIN = "uncertain"    # Cannot determine
    NOT_APPLICABLE = "na"      # Not applicable (standard does not exist)


class DamageLevel(str, Enum):
    """Severity of damage when status is DAMAGED."""

    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


@dataclass
class PartActualState:
    """Actual state of a single part, compared against the standard topology."""

    part_id: str
    part_name: str
    region: str
    side: str

    # Comparison result
    status: Status
    damage_level: DamageLevel
    damage_types: List[str] = field(default_factory=list)

    # Comparison metadata
    standard_exists: bool = True
    actual_visible: bool = False
    actual_present: bool = True

    confidence: str = "low"
    evidence_photos: List[str] = field(default_factory=list)
    notes: str = ""
    adjacent_status: Dict[str, str] = field(default_factory=dict)

    def to_legacy_dict(self) -> Dict[str, str]:
        """Return a flat dict compatible with the old synthesizer / output_validator format.

        Keys: part_id, part_name, part_category, side, status, damage_level,
              damage_type, confidence, evidence_photo, notes
        """
        damage_type_str = (
            ", ".join(self.damage_types) if self.damage_types else "none"
        )
        evidence_photo_str = (
            ", ".join(self.evidence_photos) if self.evidence_photos else ""
        )
        return {
            "part_id": self.part_id,
            "part_name": self.part_name,
            "part_category": self.region,
            "side": self.side,
            "status": self.status.value,
            "damage_level": self.damage_level.value,
            "damage_type": damage_type_str,
            "confidence": self.confidence,
            "evidence_photo": evidence_photo_str,
            "notes": self.notes,
        }

    def to_frontend_dict(self) -> Dict[str, Any]:
        """Return a frontend-friendly dict with arrays for multi-value fields.

        The existing index.html calls ``.join(', ')`` on ``evidence_photo`` and
        ``damage_type``, so these are returned as lists.  ``damage_type`` uses
        ``["none"]`` for intact parts so the column is never empty.
        """
        damage_type_list = list(self.damage_types) if self.damage_types else ["none"]
        evidence_photo_list = list(self.evidence_photos) if self.evidence_photos else []
        return {
            "part_id": self.part_id,
            "part_name": self.part_name,
            "part_category": self.region,
            "side": self.side,
            "status": self.status.value,
            "damage_level": self.damage_level.value,
            "damage_type": damage_type_list,
            "confidence": self.confidence,
            "evidence_photo": evidence_photo_list,
            "notes": self.notes,
        }

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, str]) -> PartActualState:
        """Create a PartActualState from a legacy flat dict."""
        status_str = data.get("status", "uncertain")
        try:
            status = Status(status_str)
        except ValueError:
            status = Status.UNCERTAIN

        damage_level_str = data.get("damage_level", "unknown")
        try:
            damage_level = DamageLevel(damage_level_str)
        except ValueError:
            damage_level = DamageLevel.UNKNOWN

        damage_types = []
        damage_type_raw = data.get("damage_type", "")
        if isinstance(damage_type_raw, list):
            damage_types = [str(t).strip() for t in damage_type_raw if str(t).strip()]
        elif damage_type_raw and damage_type_raw != "none":
            damage_types = [t.strip() for t in damage_type_raw.split(",") if t.strip()]

        evidence_photos = []
        photo_raw = data.get("evidence_photo", "")
        if isinstance(photo_raw, list):
            evidence_photos = [str(p).strip() for p in photo_raw if str(p).strip()]
        elif photo_raw:
            evidence_photos = [p.strip() for p in photo_raw.split(",") if p.strip()]

        return cls(
            part_id=data.get("part_id", ""),
            part_name=data.get("part_name", ""),
            region=data.get("part_category", ""),
            side=data.get("side", ""),
            status=status,
            damage_level=damage_level,
            damage_types=damage_types,
            confidence=data.get("confidence", "low"),
            evidence_photos=evidence_photos,
            notes=data.get("notes", ""),
        )

    @classmethod
    def from_region_part(
        cls,
        part_id: str,
        part_name: str,
        region: str,
        side: str,
        status: Status = Status.UNCERTAIN,
        damage_level: DamageLevel = DamageLevel.UNKNOWN,
    ) -> PartActualState:
        """Factory for creating a PartActualState from region-part info."""
        return cls(
            part_id=part_id,
            part_name=part_name,
            region=region,
            side=side,
            status=status,
            damage_level=damage_level,
        )
