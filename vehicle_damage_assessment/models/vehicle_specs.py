"""Vehicle specifications model — structured body-style and exterior features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


ALLOWED_BODY_STYLES = {
    "sedan", "hatchback", "suv", "mpv", "van", "pickup", "coupe", "convertible", "wagon"
}

DEFAULT_BODY_STYLE = "sedan"


@dataclass(frozen=True)
class VehicleSpecs:
    """Structured vehicle exterior specifications derived from vehicle_prior output."""

    body_style: str = DEFAULT_BODY_STYLE
    doors: int = 4
    has_sunroof: bool = False
    has_roof_rack: bool = False
    headlight_layout: str = "separate"
    rear_door_type: str = "trunk_lid"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "body_style": self.body_style,
            "doors": self.doors,
            "has_sunroof": self.has_sunroof,
            "has_roof_rack": self.has_roof_rack,
            "headlight_layout": self.headlight_layout,
            "rear_door_type": self.rear_door_type,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VehicleSpecs:
        """Deserialize from a plain dict with normalization and defaults.

        Normalization rules:
        - body_style: lowercased; if not in ALLOWED_BODY_STYLES, defaults to "sedan"
        - doors: coerced to int; default 4
        - has_sunroof / has_roof_rack: coerced to bool; default False
        - rear_door_type: if missing/invalid, inferred from body_style
        """
        if not isinstance(data, dict):
            data = {}

        # body_style
        raw_body = str(data.get("body_style", "")).lower().strip()
        body_style = raw_body if raw_body in ALLOWED_BODY_STYLES else DEFAULT_BODY_STYLE

        # doors
        try:
            doors = int(data.get("doors", 4))
        except (ValueError, TypeError):
            doors = 4

        # booleans
        has_sunroof = bool(data.get("has_sunroof", False))
        has_roof_rack = bool(data.get("has_roof_rack", False))

        # headlight_layout
        headlight_layout = str(data.get("headlight_layout", "separate"))

        # rear_door_type — infer from body_style if missing/empty/invalid
        raw_rear = str(data.get("rear_door_type", "")).strip().lower()
        if raw_rear in ("trunk_lid", "tailgate", "sliding", "none"):
            rear_door_type = raw_rear
        else:
            rear_door_type = cls._infer_rear_door_type(body_style)

        # notes
        notes = str(data.get("notes", ""))

        return cls(
            body_style=body_style,
            doors=doors,
            has_sunroof=has_sunroof,
            has_roof_rack=has_roof_rack,
            headlight_layout=headlight_layout,
            rear_door_type=rear_door_type,
            notes=notes,
        )

    @staticmethod
    def _infer_rear_door_type(body_style: str) -> str:
        """Infer rear_door_type from body_style.

        sedan / coupe / convertible -> "trunk_lid"
        hatchback / suv / mpv / van / pickup / wagon -> "tailgate"
        """
        if body_style in ("sedan", "coupe", "convertible"):
            return "trunk_lid"
        return "tailgate"


def normalize_vehicle_specs(specs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and coerce raw vehicle_specs dict into a normalized dict.

    Rules
    -----
    - body_style: lowercased; must be one of ALLOWED_BODY_STYLES; unknown -> "sedan"
    - doors: coerced to int; default 4
    - has_sunroof / has_roof_rack: coerced to bool; default False
    - rear_door_type: inferred from body_style if missing or empty
    """
    result: Dict[str, Any] = {}

    # body_style
    raw_body = str(specs.get("body_style", "")).lower().strip()
    if raw_body in ALLOWED_BODY_STYLES:
        result["body_style"] = raw_body
    else:
        result["body_style"] = DEFAULT_BODY_STYLE

    # doors
    try:
        result["doors"] = int(specs.get("doors", 4))
    except (ValueError, TypeError):
        result["doors"] = 4

    # booleans
    result["has_sunroof"] = bool(specs.get("has_sunroof", False))
    result["has_roof_rack"] = bool(specs.get("has_roof_rack", False))

    # headlight_layout
    result["headlight_layout"] = str(specs.get("headlight_layout", "separate"))

    # rear_door_type — infer from body_style if missing/empty
    raw_rear = str(specs.get("rear_door_type", "")).strip()
    if raw_rear:
        result["rear_door_type"] = raw_rear
    else:
        body = result["body_style"]
        if body in ("sedan", "coupe", "convertible"):
            result["rear_door_type"] = "trunk_lid"
        else:
            result["rear_door_type"] = "tailgate"

    # notes
    result["notes"] = str(specs.get("notes", ""))

    return result
