"""Cross-view evidence fusion.

Vision subagents work per-view: each subagent sees a small subset of photos
and may report a part as ``uncertain`` if its visibility from that view is
limited.  But the same part is often visible from multiple views (front,
front_left_45, rear_left_45 all see the roof edges / A-pillars from different
angles).  When several sources independently report damage on the same part,
we should treat that as strong evidence and not let the conservative
"primary-view-only" rule downgrade the conclusion to ``uncertain`` or
``intact``.

This module collects per-part evidence from all subagent outputs and
re-evaluates status/level/confidence for safety-critical parts based on the
multi-source agreement.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from config import PARTS_BY_ID
from models.part_state import DamageLevel, Status


#: Parts that should benefit from cross-view evidence fusion.  These are
#: safety-critical or structural parts whose damage should not be hidden by a
#: single conservative subagent.
_CRITICAL_FUSION_PARTS: Set[str] = {
    "windshield_front",
    "windshield_rear",
    "sunroof_glass",
    "roof_front",
    "roof_middle",
    "roof_rear",
    "headlight_front_left",
    "headlight_front_right",
    "taillight_rear_left",
    "taillight_rear_right",
    "mirror_left",
    "mirror_right",
    "door_front_left",
    "door_front_right",
    "door_rear_left",
    "door_rear_right",
    "pillar_a_left",
    "pillar_a_right",
    "pillar_b_left",
    "pillar_b_right",
    "pillar_c_left",
    "pillar_c_right",
}

#: Mapping from view id to a (region, side) tag so we can prefer conclusions
#: from a view that is the canonical vantage for a part.
_VIEW_REGION_TAG: Dict[str, str] = {
    "front": "front",
    "front_left_45": "front_left",
    "front_right_45": "front_right",
    "rear": "rear",
    "rear_left_45": "rear_left",
    "rear_right_45": "rear_right",
    "left_90": "left",
    "right_90": "right",
    "top": "roof",
}


#: View priorities for each critical part: lower number = more authoritative.
_PART_VIEW_PRIORITY: Dict[str, Dict[str, int]] = {
    "windshield_front": {
        "front": 0, "front_left_45": 1, "front_right_45": 1,
        "rear": 99, "rear_left_45": 99, "rear_right_45": 99,
    },
    "windshield_rear": {
        "rear": 0, "rear_left_45": 1, "rear_right_45": 1,
        "front": 99, "front_left_45": 99, "front_right_45": 99,
    },
    "sunroof_glass": {
        "top": 0, "left_90": 2, "right_90": 2,
        "front": 1, "front_left_45": 1, "front_right_45": 1,
    },
    "roof_front": {
        "front": 0, "front_left_45": 1, "front_right_45": 1,
    },
    "roof_middle": {
        "left_90": 0, "right_90": 0, "top": 0,
        "front": 1, "rear": 1,
    },
    "roof_rear": {
        "rear": 0, "rear_left_45": 1, "rear_right_45": 1,
    },
    "headlight_front_left": {
        "front": 0, "front_left_45": 0, "left_90": 0,
    },
    "headlight_front_right": {
        "front": 0, "front_right_45": 0, "right_90": 0,
    },
    "taillight_rear_left": {
        "rear": 0, "rear_left_45": 0, "left_90": 0,
    },
    "taillight_rear_right": {
        "rear": 0, "rear_right_45": 0, "right_90": 0,
    },
    "mirror_left": {
        "left_90": 0, "front_left_45": 1, "rear_left_45": 1,
    },
    "mirror_right": {
        "right_90": 0, "front_right_45": 1, "rear_right_45": 1,
    },
    "door_front_left": {
        "left_90": 0, "front_left_45": 1, "rear_left_45": 1,
    },
    "door_front_right": {
        "right_90": 0, "front_right_45": 1, "rear_right_45": 1,
    },
    "door_rear_left": {
        "left_90": 0, "rear_left_45": 1, "front_left_45": 1,
    },
    "door_rear_right": {
        "right_90": 0, "rear_right_45": 1, "front_right_45": 1,
    },
}


def _view_priority(part_id: str, view_id: str) -> int:
    """Lower priority = more authoritative vantage for this part."""
    table = _PART_VIEW_PRIORITY.get(part_id)
    if not table:
        return 5  # neutral
    return table.get(view_id, 5)


def _as_photo_list(raw: Any) -> List[str]:
    """Normalize an ``evidence_photo`` field to a list of strings.

    The LLM may emit either a JSON array ``["172852-04.png", "172852-03.png"]``
    or a comma-separated string.  Without this normalization, iterating a raw
    string with ``for p in raw`` would walk it character-by-character.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw or raw == "none":
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(p) for p in raw if p]
    return []


def _has_damage_signal(candidate: Dict[str, Any]) -> bool:
    """Return True if a candidate has any indication of damage in its notes."""
    if candidate.get("status") in ("damaged", "missing"):
        return True
    notes = (candidate.get("notes", "") or "").lower()
    keywords = [
        "碎裂", "裂纹", "裂", "蛛网", "破损", "破裂", "缺", "撕裂", "塌陷",
        "变形", "凹陷", "褶皱", "脱落", "缺失", "暴露", "翘起", "扭曲",
        "crack", "shatter", "broken", "tear", "deform", "missing", "dent",
    ]
    return any(kw in notes for kw in keywords)


def _level_value(level: str) -> int:
    return {
        "none": 0, "unknown": 1, "light": 2, "moderate": 3, "severe": 4,
    }.get(level, 0)


def collect_part_evidence(
    subagent_results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Index per-part candidates from all subagent outputs.

    Each candidate is enriched with the originating view_id.  Duplicate
    (part_id, view_id) candidates are merged by taking the most severe
    observation.
    """
    evidence: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = defaultdict(dict)
    for result in subagent_results:
        view_id = result.get("view_id", "unknown")
        for part in result.get("parts", []):
            if not isinstance(part, dict):
                continue
            part_id = part.get("part_id", "")
            if not part_id:
                continue
            key = (part_id, view_id)
            existing = evidence[part_id].get(key)
            if existing is None:
                evidence[part_id][key] = dict(part, _origin_view=view_id)
                # Normalize photos immediately so downstream code can iterate
                # them safely without re-checking types.
                existing = evidence[part_id][key]
                existing["evidence_photo"] = _as_photo_list(existing.get("evidence_photo"))
                continue
            # Merge: keep the most severe status/level between duplicates.
            existing_level = _level_value(existing.get("damage_level", "unknown"))
            new_level = _level_value(part.get("damage_level", "unknown"))
            if new_level > existing_level:
                existing["damage_level"] = part.get("damage_level", existing.get("damage_level"))
                existing["damage_type"] = part.get("damage_type", existing.get("damage_type"))
            if part.get("status") == "damaged" and existing.get("status") != "damaged":
                existing["status"] = part.get("status")
            # Keep all evidence photos.
            existing.setdefault("evidence_photo", [])
            for p in _as_photo_list(part.get("evidence_photo")):
                if p not in existing["evidence_photo"]:
                    existing["evidence_photo"].append(p)

    return {pid: list(candidates.values()) for pid, candidates in evidence.items()}


def fuse_evidence(
    part_id: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Re-evaluate a critical part from its full multi-view evidence pool.

    Returns a merged candidate dict, or ``None`` when there is not enough
    cross-view evidence to override the existing synthesis.
    """
    if part_id not in _CRITICAL_FUSION_PARTS:
        return None
    if not candidates:
        return None

    # Group by status.
    damaged = [c for c in candidates if c.get("status") == "damaged"]
    missing = [c for c in candidates if c.get("status") == "missing"]
    intact = [c for c in candidates if c.get("status") == "intact"]
    uncertain = [c for c in candidates if c.get("status") == "uncertain"]
    has_signal = [c for c in uncertain if _has_damage_signal(c)]

    # Rule 1: any source reports damage and no primary-view intact contradicts.
    if damaged or missing:
        # If a primary view reports intact, trust it (primary view beats
        # diagonal edge views).  Otherwise prefer damaged/missing.
        primary_priority = _PART_VIEW_PRIORITY.get(part_id, {})
        primary_intact = [
            c for c in intact
            if _view_priority(part_id, c.get("_origin_view", "")) <= 1
        ]
        if primary_intact and not damaged and not missing:
            return None
        chosen = damaged[0] if damaged else missing[0]
        best_level = max(
            (_level_value(c.get("damage_level", "unknown")) for c in damaged + missing),
            default=2,
        )
        return {
            "status": chosen.get("status", "damaged"),
            "damage_level": (
                "severe" if best_level >= 4
                else "moderate" if best_level >= 3
                else "light" if best_level >= 2
                else "unknown"
            ),
            "damage_type": chosen.get("damage_type", []),
            "confidence": "medium",
            "evidence_photo": list(dict.fromkeys(
                p for c in damaged + missing for p in _as_photo_list(c.get("evidence_photo")) if p
            )),
            "_origin_view": chosen.get("_origin_view", ""),
            "_fused": True,
        }

    # Rule 2: multiple uncertain sources with damage signal in notes.
    if len(has_signal) >= 2:
        return {
            "status": "damaged",
            "damage_level": "moderate",
            "damage_type": ["deformation"],
            "confidence": "low",
            "evidence_photo": list(dict.fromkeys(
                p for c in has_signal for p in _as_photo_list(c.get("evidence_photo")) if p
            )),
            "_origin_view": has_signal[0].get("_origin_view", ""),
            "_fused": True,
        }

    # Rule 3: one uncertain with strong signal + no intact primary contradicting.
    if has_signal and not intact:
        return {
            "status": "damaged",
            "damage_level": "moderate",
            "damage_type": ["deformation"],
            "confidence": "low",
            "evidence_photo": list(dict.fromkeys(
                p for c in has_signal for p in _as_photo_list(c.get("evidence_photo")) if p
            )),
            "_origin_view": has_signal[0].get("_origin_view", ""),
            "_fused": True,
        }

    return None


def apply_fusion(
    subagent_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return a dict of fused overrides ``{part_id: merged_candidate}``.

    Only critical parts that pass the fusion rules are returned.  Callers
    can inject these into the synthesizer's evidence pool so that downstream
    consensus no longer treats them as ``uncertain``.
    """
    evidence = collect_part_evidence(subagent_results)
    overrides: Dict[str, Dict[str, Any]] = {}
    for part_id, candidates in evidence.items():
        merged = fuse_evidence(part_id, candidates)
        if merged is not None:
            overrides[part_id] = merged
    return overrides


def merge_uncertain_evidence(uncertain_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop uncertain items that are now covered by a fused override."""
    return list(uncertain_items)
