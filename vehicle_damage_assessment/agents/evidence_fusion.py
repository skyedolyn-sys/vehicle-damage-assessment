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

from agents.rules import load_part_view_priority


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
    "hood",
    "trunk_lid",
}

# DAMAGE_RECOGNITION_POLICY §2.1: 高敏感部件集合——这些部件的 severe 损伤信号必须穿透 primary-intact。
_HIGH_SENSITIVITY_PARTS = {
    "windshield_front", "windshield_rear", "sunroof_glass",
    "roof_front", "roof_middle", "roof_rear",
    "pillar_a_left", "pillar_a_right",
    "pillar_b_left", "pillar_b_right",
    "pillar_c_left", "pillar_c_right",
    "hood", "trunk_lid",
}

#: Mapping from view id to a (region, side) tag so we can prefer conclusions
#: from a view that is the canonical vantage for a part.
_VIEW_REGION_TAG: Dict[str, str] = {
    "front": "front",
    "front_left": "front_left",
    "front_right": "front_right",
    "rear": "rear",
    "rear_left": "rear_left",
    "rear_right": "rear_right",
    "left": "left",
    "right": "right",
    "top": "roof",
}


#: View priorities for each critical part: lower number = more authoritative.
#: Loaded from agents/rules/config/view_weights.yaml#part_view_priority.
_PART_VIEW_PRIORITY: Dict[str, Dict[str, int]] = load_part_view_priority()


def _view_priority(part_id: str, view_id: str) -> int:
    """Lower priority = more authoritative vantage for this part."""
    table = _PART_VIEW_PRIORITY.get(part_id)
    if not table:
        return 5  # neutral
    return table.get(view_id, 5)


def _as_photo_list(raw: Any) -> List[str]:
    """Normalize an ``evidence_photo`` field to a list of strings.

    Thin backward-compatible alias for :func:`agents.evidence_photo.to_photo_list`.
    All actual shape-conversion logic lives in that module so we have one
    canonical normaliser across the codebase.
    """
    from agents.evidence_photo import to_photo_list
    return to_photo_list(raw)


# Negation prefixes that, when attached to a damage keyword, invert the
# meaning (e.g. "无变形" / "未变形" / "未发现凹陷" all mean NO damage).  These
# are stripped from the notes before keyword matching so the negative form
# does not trigger a false damage signal.
# Negation phrases that invert the meaning of a following damage keyword
# (e.g. "未发现凹陷" / "无明显变形" / "无显著破损" all mean NO damage).
# We replace the WHOLE negated phrase with a neutral token so the damage
# keyword does not survive and trigger a false signal.
_NEGATION_PHRASES = (
    "无明显", "无显著", "无可见", "无发现", "无异常",
    "未见", "未发现", "未观察到", "未出现", "未明显", "未出现明显",
    "没有", "无任何",
)

# Negation adverbs that may appear immediately before a damage keyword
# without an intervening space or character.  These are converted to a
# neutral placeholder so the keyword itself is not matched.
_NEGATION_ADVERBS = ("无", "未", "没有", "不")

# Markers emitted by vision_subagent._enforce_positive_anchor (§2.1) and
# _enforce_edge_visible (§2.2) when an originally-intact high-sensitivity part
# was downgraded to uncertain for missing positive evidence / edge visibility.
# When such a marker is present, the candidate's underlying observation is
# INTACT — not a true UNCERTAIN — and fusion Rule 3 must NOT escalate it back
# to ``damaged`` (mirrors the topology_comparator FP fix for Rule 9/10/11).
_INTACT_ORIGIN_MARKERS = (
    "缺少正向证据",
    "已按政策 §2.1 降级为 uncertain",
    "仅边缘可见(≤1 张证据照片)",
    "已按政策 §2.2 降级为 uncertain",
)


def _has_intact_origin_marker_in_candidate(candidate: Dict[str, Any]) -> bool:
    """Return True if the candidate's notes indicate an intact origin.

    The vision-subagent validation layer (DAMAGE_RECOGNITION_POLICY §2.1 / §2.2)
    downgrades originally-intact high-sensitivity parts to ``uncertain`` when
    the LLM fails to include a positive anchor phrase or only an edge view
    is available.  Those parts retain an "intact origin" marker in their
    notes.  Fusion Rule 3 must respect this signal: the underlying observation
    is intact, so the part must not be escalated to damaged via the
    "one uncertain with strong signal" path.
    """
    notes = (candidate.get("notes", "") or "")
    return any(marker in notes for marker in _INTACT_ORIGIN_MARKERS)


def _strip_negations(notes: str) -> str:
    """Remove Chinese/English negation forms so subsequent keyword matching
    does not flag a sentence describing the ABSENCE of damage.

    DAMAGE_RECOGNITION_POLICY §4.2: P1-B discovered that pillar parts
    reported as "结构完整，无变形、断裂或褶皱" leak through into the
    fusion Rule 3 ("one uncertain with strong signal") path because the
    substring "裂" inside "断裂", "变形" inside the negation "无变形",
    etc. all match the damage keyword list.  We replace the whole
    "无X" / "未X" / "没有X" phrase with a neutral token so the keyword
    matching pass does not count the negated phrase as positive evidence.
    """
    cleaned = notes

    # Build a sorted list of damage keywords so longer keywords are matched
    # before shorter substrings (e.g. "断裂" before "裂").
    damage_keywords = [
        "碎裂", "裂纹", "破裂", "断裂", "撕裂", "蛛网", "破损",
        "变形", "凹陷", "塌陷", "褶皱", "脱落", "缺失", "暴露",
        "翘起", "扭曲", "错位", "crack", "shatter", "broken", "tear",
        "deform", "dent", "missing",
    ]
    damage_keywords.sort(key=len, reverse=True)

    # Scan for negation adverbs and, if a damage keyword appears within a
    # short window after the adverb, replace the whole span with a neutral
    # token.  This handles both simple cases ("无变形") and decorated forms
    # ("未发现凹陷", "未出现明显裂纹", "左侧无明显变形").
    def _replace_negated_span(text: str, start: int, adverb: str) -> str:
        window_end = min(start + len(adverb) + 12, len(text))
        window = text[start + len(adverb):window_end]
        for kw in damage_keywords:
            idx = window.find(kw)
            if idx != -1:
                end = start + len(adverb) + idx + len(kw)
                return text[:start] + "完好" + text[end:]
        # English window can be a bit longer.
        window_end_en = min(start + len(adverb) + 20, len(text))
        window_en = text[start + len(adverb):window_end_en]
        for kw in ("deform", "dent", "crack", "broken", "missing"):
            idx = window_en.find(kw)
            if idx != -1:
                end = start + len(adverb) + idx + len(kw)
                return text[:start] + "intact" + text[end:]
        return text

    # Handle multi-character negation phrases first so they are treated as
    # single units and do not leave damage keywords behind.
    negation_phrases = sorted(_NEGATION_PHRASES, key=len, reverse=True)
    for phrase in negation_phrases:
        start = cleaned.find(phrase)
        while start != -1:
            new_cleaned = _replace_negated_span(cleaned, start, phrase)
            if new_cleaned is cleaned:
                # No damage keyword followed this phrase; remove it anyway
                # so it does not pollute downstream matching.
                new_cleaned = cleaned[:start] + cleaned[start + len(phrase):]
            cleaned = new_cleaned
            start = cleaned.find(phrase)

    # Single-character adverbs (无, 未, 没有, 不).
    for adv in sorted(_NEGATION_ADVERBS, key=len, reverse=True):
        start = cleaned.find(adv)
        while start != -1:
            new_cleaned = _replace_negated_span(cleaned, start, adv)
            if new_cleaned is cleaned:
                new_cleaned = cleaned[:start] + cleaned[start + len(adv):]
            cleaned = new_cleaned
            start = cleaned.find(adv)

    return cleaned


def _has_damage_signal(candidate: Dict[str, Any]) -> bool:
    """Return True if a candidate has any indication of damage in its notes.

    DAMAGE_RECOGNITION_POLICY §4.2: only true positive damage phrases count;
    a sentence like "无变形" (no deformation) must not be treated as a damage
    signal.  We strip common Chinese/English negation forms before keyword
    matching so that pillar parts reported as "结构完整，无变形、断裂或褶皱"
    do not leak through into the fusion "Rule 3: one uncertain with strong
    signal" path.

    DAMAGE_RECOGNITION_POLICY §2.1 / §2.2: an originally-intact part that was
    downgraded to uncertain by the validation layer (carrying the "缺少正向证据"
    or "仅边缘可见" marker) is fundamentally an INTACT observation.  Even if the
    notes still contain a positive damage keyword (e.g. "结构完整，无变形、断裂或
    褶皱" still contains "裂" inside "断裂"), the intact origin must dominate
    and the part must NOT be promoted to damaged by fusion Rule 3.
    """
    if candidate.get("status") in ("damaged", "missing"):
        return True
    # P1-B FP fix (fusion layer): an intact-origin marker means the
    # underlying observation was INTACT, even though the status field now
    # reads "uncertain".  Short-circuit so the negation-stripping /
    # keyword-matching pass below cannot flip this back to damaged.
    if _has_intact_origin_marker_in_candidate(candidate):
        return False
    notes = (candidate.get("notes", "") or "").lower()
    cleaned = _strip_negations(notes)
    keywords = [
        "碎裂", "裂纹", "蛛网", "破损", "破裂", "缺", "撕裂", "塌陷",
        "变形", "凹陷", "褶皱", "脱落", "缺失", "暴露", "翘起", "扭曲",
        "crack", "shatter", "broken", "tear", "deform", "missing", "dent",
    ]
    # "裂" is too short to use after negation stripping because it
    # also appears in many positive damage phrases (裂纹, 破裂, 碎裂).
    # We only require it as a standalone damage signal in the *cleaned*
    # string, which has had the negation forms removed.
    if "裂" in cleaned:
        return True
    return any(kw in cleaned for kw in keywords)


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
        # Rule 1 (DAMAGE_RECOGNITION_POLICY §3.1 / §4.1):
        # primary-intact 仅在 high confidence 时 dominate;高敏感部件 severe 永远穿透。
        primary_priority = _PART_VIEW_PRIORITY.get(part_id, {})
        primary_intact = [
            c for c in intact
            if _view_priority(part_id, c.get("_origin_view", "")) <= 1
        ]
        high_conf_primary_intact = [c for c in primary_intact if c.get("confidence") == "high"]
        is_high_sensitivity = part_id in _HIGH_SENSITIVITY_PARTS
        any_severe_secondary = any(
            c.get("damage_level") in ("moderate", "severe")
            for c in damaged + missing
            if _view_priority(part_id, c.get("_origin_view", "")) > 1
        )
        # §4.1: 高敏感部件 severe 信号永远穿透,即便 primary-intact 是 high confidence。
        if is_high_sensitivity and any(c.get("damage_level") == "severe" for c in damaged + missing):
            for c in damaged + missing:
                c["_policy_override"] = "high_sensitivity_severe_passthrough"
        # §3.1: 仅当 high_conf_primary_intact 且(不是高敏感 或 没有 moderate+ secondary 信号)时才 dominate。
        elif high_conf_primary_intact and not (is_high_sensitivity and any_severe_secondary):
            if not damaged and not missing:
                return None
            # 否则 fall through 到 damaged,但降一档
            for c in damaged + missing:
                if c.get("damage_level") == "severe":
                    c["damage_level"] = "moderate"
                    c["_downgraded_for_low_confidence"] = True
                elif c.get("damage_level") == "moderate":
                    c["damage_level"] = "light"
                    c["_downgraded_for_low_confidence"] = True
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
