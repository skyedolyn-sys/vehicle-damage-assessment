"""Damage overview generator — produces a professional, human-readable summary
of the vehicle damage condition based on topology-aware assessment results.

The overview helps users quickly understand the overall damage pattern and
provides a concise reference for human reviewers to judge whether the sample
matches expectations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from config import PARTS_BY_ID
from models.part_state import DamageLevel, PartActualState, Status


# Region display names (Chinese)
_REGION_NAMES: Dict[str, str] = {
    "front": "前部",
    "rear": "后部",
    "left": "左侧",
    "right": "右侧",
    "roof": "车顶",
}

# Severity display names
_SEVERITY_NAMES: Dict[str, str] = {
    "none": "无明显损伤",
    "light": "轻微损伤",
    "moderate": "中度损伤",
    "severe": "严重损伤",
    "unknown": "未知",
}

# Accident type inference rules based on damaged regions, structural and
# glass damage signals.  Each rule returns a dict with ``type_id`` and
# ``type_name``; first match wins.
#
# The signature accepts the damaged region set plus a damage-signals object
# describing damaged/sever parts and structural/glass fragments.  This lets
# us distinguish a side-collision (only side parts damaged) from a rollover
# (roof + glass + side + sometimes A-pillar, with multiple roof sections).
def _accident_has_roof_collapse(parts: List[PartActualState]) -> bool:
    return sum(
        1 for p in parts
        if p.part_id in ("roof_front", "roof_middle", "roof_rear")
        and p.status in (Status.DAMAGED, Status.MISSING)
    ) >= 2


def _accident_multiple_glass(parts: List[PartActualState]) -> bool:
    glass = {"windshield_front", "windshield_rear", "sunroof_glass"}
    return sum(
        1 for p in parts
        if p.part_id in glass
        and p.status in (Status.DAMAGED, Status.MISSING)
    ) >= 2


def _accident_single_glass_damaged(parts: List[PartActualState]) -> bool:
    glass = {"windshield_front", "windshield_rear", "sunroof_glass"}
    return any(
        p.part_id in glass
        and p.status in (Status.DAMAGED, Status.MISSING)
        for p in parts
    )


def _accident_has_a_pillar_damage(parts: List[PartActualState]) -> bool:
    pillar_ids = {"pillar_a_left", "pillar_a_right", "pillar_b_left",
                  "pillar_b_right", "pillar_c_left", "pillar_c_right"}
    return any(
        p.part_id in pillar_ids
        and p.status in (Status.DAMAGED, Status.MISSING)
        for p in parts
    )


def _accident_left_only_collision(regions: Set[str], parts: List[PartActualState]) -> bool:
    if regions != {"left"}:
        return False
    # Distinguish side collision from rollover: pure left with no roof/glass.
    return not _accident_has_roof_collapse(parts) and not _accident_single_glass_damaged(parts)


def _accident_right_only_collision(regions: Set[str], parts: List[PartActualState]) -> bool:
    if regions != {"right"}:
        return False
    return not _accident_has_roof_collapse(parts) and not _accident_single_glass_damaged(parts)


def _accident_pure_front(regions: Set[str]) -> bool:
    return regions == {"front"} or (
        regions.issubset({"front"}) and "rear" not in regions
        and "left" not in regions and "right" not in regions and "roof" not in regions
    )


def _accident_pure_rear(regions: Set[str]) -> bool:
    return regions == {"rear"} or (
        regions.issubset({"rear"}) and "front" not in regions
        and "left" not in regions and "right" not in regions and "roof" not in regions
    )


def _infer_accident_type_impl(regions: Set[str], parts: List[PartActualState]) -> Dict[str, str]:
    """Resolve the accident type from regions + structural/glass signals."""
    if not regions:
        return {"type_id": "none", "type_name": "无明显碰撞痕迹"}

    # Roll-over signatures take priority because they imply a stronger event
    # than a side or front-end collision alone.
    if _accident_has_roof_collapse(parts) and (
        _accident_multiple_glass(parts) or len(regions) >= 2 or "left" in regions or "right" in regions
    ):
        return {"type_id": "rollover", "type_name": "翻滚/顶部受压"}

    # Multiple glass panels shattered + damaged regions on more than one axis
    # also suggests a rollover or severe structural event.
    if _accident_multiple_glass(parts) and len(regions) >= 2:
        return {"type_id": "rollover", "type_name": "翻滚/顶部受压"}

    if _accident_pure_front(regions):
        return {"type_id": "front_collision", "type_name": "前部碰撞"}

    if _accident_pure_rear(regions):
        return {"type_id": "rear_collision", "type_name": "后部碰撞"}

    if _accident_left_only_collision(regions, parts):
        return {"type_id": "side_collision_left", "type_name": "左侧碰撞"}

    if _accident_right_only_collision(regions, parts):
        return {"type_id": "side_collision_right", "type_name": "右侧碰撞"}

    # Roof-only — when no other region has damage but the roof is damaged
    # we treat it as roof impact (e.g. falling object, hail crush).
    if regions == {"roof"}:
        return {"type_id": "roof_impact", "type_name": "车顶受损（翻滚/受压/落物）"}

    return {"type_id": "multi_region", "type_name": "多区域复合碰撞"}


def _infer_accident_type(parts: List[PartActualState]) -> Dict[str, Any]:
    """Infer the most likely accident type from damaged regions and parts."""
    damaged_regions: Set[str] = set()
    for part in parts:
        if part.status in (Status.DAMAGED, Status.MISSING):
            damaged_regions.add(part.region)
    return _infer_accident_type_impl(damaged_regions, parts)


# Safety-critical part IDs and their display names.
_SAFETY_CRITICAL_PARTS: Dict[str, str] = {
    "headlight_front_left": "左前大灯",
    "headlight_front_right": "右前大灯",
    "taillight_rear_left": "左后尾灯",
    "taillight_rear_right": "右后尾灯",
    "mirror_left": "左后视镜",
    "mirror_right": "右后视镜",
    "windshield_front": "前挡风玻璃",
    "windshield_rear": "后挡风玻璃",
    "sunroof_glass": "天窗玻璃",
}

# Structural part IDs.
_STRUCTURAL_PARTS: Set[str] = {"roof_front", "roof_middle", "roof_rear"}


def _region_name(region: str) -> str:
    return _REGION_NAMES.get(region, region)


def _severity_name(level: str) -> str:
    return _SEVERITY_NAMES.get(level, level)


def _count_by_region(parts: List[PartActualState], status: Status) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for part in parts:
        if part.status == status:
            counts[part.region] = counts.get(part.region, 0) + 1
    return counts


def _max_level_in_region(parts: List[PartActualState], region: str) -> Optional[str]:
    """Return the highest damage level in a given region."""
    level_priority = {"none": 0, "unknown": 1, "light": 2, "moderate": 3, "severe": 4}
    best_level: Optional[str] = None
    best_score = -1
    for part in parts:
        if part.region != region or part.status != Status.DAMAGED:
            continue
        score = level_priority.get(part.damage_level.value, 0)
        if score > best_score:
            best_score = score
            best_level = part.damage_level.value
    return best_level


def _infer_accident_type(parts: List[PartActualState]) -> Dict[str, Any]:
    """Infer the most likely accident type from damaged regions and parts."""
    damaged_regions: Set[str] = set()
    for part in parts:
        if part.status in (Status.DAMAGED, Status.MISSING):
            damaged_regions.add(part.region)
    return _infer_accident_type_impl(damaged_regions, parts)


def _build_region_summary(parts: List[PartActualState]) -> List[Dict[str, Any]]:
    """Summarize damage per region with counts, severity and representative parts."""
    region_data: Dict[str, Dict[str, Any]] = {}
    for part in parts:
        if part.status not in (Status.DAMAGED, Status.MISSING):
            continue
        region = part.region
        if region not in region_data:
            region_data[region] = {"damaged_count": 0, "missing_count": 0, "max_level": "none", "parts": []}
        if part.status == Status.MISSING:
            region_data[region]["missing_count"] += 1
        else:
            region_data[region]["damaged_count"] += 1
        region_data[region]["parts"].append(part.part_name)

    for region, data in region_data.items():
        data["max_level"] = _max_level_in_region(parts, region) or "none"

    # Sort by severity then by part count.
    level_priority = {"severe": 0, "moderate": 1, "light": 2, "none": 3, "unknown": 4}
    sorted_regions = sorted(
        region_data.items(),
        key=lambda item: (level_priority.get(item[1]["max_level"], 99), -(item[1]["damaged_count"] + item[1]["missing_count"])),
    )

    return [
        {
            "region": region,
            "region_name": _region_name(region),
            "damaged_count": data["damaged_count"],
            "missing_count": data["missing_count"],
            "max_level": data["max_level"],
            "max_level_name": _severity_name(data["max_level"]),
            "representative_parts": data["parts"][:5],
        }
        for region, data in sorted_regions
    ]


def _build_safety_summary(parts: List[PartActualState]) -> Dict[str, Any]:
    """Summarize status of safety-critical parts.

    Only count a safety-critical part as "affected" when it is genuinely
    damaged or missing. Intact or uncertain parts are listed separately so
    the overview does not overstate the condition.
    """
    affected: List[Dict[str, str]] = []
    uncertain: List[str] = []
    for part in parts:
        if part.part_id not in _SAFETY_CRITICAL_PARTS:
            continue
        if part.status in (Status.DAMAGED, Status.MISSING):
            affected.append(
                {
                    "part_id": part.part_id,
                    "part_name": _SAFETY_CRITICAL_PARTS[part.part_id],
                    "status": part.status.value,
                    "damage_level": part.damage_level.value,
                }
            )
        elif part.status == Status.UNCERTAIN:
            uncertain.append(_SAFETY_CRITICAL_PARTS[part.part_id])
    return {
        "affected_count": len(affected),
        "affected_parts": affected,
        "uncertain_count": len(uncertain),
        "uncertain_parts": uncertain,
    }


def _build_structural_summary(parts: List[PartActualState]) -> Dict[str, Any]:
    """Summarize structural component damage.

    Reports genuinely damaged/missing structural parts and lists uncertain
    structural parts separately for transparency.
    """
    affected: List[Dict[str, str]] = []
    uncertain: List[str] = []
    for part in parts:
        if part.part_id not in _STRUCTURAL_PARTS:
            continue
        if part.status in (Status.DAMAGED, Status.MISSING):
            affected.append(
                {
                    "part_id": part.part_id,
                    "part_name": PARTS_BY_ID.get(part.part_id, {}).get("part_name", part.part_id),
                    "status": part.status.value,
                    "damage_level": part.damage_level.value,
                }
            )
        elif part.status == Status.UNCERTAIN:
            uncertain.append(PARTS_BY_ID.get(part.part_id, {}).get("part_name", part.part_id))
    return {
        "affected_count": len(affected),
        "structural_damage_flag": len(affected) > 0,
        "affected_parts": affected,
        "uncertain_count": len(uncertain),
        "uncertain_parts": uncertain,
    }


def _detect_symmetric_damage(parts: List[PartActualState]) -> Dict[str, Any]:
    """Detect whether left/right symmetric parts are both damaged/missing."""
    symmetric_pairs = [
        ("headlight_front_left", "headlight_front_right", "前大灯"),
        ("taillight_rear_left", "taillight_rear_right", "后尾灯"),
        ("fender_front_left", "fender_front_right", "前翼子板"),
        ("fender_rear_left", "fender_rear_right", "后翼子板"),
        ("door_front_left", "door_front_right", "前车门"),
        ("door_rear_left", "door_rear_right", "后车门"),
        ("mirror_left", "mirror_right", "后视镜"),
    ]
    symmetric_damaged: List[Dict[str, str]] = []
    for left_id, right_id, group_name in symmetric_pairs:
        left_status = next((p.status for p in parts if p.part_id == left_id), None)
        right_status = next((p.status for p in parts if p.part_id == right_id), None)
        if left_status in (Status.DAMAGED, Status.MISSING) and right_status in (Status.DAMAGED, Status.MISSING):
            symmetric_damaged.append(
                {
                    "group_name": group_name,
                    "left_part_id": left_id,
                    "right_part_id": right_id,
                }
            )
    return {"symmetric_damage_detected": len(symmetric_damaged) > 0, "pairs": symmetric_damaged}


def _coverage_assessment(parts: List[PartActualState]) -> Dict[str, Any]:
    """Assess photo/evidence coverage quality from part states."""
    total = len(parts)
    uncertain_count = sum(1 for p in parts if p.status == Status.UNCERTAIN)
    low_confidence_count = sum(1 for p in parts if p.status != Status.UNCERTAIN and p.confidence == "low")
    if total == 0:
        coverage_ratio = 0.0
    else:
        coverage_ratio = 1.0 - (uncertain_count / total)

    if coverage_ratio >= 0.9 and low_confidence_count <= 2:
        quality = "high"
        quality_name = "良好"
    elif coverage_ratio >= 0.7:
        quality = "medium"
        quality_name = "一般"
    else:
        quality = "low"
        quality_name = "不足"

    return {
        "total_parts": total,
        "uncertain_count": uncertain_count,
        "low_confidence_count": low_confidence_count,
        "coverage_ratio": round(coverage_ratio, 2),
        "coverage_quality": quality,
        "coverage_quality_name": quality_name,
    }


def _repair_attention(parts: List[PartActualState], structural_flag: bool) -> Dict[str, Any]:
    """Recommend repair attention level based on damage severity and structural involvement."""
    severe_count = sum(1 for p in parts if p.status == Status.DAMAGED and p.damage_level == DamageLevel.SEVERE)
    missing_count = sum(1 for p in parts if p.status == Status.MISSING)
    damaged_count = sum(1 for p in parts if p.status == Status.DAMAGED)

    if structural_flag or severe_count >= 3 or missing_count >= 2:
        level = "high"
        level_name = "建议重点检修"
        reason = "结构件受损或存在多处严重/缺失部件，需专业定损和全面检测"
    elif severe_count >= 1 or damaged_count >= 3:
        level = "medium"
        level_name = "建议常规检修"
        reason = "存在局部严重或多个受损部件，建议到维修点进一步检查"
    elif damaged_count >= 1:
        level = "low"
        level_name = "建议关注"
        reason = "存在轻微损伤，可根据实际情况安排修复"
    else:
        level = "none"
        level_name = "无需维修"
        reason = "未检测到明显损伤"

    return {"level": level, "level_name": level_name, "reason": reason}


def _compute_photo_type_summary(plan: Dict[str, Any]) -> Dict[str, int]:
    """Aggregate photo counts by photo_type from a planner plan.

    Reads each view group's photo list and looks for a ``_planner_photo_type``
    field on individual photos. Views tagged as ``interior`` / ``auxiliary`` /
    ``unknown`` (per :data:`agents.view_mapping.NON_EXTERIOR_VIEWS`) are skipped
    so non-exterior photos do not dilute the wide_shot vs close-up ratio.

    Returns a dict with the four canonical keys (``wide_shot``,
    ``close_up_damage``, ``close_up_detail``, ``unknown``) so callers can
    compare against fixed thresholds without worrying about missing keys.
    """
    try:
        from agents.view_mapping import NON_EXTERIOR_VIEWS
    except Exception:
        NON_EXTERIOR_VIEWS = {"interior", "auxiliary", "unknown"}

    summary: Dict[str, int] = {
        "wide_shot": 0,
        "close_up_damage": 0,
        "close_up_detail": 0,
        "unknown": 0,
    }
    view_groups = plan.get("view_groups", {}) if isinstance(plan, dict) else {}
    for view_id, photo_list in view_groups.items():
        if view_id in NON_EXTERIOR_VIEWS:
            continue
        if not isinstance(photo_list, list):
            continue
        for photo in photo_list:
            if not isinstance(photo, dict):
                continue
            pt = photo.get("_planner_photo_type", "unknown")
            if pt not in summary:
                pt = "unknown"
            summary[pt] += 1
    return summary


def _human_review_notes(
    parts: List[PartActualState],
    coverage: Dict[str, Any],
    photo_type_summary: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate concise notes for human reviewers.

    Prioritises actionable, evidence-driven prompts:

    - Structural / glass parts that are still uncertain after evidence fusion
      deserve explicit review even if the rest of the car looks fine, because
      a single missed windshield crack can hide a structural event.
    - Parts judged ``damaged`` from one corner view are downgraded by adjacency
      rules; surface them so the reviewer can confirm or override.
    - Coverage-quality tips are kept generic to avoid spamming non-actionable
      suggestions like "底盘区域不可见" (chassis is never visible).

    The optional ``photo_type_summary`` dict maps photo_type names to
    counts.  When more than 70% of the photos are close-up shots
    (``close_up_damage`` + ``close_up_detail``) and fewer than 2 are
    wide-angle (``wide_shot``), a warning note is added at the top of the
    list to flag the evidence-quality concern.
    """
    notes: List[str] = []

    # 0. Photo-type evidence-quality warning.
    if photo_type_summary:
        total = sum(int(v) for v in photo_type_summary.values() if v)
        if total > 0:
            close_up = int(photo_type_summary.get("close_up_damage", 0)) + int(
                photo_type_summary.get("close_up_detail", 0)
            )
            wide = int(photo_type_summary.get("wide_shot", 0))
            if close_up / total >= 0.7 and wide < 2:
                pct = round(close_up / total * 100)
                notes.append(
                    f"证据质量提示：照片以近距离特写为主（{pct}%为特写/近景，仅 {wide} 张广角全景），"
                    f"识别结论对远景覆盖不足的部件需谨慎复核"
                )

    if coverage["coverage_quality"] == "low":
        notes.append("照片覆盖不足，部分部件无法判断，建议补拍关键视角")

    # 1. Uncertain safety-critical / structural parts (most actionable).
    uncertain_critical = sorted({
        PARTS_BY_ID.get(p.part_id, {}).get("part_name", p.part_id)
        for p in parts
        if p.status == Status.UNCERTAIN and p.part_id in _SAFETY_CRITICAL_PARTS
    })
    if uncertain_critical:
        notes.append(f"安全关键部件状态不确定：{', '.join(uncertain_critical)}")

    uncertain_structural = sorted({
        PARTS_BY_ID.get(p.part_id, {}).get("part_name", p.part_id)
        for p in parts
        if p.status == Status.UNCERTAIN and p.part_id in _STRUCTURAL_PARTS
    })
    if uncertain_structural:
        notes.append(f"结构件状态不确定，建议重点复核：{', '.join(uncertain_structural)}")

    # 2. Single-view severe damage that the adjacency rules downgraded.
    single_view_severe = sorted({
        p.part_name
        for p in parts
        if p.status == Status.DAMAGED and len(p.evidence_photos) == 1
        and p.damage_level == DamageLevel.SEVERE
    })
    if single_view_severe:
        notes.append(
            f"以下部件仅从单一视角判为严重损伤，建议人工复核："
            f"{', '.join(single_view_severe[:5])}"
        )

    # 3. Damage inversions: a part has many uncertain neighbours and is the
    # only path-side evidence for a region — flag for second look.
    inversions = []
    for p in parts:
        if p.status != Status.DAMAGED or p.damage_level not in (
            DamageLevel.LIGHT, DamageLevel.MODERATE
        ):
            continue
        if len(p.evidence_photos) > 1:
            continue
        if p.part_id.startswith(("door_", "mirror_", "roof_", "windshield", "sunroof_glass")):
            inversions.append(p.part_name)
    if inversions:
        notes.append(
            f"以下部件损伤证据较弱，建议复核：{', '.join(sorted(set(inversions))[:5])}"
        )

    return notes


def _build_overview_text(overview: Dict[str, Any]) -> str:
    """Build a single professional overview paragraph."""
    accident = overview["accident_type"]["type_name"]
    overall = overview["overall_assessment"]
    region_lines = [
        f"{r['region_name']}（{r['max_level_name']}，{r['damaged_count']}处受损{r['missing_count']}处缺失）"
        for r in overview["region_summary"]
    ]
    region_text = "；".join(region_lines) if region_lines else "无明显受损区域"

    structural = overview["structural_summary"]
    structural_text = ""
    if structural["structural_damage_flag"]:
        names = [p["part_name"] for p in structural["affected_parts"]]
        structural_text = f"结构件（{', '.join(names)}）受损，"

    safety = overview["safety_summary"]
    safety_text = ""
    if safety["affected_count"] > 0:
        names = [p["part_name"] for p in safety["affected_parts"]]
        safety_text = f"安全关键部件（{', '.join(names)}）受影响，"

    symmetric = overview["symmetric_damage"]
    symmetric_text = ""
    if symmetric["symmetric_damage_detected"]:
        groups = [p["group_name"] for p in symmetric["pairs"]]
        symmetric_text = f"左右对称部位（{', '.join(groups)}）同时损伤，"

    coverage = overview["coverage_assessment"]
    attention = overview["repair_attention"]
    coverage_quality_name = coverage["coverage_quality_name"]

    text = (
        f"{accident}。{structural_text}{safety_text}{symmetric_text}"
        f"整体评估为{overall['overall_severity_name']}，主要受损区域为{overall['primary_damage_zone_name']}。"
        f"区域分布：{region_text}。"
        f"证据覆盖{coverage_quality_name}（覆盖率{coverage['coverage_ratio']:.0%}）。"
        f"{attention['level_name']}：{attention['reason']}。"
    )
    return text


def generate_damage_overview(
    parts: List[PartActualState],
    structural_damage_flag: bool = False,
    overall_severity: str = "",
    primary_damage_zone: str = "",
    photo_type_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a professional damage overview from part actual states.

    Parameters
    ----------
    parts:
        List of part actual states (all topology nodes).
    structural_damage_flag:
        Whether the assessment triggered the structural damage flag.
    overall_severity:
        Overall severity string (e.g. "severe", "moderate", "light", "none").
    primary_damage_zone:
        Region ID with the highest damage score.
    photo_type_summary:
        Optional mapping of photo_type -> count. When more than 70% of
        exterior photos are close-up shots and fewer than 2 are wide-angle,
        a warning note is prepended to ``human_review_notes``.

    Returns
    -------
    dict
        Structured damage overview with human-readable text and supporting data.
    """
    accident_type = _infer_accident_type(parts)
    region_summary = _build_region_summary(parts)
    safety_summary = _build_safety_summary(parts)
    structural_summary = _build_structural_summary(parts)
    symmetric_damage = _detect_symmetric_damage(parts)
    coverage = _coverage_assessment(parts)
    repair_attention = _repair_attention(parts, structural_damage_flag or structural_summary["structural_damage_flag"])
    review_notes = _human_review_notes(parts, coverage, photo_type_summary)

    overview = {
        "accident_type": accident_type,
        "overall_assessment": {
            "overall_severity": overall_severity,
            "overall_severity_name": _severity_name(overall_severity),
            "primary_damage_zone": primary_damage_zone,
            "primary_damage_zone_name": _region_name(primary_damage_zone) if primary_damage_zone != "multiple" else "多区域",
            "structural_damage_flag": structural_damage_flag or structural_summary["structural_damage_flag"],
        },
        "region_summary": region_summary,
        "safety_summary": safety_summary,
        "structural_summary": structural_summary,
        "symmetric_damage": symmetric_damage,
        "coverage_assessment": coverage,
        "repair_attention": repair_attention,
        "human_review_notes": review_notes,
    }

    overview["overview_text"] = _build_overview_text(overview)
    return overview
