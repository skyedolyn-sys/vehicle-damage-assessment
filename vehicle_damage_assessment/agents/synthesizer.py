import json
from typing import List, Dict, Any, Optional, Set
from config import PARTS_CATALOG, PARTS_BY_ID, PARTS_TOPOLOGY
from models.topology import VehicleTopology


STATUS_PRIORITY = {"damaged": 3, "uncertain": 2, "intact": 1, "missing": 4}
LEVEL_PRIORITY = {"severe": 4, "moderate": 3, "light": 2, "none": 1, "unknown": 0}
CONFIDENCE_PRIORITY = {"low": 0, "medium": 1, "high": 2}

# When a source is uncertain, prefer a non-uncertain status from another
# source that covers the same part.  This implements the "adjacent view
# fallback" without extra LLM calls.
UNCERTAIN_STATUS_PRIORITY = {"missing": 4, "damaged": 3, "intact": 1, "uncertain": 0}

# Parts that are frequently only partially visible from a single view and
# therefore need conservative handling in the synthesizer.
CONSERVATIVE_PARTS = {
    "door_front_left", "door_rear_left",
    "door_front_right", "door_rear_right",
    "mirror_left", "mirror_right",
    "roof_middle", "roof_rear", "sunroof_glass",
}

ROOF_PARTS = {"roof_front", "roof_middle", "roof_rear", "sunroof_glass", "roof_rack"}

# Region units: groups of parts that should share a unified conclusion.
# - rear_unit: tailgate and rear windshield are physically one damaged area.
REGION_UNITS = {
    "rear_unit": {"tailgate", "windshield_rear"},
}

# Best canonical view for observing each conservative part.  Damage reports
# from non-primary views are downgraded because they may only show an edge.
PRIMARY_VIEW = {
    "door_front_left": ["left"],
    "door_rear_left": ["left"],
    "door_front_right": ["right"],
    "door_rear_right": ["right"],
    "mirror_left": ["left"],
    "mirror_right": ["right"],
    "roof_middle": ["front", "rear"],  # side views are acceptable but not ideal
    "roof_rear": ["rear"],
    "sunroof_glass": [],  # side views are poor for the sunroof
}

# View-based conflict resolution weights for conservative parts.
# "primary" views have the best vantage point for the part and override secondary
# edge/glance views.  "secondary" views may see only an edge and are downgraded
# unless multiple secondary sources agree.
VIEW_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "door_front_left": {
        "primary": {"left"},
        "secondary": {"front_left", "rear_left"},
    },
    "door_rear_left": {
        "primary": {"left"},
        "secondary": {"front_left", "rear_left"},
    },
    "door_front_right": {
        "primary": {"right"},
        "secondary": {"front_right", "rear_right"},
    },
    "door_rear_right": {
        "primary": {"right"},
        "secondary": {"front_right", "rear_right"},
    },
    "mirror_left": {
        "primary": {"left"},
        "secondary": {"front_left"},
    },
    "mirror_right": {
        "primary": {"right"},
        "secondary": {"front_right"},
    },
}

# Roof parts are handled separately because they are easily confused with rear
# structure damage from side views.  A symmetric roof should only be trusted as
# damaged when multiple non-rear views agree.
ROOF_PRIMARY_REGIONS = {"left", "right", "front_left", "front_right"}
ROOF_SECONDARY_REGIONS = {"rear", "rear_left", "rear_right"}

# Front parts that are frequently misclassified as damaged from single
# corner/side views due to occlusion, shadows, or spill-over from adjacent
# severe rear damage.
FRONT_FALSE_DAMAGE_PARTS = {
    "bumper_front",
    "door_front_left",
    "door_front_right",
    "fender_front_left",
    "fender_front_right",
    "headlight_front_left",
    "headlight_front_right",
}

# Parts whose damage level should be capped when only a single secondary view
# reports damage and there is adjacent severe damage that may cause spill-over.
SPILL_OVER_PRONE_PARTS = {
    "door_front_left",
    "door_front_right",
    "door_rear_left",
    "door_rear_right",
    "fender_rear_right",
    "taillight_rear_right",
}


def _extract_evidence_photos(candidate: Dict[str, Any]) -> List[str]:
    """Extract evidence photo ids from a candidate, handling string or list."""
    raw_photos = candidate.get("evidence_photo", [])
    if isinstance(raw_photos, str):
        raw_photos = [p.strip() for p in raw_photos.split(",") if p.strip()] if raw_photos else []
    return list(dict.fromkeys(p for p in raw_photos if p))


def _has_photo_coverage_for_part(part_id: str, region_results: List[Dict[str, Any]]) -> bool:
    """Return True if any region result claims to cover the part's region."""
    part_info = PARTS_BY_ID.get(part_id, {})
    part_region = part_info.get("part_category", "")
    if not part_region:
        return False
    for region_result in region_results:
        region = region_result.get("region", "")
        if region == part_region and region_result.get("parts") is not None:
            return True
    return False


def _resolve_status(candidates: List[Dict[str, Any]]) -> str:
    """Conservatively resolve status across evidence sources."""
    statuses = [c.get("status", "uncertain") for c in candidates]
    if any(s == "missing" for s in statuses):
        return "missing"
    if any(s == "damaged" for s in statuses):
        return "damaged"
    return max(statuses, key=lambda s: UNCERTAIN_STATUS_PRIORITY.get(s, 0))


def _resolve_damage_level(candidates: List[Dict[str, Any]], status: str, part_id: str = "") -> str:
    """Resolve damage level conservatively."""
    if status == "intact":
        return "none"
    levels = [c.get("damage_level", "unknown") for c in candidates if c.get("status") == status]
    if not levels:
        levels = [c.get("damage_level", "unknown") for c in candidates]
    resolved = max(levels, key=lambda lvl: LEVEL_PRIORITY.get(lvl, 0))

    # Downgrade severe reports from non-primary views for partially visible parts.
    if status == "damaged" and part_id in CONSERVATIVE_PARTS and len(candidates) == 1:
        source_regions = [c.get("_region", "") for c in candidates]
        primary = PRIMARY_VIEW.get(part_id, [])
        if primary and not any(r in primary for r in source_regions):
            if resolved == "severe":
                resolved = "moderate"
            elif resolved == "moderate":
                resolved = "light"
    return resolved


def _resolve_confidence(candidates: List[Dict[str, Any]], status: str, part_id: str = "") -> str:
    """Resolve confidence based on consensus and conflict."""
    statuses = [c.get("status", "uncertain") for c in candidates]
    confidences = [c.get("confidence", "low") for c in candidates]
    worst = min(confidences, key=lambda c: CONFIDENCE_PRIORITY.get(c, 0))

    damaged_missing_count = sum(1 for s in statuses if s in ("damaged", "missing"))
    if status in ("damaged", "missing") and damaged_missing_count >= 2:
        return "medium" if worst == "low" else worst
    if status in ("damaged", "missing") and damaged_missing_count == 1:
        # Single source reports damage vs others intact: use the worst confidence
        # among all sources (do not force an extra downgrade below that).
        return worst

    # Partial-visibility parts seen from only one view should not claim high confidence.
    if part_id in CONSERVATIVE_PARTS and len(candidates) == 1 and worst == "high":
        worst = "medium"

    return worst


def _split_candidates_by_weight(
    candidates: List[Dict[str, Any]], part_id: str
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split candidates into primary and secondary views for a conservative part."""
    weights = VIEW_WEIGHTS.get(part_id, {})
    primary_regions = set(weights.get("primary", set()))
    secondary_regions = set(weights.get("secondary", set()))
    primary = [c for c in candidates if c.get("_region", "") in primary_regions]
    secondary = [c for c in candidates if c.get("_region", "") in secondary_regions]
    return primary, secondary


def _resolve_status_weighted(candidates: List[Dict[str, Any]], part_id: str) -> str:
    """Resolve status using view weights for conservative parts.

    Primary (best vantage) views override secondary edge/glance views.
    This prevents damage spill-over from adjacent modules (e.g. rear damage
    making a rear door look damaged in a rear-diagonal photo).
    """
    if part_id not in VIEW_WEIGHTS or not candidates:
        return _resolve_status(candidates)

    statuses = [c.get("status", "uncertain") for c in candidates]
    if any(s == "missing" for s in statuses):
        return "missing"

    primary, secondary = _split_candidates_by_weight(candidates, part_id)

    if primary:
        primary_statuses = [c.get("status", "uncertain") for c in primary]
        if any(s == "damaged" for s in primary_statuses):
            return "damaged"
        if any(s == "intact" for s in primary_statuses):
            return "intact"
        # primary all uncertain: trust secondary if it reports damage.
        if any(c.get("status") == "damaged" for c in secondary):
            return "damaged"
        return "uncertain"

    # No primary coverage: fall back to conservative merge.
    return _resolve_status(candidates)


def _resolve_status_roof(candidates: List[Dict[str, Any]]) -> str:
    """Resolve status for roof parts.

    Roof parts are easily confused with rear structure damage from side/rear
    views.  Only mark damaged when at least two non-rear primary views agree,
    or when every visible source reports damage.  Otherwise prefer intact.
    """
    if not candidates:
        return "uncertain"

    statuses = [c.get("status", "uncertain") for c in candidates]
    if any(s == "missing" for s in statuses):
        return "missing"

    primary = [c for c in candidates if c.get("_region", "") in ROOF_PRIMARY_REGIONS]
    secondary = [c for c in candidates if c.get("_region", "") in ROOF_SECONDARY_REGIONS]

    # For sunroof_glass we require even stronger consensus because side views
    # only show the glass edge and cannot distinguish the sunroof from the
    # rear windshield damage behind it.
    part_id = candidates[0].get("part_id", "")
    is_sunroof = part_id == "sunroof_glass"

    if primary:
        primary_damaged = [c for c in primary if c.get("status") == "damaged"]
        primary_intact = [c for c in primary if c.get("status") == "intact"]
        # If multiple primary views agree on damaged, trust it.
        if len(primary_damaged) >= 2:
            return "damaged"
        # Sunroof: a single primary damaged is not enough unless there is no
        # intact primary source to contradict it.
        if is_sunroof and primary_damaged and not primary_intact:
            return "damaged"
        # If at least one primary view says intact, prefer intact (low confidence).
        if primary_intact:
            return "intact"
        # Primary all uncertain/damaged single source: fall back to secondary.
        if any(c.get("status") == "damaged" for c in secondary):
            return "damaged"
        return "uncertain"

    # No primary coverage: only trust secondary if all agree on damage.
    # Sunroof is never trusted from secondary-only coverage because side/rear
    # views cannot see the actual sunroof glass surface.
    if is_sunroof:
        return "uncertain"
    if secondary and all(c.get("status") == "damaged" for c in secondary):
        return "damaged"
    return "uncertain"


def _resolve_damage_level_roof(
    candidates: List[Dict[str, Any]], status: str
) -> str:
    """Resolve damage level for roof parts.

    Sunroof glass should never be classified as severe from side/rear views
    because those views cannot see the actual glass surface clearly.  Roof
    rear/middle levels derived only from rear secondary views are capped to
    avoid being inflated by tailgate/windshield damage.
    """
    if status != "damaged":
        return "none"

    part_id = candidates[0].get("part_id", "") if candidates else ""
    is_sunroof = part_id == "sunroof_glass"

    primary = [c for c in candidates if c.get("_region", "") in ROOF_PRIMARY_REGIONS]
    secondary = [c for c in candidates if c.get("_region", "") in ROOF_SECONDARY_REGIONS]

    if primary:
        primary_damaged = [c for c in primary if c.get("status") == "damaged"]
        if primary_damaged:
            level = max(
                (c.get("damage_level", "unknown") for c in primary_damaged),
                key=lambda lvl: LEVEL_PRIORITY.get(lvl, 0),
            )
            # Sunroof severe requires at least two agreeing primary damaged views.
            if is_sunroof and level == "severe" and len(primary_damaged) < 2:
                level = "moderate"
            return level

    secondary_damaged = [c for c in secondary if c.get("status") == "damaged"]
    if secondary_damaged:
        base = max(
            (c.get("damage_level", "unknown") for c in secondary_damaged),
            key=lambda lvl: LEVEL_PRIORITY.get(lvl, 0),
        )
        # Sunroof is never trusted from secondary-only coverage.
        if is_sunroof:
            return "light"
        # roof_rear/middle from rear-only views should not inherit severe from
        # the rear windshield/tailgate area.
        if base == "severe":
            return "moderate"
        return _downgrade_level(base)

    return _resolve_damage_level(candidates, status)


def _resolve_confidence_roof(
    candidates: List[Dict[str, Any]], status: str
) -> str:
    """Resolve confidence for roof parts."""
    if not candidates:
        return "low"

    primary_damaged = [
        c for c in candidates
        if c.get("_region", "") in ROOF_PRIMARY_REGIONS and c.get("status") == "damaged"
    ]
    if len(primary_damaged) >= 2:
        return _resolve_confidence(primary_damaged, "damaged")

    secondary = [c for c in candidates if c.get("_region", "") in ROOF_SECONDARY_REGIONS]
    if secondary and all(c.get("status") == "damaged" for c in secondary):
        return _resolve_confidence(secondary, "damaged")
    return "low"


def _resolve_damage_level_weighted(
    candidates: List[Dict[str, Any]], status: str, part_id: str
) -> str:
    """Resolve damage level using view weights."""
    if status != "damaged" or part_id not in VIEW_WEIGHTS or not candidates:
        return _resolve_damage_level(candidates, status, part_id=part_id)

    primary, secondary = _split_candidates_by_weight(candidates, part_id)

    if primary:
        primary_damaged = [c for c in primary if c.get("status") == "damaged"]
        if primary_damaged:
            return _resolve_damage_level(primary_damaged, "damaged", part_id=part_id)
        # primary says intact but secondary says damaged -> final intact, no level.
        return "none"

    # Only secondary coverage: downgrade one level unless multiple sources agree.
    secondary_damaged = [c for c in secondary if c.get("status") == "damaged"]
    if not secondary_damaged:
        return _resolve_damage_level(candidates, status, part_id=part_id)

    base = max(
        (c.get("damage_level", "unknown") for c in secondary_damaged),
        key=lambda lvl: LEVEL_PRIORITY.get(lvl, 0),
    )
    if len(secondary_damaged) >= 2:
        return base
    return _downgrade_level(base)


def _resolve_confidence_weighted(
    candidates: List[Dict[str, Any]], status: str, part_id: str
) -> str:
    """Resolve confidence using view weights."""
    if part_id not in VIEW_WEIGHTS or not candidates:
        return _resolve_confidence(candidates, status, part_id=part_id)

    primary, secondary = _split_candidates_by_weight(candidates, part_id)

    if primary:
        primary_statuses = [c.get("status", "uncertain") for c in primary]
        if "damaged" in primary_statuses:
            return _resolve_confidence(primary, "damaged", part_id=part_id)
        if "intact" in primary_statuses:
            # primary intact overrides secondary damage -> cap at medium.
            conf = _resolve_confidence(primary, "intact", part_id=part_id)
            return "medium" if CONFIDENCE_PRIORITY.get(conf, 0) > 1 else conf
        # primary uncertain
        if any(c.get("status") == "damaged" for c in secondary):
            return _resolve_confidence(secondary, "damaged", part_id=part_id)
        return _resolve_confidence(primary, "uncertain", part_id=part_id)

    # Only secondary coverage: keep low confidence unless multiple agree.
    secondary_damaged = [c for c in secondary if c.get("status") == "damaged"]
    if len(secondary_damaged) >= 2:
        return _resolve_confidence(secondary_damaged, "damaged", part_id=part_id)
    return "low"


def _downgrade_level(level: str) -> str:
    """Downgrade a damage level by one step."""
    return {"severe": "moderate", "moderate": "light"}.get(level, level)


def _append_note(part: Dict[str, Any], note: str) -> None:
    """Append a note to a part dict, preserving existing notes."""
    existing = part.get("notes", "")
    part["notes"] = f"{existing}；{note}" if existing else note



def _build_evidence_sources(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build traceable evidence sources for each part conclusion."""
    sources: List[Dict[str, Any]] = []
    seen: set = set()
    for c in candidates:
        photos = _extract_evidence_photos(c)
        region = c.get("_region", "")
        key = (region, c.get("status"), c.get("confidence"), tuple(photos))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "region": region,
            "status": c.get("status", "uncertain"),
            "damage_level": c.get("damage_level", "unknown"),
            "confidence": c.get("confidence", "low"),
            "evidence_photo": photos,
            "notes": c.get("notes", "").strip(),
        })
    return sources


def _apply_adjacency_rules(
    merged_parts: List[Dict[str, Any]],
    topology: Optional[VehicleTopology],
) -> List[Dict[str, Any]]:
    """Enforce geometric consistency between adjacent parts."""
    if topology is None:
        return merged_parts

    merged_by_id = {p["part_id"]: p for p in merged_parts}
    updated = []

    for part in merged_parts:
        part_id = part["part_id"]
        node = topology.get_node(part_id)
        if node is None:
            updated.append(part)
            continue

        new_part = dict(part)
        notes = part.get("notes", "")
        adjacents = topology.get_adjacent(part_id)

        neighbor_status: Dict[str, str] = {
            adj.part_id: merged_by_id.get(adj.part_id, {}).get("status", "")
            for adj in adjacents
        }
        neighbor_level: Dict[str, str] = {
            adj.part_id: merged_by_id.get(adj.part_id, {}).get("damage_level", "")
            for adj in adjacents
        }

        # Rule 1: intact next to damaged/missing -> cap confidence at medium.
        if part.get("status") == "intact":
            damaged_neighbors = [pid for pid, status in neighbor_status.items() if status in ("damaged", "missing")]
            if damaged_neighbors and part.get("confidence") == "high":
                new_part["confidence"] = "medium"
                _append_note(new_part, f"相邻部件存在损伤，降低置信度：{', '.join(damaged_neighbors)}")

        # Rule 2: sunroof level should not be lower than adjacent damaged roof.
        if part_id == "sunroof_glass" and part.get("status") == "damaged":
            for adj in adjacents:
                adj_pid = adj.part_id
                if adj_pid not in ROOF_PARTS or neighbor_status.get(adj_pid) != "damaged":
                    continue
                adj_level = neighbor_level.get(adj_pid, "unknown")
                current_level = new_part.get("damage_level", "unknown")
                if LEVEL_PRIORITY.get(adj_level, 0) > LEVEL_PRIORITY.get(current_level, 0):
                    new_part["damage_level"] = adj_level
                    _append_note(new_part, f"相邻车顶部件 {adj_pid} 为 {adj_level}，天窗级别同步上调")

        # Rule 3: door next to severely damaged fender/bumper cannot stay intact.
        if part_id.startswith("door_") and part.get("status") == "intact":
            severe_neighbors = [
                pid for pid, status in neighbor_status.items()
                if status == "damaged" and neighbor_level.get(pid) == "severe"
            ]
            if severe_neighbors:
                new_part["confidence"] = "low"
                _append_note(new_part, f"相邻部件严重受损，降低车门置信度：{', '.join(severe_neighbors)}")

        # Rule 4: front parts damaged from a single corner view with adjacent
        # severe rear damage are likely spill-over false positives: downgrade.
        if (
            part_id in FRONT_FALSE_DAMAGE_PARTS
            and new_part.get("status") == "damaged"
            and new_part.get("confidence") in ("low", "medium")
        ):
            severe_neighbors = [
                pid for pid, status in neighbor_status.items()
                if status == "damaged" and neighbor_level.get(pid) == "severe"
            ]
            if severe_neighbors:
                current_level = new_part.get("damage_level", "unknown")
                # Only downgrade moderate/light, keep severe if the model really
                # reported severe from the front view itself.
                if current_level in ("moderate", "light"):
                    new_level = _downgrade_level(current_level)
                    new_part["damage_level"] = new_level
                    _append_note(
                        new_part,
                        f"前部部件从单一/低置信度视角判损，且相邻后部严重受损，疑似误检，级别从 {current_level} 降至 {new_level}",
                    )
                # Also cap confidence to low when the only evidence is low/medium.
                if new_part.get("confidence") == "medium":
                    new_part["confidence"] = "low"

        # Rule 5: spill-over prone parts (doors, right rear lights/fenders) that
        # are damaged from a single secondary source should be capped at moderate.
        if (
            part_id in SPILL_OVER_PRONE_PARTS
            and new_part.get("status") == "damaged"
            and new_part.get("damage_level") == "severe"
        ):
            evidence_regions = {
                src.get("region", "") for src in new_part.get("evidence_sources", [])
                if src.get("status") == "damaged"
            }
            primary = set(VIEW_WEIGHTS.get(part_id, {}).get("primary", []))
            if primary and not any(r in primary for r in evidence_regions):
                new_part["damage_level"] = "moderate"
                _append_note(
                    new_part,
                    f"{part_id} 从非主视角单源判为 severe，易受相邻严重损伤传染，降级为 moderate",
                )

        updated.append(new_part)

    return updated


def _infer_missing_roof_part(
    part_id: str,
    topology: Optional[VehicleTopology],
    merged_by_id: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Infer an intact roof part when it has no candidates but neighbors are intact."""
    if topology is None or part_id not in ROOF_PARTS:
        return None

    node = topology.get_node(part_id)
    if node is None:
        return None

    adjacents = topology.get_adjacent(part_id)
    intact_roof_neighbors = [
        adj.part_id for adj in adjacents
        if adj.part_id in ROOF_PARTS
        and merged_by_id.get(adj.part_id, {}).get("status") == "intact"
    ]
    if not intact_roof_neighbors:
        return None

    base_info = PARTS_BY_ID.get(part_id, {})
    return {
        "part_id": part_id,
        "part_name": base_info.get("part_name", part_id),
        "part_category": base_info.get("part_category", ""),
        "side": base_info.get("side", ""),
        "status": "intact",
        "damage_level": "none",
        "damage_type": [],
        "confidence": "low",
        "evidence_photo": [],
        "evidence_sources": [],
        "notes": f"该部件无直接视角覆盖，从相邻 intact 车顶部件推断：{', '.join(intact_roof_neighbors)}",
    }


def synthesizer_agent(
    region_results: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any] = None,
    topology: Optional[VehicleTopology] = None,
) -> Dict[str, Any]:
    """
    确定性汇总各区域 Worker 的输出，生成整车统一的损伤评估报告。
    不使用 LLM，避免大请求体导致服务端断开。
    """
    # Collect all part conclusions grouped by part_id.
    parts_by_id: Dict[str, List[Dict[str, Any]]] = {}
    all_uncertain_items: List[Dict[str, Any]] = []

    for region_result in region_results:
        region = region_result.get("region", "未知区域")
        for part in region_result.get("parts", []):
            if not isinstance(part, dict):
                continue
            part_id = part.get("part_id")
            if not part_id:
                continue
            part_copy = dict(part)
            part_copy["_region"] = region
            parts_by_id.setdefault(part_id, []).append(part_copy)

        for item in region_result.get("uncertain_items", []):
            if isinstance(item, dict):
                all_uncertain_items.append(item)

    # Determine which part IDs to iterate.
    if topology is not None:
        part_ids_to_iterate = list(topology.nodes.keys())
    else:
        part_ids_to_iterate = list(PARTS_BY_ID.keys())

    # First pass: resolve each part independently.
    merged_parts: List[Dict[str, Any]] = []
    for part_id in part_ids_to_iterate:
        candidates = parts_by_id.get(part_id, [])
        base_info = PARTS_BY_ID.get(part_id, {})

        if not candidates:
            has_coverage = _has_photo_coverage_for_part(part_id, region_results)
            if has_coverage:
                notes = "该部件所在区域有照片覆盖，但无法从照片中确认状态"
            else:
                notes = "该区域无照片覆盖，无法判断"
            merged_parts.append({
                "part_id": part_id,
                "part_name": base_info.get("part_name", part_id),
                "part_category": base_info.get("part_category", ""),
                "side": base_info.get("side", ""),
                "status": "uncertain",
                "damage_level": "unknown",
                "damage_type": [],
                "confidence": "low",
                "evidence_photo": [],
                "evidence_sources": [],
                "notes": notes,
            })
            continue

        if part_id in VIEW_WEIGHTS:
            best_status = _resolve_status_weighted(candidates, part_id)
            best_level = _resolve_damage_level_weighted(candidates, best_status, part_id)
            worst_confidence = _resolve_confidence_weighted(candidates, best_status, part_id)
        elif part_id in ROOF_PARTS:
            best_status = _resolve_status_roof(candidates)
            best_level = _resolve_damage_level_roof(candidates, best_status)
            worst_confidence = _resolve_confidence_roof(candidates, best_status)
        else:
            best_status = _resolve_status(candidates)
            best_level = _resolve_damage_level(candidates, best_status, part_id=part_id)
            worst_confidence = _resolve_confidence(candidates, best_status, part_id=part_id)

        evidence_sources = _build_evidence_sources(candidates)
        evidence_photos = list(dict.fromkeys(
            ep for src in evidence_sources for ep in src.get("evidence_photo", []) if ep
        ))

        damage_types: Set[str] = set()
        notes_parts = []
        for c in candidates:
            raw_types = c.get("damage_type", [])
            if isinstance(raw_types, str):
                if raw_types and raw_types != "none":
                    damage_types.update(dt.strip() for dt in raw_types.split(",") if dt.strip())
            elif isinstance(raw_types, list):
                damage_types.update(str(dt) for dt in raw_types if dt)
            note = c.get("notes", "").strip()
            region = c.get("_region", "")
            if note:
                notes_parts.append(f"[{region}] {note}")

        merged_parts.append({
            "part_id": part_id,
            "part_name": base_info.get("part_name", part_id),
            "part_category": base_info.get("part_category", ""),
            "side": base_info.get("side", ""),
            "status": best_status,
            "damage_level": best_level,
            "damage_type": sorted(damage_types),
            "confidence": worst_confidence,
            "evidence_photo": evidence_photos,
            "evidence_sources": evidence_sources,
            "notes": "；".join(notes_parts) if notes_parts else "",
        })

    # Second pass: unify physically connected region units.
    merged_parts = _unify_region_units(merged_parts, topology)

    # Third pass: topology-based post-processing and roof inference.
    merged_by_id = {p["part_id"]: p for p in merged_parts}

    # Infer missing roof parts from intact neighbors.
    final_parts: List[Dict[str, Any]] = []
    for part in merged_parts:
        if part["status"] == "uncertain" and part.get("damage_level") == "unknown":
            inferred = _infer_missing_roof_part(part["part_id"], topology, merged_by_id)
            if inferred is not None:
                final_parts.append(inferred)
                continue
        final_parts.append(part)

    # Apply adjacency consistency rules.
    final_parts = _apply_adjacency_rules(final_parts, topology)

    return {
        "parts": final_parts,
        "uncertain_items": all_uncertain_items,
    }


def _unify_region_units(
    merged_parts: List[Dict[str, Any]], topology: Optional[VehicleTopology] = None
) -> List[Dict[str, Any]]:
    """Force unified conclusions within physically connected region units.

    - rear_unit (tailgate + windshield_rear): use the worst status/level among members,
      but protect against missing being selected when damaged evidence exists from any view.
    """
    merged_by_id = {p["part_id"]: p for p in merged_parts}

    # rear_unit: worst wins, but missing should not override damaged.
    rear_members = [merged_by_id[pid] for pid in REGION_UNITS.get("rear_unit", []) if pid in merged_by_id]
    if rear_members:
        statuses = [m["status"] for m in rear_members]
        levels = [m["damage_level"] for m in rear_members]

        # If any member is damaged, prefer damaged over missing to avoid the
        # severe rear collision being misclassified as missing parts.
        rear_status = "damaged" if "damaged" in statuses else max(statuses, key=lambda s: STATUS_PRIORITY.get(s, 0))
        rear_level = "severe" if "severe" in levels else max(levels, key=lambda lvl: LEVEL_PRIORITY.get(lvl, 0))

        note = "车尾区域单元统一结论（tailgate + 后挡风玻璃）"
        for m in rear_members:
            m["status"] = rear_status
            m["damage_level"] = rear_level
            existing = m.get("notes", "")
            m["notes"] = f"{note}；{existing}" if existing and not existing.startswith(note) else note or existing

    return list(merged_by_id.values())
