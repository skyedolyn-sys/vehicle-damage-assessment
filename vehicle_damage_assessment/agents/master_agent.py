"""MasterAgent — coordinates the ViewAgent Team damage-assessment pipeline.

Responsibilities:
1. Obtain vehicle prior and build topology.
2. Run PlannerAgent for 4-class photo pre-screening.
3. Dispatch one ViewAgent per exterior photo (parallel, concurrency-limited).
4. Aggregate per-part evidence, boost confidence when multiple photos agree.
5. Run deterministic reviewer and synthesizer.
6. Compare against topology and return a DamageAssessment.

The orchestrator keeps ``assessment_orchestrator.py`` as a thin compatibility
wrapper around this module.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agents.minimax_client import build_image_content
from agents.planner_agent import planner_agent
from agents.reviewer_subagent import reviewer_subagent
from agents.synthesizer import synthesizer_agent
from agents.topology_builder import build_vehicle_topology
from agents.topology_comparator import compare_topology
from agents.vehicle_prior import vehicle_prior_agent
from agents.view_agent import view_agent, view_agent_result_to_part_actual_states
from agents.view_mapping import (
    EXTERIOR_VIEWS,
    PHOTO_TYPE_CATEGORIES,
    get_display_name,
    is_exterior_view,
)
from config import MAX_CONCURRENT_API_CALLS
from models import DamageAssessment, PartActualState

logger = logging.getLogger(__name__)

_master_file_handler = logging.FileHandler(
    os.path.expanduser("~/vehicle_damage_assessment_master.log"), mode="a", encoding="utf-8"
)
_master_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_master_file_handler.setLevel(logging.INFO)
logger.addHandler(_master_file_handler)
logger.setLevel(logging.INFO)


async def master_assessment_agent(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
    plan: Optional[Dict[str, Any]] = None,
) -> DamageAssessment:
    """Run the full assessment pipeline using the ViewAgent Team architecture.

    Parameters
    ----------
    files:
        List of photo dicts with at least ``id`` and ``path``.
    vehicle_info:
        Basic vehicle info (``vehicle_id``, ``vehicle_name``).
    plan:
        Optional pre-computed planner output. If omitted, PlannerAgent is called.

    Returns
    -------
    DamageAssessment
        Final damage assessment with topology comparison applied.
    """
    logger.info("[master] start photos=%d vehicle=%s", len(files), vehicle_info.get("vehicle_name"))

    # 1. Vehicle prior and topology
    vehicle_prior = await vehicle_prior_agent(vehicle_info)
    topology = build_vehicle_topology(vehicle_info, vehicle_prior)

    # 2. PlannerAgent 4-class pre-screening
    if plan is None:
        plan = await planner_agent(files, vehicle_prior)

    classifications = _extract_photo_classifications(plan)

    # 3. Filter exterior photos
    exterior_photos = [
        {**photo, "category": classifications.get(photo.get("id"), "unknown")}
        for photo in files
        if classifications.get(photo.get("id"), "unknown") == "exterior"
    ]
    logger.info("[master] exterior photos=%d total=%d", len(exterior_photos), len(files))

    # 4. Dispatch ViewAgent Team in parallel
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

    async def _run_one(photo: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            try:
                return await view_agent(photo, vehicle_prior)
            except Exception as exc:
                logger.warning("[master] view_agent failed for photo_id=%s: %s", photo.get("id"), exc)
                return {"photo_id": photo.get("id"), "primary_view": None, "view_detections": [], "parts": []}

    view_results = await asyncio.gather(*[_run_one(p) for p in exterior_photos])
    view_results = [r for r in view_results if r.get("parts")]

    # 5. Aggregate into PartEvidence and region_results
    part_evidence = _aggregate_part_evidence(view_results)
    region_results = _build_region_results(view_results, part_evidence)

    # 6. Reviewer
    review = await reviewer_subagent(region_results, plan, vehicle_prior)

    # 7. Synthesizer
    merged = synthesizer_agent(region_results, vehicle_prior, topology)

    # 8. Apply reviewer overrides
    actual_states = _apply_review_overrides(
        merged.get("part_actual_states", []),
        review.get("reviewed_part_actual_states", []),
    )

    # 9. Topology comparison
    assessment = compare_topology(topology, actual_states)
    assessment._plan = plan  # type: ignore[attr-defined]

    logger.info(
        "[master] done findings=%d damaged=%d uncertain=%d",
        len(assessment.parts),
        len(assessment.damaged_parts),
        len(assessment.uncertain_parts),
    )
    return assessment


def _extract_photo_classifications(plan: Dict[str, Any]) -> Dict[str, str]:
    """Extract photo_id -> category map from planner output.

    Supports both the new ``photo_classifications`` field and the legacy
    ``view_groups`` / ``photo_views`` fields so migration can be incremental.
    """
    classifications: Dict[str, str] = {}

    # New schema: explicit photo_classifications
    for item in plan.get("photo_classifications", []) or []:
        category = item.get("category", "unknown")
        if category in PHOTO_TYPE_CATEGORIES:
            classifications[item.get("photo_id")] = category
        else:
            classifications[item.get("photo_id")] = "unknown"

    # Legacy schema: derive from view_groups
    if not classifications:
        for view_id, photos in (plan.get("view_groups") or {}).items():
            category = _legacy_view_to_category(view_id)
            for photo in photos:
                pid = photo.get("id") if isinstance(photo, dict) else str(photo)
                classifications[pid] = category

    # Legacy schema: photo_views list
    for item in plan.get("photo_views", []) or []:
        pid = item.get("photo_id")
        if pid and pid not in classifications:
            view_id = item.get("view_id", "unknown")
            classifications[pid] = _legacy_view_to_category(view_id)

    return classifications


def _legacy_view_to_category(view_id: str) -> str:
    """Map legacy planner view_id to a 4-class category."""
    if is_exterior_view(view_id):
        return "exterior"
    if view_id in ("interior",):
        return "interior"
    if view_id in ("auxiliary", "vehicle_info"):
        return "vehicle_info"
    return "exclude"


def _aggregate_part_evidence(view_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group observations by part_id and compute aggregated fields."""
    evidence: Dict[str, Dict[str, Any]] = {}

    for result in view_results:
        photo_id = result.get("photo_id", "unknown")
        primary_view = result.get("primary_view")
        for obs in result.get("parts", []):
            part_id = obs["part_id"]
            entry = evidence.setdefault(part_id, {
                "part_id": part_id,
                "observations": [],
                "aggregated_status": "uncertain",
                "aggregated_level": "unknown",
                "aggregated_confidence": "low",
                "conflicting": False,
            })
            entry["observations"].append({**obs, "photo_id": photo_id, "view_id": primary_view})

    for part_id, entry in evidence.items():
        statuses = [o["status"] for o in entry["observations"]]
        levels = [o["damage_level"] for o in entry["observations"]]
        confidences = [o["confidence"] for o in entry["observations"]]
        views = [o.get("view_id") for o in entry["observations"]]

        # DAMAGE_RECOGNITION_POLICY §3.1 / §4.1: primary-view signals carry
        # more authority than secondary views.  Count votes for damaged/missing
        # restricted to primary-view observations; require at least 1 primary
        # damaged (or ≥2 secondary damaged for low-risk parts) before
        # accepting damage.  This prevents a single secondary-view damaged
        # observation from dominating the aggregated status.
        primary_views = _primary_views_for_part(part_id)
        primary_strong_views = _primary_strong_views_for_part(part_id)
        damaged_votes = sum(
            1 for o in entry["observations"]
            if o["status"] == "damaged"
        )
        missing_votes = sum(
            1 for o in entry["observations"]
            if o["status"] == "missing"
        )
        primary_strong_damaged_votes = sum(
            1 for o in entry["observations"]
            if o["status"] == "damaged"
            and o.get("view_id") in primary_strong_views
        )
        primary_strong_intact_votes = sum(
            1 for o in entry["observations"]
            if o["status"] == "intact"
            and o.get("view_id") in primary_strong_views
        )
        primary_intact_votes = sum(
            1 for o in entry["observations"]
            if o["status"] == "intact"
            and o.get("view_id") in primary_views
        )
        primary_observations = sum(
            1 for o in entry["observations"] if o.get("view_id") in primary_views
        )

        if missing_votes > 0:
            entry["aggregated_status"] = "missing"
        elif primary_strong_intact_votes >= 2 and primary_strong_damaged_votes == 0:
            # §3.1: at least TWO independent strong-primary observations
            # agree intact and no strong-primary damaged signal exists.
            # A single strong-primary intact is not enough to override
            # broader secondary damaged consensus.
            entry["aggregated_status"] = "intact"
        elif primary_strong_damaged_votes >= 1 and primary_strong_intact_votes < 2:
            # Primary view says damaged — accept unless contradicted by
            # strong-primary intact (handled above).
            entry["aggregated_status"] = "damaged"
        elif damaged_votes >= 2 and primary_intact_votes == 0 and primary_observations == 0:
            # No primary observations but ≥2 damaged from secondary views —
            # trust the consensus when primary view didn't reach the part.
            entry["aggregated_status"] = "damaged"
        elif all(s == "intact" for s in statuses):
            entry["aggregated_status"] = "intact"
        else:
            entry["aggregated_status"] = "uncertain"

        # Highest damage level
        level_priority = {"severe": 3, "moderate": 2, "light": 1, "unknown": 0, "none": -1}
        entry["aggregated_level"] = max(levels, key=lambda lvl: level_priority.get(lvl, 0))

        # Lowest confidence as baseline
        conf_priority = {"high": 3, "medium": 2, "low": 1}
        base_confidence = min(confidences, key=lambda c: conf_priority.get(c, 0))

        # Evidence-count boost (schema §4.4)
        consistent_count = sum(
            1 for o in entry["observations"]
            if o["status"] == entry["aggregated_status"]
        )
        entry["aggregated_confidence"] = _boost_confidence(base_confidence, consistent_count)

        # Conflict detection
        unique_statuses = set(statuses)
        if len(unique_statuses) > 1 and not (unique_statuses == {"intact", "uncertain"}):
            entry["conflicting"] = True

    return evidence


_PRIMARY_VIEW_CACHE: Dict[str, set] = {}
_PRIMARY_STRONG_VIEW_CACHE: Dict[str, set] = {}


def _primary_views_for_part(part_id: str) -> set:
    """Return the set of primary view ids (priority <= 1) for a given part.

    Cached in a module-level dict because we query this on every part for
    every assessment.
    """
    if part_id in _PRIMARY_VIEW_CACHE:
        return _PRIMARY_VIEW_CACHE[part_id]
    primary = _load_primary_views(part_id, threshold=1)
    _PRIMARY_VIEW_CACHE[part_id] = primary
    return primary


def _primary_strong_views_for_part(part_id: str) -> set:
    """Return the strict-primary view ids (priority == 0) for a given part."""
    if part_id in _PRIMARY_STRONG_VIEW_CACHE:
        return _PRIMARY_STRONG_VIEW_CACHE[part_id]
    primary = _load_primary_views(part_id, threshold=0)
    _PRIMARY_STRONG_VIEW_CACHE[part_id] = primary
    return primary


def _load_primary_views(part_id: str, threshold: int) -> set:
    """Helper that loads the priority table once per part."""
    try:
        from agents.rules import load_part_view_priority
        priority_table = load_part_view_priority()
    except Exception:
        priority_table = {}
    return {
        view_id
        for view_id, pri in priority_table.get(part_id, {}).items()
        if pri <= threshold
    }


def _boost_confidence(base: str, evidence_count: int) -> str:
    """Schema §4.4: boost confidence when multiple photos agree."""
    if evidence_count >= 3:
        return "high"
    if evidence_count == 2:
        return base
    return base


def _build_region_results(
    view_results: List[Dict[str, Any]],
    part_evidence: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert ViewAgent results into the region_results shape expected by synthesizer."""
    region_results: Dict[str, Dict[str, Any]] = {}

    for result in view_results:
        primary_view = result.get("primary_view")
        if not primary_view:
            continue

        region_entry = region_results.setdefault(primary_view, {
            "view_id": primary_view,
            "region": primary_view,
            "display_name": get_display_name(primary_view),
            "parts": [],
            "uncertain_items": [],
            "cross_view_candidates": [],
            "additional_findings": [],
        })

        for obs in result.get("parts", []):
            part_id = obs["part_id"]
            evidence = part_evidence.get(part_id, {})
            candidate = {
                "part_id": part_id,
                "part_name": obs.get("part_name", ""),
                "status": obs.get("status", "uncertain"),
                "damage_level": obs.get("damage_level", "unknown"),
                "damage_type": obs.get("damage_types", ["none"]),
                "confidence": obs.get("confidence", "low"),
                "evidence_photo": obs.get("evidence_photo", result.get("photo_id")),
                "notes": obs.get("description", ""),
                "_region": primary_view,
                "_aggregated_status": evidence.get("aggregated_status"),
                "_aggregated_confidence": evidence.get("aggregated_confidence"),
                "_conflicting": evidence.get("conflicting", False),
            }
            region_entry["parts"].append(candidate)

    return list(region_results.values())


def _apply_review_overrides(
    synthesized_states: List[PartActualState],
    reviewed_states: List[PartActualState],
) -> List[PartActualState]:
    """Merge reviewer overrides into synthesized part states."""
    by_id = {s.part_id: s for s in synthesized_states}
    for state in reviewed_states:
        if state.part_id in by_id:
            existing = by_id[state.part_id]
            # Conservative merge: prefer more severe status, higher level
            status_priority = {"missing": 4, "damaged": 3, "uncertain": 2, "intact": 1}
            if status_priority.get(state.status.value, 0) >= status_priority.get(existing.status.value, 0):
                by_id[state.part_id] = state
        else:
            by_id[state.part_id] = state
    return list(by_id.values())


def master_assessment_to_legacy_dict(assessment: DamageAssessment) -> Dict[str, Any]:
    """Convert a DamageAssessment to the legacy orchestrator output shape."""
    return assessment.to_legacy_result()
