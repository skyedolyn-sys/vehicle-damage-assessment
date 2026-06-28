"""Assessment orchestrator — coordinates planner, vision subagents and reviewer.

This is the top-level agent that replaces the older
``vehicle_prior -> photo_locator -> damage_assessor`` linear pipeline with a
multi-subagent workflow:

1. Get vehicle prior and topology.
2. Run ``planner_agent`` to assign a canonical view to every photo.
3. Dispatch ``vision_subagent`` calls concurrently, one per view group.
4. Run ``reviewer_subagent`` to resolve conflicts and identify coverage gaps.
5. Synthesise final part states and run topology comparison.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import logging
import traceback

from agents import build_vehicle_topology, vehicle_prior_agent
from agents.planner_agent import planner_agent
from agents.reviewer_subagent import reviewer_subagent
from agents.synthesizer import synthesizer_agent
from agents.view_mapping import (
    EXTERIOR_VIEWS,
    NON_EXTERIOR_VIEWS,
    get_regions_for_view,
    is_exterior_view,
)
from agents.vision_subagent import vision_subagent
from config import MAX_CONCURRENT_API_CALLS
from models.topology import VehicleTopology


logger = logging.getLogger(__name__)


async def assessment_orchestrator(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
    plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the full multi-subagent assessment workflow.

    Parameters
    ----------
    files:
        Photo dicts with ``id``, ``path`` and optionally ``url``.
    vehicle_info:
        ``{"brand": ..., "model": ..., "year": ...}``.
    plan:
        Optional pre-computed planner result.  When provided, the orchestrator
        skips its internal planner call, avoiding redundant LLM invocations in
        streaming API contexts.

    Returns
    -------
    dict
        Assessment result compatible with ``output_validator``.
    """
    # ------------------------------------------------------------------
    # Step 1: Vehicle prior + topology
    # ------------------------------------------------------------------
    vehicle_prior = await vehicle_prior_agent(vehicle_info)
    topology = build_vehicle_topology(vehicle_info, vehicle_prior)

    # ------------------------------------------------------------------
    # Step 2: Planner assigns views to every photo (reuse if provided)
    # ------------------------------------------------------------------
    if plan is None:
        plan = await planner_agent(files, vehicle_prior)

    # ------------------------------------------------------------------
    # Step 3: Dispatch vision subagents concurrently, one per view group
    # ------------------------------------------------------------------
    view_groups = plan.get("view_groups", {})
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

    def _expected_part_count(view_id: str, topology: VehicleTopology) -> int:
        """Estimate how many distinct parts a healthy subagent result should contain."""
        regions = get_regions_for_view(view_id)
        expected_ids: set = set()
        for region in regions:
            for part_id in topology.regions.get(region, []):
                expected_ids.add(part_id)
        return len(expected_ids)

    async def run_view_subagent(view_id: str) -> Dict[str, Any]:
        async with semaphore:
            try:
                return await vision_subagent(view_id, view_groups.get(view_id, []), vehicle_prior, topology)
            except Exception as exc:
                logger.error("Vision subagent failed for %s: %s", view_id, exc)
                logger.debug(traceback.format_exc())
                raise

    def _is_anomaly(result: Dict[str, Any], view_id: str) -> bool:
        """Detect empty or suspiciously incomplete subagent output."""
        parts = result.get("parts", [])
        states = result.get("part_actual_states", [])
        if not parts and not states:
            return True
        expected = _expected_part_count(view_id, topology)
        actual = len(states) if states else len(parts)
        # Retry when fewer than half of expected parts are reported.
        if expected > 0 and actual < expected / 2:
            logger.warning(
                "Anomaly detected for %s: %d/%d parts returned",
                view_id, actual, expected,
            )
            return True
        return False

    # Only dispatch for views that actually have photos and are exterior.
    views_to_run = [
        view_id for view_id in EXTERIOR_VIEWS
        if view_groups.get(view_id)
    ]

    subagent_tasks = [asyncio.create_task(run_view_subagent(view_id), name=view_id) for view_id in views_to_run]
    subagent_results = await asyncio.gather(*subagent_tasks, return_exceptions=True)

    successful_results: List[Dict[str, Any]] = []
    retry_views: List[str] = []
    for view_id, result in zip(views_to_run, subagent_results):
        if isinstance(result, Exception):
            logger.warning("Vision subagent %s failed: %s", view_id, result)
            retry_views.append(view_id)
            continue
        if _is_anomaly(result, view_id):
            retry_views.append(view_id)
        successful_results.append(result)

    # Retry failed or anomalous views once, sequentially to reduce API pressure.
    if retry_views:
        logger.info("Retrying vision subagents sequentially: %s", retry_views)
        for view_id in retry_views:
            try:
                retry_result = await vision_subagent(view_id, view_groups.get(view_id, []), vehicle_prior, topology)
                if _is_anomaly(retry_result, view_id):
                    logger.warning("Retry for %s still anomalous; keeping best effort", view_id)
                # Replace the original anomalous result if retry is healthier.
                original_index = next(
                    (i for i, r in enumerate(successful_results) if r.get("view_id") == view_id), None
                )
                if original_index is not None:
                    successful_results[original_index] = retry_result
                else:
                    successful_results.append(retry_result)
            except Exception as exc:
                logger.error("Retry failed for %s: %s", view_id, exc)

    # ------------------------------------------------------------------
    # Step 4: Reviewer checks conflicts, gaps and low-confidence items
    # ------------------------------------------------------------------
    review = await reviewer_subagent(successful_results, plan, vehicle_prior)

    # ------------------------------------------------------------------
    # Step 5: Synthesise region results into unified part states
    # ------------------------------------------------------------------
    # Convert subagent results into the shape expected by synthesizer_agent.
    region_results: List[Dict[str, Any]] = []
    for result in successful_results:
        region_results.append(
            {
                "region": result.get("view_id", "unknown"),
                "parts": result.get("parts", []),
                "uncertain_items": result.get("uncertain_items", []),
            }
        )

    # Merge uncertain items from reviewer.
    all_uncertain_items: List[Dict[str, Any]] = []
    for result in successful_results:
        all_uncertain_items.extend(result.get("uncertain_items", []))
    all_uncertain_items.extend(review.get("added_uncertain_items", []))

    # Use the synthesizer for deterministic merging with traceability.
    merged = synthesizer_agent(region_results, vehicle_prior, topology)

    # Apply reviewer overrides when the reviewer produced revised part states.
    for override in review.get("reviewed_part_actual_states", []):
        if not isinstance(override, PartActualState):
            continue
        for part in merged.get("parts", []):
            if part.get("part_id") == override.part_id:
                part["status"] = override.status.value
                part["damage_level"] = override.damage_level.value
                part["confidence"] = override.confidence
                part["notes"] = override.notes
                part["evidence_photo"] = list(override.evidence_photos)
                break

    merged["uncertain_items"] = all_uncertain_items

    # Collect photos excluded from exterior assessment for transparency.
    excluded_photos: List[Dict[str, Any]] = []
    seen_excluded_ids: set = set()
    for view_id in NON_EXTERIOR_VIEWS:
        for photo in plan.get("view_groups", {}).get(view_id, []):
            photo_id = photo.get("id", "")
            if photo_id and photo_id not in seen_excluded_ids:
                excluded_photos.append(
                    {
                        "id": photo_id,
                        "name": photo.get("name", photo_id),
                        "reason": view_id,
                    }
                )
                seen_excluded_ids.add(photo_id)

    # Collect additional_findings from all subagents for the final report.
    all_additional_findings: List[Dict[str, Any]] = []
    for result in successful_results:
        all_additional_findings.extend(result.get("additional_findings", []))

    return {
        "vehicle_info": vehicle_info,
        "vehicle_prior": vehicle_prior,
        "topology": topology.to_dict(),
        "plan": plan,
        "subagent_results": successful_results,
        "review": review,
        "excluded_photos": excluded_photos,
        "additional_findings": all_additional_findings,
        **merged,
    }


def _merge_subagent_results(
    subagent_results: List[Dict[str, Any]],
    review: Dict[str, Any],
    topology: VehicleTopology,
) -> Dict[str, Any]:
    """Merge vision subagent outputs and reviewer overrides into a flat result.

    The returned dict has the same ``{"parts": [...], "uncertain_items": [...]}``
    shape that ``output_validator.validate_and_enrich`` expects.
    """
    from config import PARTS_BY_ID
    from models.part_state import PartActualState

    # Collect all PartActualState objects keyed by part_id.
    state_by_id: Dict[str, PartActualState] = {}

    for result in subagent_results:
        for state in result.get("part_actual_states", []):
            if not isinstance(state, PartActualState):
                continue
            existing = state_by_id.get(state.part_id)
            if existing is None:
                state_by_id[state.part_id] = state
            else:
                state_by_id[state.part_id] = _merge_two_states(existing, state)

    # Apply reviewer overrides.
    for override in review.get("reviewed_part_actual_states", []):
        if isinstance(override, PartActualState) and override.part_id:
            state_by_id[override.part_id] = override

    # Ensure every topology node has a state.
    parts: List[PartActualState] = []
    for node_id in topology.nodes:
        if node_id in state_by_id:
            parts.append(state_by_id[node_id])
        else:
            node = topology.get_node(node_id)
            parts.append(
                PartActualState(
                    part_id=node_id,
                    part_name=node.node_name if node else PARTS_BY_ID.get(node_id, {}).get("part_name", node_id),
                    region=node.region if node else "",
                    side=node.side if node else "",
                    status=Status.UNCERTAIN,
                    damage_level=DamageLevel.UNKNOWN,
                    standard_exists=True,
                    actual_visible=False,
                    actual_present=False,
                    confidence="low",
                    notes="该部件所在区域无照片覆盖，无法判断",
                )
            )

    return {
        "parts": [p.to_legacy_dict() for p in parts],
        "uncertain_items": [],
    }


def _merge_two_states(a: Any, b: Any) -> Any:
    """Merge two PartActualState objects conservatively."""
    from models.part_state import PartActualState, Status, DamageLevel

    # Status priority: missing > damaged > uncertain > intact
    status_priority = {
        Status.MISSING: 4,
        Status.DAMAGED: 3,
        Status.UNCERTAIN: 2,
        Status.INTACT: 1,
        Status.NOT_APPLICABLE: 0,
    }
    best_status = a.status if status_priority.get(a.status, 0) >= status_priority.get(b.status, 0) else b.status

    # Damage level priority
    level_priority = {
        DamageLevel.SEVERE: 4,
        DamageLevel.MODERATE: 3,
        DamageLevel.LIGHT: 2,
        DamageLevel.UNKNOWN: 1,
        DamageLevel.NONE: 0,
    }
    best_level = a.damage_level if level_priority.get(a.damage_level, 0) >= level_priority.get(b.damage_level, 0) else b.damage_level

    # Confidence: lower is worse
    confidence_priority = {"high": 2, "medium": 1, "low": 0}
    worst_confidence = (
        a.confidence
        if confidence_priority.get(a.confidence, 0) <= confidence_priority.get(b.confidence, 0)
        else b.confidence
    )

    damage_types = list(set(a.damage_types) | set(b.damage_types))
    evidence_photos = list(dict.fromkeys(a.evidence_photos + b.evidence_photos))
    notes = "；".join(filter(None, [a.notes, b.notes]))

    return PartActualState(
        part_id=a.part_id,
        part_name=a.part_name,
        region=a.region,
        side=a.side,
        status=best_status,
        damage_level=best_level,
        damage_types=damage_types,
        standard_exists=a.standard_exists,
        actual_visible=a.actual_visible or b.actual_visible,
        actual_present=a.actual_present and b.actual_present,
        confidence=worst_confidence,
        evidence_photos=evidence_photos,
        notes=notes,
    )


# Avoid circular import issues by importing Status/DamageLevel at the bottom.
from models.part_state import Status, DamageLevel
