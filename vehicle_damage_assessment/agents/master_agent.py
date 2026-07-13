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
from agents.face_mapping import build_face_prior
from agents.face_profiler import face_profiler_agent
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
    use_face_path: bool = False,
) -> DamageAssessment:
    """Run the full assessment pipeline using the ViewAgent Team architecture.

    Parameters
    ----------
    use_face_path:
        When ``False`` (default) the legacy path is preserved unchanged: each
        ViewAgent self-derives its camera view and applies the Step B flip.
        When ``True`` the new face path is used: a ``face_profiler`` first
        determines each photo's facing + visible-face coverage, deterministic
        ``face_mapping`` flips that into a locked ``camera_side`` and a
        candidate part set, and ViewAgent then only assesses damage inside
        that set (no self-facing, no flip).  This removes the left/right
        mirror-flip failure mode at the root.

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

    # 3b. (face path only) Profile each exterior photo's facing + face coverage,
    # then deterministically derive a locked camera_side + candidate part set.
    face_priors: Dict[str, Dict[str, Any]] = {}
    if use_face_path and exterior_photos:
        profiles = await face_profiler_agent(exterior_photos, vehicle_prior)
        for prof in profiles:
            pid = prof.get("photo_id")
            if pid:
                face_priors[pid] = build_face_prior(pid, prof)
        logger.info(
            "[master] face path: profiled %d photos, %d with a locked camera_side, %d usable",
            len(face_priors),
            sum(1 for fp in face_priors.values() if fp.get("camera_side")),
            sum(1 for fp in face_priors.values() if fp.get("usable")),
        )

    # 4. Dispatch ViewAgent Team in parallel
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

    async def _run_one(photo: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            try:
                prior = face_priors.get(photo.get("id")) if use_face_path else None
                return await view_agent(photo, vehicle_prior, face_prior=prior)
            except Exception as exc:
                logger.warning("[master] view_agent failed for photo_id=%s: %s", photo.get("id"), exc)
                return {"photo_id": photo.get("id"), "primary_view": None, "view_detections": [], "parts": []}

    view_results = await asyncio.gather(*[_run_one(p) for p in exterior_photos])
    # Keep a result when it has kept parts OR unmapped parts — the latter carry
    # the model's out-of-catalog / out-of-scope observations, which downstream
    # diagnostics (and the facing consensus) must not silently lose.
    view_results = [
        r for r in view_results if r.get("parts") or r.get("unmapped_parts")
    ]

    # 4b. (face path only) Cross-photo facing consensus.  A low-confidence
    # facing may simply be wrong (a front-windshield close-up misread as
    # rear).  When high-confidence photos establish a dominant facing/side,
    # low-confidence photos defer to it: we re-interpret the suspect photo
    # under the consensus frame and keep only the damage observations whose
    # location is self-consistent with the consensus damaged region.
    if use_face_path and face_priors:
        _apply_facing_consensus(face_priors, view_results)

    # Persist each exterior photo's resolved primary view into the plan so the
    # per-photo camera_position is auditable downstream (screen/vehicle-side
    # mismatches are a known error source; without this the only record is the
    # view log).  ``plan["photo_views"]`` maps photo_id -> primary_view.  On the
    # face path the primary view comes from the deterministic facing (already
    # flip-free), so this record is also the cross-photo camera_side consensus
    # input.
    plan["photo_views"] = [
        {"photo_id": r.get("photo_id"), "view_id": r.get("primary_view")}
        for r in view_results
        if r.get("photo_id")
    ]
    if use_face_path:
        plan["photo_faces"] = [
            {
                "photo_id": pid,
                "facing": fp.get("facing"),
                "camera_side": fp.get("camera_side"),
                "assignable_faces": fp.get("assignable_faces"),
            }
            for pid, fp in face_priors.items()
        ]

    # Persist the per-photo unmapped parts so out-of-catalog / out-of-scope
    # observations are auditable downstream instead of vanishing at the
    # normalization boundary.
    unmapped = [
        {"photo_id": r.get("photo_id"), "unmapped_parts": r.get("unmapped_parts")}
        for r in view_results
        if r.get("unmapped_parts")
    ]
    if unmapped:
        plan["unmapped_parts"] = unmapped

    # 5. Aggregate into PartEvidence, run side-consistency check, then
    # build region_results.  The side check downgrades mismatched
    # observations' confidence (never rewrites the suffix) so the
    # primary-strong gating in _aggregate_part_evidence has a chance
    # of rejecting them without breaking legitimately-right observations.
    part_evidence = _aggregate_part_evidence(view_results)
    side_violations = _check_side_consistency(part_evidence)
    if side_violations:
        logger.info(
            "[master] side-consistency: %d observation(s) downgraded for "
            "screen/vehicle-side mismatch",
            side_violations,
        )
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
        elif (
            damaged_votes >= 1
            and primary_strong_intact_votes < 2
            and any(
                o["status"] == "damaged" and o.get("view_id") in primary_views
                for o in entry["observations"]
            )
        ):
            # A primary-view (priority <= 1, not necessarily strong) damaged
            # observation convicts the part unless at least two strong-primary
            # intact observations disagree.  A single strong-primary intact is
            # not a consensus (§3.1), so it cannot alone override a primary
            # damaged signal.  This is the conflict case the aggregator must
            # still surface as damaged (with ``conflicting`` set below).
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


def _part_side(part_id: str) -> str:
    """Return the vehicle side a part belongs to: left / right / center."""
    if part_id.endswith("_left"):
        return "left"
    if part_id.endswith("_right"):
        return "right"
    return "center"


def _apply_facing_consensus(
    face_priors: Dict[str, Dict[str, Any]],
    view_results: List[Dict[str, Any]],
) -> None:
    """Cross-photo facing consensus via damage-location self-consistency.

    A low-confidence facing may simply be *wrong* — e.g. a front-windshield
    close-up misread as ``rear`` because shattered glass looks the same front
    and back.  The vehicle is a rigid body, so one batch of photos should
    agree on which side/region is damaged.  This routine:

    1. Lets high-confidence photos vote the consensus damaged side(s) from
       where THEIR damage actually landed (not from the facing label — the
       label can be wrong, the damaged parts are the ground signal).
    2. For each low-confidence (or unusable) photo, checks whether its OWN
       damaged observations are self-consistent with that consensus: does the
       side its damaged parts fall on match a consensus damaged side?
    3. Keeps consistent observations; soft-downgrades (confidence=low) the
       ones that contradict the consensus — those are the likely mis-faced
       false positives.  A photo whose damage IS consistent keeps full weight,
       which "rescues" a mis-faced photo's real damage instead of dropping it.

    This never touches high-confidence photos and never rewrites a part id —
    it only downgrades confidence on contradictory observations, so the
    downstream consensus rules (which already require high confidence or
    corroboration) stop a mis-faced close-up from convicting a part alone.
    """
    damaged_by_photo: Dict[str, List[Dict[str, Any]]] = {}
    for r in view_results:
        pid = r.get("photo_id")
        if not pid:
            continue
        damaged_by_photo[pid] = [
            p for p in r.get("parts", []) if p.get("status") in ("damaged", "missing")
        ]

    # 1. Consensus damaged sides, voted by USABLE (high/medium-confidence,
    #    clearly-faced) photos from the sides their damage actually fell on.
    side_votes: Dict[str, int] = {}
    for pid, fp in face_priors.items():
        if not fp.get("usable"):
            continue
        for obs in damaged_by_photo.get(pid, []):
            side = _part_side(obs.get("part_id", ""))
            if side in ("left", "right"):
                side_votes[side] = side_votes.get(side, 0) + 1
    if not side_votes:
        return  # no reliable damaged-side signal; nothing to check against
    consensus_sides = {s for s, n in side_votes.items() if n == max(side_votes.values())}

    # 2+3. For low-confidence / unusable photos, downgrade damage that
    #      contradicts the consensus damaged side.
    downgraded = 0
    for pid, fp in face_priors.items():
        if fp.get("usable"):
            continue  # trusted photo — leave its observations alone
        for obs in damaged_by_photo.get(pid, []):
            side = _part_side(obs.get("part_id", ""))
            if side == "center":
                continue  # center parts (hood/roof/windshield) carry no side signal
            if side not in consensus_sides:
                obs["confidence"] = "low"
                obs["model_confidence_score"] = min(
                    float(obs.get("model_confidence_score", 0.5)), 0.4
                )
                obs["_consensus_downgraded"] = True
                downgraded += 1
    if downgraded:
        logger.info(
            "[master] facing consensus sides=%s downgraded %d contradictory "
            "low-confidence observation(s)",
            sorted(consensus_sides), downgraded,
        )


def _check_side_consistency(evidence: Dict[str, Dict[str, Any]]) -> int:
    """Downgrade confidence on damaged observations whose view is
    outside the part's primary view set.

    Replaces the previous mirror-flip post-processor, which had a
    self-referential fallback (``screen_side`` was derived from
    ``part_id.endswith`` — the same vehicle suffix it was being
    compared against) and fired on every side-bearing observation
    regardless of whether anything was actually wrong.

    Why this is the right shape:

    * Per-part *primary* view set is homogeneous on suffix
      (``headlight_front_left`` → ``{front, front_left, left}``, all
      ``_left``-bearing).  Therefore the *only* "view that disagrees
      with the part's vehicle side" we could potentially see is
      **already not in the primary set** — i.e. the model emitted
      a side-bearing part from a view the part does not canonically
      belong to.  That is the exact side-confusion failure mode.
    * In ``_aggregate_part_evidence`` such observations cannot
      contribute to ``primary_strong_damaged_votes`` (already filtered
      out by the primary set), but they DO contribute to
      ``damaged_votes``, and a count of ``damaged_votes >= 2`` with
      no primary observations is enough to set the part to
      ``damaged`` via the secondary-consensus fallback.  That is the
      pathway this routine blocks.
    * The fix is **only** to downgrade ``confidence`` to ``low`` on
      those out-of-primary damaged observations.  The part's suffix
      and the observation's status are **never** rewritten — the
      DAMAGE_RECOGNITION_POLICY guardrail (do not silently flip
      ``_right`` to ``_left``) holds.
    * No mirror table is consulted.  The check operates entirely on
      the part's own primary view set, which is the authoritative
      answer to "what views is this part native to?".

    Returns
    -------
    int
        Number of downgraded observations.
    """
    violations = 0
    for part_id, entry in evidence.items():
        if not (part_id.endswith("_left") or part_id.endswith("_right")):
            continue  # non-side parts

        primary_views = _primary_views_for_part(part_id)
        if not primary_views:
            continue  # part has no primary view set; skip

        for obs in entry["observations"]:
            view_id = obs.get("view_id")
            if not view_id:
                continue
            if obs.get("status") != "damaged":
                continue
            if view_id in primary_views:
                continue  # in-primary — trust the primary gating

            # Out-of-primary damaged observation.  Downgrade so it
            # cannot drive the secondary-consensus path.
            violations += 1
            if not obs.get("_side_mismatch"):
                obs["_side_mismatch"] = True
                obs["_side_mismatch_view"] = view_id
                obs["_side_mismatch_reason"] = (
                    f"part={part_id} has a damaged observation from "
                    f"view={view_id!r}, which is not in the part's "
                    f"primary view set {sorted(primary_views)}; "
                    f"downgrading to confidence=low to block the "
                    f"secondary-consensus fallback"
                )
                if obs.get("confidence") != "low":
                    obs["confidence"] = "low"
            entry["conflicting"] = True
            logger.info(
                "[master.side] downgrade part=%s view=%s "
                "(view not in primary %s)",
                part_id, view_id, sorted(primary_views),
            )

    return violations


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
    """Helper that loads the priority table once per part.

    Merges two sources from ``view_weights.yaml`` so the
    side-consistency check sees the same authoritative "which views
    are native to this part?" answer that the rest of the pipeline
    uses:

    1. ``part_view_priority`` (dict form) — primary strong views are
       those with priority ``<= threshold`` (0 for strict, 1 for
       primary).
    2. ``primary_view`` (list form) — a legacy compact list of
       canonical views for parts that do not have a full priority
       table (e.g. ``pillar_a_left``).  These are treated as
       priority 0.
    """
    try:
        from agents.rules import load_part_view_priority, load_view_weights
        priority_table = load_part_view_priority()
        view_weights = load_view_weights()
    except Exception:
        priority_table, view_weights = {}, {}

    result: set = {
        view_id
        for view_id, pri in priority_table.get(part_id, {}).items()
        if pri <= threshold
    }

    # Merge in the ``primary_view`` list block at priority 0.
    if threshold >= 0:
        primary_list = view_weights.get("primary_view", {}).get(part_id, []) or []
        result.update(primary_list)

    return result


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
                # Carry the model's self-reported confidence score so the
                # synthesizer can distinguish a checklist backfill (score 0.0 —
                # the model never observed the part) from a real observation.
                # Without this the backfill signal was dropped at this boundary
                # and topology Rules 9-11 could not tell a backfill from a real
                # observation (172852 pillar_b_right systematic FP).
                "model_confidence_score": obs.get("model_confidence_score", 1.0),
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
