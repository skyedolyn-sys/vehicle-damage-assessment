"""Assessment orchestrator — compatibility wrapper around MasterAgent.

The multi-subagent workflow has moved into ``agents.master_agent``.  This
module keeps the original public API so existing callers and tests continue
to work:

* ``assessment_orchestrator`` — returns the complete legacy result dict.
* ``assessment_orchestrator_stream`` — yields progress events (SSE-compatible).

The streaming implementation currently yields a single ``final`` event because
MasterAgent does not yet expose fine-grained per-photo progress.  This can be
extended once MasterAgent supports streaming.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Dict, List

from agents.master_agent import master_assessment_agent
from agents.output_validator import _filter_uncertain_items
from agents.rules import load_priority_map
from models.part_state import PartActualState


# Load priority maps from the centralized rules config so the orchestrator's
# module-level constants stay aligned with the synthesizer and topology_comparator.
# These constants are retained for backward compatibility with existing tests.
_PRIORITIES = load_priority_map()
_STATUS_PRIORITY: Dict[Any, int] = _PRIORITIES["status"]
_LEVEL_PRIORITY: Dict[Any, int] = _PRIORITIES["level"]
_CONFIDENCE_PRIORITY: Dict[str, int] = _PRIORITIES["confidence"]


logger = logging.getLogger(__name__)
_orchestrator_file_handler = logging.FileHandler(
    os.path.expanduser("~/vehicle_damage_assessment_orchestrator.log"), mode="a", encoding="utf-8"
)
_orchestrator_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_orchestrator_file_handler.setLevel(logging.INFO)
logger.addHandler(_orchestrator_file_handler)
logger.setLevel(logging.INFO)


async def assessment_orchestrator(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
    plan: Dict[str, Any] | None = None,
    use_face_path: bool = True,
) -> Dict[str, Any]:
    """Run the full assessment workflow and return the legacy result dict.

    Defaults to ``use_face_path=True`` because the face path (face_profiler +
    deterministic camera_side + candidate-part filtering) is the production
    pipeline verified on the 172852 sample (12 true-damaged, 0 false-positive
    flips).  The legacy view path is kept for ablation and explicit
    ``use_face_path=False`` overrides only.
    """
    final_event = None
    async for event in assessment_orchestrator_stream(files, vehicle_info, plan=plan, use_face_path=use_face_path):
        if event.get("type") == "final":
            final_event = event
    if final_event is None:
        raise RuntimeError("Orchestrator workflow did not produce a final event")
    return final_event["result"]


async def assessment_orchestrator_stream(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
    plan: Dict[str, Any] | None = None,
    use_face_path: bool = True,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream assessment workflow events.

    Currently yields only a ``final`` event.  Fine-grained progress events
    can be added once MasterAgent supports streaming.
    """
    async for event in _assessment_orchestrator_impl(files, vehicle_info, plan=plan, use_face_path=use_face_path):
        yield event


async def _assessment_orchestrator_impl(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
    plan: Dict[str, Any] | None = None,
    use_face_path: bool = True,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Delegate to MasterAgent and adapt its output to the legacy shape."""
    from agents.view_mapping import NON_EXTERIOR_VIEWS

    assessment = await master_assessment_agent(files, vehicle_info, plan=plan, use_face_path=use_face_path)
    assessment_result = assessment.to_legacy_result()
    plan = getattr(assessment, "_plan", plan) or {}

    # Rebuild excluded_photos from both legacy view_groups and new photo_classifications.
    excluded_photos: List[Dict[str, Any]] = []
    seen_excluded_ids: set = set()
    if plan:
        for classification in plan.get("photo_classifications", []) or []:
            photo_id = classification.get("photo_id", "")
            category = classification.get("category", "")
            if category in NON_EXTERIOR_VIEWS and photo_id and photo_id not in seen_excluded_ids:
                excluded_photos.append({
                    "id": photo_id,
                    "name": photo_id,
                    "reason": category,
                })
                seen_excluded_ids.add(photo_id)
        for view_id in NON_EXTERIOR_VIEWS:
            for photo in plan.get("view_groups", {}).get(view_id, []):
                photo_id = photo.get("id", "")
                if photo_id and photo_id not in seen_excluded_ids:
                    excluded_photos.append({
                        "id": photo_id,
                        "name": photo.get("name", photo_id),
                        "reason": view_id,
                    })
                    seen_excluded_ids.add(photo_id)

    uncertain_items = _filter_uncertain_items(
        assessment_result.get("uncertain_items", []),
        assessment_result.get("parts", []),
    )

    final_result = {
        "vehicle_info": vehicle_info,
        "vehicle_prior": getattr(assessment, "vehicle_prior", {}),
        "topology": getattr(assessment, "topology_model", {}),
        "plan": plan or {},
        "subagent_results": [],
        "review": {"reviewed_parts": [], "added_uncertain_items": []},
        "excluded_photos": excluded_photos,
        "additional_findings": [],
        **assessment_result,
        "uncertain_items": uncertain_items,
    }

    yield {"type": "final", "result": final_result}


def _merge_two_states(a: PartActualState, b: PartActualState) -> PartActualState:
    """Merge two PartActualState objects conservatively.

    Kept for backward compatibility with tests; new code should use the
    synthesizer's merge logic directly.
    """
    best_status = (
        a.status
        if _STATUS_PRIORITY.get(a.status, 0) >= _STATUS_PRIORITY.get(b.status, 0)
        else b.status
    )

    best_level = (
        a.damage_level
        if _LEVEL_PRIORITY.get(a.damage_level, 0) >= _LEVEL_PRIORITY.get(b.damage_level, 0)
        else b.damage_level
    )

    worst_confidence = (
        a.confidence
        if _CONFIDENCE_PRIORITY.get(a.confidence, 0) <= _CONFIDENCE_PRIORITY.get(b.confidence, 0)
        else b.confidence
    )

    damage_types = list(set(a.damage_types) | set(b.damage_types))
    evidence_photos = list(dict.fromkeys(a.evidence_photos + b.evidence_photos))

    seen_keys: set = set()
    evidence_sources: List[Dict[str, Any]] = []
    for src in a.evidence_sources + b.evidence_sources:
        key = (
            src.get("region", ""),
            src.get("status", ""),
            src.get("damage_level", ""),
            src.get("confidence", ""),
            tuple(sorted(src.get("evidence_photo", []))),
        )
        if key not in seen_keys:
            seen_keys.add(key)
            evidence_sources.append(dict(src))

    notes = "；".join(filter(None, [a.notes, b.notes]))

    return PartActualState(
        part_id=a.part_id,
        part_name=a.part_name,
        part_category=a.part_category,
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
        photo_type=a.photo_type if a.photo_type != "unknown" else b.photo_type,
        evidence_sources=evidence_sources,
    )
