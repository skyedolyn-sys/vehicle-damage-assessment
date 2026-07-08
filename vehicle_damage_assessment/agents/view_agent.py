"""ViewAgent — per-photo view detection and damage observation.

Each ViewAgent instance processes a single exterior photo:
- Determines the primary exterior view (and alternate view confidences).
- Evaluates every part visible from that view according to the fixed 33-part
  catalog and 9-view mapping in ``agents/view_mapping.py``.
- Returns structured observations with calibrated confidence.

The agent is deliberately stateless: all context comes from the photo, the
fixed catalog, and the vehicle prior (vehicle name / specs).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agents.minimax_client import call_minimax, build_image_content, extract_json
from agents.rules import render_prompt_template
from agents.view_mapping import get_parts_for_view, get_display_name
from config import IMAGE_MAX_WIDTH, MINIMAX_MODEL, PARTS_BY_ID
from models import PartActualState, Status, DamageLevel

logger = logging.getLogger(__name__)

_view_file_handler = logging.FileHandler(
    os.path.expanduser("~/vehicle_damage_assessment_view.log"), mode="a", encoding="utf-8"
)
_view_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_view_file_handler.setLevel(logging.INFO)
logger.addHandler(_view_file_handler)
logger.setLevel(logging.INFO)

# Dedicated trace log for per-photo verdicts so the human auditor can read
# 32 one-line summaries without scrolling through other diagnostic output.
_view_trace_handler = logging.FileHandler(
    os.path.expanduser("~/vehicle_damage_assessment_view_trace.log"), mode="w", encoding="utf-8"
)
_view_trace_handler.setFormatter(logging.Formatter("%(message)s"))
_view_trace_handler.setLevel(logging.INFO)
logger.addHandler(_view_trace_handler)


#: Status ordering for conservative aggregation.
_STATUS_PRIORITY = {
    "missing": 4,
    "damaged": 3,
    "uncertain": 2,
    "intact": 1,
}

#: Damage level ordering (higher = more severe).
_LEVEL_PRIORITY = {
    "severe": 4,
    "moderate": 3,
    "light": 2,
    "unknown": 1,
    "none": 0,
}

#: Confidence ordering (higher = more confident).
_CONFIDENCE_PRIORITY = {
    "high": 3,
    "medium": 2,
    "low": 1,
}

#: Canonical damage types allowed in ViewAgent output.
_DAMAGE_TYPES = {
    "none",
    "scratch",
    "dent",
    "deformation",
    "crack",
    "breakage",
    "paint_loss",
    "corrosion",
}


async def view_agent(
    photo: Dict[str, Any],
    vehicle_prior: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single-photo view + damage assessment.

    Parameters
    ----------
    photo:
        ``{"id": str, "path": str, "category": "exterior"}``
    vehicle_prior:
        Output from ``vehicle_prior_agent``.

    Returns
    -------
    dict
        ``ViewAgentResult`` shape (see ``docs/schema-viewagent-team.md``).
    """
    photo_id = photo.get("id") or photo.get("photo_id") or "unknown"
    vehicle_name = vehicle_prior.get("vehicle", "待识别车辆")

    logger.info("[view_agent] start photo_id=%s", photo_id)

    image_content = build_image_content(photo.get("path") or photo.get("url"), max_width=IMAGE_MAX_WIDTH)

    # Two-role prompt architecture (per the team's SP separation rule):
    # - system role: the view_agent SP (identity, methodology, output protocol)
    # - user role: the task input (catalog, few-shot, vehicle name)
    # The image lives in the user role because M3-Vision takes its cues from
    # the same role as the rules, so the two must co-occur.
    system_prompt = render_prompt_template("view_agent_system")
    task_prompt = render_prompt_template(
        "view_agent_task",
        photo_id=photo_id,
        vehicle_name=vehicle_name,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": task_prompt},
                image_content,
            ],
        },
    ]

    raw_text = await call_minimax(
        messages=messages,
        temperature=0.2,
        max_tokens=4000,
        model=MINIMAX_MODEL,
        response_format={"type": "json_object"},
        # MiniMax M3 default reasoning spends the full max_tokens budget
        # on a single <think>...</think> block and leaves no room for
        # the JSON answer (see minimax_diagnostic_*.txt for failures).
        # Pinning reasoning_effort="low" caps the thinking so the model
        # has to commit to a structured output.
        reasoning_effort="low",
    )

    raw = extract_json(raw_text)
    if not isinstance(raw, dict):
        logger.warning("[view_agent] photo_id=%s returned non-dict: %s", photo_id, type(raw))
        raw = {}

    # Diagnostic dump: capture the model's own view_detections + descriptions
    # so we can audit whether it noticed "右后视镜朝外/车标在右" cues.
    _dump_minimax_raw(photo_id, raw)

    result = _normalize_view_agent_result(photo_id, raw)
    result = _backfill_missing_parts(result)
    result = _calibrate_result_confidence(result)

    # Per-photo debug dump (so a human can audit each ViewAgent verdict).
    _dump_per_photo_verdict(photo_id, result)

    logger.info(
        "[view_agent] done photo_id=%s primary_view=%s parts=%d",
        photo_id,
        result.get("primary_view"),
        len(result.get("parts", [])),
    )
    return result


def _dump_minimax_raw(photo_id: str, raw: Dict[str, Any]) -> None:
    """Dump the raw M3-Vision verdict to a dedicated trace log.

    For calibration, we need to know:
    - which views the model believes are visible (view_detections list)
    - which view it picked as primary and why
    - what textual evidence it gave in part descriptions (so we can audit
      whether it actually noticed right-mirror-facing-out, badge-on-right-side,
      wheel-facing-camera, etc.)
    """
    parts = raw.get("parts", []) or []
    detections = raw.get("view_detections", []) or []
    primary = raw.get("primary_view")

    # Build "<view>=<score>" summary, sorted by score descending.
    det_summary = ",".join(
        f"{d.get('view_id')}={d.get('confidence_score'):.2f}"
        for d in detections
    ) or "—"

    # Concatenate damaged-only descriptions so a human can scan evidence
    # in the trace log.  Intact descriptions are omitted to keep the log short.
    damaged_evidence = []
    for p in parts:
        if p.get("status") == "damaged":
            damaged_evidence.append(
                f"{p.get('part_id')}={p.get('description', '')[:120]}"
            )
    damaged_str = " | ".join(damaged_evidence) or "—"

    logger.info(
        "[view_agent.raw] %s | primary=%s | detections=[%s] | damaged=[%s]",
        photo_id,
        primary,
        det_summary,
        damaged_str,
    )


def _dump_per_photo_verdict(photo_id: str, result: Dict[str, Any]) -> None:
    """Write a one-line-per-damage observation so the human can scan 32 photos.

    Format: ``<photo_id> | view=<primary> | <part_id>:<status>:<level>[:<conf>] ...``
    Only damaged/missing observations are listed; intact parts are summarised
    by count.  Written to a dedicated log so it does not pollute normal logs.
    """
    primary = result.get("primary_view") or "unknown"
    damaged: list[str] = []
    intact_count = 0
    for obs in result.get("parts", []):
        if obs.get("status") in ("damaged", "missing"):
            damaged.append(
                f"{obs['part_id']}={obs.get('status', '?')}/{obs.get('damage_level', '?')}/{obs.get('confidence', '?')}"
            )
        elif obs.get("status") == "intact":
            intact_count += 1
    damaged_str = ", ".join(damaged) if damaged else "—"
    logger.info(
        "[view_agent.trace] %s | view=%s | damaged=[%s] | intact=%d",
        photo_id,
        primary,
        damaged_str,
        intact_count,
    )


def _normalize_view_agent_result(photo_id: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize raw LLM output into the canonical ViewAgentResult shape."""
    view_detections = _normalize_view_detections(raw.get("view_detections"))
    primary_view = _resolve_primary_view(view_detections, raw.get("primary_view"))

    normalized_parts: List[Dict[str, Any]] = []
    for part in raw.get("parts", []) or []:
        if not isinstance(part, dict):
            continue
        normalized = _normalize_part_observation(part, photo_id)
        if normalized:
            normalized_parts.append(normalized)

    return {
        "photo_id": photo_id,
        "primary_view": primary_view,
        "view_detections": view_detections,
        "parts": normalized_parts,
        "raw": raw,
    }


def _normalize_view_detections(detections: Any) -> List[Dict[str, Any]]:
    """Normalize view_detections list and ensure exactly one is_primary=True."""
    if not isinstance(detections, list):
        return []

    normalized = []
    for d in detections:
        if not isinstance(d, dict):
            continue
        view_id = _canonical_part_id_alias(d.get("view_id", ""))
        # Some models may emit legacy view ids; map them to new short ids.
        from agents.view_mapping import _normalize_view_id
        view_id = _normalize_view_id(view_id)
        score = float(d.get("confidence_score", 0.0))
        normalized.append({
            "view_id": view_id,
            "confidence_score": score,
            "is_primary": bool(d.get("is_primary", False)),
        })

    # If no primary flag set, mark the highest-confidence detection.
    if normalized and not any(d.get("is_primary") for d in normalized):
        best = max(normalized, key=lambda x: x["confidence_score"])
        best["is_primary"] = True

    return sorted(normalized, key=lambda x: x["confidence_score"], reverse=True)


def _resolve_primary_view(
    view_detections: List[Dict[str, Any]],
    raw_primary: Any,
) -> Optional[str]:
    """Pick the primary view from detections or raw field."""
    from agents.view_mapping import _normalize_view_id

    if view_detections:
        primary = next((d for d in view_detections if d.get("is_primary")), view_detections[0])
        if primary.get("confidence_score", 0.0) >= 0.5:
            return _normalize_view_id(primary["view_id"])
        return None

    if isinstance(raw_primary, str) and raw_primary:
        return _normalize_view_id(raw_primary)

    return None


def _normalize_part_observation(part: Dict[str, Any], photo_id: str) -> Optional[Dict[str, Any]]:
    """Normalize a single part observation."""
    raw_part_id = part.get("part_id")
    if not raw_part_id:
        return None

    part_id = _canonical_part_id_alias(str(raw_part_id))
    if part_id not in PARTS_BY_ID:
        logger.warning("[view_agent] unknown part_id=%s from photo_id=%s", part_id, photo_id)
        return None

    part_info = PARTS_BY_ID[part_id]
    status = _normalize_enum(part.get("status"), {"intact", "damaged", "missing", "uncertain"}, "uncertain")

    damage_level = _normalize_enum(
        part.get("damage_level"),
        {"none", "light", "moderate", "severe", "unknown"},
        "unknown",
    )
    damage_types = _normalize_damage_types(part.get("damage_types"))

    # Enforce consistency between status and damage fields.
    if status == "intact":
        damage_level = "none"
        damage_types = ["none"]
    elif status == "missing":
        damage_level = "severe"
        if "breakage" not in damage_types:
            damage_types = ["breakage"]
    elif status == "damaged" and damage_level == "none":
        damage_level = "light"
    elif status == "uncertain" and damage_level == "none":
        damage_level = "unknown"

    model_score = float(part.get("model_confidence_score", 0.5))
    confidence = _normalize_enum(
        part.get("confidence"),
        {"high", "medium", "low"},
        "low",
    )

    return {
        "part_id": part_id,
        "part_name": part_info["part_name"],
        "status": status,
        "damage_level": damage_level,
        "damage_types": damage_types,
        "model_confidence_score": model_score,
        "confidence": confidence,
        "description": str(part.get("description", "")).strip(),
        "evidence_photo": photo_id,
    }


def _backfill_missing_parts(result: Dict[str, Any]) -> Dict[str, Any]:
    """Fill any parts from the primary view checklist that the LLM omitted."""
    primary_view = result.get("primary_view")
    if not primary_view:
        return result

    checklist = get_parts_for_view(primary_view)
    seen = {p["part_id"] for p in result.get("parts", [])}

    for part_id in checklist:
        if part_id in seen:
            continue
        part_info = PARTS_BY_ID.get(part_id)
        if not part_info:
            continue
        result["parts"].append({
            "part_id": part_id,
            "part_name": part_info["part_name"],
            "status": "uncertain",
            "damage_level": "unknown",
            "damage_types": ["none"],
            "model_confidence_score": 0.0,
            "confidence": "low",
            "description": "该部件在照片中未被识别到，按视角清单补齐为uncertain",
            "evidence_photo": result.get("photo_id"),
        })

    return result


def _calibrate_result_confidence(result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply in-agent confidence calibration to every part observation."""
    for part in result.get("parts", []):
        part["confidence"] = _score_to_level(_calibrate_damage_confidence(part))
    return result


def _calibrate_damage_confidence(raw: Dict[str, Any]) -> float:
    """Schema §4.2: combine model score, description, damage types, and status."""
    signals = [float(raw.get("model_confidence_score", 0.5))]

    desc = raw.get("description", "")
    if any(kw in desc for kw in ["明显", "严重", "大面积", "变形", "凹陷", "断裂"]):
        signals.append(0.9)
    elif any(kw in desc for kw in ["轻微", "小", "局部", "细微"]):
        signals.append(0.6)
    else:
        signals.append(0.4)

    types = [t for t in raw.get("damage_types", []) if t != "none"]
    if len(types) >= 2:
        signals.append(0.85)
    elif len(types) == 1:
        signals.append(0.7)
    else:
        signals.append(0.5)

    status = raw.get("status")
    if status == "intact" and "无" in desc:
        signals.append(0.9)
    elif status == "damaged" and types:
        signals.append(0.8)
    else:
        signals.append(0.4)

    return sum(signals) / len(signals)


def _score_to_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _normalize_damage_types(raw: Any) -> List[str]:
    """Normalize damage_types to the canonical enum."""
    if raw is None:
        return ["none"]
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return ["none"]

    normalized = []
    for item in items:
        item = str(item).strip().lower()
        if item in _DAMAGE_TYPES:
            normalized.append(item)
        elif item in {"paint_damage", "paintloss"}:
            normalized.append("paint_loss")
        elif item in {"broken", "shattered", "glass_breakage"}:
            normalized.append("breakage")
        elif item == "tear":
            normalized.append("crack")
        elif item == "missing":
            normalized.append("breakage")

    if not normalized:
        return ["none"]
    return normalized


def _normalize_enum(value: Any, allowed: set, default: str) -> str:
    if isinstance(value, str) and value.lower() in {a.lower() for a in allowed}:
        return value.lower()
    return default


def _canonical_part_id_alias(raw_name: str) -> str:
    """Map a raw part id/name to its canonical part id."""
    from agents.rules import resolve_part_alias
    return resolve_part_alias(str(raw_name))


def view_agent_result_to_part_actual_states(
    result: Dict[str, Any],
    topology: Any = None,
) -> List[PartActualState]:
    """Convert a ViewAgentResult into PartActualState objects for downstream modules."""
    photo_id = result.get("photo_id", "unknown")
    primary_view = result.get("primary_view")
    part_states: List[PartActualState] = []

    for obs in result.get("parts", []):
        part_id = obs["part_id"]
        part_info = PARTS_BY_ID.get(part_id)
        if not part_info:
            continue

        status = Status(obs.get("status", "uncertain"))
        level = DamageLevel(obs.get("damage_level", "unknown"))
        damage_types = list(obs.get("damage_types", ["none"]))

        state = PartActualState(
            part_id=part_id,
            part_name=part_info["part_name"],
            part_category=part_info["part_category"],
            side=part_info.get("side", "center"),
            status=status,
            damage_level=level,
            damage_types=damage_types,
            standard_exists=True,
            actual_visible=True,
            actual_present=status != Status.MISSING,
            confidence=obs.get("confidence", "low"),
            evidence_photos=[photo_id],
            notes=obs.get("description", ""),
            photo_type=primary_view,
            evidence_sources=[{"photo_id": photo_id, "view_id": primary_view}],
        )
        part_states.append(state)

    return part_states
