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
from agents.view_mapping import get_parts_for_view
from config import IMAGE_MAX_WIDTH, MINIMAX_MODEL, PARTS_BY_ID
from models import PartActualState, Status, DamageLevel

logger = logging.getLogger(__name__)

# Centralized file logging — see agents/_log_init.py.  Two log streams:
#  - view.log       — full diagnostic log (rotates by append, dedup-safe)
#  - view_trace.log — one-line-per-photo trace for human auditability;
#    mode="a" preserves history across Django runserver reloads (the old
#    mode="w" silently truncated the file on every import).
from agents._log_init import attach_file_handler
attach_file_handler(logger, "view.log")
attach_file_handler(logger, "view_trace.log", level=logging.INFO)
logger.setLevel(logging.INFO)


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
    face_prior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a single-photo view + damage assessment.

    Parameters
    ----------
    photo:
        ``{"id": str, "path": str, "category": "exterior"}``
    vehicle_prior:
        Output from ``vehicle_prior_agent``.
    face_prior:
        Optional output of ``face_mapping.build_face_prior``.  When provided
        the agent does NOT re-derive the camera facing or flip left/right —
        the facing and camera_side are already locked by the deterministic
        face path, and damage is only assessed within ``candidate_parts``.
        When ``None`` the legacy behaviour (self-derived view + Step B flip)
        is preserved unchanged.

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
    if face_prior:
        # Face path: facing + camera_side already locked deterministically.
        # The agent only assesses damage inside the candidate part set; it does
        # not judge facing, flip left/right, or stray outside the part list.
        candidate_parts = face_prior.get("candidate_parts") or {}
        camera_side = face_prior.get("camera_side")
        side_cn = {"left": "左", "right": "右"}.get(camera_side, "未确定")
        candidate_parts_named = [
            (pid, PARTS_BY_ID[pid]["part_name"])
            for pid in candidate_parts
            if pid in PARTS_BY_ID
        ]
        system_prompt = (
            "你是车辆外观损伤识别专家。相机朝向与左右已由前置分析确定并锁定，"
            "你只负责在指定部件范围内判断损伤状态，严禁重新判断朝向或左右。"
            "输出必须是合法 JSON，part_id 必须逐字使用任务中给出的规范 id。"
        )
        task_prompt = render_prompt_template(
            "view_agent_face_task",
            photo_id=photo_id,
            vehicle_name=vehicle_name,
            facing=face_prior.get("facing") or "unclear",
            camera_side=side_cn if camera_side else "",
            anchor=face_prior.get("anchor") or "无",
            candidate_part_ids=list(candidate_parts.keys()),
            candidate_parts=candidate_parts,
            candidate_parts_named=candidate_parts_named,
        )
    else:
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
        # 8000 gives the model room for both its reasoning and the JSON body.
        # At 4000 (esp. with 1024px images) M3 often burns the whole budget in
        # a <think> block and emits no JSON (finish_reason=length), silently
        # dropping the photo.  call_minimax escalates further on truncation.
        max_tokens=8000,
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

    candidate_parts = face_prior.get("candidate_parts") or {} if face_prior else {}
    candidate_set = set(candidate_parts.keys()) if candidate_parts else None
    result = _normalize_view_agent_result(photo_id, raw, candidate_set)
    if face_prior:
        # Face path: primary_view comes from the deterministic facing, and
        # missing parts are backfilled from the candidate set (not the legacy
        # view checklist), so damage can only be assigned to faces this photo
        # can reliably map.  Out-of-scope parts are never invented.
        result["primary_view"] = _facing_to_view_id(face_prior.get("facing"), face_prior.get("camera_side"))
        result = _backfill_from_candidates(result, candidate_parts)
        if not face_prior.get("usable", True):
            # Soft downgrade: this photo's facing was unclear/low-confidence,
            # so its damage observations are kept as evidence but capped at
            # low confidence — they cannot alone convict a part downstream,
            # only corroborate a high-confidence observation of the same part.
            result = _soft_downgrade_damage(result)
    else:
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


def _normalize_view_agent_result(
    photo_id: str,
    raw: Dict[str, Any],
    candidate_parts: Optional[set] = None,
) -> Dict[str, Any]:
    """Validate and normalize raw LLM output into the canonical ViewAgentResult shape.

    Part names the LLM emitted that could not be mapped onto the canonical
    catalog — whether unknown to the catalog or filtered out as out-of-scope —
    are collected into ``unmapped_parts`` rather than silently dropped, so the
    whole data flow keeps a trace of what the model tried to report.  Each entry
    records the raw name, the alias-resolved id, and why it was not kept.
    """
    view_detections = _normalize_view_detections(raw.get("view_detections"))
    primary_view = _resolve_primary_view(view_detections, raw.get("primary_view"))

    normalized_parts: List[Dict[str, Any]] = []
    unmapped_parts: List[Dict[str, Any]] = []
    for part in raw.get("parts", []) or []:
        if not isinstance(part, dict):
            continue
        raw_name = str(part.get("part_id") or part.get("part_name") or "")
        normalized = _normalize_part_observation(part, photo_id, candidate_parts)
        if normalized:
            normalized_parts.append(normalized)
        elif raw_name:
            resolved = _canonical_part_id_alias(raw_name)
            reason = (
                "unknown_part_id" if resolved not in PARTS_BY_ID
                else "out_of_candidate"
            )
            unmapped_parts.append({
                "raw_name": raw_name,
                "resolved_part_id": resolved,
                "status": part.get("status"),
                "damage_level": part.get("damage_level"),
                "description": part.get("description"),
                "drop_reason": reason,
            })

    return {
        "photo_id": photo_id,
        "primary_view": primary_view,
        "view_detections": view_detections,
        "parts": normalized_parts,
        "unmapped_parts": unmapped_parts,
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


def _normalize_part_observation(
    part: Dict[str, Any], photo_id: str,
    candidate_parts: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a single part observation.

    When ``candidate_parts`` is supplied (the face path), observations outside
    the candidate set are dropped — the prompt says "do not assess parts
    outside the candidate list", but models violate that soft constraint, so
    we enforce it here as a hard filter.  Out-of-scope parts are silently
    dropped (logged) to keep the per-photo part list on the correct side.
    """
    raw_part_id = part.get("part_id") or part.get("part_name")
    if not raw_part_id:
        return None

    part_id = _canonical_part_id_alias(str(raw_part_id))
    if part_id not in PARTS_BY_ID:
        logger.warning("[view_agent] unknown part_id=%s from photo_id=%s", part_id, photo_id)
        return None

    if candidate_parts is not None and part_id not in candidate_parts:
        logger.info(
            "[view_agent] out-of-candidate part_id=%s from photo_id=%s (dropped)",
            part_id, photo_id,
        )
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


def _facing_to_view_id(facing: Optional[str], camera_side: Optional[str]) -> Optional[str]:
    """Map the deterministic facing + camera_side back to a 9-view id.

    Downstream aggregation (master_agent._aggregate_part_evidence) still keys
    primary views on the 9-view vocabulary, so the face path translates its
    facing/side into the closest canonical view.  This is a pure lookup — no
    model inference — so it cannot reintroduce a flip.
    """
    if facing == "front":
        if camera_side == "left":
            return "front_left"
        if camera_side == "right":
            return "front_right"
        return "front"
    if facing == "rear":
        if camera_side == "left":
            return "rear_left"
        if camera_side == "right":
            return "rear_right"
        return "rear"
    if facing == "side":
        if camera_side == "left":
            return "left"
        if camera_side == "right":
            return "right"
        return None
    if facing == "top":
        return "top"
    return None


def _backfill_from_candidates(
    result: Dict[str, Any], candidate_parts: Dict[str, str]
) -> Dict[str, Any]:
    """Fill any candidate parts the LLM omitted, scoped to the face path.

    Unlike ``_backfill_missing_parts`` (which uses the legacy per-view
    checklist), this only ever adds parts already in the deterministic
    candidate set — so backfill can never introduce a part from the wrong
    side/face.

    Status semantics:
    - ``"absent"`` is a sentinel meaning "this part is in the camera's
      candidate set but the LLM emitted no observation for it".  It carries
      ``_backfill=True`` so downstream code (aggregator, synthesizer) can
      recognize it and *exclude it from consensus voting*.  Without this,
      a photo whose clear observation is "sunroof is crushed" would also
      backfill ``fender_front_left``/``mirror_right``/``bumper_front`` and
      every one of those parts would never reach consensus because the
      backfilled uncertain observation cancels out the real damaged ones
      from other photos.
    - The legacy per-view ``_backfill_missing_parts`` uses the same
      ``_backfill=True`` tag, so downstream code can treat both uniformly.
    """
    seen = {p["part_id"] for p in result.get("parts", [])}
    for part_id in candidate_parts:
        if part_id in seen:
            continue
        part_info = PARTS_BY_ID.get(part_id)
        if not part_info:
            continue
        result["parts"].append({
            "part_id": part_id,
            "part_name": part_info["part_name"],
            "status": "absent",          # sentinel, NOT "uncertain"
            "damage_level": "unknown",
            "damage_types": ["none"],
            "model_confidence_score": 0.0,
            "confidence": "low",
            "_backfill": True,           # downstream skip tag
            "description": "候选清单里的部件，LLM 未给出观察；标记为 absent（不参与投票）",
            "evidence_photo": result.get("photo_id"),
        })
    return result


def _soft_downgrade_damage(result: Dict[str, Any]) -> Dict[str, Any]:
    """Cap damage confidence at low for an unclear/low-confidence photo.

    The observation is preserved (status/damage_level/description unchanged)
    so it can still corroborate the same part seen clearly from another photo,
    but its standalone authority is removed by forcing ``confidence=low`` and
    tagging it.  Downstream consensus rules already require high confidence
    (or multiple corroborating sources) before a single damaged observation
    can convict a part, so a low-confidence tag is enough to stop a shaky
    close-up from producing a false positive on its own.
    """
    for part in result.get("parts", []):
        if part.get("status") in ("damaged", "missing"):
            part["confidence"] = "low"
            part["model_confidence_score"] = min(
                float(part.get("model_confidence_score", 0.5)), 0.4
            )
            part["_soft_downgraded"] = True
    return result


def _calibrate_result_confidence(result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply in-agent confidence calibration to every part observation.

    Parts flagged ``_soft_downgraded`` (from an unclear/low-confidence photo)
    keep their forced low confidence — calibration must not re-inflate them.
    """
    for part in result.get("parts", []):
        if part.get("_soft_downgraded"):
            part["confidence"] = "low"
            continue
        part["confidence"] = _score_to_level(_calibrate_damage_confidence(part))
    return result


#: Calibration weights used by ``_calibrate_damage_confidence`` to combine
#: model score, description keywords, damage-type count, and status into a
#: single confidence score in ``[0.0, 1.0]``.  Threshold band edges are in
#: ``_HIGH_CONF_THRESHOLD`` / ``_MEDIUM_CONF_THRESHOLD``.
#: Calibration was hand-tuned against the 172852 sample; if you A/B against a
#: new sample, start by changing these and only fall back to feature
#: changes if the calibration is structurally wrong.
_DMG_DESC_STRONG = 0.9     # description has 严重/明显/大面积/变形/凹陷/断裂
_DMG_DESC_LIGHT = 0.6      # description has 轻微/小/局部/细微
_DMG_DESC_NEUTRAL = 0.4    # description has none of the above
_DMG_TYPES_DOMINANT = 0.85 # ≥2 distinct damage types
_DMG_TYPES_SINGLE = 0.7    # exactly 1 damage type
_DMG_TYPES_NONE = 0.5      # only "none" / no damage types
_DMG_STATUS_INTACT_OK = 0.9  # status=intact AND description contains 无
_DMG_STATUS_DAMAGED_OK = 0.8 # status=damaged AND has damage types
_DMG_STATUS_NEUTRAL = 0.4
_DMG_BASE_MODEL_SCORE = 0.5  # neutral default when model didn't self-score

#: Confidence-band edges used by ``_score_to_level`` and downstream consumers
#: to bucket the calibrated score into high / medium / low buckets.
_HIGH_CONF_THRESHOLD = 0.75
_MEDIUM_CONF_THRESHOLD = 0.5


def _calibrate_damage_confidence(raw: Dict[str, Any]) -> float:
    """Schema §4.2: combine model score, description, damage types, and status."""
    signals = [float(raw.get("model_confidence_score", _DMG_BASE_MODEL_SCORE))]

    desc = raw.get("description", "")
    if any(kw in desc for kw in ["明显", "严重", "大面积", "变形", "凹陷", "断裂"]):
        signals.append(_DMG_DESC_STRONG)
    elif any(kw in desc for kw in ["轻微", "小", "局部", "细微"]):
        signals.append(_DMG_DESC_LIGHT)
    else:
        signals.append(_DMG_DESC_NEUTRAL)

    types = [t for t in raw.get("damage_types", []) if t != "none"]
    if len(types) >= 2:
        signals.append(_DMG_TYPES_DOMINANT)
    elif len(types) == 1:
        signals.append(_DMG_TYPES_SINGLE)
    else:
        signals.append(_DMG_TYPES_NONE)

    status = raw.get("status")
    if status == "intact" and "无" in desc:
        signals.append(_DMG_STATUS_INTACT_OK)
    elif status == "damaged" and types:
        signals.append(_DMG_STATUS_DAMAGED_OK)
    else:
        signals.append(_DMG_STATUS_NEUTRAL)

    return sum(signals) / len(signals)


def _score_to_level(score: float) -> str:
    if score >= _HIGH_CONF_THRESHOLD:
        return "high"
    if score >= _MEDIUM_CONF_THRESHOLD:
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
