"""Planner agent — 视觉驱动的照片三类分拣。

The planner no longer assigns canonical views, and it no longer classifies by
filename keyword or aspect ratio.  Its only job is to look at every uploaded
photo with the vision model and sort it into one of three buckets:

- ``exterior``     — vehicle exterior photos that go to the ViewAgent Team.
                     Gated: the model must be able to name a rough position
                     (front/rear/left/right/roof) AND make out part of the
                     vehicle outline.  A pure close-up with no discernible
                     outline does NOT qualify and is stripped from the stream.
- ``interior``     — cabin photos (airbag / dashboard / seat / headliner).
                     Stripped from the exterior evidence chain so their damage
                     is not mis-attributed to exterior structural parts
                     (172852: an airbag/dashboard shot fed the pillar_a_left
                     false positive).
- ``vehicle_info`` — license, VIN plate, nameplate, policy, invoice, etc.

There is NO deterministic fallback.  If the vision call fails or a photo
cannot be confidently classified, the photo is marked ``exclude`` and dropped
from the downstream damage-assessment stream rather than guessed into the
exterior pool.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from agents.minimax_client import build_image_content, call_minimax, extract_json
from agents.rules import render_prompt_template
from agents.view_mapping import PHOTO_TYPE_CATEGORIES
from config import IMAGE_MAX_WIDTH

logger = logging.getLogger(__name__)
_planner_file_handler = logging.FileHandler(
    os.path.expanduser("~/vehicle_damage_assessment_planner.log"), mode="a", encoding="utf-8"
)
_planner_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_planner_file_handler.setLevel(logging.INFO)
logger.addHandler(_planner_file_handler)
logger.setLevel(logging.INFO)


#: Vision-emitted categories (before the exterior gate is applied).
_VISION_CATEGORIES = {"exterior", "interior", "vehicle_info"}
_ALLOWED_POSITION = {"front", "rear", "left", "right", "roof", "unclear"}
_ALLOWED_CONFIDENCE = {"high", "medium", "low"}

#: Photos per LLM call.  Small batches keep M3 from spending the whole token
#: budget on one long <think> block (finish=length truncation → no JSON).
#: Mirrors the face_profiler batch size, tuned against the same failure mode.
_BATCH_SIZE = 4

_CONFIDENCE_SCORE = {"high": 0.95, "medium": 0.7, "low": 0.4}


def _build_prior_block(vehicle_prior: Dict[str, Any]) -> str:
    """Build the vehicle-prior context block injected into the system prompt."""
    vehicle_name = vehicle_prior.get("vehicle", "该车")
    topology = vehicle_prior.get("topology")
    anchors = vehicle_prior.get("key_anchors")

    block = f"车型：{vehicle_name}\n"
    if topology:
        block += f"车型拓扑：\n{json.dumps(topology, ensure_ascii=False, indent=2)}\n"
    if anchors:
        block += f"关键锚点：\n{json.dumps(anchors, ensure_ascii=False, indent=2)}\n"
    return block


def _normalize_result(raw: Any) -> List[Dict[str, Any]] | None:
    """Normalize model output into a list of per-photo dicts.

    Accepts a bare list, a dict wrapping a ``results``/``classifications``
    list, or a single dict.  Returns None when the shape is unusable.
    """
    result = raw
    if isinstance(result, dict):
        for key in ("results", "classifications"):
            if key in result:
                result = result[key]
                break
        else:
            result = [result]
    if not isinstance(result, list):
        return None
    return [item for item in result if isinstance(item, dict)]


def _sanitize_item(item: Dict[str, Any], photo_id: str) -> Dict[str, Any]:
    """Coerce a single model item into the vision-classification contract."""
    category = item.get("category")
    if category not in _VISION_CATEGORIES:
        # Unknown/garbage category → strip from the stream downstream.
        category = "exclude"

    position = item.get("position")
    if position not in _ALLOWED_POSITION:
        position = "unclear"

    confidence = item.get("confidence")
    if confidence not in _ALLOWED_CONFIDENCE:
        confidence = "low"

    return {
        "photo_id": item.get("photo_id", photo_id),
        "category": category,
        "position": position,
        "has_vehicle_outline": bool(item.get("has_vehicle_outline", False)),
        "cabin_evidence": str(item.get("cabin_evidence", "无")),
        "confidence": confidence,
        "reason": str(item.get("reason", "")),
    }


def _realign_to_input(
    items: List[Dict[str, Any]], photos: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Align model output to input order, falling back per missing photo."""
    by_id = {item.get("photo_id"): item for item in items if item.get("photo_id")}
    aligned: List[Dict[str, Any]] = []
    for index, photo in enumerate(photos):
        photo_id = photo["id"]
        item = by_id.get(photo_id)
        if item is None and index < len(items):
            # Model may have dropped/renamed ids; fall back to positional match.
            item = items[index]
        if item is None:
            # No usable output for this photo → exclude (no deterministic guess).
            aligned.append(
                {
                    "photo_id": photo_id,
                    "category": "exclude",
                    "position": "unclear",
                    "has_vehicle_outline": False,
                    "cabin_evidence": "无",
                    "confidence": "low",
                    "reason": "模型未返回该照片的结果，剥离出数据流",
                }
            )
            continue
        sanitized = _sanitize_item(item, photo_id)
        # The photo_id is the authoritative key — always pin it to the input id,
        # never trust a model-renamed id from a positional fallback.
        sanitized["photo_id"] = photo_id
        aligned.append(sanitized)
    return aligned


def _apply_exterior_gate(vision_item: Dict[str, Any]) -> str:
    """Map a vision-classification item to a downstream PHOTO_TYPE category.

    The exterior bucket is gated: a photo only counts as exterior when the
    model can name a rough position AND make out the vehicle outline.  Anything
    short of that — an interior shot, a document, a position-less close-up, or
    an unparseable result — is stripped from the exterior evidence stream.
    """
    category = vision_item.get("category")
    if category == "interior":
        return "interior"
    if category == "vehicle_info":
        return "vehicle_info"
    if category == "exterior":
        has_outline = vision_item.get("has_vehicle_outline", False)
        position = vision_item.get("position", "unclear")
        if has_outline and position != "unclear":
            return "exterior"
        # Exterior-looking but fails the outline/position gate → strip.
        return "exclude"
    # Unknown / garbage category → strip.
    return "exclude"


async def _classify_batch(
    photos: List[Dict[str, Any]], vehicle_prior: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Classify one small batch of photos (single LLM call)."""
    system_prompt = render_prompt_template(
        "planner_vision_classify", prior_block=_build_prior_block(vehicle_prior)
    )

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": "以下是待分类的照片，请逐张判定："},
    ]
    for photo in photos:
        content.append({"type": "text", "text": f"照片编号: {photo['id']}"})
        content.append(build_image_content(photo["path"], max_width=IMAGE_MAX_WIDTH))

    messages = [{"role": "user", "content": content}]

    logger.info("[planner] vision classify start batch_size=%d", len(photos))
    raw = await call_minimax(
        messages,
        temperature=0.1,
        max_tokens=2000 * len(photos),
        response_format={"type": "json_object"},
        reasoning_effort="low",
    )

    parsed = extract_json(raw)
    if parsed is None:
        logger.warning("[planner] unparseable vision output; excluding batch")
        return _realign_to_input([], photos)

    items = _normalize_result(parsed)
    if items is None:
        logger.warning("[planner] vision output not a list; excluding batch")
        return _realign_to_input([], photos)

    return _realign_to_input(items, photos)


async def _classify_by_vision(
    photos: List[Dict[str, Any]], vehicle_prior: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Classify every photo with the vision model, batched to avoid truncation."""
    type_map: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(photos), _BATCH_SIZE):
        batch = photos[start : start + _BATCH_SIZE]
        for item in await _classify_batch(batch, vehicle_prior):
            type_map[item["photo_id"]] = item
    return type_map


async def planner_agent(
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify every photo via the vision model into a downstream category.

    Parameters
    ----------
    photos:
        List of photo dicts with at least ``id`` and ``path`` (or ``url``).
    vehicle_prior:
        Output from ``vehicle_prior_agent``; supplies topology/anchors context.

    Returns
    -------
    dict
        ``{"photo_classifications": [...]}`` where each item contains
        ``photo_id``, ``category``, ``confidence_score``, ``confidence``, and
        ``reason``.  ``category`` is one of ``PHOTO_TYPE_CATEGORIES``; photos
        that fail the exterior gate or cannot be classified are ``exclude``.
    """
    if not photos:
        return {"photo_classifications": []}

    vision_map = await _classify_by_vision(photos, vehicle_prior)

    photo_classifications: List[Dict[str, Any]] = []
    for photo in photos:
        photo_id = photo.get("id", "")
        vision_item = vision_map.get(photo_id, {})
        category = _apply_exterior_gate(vision_item)
        confidence = vision_item.get("confidence", "low")
        score = _CONFIDENCE_SCORE.get(confidence, 0.4)

        reason = vision_item.get("reason", "")
        position = vision_item.get("position", "unclear")
        cabin = vision_item.get("cabin_evidence", "无")
        gate_note = ""
        if vision_item.get("category") == "exterior" and category == "exclude":
            gate_note = "（外观门槛未过：无轮廓或位置不明，剥离）"
        full_reason = f"{reason} | position={position} cabin={cabin}{gate_note}".strip(" |")

        photo_classifications.append(
            {
                "photo_id": photo_id,
                "category": category,
                "confidence_score": score,
                "confidence": confidence,
                "reason": full_reason,
            }
        )
        logger.info("[planner] photo %s → %s (%s)", photo_id, category, confidence)

    logger.info(
        "[planner] vision classified %d photos: %s",
        len(photo_classifications),
        {c: sum(1 for p in photo_classifications if p["category"] == c) for c in PHOTO_TYPE_CATEGORIES},
    )

    return {"photo_classifications": photo_classifications}
