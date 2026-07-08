"""Planner agent — 4-class photo pre-screening.

The planner no longer assigns canonical views.  Its only job is to classify
every uploaded photo into one of five categories:

- ``exterior``   — vehicle exterior photos that should go to the ViewAgent Team
- ``interior``   — cabin / dashboard / seat photos
- ``vehicle_info`` — license, VIN plate, nameplate, policy, invoice, etc.
- ``exclude``    — photos that are irrelevant or unusable for assessment
- ``scene_intake`` — scene/context photos that need dedicated handling later

Classification is deterministic: filename keywords are checked first, and
image aspect ratio is used only as a tie-breaker for exterior vs close-up.
No LLM call is made by the planner in the new architecture.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

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


#: Filename keywords that strongly indicate a vehicle_info photo.
_VEHICLE_INFO_KEYWORDS = (
    "行驶证", "证件", "vin", "铭牌", "license", "plate", "车牌", "牌照",
    "车架号", "登记证书", "保单", "发票", "合格证", "一致性证书",
)

#: Filename keywords that strongly indicate an interior photo.
_INTERIOR_KEYWORDS = (
    "车内", "内饰", "驾驶舱", "座椅", "方向盘", "仪表盘", "中控", "后排",
    "安全带", "气囊", "仪表台",
)

#: Filename keywords that strongly indicate an exterior photo.
_EXTERIOR_KEYWORDS = (
    "车头", "车前", "前部", "正面", "前脸",
    "车尾", "车后", "后部", "背面", "后备箱",
    "左前", "前左", "右前", "前右",
    "左后", "后左", "右后", "后右",
    "左侧", "左侧面", "右侧", "右侧面",
    "顶部", "俯视", "车顶", "天窗",
)

#: Filename keywords that strongly indicate a scene/context photo.
_SCENE_INTAKE_KEYWORDS = (
    "现场", "全景", "环境", "事故", "碰撞点", "路牌", "路面", "第三方",
    "整体", "全貌", "场景",
)

#: Keywords that indicate the photo should be excluded.
_EXCLUDE_KEYWORDS = (
    "无关", "错误", "重复", "模糊", "黑屏", "空白", "截图",
)


def _decode_image_dimensions(photo: Dict[str, Any]) -> Tuple[int, int]:
    """Decode image width/height for deterministic signals."""
    path = photo.get("path", "") or photo.get("url", "")
    if not path or path.startswith(("http://", "https://")):
        return 0, 0
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size
    except Exception:
        return 0, 0


def _classify_by_filename(filename: str) -> str:
    """Return a deterministic category based on filename keywords."""
    if not filename:
        return ""
    lowered = filename.lower()

    if any(kw in lowered for kw in _EXCLUDE_KEYWORDS):
        return "exclude"
    if any(kw in lowered for kw in _VEHICLE_INFO_KEYWORDS):
        return "vehicle_info"
    if any(kw in lowered for kw in _INTERIOR_KEYWORDS):
        return "interior"
    if any(kw in lowered for kw in _SCENE_INTAKE_KEYWORDS):
        return "scene_intake"
    if any(kw in lowered for kw in _EXTERIOR_KEYWORDS):
        return "exterior"
    return ""


def _classify_by_signals(photo: Dict[str, Any]) -> str:
    """Deterministic classification using filename + image aspect ratio."""
    filename = photo.get("id", "") or photo.get("name", "")
    by_name = _classify_by_filename(filename)
    if by_name:
        return by_name

    width = photo.get("_decoded_width", 0)
    height = photo.get("_decoded_height", 0)
    if width and height:
        ratio = width / height
        # Extreme aspect ratios are often close-up damage shots; treat them as
        # exterior so ViewAgent can still inspect them.
        if ratio < 0.6 or ratio > 1.6:
            return "exterior"

    return "exterior"


def _category_confidence(category: str) -> Tuple[float, str]:
    """Return (score, level) for a deterministic classification."""
    if category in ("vehicle_info", "interior"):
        return 0.95, "high"
    if category == "exclude":
        return 0.90, "high"
    if category == "scene_intake":
        return 0.80, "medium"
    return 0.70, "medium"


def _category_reason(category: str) -> str:
    reasons = {
        "exterior": "车身外观照片",
        "interior": "车内/内饰照片",
        "vehicle_info": "车辆证件/铭牌/保单类照片",
        "exclude": "与损伤评估无关或无法使用的照片",
        "scene_intake": "现场/环境/全景照片，需 scene_intake 专属处理",
    }
    return reasons.get(category, "未匹配到明确关键词，默认按外观处理")


async def _classify_photo_types(
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
) -> Dict[str, str]:
    """Classify each photo into a 4-class category.

    Primary path: call the LLM with a small image-bearing prompt
    (planner_classification_prompt.j2) so the model actually looks at
    the photo.  Filenames like ``172852-01.png`` carry no semantic signal
    so a filename-only heuristic would mis-classify every photo as
    ``exterior`` (the default fallback).

    Fallback path: if the LLM call fails (network, parse, empty) we
    fall back to the deterministic filename + aspect-ratio classifier so
    the pipeline never deadlocks.
    """
    if not photos:
        return {}

    type_map: Dict[str, str] = await _classify_with_llm(photos)
    if type_map:
        return type_map

    logger.warning(
        "[planner] LLM classification returned empty/errored; "
        "falling back to filename + aspect-ratio heuristic"
    )
    return _classify_deterministic(photos)


def _classify_deterministic(photos: List[Dict[str, Any]]) -> Dict[str, str]:
    """Pure-Python fallback that needs no LLM call.

    Useful as a safety net when the LLM call errors out so the
    pipeline can still return a complete plan.  Filenames like
    ``172852-01.png`` carry no semantic signal so this will usually
    return ``exterior`` for everything — exactly the problem the LLM
    path solves.  Keep this as a fallback only.
    """
    type_map: Dict[str, str] = {}
    for photo in photos:
        if "_decoded_width" not in photo:
            w, h = _decode_image_dimensions(photo)
            photo["_decoded_width"] = w
            photo["_decoded_height"] = h

    for photo in photos:
        photo_id = photo.get("id", "")
        if not photo_id:
            continue
        category = _classify_by_signals(photo)
        type_map[photo_id] = category
        logger.info("[planner] photo %s classified as %s (fallback)", photo_id, category)

    return type_map


async def _classify_with_llm(photos: List[Dict[str, Any]]) -> Dict[str, str]:
    """Classify each photo into a 4-class category via the vision model.

    Primary path: per-photo ``understand_image`` call (MCP tool when
    configured, chat-completions image path otherwise).  Calling the
    model once per photo avoids the long <think> blocks that broke the
    earlier batched approach — the model has more thinking headroom
    for a single image and the JSON it returns is small.

    Returns
    -------
    dict[photo_id, category]
        Empty dict if every call errors or returns an unexpected
        shape; callers should fall back to the deterministic
        classifier in that case.
    """
    from agents.mcp_client import understand_image
    from agents.rules import render_prompt_template

    if not photos:
        return {}

    type_map: Dict[str, str] = {}

    for photo in photos:
        path = photo.get("path") or photo.get("url")
        photo_id = photo.get("id")
        if not path or not photo_id:
            continue

        prompt = _build_classify_prompt(photo_id)
        try:
            raw_text = await understand_image(path, prompt)
        except Exception as exc:
            logger.warning("[planner] understand_image failed for %s: %s", photo_id, exc)
            continue

        raw = _parse_classify_response(raw_text)
        category = raw.get("category") if isinstance(raw, dict) else None
        category = _normalize_category(category) if category else None
        if category:
            type_map[photo_id] = category
            logger.info("[planner] photo %s classified as %s (vision)", photo_id, category)
        else:
            logger.warning(
                "[planner] photo %s vision response unparseable: %s",
                photo_id, raw_text[:200],
            )

    if not type_map:
        logger.warning(
            "[planner] LLM classify returned empty type_map (all calls errored)"
        )

    return type_map


def _build_classify_prompt(photo_id: str) -> str:
    """Build the per-photo classification prompt.

    Kept short on purpose: ``understand_image`` is a single-turn
    tool call so the model should not have to spend tokens on
    multi-photo bookkeeping.
    """
    return (
        "判断这张车辆照片属于以下哪一类（仅输出一个 JSON 对象，不要 markdown、"
        "不要 thinking）：\n"
        "- exterior: 车身外部照片（车头/车尾/侧/45度角/车顶）\n"
        "- interior: 车内照片（座椅/方向盘/仪表台/中控/后排）\n"
        "- vehicle_info: 证件、行驶证、VIN 码、铭牌、车牌特写\n"
        "- unknown: 其它或判断不了\n\n"
        "输出格式："
        '{"photo_id": "<photo_id>", "category": "<类别>", "reason": "<≤10 字理由>"}\n\n'
        f"photo_id = {photo_id}"
    )


def _parse_classify_response(raw_text: str) -> Dict[str, Any]:
    """Parse the vision response into a dict, tolerating extra prose.

    The model occasionally wraps its answer with a sentence of
    explanation.  We look for the first JSON object in the text and
    return it; an empty dict is returned if no JSON is found.
    """
    import json
    import re

    if not raw_text:
        return {}

    # Try a direct parse first.
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Look for the first balanced JSON object in the text.
    match = re.search(r"\{[^{}]*\}", raw_text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


async def _classify_with_chat(photos: List[Dict[str, Any]]) -> Dict[str, str]:
    """Batched chat-completions path kept for the test suite and as
    an explicit fallback.  The production path is now
    :func:`_classify_with_llm` which uses one ``understand_image`` call
    per photo.
    """


async def _classify_one_batch(
    batch: List[Dict[str, Any]],
    batch_idx: int,
    total_batches: int,
) -> Dict[str, str]:
    """Classify a single batch of photos (≤ BATCH_SIZE)."""
    from agents.minimax_client import build_image_content, call_minimax, extract_json
    from agents.rules import render_prompt_template
    from config import IMAGE_MAX_WIDTH, MINIMAX_MODEL

    system_prompt = render_prompt_template("planner_classification_prompt")

    user_blocks: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"批次 {batch_idx + 1}/{total_batches}，请按顺序逐张判断下述照片的类别。"
                "只输出一个 JSON 对象，no thinking, no markdown。"
            ),
        }
    ]
    for photo in batch:
        path = photo.get("path") or photo.get("url")
        if not path:
            continue
        user_blocks.append(build_image_content(path, max_width=IMAGE_MAX_WIDTH))
        user_blocks.append(
            {"type": "text", "text": f"  ↑ photo_id: {photo.get('id', 'unknown')}"}
        )

    messages = [
        {
            "role": "user",
            "content": user_blocks,
        }
    ]

    try:
        raw_text = await call_minimax(
            messages=messages,
            temperature=0.0,
            # 4 photos per batch: thinking ~3K chars + JSON ~200 chars.
            # 6000 leaves enough headroom.
            max_tokens=6000,
            model=MINIMAX_MODEL,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.error("[planner] LLM classify batch %d failed: %s", batch_idx, exc)
        return {}

    raw = extract_json(raw_text)
    if not isinstance(raw, dict):
        logger.warning(
            "[planner] batch %d returned non-dict: type=%s raw_len=%s",
            batch_idx, type(raw).__name__,
            len(raw_text) if raw_text else 0,
        )
        return {}

    return _parse_llm_classifications(raw)


def _parse_llm_classifications(raw: Dict[str, Any]) -> Dict[str, str]:
    """Tolerate two common LLM payload shapes:

    1) ``{"classifications": [{"photo_id": ..., "photo_type": ...}, ...]}``
    2) ``{"<photo_id>": "exterior", ...}``  (flat map)
    """
    type_map: Dict[str, str] = {}
    items = raw.get("classifications")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            pid = item.get("photo_id")
            cat = item.get("photo_type") or item.get("category")
            if pid and cat:
                normalized = _normalize_category(cat)
                if normalized:
                    type_map[pid] = normalized
    else:
        for pid, cat in raw.items():
            if isinstance(cat, str) and cat in PHOTO_TYPE_CATEGORIES:
                type_map[pid] = cat

    if not type_map:
        logger.warning(
            "[planner] LLM classify raw had no usable entries: keys=%s",
            list(raw.keys())[:5],
        )
    return type_map


def _normalize_category(raw_cat: str) -> str:
    """Map the LLM's free-form category string onto PHOTO_TYPE_CATEGORIES.

    The prompt asks for ``exterior / interior / auxiliary / unknown``
    but our canonical enum uses ``vehicle_info`` instead of ``auxiliary``
    and ``scene_intake`` for some scene photos.  This normalizer is
    conservative: it only forwards strings that fit the canonical enum.
    """
    if not raw_cat:
        return ""
    cat = raw_cat.lower().strip()
    aliases = {
        "auxiliary": "vehicle_info",
        "vin": "vehicle_info",
        "document": "vehicle_info",
        "license": "vehicle_info",
        "plate": "vehicle_info",
        "scene": "scene_intake",
        "scene_intake": "scene_intake",
        "interior": "interior",
        "exterior": "exterior",
        "exclude": "exclude",
        "unknown": "unknown",
        "vehicle_info": "vehicle_info",
    }
    return aliases.get(cat, "")


async def planner_agent(
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify every photo into one of the five downstream categories.

    Parameters
    ----------
    photos:
        List of photo dicts with at least ``id`` and ``path`` (or ``url``).
    vehicle_prior:
        Output from ``vehicle_prior_agent``; currently unused by the planner
        but kept for interface compatibility.

    Returns
    -------
    dict
        ``{"photo_classifications": [...]}`` where each item contains
        ``photo_id``, ``category``, ``confidence_score``, ``confidence``,
        and ``reason``.
    """
    if not photos:
        return {"photo_classifications": []}

    categories = await _classify_photo_types(photos, vehicle_prior)

    photo_classifications: List[Dict[str, Any]] = []
    for photo in photos:
        photo_id = photo.get("id", "")
        category = categories.get(photo_id, "exterior")
        score, level = _category_confidence(category)
        photo_classifications.append({
            "photo_id": photo_id,
            "category": category,
            "confidence_score": score,
            "confidence": level,
            "reason": _category_reason(category),
        })

    logger.info(
        "[planner] classified %d photos: %s",
        len(photo_classifications),
        {c: sum(1 for p in photo_classifications if p["category"] == c) for c in PHOTO_TYPE_CATEGORIES},
    )

    return {
        "photo_classifications": photo_classifications,
    }
