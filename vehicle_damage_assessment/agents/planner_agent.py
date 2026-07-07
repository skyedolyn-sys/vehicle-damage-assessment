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
    """Return a deterministic map photo_id -> category."""
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
        logger.info("[planner] photo %s classified as %s", photo_id, category)

    logger.info("[planner] deterministic classify done: %d photos", len(type_map))
    return type_map


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
