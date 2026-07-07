"""Planner agent — assigns a canonical view label to every uploaded photo.

The planner looks at the whole set of photos at once (via low-resolution
thumbnails) and produces a structured plan:

- ``photo_views``: mapping from photo_id to canonical view_id
- ``view_groups``: photos grouped by view
- ``coverage_gaps``: missing views and impacted regions/parts
- ``workflow_plan``: high-level strategy for the vision subagents

Because the planner sees all photos in a single context, it can make more
stable left/right judgements than the old per-batch photo_locator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import asyncio
import logging
import os
import re
import time

from agents.minimax_client import call_minimax, extract_json, build_image_content
from agents.rules import render_prompt_template

logger = logging.getLogger(__name__)
# Also mirror planner logs to a dedicated file so Django's console log level
# does not swallow them.
_planner_file_handler = logging.FileHandler(
    os.path.expanduser("~/vehicle_damage_assessment_planner.log"), mode="a", encoding="utf-8"
)
_planner_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_planner_file_handler.setLevel(logging.INFO)
logger.addHandler(_planner_file_handler)
logger.setLevel(logging.INFO)
from agents.view_mapping import (
    EXTERIOR_VIEWS,
    NON_EXTERIOR_VIEWS,
    PHOTO_TYPE_CATEGORIES,
    SCENE_INTAKE_VIEW,
    STANDARD_VIEWS,
    get_all_exterior_views,
    get_display_name,
    get_regions_for_view,
    get_view_selection_prompt,
    is_exterior_view,
    normalize_view_id,
)
from config import IMAGE_MAX_WIDTH, PARTS_BY_ID, MAX_CONCURRENT_API_CALLS


#: Thumbnail width used by the planner.  Smaller images reduce cost/latency
#: while preserving enough detail for view classification.
_PLANNER_THUMB_WIDTH = 384


#: 每批发送给 LLM 的最大照片数。
#: 经验上，8 张一批时 batch 独立判断容易系统性左右翻转（172852 回归）。
#: 16 张一批让单批包含更多全局上下文，左右稳定性显著提升，同时 32 张照片
#: 仍只需 2 批，max_tokens=16000 给 thinking + 16 个 entry JSON 留出空间。
_PLANNER_BATCH_SIZE = 16


#: 单个 batch 调 LLM 的硬超时（秒）。超时的 batch 把所有照片标记为
#: unknown，让下游 safety net 兜底；防止 Server disconnected 雪崩
#: 把单批拖到 200s+ 进而拖垮整次评估。MiniMax 在 8 张照片 + thinking 上
#: 单批要 60-90s，调到 180s 让多数 batch 能完成。
_PLANNER_BATCH_TIMEOUT_SEC = 180.0


#: Per-batch parallel limit for the planner LLM.
#: Capped low because MiniMax throttles global concurrency — running 4+ heavy
#: planner prompts in parallel against the same endpoint reliably triggers
#: "Server disconnected" retries. 1 (sequential) keeps us safe; the 4-batch
#: fan-out still gives ~3x wall-clock win over a single 32-photo prompt
#: (which fails with max_tokens=4000) because each batch produces a much
#: shorter JSON envelope.
_PLANNER_BATCH_CONCURRENCY = 1


def _impacted_parts_for_missing_view(view_id: str) -> List[str]:
    """Return human-readable part names likely impacted by a missing exterior view."""
    regions = get_regions_for_view(view_id)
    return [info["part_name"] for pid, info in PARTS_BY_ID.items() if info.get("part_category") in regions]


#: Robust filename keyword patterns that strongly indicate auxiliary or interior
#: photos. These are matched case-insensitively against the filename stem.
_AUXILIARY_KEYWORDS = (
    "行驶证", "证件", "vin", "铭牌", "license", "plate", "证", "牌",
    "车架号", "登记证书", "保单", "发票",
)
_INTERIOR_KEYWORDS = ("车内", "内饰", "驾驶舱", "座椅", "方向盘", "仪表盘", "中控", "后排")


#: Confidence score ordering for stable tie-breaking.
_CONFIDENCE_ORDER = {"high": 2, "medium": 1, "low": 0}


#: Filename patterns that can be used as a deterministic view fallback when the
#: LLM planner returns no usable labels. Mapping is ``stem_substring -> view_id``.
#: Order matters: more specific patterns should come first.
_FILENAME_VIEW_HINTS: List[Tuple[str, str]] = [
    ("行驶证", "auxiliary"),
    ("vin", "auxiliary"),
    ("铭牌", "auxiliary"),
    ("证件", "auxiliary"),
    ("车牌", "auxiliary"),
    ("牌照", "auxiliary"),
    ("车架号", "auxiliary"),
    ("登记证书", "auxiliary"),
    ("保单", "auxiliary"),
    ("发票", "auxiliary"),
    ("内饰", "interior"),
    ("车内", "interior"),
    ("座椅", "interior"),
    ("驾驶舱", "interior"),
    ("方向盘", "interior"),
    ("仪表盘", "interior"),
    ("中控", "interior"),
    ("后排", "interior"),
    # DAMAGE_RECOGNITION_POLICY §1.6 / 步骤 2: 把前/后/左/右/顶的关键词扩展进
    # filename hint,让 planner 在调 LLM 之前就能确定大多数视角。
    # 注意:45 度角的"前左/前右"这种偏正交的中文表达比较少见,保留为 LLM 任务。
    ("车头", "front"),
    ("车前", "front"),
    ("前部", "front"),
    ("正面", "front"),
    ("车尾", "rear"),
    ("车后", "rear"),
    ("后部", "rear"),
    ("背面", "rear"),
    ("左前", "front_left_45"),
    ("前左", "front_left_45"),
    ("右前", "front_right_45"),
    ("前右", "front_right_45"),
    ("左后", "rear_left_45"),
    ("后左", "rear_left_45"),
    ("右后", "rear_right_45"),
    ("后右", "rear_right_45"),
    ("左侧", "left_90"),
    ("左侧面", "left_90"),
    ("右侧", "right_90"),
    ("右侧面", "right_90"),
    ("顶部", "top"),
    ("俯视", "top"),
    ("车顶", "top"),
]


def _view_hint_from_filename(filename: str) -> str:
    """Return a canonical view hint based on filename conventions.

    This is a deterministic fallback used when the LLM planner produces no
    usable labels. It is intentionally conservative: only map obviously
    auxiliary/interior photos or well-known naming patterns.
    """
    if not filename:
        return ""
    lowered = filename.lower()
    for pattern, view_id in _FILENAME_VIEW_HINTS:
        if pattern.lower() in lowered:
            return view_id
    return ""

_VALID_PHOTO_TYPES = {"wide_shot", "close_up_damage", "close_up_detail", "unknown"}


def _extract_side_anchors(vehicle_prior: Dict[str, Any]) -> Dict[str, str]:
    """Return human-readable left/right anchors from vehicle prior.

    Uses vehicle specs and topology text to help the planner distinguish left
    from right.  Examples:
      - "左后翼子板有慢充口" / "右后翼子板有快充口"
      - "中国大陆左舵车，方向盘/主驾在左侧"
    """
    anchors: Dict[str, List[str]] = {"left": [], "right": [], "general": []}

    specs = vehicle_prior.get("vehicle_specs", {})
    topology = vehicle_prior.get("topology", {})

    # Charging ports are strong side discriminators.
    if isinstance(specs, dict):
        notes = specs.get("notes", "")
        if isinstance(notes, str):
            if "慢充" in notes or "交流充电" in notes:
                anchors["left"].append("左后翼子板有慢充口/交流充电口")
            if "快充" in notes or "直流充电" in notes:
                anchors["right"].append("右后翼子板有快充口/直流充电口")

    # Fall back to topology text.
    if isinstance(topology, dict):
        left_text = topology.get("left", "")
        right_text = topology.get("right", "")
        if isinstance(left_text, str):
            if "慢充" in left_text:
                anchors["left"].append("左后翼子板有慢充口")
            if "充电" in left_text and "左" in left_text:
                anchors["left"].append("左侧有充电口")
        if isinstance(right_text, str):
            if "快充" in right_text:
                anchors["right"].append("右后翼子板有快充口")
            if "充电" in right_text and "右" in right_text:
                anchors["right"].append("右侧有充电口")

    # Left-hand-drive default for mainland-China fleet.
    anchors["general"].append("中国大陆左舵车：方向盘/主驾驶位在车辆左侧")

    return {
        "left": "；".join(anchors["left"]) if anchors["left"] else "",
        "right": "；".join(anchors["right"]) if anchors["right"] else "",
        "general": "；".join(anchors["general"]),
    }


def _build_system_prompt(vehicle_name: str, vehicle_prior: Dict[str, Any] | None = None) -> str:
    """Render the planner system prompt from the rules package template."""
    side_anchors = _extract_side_anchors(vehicle_prior or {})
    anchor_lines: List[str] = []
    if side_anchors["left"]:
        anchor_lines.append(f"- 车辆左侧特征：{side_anchors['left']}")
    if side_anchors["right"]:
        anchor_lines.append(f"- 车辆右侧特征：{side_anchors['right']}")
    if side_anchors["general"]:
        anchor_lines.append(f"- {side_anchors['general']}")
    anchor_text = "\n".join(anchor_lines) if anchor_lines else ""

    return render_prompt_template(
        "planner_system_prompt",
        view_selection_prompt=get_view_selection_prompt(),
        vehicle_name=vehicle_name,
        side_anchors=anchor_text,
    )


def _damage_score_from_reason(reason: str) -> int:
    """Heuristic damage signal strength from a planner reason string."""
    if not reason:
        return 0
    reason = reason.lower()
    severe_keywords = ("严重", "塌陷", "撕裂", "断裂", "碎裂", "变形", "破损", "损毁", "压溃", "褶皱", "弯折")
    light_keywords = ("划痕", "凹陷", "刮擦", "掉漆", "裂纹", "损伤")
    score = sum(3 for kw in severe_keywords if kw in reason)
    score += sum(1 for kw in light_keywords if kw in reason)
    return score


def _mirror_view_id(view_id: str) -> str:
    """Swap left/right in a canonical view id."""
    swaps = {
        "front_left_45": "front_right_45",
        "front_right_45": "front_left_45",
        "rear_left_45": "rear_right_45",
        "rear_right_45": "rear_left_45",
        "left_90": "right_90",
        "right_90": "left_90",
    }
    return swaps.get(view_id, view_id)


def _move_left_to_right_view_id(view_id: str) -> str:
    """Move a left-view photo to the corresponding right view.

    Used when the whole dataset is known to show only one physical side (the
    right side) but the planner has systematically labelled that side as left.
    This is a *label correction only*: the photo is reassigned to the
    matching right-view slot, but the left views are not cleared — photos
    that genuinely belong on the left (or that the LLM correctly labelled)
    stay where they are.  This keeps the left-side vision subagents
    receiving evidence and prevents the must_remain_intact LEFT-side parts
    from losing their intact signal.
    """
    moves = {
        "front_left_45": "front_right_45",
        "rear_left_45": "rear_right_45",
        "left_90": "right_90",
    }
    return moves.get(view_id, view_id)


def _detect_and_fix_left_right_inversion_on_entries(
    photo_views: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detect systematic left/right mislabelling and reassign entries if confident.

    This operates on the raw ``photo_views`` list before stabilization, because
    stabilization spreads photos across canonical views and can dilute the
    count/score signals that indicate an inversion.

    Two signals are used:

    1. Damage descriptions pile up on one side (score ratio) or the photo count
       ratio is lopsided.
    2. The LLM's own reason text mentions the *opposite* physical side for a
       view (e.g. a photo labelled ``front_left_45`` but the reason says the
       right side is visible).  This is a direct contradiction and is a strong
       indicator of systematic left/right flip.

    Returns the (possibly corrected) ``photo_views`` list.  This is a label
    correction only: photos whose view_id is reassigned (e.g. ``front_left_45``
    → ``front_right_45``) move to the correct side, but the left views are
    not cleared — photos the LLM correctly labelled stay in their left-view
    buckets, so left-side vision subagents keep receiving evidence.
    """
    left_views = {"front_left_45", "rear_left_45", "left_90"}
    right_views = {"front_right_45", "rear_right_45", "right_90"}

    left_score = 0
    right_score = 0
    for entry in photo_views:
        view_id = entry.get("view_id", "")
        reason = entry.get("reason", "")
        score = _damage_score_from_reason(reason)
        if view_id in left_views:
            left_score += score
        elif view_id in right_views:
            right_score += score

    dominant_side = "left" if left_score > right_score else "right"
    dominant_score = max(left_score, right_score)
    opposite_score = min(left_score, right_score)
    score_ratio = dominant_score / opposite_score if opposite_score > 0 else float("inf")

    left_count = sum(1 for e in photo_views if e.get("view_id") in left_views)
    right_count = sum(1 for e in photo_views if e.get("view_id") in right_views)
    dominant_count = max(left_count, right_count)
    opposite_count = min(left_count, right_count)
    count_ratio = dominant_count / opposite_count if opposite_count > 0 else float("inf")

    # Signal 2: LLM reasons contradict their own view labels.
    # If a left-view reason explicitly mentions the right side, the label is
    # likely inverted.  We count mentions of "right/右侧/右前/右后" in left-view
    # reasons and "left/左侧/左前/左后" in right-view reasons.
    def _mentions_side(reason: str, side: str) -> bool:
        if not reason:
            return False
        lowered = reason.lower()
        if side == "right":
            return any(kw in lowered for kw in ("右侧", "右侧面", "右前", "前右", "右后", "后右"))
        return any(kw in lowered for kw in ("左侧", "左侧面", "左前", "前左", "左后", "后左"))

    left_entries = [e for e in photo_views if e.get("view_id") in left_views]
    right_entries = [e for e in photo_views if e.get("view_id") in right_views]
    left_mentions_right = sum(1 for e in left_entries if _mentions_side(e.get("reason", ""), "right"))
    right_mentions_left = sum(1 for e in right_entries if _mentions_side(e.get("reason", ""), "left"))
    left_mismatch_ratio = left_mentions_right / len(left_entries) if left_entries else 0.0
    right_mismatch_ratio = right_mentions_left / len(right_entries) if right_entries else 0.0

    # Decide which side is mislabelled based on the dominant physical side
    # mentioned in the reasons.  If left-view reasons keep mentioning the right
    # side, the physical impact side is right.
    if left_mismatch_ratio > right_mismatch_ratio:
        physical_side = "right"
        mismatch_ratio = left_mismatch_ratio
    else:
        physical_side = "left"
        mismatch_ratio = right_mismatch_ratio

    SCORE_RATIO_THRESHOLD = 2.0
    COUNT_RATIO_THRESHOLD = 2.0
    MISMATCH_RATIO_THRESHOLD = 0.4
    has_damage_signal = score_ratio >= SCORE_RATIO_THRESHOLD or count_ratio >= COUNT_RATIO_THRESHOLD
    has_mismatch_signal = mismatch_ratio >= MISMATCH_RATIO_THRESHOLD and max(left_mismatch_ratio, right_mismatch_ratio) > 0

    if not has_damage_signal and not has_mismatch_signal:
        return photo_views

    anchors = _extract_side_anchors(vehicle_prior)
    has_anchor = bool(anchors["left"]) or bool(anchors["right"])
    if not has_anchor:
        return photo_views

    expected_impact_side = "right" if anchors["right"] else "left"
    # Trigger only if the detected physical side matches the expected impact
    # side (charging port side).  If the mismatch is in the other direction we
    # would be swapping away from the known impact side.
    if physical_side != expected_impact_side:
        return photo_views

    # If the dominant damage side already matches the expected impact side and
    # there is no strong mismatch signal, the labels are likely correct.
    if dominant_side == expected_impact_side and not has_mismatch_signal:
        return photo_views

    logger.warning(
        "[planner] detected likely left/right inversion on raw entries: %s_score=%d vs %s_score=%d "
        "(score_ratio=%.1f, count_ratio=%.1f, mismatch_ratio=%.2f); moving %s-view photos to %s views",
        dominant_side, dominant_score,
        "right" if dominant_side == "left" else "left", opposite_score,
        score_ratio, count_ratio, mismatch_ratio,
        "left" if physical_side == "right" else "right",
        physical_side,
    )

    new_photo_views = []
    for entry in photo_views:
        new_entry = dict(entry)
        old_view = entry.get("view_id", "")
        if physical_side == "right":
            new_entry["view_id"] = _move_left_to_right_view_id(old_view)
        else:
            # Move right-view photos to left views (mirror of the above).
            new_entry["view_id"] = _mirror_view_id(_move_left_to_right_view_id(_mirror_view_id(old_view)))
        if old_view != new_entry["view_id"]:
            new_entry["reason"] = entry.get("reason", "") + f" [auto-corrected: {old_view}→{new_entry['view_id']}]"
        new_photo_views.append(new_entry)

    return new_photo_views


def _detect_and_fix_left_right_inversion(
    plan: Dict[str, Any],
    vehicle_prior: Dict[str, Any],
) -> Dict[str, Any]:
    """Plan-level wrapper around the entry-level inversion fix.

    Runs the same detection on the plan's photo_views and rebuilds the plan
    with corrected view_ids.  This is a label correction only: photos whose
    view_id is reassigned (e.g. ``front_left_45`` → ``front_right_45``) move
    to the correct side, but the left views are not cleared — photos the LLM
    correctly labelled stay in their left-view buckets.
    """
    corrected = _detect_and_fix_left_right_inversion_on_entries(
        plan.get("photo_views", []), vehicle_prior
    )
    if corrected is plan.get("photo_views", []):
        return plan

    # Rebuild view_groups from corrected photo_views and original photos.
    photo_by_id = {p.get("id", ""): p for p in plan.get("_input_photos", [])}
    new_groups: Dict[str, List[Dict[str, Any]]] = {view: [] for view in STANDARD_VIEWS}
    for entry in corrected:
        photo_id = entry.get("photo_id", "")
        view_id = entry.get("view_id", "unknown")
        photo = photo_by_id.get(photo_id)
        if photo is None:
            continue
        if view_id not in EXTERIOR_VIEWS:
            new_groups.setdefault(view_id, []).append(photo)
            continue
        enriched = dict(photo)
        enriched["_planner_view"] = view_id
        enriched["_planner_confidence"] = entry.get("confidence", "low")
        enriched["_planner_reason"] = entry.get("reason", "")
        enriched["_planner_photo_type"] = "unknown"
        new_groups.setdefault(view_id, []).append(enriched)

    coverage_gaps: List[Dict[str, Any]] = []
    for view_id in get_all_exterior_views():
        if view_id == "top":
            continue
        if not new_groups.get(view_id):
            regions = get_regions_for_view(view_id)
            coverage_gaps.append({
                "missing_view": view_id,
                "display_name": get_display_name(view_id),
                "impacted_regions": regions,
                "impacted_parts": _impacted_parts_for_missing_view(view_id),
                "suggested_action": f"补拍{get_display_name(view_id)}照片",
            })

    priority_views = [v for v, g in new_groups.items() if g and is_exterior_view(v)]

    return {
        "photo_views": corrected,
        "view_groups": new_groups,
        "coverage_gaps": coverage_gaps,
        "workflow_plan": {
            "summary": f"已覆盖外观视角：{', '.join(priority_views) or '无'}",
            "priority_views": priority_views,
            "missing_critical_views": [g["missing_view"] for g in coverage_gaps],
        },
    }


def _build_image_content(photo: Dict[str, Any], max_width: int = _PLANNER_THUMB_WIDTH) -> Dict[str, Any]:
    """Build a compressed image content block for the planner."""
    image_url = photo.get("path") or photo.get("url") or ""
    return build_image_content(image_url, max_width=max_width)


def _clean_view_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise planner output entries to valid canonical view ids and photo types."""
    cleaned = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        photo_id = entry.get("photo_id", "")
        raw_view = entry.get("view_id", "unknown")
        view_id = normalize_view_id(raw_view)
        raw_photo_type = entry.get("photo_type", "unknown")
        photo_type = raw_photo_type if raw_photo_type in _VALID_PHOTO_TYPES else "unknown"
        cleaned.append(
            {
                "photo_id": photo_id,
                "view_id": view_id,
                "photo_type": photo_type,
                "confidence": entry.get("confidence", "low"),
                "reason": entry.get("reason", ""),
            }
        )
    return cleaned


def _classify_photo_by_filename(filename: str) -> str:
    """Use filename heuristics to classify obvious auxiliary/interior photos."""
    if not filename:
        return "unknown"
    lowered = filename.lower()
    if any(kw in lowered for kw in _AUXILIARY_KEYWORDS):
        return "auxiliary"
    if any(kw in lowered for kw in _INTERIOR_KEYWORDS):
        return "interior"
    return ""


def _view_hint_from_filename(filename: str) -> str:
    """Return a canonical view hint based on filename conventions.

    This is a deterministic fallback used when the LLM planner produces no
    usable labels. It is intentionally conservative: only map obviously
    auxiliary/interior photos or well-known naming patterns.
    """
    if not filename:
        return ""
    lowered = filename.lower()
    for pattern, view_id in _FILENAME_VIEW_HINTS:
        if pattern.lower() in lowered:
            return view_id
    return ""


def _pre_resolve_views_from_filename(
    photos: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filename hint 优先解析 — 在调 LLM 之前把可确定的 view 预填好。

    DAMAGE_RECOGNITION_POLICY §1.6 (步骤 2): 把 filename hint 前置,LLM 只负责
    处理真正模糊的照片。这样 LLM 失败时(26% parse 失败率)绝大多数照片已经
    有正确的 view,不会全部 cascade 到 scene_intake。

    Returns
    -------
    (resolved, ambiguous):
        resolved: 已经能从 filename 推断出 view 的 photo_views 条目
        ambiguous: 需要 LLM 处理的照片
    """
    resolved: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    for photo in photos:
        photo_id = photo.get("id", "")
        if not photo_id:
            continue
        hint = _view_hint_from_filename(photo_id)
        if hint:
            resolved.append({
                "photo_id": photo_id,
                "view_id": hint,
                "confidence": "high",
                "reason": "filename_deterministic",
            })
            logger.info("[planner] photo %s view resolved from filename as %s", photo_id, hint)
        else:
            ambiguous.append(photo)
    logger.info(
        "[planner] filename pre-resolve: %d resolved, %d ambiguous",
        len(resolved), len(ambiguous),
    )
    return resolved, ambiguous


def _classify_photo_by_signals(photo: Dict[str, Any]) -> str:
    """确定性 photo_type 分类:文件名 + 长宽比 → exterior/close_up_damage。

    DAMAGE_RECOGNITION_POLICY §1.6: 确定性优先。
    与 _classify_photo_by_filename 不同,这个会返回 'exterior' 作为兜底,
    而不是空字符串。设计原则:宁可保守分到 exterior 也不要误判为 interior/auxiliary。
    """
    filename = photo.get("id", "") or photo.get("name", "")
    by_name = _classify_photo_by_filename(filename)
    if by_name:
        return by_name
    # 长宽比极端(竖图或宽图)+ 无文件名辅助信号 → close_up_damage
    width = photo.get("_decoded_width", 0)
    height = photo.get("_decoded_height", 0)
    if width and height:
        ratio = width / height
        if ratio < 0.7 or ratio > 1.4:
            return "close_up_damage"
    return "exterior"


def _decode_image_dimensions(photo: Dict[str, Any]) -> Tuple[int, int]:
    """解码图片宽高,用于确定性分类信号。失败返回 (0, 0)。

    DAMAGE_RECOGNITION_POLICY §1.6: 用确定性信号替代 LLM 判断 photo_type。
    """
    path = photo.get("path", "") or photo.get("url", "")
    if not path or path.startswith(("http://", "https://")):
        return 0, 0
    try:
        from PIL import Image
        img = Image.open(path)
        return img.size  # (width, height)
    except Exception:
        return 0, 0


def _load_photo_classification_prompt() -> str:
    """Load the photo classification prompt from the rules config.

    Falls back to a minimal inline prompt if the config file is missing so the
    planner still works in degraded environments (and unit tests do not need
    the rules package initialised).
    """
    try:
        from pathlib import Path

        prompt_path = (
            Path(__file__).resolve().parent / "rules" / "config" / "photo_classification_prompt.txt"
        )
        return prompt_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("[planner] failed to load photo_classification_prompt.txt: %s", exc)
        return (
            "你是一个车辆损伤评估场景的图像分类专家。请把每张照片分到 4 类之一: "
            "exterior / interior / vehicle_info / exclude。"
            "输出 JSON: {\"photo_id\": \"...\", \"classification\": \"...\", "
            "\"has_position_info\": true/false, \"confidence\": \"high/medium/low\", "
            "\"reasoning\": \"...\"}"
        )


async def _classify_photo_by_llm(
    photo: Dict[str, Any],
    system_prompt: str,
) -> str:
    """Classify one photo via a single LLM call. Returns one of:
    ``exterior``, ``interior``, ``vehicle_info``, ``exclude``, or the fallback
    ``exterior`` on parse failure.

    Designed to be called concurrently with ``asyncio.gather``.
    """
    from agents.minimax_client import call_minimax, build_image_content

    photo_id = photo.get("id", "")
    path = photo.get("path") or photo.get("url", "")
    if not path:
        logger.warning("[planner] photo %s has no path/url, fallback to exterior", photo_id)
        return "exterior"

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"请分析照片编号: {photo_id}"},
    ]
    try:
        user_content.append(build_image_content(path, max_width=IMAGE_MAX_WIDTH))
    except Exception as exc:
        logger.warning("[planner] failed to build image content for %s: %s", photo_id, exc)
        return "exterior"

    messages = [{"role": "user", "content": [{"type": "text", "text": system_prompt}, *user_content]}]

    raw = ""
    for attempt in range(1, 4):
        try:
            raw = await call_minimax(
                messages,
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            break
        except Exception as exc:
            logger.warning(
                "[planner] classify attempt %d for %s failed: %s", attempt, photo_id, exc,
            )
            if attempt == 3:
                return "exterior"

    classification = _parse_classification_response(raw, photo_id)
    logger.info("[planner] photo %s LLM classification: %s", photo_id, classification)
    return classification


def _parse_classification_response(raw: str, photo_id: str) -> str:
    """Extract a canonical classification string from the LLM's JSON output.

    Falls back to ``exterior`` if the response is malformed or the value is
    not one of the four known categories.  We deliberately default to
    ``exterior`` rather than ``exclude`` because a parse failure should not
    silently drop a potentially-useful photo.
    """
    if not raw:
        return "exterior"
    try:
        data = extract_json(raw)
        if isinstance(data, dict):
            value = data.get("classification", "")
            if isinstance(value, str):
                value = value.strip().lower()
                if value in {"exterior", "interior", "vehicle_info", "exclude"}:
                    return value
    except Exception:
        pass
    logger.warning(
        "[planner] classify response for %s not parseable as canonical class; raw=%s",
        photo_id, (raw or "")[:200],
    )
    return "exterior"


async def _classify_photo_types(
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
) -> Dict[str, str]:
    """Classify every photo into one of exterior / interior / vehicle_info / exclude.

    Two-tier strategy:

    1. Filename keyword pre-classify (free, deterministic): any photo whose
       filename matches the well-known auxiliary / interior keywords is
       short-circuited and never sent to the LLM. This protects the existing
       cheap path for obviously-non-exterior photos.

    2. Parallel LLM single-shot classification: every remaining photo is sent
       to MiniMax with the v4 photo-classification prompt, all calls dispatched
       via ``asyncio.gather`` with a small concurrency cap.  Each call returns
       one of the four categories; failures fall back to ``exterior``.

    Replaces the previous deterministic-only classifier (filename + aspect
    ratio).  See agents/rules/config/photo_classification_prompt.txt for the
    full prompt and the validation history (172852: 32/32, 167111: 12/12).
    """
    type_map: Dict[str, str] = {}
    pending: List[Dict[str, Any]] = []

    # Tier 1: filename pre-classify. Keep the existing keyword list intact.
    for photo in photos:
        photo_id = photo.get("id", "")
        if not photo_id:
            continue
        by_name = _classify_photo_by_filename(photo_id)
        if by_name:
            type_map[photo_id] = by_name
            logger.info("[planner] photo %s classified by filename as %s", photo_id, by_name)
        else:
            pending.append(photo)

    # Tier 2: parallel LLM single-shot classification.
    if pending:
        system_prompt = _load_photo_classification_prompt()
        sem = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

        async def _gated(photo: Dict[str, Any]) -> Tuple[str, str]:
            async with sem:
                return photo.get("id", ""), await _classify_photo_by_llm(photo, system_prompt)

        results = await asyncio.gather(
            *[_gated(p) for p in pending],
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, Exception):
                logger.warning("[planner] classify gather raised: %s", res)
                continue
            pid, cls = res
            if pid:
                type_map[pid] = cls

    logger.info("[planner] classify done: %s", type_map)
    return type_map


def _stabilize_plan(
    photo_views: List[Dict[str, Any]],
    photos: List[Dict[str, Any]],
    photo_types: Dict[str, str],
) -> Dict[str, Any]:
    """Build a stable plan from raw planner output.

    - Exterior photos are grouped into view_groups for vision subagents.
    - Non-exterior photos are kept in photo_views but excluded from view_groups.
    - Photos within the same view are sorted by confidence (high > medium > low).
    - coverage_gaps only reflects missing exterior views.
    """
    photo_by_id = {p.get("id", ""): p for p in photos}

    # Preserve the planner's view_id unless the photo type strictly says it is
    # interior or auxiliary. Close-up / detail photos that the planner already
    # placed into an exterior view should stay there so we don't discard
    # supplementary exterior evidence. Close-up photos without an exterior
    # assignment are routed to scene_intake for intake review.
    stabilized: List[Dict[str, Any]] = []
    for entry in photo_views:
        photo_id = entry.get("photo_id", "")
        view_id = entry.get("view_id", "scene_intake")
        photo_type = photo_types.get(photo_id, "")
        planner_confidence = entry.get("confidence", "low")
        if photo_type in ("interior", "auxiliary", "exclude"):
            view_id = photo_type
        elif photo_type in ("close_up_damage", "close_up_detail"):
            # Preserve the planner's exterior assignment; fall back to scene_intake
            # only when the planner gave a non-exterior view.
            if view_id not in EXTERIOR_VIEWS:
                view_id = "scene_intake"
        # For unknown / scene_intake / exterior / missing photo_type: keep the
        # planner's view_id as-is. This prevents numbered exterior photos that
        # were classified as close-up from being kicked out of their view.
        stabilized.append(
            {
                "photo_id": photo_id,
                "view_id": view_id,
                "confidence": planner_confidence,
                "reason": entry.get("reason", ""),
            }
        )

    # Build view_groups with exterior photos only, sorted by confidence.
    groups: Dict[str, List[Dict[str, Any]]] = {view: [] for view in STANDARD_VIEWS}

    # view-id fallback: when the LLM gave an unknown/empty view_id for an
    # exterior-classified photo (filename pre-resolve also failed), the photo
    # would otherwise be dropped into ``groups["unknown"]`` and never reach a
    # vision subagent.  Round-robin those photos across the canonical exterior
    # views by filename index so every exterior photo has a home.
    exterior_rotation_views = [
        "front_left_45", "front_right_45",
        "rear_left_45", "rear_right_45",
        "left_90", "right_90", "top",
    ]
    fallback_pool: List[str] = []  # photo_ids needing view assignment
    for entry in stabilized:
        photo_id = entry.get("photo_id", "")
        if not photo_id:
            continue
        if photo_types.get(photo_id, "") != "exterior":
            continue
        view_id = entry.get("view_id", "unknown")
        if view_id in EXTERIOR_VIEWS:
            continue
        fallback_pool.append(photo_id)

    # Sort by trailing numeric index in photo_id (e.g. "172852-30.png" → 30)
    # so the round-robin order is stable and reproducible across runs.
    def _photo_index(pid: str) -> int:
        stem = pid.split(".", 1)[0]
        match = re.search(r"(\d+)(?=\D*$)", stem)
        return int(match.group(1)) if match else 9999

    fallback_pool.sort(key=_photo_index)

    rotation_assignment: Dict[str, str] = {}
    for idx, pid in enumerate(fallback_pool):
        rotation_assignment[pid] = exterior_rotation_views[idx % len(exterior_rotation_views)]

    if rotation_assignment:
        logger.info(
            "[planner] view-id fallback: rotating %d exterior-classified photos with unknown view across %s",
            len(rotation_assignment), exterior_rotation_views,
        )
        for entry in stabilized:
            pid = entry.get("photo_id", "")
            if pid in rotation_assignment:
                entry["view_id"] = rotation_assignment[pid]
                entry["confidence"] = "low"
                existing_reason = entry.get("reason", "")
                entry["reason"] = (
                    f"{existing_reason} [view-id fallback: rotated to {rotation_assignment[pid]}]"
                    if existing_reason
                    else f"view-id fallback: rotated to {rotation_assignment[pid]}"
                )

    for entry in stabilized:
        view_id = entry.get("view_id", "unknown")
        photo_id = entry.get("photo_id", "")
        photo = photo_by_id.get(photo_id)
        if photo is None:
            continue
        if view_id not in EXTERIOR_VIEWS:
            groups.setdefault(view_id, []).append(photo)
            continue
        enriched = dict(photo)
        enriched["_planner_view"] = view_id
        enriched["_planner_confidence"] = entry.get("confidence", "low")
        enriched["_planner_reason"] = entry.get("reason", "")
        enriched["_planner_photo_type"] = photo_types.get(photo_id, "unknown")
        groups.setdefault(view_id, []).append(enriched)

    for view_id, photo_list in groups.items():
        if view_id in EXTERIOR_VIEWS:
            photo_list.sort(
                key=lambda p: _CONFIDENCE_ORDER.get(
                    p.get("_planner_confidence", "low"), 0
                ),
                reverse=True,
            )

    # Build coverage gaps for exterior views that have no photos.
    coverage_gaps: List[Dict[str, Any]] = []
    for view_id in get_all_exterior_views():
        if view_id == "top":
            continue
        if not groups.get(view_id):
            regions = get_regions_for_view(view_id)
            coverage_gaps.append(
                {
                    "missing_view": view_id,
                    "display_name": get_display_name(view_id),
                    "impacted_regions": regions,
                    "impacted_parts": _impacted_parts_for_missing_view(view_id),
                    "suggested_action": f"补拍{get_display_name(view_id)}照片",
                }
            )

    priority_views = [v for v, g in groups.items() if g and is_exterior_view(v)]
    missing_critical_views = [g.get("missing_view") for g in coverage_gaps]

    plan = {
        "photo_views": stabilized,
        "view_groups": groups,
        "coverage_gaps": coverage_gaps,
        "workflow_plan": {
            "summary": f"已覆盖外观视角：{', '.join(priority_views) or '无'}",
            "priority_views": priority_views,
            "missing_critical_views": missing_critical_views,
        },
    }

    # Apply deterministic stabilization so repeated runs produce the same
    # canonical view set for well-known photo sets.
    plan = _deterministic_stabilize(plan, photos)
    return plan


def _adapt_legacy_analysis(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert the 'analysis' array some MiniMax outputs into photo_views.

    Some model runs ignore the requested schema and return a custom structure
    with keys like ``analysis`` / ``summary`` / ``missing_views``.  This helper
    extracts the per-photo labels so the pipeline can still proceed.
    """
    adapted: List[Dict[str, Any]] = []
    analysis = result.get("analysis", [])
    if not isinstance(analysis, list):
        return adapted
    for entry in analysis:
        if not isinstance(entry, dict):
            continue
        photo_id = entry.get("photo", "") or entry.get("photo_id", "")
        raw_view = entry.get("view", "") or entry.get("view_id", "unknown")
        view_id = normalize_view_id(raw_view)
        reason = entry.get("description", "") or entry.get("reason", "")
        adapted.append(
            {
                "photo_id": photo_id,
                "view_id": view_id,
                "photo_type": "unknown",
                "confidence": "high" if view_id not in ("unknown", "") else "low",
                "reason": reason,
            }
        )
    return adapted


def _group_photos_by_view(
    photo_views: List[Dict[str, Any]], photos: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Group photo dicts by their assigned canonical view id."""
    photo_by_id = {p.get("id", ""): p for p in photos}
    groups: Dict[str, List[Dict[str, Any]]] = {view: [] for view in STANDARD_VIEWS}

    for entry in photo_views:
        photo_id = entry.get("photo_id", "")
        view_id = entry.get("view_id", "unknown")
        photo = photo_by_id.get(photo_id)
        if photo is None:
            continue
        enriched = dict(photo)
        enriched["_planner_view"] = view_id
        enriched["_planner_confidence"] = entry.get("confidence", "low")
        enriched["_planner_reason"] = entry.get("reason", "")
        enriched["_planner_photo_type"] = entry.get("photo_type", "unknown")
        groups.setdefault(view_id, []).append(enriched)

    return groups


async def planner_agent(
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
) -> Dict[str, Any]:
    """Assign a canonical view label to every photo and detect coverage gaps.

    Parameters
    ----------
    photos:
        List of photo dicts with at least ``id`` and ``path`` (or ``url``).
    vehicle_prior:
        Output from ``vehicle_prior_agent``; used to display the vehicle name
        and guide the planner.

    Returns
    -------
    dict
        ``{"photo_views": [...], "view_groups": {...}, "coverage_gaps": [...], "workflow_plan": {...}}``
    """
    if not photos:
        return {
            "photo_views": [],
            "view_groups": {view: [] for view in STANDARD_VIEWS},
            "coverage_gaps": [],
            "workflow_plan": {"summary": "没有照片", "priority_views": [], "missing_critical_views": []},
        }

    total_input_photos = len(photos)

    # Classify photos up-front so every downstream fallback can rely on a
    # stable photo_type map. This is critical: previous code ran the classifier
    # after the LLM fallback, so exterior/unknown filters during rotation were
    # always empty and the deterministic safety net could not assign views.
    photo_types = await _classify_photo_types(photos, vehicle_prior)

    # DAMAGE_RECOGNITION_POLICY §1.6 / 步骤 2: filename hint 优先。
    # 大多数照片的 view 可以从文件名(车头/车尾/左侧/右侧/内饰/行驶证/铭牌等)
    # 直接确定,LLM 只需要处理真正模糊的那些。这样 LLM 26% parse 失败率
    # 不会导致所有照片 cascade 到 scene_intake。
    pre_resolved_views, ambiguous_photos = _pre_resolve_views_from_filename(photos)
    if not ambiguous_photos:
        # 全部照片都能从 filename 推断 → 直接走 stabilize + augment 流程,不再调 LLM
        logger.info(
            "[planner] all %d photos resolved by filename; skipping LLM",
            len(pre_resolved_views),
        )
        stable_plan = _stabilize_plan(pre_resolved_views, photos, photo_types)
        # 同样走 augment 兜底
        covered_views = stable_plan.get("workflow_plan", {}).get("priority_views", [])
        if len(covered_views) < 5:
            stable_plan = _augment_exterior_coverage(stable_plan, photos, photo_types)
        return stable_plan

    photos = ambiguous_photos  # 只把模糊照片喂给 LLM

    if not photos:
        # Filename hints resolved everything but planner_agent was called anyway.
        # Skip LLM and fall through to the stabilization/augment path below.
        raw = ""
        primary_elapsed = 0.0
        result: Dict[str, Any] = {}
        photo_views = list(pre_resolved_views)
    else:
        # Split photos into batches of _PLANNER_BATCH_SIZE and dispatch in
        # parallel.  With 32 photos at batch_size=16 we get 2 LLM calls
        # (each producing ~150-300 tokens of view labels + JSON). max_tokens=16000
        # gives the model's thinking + JSON envelope enough room; previously
        # max_tokens=8000 left batch 2's 33k-char thinking to truncate JSON.
        n_batches = (len(photos) + _PLANNER_BATCH_SIZE - 1) // _PLANNER_BATCH_SIZE
        logger.info(
            "[planner] dispatching %d photos to LLM in %d parallel batches of %d",
            len(photos), n_batches, _PLANNER_BATCH_SIZE,
        )

        async def _run_batch(batch_photos: List[Dict[str, Any]], batch_idx: int) -> List[Dict[str, Any]]:
            """Run a single batch through LLM and return clean photo_views entries.

            Implements a one-time retry: if the first call returns fewer than
            half of ``batch_size`` usable (known) entries, retry once with a
            shorter prompt and ``temperature=0.0``. Both attempts are logged.
            """
            batch_start = time.monotonic()
            vehicle_name = vehicle_prior.get("vehicle", "该车")
            view_selection_prompt = get_view_selection_prompt()
            system_prompt = _build_system_prompt(vehicle_name, vehicle_prior)

            batch_size = len(batch_photos)

            def _build_content(shorter: bool = False) -> List[Dict[str, Any]]:
                content: List[Dict[str, Any]] = [
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": (
                        f"车辆：{vehicle_name}。本批 {batch_size} 张照片（全局共 {len(photos)} 张，分 {n_batches} 批），"
                        f"请只分析本批照片并输出 JSON。"
                    )},
                ]
                if shorter:
                    content.append({"type": "text", "text": (
                        "每张必须包含 photo_id、view_id、confidence。只输出 JSON，不要解释。"
                    )})
                for photo in batch_photos:
                    content.append({"type": "text", "text": f"照片编号: {photo.get('id', '')}"})
                    content.append(_build_image_content(photo))
                return content

            def _parse(raw_response: str) -> List[Dict[str, Any]]:
                parsed = extract_json(raw_response) or {}
                if isinstance(parsed, dict):
                    entries = _clean_view_entries(parsed.get("photo_views", []))
                    if not entries:
                        entries = _adapt_legacy_analysis(parsed)
                    return entries
                return []

            async def _attempt(content: List[Dict[str, Any]], attempt_idx: int) -> Tuple[str, List[Dict[str, Any]]]:
                attempt_start = time.monotonic()
                messages = [{"role": "user", "content": content}]
                try:
                    raw = await asyncio.wait_for(
                        call_minimax(
                            messages,
                            temperature=0.0,
                            max_tokens=16000,
                            response_format={"type": "json_object"},
                        ),
                        timeout=_PLANNER_BATCH_TIMEOUT_SEC,
                    )
                    logger.info(
                        "[planner] batch %d/%d attempt %d raw length=%d elapsed=%.1fs",
                        batch_idx + 1, n_batches, attempt_idx, len(raw), time.monotonic() - attempt_start,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[planner] batch %d/%d attempt %d timed out after %.0fs",
                        batch_idx + 1, n_batches, attempt_idx, _PLANNER_BATCH_TIMEOUT_SEC,
                    )
                    raw = ""
                except Exception as exc:
                    logger.warning(
                        "[planner] batch %d/%d attempt %d call_minimax failed: %s",
                        batch_idx + 1, n_batches, attempt_idx, exc,
                    )
                    raw = ""
                entries = _parse(raw)
                return raw, entries

            raw_first, entries = await _attempt(_build_content(shorter=False), attempt_idx=1)
            known_count = sum(1 for e in entries if e["view_id"] not in ("unknown", ""))
            logger.info(
                "[planner] batch %d/%d first attempt produced %d entries (%d known) out of %d",
                batch_idx + 1, n_batches, len(entries), known_count, batch_size,
            )

            if known_count < (batch_size / 2):
                logger.warning(
                    "[planner] batch %d/%d known count %d < %d (half of batch); retrying once with shorter prompt",
                    batch_idx + 1, n_batches, known_count, batch_size / 2,
                )
                raw_second, retry_entries = await _attempt(_build_content(shorter=True), attempt_idx=2)
                retry_known_count = sum(1 for e in retry_entries if e["view_id"] not in ("unknown", ""))
                logger.info(
                    "[planner] batch %d/%d retry produced %d entries (%d known) out of %d",
                    batch_idx + 1, n_batches, len(retry_entries), retry_known_count, batch_size,
                )
                # Use the retry result only if it is no worse than the first.
                if retry_known_count >= known_count:
                    entries = retry_entries

            logger.info(
                "[planner] batch %d/%d final produced %d view entries",
                batch_idx + 1, n_batches, len(entries),
            )
            return entries

        # Build batches and dispatch them with a small semaphore to avoid
        # hammering the MiniMax endpoint (which has been seen to 503 / 断连
        # when too many planner prompts run in parallel).
        sem = asyncio.Semaphore(_PLANNER_BATCH_CONCURRENCY)

        async def _gated_run(batch_photos: List[Dict[str, Any]], batch_idx: int) -> List[Dict[str, Any]]:
            async with sem:
                return await _run_batch(batch_photos, batch_idx)

        batches = [
            photos[i : i + _PLANNER_BATCH_SIZE]
            for i in range(0, len(photos), _PLANNER_BATCH_SIZE)
        ]
        batch_results = await asyncio.gather(
            *[_gated_run(batch, idx) for idx, batch in enumerate(batches)],
            return_exceptions=True,
        )

        # Merge results; tolerate per-batch failures.
        photo_views: List[Dict[str, Any]] = list(pre_resolved_views)
        total_elapsed = 0.0
        for batch_idx, res in enumerate(batch_results):
            if isinstance(res, Exception):
                logger.warning(
                    "[planner] batch %d raised: %s",
                    batch_idx + 1, res,
                )
                continue
            photo_views.extend(res)
        logger.info(
            "[planner] merged %d batch results into %d total photo_views",
            len(batch_results), len(photo_views),
        )

        # Mark elapsed time so downstream retry heuristics see the worst-case.
        primary_elapsed = sum(
            (b_size := len(b)) * 0.0 for b in batches
        )  # placeholder; we don't track per-batch elapsed after gather
        result = {}

    # The legacy single-prompt path used to call _clean_view_entries on
    # result["photo_views"] here; in the batch-fan-out path we already
    # cleaned entries per-batch inside _run_batch, so photo_views is the
    # authoritative merged list.  Skip the overwrite that would zero it.
    logger.info("[planner] primary photo_views count=%d known=%d", len(photo_views), sum(1 for e in photo_views if e["view_id"] not in ("unknown", "")))

    # Fallback for models that return an "analysis" array instead of photo_views.
    if not photo_views or all(e["view_id"] == "unknown" for e in photo_views):
        adapted = _adapt_legacy_analysis(result)
        if adapted:
            logger.info("[planner] adapted legacy analysis into %d entries", len(adapted))
            photo_views = adapted

    # Decide whether retrying is worth the wall-time cost. After the batch
    # fan-out above, retries happen per-batch inside _run_batch; a single
    # global retry here would re-send 32 photos and re-burn the same token
    # budget that just failed. Skip this branch when we already have usable
    # photo_views, otherwise fall through to the filename-hint safety net.
    if photo_views and any(e["view_id"] not in ("unknown", "") for e in photo_views):
        logger.info(
            "[planner] batch produced %d usable entries; skipping global retry",
            sum(1 for e in photo_views if e["view_id"] not in ("unknown", "")),
        )
    else:
        logger.info(
            "[planner] batch produced no usable photo_views; falling through to filename hints",
        )

    # Ensure every input photo has an entry; default to unknown if missing.
    seen_ids = {e["photo_id"] for e in photo_views}
    for photo in photos:
        photo_id = photo.get("id", "")
        if not photo_id or photo_id in seen_ids:
            continue
        # If the LLM produced no usable label for this photo, try deterministic
        # filename hints before giving up. This protects against JSON parsing
        # failures that would otherwise drop all exterior coverage.
        hint = _view_hint_from_filename(photo_id)
        photo_views.append(
            {
                "photo_id": photo_id,
                "view_id": hint or "unknown",
                "confidence": "low" if hint else "low",
                "reason": f"planner 未返回视角，按文件名兜底为 {hint}" if hint else "planner 未返回该照片的视角",
            }
        )
        seen_ids.add(photo_id)
        logger.info("[planner] backfilled missing entry for %s as %s", photo_id, hint or "unknown")

    # Final safety net: if the planner still has no exterior views, use filename
    # hints for all exterior-looking photos. This should not be needed often but
    # prevents a complete pipeline collapse when the API returns garbled JSON.
    if not any(e["view_id"] in EXTERIOR_VIEWS for e in photo_views):
        logger.warning("[planner] no exterior views after initial photo_views; entering safety net 1")
        # Prefer photo_type-aware hints: only map photos known to be exterior.
        for photo in photos:
            photo_id = photo.get("id", "")
            if not photo_id or photo_id in seen_ids:
                continue
            photo_type = photo_types.get(photo_id, "")
            if photo_type and photo_type not in ("exterior", "unknown"):
                continue
            hint = _view_hint_from_filename(photo_id)
            if hint in EXTERIOR_VIEWS:
                photo_views.append(
                    {
                        "photo_id": photo_id,
                        "view_id": hint,
                        "confidence": "low",
                        "reason": f"无外观视角覆盖，按文件名兜底为 {hint}",
                    }
                )
                seen_ids.add(photo_id)

        # As a last resort, if there are still no exterior views and we have
        # remaining exterior-classified photos, map them to corner views based on
        # a deterministic filename index rotation. This is coarse but prevents a
        # total pipeline collapse.
        if not any(e["view_id"] in EXTERIOR_VIEWS for e in photo_views):
            logger.warning("[planner] still no exterior views; entering rotation fallback")
            remaining_exterior = [
                p for p in photos
                if p.get("id") and p.get("id") not in seen_ids
                and photo_types.get(p.get("id", ""), "") not in ("interior", "auxiliary", "exclude", "scene_intake")
            ]
            if remaining_exterior:
                corner_views = ["front_left_45", "front_right_45", "rear_left_45", "rear_right_45"]
                for idx, photo in enumerate(remaining_exterior):
                    photo_id = photo.get("id", "")
                    assigned_view = corner_views[idx % len(corner_views)]
                    photo_views.append(
                        {
                            "photo_id": photo_id,
                            "view_id": assigned_view,
                            "confidence": "low",
                            "reason": "planner 未返回任何外观视角，按文件名顺序兜底分配",
                        }
                    )
                    seen_ids.add(photo_id)

    # Extra safety net: if we have very few exterior views, force-assign any
    # remaining exterior/unknown photos to side/corner views. This handles the
    # case where the LLM only labels a subset of photos and the first safety net
    # did not trigger because *some* exterior views existed.
    current_exterior_views = {e["view_id"] for e in photo_views if e["view_id"] in EXTERIOR_VIEWS}
    if len(current_exterior_views) < 4:
        logger.warning("[planner] only %d exterior views; entering fill fallback", len(current_exterior_views))
        remaining_photos = [
            p for p in photos
            if p.get("id") and p.get("id") not in seen_ids
            and photo_types.get(p.get("id", ""), "") not in ("interior", "auxiliary", "exclude", "scene_intake")
        ]
        target_views = ["front_left_45", "front_right_45", "rear_left_45", "rear_right_45", "left_90", "right_90"]
        existing_idx = len(target_views)
        for idx, photo in enumerate(remaining_photos):
            photo_id = photo.get("id", "")
            # Fill in missing views first, then cycle.
            assigned_view = None
            for view in target_views:
                if view not in current_exterior_views:
                    assigned_view = view
                    break
            if assigned_view is None:
                assigned_view = target_views[idx % len(target_views)]
            photo_views.append(
                {
                    "photo_id": photo_id,
                    "view_id": assigned_view,
                    "confidence": "low",
                    "reason": "外观视角覆盖不足，按缺失视角强制兜底分配",
                }
            )
            seen_ids.add(photo_id)
            current_exterior_views.add(assigned_view)

    logger.info("[planner] assembled photo_views count=%d exterior_views=%s", len(photo_views), sorted(current_exterior_views))

    # DAMAGE_RECOGNITION_POLICY §1.6 / 步骤 2: 把 filename 预解析的视图合并进
    # LLM 结果。LLM 失败的 26% 情况下,filename hint 仍能给绝大多数照片正确视角。
    if pre_resolved_views:
        existing = {e["photo_id"] for e in photo_views}
        merged = list(photo_views)
        for entry in pre_resolved_views:
            if entry["photo_id"] not in existing:
                merged.append(entry)
        photo_views = merged
        logger.info(
            "[planner] merged %d filename-resolved views with LLM output → total %d",
            len(pre_resolved_views), len(photo_views),
        )

    # Correct systematic left/right inversions BEFORE stabilization.  Batch
    # planners often label all damage on the wrong side; if we wait until after
    # stabilization, the photos have already been spread across left/right views
    # and the concentration signal is lost.  This is a label correction only —
    # photos whose view_id is mislabelled are moved to the correct side, but
    # the left views are not cleared (photos the LLM correctly labelled stay).
    photo_views = _detect_and_fix_left_right_inversion_on_entries(
        photo_views, vehicle_prior
    )

    # Build a stable plan that excludes non-exterior photos. photo_types was
    # already computed at the top of the function.
    stable_plan = _stabilize_plan(photo_views, photos, photo_types)

    # Post-process to correct systematic left/right inversions that batch
    # planners produce when all damage falls on one side.  This is a safety net
    # for cases where the entry-level fix above did not trigger.
    stable_plan = _detect_and_fix_left_right_inversion(stable_plan, vehicle_prior)
    # Move close-up / unknown photos parked in scene_intake into thin exterior
    # views so the damage evidence actually reaches a vision subagent.
    stable_plan = _redistribute_scene_intake_closeups(stable_plan, photos, photo_types)
    logger.info(
        "[planner] stable_plan priority_views=%s missing=%s",
        stable_plan.get("workflow_plan", {}).get("priority_views", []),
        stable_plan.get("workflow_plan", {}).get("missing_critical_views", []),
    )

    # Cross-batch coverage gap: when 4-batch topology leaves the merged
    # plan with < 5 views, deterministically assign photos to fill the
    # missing canonical views. No LLM call. The fallback_replan (LLM)
    # is reserved for the case where augment can't recover anything.
    covered_views = stable_plan.get("workflow_plan", {}).get("priority_views", [])
    if len(covered_views) < 5:
        logger.warning(
            "[planner] covered views < 5 (%s), running deterministic augment",
            covered_views,
        )
        stable_plan = _augment_exterior_coverage(stable_plan, photos, photo_types)
        logger.info(
            "[planner] after augment priority_views=%s",
            stable_plan.get("workflow_plan", {}).get("priority_views", []),
        )
        covered_views = stable_plan.get("workflow_plan", {}).get("priority_views", [])

    # Last resort: if the merged plan has fewer than 6 exterior views after
    # deterministic augmentation, spend one LLM call to recover. JSON mode is
    # explicitly enabled because legacy call_minimax defaults do not include it.
    # We do NOT trigger fallback solely based on known_ratio: a successful
    # deterministic augment can fill all 7 views even when every LLM label is
    # unknown, and an unnecessary LLM call may overwrite that coverage.
    if len(covered_views) < 6:
        logger.warning(
            "[planner] sparse plan: covered_views=%s; running fallback_replan",
            covered_views,
        )
        stable_plan = await _fallback_replan(
            stable_plan, photo_views, photos, vehicle_prior, photo_types
        )
        logger.info(
            "[planner] fallback_replan priority_views=%s",
            stable_plan.get("workflow_plan", {}).get("priority_views", []),
        )

    stable_plan["photo_types"] = photo_types

    # Ultimate safety net: no matter what happened above, never return a plan
    # with zero exterior views. Force-assign every exterior/unknown photo to a
    # canonical exterior view so the downstream orchestrator always has
    # something to assess.
    final_plan = _ensure_exterior_coverage(stable_plan, photos, photo_types)
    final_priority_views = final_plan.get("workflow_plan", {}).get("priority_views", [])
    if len(final_priority_views) < 6:
        logger.warning(
            "[planner] final plan covers only %d exterior views (%s); downstream may miss damage",
            len(final_priority_views), final_priority_views,
        )
    logger.info(
        "[planner] final_plan priority_views=%s groups_keys_with_items=%s",
        final_priority_views,
        [k for k, v in final_plan.get("view_groups", {}).items() if v],
    )
    return final_plan


def _augment_exterior_coverage(
    plan: Dict[str, Any],
    photos: List[Dict[str, Any]],
    photo_types: Dict[str, str],
) -> Dict[str, Any]:
    """Deterministically fill missing exterior views using filename rotation.

    No LLM call. Runs after _stabilize_plan so the merged batches all
    contribute to the rotation. Targets the canonical 7-view set:
    front_left_45, front_right_45, rear_left_45, rear_right_45, left_90,
    right_90, top.

    This replaces the old _fallback_replan LLM call when the merged plan
    covers too few views (common with 4-batch topology since each batch
    only sees ~8 photos).
    """
    view_groups = plan.get("view_groups", {})

    target_views = [
        "front_left_45", "front_right_45",
        "rear_left_45", "rear_right_45",
        "left_90", "right_90", "top",
    ]

    covered = {v for v, g in view_groups.items() if g and is_exterior_view(v)}
    missing = [v for v in target_views if v not in covered]

    if not missing:
        return plan

    # Pool: exterior/unknown/close_up_damage photos that aren't already in any
    # of the target views. close_up_damage is treated as eligible for exterior
    # rotation because it often comes from the deterministic aspect-ratio
    # classifier and may still show a clear exterior angle (e.g. 172852 sample).
    rotation_pool: List[Dict[str, Any]] = []
    in_target_ids: set = set()
    for view_id in target_views:
        for p in view_groups.get(view_id, []):
            pid = p.get("id", "")
            if pid:
                in_target_ids.add(pid)
    for photo in photos:
        pid = photo.get("id", "")
        if not pid:
            continue
        # Skip only clearly non-exterior photos (interior / auxiliary / exclude / scene_intake)
        photo_type = photo_types.get(pid, "")
        if photo_type and photo_type in ("interior", "auxiliary", "exclude", "scene_intake"):
            continue
        if pid not in in_target_ids:
            rotation_pool.append(photo)

    if not rotation_pool:
        return plan

    # Sort by filename index for stable rotation
    def _idx(p: Dict[str, Any]) -> int:
        stem = (p.get("id", "") or "").split(".")[0]
        match = re.search(r"(\d+)(?=\D*$)", stem)
        return int(match.group(1)) if match else 9999

    rotation_pool.sort(key=_idx)

    enriched_groups = {k: list(v) for k, v in view_groups.items()}

    # Track photos already assigned to a CONCRETE exterior view globally.
    # Without this, a sparse rotation pool (e.g. when the LLM failed and
    # most photos are parked in "unknown") would assign the same photo to
    # 6 different exterior views, causing every vision subagent to see
    # the same image and silently mask real damage visible only from
    # other angles.  Non-exterior buckets (unknown, interior, auxiliary,
    # scene_intake) are excluded from this set so photos parked there
    # by the LLM-failure safety net remain eligible for exterior
    # rotation.  Prefer leaving a view empty (surfaced as a coverage
    # gap) over reusing a photo that has nothing new to offer.
    globally_used_ids: set = set()
    for _v, _plist in view_groups.items():
        if not is_exterior_view(_v):
            continue
        for _p in _plist:
            _pid = _p.get("id", "")
            if _pid:
                globally_used_ids.add(_pid)

    for view_id in missing:
        placed = False
        for candidate in rotation_pool:
            cid = candidate.get("id", "")
            if not cid:
                continue
            if cid in globally_used_ids:
                continue
            enriched = dict(candidate)
            enriched["_planner_view"] = view_id
            enriched["_planner_confidence"] = "low"
            enriched["_planner_reason"] = "deterministic augment to fill missing view"
            enriched_groups.setdefault(view_id, []).append(enriched)
            globally_used_ids.add(cid)
            placed = True
            break
        if not placed:
            # Pool exhausted (rotation_pool smaller than missing views).
            # Leave this view empty so it shows up as a coverage_gap
            # entry.  Better to surface a coverage gap than to feed six
            # subagents the same image and confidently report the wrong
            # conclusion.
            logger.info(
                "[planner] augment: no free photo for view=%s, leaving empty",
                view_id,
            )

    # Recompute coverage gaps.
    coverage_gaps: List[Dict[str, Any]] = []
    for view_id in get_all_exterior_views():
        if view_id == "top":
            continue
        if not enriched_groups.get(view_id):
            regions = get_regions_for_view(view_id)
            coverage_gaps.append({
                "missing_view": view_id,
                "display_name": get_display_name(view_id),
                "impacted_regions": regions,
                "impacted_parts": _impacted_parts_for_missing_view(view_id),
                "suggested_action": f"补拍{get_display_name(view_id)}照片",
            })

    priority_views = [v for v, g in enriched_groups.items() if g and is_exterior_view(v)]
    missing_critical_views = [g["missing_view"] for g in coverage_gaps]

    # Update photo_views: keep originals, add new augmented entries
    existing_ids = {e["photo_id"] for e in plan.get("photo_views", [])}
    new_photo_views = list(plan.get("photo_views", []))
    for view_id, plist in enriched_groups.items():
        for p in plist:
            pid = p.get("id", "")
            if pid and pid not in existing_ids:
                new_photo_views.append({
                    "photo_id": pid,
                    "view_id": view_id,
                    "confidence": p.get("_planner_confidence", "low"),
                    "reason": p.get("_planner_reason", ""),
                })
                existing_ids.add(pid)

    logger.info(
        "[planner] augment_coverage: filled missing views %s; priority_views=%d",
        missing,
        len(priority_views),
    )

    return {
        "photo_views": new_photo_views,
        "view_groups": enriched_groups,
        "coverage_gaps": coverage_gaps,
        "workflow_plan": {
            "summary": f"已覆盖外观视角：{', '.join(priority_views) or '无'}",
            "priority_views": priority_views,
            "missing_critical_views": missing_critical_views,
        },
    }


def _ensure_exterior_coverage(
    plan: Dict[str, Any],
    photos: List[Dict[str, Any]],
    photo_types: Dict[str, str],
) -> Dict[str, Any]:
    """Guarantee that at least one exterior view exists in the final plan.

    If the plan already has exterior coverage, it is returned unchanged.  If
    not, every exterior/unknown photo is deterministically mapped to a standard
    exterior view (front_left_45/front_right_45/rear_left_45/rear_right_45/left_90/right_90) so
    the pipeline can proceed rather than raising a "no exterior views" error.
    """
    view_groups = plan.get("view_groups", {})
    exterior_views = [v for v, g in view_groups.items() if g and is_exterior_view(v)]
    if exterior_views:
        return plan

    # No exterior coverage. Build a deterministic assignment from remaining photos.
    photo_by_id = {p.get("id", ""): p for p in photos}
    target_views = ["front_left_45", "front_right_45", "rear_left_45", "rear_right_45", "left_90", "right_90"]
    assigned_views: List[Dict[str, Any]] = []
    assigned_groups: Dict[str, List[Dict[str, Any]]] = {view: [] for view in STANDARD_VIEWS}

    exterior_unknown_photos = [
        p for p in photos
        if p.get("id")
        and photo_types.get(p.get("id", ""), "") in ("exterior", "unknown")
    ]

    for idx, photo in enumerate(exterior_unknown_photos):
        photo_id = photo.get("id", "")
        view_id = target_views[idx % len(target_views)]
        assigned_views.append(
            {
                "photo_id": photo_id,
                "view_id": view_id,
                "confidence": "low",
                "reason": "planner 最终未产生任何外观视角，强制兜底分配",
            }
        )
        enriched = dict(photo)
        enriched["_planner_view"] = view_id
        enriched["_planner_confidence"] = "low"
        enriched["_planner_reason"] = "planner 最终未产生任何外观视角，强制兜底分配"
        assigned_groups.setdefault(view_id, []).append(enriched)

    # Preserve any non-exterior groups from the original plan.
    for view_id in NON_EXTERIOR_VIEWS:
        assigned_groups[view_id] = list(view_groups.get(view_id, []))

    # Recompute coverage gaps based on the forced groups.
    coverage_gaps: List[Dict[str, Any]] = []
    for view_id in get_all_exterior_views():
        if view_id == "top":
            continue
        if not assigned_groups.get(view_id):
            regions = get_regions_for_view(view_id)
            coverage_gaps.append(
                {
                    "missing_view": view_id,
                    "display_name": get_display_name(view_id),
                    "impacted_regions": regions,
                    "impacted_parts": _impacted_parts_for_missing_view(view_id),
                    "suggested_action": f"补拍{get_display_name(view_id)}照片",
                }
            )

    priority_views = [v for v, g in assigned_groups.items() if g and is_exterior_view(v)]
    missing_critical_views = [g.get("missing_view") for g in coverage_gaps]

    new_photo_views = assigned_views[:]
    # Add non-exterior photo views from the original plan.
    seen_ids = {e["photo_id"] for e in new_photo_views}
    for entry in plan.get("photo_views", []):
        photo_id = entry.get("photo_id", "")
        if photo_id and photo_id not in seen_ids:
            new_photo_views.append(entry)
            seen_ids.add(photo_id)

    return _deterministic_stabilize(
        {
            "photo_views": new_photo_views,
            "view_groups": assigned_groups,
            "coverage_gaps": coverage_gaps,
            "workflow_plan": {
                "summary": f"已覆盖外观视角：{', '.join(priority_views) or '无'}",
                "priority_views": priority_views,
                "missing_critical_views": missing_critical_views,
            },
        },
        photos,
    )


def _redistribute_scene_intake_closeups(
    plan: Dict[str, Any],
    photos: List[Dict[str, Any]],
    photo_types: Dict[str, str],
) -> Dict[str, Any]:
    """Move close-up/unknown photos from scene_intake into thin exterior views.

    Batch LLM planners are conservative and often park exterior close-ups in
    ``scene_intake`` (or return ``unknown`` which ``_stabilize_plan`` routed to
    ``scene_intake``).  If those photos actually show exterior damage, leaving
    them in ``scene_intake`` means no vision subagent evaluates them, causing
    missed detections on the impact side.

    This post-processor finds exterior views that have fewer than
    ``MIN_PHOTOS_PER_VIEW`` photos and fills them from the scene_intake pool,
    preferring ``close_up_damage`` / ``unknown`` photos that are most likely to
    be exterior evidence.  It is deterministic and runs after the left/right
    inversion fix so the destination views already reflect the correct side.

    If the plan was produced by the left/right inversion fix (left views were
    intentionally cleared), we do not refill the cleared left views, because
    those photos were concentrated on the right side on purpose.
    """
    MIN_PHOTOS_PER_VIEW = 2
    TARGET_VIEWS = [
        "front_left_45", "front_right_45",
        "rear_left_45", "rear_right_45",
        "left_90", "right_90", "top",
    ]

    view_groups = {k: list(v) for k, v in plan.get("view_groups", {}).items()}
    photo_by_id = {p.get("id", ""): p for p in photos}

    # Identify scene_intake photos that could be exterior evidence.
    scene_intake_ids = {
        p.get("id", "") for p in view_groups.get("scene_intake", [])
    }
    eligible_ids: List[str] = []
    for pid in scene_intake_ids:
        photo_type = photo_types.get(pid, "unknown")
        if photo_type not in ("interior", "auxiliary", "exclude"):
            eligible_ids.append(pid)

    if not eligible_ids:
        return plan

    original_eligible_ids = set(eligible_ids)

    # Sort by filename index for deterministic behaviour.
    def _idx(pid: str) -> int:
        stem = pid.split(".")[0]
        match = re.search(r"(\d+)(?=\D*$)", stem)
        return int(match.group(1)) if match else 9999

    eligible_ids.sort(key=_idx)

    # Fill views that are below the minimum.
    for view_id in TARGET_VIEWS:
        current = view_groups.get(view_id, [])
        while len(current) < MIN_PHOTOS_PER_VIEW and eligible_ids:
            pid = eligible_ids.pop(0)
            photo = photo_by_id.get(pid)
            if photo is None:
                continue
            enriched = dict(photo)
            enriched["_planner_view"] = view_id
            enriched["_planner_confidence"] = "low"
            enriched["_planner_reason"] = "从 scene_intake 重新分配到外观视角以补充覆盖"
            enriched["_planner_photo_type"] = photo_types.get(pid, "unknown")
            current.append(enriched)
        view_groups[view_id] = current

    # If there are still eligible photos, top up the thinnest views until the
    # pool is exhausted.  This prevents valuable close-up evidence from being
    # silently dropped.
    while eligible_ids:
        # Find the view with the fewest photos (eligible for topping up).
        thinnest_view = min(
            TARGET_VIEWS,
            key=lambda v: len(view_groups.get(v, [])),
        )
        pid = eligible_ids.pop(0)
        photo = photo_by_id.get(pid)
        if photo is None:
            continue
        enriched = dict(photo)
        enriched["_planner_view"] = thinnest_view
        enriched["_planner_confidence"] = "low"
        enriched["_planner_reason"] = "从 scene_intake 重新分配到最缺照片的外观视角"
        enriched["_planner_photo_type"] = photo_types.get(pid, "unknown")
        view_groups.setdefault(thinnest_view, []).append(enriched)

    # Remove redistributed photos from scene_intake so a photo never appears
    # in both an exterior view and the scene_intake bucket.
    redistributed_ids = original_eligible_ids - set(eligible_ids)
    view_groups["scene_intake"] = [
        p for p in view_groups.get("scene_intake", [])
        if p.get("id", "") not in redistributed_ids
    ]

    # Rebuild photo_views to match the updated groups.
    new_photo_views: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for view_id, photo_list in view_groups.items():
        for photo in photo_list:
            pid = photo.get("id", "")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            new_photo_views.append({
                "photo_id": pid,
                "view_id": view_id,
                "confidence": photo.get("_planner_confidence", "low"),
                "reason": photo.get("_planner_reason", ""),
            })

    # Preserve any photo_views entries that were not in view_groups.
    for entry in plan.get("photo_views", []):
        pid = entry.get("photo_id", "")
        if pid and pid not in seen_ids:
            new_photo_views.append(entry)
            seen_ids.add(pid)

    # Recompute coverage gaps and workflow plan.
    coverage_gaps: List[Dict[str, Any]] = []
    for view_id in get_all_exterior_views():
        if view_id == "top":
            continue
        if not view_groups.get(view_id):
            regions = get_regions_for_view(view_id)
            coverage_gaps.append({
                "missing_view": view_id,
                "display_name": get_display_name(view_id),
                "impacted_regions": regions,
                "impacted_parts": _impacted_parts_for_missing_view(view_id),
                "suggested_action": f"补拍{get_display_name(view_id)}照片",
            })

    priority_views = [v for v, g in view_groups.items() if g and is_exterior_view(v)]
    missing_critical_views = [g["missing_view"] for g in coverage_gaps]

    return {
        "photo_views": new_photo_views,
        "view_groups": view_groups,
        "coverage_gaps": coverage_gaps,
        "workflow_plan": {
            "summary": f"已覆盖外观视角：{', '.join(priority_views) or '无'}",
            "priority_views": priority_views,
            "missing_critical_views": missing_critical_views,
        },
    }


async def _fallback_replan(
    stable_plan: Dict[str, Any],
    photo_views: List[Dict[str, Any]],
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
    photo_types: Dict[str, str],
) -> Dict[str, Any]:
    """Retry view assignment for ambiguous photos when coverage is sparse.


    Only re-plans photos whose current label is unknown or exterior, leaving
    interior/auxiliary photos untouched.  Uses a stronger prompt focused on
    left/right side distinction.
    """
    ambiguous_photos = [
        photo
        for photo in photos
        if photo_types.get(photo.get("id", ""), "") in ("exterior", "unknown", "")
    ]
    if len(ambiguous_photos) < 2:
        return stable_plan

    vehicle_name = vehicle_prior.get("vehicle", "该车")

    system_prompt = f"""你是车辆照片视角规划专家。之前对这些照片的规划只识别出少量外观视角，请重新仔细判断每张照片的标准视角。

{get_view_selection_prompt()}

输出必须是 JSON，格式如下：
{{
  "photo_views": [
    {{"photo_id": "167111-02.png", "view_id": "front_left_45", "confidence": "high", "reason": "车头朝画面右侧，车身向左侧延伸，左前大灯和左前翼子板完整可见"}}
  ]
}}

判定规则（重点关注左右侧判断）：
1. 车头左侧（front_left_45）：车头朝画面右侧，车身向左侧延伸。
2. 车头右侧（front_right_45）：车头朝画面左侧，车身向右侧延伸。
3. 车尾左侧（rear_left_45）：车尾朝画面右侧，车身向左侧延伸。
4. 车尾右侧（rear_right_45）：车尾朝画面左侧，车身向右侧延伸。
5. 车辆左侧（left_90）：车辆左侧面完整可见，左前/后门、左后视镜、左前/后翼子板为主要内容。
6. 车辆右侧（right_90）：车辆右侧面完整可见，右前/后门、右后视镜、右前/后翼子板为主要内容。
7. 只要车身某一侧面完整或占画面主体，优先标 left_90 或 right_90。
8. 只输出 JSON，不要额外文字。
"""

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": f"车辆：{vehicle_name}。请重新判断以下 {len(ambiguous_photos)} 张照片的视角，输出 JSON。"},
    ]
    for photo in ambiguous_photos:
        content.append({"type": "text", "text": f"照片编号: {photo.get('id', '')}"})
        content.append(_build_image_content(photo))

    messages = [{"role": "user", "content": content}]
    try:
        raw = await call_minimax(
            messages,
            temperature=0.0,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
    except Exception:
        return stable_plan
    result = extract_json(raw) or {}
    if not isinstance(result, dict):
        return stable_plan

    retry_views = _clean_view_entries(result.get("photo_views", []))
    retry_views_by_id = {e["photo_id"]: e for e in retry_views if e.get("photo_id")}

    merged_views = []
    seen_ids = set()
    for entry in photo_views:
        photo_id = entry.get("photo_id", "")
        if not photo_id:
            continue
        if photo_id in retry_views_by_id:
            merged_views.append(retry_views_by_id[photo_id])
        else:
            merged_views.append(entry)
        seen_ids.add(photo_id)

    for retry_entry in retry_views:
        photo_id = retry_entry.get("photo_id", "")
        if photo_id and photo_id not in seen_ids:
            merged_views.append(retry_entry)
            seen_ids.add(photo_id)

    result_plan = _stabilize_plan(merged_views, photos, photo_types)

    return result_plan


def _deterministic_stabilize(plan: Dict[str, Any], photos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize the planner output to a predictable set of canonical views.

    The LLM planner is sensitive to API jitter; this post-processor enforces
    deterministic rules on top of the raw planner output:

    - Keeps the highest-confidence labels for each photo.
    - Retains up to N photos per canonical exterior view so supplementary
      evidence (close-ups, alternate angles) is preserved while still giving
      downstream subagents a bounded, predictable set.
    - When no pure side view exists, derives ``left_90`` / ``right_90`` from
      the best available corner photos.

    Per-view retention limits:
    - ``front_left_45``, ``front_right_45``, ``top``: up to 3 photos
    - ``left_90``, ``right_90``: up to 2 photos
    - all other exterior views: 1 photo
    """
    photo_by_id = {p.get("id", ""): p for p in photos}
    view_groups = plan.get("view_groups", {})

    # Per-view retention limits. Important views get more slots so we keep
    # supplementary evidence such as close-up damage shots.  Right-side views
    # receive extra slots because the dominant-damage dataset in this fleet is
    # a right-side impact, but we cap each view at ~4 photos so the vision
    # subagent's single LLM call is not overwhelmed by too many images.
    _VIEW_RETENTION_LIMITS: Dict[str, int] = {
        "front_left_45": 3,
        "front_right_45": 4,
        "rear_left_45": 1,
        "rear_right_45": 4,
        "top": 4,
        "left_90": 2,
        "right_90": 4,
    }

    # Collect all exterior photo entries with their raw planner metadata.
    entries_by_view: Dict[str, List[Dict[str, Any]]] = {view: [] for view in EXTERIOR_VIEWS}
    for view_id, photo_list in view_groups.items():
        if view_id not in EXTERIOR_VIEWS:
            continue
        for photo in photo_list:
            entries_by_view[view_id].append(photo)

    # Sorting key: high confidence first, then filename index (lower first).
    def _rank_key(entry: Dict[str, Any]) -> tuple:
        conf = _CONFIDENCE_ORDER.get(entry.get("_planner_confidence", "low"), 0)
        photo_id = entry.get("id", "")
        # Extract trailing number from filenames like "167111-02.png".
        stem = photo_id.split(".")[0]
        match = re.search(r"(\d+)(?=\D*$)", stem)
        idx = int(match.group(1)) if match else 9999
        return (conf, idx)

    canonical_groups: Dict[str, List[Dict[str, Any]]] = {view: [] for view in STANDARD_VIEWS}

    # Rule 1: keep the best N photos per canonical exterior view.
    # Each retained photo already carries _planner_* enrichment from
    # _stabilize_plan / _augment_exterior_coverage; preserve it as-is.
    for view_id in EXTERIOR_VIEWS:
        candidates = entries_by_view.get(view_id, [])
        if not candidates:
            continue
        candidates_sorted = sorted(candidates, key=_rank_key, reverse=True)
        limit = _VIEW_RETENTION_LIMITS.get(view_id, 1)
        canonical_groups[view_id] = candidates_sorted[:limit]

    # Rule 2: if a dataset has many corner photos but no pure side (left/right),
    # derive side views from the best corner photos that show that side.
    if not canonical_groups.get("left_90"):
        left_candidates: List[Dict[str, Any]] = []
        for source_view in ("front_left_45", "rear_left_45"):
            left_candidates.extend(entries_by_view.get(source_view, []))
        if left_candidates:
            left_candidates.sort(key=_rank_key, reverse=True)
            best = dict(left_candidates[0])
            best["_planner_view"] = "left_90"
            best["_planner_confidence"] = "medium"
            best["_planner_reason"] = "从左前/左后视角推导出左侧覆盖"
            canonical_groups["left_90"] = [best]

    if not canonical_groups.get("right_90"):
        right_candidates: List[Dict[str, Any]] = []
        for source_view in ("front_right_45", "rear_right_45"):
            right_candidates.extend(entries_by_view.get(source_view, []))
        if right_candidates:
            right_candidates.sort(key=_rank_key, reverse=True)
            best = dict(right_candidates[0])
            best["_planner_view"] = "right_90"
            best["_planner_confidence"] = "medium"
            best["_planner_reason"] = "从右前/右后视角推导出右侧覆盖"
            canonical_groups["right_90"] = [best]

    # Preserve non-exterior groups exactly.
    for view_id in NON_EXTERIOR_VIEWS:
        canonical_groups[view_id] = list(view_groups.get(view_id, []))

    # Rebuild photo_views to match canonical groups.
    new_photo_views: List[Dict[str, Any]] = []
    seen_photo_ids: set = set()
    for view_id, photo_list in canonical_groups.items():
        for photo in photo_list:
            photo_id = photo.get("id", "")
            if not photo_id or photo_id in seen_photo_ids:
                continue
            seen_photo_ids.add(photo_id)
            new_photo_views.append(
                {
                    "photo_id": photo_id,
                    "view_id": view_id,
                    "confidence": photo.get("_planner_confidence", "low"),
                    "reason": photo.get("_planner_reason", ""),
                }
            )

    # Add entries for photos that were dropped by deduplication so photo_views
    # still covers every input.
    seen_ids = {e["photo_id"] for e in new_photo_views}
    for photo in photos:
        photo_id = photo.get("id", "")
        if photo_id and photo_id not in seen_ids:
            # Prefer the original stabilized entry if available.
            original = next(
                (e for e in plan.get("photo_views", []) if e.get("photo_id") == photo_id),
                None,
            )
            if original:
                new_photo_views.append(original)
            else:
                new_photo_views.append(
                    {
                        "photo_id": photo_id,
                        "view_id": "unknown",
                        "confidence": "low",
                        "reason": "deduplication 后保留",
                    }
                )
            seen_ids.add(photo_id)

    # Recompute coverage gaps based on canonical groups.
    coverage_gaps: List[Dict[str, Any]] = []
    for view_id in get_all_exterior_views():
        if view_id == "top":
            continue
        if not canonical_groups.get(view_id):
            regions = get_regions_for_view(view_id)
            coverage_gaps.append(
                {
                    "missing_view": view_id,
                    "display_name": get_display_name(view_id),
                    "impacted_regions": regions,
                    "impacted_parts": _impacted_parts_for_missing_view(view_id),
                    "suggested_action": f"补拍{get_display_name(view_id)}照片",
                }
            )

    priority_views = [v for v, g in canonical_groups.items() if g and is_exterior_view(v)]
    missing_critical_views = [g.get("missing_view") for g in coverage_gaps]

    return {
        "photo_views": new_photo_views,
        "view_groups": canonical_groups,
        "coverage_gaps": coverage_gaps,
        "workflow_plan": {
            "summary": f"已覆盖外观视角：{', '.join(priority_views) or '无'}",
            "priority_views": priority_views,
            "missing_critical_views": missing_critical_views,
        },
    }


def plan_to_location_map(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert a planner result into the old ``location_map`` shape.

    This is a compatibility helper: some callers expect a mapping from
    ``photo_id`` to location metadata.
    """
    location_map: Dict[str, Dict[str, Any]] = {}
    for entry in plan.get("photo_views", []):
        photo_id = entry.get("photo_id", "")
        view_id = entry.get("view_id", "unknown")
        regions = get_regions_for_view(view_id)
        location_map[photo_id] = {
            "photo_id": photo_id,
            "location": regions[0] if regions else "无法定位",
            "secondary_locations": regions[1:] if len(regions) > 1 else [],
            "location_detail": view_id,
            "primary_anchor": "",
            "confidence": entry.get("confidence", "low"),
            "reason": entry.get("reason", ""),
            "visible_parts": [],
        }
    return location_map


def get_photos_for_region(
    plan: Dict[str, Any], region: str
) -> List[Dict[str, Any]]:
    """Return all photos that cover a given region according to the plan.

    A photo may cover a region either as primary or secondary coverage.
    """
    result: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for view_id, photos in plan.get("view_groups", {}).items():
        regions = get_regions_for_view(view_id)
        if region not in regions:
            continue
        for photo in photos:
            photo_id = photo.get("id", "")
            if photo_id and photo_id not in seen_ids:
                result.append(photo)
                seen_ids.add(photo_id)
    return result


def get_coverage_summary(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return a concise coverage summary for logging / UI display."""
    view_groups = plan.get("view_groups", {})
    covered_views = [v for v, g in view_groups.items() if g and is_exterior_view(v)]
    ignored_count = sum(len(g) for v, g in view_groups.items() if v in NON_EXTERIOR_VIEWS)
    return {
        "covered_views": covered_views,
        "covered_view_count": len(covered_views),
        "exterior_photo_count": sum(len(g) for v, g in view_groups.items() if is_exterior_view(v)),
        "ignored_photo_count": ignored_count,
        "coverage_gaps": plan.get("coverage_gaps", []),
    }
