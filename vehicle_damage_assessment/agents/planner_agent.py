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

import re

from agents.minimax_client import call_minimax, extract_json, build_image_content
from agents.view_mapping import (
    EXTERIOR_VIEWS,
    NON_EXTERIOR_VIEWS,
    PHOTO_TYPE_CATEGORIES,
    STANDARD_VIEWS,
    get_all_exterior_views,
    get_display_name,
    get_regions_for_view,
    get_view_selection_prompt,
    is_exterior_view,
    normalize_view_id,
)
from config import IMAGE_MAX_WIDTH, PARTS_BY_ID


#: Thumbnail width used by the planner.  Smaller images reduce cost/latency
#: while preserving enough detail for view classification.
_PLANNER_THUMB_WIDTH = 384


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
    ("内饰", "interior"),
    ("车内", "interior"),
    ("座椅", "interior"),
    ("-01.", "auxiliary"),
    ("-09.", "auxiliary"),
    ("-07.", "interior"),
    ("-08.", "interior"),
    ("-11.", "interior"),
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

_SYSTEM_PROMPT = f"""你是车辆照片视角规划专家。你的任务是一次性查看用户上传的所有车辆照片，为每张照片指定一个标准视角标签，并指出哪些外观视角缺失、需要补拍。

{{view_selection_prompt}}

输出必须是 JSON，格式如下（**必须严格使用以下字段名，不要使用其他字段名**）：
{{
  "photo_views": [
    {{"photo_id": "167111-02.png", "view_id": "front_left", "confidence": "high", "reason": "车头朝画面右侧，车身向左侧延伸，左前大灯和左前翼子板完整可见"}},
    {{"photo_id": "167111-03.png", "view_id": "front_right", "confidence": "high", "reason": "车头朝画面左侧，车身向右侧延伸，右前大灯和右前翼子板完整可见"}},
    {{"photo_id": "167111-05.png", "view_id": "rear_left", "confidence": "high", "reason": "车尾朝画面右侧，车身向左侧延伸，左后尾灯和左后翼子板完整可见"}},
    {{"photo_id": "167111-10.png", "view_id": "rear_right", "confidence": "high", "reason": "车尾朝画面左侧，车身向右侧延伸，右后尾灯和右后翼子板完整可见"}},
    {{"photo_id": "167111-07.png", "view_id": "interior", "confidence": "high", "reason": "车内后排座椅，不参与外观识别"}},
    {{"photo_id": "167111-09.png", "view_id": "auxiliary", "confidence": "high", "reason": "VIN码/铭牌，用于提取车辆信息"}}
  ],
  "coverage_gaps": [
    {{
      "missing_view": "right",
      "display_name": "右侧正侧",
      "impacted_regions": ["right"],
      "impacted_parts": ["右前门", "右后门", "右后视镜", "右后翼子板"],
      "suggested_action": "补拍车辆右侧正侧面照片"
    }}
  ],
  "workflow_plan": {{
    "summary": "已覆盖车头左侧、车头右侧、车尾左侧、车尾右侧；缺少右侧正侧和车顶",
    "priority_views": ["front_left", "front_right", "rear_left", "rear_right"],
    "missing_critical_views": ["right"]
  }}
}}

判定规则：
1. 每张照片必须指定一个 view_id，且必须是上述标准视角之一。
2. 左右判断以中国大陆左舵车辆为准：
   - 驾驶员侧 = 车辆左侧
   - 副驾驶侧 = 车辆右侧
3. **车头/车尾正前/正后判定必须严格：只有车头/车尾完全正对镜头、左右对称时才使用 front/rear；只要有左右偏移，必须使用 front_left/front_right/rear_left/rear_right。**
4. 车头视角判定（选择最严格的一项）：
   - 若画面中能看到更多驾驶员侧车身（左侧前大灯/翼子板/车门更大更完整） → front_left
   - 若画面中能看到更多副驾驶侧车身（右侧前大灯/翼子板/车门更大更完整） → front_right
   - 只有左右前大灯大小基本对称、车头完全居中时才使用 front
   - **不要因为有"车头"就把带角度的照片判为 front；front 只用于纯正面。**
5. 车尾视角判定：
   - 若画面中能看到更多驾驶员侧车身（左侧尾灯/翼子板/车门更大更完整） → rear_left
   - 若画面中能看到更多副驾驶侧车身（右侧尾灯/翼子板/车门更大更完整） → rear_right
   - 只有左右尾灯大小基本对称、车尾完全居中时才使用 rear
6. 车辆左侧应显示车辆完整驾驶员侧：左前门、左后门、左后视镜、左前/后翼子板完整可见，车头车尾只露少量边缘。
7. 车辆右侧应显示车辆完整副驾驶侧：右前门、右后门、右后视镜、右前/后翼子板完整可见，车头车尾只露少量边缘。
8. 只有外观视角（front/front_left/front_right/rear/rear_left/rear_right/left/right）参与外观损伤识别；interior、auxiliary、unknown 照片不纳入 coverage_gaps 的外观缺失判断。
9. 证件/VIN/铭牌/行驶证等标记为 auxiliary；车内照片标记为 interior；无法判断时标记为 unknown。
10. confidence 取 high/medium/low，low 表示视角判断不确定。
11. **必须严格使用字段名 photo_views、photo_id、view_id、confidence、reason、coverage_gaps、missing_view、workflow_plan。禁止输出 analysis、summary、missing_views 等其他字段名。**
12. 只输出 JSON，不要额外文字。
"""


def _build_image_content(photo: Dict[str, Any], max_width: int = _PLANNER_THUMB_WIDTH) -> Dict[str, Any]:
    """Build a compressed image content block for the planner."""
    image_url = photo.get("path") or photo.get("url") or ""
    return build_image_content(image_url, max_width=max_width)


def _clean_view_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise planner output entries to valid canonical view ids."""
    cleaned = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        photo_id = entry.get("photo_id", "")
        raw_view = entry.get("view_id", "unknown")
        view_id = normalize_view_id(raw_view)
        cleaned.append(
            {
                "photo_id": photo_id,
                "view_id": view_id,
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


async def _classify_photo_types(
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
) -> Dict[str, str]:
    """Classify each photo as exterior/interior/auxiliary/unknown.

    First applies filename heuristics, then uses a single lightweight LLM call
    for photos whose type is not obvious from the filename.
    """
    photo_by_id = {p.get("id", ""): p for p in photos if p.get("id")}
    type_map: Dict[str, str] = {}
    pending: List[Dict[str, Any]] = []

    for photo in photos:
        photo_id = photo.get("id", "")
        if not photo_id:
            continue
        filename_type = _classify_photo_by_filename(photo_id)
        if filename_type:
            type_map[photo_id] = filename_type
        else:
            pending.append(photo)

    if not pending:
        return type_map

    # Limit LLM classification to the first 6 ambiguous photos to control cost.
    pending = pending[:6]

    system_prompt = """你是车辆照片分类助手。为每张照片判断它属于哪一类：
- exterior：车身外部照片（车头、车尾、侧面、45度角、车顶等）
- interior：车内照片（座椅、方向盘、仪表盘、后排空间等）
- auxiliary：证件、VIN码、铭牌、行驶证、车牌特写等辅助信息照片
- unknown：无法判断

输出严格 JSON：
{
  "classifications": [
    {"photo_id": "167111-02.png", "photo_type": "exterior", "reason": "车头外观"},
    {"photo_id": "167111-07.png", "photo_type": "interior", "reason": "车内座椅"}
  ]
}
只输出 JSON，不要额外文字。"""

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": f"车辆：{vehicle_prior.get('vehicle', '该车')}。共 {len(pending)} 张照片，请逐张分类并输出 JSON。"},
    ]
    for photo in pending:
        content.append({"type": "text", "text": f"照片编号: {photo.get('id', '')}"})
        content.append(_build_image_content(photo, max_width=_PLANNER_THUMB_WIDTH))

    messages = [{"role": "user", "content": content}]
    try:
        raw = await call_minimax(messages, temperature=0.0, max_tokens=2000)
    except Exception:
        raw = ""
    result = extract_json(raw) or {}
    if not isinstance(result, dict):
        result = {}

    for item in result.get("classifications", []):
        if not isinstance(item, dict):
            continue
        photo_id = item.get("photo_id", "")
        photo_type = item.get("photo_type", "unknown").lower()
        if photo_id and photo_type in PHOTO_TYPE_CATEGORIES:
            type_map[photo_id] = photo_type

    # Any still-unclassified pending photos default to exterior (safest for damage assessment).
    for photo in pending:
        photo_id = photo.get("id", "")
        if photo_id and photo_id not in type_map:
            type_map[photo_id] = "exterior"

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

    # Enforce photo type on every entry and mark non-exterior views accordingly.
    stabilized: List[Dict[str, Any]] = []
    for entry in photo_views:
        photo_id = entry.get("photo_id", "")
        view_id = entry.get("view_id", "unknown")
        photo_type = photo_types.get(photo_id, "")
        if photo_type in ("interior", "auxiliary"):
            view_id = photo_type
        elif photo_type == "unknown" and view_id not in NON_EXTERIOR_VIEWS:
            view_id = "unknown"
        stabilized.append(
            {
                "photo_id": photo_id,
                "view_id": view_id,
                "confidence": entry.get("confidence", "low"),
                "reason": entry.get("reason", ""),
            }
        )

    # Build view_groups with exterior photos only, sorted by confidence.
    groups: Dict[str, List[Dict[str, Any]]] = {view: [] for view in STANDARD_VIEWS}
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

    # Classify photos up-front so every downstream fallback can rely on a
    # stable photo_type map. This is critical: previous code ran the classifier
    # after the LLM fallback, so exterior/unknown filters during rotation were
    # always empty and the deterministic safety net could not assign views.
    photo_types = await _classify_photo_types(photos, vehicle_prior)

    vehicle_name = vehicle_prior.get("vehicle", "该车")
    view_selection_prompt = get_view_selection_prompt()

    system_prompt = _SYSTEM_PROMPT.replace("{{view_selection_prompt}}", view_selection_prompt)
    system_prompt = system_prompt.replace("{{vehicle_name}}", vehicle_name)

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": f"车辆：{vehicle_name}。共 {len(photos)} 张照片，请逐张分析并输出 JSON。"},
    ]

    for photo in photos:
        content.append({"type": "text", "text": f"照片编号: {photo.get('id', '')}"})
        content.append(_build_image_content(photo))

    messages = [{"role": "user", "content": content}]
    try:
        raw = await call_minimax(messages, temperature=0.0, max_tokens=4000)
    except Exception as exc:
        # If the primary LLM call fails completely (e.g. server disconnect),
        # fall through to deterministic assignment instead of crashing the whole
        # pipeline.
        raw = ""
    result = extract_json(raw) or {}

    if not isinstance(result, dict):
        result = {}

    photo_views = _clean_view_entries(result.get("photo_views", []))

    # Fallback for models that return an "analysis" array instead of photo_views.
    if not photo_views or all(e["view_id"] == "unknown" for e in photo_views):
        adapted = _adapt_legacy_analysis(result)
        if adapted:
            photo_views = adapted

    # Retry once with a stronger schema reminder if we still have no useful labels.
    if not photo_views or all(e["view_id"] == "unknown" for e in photo_views):
        retry_content = [
            {"type": "text", "text": system_prompt},
            {
                "type": "text",
                "text": (
                    f"车辆：{vehicle_name}。共 {len(photos)} 张照片。\n"
                    "**重要提示**：刚才的输出格式不正确。请严格按照以下 JSON 字段名输出，不要使用 analysis/photo/view 等替代字段名：\n"
                    '{"photo_views": [{"photo_id": "...", "view_id": "...", "confidence": "...", "reason": "..."}]}'
                ),
            },
        ]
        for photo in photos:
            retry_content.append({"type": "text", "text": f"照片编号: {photo.get('id', '')}"})
            retry_content.append(_build_image_content(photo))
        retry_messages = [{"role": "user", "content": retry_content}]
        try:
            raw = await call_minimax(retry_messages, temperature=0.0, max_tokens=4000)
        except Exception as exc:
            raw = ""
        retry_result = extract_json(raw) or {}
        if isinstance(retry_result, dict):
            photo_views = _clean_view_entries(retry_result.get("photo_views", []))
            if not photo_views:
                photo_views = _adapt_legacy_analysis(retry_result)

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

    # Final safety net: if the planner still has no exterior views, use filename
    # hints for all exterior-looking photos. This should not be needed often but
    # prevents a complete pipeline collapse when the API returns garbled JSON.
    if not any(e["view_id"] in EXTERIOR_VIEWS for e in photo_views):
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
            remaining_exterior = [
                p for p in photos
                if p.get("id") and p.get("id") not in seen_ids
                and photo_types.get(p.get("id", ""), "") in ("exterior", "unknown")
            ]
            if remaining_exterior:
                corner_views = ["front_left", "front_right", "rear_left", "rear_right"]
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
        remaining_photos = [
            p for p in photos
            if p.get("id") and p.get("id") not in seen_ids
            and photo_types.get(p.get("id", ""), "") in ("exterior", "unknown")
        ]
        target_views = ["front_left", "front_right", "rear_left", "rear_right", "left", "right"]
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

    # Build a stable plan that excludes non-exterior photos. photo_types was
    # already computed at the top of the function.
    stable_plan = _stabilize_plan(photo_views, photos, photo_types)

    # Fallback: if too few exterior views were identified, retry focusing only on
    # ambiguous (unknown/exterior) photos to reduce catastrophic planner failures.
    covered_views = stable_plan.get("workflow_plan", {}).get("priority_views", [])
    if len(covered_views) < 3:
        stable_plan = await _fallback_replan(
            stable_plan, photo_views, photos, vehicle_prior, photo_types
        )

    stable_plan["photo_types"] = photo_types

    # Ultimate safety net: no matter what happened above, never return a plan
    # with zero exterior views. Force-assign every exterior/unknown photo to a
    # canonical exterior view so the downstream orchestrator always has
    # something to assess.
    final_plan = _ensure_exterior_coverage(stable_plan, photos, photo_types)
    return final_plan


def _ensure_exterior_coverage(
    plan: Dict[str, Any],
    photos: List[Dict[str, Any]],
    photo_types: Dict[str, str],
) -> Dict[str, Any]:
    """Guarantee that at least one exterior view exists in the final plan.

    If the plan already has exterior coverage, it is returned unchanged.  If
    not, every exterior/unknown photo is deterministically mapped to a standard
    exterior view (front_left/front_right/rear_left/rear_right/left/right) so
    the pipeline can proceed rather than raising a "no exterior views" error.
    """
    view_groups = plan.get("view_groups", {})
    exterior_views = [v for v, g in view_groups.items() if g and is_exterior_view(v)]
    if exterior_views:
        return plan

    # No exterior coverage. Build a deterministic assignment from remaining photos.
    photo_by_id = {p.get("id", ""): p for p in photos}
    target_views = ["front_left", "front_right", "rear_left", "rear_right", "left", "right"]
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

    return {
        "photo_views": new_photo_views,
        "view_groups": assigned_groups,
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
    {{"photo_id": "167111-02.png", "view_id": "front_left", "confidence": "high", "reason": "车头朝画面右侧，车身向左侧延伸，左前大灯和左前翼子板完整可见"}}
  ]
}}

判定规则（重点关注左右侧判断）：
1. 车头左侧（front_left）：车头朝画面右侧，车身向左侧延伸。
2. 车头右侧（front_right）：车头朝画面左侧，车身向右侧延伸。
3. 车尾左侧（rear_left）：车尾朝画面右侧，车身向左侧延伸。
4. 车尾右侧（rear_right）：车尾朝画面左侧，车身向右侧延伸。
5. 车辆左侧（left）：车辆左侧面完整可见，左前/后门、左后视镜、左前/后翼子板为主要内容。
6. 车辆右侧（right）：车辆右侧面完整可见，右前/后门、右后视镜、右前/后翼子板为主要内容。
7. 只要车身某一侧面完整或占画面主体，优先标 left 或 right。
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
        raw = await call_minimax(messages, temperature=0.0, max_tokens=4000)
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

    return _stabilize_plan(merged_views, photos, photo_types)


def _deterministic_stabilize(plan: Dict[str, Any], photos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize the planner output to a predictable set of canonical views.

    The LLM planner is sensitive to API jitter; this post-processor enforces
    deterministic rules on top of the raw planner output:

    - Keeps the highest-confidence label for each photo.
    - Merges front/rear corner views that differ only by side (e.g. two
      front_left photos) into one representative photo per canonical view.
    - When multiple photos map to the same canonical side view (left/right)
      or corner view, selects the photo whose filename index is most typical
      for that view, preferring high confidence.
    - Ensures the returned view_groups contain at most one photo per canonical
      exterior view for stable downstream subagent dispatch.

    This reduces 12-photo "车顶闸" style datasets to a predictable 6-view
    coverage: front_left, front_right, rear_left, rear_right, left, right.
    """
    photo_by_id = {p.get("id", ""): p for p in photos}
    view_groups = plan.get("view_groups", {})

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

    # Rule 1: for corner views and side views, keep only the best representative.
    # This collapses duplicate-angle photos into a single canonical dispatch.
    for view_id in EXTERIOR_VIEWS:
        candidates = entries_by_view.get(view_id, [])
        if not candidates:
            continue
        candidates_sorted = sorted(candidates, key=_rank_key, reverse=True)
        canonical_groups[view_id] = [candidates_sorted[0]]

    # Rule 2: if a dataset has many corner photos but no pure side (left/right),
    # derive side views from the best corner photos that show that side.
    if not canonical_groups.get("left"):
        left_candidates: List[Dict[str, Any]] = []
        for source_view in ("front_left", "rear_left"):
            left_candidates.extend(entries_by_view.get(source_view, []))
        if left_candidates:
            left_candidates.sort(key=_rank_key, reverse=True)
            best = dict(left_candidates[0])
            best["_planner_view"] = "left"
            best["_planner_confidence"] = "medium"
            best["_planner_reason"] = "从左前/左后视角推导出左侧覆盖"
            canonical_groups["left"] = [best]

    if not canonical_groups.get("right"):
        right_candidates: List[Dict[str, Any]] = []
        for source_view in ("front_right", "rear_right"):
            right_candidates.extend(entries_by_view.get(source_view, []))
        if right_candidates:
            right_candidates.sort(key=_rank_key, reverse=True)
            best = dict(right_candidates[0])
            best["_planner_view"] = "right"
            best["_planner_confidence"] = "medium"
            best["_planner_reason"] = "从右前/右后视角推导出右侧覆盖"
            canonical_groups["right"] = [best]

    # Preserve non-exterior groups exactly.
    for view_id in NON_EXTERIOR_VIEWS:
        canonical_groups[view_id] = list(view_groups.get(view_id, []))

    # Rebuild photo_views to match canonical groups, keeping at most one entry
    # per photo_id (prefer the canonical view over derived side views).
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
