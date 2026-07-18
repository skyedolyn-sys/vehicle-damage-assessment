"""Face profiler agent — 朝向 + 面占比判定。

This agent answers the *smallest* reliable question the vision model (MiniMax M3)
can answer without breaking: the vehicle's facing direction, the side panel's
position within the frame, and the relative coverage of each visible face.

It deliberately does NOT:
- flip left/right (that is deferred to downstream deterministic code), and
- identify damage (a separate agent owns that).

Front/rear and left/right are mutually exclusive within a single photo: one
photo shows at most one front/rear orientation plus one side panel.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from agents.minimax_client import build_image_content, call_minimax, extract_json
from config import IMAGE_MAX_WIDTH

logger = logging.getLogger(__name__)
_faceprofiler_file_handler = logging.FileHandler(
    os.path.expanduser("~/vehicle_damage_assessment_faceprofiler.log"),
    mode="a",
    encoding="utf-8",
)
_faceprofiler_file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
)
_faceprofiler_file_handler.setLevel(logging.INFO)
logger.addHandler(_faceprofiler_file_handler)
logger.setLevel(logging.INFO)

# Faces the model is allowed to emit.  ``side`` is a side-agnostic marker for
# "a side panel is visible"; its left/right is resolved downstream from
# side_panel_pos.  left/right are also accepted for backward tolerance but are
# overridden by camera_side in face_mapping.
_ALLOWED_FACES = {"front", "rear", "left", "right", "roof", "side"}
_ALLOWED_COVERAGE = {"dominant", "partial", "glimpse"}
_ALLOWED_FACING = {"front", "rear", "side", "top", "unclear"}
_ALLOWED_SIDE_POS = {"left", "right", "center", "fills_frame", "none"}
_MAX_VISIBLE_FACES = 3

#: Photos per LLM call.  Small batches keep M3 from spending the whole token
#: budget on a single long <think> block (which truncates with finish=length
#: and yields no JSON).  4 photos balances throughput vs. truncation risk.
_BATCH_SIZE = 4


def _build_system_prompt(vehicle_prior: Dict[str, Any]) -> str:
    """Build the system prompt, injecting vehicle topology/anchors when present."""
    vehicle_name = vehicle_prior.get("vehicle", "该车")
    topology = vehicle_prior.get("topology")
    anchors = vehicle_prior.get("key_anchors")

    prior_block = ""
    if topology:
        prior_block += f"\n车型拓扑：\n{json.dumps(topology, ensure_ascii=False, indent=2)}\n"
    if anchors:
        prior_block += f"\n关键锚点：\n{json.dumps(anchors, ensure_ascii=False, indent=2)}\n"

    return f"""你是汽车照片朝向与面占比识别专家。给定 {vehicle_name} 的车型先验信息，判断每张照片的【朝向】、【车头/车尾在画面中的水平位置】、以及【各可见面的占比】。

{ prior_block }
你的任务非常窄：只判断朝向和面占比。不做左右翻转校正，不识别任何损伤。

对每张照片，输出一个 JSON 对象：
{{
  "photo_id": "照片编号",
  "facing": "front | rear | side | top | unclear",
  "side_panel_pos": "left | right | center | fills_frame | none",
  "visible_faces": [
    {{"face": "right", "coverage": "dominant"}},
    {{"face": "front", "coverage": "partial"}},
    {{"face": "roof",  "coverage": "glimpse"}}
  ],
  "anchor": "最强视觉锚点，一句话",
  "confidence": "high | medium | low",
  "reason": "判断理由"
}}

必须返回一个 JSON 数组，每个元素对应一张照片，顺序与输入照片一致。只输出 JSON 数组，不要额外文字。

【facing 判定规则 — 锚点约束(重要)】
- front: 必须能指出**明确车头特征**(车标/进气格栅/前大灯组/引擎盖前沿)才允许判 front。
- rear: 必须能指出**明确车尾特征**(尾灯组/牌照框/后备箱盖或尾门/后保险杠)才允许判 rear。
- **碎玻璃/钣金特写警告**: 前挡风玻璃和后挡风玻璃碎裂后外观几乎一样,单看碎玻璃**无法**区分 front 和 rear。若画面只有碎玻璃、A柱、车顶、钣金而**看不到上述明确车头/车尾特征**,**严禁**猜 front 或 rear,必须给 facing=unclear 且 confidence=low。
- **纯玻璃/塌顶特写 → 必判 unclear（重要）**: 当一张俯拍/特写只能看到**碎玻璃 + 塌陷车顶 + 柱/翼子板**,**看不到车身侧面、也看不到完整可辨的车头（格栅/车标/大灯组轮廓）或车尾（尾灯组/牌照框/尾门）结构**时,front/rear 与 left/right **全都不可判**。此时必须 facing=unclear、side_panel_pos=none、confidence=low,**绝不**据"这片玻璃像前挡/后挡"去猜朝向——猜错会把损伤挂到完全相反的一侧。
- side: 车身侧面占满,看不到明确车头或车尾特征(不要猜左右,一律 side)。
- top: 俯视,车顶/天窗/车顶钣金占主体。
- unclear: 内饰/证件/局部特写/纯碎玻璃,无法可靠判大方向。
- front 和 rear 互斥:一张图只能选一个。
- 3/4 角度(车头+侧面同现)时,facing 仍选 front 或 rear(看哪个更清晰),侧面进入 visible_faces。
- 绝不输出 left/right 作为 facing;侧面一律 side。
- **宁可 unclear 也不要硬猜**: 判 facing=front/rear 必须有锚点证据写进 anchor 字段;写不出具体锚点就给 unclear。

【side_panel_pos 判定规则 — 指"车身侧面"相对于车头/车尾的位置,不是车头本身的位置】
- 这是左右归属的唯一依据,务必看准**车身侧面**(车门/翼子板/后视镜所在的那一侧车身)在画面里位于车头/车尾的哪一边。
- facing=front(看到车头)时:
    车身侧面在车头**左边** → 给 left;在车头**右边** → 给 right;正对车头、看不到侧面 → center。
- facing=rear(看到车尾)时:
    车身侧面在车尾**左边** → 给 left;在车尾**右边** → 给 right;正对车尾 → center。
- facing=side 时：给 fills_frame（侧面占满画面，看不到车头/车尾）。
- facing=top 或 unclear 时：给 none。
- 例:车头朝画面右、车身向画面左延伸(侧面在车头左边) → side_panel_pos=left。
- 只报相对位置,**不要**自己做左右翻转,翻转由下游确定性代码完成。
- **对称/居中视角不得锁侧（重要）**：当车头(或车尾)在画面中**基本居中**、能同时看到
  左右两侧（如正面俯拍同时看到前挡风+车顶+两根A柱、正中正面、正中正后）时,
  `side_panel_pos` **必须**给 `center`（侧面占满给 `fills_frame`），**严禁**据"损伤
  看着偏哪边"去猜 left/right。**损伤位置不能用来推侧**——画面中央一团碎玻璃/塌顶
  不代表某一侧；只有**车身侧面本身明显偏向画面某一侧**时才允许给 left/right。
  拿不准就给 `center`。

【visible_faces 规则】
- 最多 3 面，按占比降序排列。face 只能取自 front/rear/roof/left/right。
- **front/rear/roof 直接报**。侧面（left/right）你只需判断"是否看到车身侧面 + 它在车头/车尾的哪一边"并填到 side_panel_pos；visible_faces 里如果包含侧面，统一用 `side` 表示（不要区分 left/right，左右归属由下游根据 side_panel_pos 确定）。
- 前与后互斥：同一张图里不要同时出现 front 和 rear。

【coverage 三档（以该面在画面中的相对面积为主要代理，兼作"可归面"判定）】
- dominant: 该面是画面主体（≥~40% 或明显视觉主体）→ 主损区 + 可归面
- partial: 清楚可见多个部件但非主体（~15-40%）→ 次损区 + 可归面
- glimpse: 只露边缘/一角（<~15%）→ 非主损 + 不单独归面（仅佐证）
- 判定以"这个面在画面里是大块/中等/一条边"的相对面积为主；若该面虽占面积但糊/反光/严重遮挡到无法判部件，降一档。

【confidence】
- high: 朝向与主面都非常明确，有强锚点
- medium: 朝向明确但侧面左右不确定，或主面占比临界
- low: 严重变形/反光/遮挡/特写，难以可靠判断

【输出纪律 — 最重要，必须严格遵守】
- **不要写长篇推理**。每张照片的 reason 和 anchor 各用**一句话**，不要逐张展开分析过程。
- **绝对禁止**在输出前进行左右翻转的多步推理（"车头在右所以是左侧……不对再想想……"）。左右翻转由下游确定性代码处理，与你无关。你只需凭**第一视觉印象**给出 visible_faces 里的 left/right，不确定就给 glimpse。
- 把 token 预算全部留给 JSON 数组本身。先想清楚结论，再一次性输出完整 JSON，不要边想边写。
- 每张照片的输出控制在 5 个字段内、总计不超过约 60 字。
"""


def _fallback_results(photos: List[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
    """Build a per-photo fallback result when parsing fails."""
    return [
        {
            "photo_id": photo["id"],
            "facing": "unclear",
            "side_panel_pos": "none",
            "visible_faces": [],
            "anchor": "无",
            "confidence": "low",
            "reason": reason,
        }
        for photo in photos
    ]


def _normalize_result(raw: Any) -> List[Dict[str, Any]] | None:
    """Normalize model output into a list of per-photo dicts.

    Accepts a bare list, a dict wrapping a ``results`` list, or a single dict.
    Returns None when the shape is unusable.
    """
    result = raw
    if isinstance(result, dict) and "results" in result:
        result = result["results"]
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        return None
    return [item for item in result if isinstance(item, dict)]


def _sanitize_visible_faces(faces: Any) -> List[Dict[str, str]]:
    """Clamp visible_faces to allowed faces/coverage and the 3-face cap."""
    if not isinstance(faces, list):
        return []
    cleaned: List[Dict[str, str]] = []
    seen_faces: set[str] = set()
    for entry in faces:
        if not isinstance(entry, dict):
            continue
        face = entry.get("face")
        coverage = entry.get("coverage")
        if face not in _ALLOWED_FACES or coverage not in _ALLOWED_COVERAGE:
            continue
        if face in seen_faces:
            continue
        seen_faces.add(face)
        cleaned.append({"face": face, "coverage": coverage})
        if len(cleaned) >= _MAX_VISIBLE_FACES:
            break
    return cleaned


def _sanitize_item(item: Dict[str, Any], photo_id: str) -> Dict[str, Any]:
    """Coerce a single model item into the output contract."""
    facing = item.get("facing")
    if facing not in _ALLOWED_FACING:
        facing = "unclear"

    side_panel_pos = item.get("side_panel_pos")
    if side_panel_pos not in _ALLOWED_SIDE_POS:
        side_panel_pos = "none"

    confidence = item.get("confidence")
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "photo_id": item.get("photo_id", photo_id),
        "facing": facing,
        "side_panel_pos": side_panel_pos,
        "visible_faces": _sanitize_visible_faces(item.get("visible_faces")),
        "anchor": str(item.get("anchor", "无")),
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
            aligned.append(
                _fallback_results([photo], "模型未返回该照片的结果")[0]
            )
            continue
        aligned.append(_sanitize_item(item, photo_id))
    return aligned


async def face_profiler_agent(
    photos: List[Dict[str, Any]], vehicle_prior: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """判定每张照片的朝向 + 画面内相对方位 + 各面占比。

    photos: [{id, path, ...}]；返回 per-photo dict 列表，顺序与输入一致。

    大批次会让 M3 在单个 <think> 块里逐张长篇推理、烧光 token 预算导致
    finish=length 截断且无 JSON 输出。因此按 ``_BATCH_SIZE`` 分批调用，
    每批只处理少量照片，既降低单次推理量又缩短输出，规避截断。
    """
    if not photos:
        return []

    results: List[Dict[str, Any]] = []
    for start in range(0, len(photos), _BATCH_SIZE):
        batch = photos[start : start + _BATCH_SIZE]
        results.extend(await _profile_batch(batch, vehicle_prior))
    return results


async def _profile_batch(
    photos: List[Dict[str, Any]], vehicle_prior: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Profile one small batch of photos (single LLM call).

    N=2 sampling for facing stability: the same batch is profiled twice with
    different sampling temperatures.  When the two runs agree on a photo's
    ``facing`` the verdict is trusted; when they disagree the photo is
    soft-downgraded to ``unclear`` so it cannot alone convict a part (its
    damage observations still corroborate).  This targets the run-to-run
    facing flip that produced run3-style cascades in 172852 stability tests
    (single-sampling face_profiler occasionally flipped front→rear on the
    collapsed-roof shot, poisoning every downstream candidate).
    """
    first = await _profile_batch_once(
        photos, vehicle_prior, temperature=0.1, reasoning_effort="low"
    )
    second = await _profile_batch_once(
        photos, vehicle_prior, temperature=0.4, reasoning_effort="medium"
    )
    return _merge_double_sample(first, second)


def _merge_double_sample(
    first: List[Dict[str, Any]], second: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge two face_profiler passes over the same batch.

    Per photo:
    - ``facing`` agreement → keep the verdict; merge ``visible_faces`` (union,
      higher coverage wins); ``side_panel_pos`` takes the first non-"none".
    - ``facing`` disagreement → downgrade to ``unclear`` + ``confidence=low``
      so the photo is soft-flagged as unusable; downstream it can corroborate
      but cannot alone convict.  ``visible_faces`` still merged (the geometry
      signal is real even when facing is unstable).
    """
    second_by_id = {p.get("photo_id"): p for p in second}
    merged: List[Dict[str, Any]] = []
    for a in first:
        pid = a.get("photo_id")
        b = second_by_id.get(pid)
        if b is None:
            merged.append(a)
            continue
        faces = _merge_visible_faces(
            a.get("visible_faces") or [], b.get("visible_faces") or []
        )
        side_pos = a.get("side_panel_pos")
        if side_pos in (None, "none"):
            side_pos = b.get("side_panel_pos")
        if a.get("facing") == b.get("facing"):
            conf_a = a.get("confidence", "low")
            conf_b = b.get("confidence", "low")
            conf = "high" if "high" in (conf_a, conf_b) else (
                "medium" if "medium" in (conf_a, conf_b) else "low"
            )
            merged.append({
                "photo_id": pid,
                "facing": a.get("facing"),
                "side_panel_pos": side_pos,
                "visible_faces": faces,
                "anchor": a.get("anchor") or b.get("anchor") or "无",
                "confidence": conf,
                "reason": a.get("reason") or b.get("reason") or "",
            })
        else:
            merged.append({
                "photo_id": pid,
                "facing": "unclear",
                "side_panel_pos": "none",
                "visible_faces": faces,
                "anchor": a.get("anchor") or b.get("anchor") or "无",
                "confidence": "low",
                "reason": (
                    f"double-sample facing disagreement: "
                    f"{a.get('facing')} vs {b.get('facing')}"
                ),
            })
    return merged


def _merge_visible_faces(
    a: List[Dict[str, str]], b: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Union of visible_faces; higher coverage wins per face; cap at 3."""
    rank = {"glimpse": 0, "partial": 1, "dominant": 2}
    by_face: Dict[str, str] = {}
    for entry in list(a) + list(b):
        face = entry.get("face")
        cov = entry.get("coverage")
        if not face or not cov:
            continue
        existing = by_face.get(face)
        if existing is None or rank.get(cov, -1) > rank.get(existing, -1):
            by_face[face] = cov
    # deterministic order: dominant first, then partial, then glimpse
    ordered = sorted(
        by_face.items(), key=lambda kv: -rank.get(kv[1], 0)
    )[:3]
    return [{"face": f, "coverage": c} for f, c in ordered]


async def _profile_batch_once(
    photos: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
    *,
    temperature: float,
    reasoning_effort: str,
) -> List[Dict[str, Any]]:
    """Profile one small batch of photos (single LLM call)."""
    system_prompt = _build_system_prompt(vehicle_prior)

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": "以下是待识别的照片，请逐张分析："},
    ]
    for photo in photos:
        content.append({"type": "text", "text": f"照片编号: {photo['id']}"})
        content.append(build_image_content(photo["path"], max_width=IMAGE_MAX_WIDTH))

    messages = [{"role": "user", "content": content}]

    logger.info(
        "[faceprofiler] start batch_size=%d temperature=%.2f effort=%s",
        len(photos), temperature, reasoning_effort,
    )
    raw = await call_minimax(
        messages,
        temperature=temperature,
        # 每张照片输出 ~60 字；留足余量但不过度,避免诱导模型写长推理。
        max_tokens=2000 * len(photos),
        response_format={"type": "json_object"},
        reasoning_effort=reasoning_effort,
    )

    parsed = extract_json(raw)
    if parsed is None:
        logger.warning("[faceprofiler] unparseable output; falling back for batch")
        return _fallback_results(
            photos,
            f"模型返回内容无法解析为 JSON: {raw[:200] if raw else '(empty)'}",
        )

    items = _normalize_result(parsed)
    if items is None:
        logger.warning("[faceprofiler] output not a list; falling back for batch")
        return _fallback_results(photos, "模型返回不是照片结果数组")

    return _realign_to_input(items, photos)
