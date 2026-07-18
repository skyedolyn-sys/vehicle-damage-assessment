"""Face-to-part mapping — flip "facing + frame position" into vehicle left/right.

Pure deterministic module (zero LLM, zero model inference). It converts the
face_profiler's per-photo output into:

- ``camera_side``: which side of the vehicle the camera sees ("left"/"right"/None)
- ``candidate_parts``: parts determinable from the visible faces
- ``assignable_faces``: faces solid enough to assign damage to

Left/right flipping only makes sense when the vehicle front or rear is seen,
and front/back is mutually exclusive with left/right within one photo:

- facing=front, side_panel_pos=left  → head is on the frame's left, so the
  side panel visible to its left is the vehicle's RIGHT side → "right"
- facing=front, side_panel_pos=right → vehicle's LEFT side → "left"
- facing=rear,  side_panel_pos=left  → vehicle's LEFT side → "left"
- facing=rear,  side_panel_pos=right → vehicle's RIGHT side → "right"
- side_panel_pos=center (dead-on front/rear, no side panel) → None
- facing in {side, top, unclear} or side_panel_pos in {fills_frame, none} → None
"""

from __future__ import annotations

from typing import List, Optional

from config import PARTS_CATALOG

#: Coverage ranking — higher value wins when merging duplicate parts.
COVERAGE_RANK = {"glimpse": 0, "partial": 1, "dominant": 2}


def align_profile_ids(
    profiles: List[dict], photos: List[dict]
) -> List[tuple[str, dict]]:
    """Pair each face profile with its authoritative input photo id.

    The vision model occasionally echoes a photo id back in a mangled form —
    most commonly dropping the file extension (``172852-07.png`` →
    ``172852-07``).  Master then does ``face_priors.get(photo["id"])`` with the
    *authoritative* id, misses, and the photo silently falls back to the
    unconstrained legacy view_agent path — where the model self-guesses a
    left/right-bearing view and mis-attributes damage (172852: a side-unlocked
    A-pillar close-up convicted ``pillar_a_left``).

    ``face_profiler._realign_to_input`` already guarantees ``profiles`` is in
    the same order as ``photos``, so the authoritative pairing is positional.
    We prefer an exact id match when the model echoed it correctly, then an
    extension-tolerant match, and finally fall back to positional order.

    Returns a list of ``(authoritative_photo_id, profile)`` tuples, one per
    input photo, in input order.
    """
    by_exact = {p.get("photo_id"): p for p in profiles if p.get("photo_id")}

    def _strip_ext(pid: str) -> str:
        return pid.rsplit(".", 1)[0] if "." in pid else pid

    by_stem: dict[str, dict] = {}
    for p in profiles:
        pid = p.get("photo_id")
        if pid:
            by_stem.setdefault(_strip_ext(pid), p)

    aligned: List[tuple[str, dict]] = []
    for index, photo in enumerate(photos):
        pid = photo.get("id", "")
        prof = by_exact.get(pid)
        if prof is None:
            prof = by_stem.get(_strip_ext(pid))
        if prof is None and index < len(profiles):
            prof = profiles[index]
        aligned.append((pid, prof if prof is not None else {}))
    return aligned


#: Coverages that qualify a face for standalone damage assignment.
ASSIGNABLE_COVERAGES = ("dominant", "partial")

#: (facing, side_panel_pos) → camera side.  Only front/rear + left/right flip.
#: ``side_panel_pos`` is the position of the *side body panel* relative to the
#: front/rear in the frame (NOT the position of the front/rear itself):
#:   front + side panel on the front's LEFT  → that panel is the vehicle's RIGHT
#:   front + side panel on the front's RIGHT → vehicle's LEFT
#:   rear  + side panel on the rear's LEFT   → vehicle's LEFT
#:   rear  + side panel on the rear's RIGHT  → vehicle's RIGHT
_SIDE_FLIP_TABLE = {
    ("front", "left"): "right",
    ("front", "right"): "left",
    ("rear", "left"): "left",
    ("rear", "right"): "right",
}


def derive_camera_side(facing: str, side_panel_pos: str) -> Optional[str]:
    """Return "left" | "right" | None for the camera's view of the vehicle.

    The flip only applies when the front or rear face is seen and a side
    panel appears off-centre in the frame. Anything else (centred, pure
    side/top shots, frame-filling or absent panels) yields no single side.
    """
    return _SIDE_FLIP_TABLE.get((facing, side_panel_pos))


def _higher_coverage(current: str, candidate: str) -> str:
    """Return the coverage with the higher rank (dominant > partial > glimpse)."""
    if COVERAGE_RANK.get(candidate, -1) > COVERAGE_RANK.get(current, -1):
        return candidate
    return current


#: Roof-category parts are visible not only from a top-down shot but also from
#: any elevated oblique angle.  A front/rear/side 3/4 shot looking slightly down
#: at the vehicle shows the roof surface (roof_front from the front, roof_rear
#: from the rear, sunroof/roof_middle from a side), so when any of these faces
#: is visible we admit roof parts into the candidate set at ``glimpse`` coverage
#: (below the assignable threshold, so they cannot alone convict, but can be
#: corroborated or upgraded when a real roof/top observation exists).
_ROOF_OBLIQUE_FACES = ("front", "rear", "side", "left", "right")


def parts_for_faces(visible_faces: list[dict]) -> dict[str, str]:
    """Map visible faces to the determinable parts and their coverage.

    ``visible_faces``: [{"face": "right", "coverage": "dominant"}, ...]
    Returns ``{part_id: coverage}`` where a part belongs to a face when
    ``part["part_category"] == face``. Faces without a coverage are skipped.
    If a part ever matched multiple faces, the higher coverage wins.

    Roof-category parts additionally enter at ``glimpse`` whenever an elevated
    oblique face (front/rear/side) is visible — without this, a sample shot
    entirely at garage eye-level (no top-down photo) would never admit any roof
    part, and correctly-observed roof damage would be dropped as out-of-scope.

    Front-side / rear-side structural pairs (A-pillar, C-pillar, side doors,
    side fender) are geometrically invisible from a side-unlocked front/rear
    head-on view — the camera only sees the front/rear face.  Earlier code
    admitted them at ``glimpse`` whenever the front/rear face was visible,
    which surfaced pillar_a_left / pillar_b_left / pillar_c_left / trunk_lid /
    windshield_rear / bumper_rear as ViewAgent candidates and produced
    confident-but-false "damaged" verdicts (172852 geometry audit).  Tier 1
    fix: a side-unlocked head-on view admits only the front/rear face parts
    (plus the roof oblique fallback); the side pairs stay out of candidates.
    """
    face_coverage = {
        f.get("face"): f.get("coverage")
        for f in visible_faces
        if f.get("face") and f.get("coverage")
    }
    roof_obliquely_visible = any(f in face_coverage for f in _ROOF_OBLIQUE_FACES)
    locked_side: Optional[str] = next(
        (s for s in ("left", "right") if s in face_coverage), None
    )
    front_visible = "front" in face_coverage
    rear_visible = "rear" in face_coverage
    # Head-on front/rear view (no locked side): the camera cannot resolve which
    # side it sees, so front/rear-side structural pairs stay out of candidates.
    side_unlocked = (front_visible or rear_visible) and locked_side is None

    # 纯侧面照（facing=side, 未锁定左右）的双侧候选容错。
    #
    # 几何事实：一张侧面照**一定**看到车的某一侧——问题只是不知道是哪一侧。
    # 让 candidate 为空等于让 view_agent 无法定罪任何门/镜/柱（172852-10/21
    # 纯右侧照就是这样丢掉 door_rear_right / mirror_right 的）。
    #
    # 解决：给左右两套候选。view_agent 报哪侧就是哪侧——模型对"画面里这侧
    # 有没有损伤"的判断是可靠的，**不可靠的只是把这侧叫 left 还是 right**。
    # 归侧的最终决定交给 master_agent 的跨照片共识（少数服从多数），单张
    # 照片只需负责"看到什么报什么"。
    #
    # 触发条件：visible_faces 里有 side（模型报告了"看到侧面"）但没有锁定
    # left/right。**不限于纯侧面照**——3/4 视角（side + front 或 side + rear）
    # 同样需要双侧候选，因为 facing=side 意味着侧面是主体，front/rear 只是
    # 边缘可见。172852-21（side + front + roof）就是这种case：主体是右侧
    # 面，但 candidate 只有 front 部件，导致 door_rear_right 被 out_of_candidate
    # drop。
    side_only_unlocked = "side" in face_coverage and locked_side is None

    result: dict[str, str] = {}
    for part in PARTS_CATALOG:
        part_id = part["part_id"]
        coverage = face_coverage.get(part["part_category"])
        if coverage is None:
            if part["part_category"] == "roof" and roof_obliquely_visible:
                coverage = "glimpse"
            elif side_only_unlocked and part["part_category"] in ("left", "right"):
                # 纯侧面照 / 侧面 3/4 视角：左右两套侧面部件都进候选，coverage
                # 沿用模型给的 side 覆盖度（glimpse 降级为 corroborate-only，
                # 防止单侧 glimpse 证据单独定罪）。
                #
                # 优先于 side_unlocked 检查：visible_faces 有 side + front 时
                # side_unlocked 会把 left/right 部件 continue 掉，但 facing=side
                # 意味着侧面是主体，侧面部件必须进 candidate（172852-21 的
                # door_rear_right 就是这样被丢的）。
                coverage = face_coverage["side"]
            elif side_unlocked:
                continue  # head-on view: skip side structural pairs
            else:
                continue
        result[part_id] = _higher_coverage(result.get(part_id, coverage), coverage)
    return result


def assignable_faces(visible_faces: list[dict]) -> list[str]:
    """Return faces with coverage dominant/partial (glimpse is excluded)."""
    return [
        f["face"]
        for f in visible_faces
        if f.get("face") and f.get("coverage") in ASSIGNABLE_COVERAGES
    ]


def build_face_prior(photo_id: str, profile: dict) -> dict:
    """Build the face_prior dict fed to view_agent for one photo.

    ``profile`` is the face_profiler output for a single photo, with keys:
    ``facing``, ``side_panel_pos``, ``visible_faces``, ``anchor``,
    ``confidence``.

    Single source of truth for left/right: the model is unreliable at naming
    a side panel's left/right directly (it flips), so any ``left``/``right``
    face it reported is overridden by the deterministic ``camera_side``
    derived from ``side_panel_pos``.  Front/rear/roof faces pass through
    unchanged.  This collapses the two left/right signals into one and stops
    the model's side-naming errors from reaching the part list.
    """
    facing = profile.get("facing")
    side_panel_pos = profile.get("side_panel_pos")
    camera_side = derive_camera_side(facing, side_panel_pos)

    raw_faces = list(profile.get("visible_faces") or [])

    # 对称视角降侧（确定性，零方差兜底）。
    #
    # 几何事实：正中前/后拍（能同时看到左右两侧，如正面俯拍看到前挡风+车顶+两根
    # A柱）无法把画面中央的损伤侧化到某一侧立柱。
    #
    # 判定锚点：front/rear 的朝向只有在画面里有**真实可辨的车身侧面**时才可锚定
    # 左右。"真实侧面" = visible_faces 里有 dominant/partial 的 left/right/side——
    # 一条 glimpse 的侧面边缘（如纯玻璃俯拍偶尔漏出的一丝车身）**不足以**锚定
    # 左右。
    #
    # 对称降侧的边界：当模型**确实看到了车头或车尾的结构特征**（visible_faces 里
    # 包含 front 或 rear，说明模型识别出了格栅/车标/引擎盖/尾灯/牌照框等），
    # facing 本身是可靠的，**只降 camera_side 不降 facing**。这样正中正面照
    # （172852-20：能看到车头+塌陷车顶+天窗）仍能贡献 roof/sunroof/windshield 的
    # 损伤观察，不会被误降为 unclear 整张照片作废。
    #
    # 诊断实证（172852 face_profiler 3 轮）：
    #   - 24 纯玻璃俯拍：visible_faces 里**没有** front/rear（只有 roof + 边缘
    #     glimpse），facing 是模型猜的——这种已经被 face_profiler prompt 约束
    #     必须给 unclear，不需要 face_mapping 再兜底降朝向。
    #   - 02/03/21 真实右前 3/4：稳定给 side=partial/dominant → 有真实侧面锚点，
    #     保留 front+side，右前损伤正常定罪。
    #   - 20 正中正面俯拍：visible_faces 有 front + roof，没有 side → 该降
    #     camera_side（不能把天窗/A柱损伤挂到具体左右），但 facing=front 是
    #     对的（模型确实看到了车头），不能降为 unclear。
    has_real_side_anchor = any(
        f.get("face") in ("left", "right", "side")
        and COVERAGE_RANK.get(f.get("coverage"), -1) >= COVERAGE_RANK["partial"]
        for f in raw_faces
    )
    facing = profile.get("facing")
    if facing in ("front", "rear") and not has_real_side_anchor:
        # 无真实侧面锚点：对称视角。只降侧——damage 不能挂到 left/right 后缀部件。
        # facing 保留：模型既然能识别出 front/rear 的结构特征（visible_faces 里
        # 有 front/rear），facing 本身是可靠的。
        camera_side = None
    normalized_faces: list[dict] = []
    side_coverage: Optional[str] = None
    for f in raw_faces:
        face = f.get("face")
        coverage = f.get("coverage")
        if face in ("left", "right"):
            # Defer the side panel's left/right to camera_side; remember the
            # strongest coverage the model gave the side panel.
            if camera_side is None:
                continue  # no locked side -> drop the ambiguous side face
            if COVERAGE_RANK.get(coverage, -1) > COVERAGE_RANK.get(side_coverage, -1):
                side_coverage = coverage
            continue
        if face == "side":
            # ``side`` is the model's side-agnostic marker for "a side panel is
            # visible".  When camera_side is locked we still defer to it (the
            # locked side wins); when camera_side is None the ``side`` face is
            # the ONLY signal that a side body panel is visible — dropping it
            # would leave pure-side photos with an empty candidate list
            # (172852-10/21/28/31 lost door_rear_right / mirror_right this way).
            # Keep it so parts_for_faces can fan out into both left and right
            # candidates for pure-side shots; the final left/right attribution
            # is deferred to master_agent's cross-photo consensus.
            if camera_side is not None:
                if COVERAGE_RANK.get(coverage, -1) > COVERAGE_RANK.get(side_coverage, -1):
                    side_coverage = coverage
                continue
            normalized_faces.append({"face": face, "coverage": coverage})
            continue
        normalized_faces.append({"face": face, "coverage": coverage})
    if camera_side is not None and side_coverage is not None:
        normalized_faces.append({"face": camera_side, "coverage": side_coverage})

    # A photo is "usable" for a standalone damage conclusion only when its
    # facing was judged with enough confidence and is not the catch-all
    # "unclear".  Low-confidence / unclear photos are soft-downgraded: their
    # damage observations are kept but flagged so they cannot alone convict a
    # part in the final assessment (they may still corroborate).  ``facing`` is
    # the (possibly downgraded) facing — a pure-glass close-up already forced to
    # "unclear" above is therefore also unusable.
    facing_val = facing
    confidence_val = profile.get("confidence")
    usable = facing_val not in (None, "unclear") and confidence_val != "low"

    return {
        "photo_id": photo_id,
        "facing": facing_val,
        "camera_side": camera_side,
        "candidate_parts": parts_for_faces(normalized_faces),
        "assignable_faces": assignable_faces(normalized_faces),
        "anchor": profile.get("anchor"),
        "confidence": confidence_val,
        "usable": usable,
    }
