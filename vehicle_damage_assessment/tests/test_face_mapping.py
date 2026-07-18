"""Unit tests for agents.face_mapping — deterministic face-to-part mapping."""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
try:
    import django

    django.setup()
except Exception:
    pass

import pytest

from agents.face_mapping import (
    align_profile_ids,
    assignable_faces,
    build_face_prior,
    derive_camera_side,
    parts_for_faces,
)
from config import PARTS_BY_ID, PARTS_CATALOG


# ---------------------------------------------------------------------------
# align_profile_ids — authoritative photo-id pairing (172852 pillar_a_left FP)
# ---------------------------------------------------------------------------

def _photo(pid):
    return {"id": pid, "path": f"/tmp/{pid}"}


def test_align_exact_id_match():
    photos = [_photo("172852-07.png"), _photo("172852-01.png")]
    profiles = [{"photo_id": "172852-07.png", "facing": "unclear"},
                {"photo_id": "172852-01.png", "facing": "front"}]
    aligned = align_profile_ids(profiles, photos)
    assert [pid for pid, _ in aligned] == ["172852-07.png", "172852-01.png"]
    assert aligned[0][1]["facing"] == "unclear"
    assert aligned[1][1]["facing"] == "front"


def test_align_extension_dropped_by_model():
    """Model echoes '172852-07' (no .png) — must still pair with 172852-07.png."""
    photos = [_photo("172852-07.png")]
    profiles = [{"photo_id": "172852-07", "facing": "unclear"}]
    aligned = align_profile_ids(profiles, photos)
    assert aligned[0][0] == "172852-07.png"  # authoritative id wins
    assert aligned[0][1]["facing"] == "unclear"


def test_align_positional_fallback_when_id_unrecognizable():
    photos = [_photo("a.png"), _photo("b.png")]
    profiles = [{"photo_id": "GARBAGE", "facing": "front"},
                {"photo_id": "ALSO_GARBAGE", "facing": "rear"}]
    aligned = align_profile_ids(profiles, photos)
    assert [pid for pid, _ in aligned] == ["a.png", "b.png"]
    assert aligned[0][1]["facing"] == "front"
    assert aligned[1][1]["facing"] == "rear"


def test_align_missing_profile_yields_empty_dict():
    photos = [_photo("a.png"), _photo("b.png")]
    profiles = [{"photo_id": "a.png", "facing": "front"}]
    aligned = align_profile_ids(profiles, photos)
    assert aligned[0][1]["facing"] == "front"
    assert aligned[1][1] == {}  # no profile → empty, not a KeyError


# ---------------------------------------------------------------------------
# derive_camera_side — all rule branches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "facing,side_panel_pos,expected",
    [
        # front seen: flip — head-left means the right side panel is visible
        ("front", "left", "right"),
        ("front", "right", "left"),
        # rear seen: no flip — tail-left means the left side panel is visible
        ("rear", "left", "left"),
        ("rear", "right", "right"),
        # dead-on front/rear: no single side
        ("front", "center", None),
        ("rear", "center", None),
        # pure side / top / unclear facing: no flip possible
        ("side", "left", None),
        ("side", "right", None),
        ("top", "left", None),
        ("top", "right", None),
        ("unclear", "left", None),
        ("unclear", "right", None),
        # frame-filling or absent side panel: no flip possible
        ("front", "fills_frame", None),
        ("rear", "fills_frame", None),
        ("front", "none", None),
        ("rear", "none", None),
    ],
)
def test_derive_camera_side(facing, side_panel_pos, expected):
    assert derive_camera_side(facing, side_panel_pos) == expected


# ---------------------------------------------------------------------------
# parts_for_faces
# ---------------------------------------------------------------------------

def test_parts_for_faces_right_and_front():
    visible = [
        {"face": "right", "coverage": "dominant"},
        {"face": "front", "coverage": "partial"},
    ]
    result = parts_for_faces(visible)

    assert result, "expected some candidate parts"

    for part_id, coverage in result.items():
        part = PARTS_BY_ID[part_id]
        # front is an elevated oblique face, so roof parts are also admitted at
        # glimpse coverage (they cannot convict, only corroborate).
        assert part["part_category"] in ("right", "front", "roof")
        if part["part_category"] == "right":
            assert coverage == "dominant"
        elif part["part_category"] == "front":
            assert coverage == "partial"
        else:  # roof via oblique visibility
            assert coverage == "glimpse"

    right_parts = {p["part_id"] for p in PARTS_CATALOG if p["part_category"] == "right"}
    front_parts = {p["part_id"] for p in PARTS_CATALOG if p["part_category"] == "front"}
    roof_parts = {p["part_id"] for p in PARTS_CATALOG if p["part_category"] == "roof"}
    assert set(result) == right_parts | front_parts | roof_parts


def test_parts_for_faces_roof_admitted_from_oblique_front():
    """A front 3/4 shot (no top-down photo) must still admit roof parts."""
    result = parts_for_faces([{"face": "front", "coverage": "dominant"}])
    roof_parts = {p["part_id"] for p in PARTS_CATALOG if p["part_category"] == "roof"}
    assert roof_parts, "catalog must define roof parts"
    for pid in roof_parts:
        assert pid in result, f"{pid} should be a candidate from an oblique front view"
        assert result[pid] == "glimpse"


def test_parts_for_faces_roof_admitted_from_oblique_side():
    """A pure side shot also reveals the roof surface at glimpse coverage."""
    result = parts_for_faces([{"face": "right", "coverage": "dominant"}])
    for pid in ("sunroof_glass", "roof_middle"):
        assert pid in result
        assert result[pid] == "glimpse"


def test_parts_for_faces_roof_not_admitted_without_oblique_face():
    """With no front/rear/side face, roof parts must not enter the candidate set."""
    result = parts_for_faces([{"face": "undercarriage", "coverage": "dominant"}])
    assert result == {}


def test_parts_for_faces_roof_real_roof_face_wins_over_glimpse():
    """A genuine roof face gives roof parts their real coverage, not glimpse."""
    result = parts_for_faces([
        {"face": "front", "coverage": "dominant"},
        {"face": "roof", "coverage": "partial"},
    ])
    roof_parts = {p["part_id"] for p in PARTS_CATALOG if p["part_category"] == "roof"}
    for pid in roof_parts:
        assert result[pid] == "partial"


def test_parts_for_faces_rear_closeup_no_side_drops_rear_side_pairs():
    """A rear close-up with no locked side does NOT admit rear-side pairs.

    172852 geometry audit: the legacy ``both sides admit for consensus``
    fallback let ``pillar_c_left`` / ``trunk_lid`` / ``windshield_rear`` /
    ``bumper_rear`` enter the candidate set as ``glimpse`` in a side-unlocked
    rear view, which surfaced them to the ViewAgent prompt and produced
    confident-but-false "damaged" verdicts in the evidence chain.  The Tier 1
    fix requires the camera_side to be locked before any front-side or
    rear-side structural pair is admitted — geometry, not consensus.
    """
    result = parts_for_faces([{"face": "rear", "coverage": "dominant"}])
    for pid in ("pillar_c_left", "pillar_c_right", "door_rear_left", "door_rear_right"):
        assert pid not in result, (
            f"{pid} should NOT be a candidate in a side-unlocked rear view; "
            f"the camera cannot resolve which side of the vehicle it sees."
        )
    # Roof parts still get the oblique fallback (a rear shot tilts up to roof).
    for pid in ("roof_rear", "roof_middle"):
        assert pid in result


def test_parts_for_faces_front_closeup_no_side_drops_front_side_pairs():
    """A front close-up with no locked side does NOT admit front-side pairs.

    Mirror of the rear case: a head-on front view cannot see the A-pillars'
    front surfaces; admitting them at ``glimpse`` previously caused
    ``pillar_a_left`` / ``pillar_a_right`` / ``fender_front_*`` FPs.
    """
    result = parts_for_faces([{"face": "front", "coverage": "dominant"}])
    for pid in ("pillar_a_left", "pillar_a_right", "door_front_left", "door_front_right"):
        assert pid not in result, (
            f"{pid} should NOT be a candidate in a side-unlocked front view; "
            f"the camera only sees the front face head-on."
        )
    # Front-only parts and roof still get in.
    assert result["hood"] == "dominant"
    assert "roof_front" in result


def test_parts_for_faces_front_closeup_locked_side_admits_only_that_side():
    """A front close-up WITH a locked side admits only the matching side's
    front-side pair (Tier 1 positive case)."""
    result_right = parts_for_faces([
        {"face": "front", "coverage": "dominant"},
        {"face": "right", "coverage": "dominant"},
    ])
    assert result_right["door_front_right"] == "dominant"
    assert "door_front_left" not in result_right
    assert result_right["pillar_a_right"] == "dominant"
    assert "pillar_a_left" not in result_right

    result_left = parts_for_faces([
        {"face": "front", "coverage": "dominant"},
        {"face": "left", "coverage": "dominant"},
    ])
    assert result_left["door_front_left"] == "dominant"
    assert "door_front_right" not in result_left
    assert result_left["pillar_a_left"] == "dominant"
    assert "pillar_a_right" not in result_left


def test_parts_for_faces_side_locked_does_not_admit_opposite_side():
    """With a locked right side, left-side parts must not be admitted."""
    result = parts_for_faces([
        {"face": "rear", "coverage": "dominant"},
        {"face": "right", "coverage": "dominant"},
    ])
    assert "door_rear_right" in result
    assert "door_rear_left" not in result, "locked right side must exclude left parts"
    assert "pillar_c_left" not in result


def test_parts_for_faces_right_side_suffix():
    """All right-face parts carry a _right suffix, except center-side pillars."""
    result = parts_for_faces([{"face": "right", "coverage": "dominant"}])
    assert result
    for part_id in result:
        part = PARTS_BY_ID[part_id]
        if part["side"] != "center":
            assert part_id.endswith("_right"), part_id


def test_parts_for_faces_unknown_face_ignored():
    result = parts_for_faces([{"face": "undercarriage", "coverage": "dominant"}])
    assert result == {}


def test_parts_for_faces_empty():
    assert parts_for_faces([]) == {}


def test_parts_for_faces_does_not_mutate_input():
    visible = [{"face": "left", "coverage": "partial"}]
    snapshot = [dict(f) for f in visible]
    parts_for_faces(visible)
    assert visible == snapshot


# ---------------------------------------------------------------------------
# assignable_faces
# ---------------------------------------------------------------------------

def test_assignable_faces_excludes_glimpse():
    visible = [
        {"face": "right", "coverage": "dominant"},
        {"face": "front", "coverage": "partial"},
        {"face": "roof", "coverage": "glimpse"},
    ]
    assert assignable_faces(visible) == ["right", "front"]


def test_assignable_faces_empty():
    assert assignable_faces([]) == []


# ---------------------------------------------------------------------------
# build_face_prior — end to end
# ---------------------------------------------------------------------------

def test_build_face_prior_front_right_orientation():
    # Side panel reported agnostically as ``side``; its left/right is resolved
    # from side_panel_pos.  front + side panel on the front's right → the
    # vehicle's LEFT side is showing, so candidate parts are the LEFT-face set.
    profile = {
        "facing": "front",
        "side_panel_pos": "right",
        "visible_faces": [
            {"face": "side", "coverage": "dominant"},
            {"face": "front", "coverage": "partial"},
        ],
        "anchor": "front grille + side body",
        "confidence": 0.87,
    }
    prior = build_face_prior("photo_001", profile)

    assert prior["photo_id"] == "photo_001"
    assert prior["facing"] == "front"
    assert prior["camera_side"] == "left"

    candidate_parts = prior["candidate_parts"]
    left_parts = {p["part_id"] for p in PARTS_CATALOG if p["part_category"] == "left"}
    for part_id in left_parts:
        assert candidate_parts[part_id] == "dominant"
    for part_id, coverage in candidate_parts.items():
        part = PARTS_BY_ID[part_id]
        if part["part_category"] == "left":
            expected = "dominant"
        elif part["part_category"] == "roof":
            expected = "glimpse"  # roof admitted via oblique front/left view
        else:
            expected = "partial"
        assert coverage == expected, f"{part_id}: {coverage} != {expected}"

    assert sorted(prior["assignable_faces"]) == ["front", "left"]
    assert prior["anchor"] == "front grille + side body"
    assert prior["confidence"] == 0.87


def test_build_face_prior_centered_rear_no_side():
    # 正中 rear 特写（side_panel_pos=center，visible_faces 有 rear 但无真实侧面锚点）
    # 按 BUG2c 新规则：只降 camera_side 不降 facing——模型既然识别出了 rear 的
    # 结构特征（尾灯/牌照框/尾门），facing 是可靠的；只是不能把损伤挂到 left/right。
    profile = {
        "facing": "rear",
        "side_panel_pos": "center",
        "visible_faces": [{"face": "rear", "coverage": "dominant"}],
        "anchor": "tailgate",
        "confidence": 0.9,
    }
    prior = build_face_prior("photo_002", profile)
    assert prior["camera_side"] is None, (
        "a centered rear with no side body panel must drop camera_side"
    )
    assert prior["facing"] == "rear", (
        "visible rear structural features (tailgate/tail lights) make facing reliable"
    )
    assert prior["usable"] is True


def test_build_face_prior_side_naming_overridden_by_camera_side():
    # The model mis-names the side panel (says "left") but side_panel_pos
    # locks camera_side to "right".  The prior must use camera_side, so the
    # candidate parts are the RIGHT-face set, not left.
    profile = {
        "facing": "front",
        "side_panel_pos": "left",  # side panel on front's left → vehicle RIGHT
        "visible_faces": [
            {"face": "left", "coverage": "dominant"},  # model's (wrong) naming
            {"face": "front", "coverage": "partial"},
        ],
        "anchor": "front + side",
        "confidence": 0.8,
    }
    prior = build_face_prior("p", profile)
    assert prior["camera_side"] == "right"
    right_parts = {p["part_id"] for p in PARTS_CATALOG if p["part_category"] == "right"}
    left_parts = {p["part_id"] for p in PARTS_CATALOG if p["part_category"] == "left"}
    assert right_parts <= set(prior["candidate_parts"])
    assert not (left_parts & set(prior["candidate_parts"]))


def test_build_face_prior_usable_flag():
    # high-confidence clear facing → usable; unclear or low → not usable.
    clear = build_face_prior("a", {
        "facing": "front", "side_panel_pos": "left",
        "visible_faces": [{"face": "side", "coverage": "dominant"}],
        "anchor": "grille", "confidence": "high"})
    assert clear["usable"] is True

    unclear = build_face_prior("b", {
        "facing": "unclear", "side_panel_pos": "none",
        "visible_faces": [], "anchor": "shattered glass", "confidence": "low"})
    assert unclear["usable"] is False

    low_conf = build_face_prior("c", {
        "facing": "front", "side_panel_pos": "center",
        "visible_faces": [{"face": "front", "coverage": "partial"}],
        "anchor": "weak", "confidence": "low"})
    assert low_conf["usable"] is False


# ---------------------------------------------------------------------------
# build_face_prior — 对称视角降侧关卡（BUG2b, 172852 run5 pillar_a_left FP）
# ---------------------------------------------------------------------------

def test_symmetric_front_no_side_face_drops_camera_side():
    """正面俯拍（front+roof，无侧向面入画）即使模型给了 side_panel_pos=left
    也必须把 camera_side 降为 None——画面里没有可分辨的单侧车身，侧是凭空猜的。

    172852 run5: photo 20 对称俯拍被误判 front+left → camera_side=left →
    view=front_left → pillar_a_left（MUST_REMAIN_INTACT）被中央碎玻璃/塌顶误挂。

    但 facing 保留为 front——模型确实识别出了车头特征（visible_faces 里有 front），
    该照片仍能用来定罪 roof/sunroof/windshield 等 center 部件。
    """
    profile = {
        "facing": "front",
        "side_panel_pos": "left",  # model guessed a side
        "visible_faces": [
            {"face": "front", "coverage": "dominant"},
            {"face": "roof", "coverage": "partial"},
        ],  # NO left/right/side face — symmetric
        "anchor": "前挡风玻璃+车顶",
        "confidence": "high",
    }
    prior = build_face_prior("photo_20", profile)
    assert prior["camera_side"] is None, (
        "symmetric front view with no side face must drop camera_side"
    )
    # 新规则：visible_faces 有 front（车头结构特征可见），facing 保留
    assert prior["facing"] == "front", (
        "visible front structural features (hood/windshield frame) make facing reliable"
    )
    assert prior["usable"] is True


def test_symmetric_rear_no_side_face_drops_camera_side():
    """Mirror case: rear symmetric view with no real side anchor → camera_side
    None.  But facing stays rear because visible_faces has rear (structural
    features like tailgate/tail lights are visible)."""
    profile = {
        "facing": "rear",
        "side_panel_pos": "right",
        "visible_faces": [{"face": "rear", "coverage": "dominant"}],
        "anchor": "后挡风玻璃",
        "confidence": "high",
    }
    prior = build_face_prior("photo_x", profile)
    assert prior["camera_side"] is None
    assert prior["facing"] == "rear"
    assert prior["usable"] is True


def test_side_face_present_keeps_camera_side():
    """When the model really saw a side body panel (visible_faces has side),
    the locked side is preserved — this is NOT a symmetric view.

    Guards against the gate over-firing and dropping legitimate单侧损伤定罪.
    """
    profile = {
        "facing": "front",
        "side_panel_pos": "left",  # → vehicle RIGHT
        "visible_faces": [
            {"face": "front", "coverage": "partial"},
            {"face": "side", "coverage": "dominant"},  # side body panel seen
        ],
        "anchor": "车头+右侧车身",
        "confidence": "high",
    }
    prior = build_face_prior("photo_01", profile)
    assert prior["camera_side"] == "right", (
        "a real side face in frame must keep the locked side"
    )
    assert "pillar_a_right" in prior["candidate_parts"]
    assert "pillar_a_left" not in prior["candidate_parts"]


def test_non_front_rear_facing_not_affected_by_gate():
    """side/top/unclear facings already yield camera_side None; the gate must
    not change that behaviour."""
    for facing, pos, faces in [
        ("side", "fills_frame", [{"face": "side", "coverage": "dominant"}]),
        ("top", "none", [{"face": "roof", "coverage": "dominant"}]),
        ("unclear", "none", []),
    ]:
        prior = build_face_prior("p", {
            "facing": facing, "side_panel_pos": pos,
            "visible_faces": faces, "anchor": "a", "confidence": "high"})
        assert prior["camera_side"] is None, facing


# ---------------------------------------------------------------------------
# 纯玻璃/塌顶特写降朝向（BUG2c, 172852 photo 24 前后翻转 FP）
# ---------------------------------------------------------------------------

def test_pure_glass_closeup_keeps_facing_when_rear_visible():
    """纯碎玻璃+塌顶特写，但 visible_faces 里有 rear（模型识别出了后挡风/尾灯等
    车尾结构特征）——按新规则 facing 保留 rear，只降 camera_side。

    172852 photo 24 round1: 模型报 rear+roof，确实看到了后挡风破碎和车尾结构——
    facing 是对的，只是不能把损伤挂到 left/right。
    """
    profile = {
        "facing": "rear",               # model guessed rear
        "side_panel_pos": "left",       # ... and a side
        "visible_faces": [
            {"face": "rear", "coverage": "dominant"},
            {"face": "roof", "coverage": "partial"},
        ],  # NO real side anchor → 只降 camera_side
        "anchor": "碎裂的后挡风玻璃+塌陷车顶",
        "confidence": "high",
    }
    prior = build_face_prior("photo_24", profile)
    assert prior["facing"] == "rear", (
        "visible rear structural features make facing reliable"
    )
    assert prior["camera_side"] is None
    assert prior["usable"] is True


def test_glimpse_side_is_not_a_real_anchor():
    """A pure glass close-up that only leaks a *glimpse* of side body must still
    have camera_side dropped — a sliver of side is not a real anchor for left/right.

    172852 photo 24 round2: front/center with side=glimpse — must NOT be treated
    as a real 3/4 view (which would keep right-side conviction).

    But facing stays front because visible_faces has front (hood/grille structural
    features are visible).
    """
    profile = {
        "facing": "front",
        "side_panel_pos": "center",
        "visible_faces": [
            {"face": "front", "coverage": "dominant"},
            {"face": "roof", "coverage": "partial"},
            {"face": "side", "coverage": "glimpse"},  # only a sliver — not an anchor
        ],
        "anchor": "碎玻璃+车顶",
        "confidence": "medium",
    }
    prior = build_face_prior("photo_24", profile)
    assert prior["camera_side"] is None, (
        "a glimpse of side body is not a real left/right anchor"
    )
    assert prior["facing"] == "front", (
        "visible front structural features make facing reliable"
    )
    assert prior["usable"] is True


def test_front_pure_glass_closeup_keeps_facing_when_front_visible():
    """Mirror case: front-facing symmetric view keeps facing=front when visible_faces
    has front (hood/grille structural features visible).  Only camera_side drops."""
    profile = {
        "facing": "front",
        "side_panel_pos": "right",
        "visible_faces": [
            {"face": "front", "coverage": "dominant"},
            {"face": "roof", "coverage": "glimpse"},
        ],
        "anchor": "碎玻璃+车顶",
        "confidence": "high",
    }
    prior = build_face_prior("p", profile)
    assert prior["facing"] == "front"
    assert prior["camera_side"] is None
    assert prior["usable"] is True


def test_real_three_quarter_view_keeps_facing_and_side():
    """A real 3/4 view (visible_faces has a side body panel) keeps both its
    facing AND its locked side — the anchor to front/rear is the side itself.

    Guards the gate against over-firing on legitimate rear_right / front_left
    photos (172852 04/09/29) that genuinely resolve front/rear via the side.
    """
    profile = {
        "facing": "rear",
        "side_panel_pos": "right",  # → vehicle RIGHT
        "visible_faces": [
            {"face": "rear", "coverage": "dominant"},
            {"face": "right", "coverage": "partial"},  # real side body panel
        ],
        "anchor": "Mercedes 标+尾灯+右侧车身",
        "confidence": "high",
    }
    prior = build_face_prior("photo_04", profile)
    assert prior["facing"] == "rear", "a real 3/4 view must keep its facing"
    assert prior["camera_side"] == "right"
    assert prior["usable"] is True
    assert "pillar_c_right" in prior["candidate_parts"]
    assert "taillight_rear_right" in prior["candidate_parts"]


def test_pure_side_view_unlocked_gives_both_left_right_candidates():
    """facing=side + camera_side=None（纯侧面照，未锁定左右）→ 左右两套
    侧面部件都进 candidate。

    172852-10/21/28/31 是纯右侧照，face_profiler 稳定给 facing=side +
    side_panel_pos=fills_frame → camera_side=None。原逻辑下 normalized_faces
    只有 "side"（被 defer 到 camera_side 后丢弃），parts_for_faces 找不到
    category=left/right 的部件 → candidate 空 → view_agent 无法定罪
    door_rear_right / mirror_right 等纯侧面可见部件（核心漏报源）。

    修复：纯侧面照（visible_faces 里有 side，无 front/rear，无锁侧）给
    左右两套候选，归侧最终决定交给 master_agent 跨照片共识。
    """
    profile = {
        "facing": "side",
        "side_panel_pos": "fills_frame",
        "visible_faces": [
            {"face": "side", "coverage": "dominant"},
        ],
        "anchor": "车身侧面",
        "confidence": "high",
    }
    prior = build_face_prior("photo_10", profile)
    assert prior["camera_side"] is None
    # 左右两套侧面部件都进 candidate
    for pid in (
        "door_front_left", "door_front_right",
        "door_rear_left", "door_rear_right",
        "mirror_left", "mirror_right",
        "pillar_a_left", "pillar_a_right",
        "pillar_b_left", "pillar_b_right",
        "fender_rear_left", "fender_rear_right",
    ):
        assert pid in prior["candidate_parts"], (
            f"pure-side photo should admit {pid} as candidate"
        )
    # front/rear/roof 部件不进（侧面照看不到）
    assert "windshield_front" not in prior["candidate_parts"]
    assert "windshield_rear" not in prior["candidate_parts"]
    assert "trunk_lid" not in prior["candidate_parts"]
    # roof 可以从侧面 glimpse 看到（oblique roof fallback 仍然有效）
    assert "roof_front" in prior["candidate_parts"]
    assert prior["candidate_parts"]["roof_front"] == "glimpse"


def test_pure_side_with_front_visible_does_not_double_candidate():
    """facing=side 但同时 visible_faces 里有 front（3/4 视角）→ 不触发双侧
    候选（避免在 3/4 视角里引入反向部件的假阳性）。
    """
    profile = {
        "facing": "side",
        "side_panel_pos": "fills_frame",
        "visible_faces": [
            {"face": "side", "coverage": "dominant"},
            {"face": "front", "coverage": "partial"},  # 3/4 视角
        ],
        "anchor": "车身侧面+车头一角",
        "confidence": "high",
    }
    prior = build_face_prior("photo_x", profile)
    assert prior["camera_side"] is None
    # 不触发双侧候选：左右侧面部件都不应仅靠 side_only_unlocked 进入
    # （但若 visible_faces 里有 left/right 显式锁侧则例外——本例没有）
    side_only_parts = [
        pid for pid in prior["candidate_parts"]
        if pid.endswith("_left") or pid.endswith("_right")
    ]
    # 3/4 视角只应该看到 front 部件 + roof oblique + 少量 glimpse
    # 不应该出现 mirror_left+right 这种双侧并列
    has_mirror_both = "mirror_left" in side_only_parts and "mirror_right" in side_only_parts
    assert not has_mirror_both, "3/4 view should not admit both mirror_left and mirror_right"


