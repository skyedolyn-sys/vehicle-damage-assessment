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
    profile = {
        "facing": "rear",
        "side_panel_pos": "center",
        "visible_faces": [{"face": "rear", "coverage": "dominant"}],
        "anchor": "tailgate",
        "confidence": 0.9,
    }
    prior = build_face_prior("photo_002", profile)
    assert prior["camera_side"] is None
    assert prior["assignable_faces"] == ["rear"]
    # Tier 1 fix: side-unlocked rear close-up does NOT admit rear-side pairs.
    # Only the rear face itself + roof (oblique fallback) are in cands.
    rear_parts = {p["part_id"] for p in PARTS_CATALOG if p["part_category"] == "rear"}
    roof_parts = {p["part_id"] for p in PARTS_CATALOG if p["part_category"] == "roof"}
    assert set(prior["candidate_parts"]) == rear_parts | roof_parts


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
