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

from typing import Optional

from config import PARTS_CATALOG

#: Coverage ranking — higher value wins when merging duplicate parts.
COVERAGE_RANK = {"glimpse": 0, "partial": 1, "dominant": 2}

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

#: Front-side / rear-side structural pairs (A-pillar, C-pillar, side doors,
#: side fender) are geometrically invisible from a side-unlocked front/rear
#: head-on view — the camera only sees the front/rear face.  Earlier code
#: admitted them at ``glimpse`` whenever the front/rear face was visible, which
#: surfaced pillar_a_left / pillar_b_left / pillar_c_left / trunk_lid /
#: windshield_rear / bumper_rear as candidates to the ViewAgent prompt and
#: produced confident-but-false "damaged" verdicts (172852 geometry audit).
#: Tier 1 fix: admit only when the camera side is locked, and only the
#: matching side's pair.
def _parts_by_side(sides: tuple[str, ...]) -> dict[str, set[str]]:
    """Group PARTS_CATALOG part_ids whose ``side`` field is in ``sides``,
    keyed by the trailing side token ("left"/"right")."""
    out: dict[str, set[str]] = {"left": set(), "right": set()}
    for p in PARTS_CATALOG:
        if p["side"] in sides:
            out[p["side"].rsplit("_", 1)[-1]].add(p["part_id"])
    return out


_FRONT_SIDE_PARTS_BY_SIDE = _parts_by_side(("front_left", "front_right"))
_REAR_SIDE_PARTS_BY_SIDE = _parts_by_side(("rear_left", "rear_right"))


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

    Front-side / rear-side structural pairs only admit at ``glimpse`` when
    BOTH the front/rear face AND the corresponding side are visible (camera
    side locked).  Without a locked side, the camera only sees a head-on
    front/rear view and the side structural pairs are geometrically out of
    view — admitting them produced confident-but-false "damaged" verdicts
    for pillar_a_left / pillar_a_right / pillar_b_* / pillar_c_* /
    trunk_lid / windshield_rear / bumper_rear in the 172852 audit.
    """
    face_coverage = {
        f.get("face"): f.get("coverage")
        for f in visible_faces
        if f.get("face") and f.get("coverage")
    }
    roof_obliquely_visible = any(f in face_coverage for f in _ROOF_OBLIQUE_FACES)
    locked_side: Optional[str] = "left" if "left" in face_coverage else (
        "right" if "right" in face_coverage else None
    )
    front_visible = "front" in face_coverage
    rear_visible = "rear" in face_coverage
    # Admit structural pairs from the front/rear face only when the matching
    # side is also visible; otherwise the camera only sees the head-on face.
    front_side_cands = (
        _FRONT_SIDE_PARTS_BY_SIDE.get(locked_side or "", set())
        if front_visible and locked_side else set()
    )
    rear_side_cands = (
        _REAR_SIDE_PARTS_BY_SIDE.get(locked_side or "", set())
        if rear_visible and locked_side else set()
    )
    side_unlocked_front = front_visible and not locked_side
    side_unlocked_rear = rear_visible and not locked_side

    result: dict[str, str] = {}
    for part in PARTS_CATALOG:
        part_id = part["part_id"]
        coverage = face_coverage.get(part["part_category"])
        if coverage is None:
            if part["part_category"] == "roof" and roof_obliquely_visible:
                coverage = "glimpse"
            elif side_unlocked_front or side_unlocked_rear:
                continue  # head-on view: skip side structural pairs
            elif part_id in front_side_cands or part_id in rear_side_cands:
                coverage = "glimpse"
            else:
                continue
        result[part_id] = _higher_coverage(result[part_id], coverage) if part_id in result else coverage
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
    normalized_faces: list[dict] = []
    side_coverage: Optional[str] = None
    for f in raw_faces:
        face = f.get("face")
        coverage = f.get("coverage")
        if face in ("left", "right", "side"):
            # Defer the side panel's left/right to camera_side; remember the
            # strongest coverage the model gave the side panel.  ``side`` is
            # the model's side-agnostic marker for "a side panel is visible".
            if camera_side is None:
                continue  # no locked side -> drop the ambiguous side face
            if COVERAGE_RANK.get(coverage, -1) > COVERAGE_RANK.get(side_coverage or "", -1):
                side_coverage = coverage
            continue
        normalized_faces.append({"face": face, "coverage": coverage})
    if camera_side is not None and side_coverage is not None:
        normalized_faces.append({"face": camera_side, "coverage": side_coverage})

    # A photo is "usable" for a standalone damage conclusion only when its
    # facing was judged with enough confidence and is not the catch-all
    # "unclear".  Low-confidence / unclear photos are soft-downgraded: their
    # damage observations are kept but flagged so they cannot alone convict a
    # part in the final assessment (they may still corroborate).
    facing_val = profile.get("facing")
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
