"""Unit tests for agents.face_profiler — N=2 double-sampling merge logic."""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
try:
    import django

    django.setup()
except Exception:
    pass

from agents.face_profiler import _merge_double_sample, _merge_visible_faces


def _p(pid, facing, side_pos="none", faces=None, conf="high", anchor="a", reason=""):
    return {
        "photo_id": pid,
        "facing": facing,
        "side_panel_pos": side_pos,
        "visible_faces": faces or [],
        "anchor": anchor,
        "confidence": conf,
        "reason": reason,
    }


def test_merge_visible_faces_union_higher_coverage_wins():
    a = [{"face": "front", "coverage": "partial"}, {"face": "roof", "coverage": "glimpse"}]
    b = [{"face": "front", "coverage": "dominant"}, {"face": "side", "coverage": "partial"}]
    out = _merge_visible_faces(a, b)
    by_face = {f["face"]: f["coverage"] for f in out}
    assert by_face["front"] == "dominant", "higher coverage wins"
    assert by_face["roof"] == "glimpse"
    assert by_face["side"] == "partial"


def test_merge_visible_faces_caps_at_three():
    a = [{"face": f, "coverage": "partial"} for f in ("front", "side", "roof")]
    b = [{"face": "rear", "coverage": "dominant"}]
    out = _merge_visible_faces(a, b)
    assert len(out) <= 3


def test_double_sample_facing_agree_keeps_verdict():
    first = [_p("p1", "front", side_pos="left", conf="high")]
    second = [_p("p1", "front", side_pos="left", conf="medium")]
    out = _merge_double_sample(first, second)
    assert len(out) == 1
    assert out[0]["facing"] == "front"
    assert out[0]["side_panel_pos"] == "left"
    # confidence: max(high, medium) = high
    assert out[0]["confidence"] == "high"


def test_double_sample_facing_disagree_downgrades_to_unclear():
    """核心稳定性机制：双采样 facing 不一致 → 降级 unclear + confidence=low。

    172852 Run3 雪崩的根因就是 face_profiler 单采样把 20 的 facing 判反，
    错误的 facing 污染 candidate_parts → view_agent 在错 candidate 下报出
    大量假阳性。双采样不一致时强制 unclear，让该照片 soft-flagged unusable，
    下游不再单独定罪。
    """
    first = [_p("p1", "front", side_pos="left", conf="high")]
    second = [_p("p1", "rear", side_pos="right", conf="high")]
    out = _merge_double_sample(first, second)
    assert out[0]["facing"] == "unclear"
    assert out[0]["side_panel_pos"] == "none"
    assert out[0]["confidence"] == "low"
    assert "disagreement" in out[0]["reason"]


def test_double_sample_merges_visible_faces_even_on_disagreement():
    """facing 不一致时仍合并 visible_faces——几何信号是真实的，只是 facing 不稳。"""
    first = [_p("p1", "front", faces=[{"face": "roof", "coverage": "partial"}])]
    second = [_p("p1", "rear", faces=[{"face": "side", "coverage": "glimpse"}])]
    out = _merge_double_sample(first, second)
    faces = {f["face"] for f in out[0]["visible_faces"]}
    assert "roof" in faces
    assert "side" in faces


def test_double_sample_side_pos_fallback_to_second():
    """side_panel_pos 取第一个非 none 的——first 给 none 时 fallback 到 second。"""
    first = [_p("p1", "side", side_pos="none")]
    second = [_p("p1", "side", side_pos="fills_frame")]
    out = _merge_double_sample(first, second)
    assert out[0]["side_panel_pos"] == "fills_frame"


def test_double_sample_missing_second_photo_keeps_first():
    """second 缺某张照片时，保留 first 的结果（容错）。"""
    first = [_p("p1", "front"), _p("p2", "side")]
    second = [_p("p1", "front")]
    out = _merge_double_sample(first, second)
    assert len(out) == 2
    p2 = next(r for r in out if r["photo_id"] == "p2")
    assert p2["facing"] == "side"
