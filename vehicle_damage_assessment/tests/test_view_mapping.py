import pytest

from agents.view_mapping import (
    EXTERIOR_VIEWS,
    STANDARD_VIEWS,
    get_display_name,
    get_regions_for_view,
    is_exterior_view,
    normalize_view_id,
)


def test_standard_views_include_common_angles():
    assert "front" in STANDARD_VIEWS
    assert "front_left_45" in STANDARD_VIEWS
    assert "left_90" in STANDARD_VIEWS
    assert "top" in STANDARD_VIEWS


def test_get_regions_for_view_front_left_45():
    assert get_regions_for_view("front_left_45") == ["front", "left"]


def test_get_regions_for_view_rear_right_45():
    assert get_regions_for_view("rear_right_45") == ["rear", "right"]


def test_get_regions_for_view_top_is_roof():
    assert get_regions_for_view("top") == ["roof"]


def test_get_regions_for_unknown_returns_empty():
    assert get_regions_for_view("unknown") == []
    assert get_regions_for_view("not_a_view") == []


def test_is_exterior_view():
    assert is_exterior_view("front") is True
    assert is_exterior_view("left_90") is True
    assert is_exterior_view("interior") is False
    assert is_exterior_view("auxiliary") is False
    assert is_exterior_view("unknown") is False


def test_exterior_views_count():
    assert len(EXTERIOR_VIEWS) == 9


def test_get_display_name():
    assert get_display_name("front") == "车头正前"
    assert get_display_name("front_left_45") == "车头左前45度"


def test_normalize_view_id_direct_id():
    assert normalize_view_id("front_left_45") == "front_left_45"


def test_normalize_view_id_chinese():
    assert normalize_view_id("车头左前45度") == "front_left_45"
    assert normalize_view_id("左侧90度") == "left_90"
    assert normalize_view_id("车顶俯视") == "top"


def test_normalize_view_id_empty():
    assert normalize_view_id("") == "unknown"
    assert normalize_view_id(None) == "unknown"


def test_normalize_view_id_unrecognised():
    assert normalize_view_id("some random text") == "unknown"
