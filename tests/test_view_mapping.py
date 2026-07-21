import pytest

from agents.view_mapping import (
    EXTERIOR_VIEWS,
    NON_EXTERIOR_VIEWS,
    PHOTO_TYPE_CATEGORIES,
    VIEW_TO_PARTS,
    get_all_exterior_views,
    get_display_name,
    get_parts_for_view,
    get_regions_for_view,
    is_exterior_view,
    is_valid_photo_category,
    normalize_view_id,
)


def test_exterior_views_count():
    assert len(EXTERIOR_VIEWS) == 9


def test_exterior_views_are_short_ids():
    assert EXTERIOR_VIEWS == {
        "front",
        "front_left",
        "front_right",
        "rear",
        "rear_left",
        "rear_right",
        "left",
        "right",
        "top",
    }


def test_photo_type_categories():
    assert PHOTO_TYPE_CATEGORIES == {
        "exterior", "interior", "vehicle_info", "exclude", "scene_intake"
    }


def test_get_regions_for_view_front_left():
    assert get_regions_for_view("front_left") == ["front", "left", "roof_front"]


def test_get_regions_for_view_rear_right():
    assert get_regions_for_view("rear_right") == ["rear", "right", "roof_rear"]


def test_get_regions_for_view_top_is_roof():
    assert get_regions_for_view("top") == ["roof", "roof_front", "roof_middle", "roof_rear"]


def test_get_regions_for_unknown_returns_empty():
    assert get_regions_for_view("unknown") == []
    assert get_regions_for_view("not_a_view") == []


def test_get_parts_for_view_front():
    parts = get_parts_for_view("front")
    assert "hood" in parts
    assert "bumper_front" in parts
    assert "windshield_front" in parts
    assert len(parts) == 8


def test_get_parts_for_view_left():
    parts = get_parts_for_view("left")
    assert "door_front_left" in parts
    assert "pillar_b_left" in parts
    assert len(parts) == 7


def test_get_parts_for_view_top():
    parts = get_parts_for_view("top")
    assert "roof_front" in parts
    assert "sunroof_glass" in parts
    assert len(parts) == 5


def test_view_to_parts_total():
    """Every exterior view has a non-empty part checklist."""
    for view_id in get_all_exterior_views():
        assert len(VIEW_TO_PARTS[view_id]) > 0


def test_is_exterior_view():
    assert is_exterior_view("front") is True
    assert is_exterior_view("left") is True
    assert is_exterior_view("top") is True
    assert is_exterior_view("interior") is False
    assert is_exterior_view("vehicle_info") is False
    assert is_exterior_view("exclude") is False


def test_is_valid_photo_category():
    assert is_valid_photo_category("exterior") is True
    assert is_valid_photo_category("vehicle_info") is True
    assert is_valid_photo_category("auxiliary") is False


def test_get_display_name():
    assert get_display_name("front") == "车头正前"
    assert get_display_name("front_left") == "车头左前"
    assert get_display_name("left") == "车辆左侧"


def test_normalize_view_id_legacy_to_short():
    assert normalize_view_id("front_left_45") == "front_left"
    assert normalize_view_id("left_90") == "left"
    assert normalize_view_id("rear_right_45") == "rear_right"


def test_normalize_view_id_short_unchanged():
    assert normalize_view_id("front_left") == "front_left"
    assert normalize_view_id("top") == "top"


def test_normalize_view_id_non_string():
    assert normalize_view_id(None) == "None"
    assert normalize_view_id(123) == "123"


def test_normalize_view_id_unrecognised():
    assert normalize_view_id("some random text") == "some random text"
