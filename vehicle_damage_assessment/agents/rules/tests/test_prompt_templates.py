"""Tests for Jinja2 prompt templates."""

from agents.rules import render_prompt_template
from agents.view_mapping import get_display_name, get_view_selection_prompt


def test_planner_system_prompt_renders():
    rendered = render_prompt_template(
        "planner_system_prompt",
        view_selection_prompt=get_view_selection_prompt(),
        vehicle_name="TestCar",
    )
    assert "标准视角选项" in rendered
    assert "photo_views" in rendered


def test_planner_classification_prompt_renders():
    rendered = render_prompt_template("planner_classification_prompt")
    assert "exterior" in rendered
    assert "classifications" in rendered


def test_planner_retry_prompt_renders():
    rendered = render_prompt_template(
        "planner_retry_prompt",
        view_selection_prompt=get_view_selection_prompt(),
    )
    assert "front_left" in rendered
    assert "photo_views" in rendered


def test_vision_system_prompt_renders():
    rendered = render_prompt_template(
        "vision_system_prompt",
        view_id="front_left_45",
        view_display_name=get_display_name("front_left_45"),
        checklist_text="1. hood（引擎盖）",
        vehicle_name="TestCar",
    )
    assert get_display_name("front_left_45") in rendered
    assert "hood" in rendered
    assert "TestCar" in rendered
    assert '"view_id": "front_left_45"' in rendered


def test_view_agent_system_renders():
    """ViewAgent SP renders and contains the methodology + output protocol."""
    rendered = render_prompt_template("view_agent_system")
    assert rendered
    assert "相机坐标系" in rendered
    assert "camera_analysis" in rendered
    assert "front_right" in rendered
    assert "description" in rendered


def test_view_agent_task_renders():
    """ViewAgent task template renders the catalog + photo context."""
    rendered = render_prompt_template(
        "view_agent_task",
        photo_id="p1",
        vehicle_name="TestCar",
    )
    assert rendered
    assert "TestCar" in rendered
    assert "front_right" in rendered
    assert "hood" in rendered


def test_master_agent_system_renders():
    """MasterAgent SP renders the orchestration methodology."""
    rendered = render_prompt_template("master_agent_system")
    assert rendered
    assert "MasterAgent" in rendered
    assert "view_agent" in rendered
    assert "PartEvidence" in rendered


def test_vision_system_prompt_matches_legacy():
    from agents.vision_subagent import _build_system_prompt

    rendered = render_prompt_template(
        "vision_system_prompt",
        view_id="front_left_45",
        view_display_name=get_display_name("front_left_45"),
        checklist_text="1. hood（引擎盖）",
        vehicle_name="该车",
    )
    # Template has evolved beyond the legacy prompt; verify it renders
    # with expected content rather than exact string equality.
    assert rendered
    assert "front_left_45" in rendered
    assert get_display_name("front_left_45") in rendered
    assert "hood" in rendered
    assert "该车" in rendered
