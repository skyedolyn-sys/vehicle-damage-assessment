from typing import Dict

import pytest
from unittest.mock import AsyncMock, patch

from agents.planner_agent import (
    _classify_photo_by_filename,
    _classify_photo_types,
    _stabilize_plan,
    _group_photos_by_view,
    get_coverage_summary,
    get_photos_for_region,
    normalize_view_id,
    plan_to_location_map,
    planner_agent,
)
from agents.view_mapping import EXTERIOR_VIEWS, STANDARD_VIEWS


def test_normalize_view_id_variants():
    assert normalize_view_id("front_left_45") == "front_left_45"
    assert normalize_view_id("车头左前45度") == "front_left_45"
    assert normalize_view_id("右侧") == "right_90"
    assert normalize_view_id("车顶俯视") == "top"


def test_classify_photo_by_filename():
    assert _classify_photo_by_filename("行驶证.png") == "auxiliary"
    assert _classify_photo_by_filename("vin.png") == "auxiliary"
    assert _classify_photo_by_filename("车内座椅.png") == "interior"
    assert _classify_photo_by_filename("车头.png") == ""


def test_stabilize_plan_filters_non_exterior():
    photos = [
        {"id": "a.png", "path": "/a.png"},
        {"id": "b.png", "path": "/b.png"},
        {"id": "c.png", "path": "/c.png"},
    ]
    photo_views = [
        {"photo_id": "a.png", "view_id": "front", "confidence": "high", "reason": ""},
        {"photo_id": "b.png", "view_id": "interior", "confidence": "high", "reason": ""},
        {"photo_id": "c.png", "view_id": "unknown", "confidence": "low", "reason": ""},
    ]
    photo_types = {"a.png": "exterior", "b.png": "interior", "c.png": "unknown"}
    plan = _stabilize_plan(photo_views, photos, photo_types)

    assert len(plan["view_groups"]["front"]) == 1
    assert plan["view_groups"]["front"][0]["id"] == "a.png"
    assert len(plan["view_groups"]["interior"]) == 1
    assert plan["view_groups"]["interior"][0]["id"] == "b.png"
    assert len(plan["view_groups"]["unknown"]) == 1
    assert "right_90" in [g["missing_view"] for g in plan["coverage_gaps"]]
    assert plan["workflow_plan"]["priority_views"] == ["front"]


def test_stabilize_plan_sorts_and_deduplicates_by_confidence():
    photos = [
        {"id": "low.png", "path": "/low.png"},
        {"id": "high.png", "path": "/high.png"},
    ]
    photo_views = [
        {"photo_id": "low.png", "view_id": "front", "confidence": "low", "reason": ""},
        {"photo_id": "high.png", "view_id": "front", "confidence": "high", "reason": ""},
    ]
    photo_types = {"low.png": "exterior", "high.png": "exterior"}
    plan = _stabilize_plan(photo_views, photos, photo_types)

    assert len(plan["view_groups"]["front"]) == 1
    assert plan["view_groups"]["front"][0]["id"] == "high.png"


def test_group_photos_by_view():
    photos = [
        {"id": "a.png", "path": "/a.png"},
        {"id": "b.png", "path": "/b.png"},
    ]
    photo_views = [
        {"photo_id": "a.png", "view_id": "front", "confidence": "high", "reason": ""},
        {"photo_id": "b.png", "view_id": "left_90", "confidence": "high", "reason": ""},
    ]
    groups = _group_photos_by_view(photo_views, photos)
    assert len(groups["front"]) == 1
    assert len(groups["left_90"]) == 1
    assert groups["front"][0]["_planner_view"] == "front"


def test_plan_to_location_map():
    plan = {
        "photo_views": [
            {"photo_id": "a.png", "view_id": "front_left_45", "confidence": "high", "reason": ""},
            {"photo_id": "b.png", "view_id": "rear", "confidence": "high", "reason": ""},
        ]
    }
    location_map = plan_to_location_map(plan)
    assert location_map["a.png"]["location"] == "front"
    assert location_map["a.png"]["secondary_locations"] == ["left", "roof_front"]
    assert location_map["b.png"]["location"] == "rear"


def test_get_photos_for_region():
    plan = {
        "view_groups": {
            "front_left_45": [
                {"id": "a.png", "path": "/a.png", "_planner_view": "front_left_45"},
            ],
            "left_90": [
                {"id": "b.png", "path": "/b.png", "_planner_view": "left_90"},
            ],
            "rear": [
                {"id": "c.png", "path": "/c.png", "_planner_view": "rear"},
            ],
        }
    }
    left_photos = get_photos_for_region(plan, "left")
    assert len(left_photos) == 2
    assert {p["id"] for p in left_photos} == {"a.png", "b.png"}


def test_get_coverage_summary():
    plan = {
        "view_groups": {
            "front": [{"id": "a.png"}],
            "rear": [{"id": "b.png"}],
            "interior": [{"id": "c.png"}],
            "unknown": [],
        },
        "coverage_gaps": [
            {"missing_view": "left_90", "suggested_action": "补拍左侧"}
        ],
    }
    summary = get_coverage_summary(plan)
    assert summary["covered_views"] == ["front", "rear"]
    assert summary["exterior_photo_count"] == 2
    assert summary["ignored_photo_count"] == 1


@pytest.mark.asyncio
async def test_planner_agent_empty_photos():
    result = await planner_agent([], {})
    assert result["photo_views"] == []
    assert result["coverage_gaps"] == []
    for view in STANDARD_VIEWS:
        assert result["view_groups"][view] == []


@pytest.mark.asyncio
async def test_planner_agent_adds_missing_entries():
    photos = [{"id": "a.png", "path": "/a.png"}]
    fake_json = {
        "photo_views": [],
        "coverage_gaps": [],
        "workflow_plan": {},
    }
    with patch("agents.planner_agent._build_image_content", return_value={"type": "image_url", "image_url": {"url": "data:fake"}}):
        with patch("agents.planner_agent.call_minimax", new=AsyncMock(return_value="{}")):
            with patch("agents.planner_agent.extract_json", return_value=fake_json):
                result = await planner_agent(photos, {"vehicle": "test"})
    # DAMAGE_RECOGNITION_POLICY §1.1: 任何 photo 不得因 planner 无法识别视角就被丢弃。
    # 当 LLM 完全失败(返回空对象)时,planner 必须仍然为 photo 留一个 photo_views
    # 条目(view_id 可以是 'unknown'/'scene_intake',会路由到 intake subagent)。
    assert len(result["photo_views"]) == 1
    # augment 阶段会把 photo 分配到 exterior view(写进 view_groups),但
    # photo_views 里 LLM 给的 'unknown' 不会被覆盖。这是已知行为——
    # downstream 依赖 view_groups 而不是 photo_views.view_id。
    augmented_views = result.get("view_groups", {})
    has_exterior = any(
        v in EXTERIOR_VIEWS and len(augmented_views.get(v, [])) > 0
        for v in EXTERIOR_VIEWS
    )
    assert has_exterior, f"expected augment to fill at least one exterior view, got groups={list(augmented_views.keys())}"



class TestPlannerViewDerivation:
    """Planner should derive left_90/right_90 from corner views using standard ids."""

    def test_derives_left_90_from_corner_views(self):
        from agents.planner_agent import _deterministic_stabilize
        photos = [
            {"id": "fl.png", "path": "/fl.png"},
            {"id": "rl.png", "path": "/rl.png"},
        ]
        plan = _deterministic_stabilize(
            {
                "photo_views": [
                    {"photo_id": "fl.png", "view_id": "front_left_45", "confidence": "high", "reason": ""},
                    {"photo_id": "rl.png", "view_id": "rear_left_45", "confidence": "high", "reason": ""},
                ],
                "view_groups": {
                    "front_left_45": [{"id": "fl.png", "path": "/fl.png", "_planner_view": "front_left_45", "_planner_confidence": "high", "_planner_reason": ""}],
                    "rear_left_45": [{"id": "rl.png", "path": "/rl.png", "_planner_view": "rear_left_45", "_planner_confidence": "high", "_planner_reason": ""}],
                },
                "coverage_gaps": [],
                "workflow_plan": {"priority_views": ["front_left_45", "rear_left_45"], "missing_critical_views": []},
            },
            photos,
        )
        assert "left_90" in plan["view_groups"]
        assert len(plan["view_groups"]["left_90"]) == 1
        assert plan["view_groups"]["left_90"][0]["_planner_view"] == "left_90"

    def test_derives_right_90_from_corner_views(self):
        from agents.planner_agent import _deterministic_stabilize
        photos = [
            {"id": "fr.png", "path": "/fr.png"},
            {"id": "rr.png", "path": "/rr.png"},
        ]
        plan = _deterministic_stabilize(
            {
                "photo_views": [
                    {"photo_id": "fr.png", "view_id": "front_right_45", "confidence": "high", "reason": ""},
                    {"photo_id": "rr.png", "view_id": "rear_right_45", "confidence": "high", "reason": ""},
                ],
                "view_groups": {
                    "front_right_45": [{"id": "fr.png", "path": "/fr.png", "_planner_view": "front_right_45", "_planner_confidence": "high", "_planner_reason": ""}],
                    "rear_right_45": [{"id": "rr.png", "path": "/rr.png", "_planner_view": "rear_right_45", "_planner_confidence": "high", "_planner_reason": ""}],
                },
                "coverage_gaps": [],
                "workflow_plan": {"priority_views": ["front_right_45", "rear_right_45"], "missing_critical_views": []},
            },
            photos,
        )
        assert "right_90" in plan["view_groups"]
        assert plan["view_groups"]["right_90"][0]["_planner_view"] == "right_90"


@pytest.mark.asyncio
async def test_single_batch_skips_retry_when_response_too_short(monkeypatch):
    """Primary returns empty/very-short response → planner must not regress.

    Guard test: when the LLM returns a clearly broken response (rate-limited
    or disconnected), the planner should fall back to deterministic filename
    hints / safety nets rather than crashing or dropping photos. The safety
    nets in ``_ensure_exterior_coverage`` and the filename backfill must
    still produce a ``photo_views`` entry for every input photo.
    """
    import importlib
    pa = importlib.import_module("agents.planner_agent")

    call_count = {"n": 0}

    async def fake_call_minimax(messages, **kwargs):
        call_count["n"] += 1
        return ""  # empty → simulates disconnected/rate-limited

    monkeypatch.setattr(pa, "call_minimax", fake_call_minimax)
    # Avoid hitting the filesystem for image encoding.
    monkeypatch.setattr(
        pa, "_build_image_content",
        lambda photo, max_width=None: {"type": "image_url", "image_url": {"url": "data:fake"}},
    )

    photos = [
        {"id": f"front_{i}.png", "path": f"/tmp/p{i}.png"} for i in range(8)
    ]
    vehicle_prior = {"vehicle": "Test Car"}

    result = await pa.planner_agent(photos=photos, vehicle_prior=vehicle_prior)

    # Every photo got a view_id entry via filename backfill / safety nets —
    # the planner must never drop a photo even when the LLM is dead.
    photo_views = result.get("photo_views", [])
    assert len(photo_views) == 8, (
        f"expected 8 photo_views (one per input photo), got {len(photo_views)}"
    )
    for entry in photo_views:
        assert entry.get("view_id"), f"missing view_id: {entry}"
    # LLM was actually invoked (not bypassed entirely).
    assert call_count["n"] >= 1


@pytest.mark.asyncio
async def test_planner_returns_min_4_views_when_all_llm_calls_fail(monkeypatch):
    """With 32 photos and a fully-broken LLM, deterministic safety nets
    must still produce a plan the orchestrator can act on.

    This guards against a regression where the planner would return 0
    priority views or drop photo coverage when MiniMax is unavailable.
    """
    import importlib
    pa = importlib.import_module("agents.planner_agent")

    async def always_fail(messages, **kwargs):
        raise RuntimeError("simulated MiniMax outage")

    monkeypatch.setattr(pa, "call_minimax", always_fail)
    # Avoid hitting the filesystem for image encoding across every call.
    monkeypatch.setattr(
        pa, "_build_image_content",
        lambda photo, max_width=None: {"type": "image_url", "image_url": {"url": "data:fake"}},
    )

    photos = [
        {"id": f"172852-{i:02d}.png", "path": f"/tmp/p{i}.png"}
        for i in range(1, 33)
    ]
    vehicle_prior = {"vehicle": "Unknown"}

    plan = await pa.planner_agent(photos=photos, vehicle_prior=vehicle_prior)

    priority = plan.get("workflow_plan", {}).get("priority_views", [])
    assert len(priority) >= 4, (
        f"planner produced only {len(priority)} priority views: {priority}; "
        "deterministic safety nets must recover ≥ 4 views"
    )

    # Every photo got a view assignment (no orphans)
    photo_views = plan.get("photo_views", [])
    seen = {e["photo_id"] for e in photo_views}
    missing = {p["id"] for p in photos} - seen
    assert not missing, f"photos without view: {missing}"


class TestClassifyPhotoBySignals:
    """DAMAGE_RECOGNITION_POLICY §1.6: 确定性 photo_type 分类。"""

    def test_auxiliary_by_filename(self):
        from agents.planner_agent import _classify_photo_by_signals
        assert _classify_photo_by_signals({"id": "172852-行驶证.png"}) == "auxiliary"
        assert _classify_photo_by_signals({"id": "vin_plate.png"}) == "auxiliary"

    def test_interior_by_filename(self):
        from agents.planner_agent import _classify_photo_by_signals
        assert _classify_photo_by_signals({"id": "172852-内饰.png"}) == "interior"
        assert _classify_photo_by_signals({"id": "驾驶舱-左侧.png"}) == "interior"

    def test_close_up_by_portrait_aspect(self):
        """长宽比 < 0.7(竖图)→ close_up_damage"""
        from agents.planner_agent import _classify_photo_by_signals
        assert _classify_photo_by_signals({
            "id": "172852-15.png",
            "_decoded_width": 400,
            "_decoded_height": 800,  # ratio = 0.5
        }) == "close_up_damage"

    def test_close_up_by_landscape_aspect(self):
        """长宽比 > 1.4(横图特写)→ close_up_damage"""
        from agents.planner_agent import _classify_photo_by_signals
        assert _classify_photo_by_signals({
            "id": "172852-20.png",
            "_decoded_width": 1600,  # ratio = 2.0
            "_decoded_height": 800,
        }) == "close_up_damage"

    def test_default_exterior(self):
        """普通文件名 + 标准比例 → exterior"""
        from agents.planner_agent import _classify_photo_by_signals
        assert _classify_photo_by_signals({
            "id": "172852-15.png",
            "_decoded_width": 1024,
            "_decoded_height": 768,  # ratio = 1.33
        }) == "exterior"

    def test_no_llm_call(self):
        """DAMAGE_RECOGNITION_POLICY §1.6: _classify_photo_types 绝不应调 LLM。"""
        import asyncio
        from unittest.mock import patch

        with patch(
            "agents.planner_agent.call_minimax",
            side_effect=AssertionError("LLM was called - policy §1.6 violated"),
        ):
            asyncio.run(_classify_photo_types(
                [{"id": "172852-01.png", "_decoded_width": 1024, "_decoded_height": 768}],
                {"vehicle": "Test"},
            ))


class TestPreResolveViewsFromFilename:
    """DAMAGE_RECOGNITION_POLICY §1.6 / 步骤 2: filename hint 优先解析。"""

    def test_front_keyword(self):
        from agents.planner_agent import _pre_resolve_views_from_filename
        resolved, ambiguous = _pre_resolve_views_from_filename(
            [{"id": "172852-车头.png"}, {"id": "172852-前部.png"}]
        )
        assert len(resolved) == 2
        assert all(r["view_id"] == "front" for r in resolved)
        assert all(r["confidence"] == "high" for r in resolved)
        assert ambiguous == []

    def test_left_right_45(self):
        from agents.planner_agent import _pre_resolve_views_from_filename
        resolved, _ = _pre_resolve_views_from_filename(
            [{"id": "左前损伤.png"}, {"id": "右后碎裂.png"}]
        )
        views = {r["photo_id"]: r["view_id"] for r in resolved}
        assert views["左前损伤.png"] == "front_left_45"
        assert views["右后碎裂.png"] == "rear_right_45"

    def test_top_keyword(self):
        from agents.planner_agent import _pre_resolve_views_from_filename
        resolved, _ = _pre_resolve_views_from_filename([{"id": "172852-顶部俯视.png"}])
        assert resolved[0]["view_id"] == "top"

    def test_auxiliary_takes_priority(self):
        from agents.planner_agent import _pre_resolve_views_from_filename
        # 文件名含"行驶证"应被识别为 auxiliary(在 _FILENAME_VIEW_HINTS 列表靠前)
        resolved, _ = _pre_resolve_views_from_filename([{"id": "行驶证-01.png"}])
        assert resolved[0]["view_id"] == "auxiliary"

    def test_ambiguous_passes_through(self):
        from agents.planner_agent import _pre_resolve_views_from_filename
        resolved, ambiguous = _pre_resolve_views_from_filename(
            [{"id": "167111-01.png"}, {"id": "167111-02.png"}]
        )
        # 167111-* 是中性命名,不应该匹配任何 hint
        assert resolved == []
        assert len(ambiguous) == 2

    @pytest.mark.asyncio
    async def test_merges_into_final_plan(self):
        """End-to-end: filename 预解析应该出现在最终 photo_views 里。"""
        from unittest.mock import patch, AsyncMock
        from agents.planner_agent import planner_agent

        async def fake_call_minimax(messages, **kwargs):
            # LLM 返回空(模拟 26% parse 失败路径)→ filename 预解析必须保留
            return ""

        photos = [
            {"id": "172852-车头.png", "path": "/tmp/fake.png"},
            {"id": "172852-右侧.png", "path": "/tmp/fake.png"},
        ]
        with patch("agents.planner_agent._build_image_content", return_value={"type": "image_url", "image_url": {"url": "data:fake"}}):
            with patch("agents.planner_agent.call_minimax", new=AsyncMock(side_effect=fake_call_minimax)):
                with patch("agents.planner_agent.extract_json", return_value={}):
                    result = await planner_agent(photos, {"vehicle": "Test"})

        # 全部 photo 都该有 view_id(从 filename 来,即便 LLM 失败)
        photo_views = result.get("photo_views", [])
        assert len(photo_views) == 2, f"expected 2 photo_views, got {len(photo_views)}: {photo_views}"
        for entry in photo_views:
            assert entry["view_id"] in EXTERIOR_VIEWS or entry["view_id"] in {"front", "rear", "top"}, (
                f"photo {entry['photo_id']} view_id={entry['view_id']!r} not in expected set"
            )


class TestAugmentCoverageNoPhotoReuse:
    """DAMAGE_RECOGNITION_POLICY §1.6: planner must never assign the same
    photo to multiple view groups via deterministic augmentation.

    Observed failure mode (2026-06-22 sample): when the LLM planner
    returns nothing usable, the old _augment_exterior_coverage rotated
    the same single photo into 6 different view groups.  Every vision
    subagent then saw the same image, silently masking real damage
    visible only from other angles.  The fix: each photo may fill at
    most one target view; remaining target views stay empty and surface
    as coverage_gaps.
    """

    def test_does_not_reuse_photo_across_target_views(self):
        from agents.planner_agent import _augment_exterior_coverage

        photos = [
            {"id": f"172852-{i:02d}.png", "path": f"/tmp/{i}.png"}
            for i in range(1, 33)
        ]
        plan = {
            "photo_views": [],
            "view_groups": {"unknown": photos, "auxiliary": []},
            "coverage_gaps": [],
            "workflow_plan": {"priority_views": [], "missing_critical_views": []},
        }
        photo_types = {p["id"]: "exterior" for p in photos}

        out = _augment_exterior_coverage(plan, photos, photo_types)

        target_views = {
            "front_left_45", "front_right_45",
            "rear_left_45", "rear_right_45",
            "left_90", "right_90",
        }
        seen = {}
        for view_id, group in out["view_groups"].items():
            if view_id not in target_views:
                continue
            for p in group:
                pid = p.get("id")
                assert pid not in seen, (
                    f"photo {pid} appears in both {seen[pid]} and {view_id}"
                )
                seen[pid] = view_id
        assert len(seen) == 6, f"expected 6 unique photos, got {len(seen)}"

    def test_leaves_views_empty_when_pool_smaller_than_missing(self):
        """3 photos, 6 missing views → only 3 should be filled; rest stay
        empty and appear in coverage_gaps."""
        from agents.planner_agent import _augment_exterior_coverage

        photos = [
            {"id": f"172852-{i:02d}.png", "path": f"/tmp/{i}.png"}
            for i in range(1, 4)
        ]
        plan = {
            "photo_views": [],
            "view_groups": {"unknown": photos, "auxiliary": []},
            "coverage_gaps": [],
            "workflow_plan": {"priority_views": [], "missing_critical_views": []},
        }
        photo_types = {p["id"]: "exterior" for p in photos}

        out = _augment_exterior_coverage(plan, photos, photo_types)

        target_views = (
            "front_left_45", "front_right_45", "rear_left_45",
            "rear_right_45", "left_90", "right_90",
        )
        filled = sum(1 for v in target_views if out["view_groups"].get(v))
        assert filled == 3, f"expected 3 filled views, got {filled}"

    def test_keeps_priority_views_at_seven(self):
        """The 7-view priority list (6 sides + top) must remain so the
        orchestrator can dispatch vision subagents — never collapse to 1 (the
        pre-fix bug).
        """
        from agents.planner_agent import _augment_exterior_coverage

        photos = [
            {"id": f"172852-{i:02d}.png", "path": f"/tmp/{i}.png"}
            for i in range(1, 33)
        ]
        plan = {
            "photo_views": [],
            "view_groups": {"unknown": photos, "auxiliary": []},
            "coverage_gaps": [],
            "workflow_plan": {"priority_views": [], "missing_critical_views": []},
        }
        photo_types = {p["id"]: "exterior" for p in photos}

        out = _augment_exterior_coverage(plan, photos, photo_types)
        priority = out["workflow_plan"]["priority_views"]
        assert len(priority) == 7, (
            f"expected 7 priority views, got {len(priority)}: {priority}"
        )


@pytest.mark.asyncio
async def test_2026_06_22_sample_simulation():
    """End-to-end: reproduce the bug pattern from the 172852 sample.

    32 photos, LLM returns empty. The planner must:
    1. Give every photo a view_id entry (no silent drops).
    2. NOT reuse the same photo across 6 different view groups.
    3. Surface front/rear as coverage gaps.
    """
    from agents.planner_agent import planner_agent

    photos = [
        {"id": f"172852-{i:02d}.png", "path": f"/tmp/{i}.png"}
        for i in range(1, 33)
    ]

    async def fake_call_minimax(messages, **kwargs):
        return ""

    with patch(
        "agents.planner_agent._build_image_content",
        return_value={"type": "image_url", "image_url": {"url": "data:fake"}},
    ):
        with patch(
            "agents.planner_agent.call_minimax",
            new=AsyncMock(side_effect=fake_call_minimax),
        ):
            result = await planner_agent(photos, {"vehicle": "Unknown"})

    view_groups = result.get("view_groups", {})
    target_views = (
        "front_left_45", "front_right_45", "rear_left_45",
        "rear_right_45", "left_90", "right_90", "top",
    )
    seen = {}
    for v in target_views:
        for p in view_groups.get(v, []):
            pid = p.get("id")
            assert pid not in seen, (
                f"BUG: photo {pid} appears in both {seen[pid]} and {v}"
            )
            seen[pid] = v
    priority = result.get("workflow_plan", {}).get("priority_views", [])
    assert len(priority) == 7
    coverage_gaps = [g["missing_view"] for g in result.get("coverage_gaps", [])]
    assert "front" in coverage_gaps
    assert "rear" in coverage_gaps


class TestStabilizePlanPreservesCloseUpExteriorViews:
    """close_up_damage / close_up_detail photos with a usable exterior view_id
    must stay in that exterior group, not be rerouted to scene_intake.
    """

    def test_close_up_damage_keeps_exterior_view_id(self):
        photos = [
            {"id": "172852-15.png", "path": "/tmp/15.png"},
        ]
        photo_views = [
            {"photo_id": "172852-15.png", "view_id": "front_left_45", "confidence": "medium", "reason": "llm"},
        ]
        photo_types = {"172852-15.png": "close_up_damage"}
        plan = _stabilize_plan(photo_views, photos, photo_types)

        assert len(plan["view_groups"]["front_left_45"]) == 1
        assert plan["view_groups"]["front_left_45"][0]["id"] == "172852-15.png"
        assert plan["view_groups"]["front_left_45"][0]["_planner_photo_type"] == "close_up_damage"

    def test_close_up_detail_keeps_exterior_view_id(self):
        photos = [
            {"id": "172852-20.png", "path": "/tmp/20.png"},
        ]
        photo_views = [
            {"photo_id": "172852-20.png", "view_id": "right_90", "confidence": "high", "reason": "llm"},
        ]
        photo_types = {"172852-20.png": "close_up_detail"}
        plan = _stabilize_plan(photo_views, photos, photo_types)

        assert len(plan["view_groups"]["right_90"]) == 1
        assert plan["view_groups"]["right_90"][0]["id"] == "172852-20.png"

    def test_close_up_without_exterior_view_goes_to_scene_intake(self):
        photos = [
            {"id": "172852-21.png", "path": "/tmp/21.png"},
        ]
        photo_views = [
            {"photo_id": "172852-21.png", "view_id": "unknown", "confidence": "low", "reason": "llm"},
        ]
        photo_types = {"172852-21.png": "close_up_damage"}
        plan = _stabilize_plan(photo_views, photos, photo_types)

        assert len(plan["view_groups"]["scene_intake"]) == 1
        assert plan["view_groups"]["scene_intake"][0]["id"] == "172852-21.png"


class TestDeterministicStabilizeRetainsMultiplePhotos:
    """_deterministic_stabilize should keep supplementary evidence for key views."""

    def test_retains_up_to_three_for_corner_and_top_views(self):
        from agents.planner_agent import _deterministic_stabilize

        photos = [{"id": f"fl{i:02d}.png", "path": f"/tmp/fl{i:02d}.png"} for i in range(1, 5)]
        plan = {
            "photo_views": [],
            "view_groups": {
                "front_left_45": [
                    {
                        "id": f"fl{i:02d}.png",
                        "path": f"/tmp/fl{i:02d}.png",
                        "_planner_view": "front_left_45",
                        "_planner_confidence": "high" if i == 1 else "medium",
                        "_planner_reason": "",
                        "_planner_photo_type": "exterior",
                    }
                    for i in range(1, 5)
                ],
            },
            "coverage_gaps": [],
            "workflow_plan": {"priority_views": ["front_left_45"], "missing_critical_views": []},
        }
        out = _deterministic_stabilize(plan, photos)
        assert len(out["view_groups"]["front_left_45"]) == 3
        retained_ids = {p["id"] for p in out["view_groups"]["front_left_45"]}
        # Highest-confidence first three by filename order should be retained.
        assert "fl01.png" in retained_ids
        for p in out["view_groups"]["front_left_45"]:
            assert p["_planner_view"] == "front_left_45"
            assert p["_planner_photo_type"] == "exterior"

    def test_retains_up_to_two_for_side_views(self):
        from agents.planner_agent import _deterministic_stabilize

        photos = [{"id": f"left{i:02d}.png", "path": f"/tmp/left{i:02d}.png"} for i in range(1, 4)]
        plan = {
            "photo_views": [],
            "view_groups": {
                "left_90": [
                    {
                        "id": f"left{i:02d}.png",
                        "path": f"/tmp/left{i:02d}.png",
                        "_planner_view": "left_90",
                        "_planner_confidence": "high",
                        "_planner_reason": "",
                        "_planner_photo_type": "exterior",
                    }
                    for i in range(1, 4)
                ],
            },
            "coverage_gaps": [],
            "workflow_plan": {"priority_views": ["left_90"], "missing_critical_views": []},
        }
        out = _deterministic_stabilize(plan, photos)
        assert len(out["view_groups"]["left_90"]) == 2

    def test_retains_one_for_other_exterior_views(self):
        from agents.planner_agent import _deterministic_stabilize

        photos = [{"id": f"front{i:02d}.png", "path": f"/tmp/front{i:02d}.png"} for i in range(1, 4)]
        plan = {
            "photo_views": [],
            "view_groups": {
                "front": [
                    {
                        "id": f"front{i:02d}.png",
                        "path": f"/tmp/front{i:02d}.png",
                        "_planner_view": "front",
                        "_planner_confidence": "high",
                        "_planner_reason": "",
                        "_planner_photo_type": "exterior",
                    }
                    for i in range(1, 4)
                ],
            },
            "coverage_gaps": [],
            "workflow_plan": {"priority_views": ["front"], "missing_critical_views": []},
        }
        out = _deterministic_stabilize(plan, photos)
        assert len(out["view_groups"]["front"]) == 1


class TestAugmentCoverage172852:
    """172852-style numbered datasets must fill front_right_45, right_90 and top
    deterministically across repeated runs.
    """

    def test_augment_fills_front_right_right_90_and_top_across_three_runs(self):
        from agents.planner_agent import _augment_exterior_coverage

        # 32 numbered photos, no useful LLM labels, mix of exterior/close-up aspect ratios.
        photos = [
            {"id": f"172852-{i:02d}.png", "path": f"/tmp/{i}.png"}
            for i in range(1, 33)
        ]
        plan = {
            "photo_views": [],
            "view_groups": {"unknown": photos, "auxiliary": [], "interior": []},
            "coverage_gaps": [],
            "workflow_plan": {"priority_views": [], "missing_critical_views": []},
        }
        # Simulate deterministic classification: portrait/landscape -> close_up_damage,
        # square-ish -> exterior. This is the realistic mix that caused the original bug.
        photo_types: Dict[str, str] = {}
        for i, p in enumerate(photos, start=1):
            if i % 5 in (1, 4):  # portrait / landscape-ish
                photo_types[p["id"]] = "close_up_damage"
            else:
                photo_types[p["id"]] = "exterior"

        for _ in range(3):
            out = _augment_exterior_coverage(plan, photos, photo_types)
            groups = out["view_groups"]
            assert groups["front_right_45"], "front_right_45 should be filled"
            assert groups["right_90"], "right_90 should be filled"
            assert groups["top"], "top should be filled"

    def test_close_up_photos_eligible_for_augment_rotation(self):
        from agents.planner_agent import _augment_exterior_coverage

        photos = [
            {"id": f"172852-{i:02d}.png", "path": f"/tmp/{i}.png"}
            for i in range(1, 8)
        ]
        plan = {
            "photo_views": [],
            "view_groups": {"unknown": photos, "auxiliary": []},
            "coverage_gaps": [],
            "workflow_plan": {"priority_views": [], "missing_critical_views": []},
        }
        # All close_up_damage: previously they were excluded from rotation.
        photo_types = {p["id"]: "close_up_damage" for p in photos}

        out = _augment_exterior_coverage(plan, photos, photo_types)
        filled = sum(
            1 for v in ("front_right_45", "right_90", "top")
            if out["view_groups"].get(v)
        )
        assert filled >= 2, f"expected close-up photos to fill missing views, got filled={filled}"
