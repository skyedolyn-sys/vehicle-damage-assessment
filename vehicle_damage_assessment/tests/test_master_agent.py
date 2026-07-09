import pytest
from unittest.mock import AsyncMock, patch

from agents.master_agent import (
    _extract_photo_classifications,
    _aggregate_part_evidence,
    _boost_confidence,
    _build_region_results,
    _apply_sunroof_roof_propagation,
    master_assessment_agent,
)
from models.part_state import PartActualState, Status, DamageLevel
from models.assessment import DamageAssessment


def test_extract_photo_classifications_new_schema():
    plan = {
        "photo_classifications": [
            {"photo_id": "a", "category": "exterior"},
            {"photo_id": "b", "category": "interior"},
        ]
    }
    result = _extract_photo_classifications(plan)
    assert result == {"a": "exterior", "b": "interior"}


def test_extract_photo_classifications_legacy_view_groups():
    plan = {
        "view_groups": {
            "front": [{"id": "a"}],
            "interior": [{"id": "b"}],
            "auxiliary": [{"id": "c"}],
        }
    }
    result = _extract_photo_classifications(plan)
    assert result["a"] == "exterior"
    assert result["b"] == "interior"
    assert result["c"] == "vehicle_info"


def test_aggregate_part_evidence_consensus_boosts_confidence():
    view_results = [
        {
            "photo_id": "p1",
            "primary_view": "front",
            "parts": [
                {
                    "part_id": "hood",
                    "status": "damaged",
                    "damage_level": "moderate",
                    "damage_types": ["dent"],
                    "confidence": "medium",
                    "description": "凹陷",
                },
            ],
        },
        {
            "photo_id": "p2",
            "primary_view": "front_left",
            "parts": [
                {
                    "part_id": "hood",
                    "status": "damaged",
                    "damage_level": "moderate",
                    "damage_types": ["dent"],
                    "confidence": "medium",
                    "description": "凹陷",
                },
            ],
        },
        {
            "photo_id": "p3",
            "primary_view": "front_right",
            "parts": [
                {
                    "part_id": "hood",
                    "status": "damaged",
                    "damage_level": "moderate",
                    "damage_types": ["dent"],
                    "confidence": "low",
                    "description": "凹陷",
                },
            ],
        },
    ]
    evidence = _aggregate_part_evidence(view_results)
    hood = evidence["hood"]
    assert hood["aggregated_status"] == "damaged"
    assert hood["aggregated_confidence"] == "high"
    assert hood["conflicting"] is False


def test_aggregate_part_evidence_conflict_detected():
    view_results = [
        {
            "photo_id": "p1",
            "primary_view": "front_left",
            "parts": [
                {"part_id": "hood", "status": "intact", "damage_level": "none", "damage_types": ["none"], "confidence": "high", "description": "平整"},
            ],
        },
        {
            "photo_id": "p2",
            "primary_view": "rear",
            "parts": [
                {"part_id": "hood", "status": "damaged", "damage_level": "moderate", "damage_types": ["dent"], "confidence": "high", "description": "凹陷"},
            ],
        },
    ]
    evidence = _aggregate_part_evidence(view_results)
    # Neither primary strong intact nor primary strong damaged; consensus
    # is uncertain and conflicting is True.
    assert evidence["hood"]["conflicting"] is True
    assert evidence["hood"]["aggregated_status"] == "uncertain"


def test_boost_confidence_schema():
    assert _boost_confidence("low", 3) == "high"
    assert _boost_confidence("medium", 2) == "medium"
    assert _boost_confidence("low", 1) == "low"


def test_build_region_results_groups_by_primary_view():
    view_results = [
        {
            "photo_id": "p1",
            "primary_view": "front",
            "parts": [
                {"part_id": "hood", "part_name": "引擎盖", "status": "intact", "damage_level": "none", "damage_types": ["none"], "confidence": "high", "description": "平整"},
            ],
        },
    ]
    evidence = _aggregate_part_evidence(view_results)
    region_results = _build_region_results(view_results, evidence)
    assert len(region_results) == 1
    assert region_results[0]["region"] == "front"
    assert region_results[0]["parts"][0]["part_id"] == "hood"


@pytest.mark.asyncio
async def test_master_agent_dispatches_view_agent_per_exterior_photo():
    fake_prior = {
        "vehicle": "Test Car",
        "vehicle_specs": {},
        "topology": {},
        "key_anchors": {},
    }
    fake_plan = {
        "photo_classifications": [
            {"photo_id": "a", "category": "exterior"},
            {"photo_id": "b", "category": "exterior"},
            {"photo_id": "c", "category": "vehicle_info"},
        ],
    }

    async def fake_view_agent(photo, vehicle_prior):
        return {
            "photo_id": photo["id"],
            "primary_view": "front",
            "view_detections": [],
            "parts": [
                {"part_id": "hood", "part_name": "引擎盖", "status": "intact", "damage_level": "none", "damage_types": ["none"], "confidence": "high", "description": "", "evidence_photo": photo["id"]},
            ],
        }

    with patch("agents.master_agent.vehicle_prior_agent", new=AsyncMock(return_value=fake_prior)):
        with patch("agents.master_agent.build_vehicle_topology") as mock_topology:
            from models.topology import TopologyNode, VehicleTopology
            topology = VehicleTopology(
                vehicle_id="test",
                vehicle_name="Test",
                nodes={
                    "hood": TopologyNode(
                        node_id="hood",
                        part_id="hood",
                        node_name="引擎盖",
                        node_type="panel",
                        region="front",
                        side="center",
                        visibility_from=["front"],
                    ),
                },
                regions={"front": ["hood"]},
            )
            mock_topology.return_value = topology
            with patch("agents.master_agent.planner_agent", new=AsyncMock(return_value=fake_plan)):
                with patch("agents.master_agent.view_agent", new=AsyncMock(side_effect=fake_view_agent)):
                    with patch("agents.master_agent.reviewer_subagent", new=AsyncMock(return_value={
                        "reviewed_parts": [],
                        "reviewed_part_actual_states": [],
                    })):
                        result = await master_assessment_agent(
                            [{"id": "a", "path": "/a.png"}, {"id": "b", "path": "/b.png"}, {"id": "c", "path": "/c.png"}],
                            {"vehicle_name": "Test"},
                        )

    assert result is not None
    # Two exterior photos should produce two view_agent calls.
    view_calls = [c for c in result._plan.get("photo_classifications", []) if c["category"] == "exterior"]
    assert len(view_calls) == 2


# --- _apply_sunroof_roof_propagation tests --------------------------------


def _make_state(part_id: str, status: Status, level: DamageLevel = DamageLevel.UNKNOWN) -> PartActualState:
    """Helper to build a minimal PartActualState for the sunroof helper tests."""
    return PartActualState(
        part_id=part_id,
        part_name=part_id,
        part_category="roof" if "roof" in part_id else "other",
        side="center",
        status=status,
        damage_level=level,
    )


def _make_assessment(parts: List[PartActualState]) -> DamageAssessment:
    """Build a DamageAssessment with classified lists auto-populated."""
    assessment = DamageAssessment(parts=parts)
    assessment.damaged_parts = [p.part_id for p in parts if p.status == Status.DAMAGED]
    assessment.intact_parts = [p.part_id for p in parts if p.status == Status.INTACT]
    assessment.uncertain_parts = [p.part_id for p in parts if p.status == Status.UNCERTAIN]
    assessment.missing_parts = [p.part_id for p in parts if p.status == Status.MISSING]
    return assessment


def test_sunroof_promoted_when_roof_metal_damaged():
    """sunroof_glass should flip UNCERTAIN → DAMAGED SEVERE when ≥1 of
    roof_front / roof_middle / roof_rear is DAMAGED (post-topology Rule 10/11)."""
    parts = [
        _make_state("roof_front", Status.DAMAGED, DamageLevel.SEVERE),
        _make_state("sunroof_glass", Status.UNCERTAIN),
    ]
    assessment = _make_assessment(parts)
    assert "sunroof_glass" in assessment.uncertain_parts
    assert "sunroof_glass" not in assessment.damaged_parts

    _apply_sunroof_roof_propagation(assessment)

    sunroof = next(p for p in assessment.parts if p.part_id == "sunroof_glass")
    assert sunroof.status == Status.DAMAGED
    assert sunroof.damage_level == DamageLevel.SEVERE
    assert "deformation" in sunroof.damage_types
    # Classified lists must reflect the promotion.
    assert "sunroof_glass" in assessment.damaged_parts
    assert "sunroof_glass" not in assessment.uncertain_parts


def test_sunroof_intact_when_no_roof_metal_damaged():
    """When no roof_metal is DAMAGED, sunroof_glass must NOT be promoted
    even if it's currently UNCERTAIN (preserves the synthesizer's
    'no primary → uncertain' guard)."""
    parts = [
        _make_state("roof_front", Status.INTACT),
        _make_state("roof_middle", Status.UNCERTAIN),  # not enough
        _make_state("sunroof_glass", Status.UNCERTAIN),
    ]
    assessment = _make_assessment(parts)
    _apply_sunroof_roof_propagation(assessment)
    sunroof = next(p for p in assessment.parts if p.part_id == "sunroof_glass")
    assert sunroof.status == Status.UNCERTAIN
    assert "sunroof_glass" in assessment.uncertain_parts
    assert "sunroof_glass" not in assessment.damaged_parts


def test_sunroof_already_damaged_not_overwritten():
    """When sunroof_glass is already DAMAGED from a primary top-view
    source, the helper must NOT touch it (would overwrite the higher-quality
    evidence-based status)."""
    parts = [
        _make_state("roof_front", Status.DAMAGED, DamageLevel.SEVERE),
        PartActualState(
            part_id="sunroof_glass",
            part_name="天窗玻璃",
            part_category="roof",
            side="center",
            status=Status.DAMAGED,
            damage_level=DamageLevel.MODERATE,
            damage_types=["crack"],
            confidence="high",
            notes="top-down observation",
        ),
    ]
    assessment = _make_assessment(parts)
    _apply_sunroof_roof_propagation(assessment)
    sunroof = next(p for p in assessment.parts if p.part_id == "sunroof_glass")
    # Original primary-view status preserved — no double-write.
    assert sunroof.status == Status.DAMAGED
    assert sunroof.damage_level == DamageLevel.MODERATE
    assert sunroof.damage_types == ["crack"]
    assert "top-down observation" in sunroof.notes
    assert "推断天窗" not in sunroof.notes


def test_sunroof_missing_with_damaged_evidence_promoted_to_damaged():
    """When sunroof_glass status=MISSING but the view_agent observed
    breakage/crack damage_types, the helper should promote to DAMAGED SEVERE
    (matches the _apply_rear_missing_to_damaged_fallback pattern).

    Real-world case: top-view shows shattered glass, the synthesizer picks
    'missing' as terminal status, but the sunroof was clearly damaged
    (not just absent because the car never had one).
    """
    parts = [
        _make_state("roof_front", Status.DAMAGED, DamageLevel.SEVERE),
        _make_state("roof_middle", Status.DAMAGED, DamageLevel.SEVERE),
        PartActualState(
            part_id="sunroof_glass",
            part_name="天窗玻璃",
            part_category="roof",
            side="center",
            status=Status.MISSING,
            damage_level=DamageLevel.NONE,
            damage_types=["breakage", "crack", "none"],
            confidence="low",
            notes="top view shows fragmented glass",
        ),
    ]
    assessment = _make_assessment(parts)
    assert "sunroof_glass" in assessment.missing_parts
    assert "sunroof_glass" not in assessment.damaged_parts

    _apply_sunroof_roof_propagation(assessment)

    sunroof = next(p for p in assessment.parts if p.part_id == "sunroof_glass")
    assert sunroof.status == Status.DAMAGED
    assert sunroof.damage_level == DamageLevel.SEVERE
    # damage_types preserved from view_agent — breakage / crack kept.
    assert "breakage" in sunroof.damage_types
    assert "crack" in sunroof.damage_types
    # Classified lists must reflect the promotion.
    assert "sunroof_glass" in assessment.damaged_parts
    assert "sunroof_glass" not in assessment.missing_parts
    assert "sunroof_glass" not in assessment.uncertain_parts
    # Notes should mention the missing→damaged rollback, not pure inference.
    assert "回退为 damaged severe" in sunroof.notes or "推断天窗" in sunroof.notes


def test_sunroof_missing_no_damaged_evidence_not_promoted():
    """When sunroof_glass status=MISSING and the synthesizer left no
    damage_types (truly absent, never had a sunroof), the helper must NOT
    promote (no evidence of damage)."""
    parts = [
        _make_state("roof_front", Status.DAMAGED, DamageLevel.SEVERE),
        PartActualState(
            part_id="sunroof_glass",
            part_name="天窗玻璃",
            part_category="roof",
            side="center",
            status=Status.MISSING,
            damage_level=DamageLevel.NONE,
            damage_types=["none"],
            confidence="low",
            notes="",
        ),
    ]
    assessment = _make_assessment(parts)
    _apply_sunroof_roof_propagation(assessment)
    sunroof = next(p for p in assessment.parts if p.part_id == "sunroof_glass")
    # No damaged evidence → don't override the synthesizer's missing verdict.
    assert sunroof.status == Status.MISSING
    assert "sunroof_glass" in assessment.missing_parts


def test_sunroof_intact_never_promoted():
    """When sunroof_glass status=INTACT (clear primary observation), the
    helper must NEVER promote — would override a higher-confidence
    observation."""
    parts = [
        _make_state("roof_front", Status.DAMAGED, DamageLevel.SEVERE),
        PartActualState(
            part_id="sunroof_glass",
            part_name="天窗玻璃",
            part_category="roof",
            side="center",
            status=Status.INTACT,
            damage_level=DamageLevel.NONE,
            confidence="high",
            notes="intact in primary view",
        ),
    ]
    assessment = _make_assessment(parts)
    _apply_sunroof_roof_propagation(assessment)
    sunroof = next(p for p in assessment.parts if p.part_id == "sunroof_glass")
    assert sunroof.status == Status.INTACT
    assert "sunroof_glass" in assessment.intact_parts
    assert "推断天窗" not in sunroof.notes
