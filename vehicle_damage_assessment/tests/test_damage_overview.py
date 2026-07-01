"""Tests for damage_overview generator."""

import pytest

from models.part_state import DamageLevel, PartActualState, Status
from agents.topology_builder import build_vehicle_topology
from agents.topology_comparator import compare_topology
from agents.damage_overview import generate_damage_overview


@pytest.fixture
def sample_topology():
    info = {"vehicle_id": "v-overview-001", "vehicle_name": "Overview Test"}
    prior = {
        "topology": {
            "front": "前部",
            "rear": "后部",
            "left": "左侧",
            "right": "右侧",
            "roof": "车顶",
        },
        "key_anchors": {
            "front": [],
            "rear": [],
            "left": [],
            "right": [],
            "roof": [],
        },
    }
    return build_vehicle_topology(info, prior)


class TestGenerateDamageOverview:
    """Unit tests for the damage overview generator."""

    def test_overall_structure(self, sample_topology):
        """The overview contains all expected top-level keys."""
        states = [
            PartActualState(
                part_id="hood",
                part_name="引擎盖",
                region="front",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.MODERATE,
                confidence="high",
                evidence_photos=["p1.jpg"],
            ),
        ]
        result = compare_topology(sample_topology, states)
        overview = generate_damage_overview(
            result.parts,
            result.structural_damage_flag,
            result.overall_severity,
            result.primary_damage_zone,
        )

        assert "overview_text" in overview
        assert "accident_type" in overview
        assert "overall_assessment" in overview
        assert "region_summary" in overview
        assert "safety_summary" in overview
        assert "structural_summary" in overview
        assert "symmetric_damage" in overview
        assert "coverage_assessment" in overview
        assert "repair_attention" in overview
        assert "human_review_notes" in overview

    def test_front_collision_inference(self, sample_topology):
        """Damage limited to the front region is inferred as a front collision."""
        states = [
            PartActualState(
                part_id="hood",
                part_name="引擎盖",
                region="front",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.MODERATE,
                confidence="high",
                evidence_photos=["p1.jpg"],
            ),
        ]
        result = compare_topology(sample_topology, states)
        overview = generate_damage_overview(
            result.parts,
            result.structural_damage_flag,
            result.overall_severity,
            result.primary_damage_zone,
        )

        assert overview["accident_type"]["type_id"] == "front_collision"
        assert "前部碰撞" in overview["overview_text"]

    def test_rear_collision_inference(self, sample_topology):
        """Damage limited to the rear region is inferred as a rear collision."""
        states = [
            PartActualState(
                part_id="bumper_rear",
                part_name="后保险杠",
                region="rear",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.MODERATE,
                confidence="high",
                evidence_photos=["p1.jpg"],
            ),
        ]
        result = compare_topology(sample_topology, states)
        overview = generate_damage_overview(
            result.parts,
            result.structural_damage_flag,
            result.overall_severity,
            result.primary_damage_zone,
        )

        assert overview["accident_type"]["type_id"] == "rear_collision"
        assert "后部碰撞" in overview["overview_text"]

    def test_symmetric_damage_detected(self, sample_topology):
        """Symmetric left/right damage is detected and reported."""
        states = [
            PartActualState(
                part_id="headlight_front_left",
                part_name="左前大灯",
                region="front",
                side="front_left",
                status=Status.DAMAGED,
                damage_level=DamageLevel.MODERATE,
                confidence="high",
                evidence_photos=["p1.jpg"],
            ),
            PartActualState(
                part_id="headlight_front_right",
                part_name="右前大灯",
                region="front",
                side="front_right",
                status=Status.DAMAGED,
                damage_level=DamageLevel.MODERATE,
                confidence="high",
                evidence_photos=["p2.jpg"],
            ),
        ]
        result = compare_topology(sample_topology, states)
        overview = generate_damage_overview(
            result.parts,
            result.structural_damage_flag,
            result.overall_severity,
            result.primary_damage_zone,
        )

        assert overview["symmetric_damage"]["symmetric_damage_detected"] is True
        assert any(p["group_name"] == "前大灯" for p in overview["symmetric_damage"]["pairs"])
        assert "左右对称部位" in overview["overview_text"]

    def test_safety_critical_summary(self, sample_topology):
        """Safety-critical part damage appears in the safety summary."""
        states = [
            PartActualState(
                part_id="headlight_front_left",
                part_name="左前大灯",
                region="front",
                side="front_left",
                status=Status.MISSING,
                damage_level=DamageLevel.SEVERE,
                confidence="high",
                evidence_photos=["p1.jpg"],
            ),
        ]
        result = compare_topology(sample_topology, states)
        overview = generate_damage_overview(
            result.parts,
            result.structural_damage_flag,
            result.overall_severity,
            result.primary_damage_zone,
        )

        assert overview["safety_summary"]["affected_count"] == 1
        assert overview["safety_summary"]["affected_parts"][0]["part_id"] == "headlight_front_left"
        assert "左前大灯" in overview["overview_text"]

    def test_repair_attention_high_for_structural(self, sample_topology):
        """Structural damage raises repair attention to high."""
        states = [
            PartActualState(
                part_id="roof_front",
                part_name="车顶前部",
                region="roof",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.SEVERE,
                confidence="high",
                evidence_photos=["p1.jpg"],
            ),
        ]
        result = compare_topology(sample_topology, states)
        overview = generate_damage_overview(
            result.parts,
            result.structural_damage_flag,
            result.overall_severity,
            result.primary_damage_zone,
        )

        assert overview["repair_attention"]["level"] == "high"
        assert "重点检修" in overview["repair_attention"]["level_name"]

    def test_coverage_assessment(self, sample_topology):
        """Coverage assessment reflects the ratio of certain vs uncertain parts."""
        states = [
            PartActualState(
                part_id="hood",
                part_name="引擎盖",
                region="front",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.LIGHT,
                confidence="high",
                evidence_photos=["p1.jpg"],
            ),
        ]
        result = compare_topology(sample_topology, states)
        overview = generate_damage_overview(
            result.parts,
            result.structural_damage_flag,
            result.overall_severity,
            result.primary_damage_zone,
        )

        coverage = overview["coverage_assessment"]
        assert coverage["total_parts"] == len(sample_topology.nodes)
        assert coverage["uncertain_count"] > 0
        assert 0.0 < coverage["coverage_ratio"] < 1.0

    def test_no_damage_overall(self, sample_topology):
        """When all parts are intact the overview reflects no damage."""
        states = [
            PartActualState(
                part_id="hood",
                part_name="引擎盖",
                region="front",
                side="center",
                status=Status.INTACT,
                damage_level=DamageLevel.NONE,
                confidence="high",
                evidence_photos=["p1.jpg"],
            ),
        ]
        result = compare_topology(sample_topology, states)
        overview = generate_damage_overview(
            result.parts,
            result.structural_damage_flag,
            result.overall_severity,
            result.primary_damage_zone,
        )

        assert overview["accident_type"]["type_id"] == "none"
        assert "无明显碰撞" in overview["overview_text"]
        assert overview["repair_attention"]["level"] == "none"


class TestPhotoTypeEvidenceNote:
    """generate_damage_overview should flag close-up-skewed photo sets."""

    def _build_parts(self):
        return [
            PartActualState(
                part_id="hood",
                part_name="引擎盖",
                region="front",
                side="center",
                status=Status.DAMAGED,
                damage_level=DamageLevel.SEVERE,
                damage_types=["deformation"],
                confidence="low",
                evidence_photos=["a.png"],
                notes="",
            ),
        ]

    def test_close_up_skew_triggers_warning(self):
        """When 80% of photos are close-up and only 1 is wide-shot, warning fires."""
        summary = {"wide_shot": 1, "close_up_damage": 8, "close_up_detail": 0, "unknown": 1}
        result = generate_damage_overview(
            self._build_parts(),
            structural_damage_flag=True,
            overall_severity="severe",
            primary_damage_zone="front",
            photo_type_summary=summary,
        )
        notes = result["human_review_notes"]
        assert any("近距离特写" in n for n in notes), f"expected close-up warning, got {notes}"

    def test_close_up_warning_is_first_note(self):
        """The close-up warning should appear at the top of the note list."""
        summary = {"wide_shot": 1, "close_up_damage": 8, "close_up_detail": 0, "unknown": 1}
        result = generate_damage_overview(
            self._build_parts(),
            structural_damage_flag=True,
            overall_severity="severe",
            primary_damage_zone="front",
            photo_type_summary=summary,
        )
        notes = result["human_review_notes"]
        assert notes, "expected at least one note"
        assert "近距离特写" in notes[0], f"expected warning first, got {notes}"

    def test_balanced_photo_set_no_warning(self):
        """When ≥2 wide_shot photos, no close-up warning fires."""
        summary = {"wide_shot": 5, "close_up_damage": 3, "close_up_detail": 0, "unknown": 0}
        result = generate_damage_overview(
            self._build_parts(),
            structural_damage_flag=False,
            overall_severity="moderate",
            primary_damage_zone="front",
            photo_type_summary=summary,
        )
        notes = result["human_review_notes"]
        assert not any("近距离特写" in n for n in notes), f"unexpected warning, got {notes}"

    def test_below_seventy_percent_close_up_no_warning(self):
        """When close-ups are <70%, no warning fires even with few wide-shots."""
        # 6 close-up out of 10 = 60%, below the 70% threshold.
        summary = {"wide_shot": 1, "close_up_damage": 4, "close_up_detail": 2, "unknown": 3}
        result = generate_damage_overview(
            self._build_parts(),
            structural_damage_flag=False,
            overall_severity="moderate",
            primary_damage_zone="front",
            photo_type_summary=summary,
        )
        notes = result["human_review_notes"]
        assert not any("近距离特写" in n for n in notes), f"unexpected warning, got {notes}"

    def test_no_photo_type_summary_no_warning(self):
        """When photo_type_summary is None, no warning fires (backward compatible)."""
        result = generate_damage_overview(
            self._build_parts(),
            structural_damage_flag=False,
            overall_severity="moderate",
            primary_damage_zone="front",
        )
        notes = result["human_review_notes"]
        assert not any("近距离特写" in n for n in notes)

    def test_empty_photo_type_summary_no_warning(self):
        """When photo_type_summary is an empty dict, no warning fires."""
        result = generate_damage_overview(
            self._build_parts(),
            structural_damage_flag=False,
            overall_severity="moderate",
            primary_damage_zone="front",
            photo_type_summary={},
        )
        notes = result["human_review_notes"]
        assert not any("近距离特写" in n for n in notes)


class TestComputePhotoTypeSummary:
    """Helper that aggregates photo counts by photo_type from a planner plan."""

    def test_counts_by_planner_photo_type(self):
        from agents.damage_overview import _compute_photo_type_summary
        plan = {
            "view_groups": {
                "front": [
                    {"id": "p1.png", "_planner_photo_type": "wide_shot"},
                    {"id": "p2.png", "_planner_photo_type": "close_up_damage"},
                ],
                "rear": [
                    {"id": "p3.png", "_planner_photo_type": "close_up_damage"},
                ],
            }
        }
        summary = _compute_photo_type_summary(plan)
        assert summary["wide_shot"] == 1
        assert summary["close_up_damage"] == 2
        assert summary["close_up_detail"] == 0
        assert summary["unknown"] == 0

    def test_skips_non_exterior_views(self):
        from agents.damage_overview import _compute_photo_type_summary
        plan = {
            "view_groups": {
                "front": [
                    {"id": "p1.png", "_planner_photo_type": "wide_shot"},
                ],
                "interior": [
                    {"id": "p2.png", "_planner_photo_type": "close_up_damage"},
                ],
                "auxiliary": [
                    {"id": "p3.png", "_planner_photo_type": "close_up_damage"},
                ],
            }
        }
        summary = _compute_photo_type_summary(plan)
        assert summary["wide_shot"] == 1
        assert summary["close_up_damage"] == 0
        assert summary["unknown"] == 0

    def test_untyped_photos_count_as_unknown(self):
        from agents.damage_overview import _compute_photo_type_summary
        plan = {
            "view_groups": {
                "front": [
                    {"id": "p1.png"},  # no _planner_photo_type
                    {"id": "p2.png", "_planner_photo_type": "bogus_value"},
                ],
            }
        }
        summary = _compute_photo_type_summary(plan)
        assert summary["unknown"] == 2

    def test_handles_missing_view_groups(self):
        from agents.damage_overview import _compute_photo_type_summary
        # No view_groups key, no error.
        summary = _compute_photo_type_summary({})
        assert summary == {"wide_shot": 0, "close_up_damage": 0, "close_up_detail": 0, "unknown": 0}
