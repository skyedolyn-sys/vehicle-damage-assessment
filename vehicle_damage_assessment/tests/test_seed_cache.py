"""Tests for seed_vehicle_specs_cache script.

Verifies that COMMON_CHINESE_VEHICLES entries are well-formed and consistent.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so imports resolve
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.seed_vehicle_specs_cache import COMMON_CHINESE_VEHICLES  # noqa: E402
from models.vehicle_specs import VehicleSpecs, ALLOWED_BODY_STYLES  # noqa: E402


class TestSeedVehicleEntries:
    """Validate every entry in COMMON_CHINESE_VEHICLES."""

    @pytest.fixture(scope="module")
    def entries(self):
        return COMMON_CHINESE_VEHICLES

    def test_at_least_200_entries(self, entries):
        assert len(entries) >= 200, f"Expected at least 200 entries, got {len(entries)}"

    def test_all_entries_have_vehicle_info(self, entries):
        for i, entry in enumerate(entries):
            assert "vehicle_info" in entry, f"Entry {i} missing vehicle_info"
            info = entry["vehicle_info"]
            assert "brand" in info and info["brand"], f"Entry {i} missing brand"
            assert "model" in info and info["model"], f"Entry {i} missing model"
            assert "year" in info and info["year"], f"Entry {i} missing year"

    def test_all_entries_have_specs(self, entries):
        for i, entry in enumerate(entries):
            assert "specs" in entry, f"Entry {i} missing specs"
            assert isinstance(entry["specs"], VehicleSpecs), f"Entry {i} specs is not VehicleSpecs"

    def test_all_body_styles_valid(self, entries):
        for i, entry in enumerate(entries):
            body = entry["specs"].body_style
            assert body in ALLOWED_BODY_STYLES, (
                f"Entry {i} ({entry['vehicle_info']}) has invalid body_style: {body}"
            )

    def test_rear_door_type_consistency_sedan_coupe_convertible(self, entries):
        """Sedans, coupes, and convertibles should have trunk_lid."""
        for i, entry in enumerate(entries):
            body = entry["specs"].body_style
            rear = entry["specs"].rear_door_type
            if body in ("sedan", "coupe", "convertible"):
                assert rear == "trunk_lid", (
                    f"Entry {i} ({entry['vehicle_info']}) {body} should have trunk_lid, got {rear}"
                )

    def test_rear_door_type_consistency_suv_hatchback_wagon_pickup(self, entries):
        """SUVs, hatchbacks, wagons, and pickups should have tailgate."""
        for i, entry in enumerate(entries):
            body = entry["specs"].body_style
            rear = entry["specs"].rear_door_type
            if body in ("suv", "hatchback", "wagon", "pickup"):
                assert rear == "tailgate", (
                    f"Entry {i} ({entry['vehicle_info']}) {body} should have tailgate, got {rear}"
                )

    def test_mpv_van_rear_door_type(self, entries):
        """MPVs and vans should have sliding or tailgate rear door."""
        for i, entry in enumerate(entries):
            body = entry["specs"].body_style
            rear = entry["specs"].rear_door_type
            if body in ("mpv", "van"):
                assert rear in ("sliding", "tailgate"), (
                    f"Entry {i} ({entry['vehicle_info']}) {body} should have sliding or tailgate, got {rear}"
                )

    def test_doors_realistic(self, entries):
        """Doors count should be realistic (2, 3, 4, or 5)."""
        for i, entry in enumerate(entries):
            doors = entry["specs"].doors
            assert doors in (2, 3, 4, 5), (
                f"Entry {i} ({entry['vehicle_info']}) has unrealistic doors count: {doors}"
            )

    def test_sedans_have_4_doors(self, entries):
        """Sedans should typically have 4 doors."""
        for i, entry in enumerate(entries):
            if entry["specs"].body_style == "sedan":
                assert entry["specs"].doors == 4, (
                    f"Entry {i} ({entry['vehicle_info']}) sedan should have 4 doors"
                )

    def test_suvs_mpvs_have_5_doors(self, entries):
        """SUVs and MPVs should typically have 5 doors."""
        for i, entry in enumerate(entries):
            if entry["specs"].body_style in ("suv", "mpv"):
                assert entry["specs"].doors == 5, (
                    f"Entry {i} ({entry['vehicle_info']}) {entry['specs'].body_style} should have 5 doors"
                )

    def test_hatchbacks_have_3_or_5_doors(self, entries):
        """Hatchbacks should have 3 or 5 doors."""
        for i, entry in enumerate(entries):
            if entry["specs"].body_style == "hatchback":
                assert entry["specs"].doors in (3, 5), (
                    f"Entry {i} ({entry['vehicle_info']}) hatchback should have 3 or 5 doors"
                )

    def test_notes_not_empty(self, entries):
        """Every entry should have a non-empty notes field."""
        for i, entry in enumerate(entries):
            notes = entry["specs"].notes
            assert notes and isinstance(notes, str), (
                f"Entry {i} ({entry['vehicle_info']}) missing or empty notes"
            )

    def test_headlight_layout_not_empty(self, entries):
        for i, entry in enumerate(entries):
            hl = entry["specs"].headlight_layout
            assert hl and isinstance(hl, str), (
                f"Entry {i} ({entry['vehicle_info']}) missing headlight_layout"
            )

    def test_required_brands_present(self, entries):
        """Verify that all required brands from the spec have at least one entry."""
        brands = {entry["vehicle_info"]["brand"] for entry in entries}
        required_brands = {
            "蔚来", "理想", "小鹏", "比亚迪", "特斯拉",
            "宝马", "奔驰", "奥迪", "丰田", "本田", "大众", "五菱",
        }
        missing = required_brands - brands
        assert not missing, f"Missing required brands: {missing}"

    def test_required_models_present(self, entries):
        """Verify that specific required models are present."""
        models = {(entry["vehicle_info"]["brand"], entry["vehicle_info"]["model"]) for entry in entries}
        required_models = {
            ("蔚来", "ES6"), ("蔚来", "ES8"), ("蔚来", "ET5"), ("蔚来", "ET7"), ("蔚来", "EC6"),
            ("理想", "L7"), ("理想", "L8"), ("理想", "L9"), ("理想", "MEGA"),
            ("小鹏", "P7"), ("小鹏", "G6"), ("小鹏", "G9"), ("小鹏", "X9"),
            ("比亚迪", "秦PLUS"), ("比亚迪", "汉"), ("比亚迪", "唐"), ("比亚迪", "海豚"),
            ("比亚迪", "海鸥"), ("比亚迪", "宋PLUS"), ("比亚迪", "元PLUS"),
            ("特斯拉", "Model 3"), ("特斯拉", "Model Y"),
            ("宝马", "3系"), ("宝马", "5系"), ("宝马", "X3"), ("宝马", "X5"),
            ("奔驰", "C级"), ("奔驰", "E级"), ("奔驰", "GLC"),
            ("奥迪", "A4L"), ("奥迪", "A6L"), ("奥迪", "Q5L"),
            ("丰田", "凯美瑞"), ("丰田", "汉兰达"), ("丰田", "赛那"),
            ("本田", "雅阁"), ("本田", "CR-V"), ("本田", "奥德赛"),
            ("大众", "迈腾"), ("大众", "帕萨特"), ("大众", "途观L"),
            ("五菱", "宏光MINIEV"),
        }
        missing = required_models - models
        assert not missing, f"Missing required models: {missing}"

    def test_vehicle_specs_frozen(self, entries):
        """VehicleSpecs instances should be immutable (frozen dataclass)."""
        for entry in entries:
            with pytest.raises(AttributeError):
                entry["specs"].body_style = "changed"

    def test_to_dict_roundtrip(self, entries):
        """Every VehicleSpecs should round-trip through to_dict / from_dict."""
        for i, entry in enumerate(entries):
            original = entry["specs"]
            d = original.to_dict()
            restored = VehicleSpecs.from_dict(d)
            assert restored == original, (
                f"Entry {i} ({entry['vehicle_info']}) round-trip failed"
            )
