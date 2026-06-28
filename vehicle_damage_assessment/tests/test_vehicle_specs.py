"""Tests for VehicleSpecs model and vehicle_specs cache."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from models.vehicle_specs import VehicleSpecs, ALLOWED_BODY_STYLES, DEFAULT_BODY_STYLE
from data.vehicle_specs_cache import (
    make_cache_key,
    get_cached_specs,
    save_cached_specs,
    _load_json_cache as _load_cache,
    _save_json_cache as _save_cache,
)


class TestVehicleSpecsCreation:
    """Test VehicleSpecs dataclass construction and defaults."""

    def test_default_values(self):
        specs = VehicleSpecs()
        assert specs.body_style == "sedan"
        assert specs.doors == 4
        assert specs.has_sunroof is False
        assert specs.has_roof_rack is False
        assert specs.headlight_layout == "separate"
        assert specs.rear_door_type == "trunk_lid"
        assert specs.notes == ""

    def test_custom_values(self):
        specs = VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="split",
            rear_door_type="tailgate",
            notes="Test vehicle",
        )
        assert specs.body_style == "suv"
        assert specs.doors == 5
        assert specs.has_sunroof is True
        assert specs.has_roof_rack is True
        assert specs.headlight_layout == "split"
        assert specs.rear_door_type == "tailgate"
        assert specs.notes == "Test vehicle"

    def test_frozen_immutable(self):
        specs = VehicleSpecs()
        with pytest.raises(AttributeError):
            specs.body_style = "hatchback"

    def test_to_dict_roundtrip(self):
        specs = VehicleSpecs(body_style="hatchback", doors=3, has_sunroof=True)
        d = specs.to_dict()
        assert d["body_style"] == "hatchback"
        assert d["doors"] == 3
        assert d["has_sunroof"] is True
        assert d["has_roof_rack"] is False


class TestVehicleSpecsFromDict:
    """Test VehicleSpecs.from_dict normalization and inference."""

    def test_from_dict_basic(self):
        d = {
            "body_style": "suv",
            "doors": 5,
            "has_sunroof": True,
            "has_roof_rack": False,
            "headlight_layout": "led",
            "rear_door_type": "tailgate",
            "notes": "Family SUV",
        }
        specs = VehicleSpecs.from_dict(d)
        assert specs.body_style == "suv"
        assert specs.doors == 5
        assert specs.has_sunroof is True
        assert specs.has_roof_rack is False
        assert specs.headlight_layout == "led"
        assert specs.rear_door_type == "tailgate"
        assert specs.notes == "Family SUV"

    def test_from_dict_empty_defaults(self):
        specs = VehicleSpecs.from_dict({})
        assert specs.body_style == DEFAULT_BODY_STYLE
        assert specs.doors == 4
        assert specs.has_sunroof is False
        assert specs.has_roof_rack is False
        assert specs.rear_door_type == "trunk_lid"  # sedan default

    def test_from_dict_invalid_body_style(self):
        specs = VehicleSpecs.from_dict({"body_style": "spaceship"})
        assert specs.body_style == DEFAULT_BODY_STYLE

    def test_from_dict_body_style_lowercased(self):
        specs = VehicleSpecs.from_dict({"body_style": "SUV"})
        assert specs.body_style == "suv"

    def test_from_dict_doors_coerced(self):
        specs = VehicleSpecs.from_dict({"doors": "5"})
        assert specs.doors == 5

    def test_from_dict_doors_invalid_default(self):
        specs = VehicleSpecs.from_dict({"doors": "abc"})
        assert specs.doors == 4

    def test_from_dict_booleans_coerced(self):
        specs = VehicleSpecs.from_dict({"has_sunroof": "yes", "has_roof_rack": 1})
        assert specs.has_sunroof is True
        assert specs.has_roof_rack is True

    def test_from_dict_rear_door_type_inference_sedan(self):
        """Sedan -> trunk_lid."""
        specs = VehicleSpecs.from_dict({"body_style": "sedan"})
        assert specs.rear_door_type == "trunk_lid"

    def test_from_dict_rear_door_type_inference_coupe(self):
        """Coupe -> trunk_lid."""
        specs = VehicleSpecs.from_dict({"body_style": "coupe"})
        assert specs.rear_door_type == "trunk_lid"

    def test_from_dict_rear_door_type_inference_convertible(self):
        """Convertible -> trunk_lid."""
        specs = VehicleSpecs.from_dict({"body_style": "convertible"})
        assert specs.rear_door_type == "trunk_lid"

    def test_from_dict_rear_door_type_inference_hatchback(self):
        """Hatchback -> tailgate."""
        specs = VehicleSpecs.from_dict({"body_style": "hatchback"})
        assert specs.rear_door_type == "tailgate"

    def test_from_dict_rear_door_type_inference_suv(self):
        """SUV -> tailgate."""
        specs = VehicleSpecs.from_dict({"body_style": "suv"})
        assert specs.rear_door_type == "tailgate"

    def test_from_dict_rear_door_type_inference_van(self):
        """Van -> tailgate."""
        specs = VehicleSpecs.from_dict({"body_style": "van"})
        assert specs.rear_door_type == "tailgate"

    def test_from_dict_rear_door_type_inference_wagon(self):
        """Wagon -> tailgate."""
        specs = VehicleSpecs.from_dict({"body_style": "wagon"})
        assert specs.rear_door_type == "tailgate"

    def test_from_dict_rear_door_type_inference_pickup(self):
        """Pickup -> tailgate."""
        specs = VehicleSpecs.from_dict({"body_style": "pickup"})
        assert specs.rear_door_type == "tailgate"

    def test_from_dict_rear_door_type_inference_mpv(self):
        """MPV -> tailgate."""
        specs = VehicleSpecs.from_dict({"body_style": "mpv"})
        assert specs.rear_door_type == "tailgate"

    def test_from_dict_rear_door_type_explicit(self):
        """Explicit rear_door_type overrides inference."""
        specs = VehicleSpecs.from_dict({"body_style": "sedan", "rear_door_type": "sliding"})
        assert specs.rear_door_type == "sliding"

    def test_from_dict_rear_door_type_invalid_infers(self):
        """Invalid rear_door_type falls back to inference."""
        specs = VehicleSpecs.from_dict({"body_style": "suv", "rear_door_type": "magic"})
        assert specs.rear_door_type == "tailgate"

    def test_from_dict_roundtrip(self):
        original = VehicleSpecs(
            body_style="hatchback",
            doors=3,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="matrix",
            rear_door_type="tailgate",
            notes="Hot hatch",
        )
        d = original.to_dict()
        restored = VehicleSpecs.from_dict(d)
        assert restored == original

    def test_all_allowed_body_styles(self):
        for style in ALLOWED_BODY_STYLES:
            specs = VehicleSpecs.from_dict({"body_style": style})
            assert specs.body_style == style


class TestVehicleSpecsInferRearDoorType:
    """Test the static _infer_rear_door_type helper."""

    def test_trunk_lid_styles(self):
        for style in ("sedan", "coupe", "convertible"):
            assert VehicleSpecs._infer_rear_door_type(style) == "trunk_lid"

    def test_tailgate_styles(self):
        for style in ("hatchback", "suv", "mpv", "van", "pickup", "wagon"):
            assert VehicleSpecs._infer_rear_door_type(style) == "tailgate"


class TestCacheKey:
    """Test make_cache_key."""

    def test_basic(self):
        key = make_cache_key({"brand": "Toyota", "model": "Camry", "year": "2023"})
        assert key == "toyota|camry|2023"

    def test_lowercased(self):
        key = make_cache_key({"brand": "BMW", "model": "X5", "year": "2024"})
        assert key == "bmw|x5|2024"

    def test_whitespace_trimmed(self):
        key = make_cache_key({"brand": "  Toyota  ", "model": " Camry ", "year": " 2023 "})
        assert key == "toyota|camry|2023"

    def test_missing_fields(self):
        key = make_cache_key({"brand": "Toyota"})
        assert key == "toyota||"

    def test_colon_in_model(self):
        """Pipe separator avoids collision with colons in model names."""
        key = make_cache_key({"brand": "Mercedes", "model": "C-Class", "year": "2023"})
        assert key == "mercedes|c-class|2023"


class TestCacheRoundTrip:
    """Test cache read/write with a temp file path for isolation."""

    @pytest.fixture
    def temp_cache(self, tmp_path, monkeypatch):
        """Override the cache file path for testing and disable ORM lookups."""
        import data.vehicle_specs_cache as cache_module
        original_file = cache_module.CACHE_FILE
        temp_file = tmp_path / "test_cache.json"
        cache_module.CACHE_FILE = temp_file
        monkeypatch.setattr(cache_module, "_get_orm_specs", lambda _vehicle_info: None)
        yield temp_file
        cache_module.CACHE_FILE = original_file

    def test_save_and_get(self, temp_cache):
        vehicle_info = {"brand": "Toyota", "model": "Camry", "year": "2023"}
        specs = VehicleSpecs(body_style="sedan", doors=4, has_sunroof=False)

        save_cached_specs(vehicle_info, specs)
        cached = get_cached_specs(vehicle_info)

        assert cached is not None
        assert cached.body_style == "sedan"
        assert cached.doors == 4
        assert cached.has_sunroof is False

    def test_cache_miss(self, temp_cache):
        vehicle_info = {"brand": "Honda", "model": "Civic", "year": "2023"}
        cached = get_cached_specs(vehicle_info)
        assert cached is None

    def test_cache_file_created(self, temp_cache):
        vehicle_info = {"brand": "Ford", "model": "F-150", "year": "2023"}
        specs = VehicleSpecs(body_style="pickup", doors=4, has_sunroof=False)

        save_cached_specs(vehicle_info, specs)
        assert temp_cache.exists()

        data = json.loads(temp_cache.read_text(encoding="utf-8"))
        key = make_cache_key(vehicle_info)
        assert key in data
        assert data[key]["body_style"] == "pickup"

    def test_multiple_entries(self, temp_cache):
        specs1 = VehicleSpecs(body_style="sedan", doors=4)
        specs2 = VehicleSpecs(body_style="suv", doors=5, has_sunroof=True)

        save_cached_specs({"brand": "Toyota", "model": "Camry", "year": "2023"}, specs1)
        save_cached_specs({"brand": "BMW", "model": "X5", "year": "2024"}, specs2)

        cached1 = get_cached_specs({"brand": "Toyota", "model": "Camry", "year": "2023"})
        cached2 = get_cached_specs({"brand": "BMW", "model": "X5", "year": "2024"})

        assert cached1.body_style == "sedan"
        assert cached2.body_style == "suv"
        assert cached2.has_sunroof is True

    def test_overwrite_existing(self, temp_cache):
        vehicle_info = {"brand": "Toyota", "model": "Camry", "year": "2023"}
        specs1 = VehicleSpecs(body_style="sedan", doors=4)
        specs2 = VehicleSpecs(body_style="coupe", doors=2)

        save_cached_specs(vehicle_info, specs1)
        save_cached_specs(vehicle_info, specs2)

        cached = get_cached_specs(vehicle_info)
        assert cached.body_style == "coupe"
        assert cached.doors == 2

    def test_corrupted_cache_returns_none(self, temp_cache):
        """If cache file has bad JSON, get_cached_specs returns None."""
        temp_cache.write_text("not json", encoding="utf-8")
        cached = get_cached_specs({"brand": "Toyota", "model": "Camry", "year": "2023"})
        assert cached is None

    def test_malformed_entry_returns_none(self, temp_cache):
        """If a specific entry is malformed, get_cached_specs returns a default VehicleSpecs."""
        key = make_cache_key({"brand": "Toyota", "model": "Camry", "year": "2023"})
        temp_cache.write_text(json.dumps({key: "not a dict"}), encoding="utf-8")
        cached = get_cached_specs({"brand": "Toyota", "model": "Camry", "year": "2023"})
        # from_dict now gracefully handles non-dict input by returning defaults
        assert cached is not None
        assert cached.body_style == "sedan"
        assert cached.doors == 4

    def test_load_and_save_helpers(self, temp_cache):
        """Test _load_cache and _save_cache directly."""
        data = {"key1": {"body_style": "sedan"}, "key2": {"body_style": "suv"}}
        _save_cache(data)
        loaded = _load_cache()
        assert loaded == data


class TestVehiclePriorCacheBehavior:
    """Bug 1: cache should not return empty topology/key_anchors.

    When cache hit occurs, vehicle_prior_agent should still call the LLM
    to get topology/key_anchors, but override vehicle_specs with cached values.
    """

    @pytest.fixture
    def temp_cache(self, tmp_path, monkeypatch):
        """Override the cache file path for testing and disable ORM lookups."""
        import data.vehicle_specs_cache as cache_module
        original_file = cache_module.CACHE_FILE
        temp_file = tmp_path / "test_cache.json"
        cache_module.CACHE_FILE = temp_file
        monkeypatch.setattr(cache_module, "_get_orm_specs", lambda _vehicle_info: None)
        yield temp_file
        cache_module.CACHE_FILE = original_file

    @pytest.mark.asyncio
    async def test_cache_hit_still_calls_llm_for_topology(self, temp_cache):
        """When cache hit, LLM is still called and topology/key_anchors are preserved."""
        import agents.vehicle_prior as vp_module

        vehicle_info = {"brand": "NIO", "model": "ES8", "year": "2024"}
        cached_specs = VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="split",
            rear_door_type="tailgate",
            notes="Cached specs",
        )
        save_cached_specs(vehicle_info, cached_specs)

        # Mock LLM response with rich topology
        llm_response = {
            "vehicle": "2024 NIO ES8",
            "vehicle_specs": {
                "body_style": "sedan",  # Wrong! Should be overridden by cache
                "doors": 4,
                "has_sunroof": False,
                "rear_door_type": "trunk_lid",
            },
            "topology": {
                "front": "X-Bar前脸，分体式大灯",
                "rear": "贯穿尾灯，尾门",
                "left": "左侧双门+尾门",
                "right": "右侧双门+尾门",
                "roof": "全景天幕",
            },
            "key_anchors": {
                "front": ["分体式大灯上部", "分体式大灯下部"],
                "rear": ["贯穿尾灯", "尾门把手"],
                "left": ["左后视镜", "左前车门"],
                "right": ["右后视镜", "右前车门"],
                "roof": ["全景天幕"],
            },
            "common_photo_angles": ["车头正前", "车尾正后"],
        }

        async def mock_call_minimax(messages, **kwargs):
            import json
            return json.dumps(llm_response, ensure_ascii=False)

        with patch.object(vp_module, "call_minimax", side_effect=mock_call_minimax):
            result = await vp_module.vehicle_prior_agent(vehicle_info, use_cache=True)

        # vehicle_specs should come from cache (SUV), not LLM (sedan)
        assert result["vehicle_specs"]["body_style"] == "suv"
        assert result["vehicle_specs"]["doors"] == 5
        assert result["vehicle_specs"]["rear_door_type"] == "tailgate"

        # topology and key_anchors should come from LLM, NOT be empty
        assert result["topology"] == llm_response["topology"]
        assert result["key_anchors"] == llm_response["key_anchors"]
        assert result["common_photo_angles"] == llm_response["common_photo_angles"]

        # Should NOT have the old _cached flag
        assert "_cached" not in result

    @pytest.mark.asyncio
    async def test_cache_miss_saves_llm_specs(self, temp_cache):
        """When cache miss, LLM specs are saved to cache."""
        import agents.vehicle_prior as vp_module

        vehicle_info = {"brand": "New", "model": "Brand", "year": "2024"}

        llm_response = {
            "vehicle": "2024 New Brand",
            "vehicle_specs": {
                "body_style": "hatchback",
                "doors": 5,
                "has_sunroof": True,
                "rear_door_type": "tailgate",
            },
            "topology": {"front": "前脸"},
            "key_anchors": {"front": ["大灯"]},
        }

        async def mock_call_minimax(messages, **kwargs):
            import json
            return json.dumps(llm_response, ensure_ascii=False)

        with patch.object(vp_module, "call_minimax", side_effect=mock_call_minimax):
            result = await vp_module.vehicle_prior_agent(vehicle_info, use_cache=True)

        # Should save LLM specs to cache
        cached = get_cached_specs(vehicle_info)
        assert cached is not None
        assert cached.body_style == "hatchback"
        assert cached.doors == 5
