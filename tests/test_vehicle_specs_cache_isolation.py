"""Tests for vehicle_specs_cache key isolation.

The cache must never read or write the shared empty (``"||"``) bucket produced
when brand/model are blank.  Caching under that bucket let a default-sedan write
(or a mis-detected vehicle) be served back to every vehicle-info-less sample,
silently poisoning unrelated cars (e.g. a Mercedes GLC read as a sedan).
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.vehicle_specs_cache import (  # noqa: E402
    get_cached_specs,
    is_cacheable_key,
    make_cache_key,
    save_cached_specs,
)
from models.vehicle_specs import VehicleSpecs  # noqa: E402


class TestMakeCacheKey:
    def test_lowercases_and_strips(self):
        info = {"brand": "  BMW ", "model": "X5", "year": "2024"}
        assert make_cache_key(info) == "bmw|x5|2024"

    def test_empty_info_collapses_to_shared_bucket(self):
        # This is exactly the dangerous key we must refuse to cache.
        assert make_cache_key({"brand": "", "model": "", "year": ""}) == "||"
        assert make_cache_key({}) == "||"

    def test_missing_fields_default_to_empty(self):
        assert make_cache_key({"brand": "奔驰"}) == "奔驰||"


class TestIsCacheableKey:
    @pytest.mark.parametrize(
        "info,expected",
        [
            ({"brand": "奔驰", "model": "GLC", "year": "2024"}, True),
            ({"brand": "奔驰", "model": "GLC"}, True),  # year optional
            ({"brand": "", "model": "", "year": ""}, False),
            ({}, False),
            ({"brand": "", "model": "GLC"}, False),  # missing brand
            ({"brand": "奔驰", "model": ""}, False),  # missing model
            ({"brand": "  ", "model": "  "}, False),  # whitespace only
        ],
    )
    def test_cacheable_requires_brand_and_model(self, info, expected):
        assert is_cacheable_key(info) is expected


class TestEmptyKeyIsolation:
    """Reading/writing a blank vehicle_info must be a no-op (no shared bucket)."""

    def test_save_blank_info_is_noop_json(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr("data.vehicle_specs_cache.CACHE_FILE", cache_file)

        save_cached_specs({"brand": "", "model": "", "year": ""}, VehicleSpecs.from_dict({}))

        # Nothing should have been written under the shared bucket.
        assert not cache_file.exists() or "||" not in cache_file.read_text(encoding="utf-8")

    def test_get_blank_info_returns_none(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        # Pre-seed the dangerous shared bucket directly on disk.
        cache_file.write_text(
            '{"||": {"body_style": "sedan", "doors": 4, "has_sunroof": false,'
            ' "has_roof_rack": false, "headlight_layout": "separate",'
            ' "rear_door_type": "trunk_lid", "notes": "default"}}',
            encoding="utf-8",
        )
        monkeypatch.setattr("data.vehicle_specs_cache.CACHE_FILE", cache_file)
        # Force the ORM lookup to miss so we exercise the JSON fallback path.
        monkeypatch.setattr("data.vehicle_specs_cache._get_orm_specs", lambda info: None)

        # Even though "||" exists on disk, a blank lookup must NOT read it.
        assert get_cached_specs({"brand": "", "model": "", "year": ""}) is None

    def test_roundtrip_for_valid_key(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr("data.vehicle_specs_cache.CACHE_FILE", cache_file)
        # Force the ORM lookup to miss so we exercise the JSON fallback path.
        monkeypatch.setattr("data.vehicle_specs_cache._get_orm_specs", lambda info: None)
        specs = VehicleSpecs.from_dict({"body_style": "suv", "doors": 5, "has_sunroof": True})
        info = {"brand": "奔驰", "model": "GLC", "year": "2024"}

        save_cached_specs(info, specs)
        got = get_cached_specs(info)

        assert got is not None
        assert got.body_style == "suv"
        assert got.has_sunroof is True
