"""Tests for rules engine utilities."""

import pytest

from agents.rules.engine.cache import CachedConfig, LRUCache, load_with_cache
from agents.rules.engine.merge import deep_merge, merge_vehicle_type_config
from agents.rules.engine.validate import (
    ConfigValidationError,
    validate_config,
    validate_filename_heuristics,
    validate_part_aliases,
    validate_part_list_config,
    validate_priorities,
    validate_thresholds,
    validate_view_weights,
)


class TestLRUCache:
    def test_cache_hit_and_miss(self, tmp_path):
        cache = LRUCache(capacity=2)
        path = tmp_path / "test.yaml"
        path.write_text("a: 1", encoding="utf-8")

        data1 = load_with_cache(path, cache=cache)
        data2 = load_with_cache(path, cache=cache)
        assert data1 == data2 == {"a": 1}
        assert len(cache) == 1

    def test_cache_invalidated_by_mtime_change(self, tmp_path):
        cache = LRUCache(capacity=2)
        path = tmp_path / "test.yaml"
        path.write_text("a: 1", encoding="utf-8")

        load_with_cache(path, cache=cache)
        path.write_text("a: 2", encoding="utf-8")
        data = load_with_cache(path, cache=cache)
        assert data == {"a": 2}

    def test_lru_eviction(self, tmp_path):
        cache = LRUCache(capacity=2)
        for i in range(3):
            path = tmp_path / f"test{i}.yaml"
            path.write_text(f"a: {i}", encoding="utf-8")
            load_with_cache(path, cache=cache)
        assert len(cache) == 2

    def test_missing_file_returns_empty_dict(self, tmp_path):
        cache = LRUCache(capacity=2)
        path = tmp_path / "missing.yaml"
        assert load_with_cache(path, cache=cache) == {}


class TestDeepMerge:
    def test_scalar_dict_replacement(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert deep_merge(base, override) == {"a": 1, "b": 3}

    def test_list_replace_mode(self):
        base = {"x": [1, 2]}
        override = {"x": [3]}
        assert deep_merge(base, override) == {"x": [3]}

    def test_list_additive_mode(self):
        base = {"x": [1, 2]}
        override = {"x": [3]}
        assert deep_merge(base, override, list_mode="additive") == {"x": [1, 2, 3]}

    def test_nested_dict_merge(self):
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"c": 3}}
        assert deep_merge(base, override) == {"a": {"b": 1, "c": 3}}

    def test_merge_vehicle_type_config_uses_default(self):
        config = {"default": {"a": 1}, "by_vehicle_type": {"suv": {"a": 2}}}
        assert merge_vehicle_type_config(config, "sedan") == {"a": 1}

    def test_merge_vehicle_type_config_applies_override(self):
        config = {"default": {"a": 1}, "by_vehicle_type": {"suv": {"a": 2}}}
        assert merge_vehicle_type_config(config, "suv") == {"a": 2}


class TestValidate:
    def test_validate_priorities_ok(self):
        data = {
            "default": {
                "status": {"damaged": 3, "intact": 1},
                "level": {"severe": 2, "light": 1},
                "confidence": {"high": 2},
                "uncertain_status": {"damaged": 1, "intact": 0},
            }
        }
        assert validate_priorities(data) == []

    def test_validate_priorities_rejects_non_integer(self):
        data = {"default": {"status": {"damaged": "high"}}}
        errors = validate_priorities(data)
        assert any("must be an integer" in e for e in errors)

    def test_validate_part_list_config_ok(self):
        data = {"default": {"conservative": ["door_front_left", "mirror_left"]}}
        assert validate_part_list_config(data, "part_profiles", known_parts={"door_front_left", "mirror_left"}) == []

    def test_validate_part_list_config_unknown_part(self):
        data = {"default": {"conservative": ["door_front_left", "unknown_part"]}}
        errors = validate_part_list_config(data, "part_profiles", known_parts={"door_front_left"})
        assert any("unknown part id" in e for e in errors)

    def test_validate_thresholds_ok(self):
        data = {"default": {"visibility_definite_ratio": 0.3}}
        assert validate_thresholds(data) == []

    def test_validate_thresholds_rejects_string(self):
        data = {"default": {"visibility_definite_ratio": "0.3"}}
        errors = validate_thresholds(data)
        assert any("must be a number" in e for e in errors)

    def test_validate_view_weights_ok(self):
        data = {
            "default": {
                "primary_view": {"door_front_left": ["left_90"]},
                "view_weights": {
                    "door_front_left": {"primary": ["left_90"], "secondary": ["front_left_45"]}
                },
                "roof_primary_regions": ["top"],
                "roof_secondary_regions": ["left_90"],
            }
        }
        assert validate_view_weights(data) == []

    def test_validate_view_weights_rejects_bad_bucket(self):
        data = {
            "default": {
                "view_weights": {"door_front_left": {"primary": ["left_90"], "other": ["right_90"]}},
            }
        }
        errors = validate_view_weights(data)
        assert any("primary or secondary" in e for e in errors)

    def test_validate_part_aliases_ok(self):
        data = {"aliases": {"hood": {"canonical": "hood", "synonyms": ["bonnet"]}}}
        assert validate_part_aliases(data, known_parts={"hood"}) == []

    def test_validate_part_aliases_unknown_canonical(self):
        data = {"aliases": {"hood": {"canonical": "hood", "synonyms": ["bonnet"]}}}
        errors = validate_part_aliases(data, known_parts={"hood"})
        assert errors == []

    def test_validate_part_aliases_unknown_canonical_rejected(self):
        data = {"aliases": {"hood": {"canonical": "hood", "synonyms": ["bonnet"]}}}
        errors = validate_part_aliases(data, known_parts=set())
        assert any("unknown canonical part id" in e for e in errors)

    def test_validate_filename_heuristics_ok(self):
        data = {"rules": [{"name": "front", "patterns": ["front"], "priority": 100}]}
        assert validate_filename_heuristics(data) == []

    def test_validate_filename_heuristics_missing_field(self):
        data = {"rules": [{"name": "front"}]}
        errors = validate_filename_heuristics(data)
        assert any("missing required field" in e for e in errors)

    def test_validate_config_dispatch(self):
        data = {"default": {"status": {"damaged": 1}}}
        assert validate_config("priorities", data) == []

    def test_validate_config_unknown_config(self):
        errors = validate_config("not_a_config", {})
        assert any("no validator registered" in e for e in errors)
