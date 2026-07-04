"""agents/rules/loader.py: verify _LEGACY_* hardcoded fallbacks are gone
and the loader fails loud when YAML is missing.

These tests guard the fail-loud contract: a missing YAML must surface as
``RuleLoadError`` instead of being silently masked by in-code defaults.
"""
import inspect
from pathlib import Path

import pytest

from agents.rules import loader
from agents.rules.loader import RuleLoadError


CONFIG_DIR = Path(__file__).resolve().parent.parent / "agents" / "rules" / "config"


def test_loader_has_no_legacy_hardcoded_dicts():
    """loader.py must not contain any _LEGACY_* hardcoded dictionary."""
    src = inspect.getsource(loader)
    assert "_LEGACY_" not in src, (
        "loader.py still contains _LEGACY_ fallback; remove it for fail-loud behavior"
    )
    # Belt-and-suspenders: the legacy constants block and _legacy_config
    # helper were the two surfaces we relied on.
    assert "_legacy_config" not in src, (
        "loader.py still defines _legacy_config; remove for fail-loud behavior"
    )


def test_yaml_files_all_present():
    """All required YAML config files must exist on disk."""
    required = [
        "view_weights.yaml",
        "part_profiles.yaml",
        "priorities.yaml",
        "damage_types.yaml",
        "thresholds.yaml",
        "filename_heuristics.yaml",
        "part_aliases.yaml",
        "checklist_hints.yaml",
        "filename_view_hints.yaml",
        "region_units.yaml",
        "trigger_sets.yaml",
    ]
    missing = [name for name in required if not (CONFIG_DIR / name).exists()]
    assert not missing, f"YAML files missing under {CONFIG_DIR}: {missing}"


def test_loader_fails_loud_on_missing_yaml(tmp_path, monkeypatch):
    """If YAML is absent, loader must raise RuleLoadError, not silently fall back."""
    # Point _CONFIG_DIR at an empty tmp dir so every load returns {}.
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)
    # Reset the global LRU cache so the previous real-path entries do not
    # satisfy the call now that _CONFIG_DIR has changed.
    loader.load_with_cache.__globals__["_GLOBAL_CACHE"].clear()

    with pytest.raises(RuleLoadError) as exc_info:
        loader.load_view_weights()

    msg = str(exc_info.value).lower()
    assert "missing" in msg or "not found" in msg, (
        f"expected fail-loud error mentioning missing YAML, got: {exc_info.value!r}"
    )