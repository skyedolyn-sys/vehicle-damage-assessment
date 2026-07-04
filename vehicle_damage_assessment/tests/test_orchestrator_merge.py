"""orchestrator._merge_two_states: verify priority maps come from rules loader.

These tests guard against future regressions where someone inlines priority
dict literals back into the orchestrator, drifting from the centralized
``agents.rules.load_priority_map()`` config that the synthesizer and
topology_comparator also use.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

# Ensure the project root is on the import path when pytest is run from
# elsewhere (the project's pytest.ini sets DJANGO_SETTINGS_MODULE so most
# tests work without this, but be defensive for direct invocation).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_orchestrator():
    import importlib

    # Importing via ``from agents import assessment_orchestrator`` would return
    # the function (re-exported by ``agents/__init__.py``), not the module.
    # We need the module object so we can read module-level priority maps and
    # inspect the source.
    return importlib.import_module("agents.assessment_orchestrator")


def _load_rules():
    from agents.rules import load_priority_map

    return load_priority_map


def test_orchestrator_uses_loader_for_status_priority():
    """orchestrator._STATUS_PRIORITY must come from load_priority_map()['status']."""
    orchestrator = _load_orchestrator()
    assert hasattr(orchestrator, "_STATUS_PRIORITY"), (
        "assessment_orchestrator must expose a module-level _STATUS_PRIORITY "
        "loaded from agents.rules.load_priority_map()"
    )

    expected = _load_rules()()["status"]
    assert orchestrator._STATUS_PRIORITY == expected, (
        "_STATUS_PRIORITY drifted from the rules loader: "
        f"module={dict(orchestrator._STATUS_PRIORITY)} loader={dict(expected)}"
    )


def test_orchestrator_uses_loader_for_level_priority():
    """orchestrator._LEVEL_PRIORITY must come from load_priority_map()['level']."""
    orchestrator = _load_orchestrator()
    assert hasattr(orchestrator, "_LEVEL_PRIORITY"), (
        "assessment_orchestrator must expose a module-level _LEVEL_PRIORITY "
        "loaded from agents.rules.load_priority_map()"
    )

    expected = _load_rules()()["level"]
    assert orchestrator._LEVEL_PRIORITY == expected, (
        "_LEVEL_PRIORITY drifted from the rules loader: "
        f"module={dict(orchestrator._LEVEL_PRIORITY)} loader={dict(expected)}"
    )


def test_orchestrator_uses_loader_for_confidence_priority():
    """orchestrator._CONFIDENCE_PRIORITY must come from load_priority_map()['confidence']."""
    orchestrator = _load_orchestrator()
    assert hasattr(orchestrator, "_CONFIDENCE_PRIORITY"), (
        "assessment_orchestrator must expose a module-level _CONFIDENCE_PRIORITY "
        "loaded from agents.rules.load_priority_map()"
    )

    expected = _load_rules()()["confidence"]
    assert orchestrator._CONFIDENCE_PRIORITY == expected, (
        "_CONFIDENCE_PRIORITY drifted from the rules loader: "
        f"module={dict(orchestrator._CONFIDENCE_PRIORITY)} loader={dict(expected)}"
    )


def test_orchestrator_does_not_hardcode_priority_dicts():
    """orchestrator source must not contain hardcoded STATUS/LEVEL/CONFIDENCE dicts.

    Only loader-driven assignments are allowed.  This regex catches the
    previous inlined pattern, e.g.::

        status_priority = {Status.MISSING: 4, Status.DAMAGED: 3, ...}

    which is exactly the drift the loader is meant to prevent.
    """
    orchestrator = _load_orchestrator()
    source = inspect.getsource(orchestrator)

    forbidden_patterns = [
        # The historical inlined status map (with status enum keys).
        re.compile(
            r"status_priority\s*=\s*\{[^}]*Status\.(MISSING|DAMAGED|UNCERTAIN|INTACT|NOT_APPLICABLE)",
            re.DOTALL,
        ),
        # The historical inlined level map.
        re.compile(
            r"level_priority\s*=\s*\{[^}]*DamageLevel\.(SEVERE|MODERATE|LIGHT|UNKNOWN|NONE)",
            re.DOTALL,
        ),
        # The historical inlined confidence map with all three levels.
        re.compile(
            r"confidence_priority\s*=\s*\{[^}]*[\"']high[\"']\s*:\s*\d+[^}]*[\"']low[\"']",
            re.DOTALL,
        ),
    ]

    for pattern in forbidden_patterns:
        match = pattern.search(source)
        assert match is None, (
            "Found hardcoded priority dict in assessment_orchestrator; "
            "must use agents.rules.load_priority_map() instead. "
            f"Matched: {match.group(0)[:160] if match else ''}"
        )


def test_merge_two_states_resolves_missing_over_damaged():
    """Sanity: _merge_two_states honors loader priorities end-to-end."""
    from models.part_state import DamageLevel, PartActualState, Status

    orchestrator = _load_orchestrator()
    a = PartActualState(
        part_id="hood",
        part_name="引擎盖",
        part_category="front",
        side="center",
        status=Status.DAMAGED,
        damage_level=DamageLevel.MODERATE,
        confidence="high",
    )
    b = PartActualState(
        part_id="hood",
        part_name="引擎盖",
        part_category="front",
        side="center",
        status=Status.MISSING,
        damage_level=DamageLevel.SEVERE,
        confidence="medium",
    )
    merged = orchestrator._merge_two_states(a, b)
    # Loader status map: missing=4, damaged=3 → MISSING wins.
    assert merged.status == Status.MISSING
    # Loader level map: severe=4, moderate=3 → SEVERE wins.
    assert merged.damage_level == DamageLevel.SEVERE
    # Loader confidence map: high=2, medium=1 → lower value (medium) is "worst".
    assert merged.confidence == "medium"
