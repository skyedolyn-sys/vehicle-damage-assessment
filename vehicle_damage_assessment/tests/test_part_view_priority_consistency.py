"""Consistency test: part_view_priority (priority 0) and view_weights.primary must match.

Both data sources are loaded from agents/rules/config/view_weights.yaml.
Per DAMAGE_RECOGNITION_POLICY §3.1, evidence_fusion uses part_view_priority
to decide whether a primary-view intact dominates secondary-view damaged.
Synthesizer uses view_weights.primary for the same purpose.  If the two
disagree, the same input could produce different conclusions in different
code paths.

This test enforces strict equality on the priority == 0 set: those views are
the most authoritative vantage for the part.  priority == 1 / priority == 2
are near-primary / secondary and are checked separately for view validity
only.
"""

from agents.rules import load_part_view_priority, load_view_weights
from agents.view_mapping import STANDARD_VIEWS


def _zero_priority_views(part_view_priority, part_id):
    """Return the set of views with priority == 0 for ``part_id``.

    If the part_id is not in part_view_priority, return None so the caller
    can decide whether missing entries are allowed.
    """
    if part_id not in part_view_priority:
        return None
    return {v for v, p in part_view_priority[part_id].items() if p == 0}


def test_part_view_priority_zero_matches_view_weights_primary():
    """For every part_id defined in BOTH sources, priority == 0 set must equal
    view_weights.primary set.

    This catches the case where evidence_fusion treats ``[front]`` as the
    authoritative vantage for ``roof_front`` while synthesizer treats ``[top]``
    as the primary view — they would produce conflicting outputs on the same
    input.
    """
    ppp = load_part_view_priority()
    vw = load_view_weights()
    vw_primary = vw.get("primary_view", {})
    vw_view_weights = vw.get("view_weights", {})

    mismatches = []
    for part_id in ppp:
        ppp_zero = _zero_priority_views(ppp, part_id)
        if not ppp_zero:
            continue

        # Compare against primary_view (legacy single-view list).
        if part_id in vw_primary:
            primary_list = set(vw_primary[part_id])
            if ppp_zero != primary_list:
                mismatches.append(
                    f"{part_id}: part_view_priority[0]={sorted(ppp_zero)} "
                    f"vs view_weights.primary_view={sorted(primary_list)}"
                )

        # Compare against view_weights.primary (multi-view set).
        vw_entry = vw_view_weights.get(part_id)
        if isinstance(vw_entry, dict):
            vw_primary_set = set(vw_entry.get("primary", set()))
            if ppp_zero != vw_primary_set:
                mismatches.append(
                    f"{part_id}: part_view_priority[0]={sorted(ppp_zero)} "
                    f"vs view_weights.view_weights.primary={sorted(vw_primary_set)}"
                )

    assert not mismatches, (
        "part_view_priority (priority 0) disagrees with view_weights.primary:\n  "
        + "\n  ".join(mismatches)
    )


def test_view_weights_part_ids_subset_of_part_view_priority():
    """view_weights.view_weights.primary should not declare a primary view
    for a part_id whose part_view_priority has NO priority-0 entry for that view.

    Reverse direction: every (part_id, view_id) pair declared as primary by
    view_weights must be flagged as priority 0 (or at least <= 1) by
    part_view_priority.  Priority 1 is acceptable here because the §3.1
    policy says priority <= 1 counts as primary.
    """
    ppp = load_part_view_priority()
    vw = load_view_weights()
    vw_view_weights = vw.get("view_weights", {})

    for part_id, weights in vw_view_weights.items():
        if not isinstance(weights, dict):
            continue
        primary_views = set(weights.get("primary", set()))
        if part_id not in ppp:
            # Allow parts that have view_weights but no part_view_priority entry.
            # The comment in view_weights.yaml notes part_view_priority only
            # covers critical parts; non-critical parts may exist in
            # view_weights alone.
            continue
        part_priorities = ppp[part_id]
        for view_id in primary_views:
            priority = part_priorities.get(view_id)
            assert priority is not None and priority <= 1, (
                f"part_id={part_id}: view_id={view_id} declared primary in "
                f"view_weights but part_view_priority gives priority={priority!r} "
                f"(must be 0 or 1 for primary)"
            )


def test_part_view_priority_keys_are_valid_views():
    """Every view_id appearing in part_view_priority must be a STANDARD_VIEW."""
    ppp = load_part_view_priority()
    standard = set(STANDARD_VIEWS)

    bad = []
    for part_id, priorities in ppp.items():
        for view_id in priorities:
            if view_id not in standard:
                bad.append(f"{part_id}: {view_id}")
    assert not bad, (
        "Unknown view_ids in part_view_priority: " + ", ".join(bad)
    )


def test_load_part_view_priority_returns_dict_of_dicts():
    """Sanity check: loader must return ``{part_id: {view_id: int}}`` shape."""
    ppp = load_part_view_priority()
    assert isinstance(ppp, dict)
    for part_id, views in ppp.items():
        assert isinstance(views, dict), f"{part_id} value is not a dict"
        for view_id, priority in views.items():
            assert isinstance(view_id, str)
            assert isinstance(priority, int)