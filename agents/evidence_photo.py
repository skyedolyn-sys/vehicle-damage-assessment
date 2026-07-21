"""Canonical evidence_photo normalisation.

There are three shapes the field takes across the codebase:
- LLM JSON output emits a JSON list ``["172852-04.png", "172852-03.png"]``.
- Some LLM prompts produce a CSV string ``"172852-04.png, 172852-03.png"``.
- The :class:`PartActualState` dataclass stores it as ``evidence_photos`` (list).

This module provides a single canonical normaliser. Always call this rather
than open-coding the split/iter logic at the call site.
"""

from __future__ import annotations

from typing import Any, List


def to_photo_list(raw: Any) -> List[str]:
    """Normalise an evidence_photo-shaped value to a list of strings.

    Returns ``[]`` for None, empty string, or the literal ``"none"``.
    Splits comma-separated strings. Drops empty fragments. Preserves
    insertion order; duplicates are NOT removed (callers that need dedup
    can wrap with ``list(dict.fromkeys(...))``).
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw or raw.strip().lower() == "none":
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(p).strip() for p in raw if p is not None and str(p).strip()]
    return []


def to_csv(raw: Any) -> str:
    """Render a normalised list of photo ids as a CSV string.

    Empty list renders as ``""`` (not ``"none"``); use ``to_photo_list``
    first if you need to handle raw inputs.
    """
    return ", ".join(str(p) for p in raw if p)