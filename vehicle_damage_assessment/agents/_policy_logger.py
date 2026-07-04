"""DAMAGE_RECOGNITION_POLICY §4.3 — structured policy conflict logging.

Logs every fusion/synthesizer/topology rule that flips a final conclusion
against ≥2 secondary view signals to a dedicated file under
``~/vehicle_damage_assessment_logs/policy_conflicts.log``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable


_LOG_DIR = Path(os.path.expanduser("~/vehicle_damage_assessment_logs"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger("policy_conflicts")
# Avoid duplicate handlers if module is reloaded.
if not any(
    isinstance(h, logging.FileHandler)
    and getattr(h, "baseFilename", "").endswith("policy_conflicts.log")
    for h in _logger.handlers
):
    _handler = logging.FileHandler(
        _LOG_DIR / "policy_conflicts.log", mode="a", encoding="utf-8"
    )
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)


def log_policy_conflict(
    part_id: str,
    final_status: str,
    conflict_sources: Iterable[str],
    rule_applied: str,
) -> None:
    """Log a policy conflict per DAMAGE_RECOGNITION_POLICY §4.3.

    Args:
        part_id: The component identifier.
        final_status: The status the synthesizer/topology finally chose.
        conflict_sources: Iterable of part_ids or view_ids that disagreed.
        rule_applied: Tag identifying which rule did the flip (e.g. ``adjacency_rule_6``).
    """
    _logger.info(
        "part_id=%s final_status=%s conflict_sources=%s rule=%s",
        part_id,
        final_status,
        list(conflict_sources),
        rule_applied,
    )