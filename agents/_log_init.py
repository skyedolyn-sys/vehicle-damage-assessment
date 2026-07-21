"""Centralized file-logger setup for agent modules.

Replaces the 8 ad-hoc module-level FileHandler blocks that wrote to
``~/vehicle_damage_assessment_<module>.log``.  Three problems the old code
had:

1. **Crashes on read-only / unset ``$HOME``** — the FileHandler was
   constructed at module import time, so any sandboxed or containerized
   deployment without a writable home dir would crash on first import.
2. **Duplicate handlers on Django autoreload** — ``runserver --reload``
   re-imports every module on file change, so each reload appended a
   fresh FileHandler.  The log file accumulated N copies of every line.
3. **Inconsistent log locations** — 8 files directly in ``$HOME`` plus
   ``~/vehicle_damage_assessment_logs/policy_conflicts.log`` from
   ``_policy_logger.py``.  A grep for "where are the logs" returned 9
   results.

This module centralizes the choice of log dir to ``<BASE_DIR>/logs`` (or
whatever the ``VEHICLE_DAMAGE_LOG_DIR`` env var points at), uses an
existence + baseFilename check to dedupe handlers, and uses ``mode="a"``
so re-imports never truncate the file.

The companion change in ``minimax_client.py`` also routes HTTPS through
the certifi bundle so we no longer need to disable SSL verification just
to talk to the LLM provider from macOS.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


def _log_dir() -> Path:
    """Resolve the directory agent file-logs go to.

    Order of precedence:
      1. ``VEHICLE_DAMAGE_LOG_DIR`` env var (set this in production)
      2. ``<BASE_DIR>/logs`` from Django settings when available
      3. ``./logs`` (cwd-relative) as a last-resort fallback that always
         works in containers where neither $HOME nor Django settings is
         available at import time.
    """
    env = os.environ.get("VEHICLE_DAMAGE_LOG_DIR")
    if env:
        return Path(env)
    try:
        from django.conf import settings
        base = getattr(settings, "BASE_DIR", None)
        if base is not None:
            return Path(base) / "logs"
    except Exception:
        # Django not yet configured (early import, management command, etc.)
        pass
    return Path.cwd() / "logs"


_LOG_DIR = _log_dir()
try:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # If we cannot even create the dir (read-only fs), fall back silently.
    # The agent will still log to console via Django's root handler.
    _LOG_DIR = Path(os.devnull)  # type: ignore[assignment]


def attach_file_handler(
    logger: logging.Logger,
    filename: str,
    level: int = logging.INFO,
) -> Optional[logging.FileHandler]:
    """Attach a deduplicated ``FileHandler`` to ``logger``.

    Idempotent: if a ``FileHandler`` for the same base filename is already
    attached (e.g. after Django autoreload re-imports the module), this is
    a no-op.  Uses ``mode="a"`` so existing log content survives reloads.

    Returns the handler if attached, ``None`` if the log dir is unwritable
    and the handler was skipped.
    """
    if str(_LOG_DIR) == os.devnull:
        return None
    target = _LOG_DIR / filename
    if any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "") == str(target)
        for h in logger.handlers
    ):
        return None
    handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.setLevel(level)
    logger.addHandler(handler)
    return handler
