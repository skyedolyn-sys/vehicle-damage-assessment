"""Simple file-based cache for YAML rule configs.

The cache is keyed by (absolute_path, mtime) so edits on disk invalidate it
automatically.  It is implemented as a module-level singleton so agents
running in the same process share loaded configs.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional


class CachedConfig(NamedTuple):
    """A cached config document plus the mtime used to validate freshness."""

    mtime: float
    data: Dict[str, Any]


class LRUCache:
    """Minimal thread-unsafe LRU cache sufficient for config loading."""

    def __init__(self, capacity: int = 32):
        self._capacity = capacity
        self._data: OrderedDict[tuple[str, float], CachedConfig] = OrderedDict()

    def get(self, key: tuple[str, float]) -> Optional[CachedConfig]:
        value = self._data.get(key)
        if value is None:
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: tuple[str, float], value: CachedConfig) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


#: Module-level singleton cache used by :func:`load_with_cache`.
_GLOBAL_CACHE = LRUCache(capacity=32)


def get_cache() -> LRUCache:
    """Return the global config cache."""
    return _GLOBAL_CACHE


def load_with_cache(path: Path, cache: Optional[LRUCache] = None) -> Dict[str, Any]:
    """Load a YAML file, using the cache when the mtime has not changed.

    Parameters
    ----------
    path:
        Absolute path to the YAML file.
    cache:
        Optional cache instance. Defaults to the module-level singleton.

    Returns
    -------
    dict
        Parsed YAML content. Empty dict if the file does not exist.
    """
    import yaml

    if cache is None:
        cache = _GLOBAL_CACHE

    abs_path = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return {}

    key = (abs_path, mtime)
    cached = cache.get(key)
    if cached is not None:
        return cached.data

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    cache.set(key, CachedConfig(mtime=mtime, data=data))
    return data
