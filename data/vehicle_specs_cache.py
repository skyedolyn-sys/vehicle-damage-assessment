"""Persistent cache for vehicle specs.

Primary storage is the ``VehicleSpec`` Django ORM model.  The legacy JSON file
at ``data/vehicle_specs_cache.json`` is kept as a fallback and as a portable
fixture that can be reloaded with ``loaddata`` or the ``seed_vehicle_specs``
management command.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from models.vehicle_specs import VehicleSpecs


CACHE_DIR = Path(__file__).parent
CACHE_FILE = CACHE_DIR / "vehicle_specs_cache.json"


def _ensure_data_dir() -> None:
    """Create the data/ directory if it does not exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def make_cache_key(vehicle_info: Dict[str, Any]) -> str:
    """Build a deterministic cache key from brand + model + year (lowercased).

    Joined by ``|`` to avoid collisions with brand/model names containing colons.
    """
    brand = str(vehicle_info.get("brand", "")).lower().strip()
    model = str(vehicle_info.get("model", "")).lower().strip()
    year = str(vehicle_info.get("year", "")).lower().strip()
    return f"{brand}|{model}|{year}"


def is_cacheable_key(vehicle_info: Dict[str, Any]) -> bool:
    """Return True only when brand AND model are both present.

    An empty/blank vehicle_info collapses to the shared ``"||"`` bucket.  Caching
    under that bucket is unsafe: any default-sedan write (or a mis-detected
    vehicle) would be served back to *every* subsequent vehicle-info-less sample,
    silently poisoning unrelated cars (e.g. a Mercedes GLC read as a sedan).

    When brand or model is missing we bypass the cache entirely and let the LLM
    infer specs fresh each time — no shared bucket, no cross-sample pollution.
    """
    brand = str(vehicle_info.get("brand", "")).strip()
    model = str(vehicle_info.get("model", "")).strip()
    return bool(brand) and bool(model)


def _load_json_cache() -> Dict[str, Any]:
    """Read the legacy cache file into a dict."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json_cache(data: Dict[str, Any]) -> None:
    """Write cache dict to disk atomically (temp file + rename)."""
    _ensure_data_dir()
    fd, temp_path = tempfile.mkstemp(dir=str(CACHE_DIR), suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, str(CACHE_FILE))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _get_orm_specs(vehicle_info: Dict[str, Any]) -> Optional[VehicleSpecs]:
    """Try to read specs from the Django ORM if available."""
    try:
        # Local import so the module stays usable outside Django runtime.
        from api.models import VehicleSpec
    except Exception:
        return None

    key = make_cache_key(vehicle_info)
    try:
        spec = VehicleSpec.objects.get(cache_key=key)
        return VehicleSpecs.from_dict(spec.to_dict())
    except VehicleSpec.DoesNotExist:
        return None


def get_cached_specs(vehicle_info: Dict[str, Any]) -> Optional[VehicleSpecs]:
    """Return cached VehicleSpecs for *vehicle_info*, or None if not cached.

    Tries the Django ORM first, then falls back to the legacy JSON file.
    Refuses to read the shared empty (``"||"``) bucket — see is_cacheable_key.
    """
    if not is_cacheable_key(vehicle_info):
        return None

    orm_specs = _get_orm_specs(vehicle_info)
    if orm_specs is not None:
        return orm_specs

    key = make_cache_key(vehicle_info)
    cache = _load_json_cache()
    raw = cache.get(key)
    if raw is None:
        return None
    try:
        return VehicleSpecs.from_dict(raw)
    except (TypeError, ValueError, KeyError):
        return None


def save_cached_specs(vehicle_info: Dict[str, Any], specs: VehicleSpecs) -> None:
    """Store *specs* in the persistent cache keyed by *vehicle_info*.

    Saves to both the Django ORM (if available) and the legacy JSON file.
    Refuses to write the shared empty (``"||"``) bucket — see is_cacheable_key.
    """
    if not is_cacheable_key(vehicle_info):
        return

    key = make_cache_key(vehicle_info)
    specs_dict = specs.to_dict()

    # Primary: Django ORM
    try:
        from api.models import VehicleSpec

        VehicleSpec.objects.update_or_create(
            cache_key=key,
            defaults={
                "brand": str(vehicle_info.get("brand", "")).strip(),
                "model": str(vehicle_info.get("model", "")).strip(),
                "year": str(vehicle_info.get("year", "")).strip(),
                "body_style": specs_dict.get("body_style", ""),
                "doors": specs_dict.get("doors", 4),
                "has_sunroof": bool(specs_dict.get("has_sunroof", False)),
                "has_roof_rack": bool(specs_dict.get("has_roof_rack", False)),
                "headlight_layout": specs_dict.get("headlight_layout", ""),
                "rear_door_type": specs_dict.get("rear_door_type", ""),
                "notes": specs_dict.get("notes", ""),
            },
        )
    except Exception:
        # ORM may not be available outside Django runtime; continue to JSON.
        pass

    # Fallback: legacy JSON file
    cache = _load_json_cache()
    cache[key] = specs_dict
    _save_json_cache(cache)
