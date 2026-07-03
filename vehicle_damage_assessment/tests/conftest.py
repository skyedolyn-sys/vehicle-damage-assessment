"""pytest configuration — set up Django so vehicle_prior cache lookup works.

DAMAGE_RECOGNITION_POLICY §1.6: 回归测试需要跑 assessment_orchestrator →
vehicle_prior_agent → get_cached_specs (Django ORM)。需要先初始化 Django。
"""
import os
import sys

import django
import pytest


@pytest.fixture(scope="session", autouse=True)
def _setup_django():
    """Initialize Django before any test that hits the ORM."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
    try:
        django.setup()
    except Exception:
        # If Django can't initialize (e.g. settings not configured), skip DB tests
        pass