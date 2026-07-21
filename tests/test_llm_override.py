"""Tests for the per-request LLM provider override plumbing in ``api.views``.

The regression these guard against: ``assess_stream`` used to enter
``_LLMOverride`` in the request thread, then ``return StreamingHttpResponse(
lazy_generator)``.  The ``try/finally`` around it fired at the ``return``
statement — long before Django iterated the generator and the worker thread
made any LLM calls — so the swapped ``config.MINIMAX_API_KEY`` /
``MINIMAX_BASE_URL`` values were restored before a single agent read them.
The UI's key/url therefore never reached the pipeline.

The fix moves the enter/exit into the worker thread that actually runs the
LLM calls.  These tests pin that behaviour.
"""
import os
import threading

import pytest

import config
from api.views import _LLMOverride


def _snapshot():
    return {
        "key": config.MINIMAX_API_KEY,
        "url": config.MINIMAX_BASE_URL,
        "model": config.MINIMAX_MODEL,
        "provider": os.environ.get("LLM_PROVIDER"),
    }


def _restore(snap):
    config.MINIMAX_API_KEY = snap["key"]
    config.MINIMAX_BASE_URL = snap["url"]
    config.MINIMAX_MODEL = snap["model"]
    if snap["provider"] is None:
        os.environ.pop("LLM_PROVIDER", None)
    else:
        os.environ["LLM_PROVIDER"] = snap["provider"]


def test_override_applies_inside_worker_thread_and_restores():
    """The override must be visible to code running in the worker thread that
    runs the LLM calls, and fully restored afterwards."""
    snap = _snapshot()
    try:
        ov = _LLMOverride(
            provider="minimax",
            api_key="sk-cp-OVERRIDE",
            base_url="https://api.minimaxi.com/v1/chat/completions",
            model="OverrideModel",
        )
        seen = {}

        def worker():
            # Mirrors _orchestrator_workflow_sync._bridge: enter inside the
            # worker thread, exit in finally.
            ov.__enter__()
            try:
                seen.update(_snapshot())
            finally:
                ov.__exit__(None, None, None)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert seen["key"] == "sk-cp-OVERRIDE"
        assert seen["url"] == "https://api.minimaxi.com/v1/chat/completions"
        assert seen["model"] == "OverrideModel"
        assert seen["provider"] == "minimax"

        # Restored to the pre-override env values, not the override values.
        after = _snapshot()
        assert after["key"] == snap["key"]
        assert after["url"] == snap["url"]
        assert after["model"] == snap["model"]
        assert after["provider"] == snap["provider"]
    finally:
        _restore(snap)


def test_override_partial_fields_leave_others_untouched():
    """Overriding only the api_key must not clobber base_url / model."""
    snap = _snapshot()
    try:
        ov = _LLMOverride(provider=None, api_key="sk-cp-KEY-ONLY", base_url=None, model=None)
        seen = {}

        def worker():
            ov.__enter__()
            try:
                seen.update(_snapshot())
            finally:
                ov.__exit__(None, None, None)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert seen["key"] == "sk-cp-KEY-ONLY"
        assert seen["url"] == snap["url"]      # untouched
        assert seen["model"] == snap["model"]  # untouched
        assert seen["provider"] == snap["provider"]  # untouched
        assert _snapshot()["key"] == snap["key"]  # restored
    finally:
        _restore(snap)


def test_has_overrides_flags_empty_vs_set():
    assert _LLMOverride().has_overrides() is False
    assert _LLMOverride(provider="", api_key="", base_url="", model="").has_overrides() is False
    assert _LLMOverride(api_key="k").has_overrides() is True
    assert _LLMOverride(provider="openai").has_overrides() is True


def test_from_request_reads_query_params():
    class _Req:
        GET = {
            "provider": "Anthropic",
            "api_key": "  sk-ant-x  ",
            "base_url": "https://api.anthropic.com/v1/messages",
            "model": "claude-x",
        }

    ov = _LLMOverride.from_request(_Req())
    assert ov._provider == "anthropic"      # lowercased
    assert ov._api_key == "sk-ant-x"        # stripped
    assert ov._base_url == "https://api.anthropic.com/v1/messages"
    assert ov._model == "claude-x"


# ── URL/provider mismatch guard ──────────────────────────────────────────────

def test_url_matches_provider_flags_minimax_with_anthropic_path():
    from api.views import _url_matches_provider
    # The exact 404 the user hit: minimax provider + cached /anthropic URL.
    assert _url_matches_provider("minimax", "https://api.minimaxi.com/anthropic") is False
    assert _url_matches_provider("openai", "https://x.test/v1/messages") is False
    assert _url_matches_provider("anthropic", "https://x.test/v1/chat/completions") is False


def test_url_matches_provider_accepts_consistent_and_custom():
    from api.views import _url_matches_provider
    assert _url_matches_provider("minimax", "https://api.minimaxi.com/v1/chat/completions") is True
    assert _url_matches_provider("anthropic", "https://api.anthropic.com/v1/messages") is True
    # Empty url / provider → no judgement (server default kicks in).
    assert _url_matches_provider("", "https://api.minimaxi.com/anthropic") is True
    assert _url_matches_provider("minimax", "") is True
    # A custom gateway host we can't classify is left alone.
    assert _url_matches_provider("minimax", "https://my-gateway.internal/llm") is True


def test_enter_drops_mismatched_base_url_and_keeps_env_default():
    """A stale mismatched base_url must NOT overwrite config.MINIMAX_BASE_URL;
    the env default stays so the request goes to the right endpoint."""
    snap = _snapshot()
    try:
        ov = _LLMOverride(provider="minimax", base_url="https://api.minimaxi.com/anthropic")
        ov.__enter__()
        try:
            # base_url was dropped — config still holds the env/default URL.
            assert config.MINIMAX_BASE_URL == snap["url"]
            assert os.environ.get("LLM_PROVIDER") == "minimax"
        finally:
            ov.__exit__(None, None, None)
    finally:
        _restore(snap)


def test_enter_applies_matching_base_url():
    snap = _snapshot()
    try:
        good = "https://api.openai.com/v1/chat/completions"
        ov = _LLMOverride(provider="openai", base_url=good)
        ov.__enter__()
        try:
            assert config.MINIMAX_BASE_URL == good
        finally:
            ov.__exit__(None, None, None)
        assert config.MINIMAX_BASE_URL == snap["url"]
    finally:
        _restore(snap)
