"""Tests for ``agents.llm_client``.

These exercise the real ``call_llm`` code path with the HTTP layer mocked.
They exist because the module shipped with ``aiohttp.ClientConnector`` — an
attribute that does not exist (the real class is ``TCPConnector``) — and it
only surfaced during a live E2E run.  Mocking ``aiohttp.ClientSession`` lets
us drive ``call_llm`` to completion without any network, so a typo in the
connector/session construction fails fast in CI instead of in the browser.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agents.llm_client as llm


# ── _normalize_base_url ──────────────────────────────────────────────────────

def test_normalize_appends_chat_completions_to_sdk_style_base():
    # MiniMax's documented OpenAI-compat base URL is the bare host path.
    assert (
        llm._normalize_base_url("openai", "https://api.minimaxi.com/v1")
        == "https://api.minimaxi.com/v1/chat/completions"
    )
    assert (
        llm._normalize_base_url("minimax", "https://api.minimaxi.com/v1/")
        == "https://api.minimaxi.com/v1/chat/completions"
    )


def test_normalize_appends_messages_for_anthropic():
    # MiniMax's Anthropic-compat base URL is the bare /anthropic host path;
    # the real terminal is /anthropic/v1/messages (verified by probing).
    assert (
        llm._normalize_base_url("anthropic", "https://api.minimaxi.com/anthropic")
        == "https://api.minimaxi.com/anthropic/v1/messages"
    )
    assert (
        llm._normalize_base_url("anthropic", "https://api.anthropic.com/v1")
        == "https://api.anthropic.com/v1/messages"
    )


def test_normalize_leaves_full_endpoint_untouched():
    full = "https://api.minimaxi.com/v1/chat/completions"
    assert llm._normalize_base_url("minimax", full) == full
    full_msg = "https://api.anthropic.com/v1/messages"
    assert llm._normalize_base_url("anthropic", full_msg) == full_msg
    # The MiniMax Anthropic-compat full endpoint we discovered by probing.
    full_mm = "https://api.minimaxi.com/anthropic/v1/messages"
    assert llm._normalize_base_url("anthropic", full_mm) == full_mm


def test_normalize_handles_empty():
    assert llm._normalize_base_url("minimax", "") == ""
    assert llm._normalize_base_url("openai", "   ") == ""


def test_from_overrides_normalizes_sdk_style_base_url():
    """End-to-end at the resolution layer: a documented SDK base URL becomes a
    callable endpoint after from_overrides."""
    cfg = llm.LLMConfig.from_overrides(
        provider="openai",
        api_key="sk-x",
        base_url="https://api.minimaxi.com/v1",
        model="MiniMax-M3",
    )
    assert cfg.base_url == "https://api.minimaxi.com/v1/chat/completions"

    cfg2 = llm.LLMConfig.from_overrides(
        provider="anthropic",
        api_key="sk-x",
        base_url="https://api.minimaxi.com/anthropic",
        model="claude-x",
    )
    assert cfg2.base_url == "https://api.minimaxi.com/anthropic/v1/messages"

    # The full Anthropic endpoint passes through untouched.
    cfg3 = llm.LLMConfig.from_overrides(
        provider="anthropic",
        api_key="sk-x",
        base_url="https://api.minimaxi.com/anthropic/v1/messages",
        model="claude-x",
    )
    assert cfg3.base_url == "https://api.minimaxi.com/anthropic/v1/messages"


def _make_cfg(provider):
    return llm.LLMConfig(
        provider=provider,
        api_key="sk-test",
        base_url="https://example.test/v1/chat",
        model="test-model",
    )


def _mock_session(payload):
    """Build an aiohttp.ClientSession mock whose post() returns ``payload``."""
    resp = MagicMock()
    resp.status = 200
    resp.json = AsyncMock(return_value=payload)
    resp.text = AsyncMock(return_value="OK")
    # async context manager on the response
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=resp)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.mark.asyncio
async def test_call_llm_openai_compatible_roundtrip():
    session = _mock_session(
        {"choices": [{"message": {"content": '{"ok": true}'}}]}
    )
    with patch.object(llm.aiohttp, "ClientSession", return_value=session), patch.object(
        llm.aiohttp, "TCPConnector", return_value=MagicMock()
    ):
        out = await llm.call_llm(
            [{"role": "user", "content": "hi"}], config=_make_cfg(llm.PROVIDER_OPENAI)
        )
    assert out == '{"ok": true}'
    session.post.assert_called_once()


@pytest.mark.asyncio
async def test_call_llm_anthropic_roundtrip():
    session = _mock_session(
        {"content": [{"type": "text", "text": '{"ok": true}'}]}
    )
    with patch.object(llm.aiohttp, "ClientSession", return_value=session), patch.object(
        llm.aiohttp, "TCPConnector", return_value=MagicMock()
    ):
        out = await llm.call_llm(
            [{"role": "user", "content": "hi"}], config=_make_cfg(llm.PROVIDER_ANTHROPIC)
        )
    assert out == '{"ok": true}'
    # Anthropic path must send the x-api-key header, not Bearer.
    _, kwargs = session.post.call_args
    assert kwargs["headers"]["x-api-key"] == "sk-test"
    assert "Authorization" not in kwargs["headers"]


@pytest.mark.asyncio
async def test_call_llm_http_error_raises_with_context():
    session = _mock_session({})
    session.post.return_value.status = 401
    session.post.return_value.text = AsyncMock(return_value='{"error":"login fail"}')
    with patch.object(llm.aiohttp, "ClientSession", return_value=session), patch.object(
        llm.aiohttp, "TCPConnector", return_value=MagicMock()
    ):
        with pytest.raises(RuntimeError) as exc:
            await llm.call_llm(
                [{"role": "user", "content": "hi"}], config=_make_cfg(llm.PROVIDER_OPENAI)
            )
    msg = str(exc.value)
    assert "401" in msg
    assert "openai_compatible" in msg
    assert "https://example.test/v1/chat" in msg
