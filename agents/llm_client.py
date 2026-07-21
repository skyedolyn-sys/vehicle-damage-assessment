"""LLM provider abstraction — MiniMax / OpenAI-compatible / Anthropic-compatible.

The face-path pipeline only needs ``call_llm(messages, ...)``; the original
``agents.minimax_client.call_minimax`` was hard-wired to the MiniMax
``/chat/completions`` endpoint with the MiniMax request envelope (which is
OpenAI-compatible anyway).  This module adds a third-party switch so a
developer can test the pipeline against OpenAI or Anthropic endpoints
without editing any agent code:

* ``provider="minimax"`` (default) — POST /chat/completions with Bearer
  auth, MiniMax base URL.
* ``provider="openai"`` — same wire format but a custom base URL.
* ``provider="anthropic"`` — POST /messages with ``x-api-key`` auth and
  the Anthropic ``messages`` request envelope.

The provider / base_url / model / api_key are taken from the
``_LLMOverride`` context manager in ``api.views``.  Outside a request
(e.g. management commands, tests) we fall back to the env defaults
loaded by ``config.py``.

Why this layer exists
---------------------
Originally a user changing providers would have to edit ``.env`` and
restart the server.  The new UI lets them paste a key + base URL + model
in the browser, forward them as query params, and the server swaps the
entire transport for the duration of one SSE response.  This module is
the seam that makes that swap possible without scattering ``if provider
== 'anthropic'`` checks across every agent.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import certifi
import ssl

logger = logging.getLogger(__name__)


#: Provider identifiers accepted by ``call_llm``.
PROVIDER_MINIMAX = "minimax"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
_VALID_PROVIDERS = {PROVIDER_MINIMAX, PROVIDER_OPENAI, PROVIDER_ANTHROPIC}


@dataclass
class LLMConfig:
    """Resolved LLM endpoint configuration for one request."""

    provider: str
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_overrides(
        cls,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> "LLMConfig":
        """Build an LLMConfig from per-request overrides, falling back to env.

        Resolution order for each field:
            1. explicit override (per-request)
            2. env var
            3. hardcoded MiniMax default (for base_url/model only)
        """
        # Lazy import: this module is imported by agents at startup, but
        # config.py itself loads dotenv + raises on missing key.  Pulling
        # config lazily avoids the chicken-and-egg on first import.
        import config as _config

        resolved_provider = (provider or "").strip().lower() or _detect_provider_from_env()
        if resolved_provider not in _VALID_PROVIDERS:
            logger.warning(
                "[llm] unknown provider %r — falling back to minimax",
                resolved_provider,
            )
            resolved_provider = PROVIDER_MINIMAX

        resolved_base_url = _normalize_base_url(
            resolved_provider,
            (base_url or "").strip() or _default_base_url(resolved_provider, _config),
        )
        return cls(
            provider=resolved_provider,
            api_key=(api_key or "").strip() or _config.MINIMAX_API_KEY,
            base_url=resolved_base_url,
            model=(model or "").strip() or _default_model(resolved_provider, _config),
        )


def _normalize_base_url(provider: str, base_url: str) -> str:
    """Ensure ``base_url`` is a fully-qualified endpoint for ``provider``.

    Why this exists: every official SDK / vendor doc hands users a *base URL*
    without the terminal path, and lets the SDK append it.  MiniMax's own
    quickstart says ``OPENAI_BASE_URL=https://api.minimaxi.com/v1`` and
    ``ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic``.  But this client
    POSTs to ``base_url`` verbatim, so a user pasting the documented SDK value
    (``.../v1``) would hit ``404 page not found``.

    The rule depends on the provider, because MiniMax routes protocols by
    host path (``/v1`` = OpenAI wire, ``/anthropic/v1/...`` = Anthropic wire).
    Discovered by probing the live service — do not change without re-probing:

    * ``openai`` / ``minimax`` (OpenAI wire): terminal path ``/chat/completions``.
      ``https://api.minimaxi.com/v1`` → ``.../v1/chat/completions``.
      ``https://api.openai.com`` → ``https://api.openai.com/chat/completions``.
    * ``anthropic`` (Anthropic wire): terminal path ``/v1/messages`` under the
      ``/anthropic`` sub-host on MiniMax, or under the official ``/v1`` path
      on Anthropic.  ``https://api.minimaxi.com/anthropic`` →
      ``.../anthropic/v1/messages``; ``https://api.anthropic.com/v1`` →
      ``.../v1/messages`` (no double ``/v1``).  (NOT ``.../anthropic/messages``
      — that 404s on MiniMax.)

    The rule accepts "SDK-style base URL" and "full endpoint URL" both,
    regardless of which the user pastes.  Cross-protocol host mismatch is
    caught separately in ``api.views._url_matches_provider``.
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return url
    if provider == PROVIDER_ANTHROPIC:
        terminal = "/v1/messages"
        # Avoid double /v1 when the user already gave us .../v1 (e.g. the
        # official Anthropic SDK style).  Skip the /v1 segment only when the
        # base URL doesn't already have its own protocol-prefix sub-path.
        if url.endswith("/v1") and not url.endswith("/anthropic/v1"):
            return url + "/messages"
        if url.lower().endswith(terminal):
            return url
        return url + terminal
    # minimax + openai both speak the OpenAI chat-completions wire format.
    if url.lower().endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def _detect_provider_from_env() -> str:
    """Best-effort provider detection when no override is given.

    Reads ``LLM_PROVIDER`` env var; defaults to ``minimax`` so existing
    deployments keep working unchanged.
    """
    env = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    return env if env in _VALID_PROVIDERS else PROVIDER_MINIMAX


def _default_base_url(provider: str, cfg) -> str:
    """Default base URL when the caller does not override it."""
    env = os.environ.get("LLM_BASE_URL")
    if env:
        return env
    if provider == PROVIDER_ANTHROPIC:
        return "https://api.anthropic.com/v1/messages"
    # MiniMax and OpenAI-compatible share the OpenAI wire format.
    return cfg.MINIMAX_BASE_URL


def _default_model(provider: str, cfg) -> str:
    env = os.environ.get("LLM_MODEL")
    if env:
        return env
    if provider == PROVIDER_ANTHROPIC:
        return "claude-3-5-sonnet-latest"
    return cfg.MINIMAX_MODEL


#: Shared TLS context that verifies certificates through certifi's bundle.
#: Created once at import.  Falls back to system defaults if certifi is
#: unavailable (rare — only happens in minimal containers).
try:
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    _SSL_CONTEXT = ssl.create_default_context()


async def call_llm(
    messages: List[Dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 8000,
    reasoning_effort: Optional[str] = "low",
    response_format: Optional[Dict[str, str]] = None,
    config: Optional[LLMConfig] = None,
    timeout: int = 300,
) -> str:
    """Call the configured LLM provider and return the assistant text.

    The agent layer only consumes the returned text; it does not need to
    know which provider answered.  ``response_format={"type": "json_object"}``
    is honoured on OpenAI-compatible endpoints and silently ignored on
    Anthropic (Anthropic uses a different prompt-side enforcement path).
    """
    cfg = config or LLMConfig.from_overrides()
    if cfg.provider == PROVIDER_ANTHROPIC:
        return await _call_anthropic(
            cfg, messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout
        )
    return await _call_openai_compatible(
        cfg, messages, temperature=temperature, max_tokens=max_tokens,
        reasoning_effort=reasoning_effort, response_format=response_format,
        timeout=timeout,
    )


async def _call_openai_compatible(
    cfg: LLMConfig,
    messages: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    response_format: Optional[Dict[str, str]],
    timeout: int,
) -> str:
    """POST ``/chat/completions`` against any OpenAI-compatible endpoint.

    Used for both MiniMax (default base URL) and OpenAI itself.
    """
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    connector = aiohttp.TCPConnector(ssl=_SSL_CONTEXT)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj) as session:
        async with session.post(
            cfg.base_url, json=payload, headers=headers
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"LLM error {resp.status} from openai_compatible at "
                    f"{cfg.base_url} (auth=Bearer, model={cfg.model}): {text}"
                )
            data = await resp.json()
    return _extract_openai_text(data)


async def _call_anthropic(
    cfg: LLMConfig,
    messages: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    """POST ``/messages`` against an Anthropic-compatible endpoint.

    Anthropic uses a different request envelope: ``system`` lives outside
    the ``messages`` array, and the user/assistant turn list is the body.
    We translate the OpenAI-style ``messages`` list into the Anthropic
    shape here so the agent code stays provider-agnostic.
    """
    system_texts: List[str] = []
    anthropic_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            # Anthropic allows multiple system blocks; concatenate text.
            if isinstance(content, str):
                system_texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_texts.append(block.get("text", ""))
            continue
        if role in ("user", "assistant"):
            if isinstance(content, list):
                # Flatten multi-part user content to the first text block;
                # the face-path agents always send a single string here.
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(p for p in text_parts if p)
            anthropic_messages.append({"role": role, "content": content or ""})

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": anthropic_messages,
    }
    if system_texts:
        payload["system"] = "\n\n".join(system_texts)

    headers = {
        "x-api-key": cfg.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    connector = aiohttp.TCPConnector(ssl=_SSL_CONTEXT)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj) as session:
        async with session.post(
            cfg.base_url, json=payload, headers=headers
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"LLM error {resp.status} from anthropic at "
                    f"{cfg.base_url} (auth=x-api-key, model={cfg.model}): {text}"
                )
            data = await resp.json()
    return _extract_anthropic_text(data)


def _extract_openai_text(data: Dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenAI-shaped response.

    Defensive against partial responses (some providers omit ``finish_reason``
    or nest ``content`` differently).  Returns an empty string if the body
    has no text — callers detect that via the same code path as a JSON
    parse failure.
    """
    try:
        choices = data.get("choices") or []
        if not choices:
            return ""
        first = choices[0]
        # Some providers put content directly on the choice; others nest
        # under ``message.content``.  Handle both.
        if "message" in first and isinstance(first["message"], dict):
            return first["message"].get("content") or ""
        return first.get("text") or ""
    except (AttributeError, TypeError, KeyError):
        logger.warning("[llm] unexpected OpenAI-shaped response: %s", json.dumps(data)[:200])
        return ""


def _extract_anthropic_text(data: Dict[str, Any]) -> str:
    """Pull the assistant text out of an Anthropic-shaped response."""
    try:
        blocks = data.get("content") or []
        text_parts = [
            b.get("text", "") for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in text_parts if p)
    except (AttributeError, TypeError, KeyError):
        logger.warning("[llm] unexpected Anthropic-shaped response: %s", json.dumps(data)[:200])
        return ""