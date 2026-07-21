"""MCP-based image understanding client.

This module wraps the ``minimax-coding-plan-mcp`` server's
``understand_image`` tool.  The MCP server is expected to be running
as a separate process (e.g. started via ``uvx minimax-coding-plan-mcp``).
It exposes the two tools ``web_search`` and ``understand_image`` over an
HTTP/SSE transport.

For our purposes the only relevant tool is ``understand_image``:

    parameters:
      prompt (string, required) — what to ask about the image
      image_url (string, required) — HTTP/HTTPS URL or local file path

It returns the model's analysis as a plain text response.

If the MCP server is not configured (no ``MCP_MINIMAX_URL`` env var)
this module falls back to the existing chat-completions image path
in :mod:`agents.minimax_client`.  That way the dev server keeps working
in environments where the MCP server is not installed yet.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

# IMPORTANT: do not `from config import MINIMAX_*` here.  The LLM provider
# override (api.views._LLMOverride) mutates config.MINIMAX_* at request
# time; reading through the `config` namespace on each call picks the
# fresh values.
import config

logger = logging.getLogger(__name__)

# Where the MCP server exposes its JSON-RPC endpoint.  When unset
# (i.e. minimax-coding-plan-mcp has not been installed) the client
# falls back to the chat-completions image path.
MCP_MINIMAX_URL = os.environ.get("MCP_MINIMAX_URL", "").strip()
MCP_DEFAULT_TIMEOUT = float(os.environ.get("MCP_TIMEOUT_SEC", "120"))


async def understand_image(
    image_path: str,
    prompt: str,
    *,
    timeout: float = MCP_DEFAULT_TIMEOUT,
) -> str:
    """Ask the model to analyse ``image_path`` according to ``prompt``.

    Returns
    -------
    str
        The model's text response.  Callers are expected to parse this
        as JSON (the planner agent, for instance, asks the model to
        return a JSON object).

    Behaviour
    ---------
    - If ``MCP_MINIMAX_URL`` is set, sends a JSON-RPC ``tools/call``
      request to the configured ``minimax-coding-plan-mcp`` endpoint.
    - Otherwise falls back to the chat-completions image path
      (single user message with image + text, expecting a JSON
      response, same as the existing planner flow).
    """
    if MCP_MINIMAX_URL:
        return await _understand_image_via_mcp(image_path, prompt, timeout=timeout)
    return await _understand_image_via_chat(image_path, prompt, timeout=timeout)


async def _understand_image_via_mcp(
    image_path: str,
    prompt: str,
    *,
    timeout: float,
) -> str:
    """Call ``understand_image`` over HTTP/JSON-RPC on the MCP server.

    The MCP HTTP transport expects a single JSON object per request::

        {"jsonrpc": "2.0", "id": <int>, "method": "tools/call",
         "params": {"name": "understand_image",
                    "arguments": {"prompt": ..., "image_url": ...}}}

    and returns::

        {"jsonrpc": "2.0", "id": <int>,
         "result": {"content": [{"type": "text", "text": "..."}]}}

    The ``text`` field is what we return to callers.
    """
    import aiohttp
    import json
    import time

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if config.MINIMAX_API_KEY:
        headers["Authorization"] = f"Bearer {config.MINIMAX_API_KEY}"

    payload = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000) % 100000,
        "method": "tools/call",
        "params": {
            "name": "understand_image",
            "arguments": {
                "prompt": prompt,
                "image_url": image_path,
            },
        },
    }

    call_id = payload["id"]
    logger.info(
        "[mcp_client] call id=%s endpoint=%s prompt_len=%d",
        call_id, MCP_MINIMAX_URL, len(prompt),
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                MCP_MINIMAX_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"MCP server {MCP_MINIMAX_URL} returned {resp.status}: {text[:500]}"
                    )
                raw = await resp.json()
    except Exception as exc:
        logger.warning("[mcp_client] id=%s MCP call failed: %s; falling back to chat", call_id, exc)
        return await _understand_image_via_chat(image_path, prompt, timeout=timeout)

    if "error" in raw and "result" not in raw:
        logger.warning(
            "[mcp_client] id=%s MCP returned error: %s; falling back to chat",
            call_id, raw.get("error"),
        )
        return await _understand_image_via_chat(image_path, prompt, timeout=timeout)

    result = raw.get("result", {})
    content = result.get("content", []) if isinstance(result, dict) else []
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    text_out = "\n".join(parts).strip()
    if not text_out:
        logger.warning(
            "[mcp_client] id=%s MCP returned empty content (raw=%s); falling back to chat",
            call_id, json.dumps(raw)[:300],
        )
        return await _understand_image_via_chat(image_path, prompt, timeout=timeout)
    logger.info("[mcp_client] id=%s success text_len=%d", call_id, len(text_out))
    return text_out


async def _understand_image_via_chat(
    image_path: str,
    prompt: str,
    *,
    timeout: float,
) -> str:
    """Fallback path: call the chat-completions endpoint with the image.

    The current ``call_minimax`` chat interface accepts a multi-part
    user message containing both the text prompt and a base64-encoded
    image.  This is the path planner_agent used before the MCP
    ``understand_image`` tool was wired in; it remains the fallback
    when no MCP server is configured.
    """
    from agents.minimax_client import build_image_content, call_minimax, extract_json

    image_content = build_image_content(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_content,
            ],
        },
    ]
    raw_text = await call_minimax(
        messages=messages,
        temperature=0.0,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )
    parsed = extract_json(raw_text)
    if isinstance(parsed, dict):
        # Some chat backends return a JSON object directly; the
        # ``understand_image`` tool returns plain text, so we re-pack it
        # in the same shape so downstream callers behave the same.
        import json
        return json.dumps(parsed, ensure_ascii=False)
    return raw_text


def is_mcp_configured() -> bool:
    """Return True if the MCP server URL is configured."""
    return bool(MCP_MINIMAX_URL)
