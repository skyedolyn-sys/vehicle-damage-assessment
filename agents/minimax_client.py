import json
import base64
import ssl
import re
import os
import asyncio
import aiohttp
import logging
import time
from typing import List, Tuple, Dict, Any
# IMPORTANT: do not `from config import MINIMAX_*` here.  The LLM provider
# override (api.views._LLMOverride) mutates config.MINIMAX_* at request
# time; if we had imported them as bound names at module load, the agent
# layer would keep reading the OLD values forever.  Instead we read them
# through the `config` namespace on every call so the swap is visible.
import config
from agents.image_utils import compress_image_to_base64
from agents.llm_client import (
    LLMConfig,
    PROVIDER_MINIMAX,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    call_llm,
)

logger = logging.getLogger(__name__)
# Dedicated file log for API diagnostics; Django console log level may swallow INFO.
# Centralized through agents._log_init so Django runserver autoreload does not
# stack duplicate FileHandlers on the same log file.
from agents._log_init import attach_file_handler
attach_file_handler(logger, "minimax.log")
logger.setLevel(logging.INFO)

# Verify TLS certificates via the certifi bundle.  Falls back to the system
# default trust store if certifi is not installed (rare — only in minimal
# containers where the user can install it explicitly).
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    SSL_CONTEXT = ssl.create_default_context()


import random

async def call_minimax(
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 4000,
    model: str | None = None,
    response_format: Dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    max_escalation_tokens: int | None = None,
) -> str:
    """Call the active LLM provider and return the assistant text.

    For ``LLM_PROVIDER=minimax`` (default) this uses the carefully tuned
    retry + token-escalation loop below — MiniMax M3 sometimes burns the
    whole ``max_tokens`` budget on a single ``<think>...</think>`` block
    (``finish_reason == "length"``), and we escalate to recover.

    For ``LLM_PROVIDER=openai`` or ``anthropic`` we delegate to
    ``agents.llm_client.call_llm``, which speaks the respective native
    protocol without retry.  The MiniMax retry logic is unnecessary on
    OpenAI/Anthropic endpoints (they have their own server-side retry).

    Truncation handling: MiniMax M3 sometimes spends the whole ``max_tokens``
    budget inside a single ``<think>...</think>`` block and emits no JSON body
    (``finish_reason == "length"``).  When that happens we do NOT silently
    return the unparseable narrative — we escalate the token budget and retry
    so the model has room for both reasoning and the JSON answer.
    ``max_escalation_tokens`` caps that escalation (default: 2× ``max_tokens``).
    """
    provider = (os.environ.get("LLM_PROVIDER") or PROVIDER_MINIMAX).strip().lower()
    if provider in (PROVIDER_OPENAI, PROVIDER_ANTHROPIC):
        # Build an LLMConfig from the current request's overrides.  The
        # config module's MINIMAX_API_KEY / MINIMAX_BASE_URL / MINIMAX_MODEL
        # are swapped by api.views._ApiKeyOverride when the UI sends
        # ?api_key=... / ?base_url=... / ?model=... — so this picks them
        # up automatically.
        cfg = LLMConfig(
            provider=provider,
            api_key=config.MINIMAX_API_KEY,
            base_url=config.MINIMAX_BASE_URL,
            model=model or config.MINIMAX_MODEL,
        )
        raw = await call_llm(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            config=cfg,
            timeout=config.REQUEST_TIMEOUT,
        )
        return clean_minimax_output(raw) if raw else raw

    call_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    request_model = model or config.MINIMAX_MODEL
    if max_escalation_tokens is None:
        max_escalation_tokens = max_tokens * 2
    logger.info("[minimax:%s] start model=%s temp=%s max_tokens=%s json_mode=%s reasoning=%s", call_id, request_model, temperature, max_tokens, bool(response_format), reasoning_effort)

    headers = {
        "Authorization": f"Bearer {config.MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    def _build_payload(token_budget: int) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "model": request_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": token_budget,
        }
        if response_format:
            p["response_format"] = response_format
        if reasoning_effort:
            # MiniMax M3 supports reasoning_effort in {"low", "medium", "high"}.
            # Default reasoning is unbounded and burns the full max_tokens
            # budget on a single <think>...</think> block, leaving no room
            # for the actual JSON answer.  Forcing "low" caps the thinking
            # so the model has to commit to a structured output.
            p["reasoning_effort"] = reasoning_effort
        return p

    async def _single_request(token_budget: int) -> tuple[str, str]:
        """One HTTP call. Returns (cleaned_content, finish_reason)."""
        payload = _build_payload(token_budget)
        connector = aiohttp.TCPConnector(limit=1, force_close=True)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                config.MINIMAX_BASE_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT, sock_connect=30),
                ssl=SSL_CONTEXT,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"MiniMax API error {resp.status}: {text}")
                data = await resp.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "")
                cleaned = clean_minimax_output(content)
                return cleaned, finish_reason

    def _has_usable_json(cleaned: str) -> bool:
        return bool(cleaned) and extract_json(cleaned) is not None

    last_exception = None
    token_budget = max_tokens
    for attempt in range(1, 4):
        start = time.perf_counter()
        try:
            cleaned, finish_reason = await _single_request(token_budget)
            elapsed = time.perf_counter() - start
            logger.info("[minimax:%s] attempt=%d success elapsed=%.2fs raw_budget=%d cleaned_len=%d finish=%s", call_id, attempt, elapsed, token_budget, len(cleaned), finish_reason)

            # Truncation: finish_reason == "length" means the model ran out of
            # token budget, so the JSON body is incomplete — even if a small
            # fragment happens to parse.  Always escalate and retry rather than
            # accept a truncated payload; only give up once the budget ceiling
            # is reached.
            if finish_reason == "length" and token_budget < max_escalation_tokens:
                token_budget = min(token_budget * 2, max_escalation_tokens)
                logger.warning("[minimax:%s] attempt=%d truncated (finish=length); escalating max_tokens→%d and retrying", call_id, attempt, token_budget)
                continue

            # Diagnostic: if cleaning stripped usable JSON, dump samples for inspection.
            if cleaned and not extract_json(cleaned):
                diagnostic_path = os.path.expanduser(f"~/minimax_diagnostic_{call_id}.txt")
                try:
                    with open(diagnostic_path, "w", encoding="utf-8") as f:
                        f.write("===== FINISH_REASON =====\n")
                        f.write(str(finish_reason))
                        f.write("\n===== CLEANED =====\n")
                        f.write(cleaned)
                    logger.warning("[minimax:%s] cleaned content not parseable as JSON; diagnostic written to %s", call_id, diagnostic_path)
                except Exception as dump_exc:
                    logger.warning("[minimax:%s] failed to write diagnostic: %s", call_id, dump_exc)
            return cleaned
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError, RuntimeError) as e:
            elapsed = time.perf_counter() - start
            last_exception = e
            logger.warning("[minimax:%s] attempt=%d failed elapsed=%.2fs error=%s", call_id, attempt, elapsed, e)
            if attempt < 3:
                # Exponential backoff with jitter: 2s, 7s, 15s base plus up to 3s jitter.
                base = [2, 7, 15][attempt - 1]
                wait = base + random.uniform(0, 3)
                logger.info("[minimax:%s] sleeping %.2fs before retry", call_id, wait)
                await asyncio.sleep(wait)
            continue

    logger.error("[minimax:%s] all 3 attempts failed: %s", call_id, last_exception)
    raise RuntimeError(f"MiniMax API failed after 3 attempts: {last_exception}") from last_exception


def clean_minimax_output(text: str) -> str:
    """清洗 MiniMax 模型输出,提取 JSON。

    优先级:
    1. 尝试整段作为 JSON(无 think 标签时)
    2. think 标签外的 JSON
    3. think 标签内嵌的 JSON
    4. narrative 中嵌入的 JSON(栈匹配找所有 { } 块)
    5. 全部失败,返回 cleaned narrative(走 fallback)
    """
    if not text:
        return ""

    text = _strip_code_fences(text)

    outside_parts, inside_parts = _split_think_tags(text)
    outside = "".join(outside_parts).strip()
    if outside:
        json_snip = _find_first_valid_json(outside)
        if json_snip:
            return json_snip

    inside = "".join(inside_parts).strip()
    if inside:
        json_snip = _find_first_valid_json(inside)
        if json_snip:
            return json_snip

    return text  # 全部失败,返回原 cleaned 文本


def _split_think_tags(text: str) -> Tuple[List[str], List[str]]:
    """Split text by <think>...</think> tags. Returns (outside_parts, inside_parts)."""
    outside_parts: List[str] = []
    inside_parts: List[str] = []
    idx = 0
    while True:
        start = text.find("<think>", idx)
        if start == -1:
            outside_parts.append(text[idx:])
            break
        outside_parts.append(text[idx:start])
        end = text.find("</think>", start + len("<think>"))
        if end == -1:
            inside_parts.append(text[start + len("<think>"):])
            break
        inside_parts.append(text[start + len("<think>"):end])
        idx = end + len("</think>")
    return outside_parts, inside_parts


def _strip_code_fences(text: str) -> str:
    """Remove markdown \`\`\`json ... \`\`\` fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _find_first_valid_json(text: str) -> str:
    """扫描文本,找所有 { } 平衡块,返回第一个能 json.loads 成功的。

    必须解析为 dict 或 list(不是单个 int/string/None)。
    """
    if not text:
        return ""
    text = text.strip()

    # 快速通道:整段以 { 开头以 } 结尾,直接试
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            if isinstance(obj, (dict, list)):
                return text
        except json.JSONDecodeError:
            pass

    # 快速通道:整段以 [ 开头以 ] 结尾,直接试
    if text.startswith("[") and text.endswith("]"):
        try:
            obj = json.loads(text)
            if isinstance(obj, (dict, list)):
                return text
        except json.JSONDecodeError:
            pass

    # 栈匹配:按出现顺序找 { } 平衡块
    candidates = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            # 从 i 开始向右找匹配的 }
            depth = 0
            in_string = False
            escape = False
            for j in range(i, n):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append((i, j + 1))
                        break
            i += 1
        else:
            i += 1

    # 同样找 [ ] 平衡块
    i = 0
    while i < n:
        if text[i] == "[":
            depth = 0
            in_string = False
            escape = False
            for j in range(i, n):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        candidates.append((i, j + 1))
                        break
            i += 1
        else:
            i += 1

    # 按出现顺序排序,返回第一个可解析的
    candidates.sort(key=lambda x: x[0])
    for start, end in candidates:
        snippet = text[start:end]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, (dict, list)):
                return snippet
        except json.JSONDecodeError:
            continue
    return ""


def build_image_content(image_url: str, max_width: int = 1024) -> Dict[str, Any]:
    """
    构建图片内容。
    支持传入 HTTP URL 或本地文件路径。
    如果传入本地路径，会自动压缩并转为 base64 data URI。
    """
    if image_url.startswith("http://") or image_url.startswith("https://"):
        return {"type": "image_url", "image_url": {"url": image_url}}

    data_uri, _ = compress_image_to_base64(image_url, max_width=max_width)
    return {"type": "image_url", "image_url": {"url": data_uri}}


def extract_json(text: str) -> Any:
    """从模型输出中提取 JSON（支持对象和数组），失败时返回 None。"""
    if not text or not text.strip():
        return None

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    # Most LLM output is already valid JSON. Try parsing it directly first so
    # heuristic cleaners cannot corrupt a perfectly good payload.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to a sequence of cleaners that try to rescue malformed JSON.
    cleaners = [
        _fix_unescaped_quotes,
        _fix_chinese_quotes,
        lambda t: _fix_unescaped_quotes(_fix_chinese_quotes(t)),
    ]

    for cleaner in cleaners:
        cleaned = cleaner(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 尝试截取外层结构
            result = _extract_outer_json(cleaned)
            if result is not None:
                return result
            continue

    # Last resort: extract any outer JSON object/array from the original text.
    return _extract_outer_json(text)


def _extract_outer_json(text: str) -> Any | None:
    """尝试从文本中提取最外层的 JSON 对象或数组"""
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    start_arr = text.find("[")
    end_arr = text.rfind("]")

    candidates = []
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidates.append((start_obj, end_obj + 1))
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidates.append((start_arr, end_arr + 1))

    # 按起始位置排序，取最外层
    candidates.sort()
    for start, end in candidates:
        snippet = text[start:end]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            # 再清洗一次
            for cleaner in [_fix_unescaped_quotes, _fix_chinese_quotes, lambda t: _fix_unescaped_quotes(_fix_chinese_quotes(t))]:
                try:
                    return json.loads(cleaner(snippet))
                except json.JSONDecodeError:
                    continue
    return None


def _fix_chinese_quotes(text: str) -> str:
    """把中文引号替换为普通字符或转义，避免破坏 JSON"""
    # 中文双引号 “ ” 在 JSON 字符串内会非法
    # 简单策略：先替换为 ASCII 双引号，再让 _fix_unescaped_quotes 处理转义
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text


def _fix_unescaped_quotes(text: str) -> str:
    """
    Fix unescaped quotes inside JSON string values.
    This handles the common case where LLMs output quotes inside string values
    without escaping them, e.g.: "desc": "has "special" marks"
    """
    result = []
    in_string = False
    escape_next = False
    i = 0
    while i < len(text):
        char = text[i]
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        if char == "\\":
            result.append(char)
            escape_next = True
            i += 1
            continue
        if char == '"':
            if not in_string:
                in_string = True
                result.append(char)
            else:
                # Check if this quote is followed by JSON structural chars
                # If so, it's a closing quote; otherwise it's an unescaped inner quote
                j = i + 1
                while j < len(text) and text[j] in " \t\n\r":
                    j += 1
                if j < len(text) and text[j] in ",:}]}":
                    in_string = False
                    result.append(char)
                else:
                    # This is an unescaped quote inside a string - escape it
                    result.append('\\"')
            i += 1
            continue
        result.append(char)
        i += 1
    return "".join(result)
