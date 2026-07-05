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
from config import MINIMAX_API_KEY, MINIMAX_BASE_URL, MINIMAX_MODEL, REQUEST_TIMEOUT
from agents.image_utils import compress_image_to_base64

logger = logging.getLogger(__name__)
# Dedicated file log for API diagnostics; Django console log level may swallow INFO.
_minimax_file_handler = logging.FileHandler(
    os.path.expanduser("~/vehicle_damage_assessment_minimax.log"), mode="a", encoding="utf-8"
)
_minimax_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_minimax_file_handler.setLevel(logging.INFO)
logger.addHandler(_minimax_file_handler)
logger.setLevel(logging.INFO)

# 临时处理 macOS SSL 证书问题
# 生产环境应安装 certifi 或正确配置证书
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


import random

async def call_minimax(
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 4000,
    model: str | None = None,
    response_format: Dict[str, Any] | None = None,
) -> str:
    """调用 MiniMax OpenAI 兼容接口（带重试和指数退避）"""
    call_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    request_model = model or MINIMAX_MODEL
    logger.info("[minimax:%s] start model=%s temp=%s max_tokens=%s json_mode=%s", call_id, request_model, temperature, max_tokens, bool(response_format))

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": request_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format

    last_exception = None
    for attempt in range(1, 4):
        start = time.perf_counter()
        try:
            connector = aiohttp.TCPConnector(limit=1, force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    MINIMAX_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT, sock_connect=30),
                    ssl=SSL_CONTEXT,
                ) as resp:
                    elapsed = time.perf_counter() - start
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning("[minimax:%s] attempt=%d status=%d elapsed=%.2fs text=%s", call_id, attempt, resp.status, elapsed, text[:500])
                        raise RuntimeError(f"MiniMax API error {resp.status}: {text}")
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    cleaned = clean_minimax_output(content)
                    logger.info("[minimax:%s] attempt=%d success elapsed=%.2fs raw_len=%d cleaned_len=%d", call_id, attempt, elapsed, len(content), len(cleaned))
                    # Diagnostic: if cleaning stripped usable JSON, dump samples for inspection.
                    if cleaned and not extract_json(cleaned):
                        diagnostic_path = os.path.expanduser(f"~/minimax_diagnostic_{call_id}.txt")
                        try:
                            with open(diagnostic_path, "w", encoding="utf-8") as f:
                                f.write("===== RAW =====\n")
                                f.write(content)
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
