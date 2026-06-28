import asyncio
import base64
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents.minimax_client import call_minimax

PHOTO_PATH = "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_02.png"


def load_b64(path: str) -> tuple[str, str]:
    with open(path, "rb") as f:
        data = f.read()
    ext = path.split(".")[-1].lower()
    mime = f"image/{ext}" if ext in ["png", "jpg", "jpeg", "webp", "gif"] else "image/jpeg"
    return mime, base64.b64encode(data).decode("utf-8")


async def try_format(name: str, content: list) -> None:
    print(f"\n{'='*60}")
    print(f"FORMAT: {name}")
    print(f"{'='*60}")
    messages = [{"role": "user", "content": content}]
    try:
        raw = await call_minimax(messages, temperature=0.1, max_tokens=1000)
        print("RAW:", repr(raw[:300]))
    except Exception as e:
        print("ERROR:", e)


async def main():
    mime, b64 = load_b64(PHOTO_PATH)

    text_intro = [
        {"type": "text", "text": "判断这张照片拍摄的是车辆哪个区域。只输出 JSON 数组。"},
        {"type": "text", "text": "照片编号: 02"},
    ]

    # 1. OpenAI image_url with base64 data URI
    await try_format(
        "OpenAI image_url + base64 data URI",
        text_intro + [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}],
    )

    # 2. Anthropic image + base64 source
    await try_format(
        "Anthropic image + base64 source",
        text_intro + [{"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}],
    )

    # 3. OpenAI image_url with HTTPS public image
    await try_format(
        "OpenAI image_url + HTTPS URL",
        text_intro + [{"type": "image_url", "image_url": {"url": "https://filecdn.minimax.chat/public/fe9d04da-f60e-444d-a2e0-18ae743add33.jpeg"}}],
    )

    # 4. Anthropic image + URL source
    await try_format(
        "Anthropic image + URL source",
        text_intro + [{"type": "image", "source": {"type": "url", "url": "https://filecdn.minimax.chat/public/fe9d04da-f60e-444d-a2e0-18ae743add33.jpeg"}}],
    )

    # 5. OpenAI image_url with localhost URL
    await try_format(
        "OpenAI image_url + localhost URL",
        text_intro + [{"type": "image_url", "image_url": {"url": "http://localhost:8000/uploads/test/蔚来ES8_02.png"}}],
    )

    # 6. OpenAI image_url with localhost URL (URL-encoded filename)
    await try_format(
        "OpenAI image_url + localhost URL encoded",
        text_intro + [{"type": "image_url", "image_url": {"url": "http://localhost:8000/uploads/test/%E8%94%9A%E6%9D%A5ES8_02.png"}}],
    )

    # 7. OpenAI image_url with detail=low
    await try_format(
        "OpenAI image_url + base64 detail=low",
        text_intro + [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}}],
    )


if __name__ == "__main__":
    asyncio.run(main())
