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


async def main():
    mime, b64 = load_b64(PHOTO_PATH)
    content = [
        {"type": "text", "text": "判断这张照片拍摄的是车辆哪个区域。只输出 JSON 数组。"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]

    for model in ["MiniMax-M3", "MiniMax-M2.7"]:
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")
        try:
            raw = await call_minimax([{"role": "user", "content": content}], temperature=0.1, max_tokens=1000, model=model)
            print("RAW:", repr(raw[:300]))
        except Exception as e:
            print("ERROR:", e)


if __name__ == "__main__":
    asyncio.run(main())
