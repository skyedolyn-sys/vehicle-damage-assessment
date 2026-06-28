import asyncio
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
import aiohttp
from agents.minimax_client import call_minimax, SSL_CONTEXT, MINIMAX_API_KEY

PHOTO_PATH = "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_02.png"


async def upload_to_minimax(path: str) -> str:
    url = "https://api.minimax.io/v1/files/upload"
    headers = {"Authorization": f"Bearer {MINIMAX_API_KEY}"}

    with open(path, "rb") as f:
        file_data = f.read()

    form = aiohttp.FormData()
    form.add_field("purpose", "video_understanding")
    form.add_field("file", file_data, filename=path.split("/")[-1])

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=form, ssl=SSL_CONTEXT) as resp:
            text = await resp.text()
            print(f"Upload status: {resp.status}")
            print(f"Upload response: {text}")
            if resp.status != 200:
                raise RuntimeError(f"Upload failed: {text}")
            data = await resp.json()
            return str(data["file"]["file_id"])


async def main():
    file_id = await upload_to_minimax(PHOTO_PATH)
    print(f"File ID: {file_id}")

    content = [
        {"type": "text", "text": "判断这张照片拍摄的是车辆哪个区域。只输出 JSON 数组。"},
        {"type": "text", "text": "照片编号: 02"},
        {"type": "image_url", "image_url": {"url": f"mm_file://{file_id}"}},
    ]

    raw = await call_minimax([{"role": "user", "content": content}], temperature=0.1, max_tokens=1000)
    print("RAW:", repr(raw[:500]))


if __name__ == "__main__":
    asyncio.run(main())
