import asyncio
import json
from agents.photo_locator import photo_locator_agent

async def main():
    vehicle_prior = {
        "vehicle": "蔚来 ES8 2019款",
        "topology": {
            "front": "X-bar 前脸、分体式大灯、NIO logo 居中",
            "rear": "贯穿式尾灯、后备箱造型",
            "left": "悬浮车顶、隐藏门把手、四个车门",
            "right": "悬浮车顶、隐藏门把手、四个车门",
            "roof": "全景天窗、行李架"
        },
        "key_anchors": {
            "front": ["NIO logo", "前大灯", "前格栅"],
            "rear": ["贯穿尾灯", "后备箱盖"],
            "left": ["车门", "后视镜"],
            "right": ["车门", "后视镜"],
            "roof": ["全景天窗", "行李架"]
        }
    }
    photos = [
        {"id": "蔚来ES8_02.png", "path": "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_02.png"},
        {"id": "蔚来ES8_03.png", "path": "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_03.png"},
        {"id": "蔚来ES8_04.png", "path": "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_04.png"},
    ]
    try:
        result = await photo_locator_agent(photos, vehicle_prior)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(main())
