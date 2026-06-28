import asyncio
import json
import sys
sys.path.insert(0, '/Users/sky/Downloads/vehicle_damage_assessment')
from agents.photo_locator import photo_locator_agent

async def main():
    vehicle_prior = {
        "vehicle": "蔚来 ES8 2019款",
        "topology": {"front": "X-bar 前脸", "rear": "贯穿尾灯"},
        "key_anchors": {"front": ["NIO logo"], "rear": ["贯穿尾灯"]}
    }
    photos = [
        {"id": "02", "path": "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_02.png", "url": "http://localhost:8000/uploads/test/蔚来ES8_02.png"},
    ]
    try:
        result = await photo_locator_agent(photos, vehicle_prior)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(main())
