import asyncio
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents.photo_locator import photo_locator_agent

async def main():
    vehicle_prior = {
        "vehicle": "蔚来 ES8 2019款",
        "topology": {"front": "X-bar 前脸", "rear": "贯穿尾灯"},
        "key_anchors": {"front": ["NIO logo"], "rear": ["贯穿尾灯"]}
    }
    photos = [
        {"id": "02", "path": "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_02.png", "url": "http://localhost:8000/uploads/test/蔚来ES8_02.png"},
        {"id": "03", "path": "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_03.png", "url": "http://localhost:8000/uploads/test/蔚来ES8_03.png"},
        {"id": "04", "path": "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_04.png", "url": "http://localhost:8000/uploads/test/蔚来ES8_04.png"},
    ]
    result = await photo_locator_agent(photos, vehicle_prior)
    print(result)

asyncio.run(main())
