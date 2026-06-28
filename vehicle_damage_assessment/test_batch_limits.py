import asyncio
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents.photo_locator import photo_locator_agent

async def test_batch(count: int):
    vehicle_prior = {
        "vehicle": "蔚来 ES8 2019款",
        "topology": {"front": "X-bar 前脸", "rear": "贯穿尾灯"},
        "key_anchors": {"front": ["NIO logo"], "rear": ["贯穿尾灯"]}
    }
    folder = "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868"
    photos = []
    for i in range(1, count + 1):
        fname = f"蔚来ES8_{i:02d}.png"
        path = f"{folder}/{fname}"
        photos.append({"id": f"{i:02d}", "path": path, "url": f"http://localhost:8000/uploads/test/{fname}"})
    try:
        result = await photo_locator_agent(photos, vehicle_prior)
        print(f"count={count}: OK, got {len(result)} results")
        return True
    except Exception as e:
        print(f"count={count}: FAILED - {type(e).__name__}: {e}")
        return False

async def main():
    for c in [1, 2, 3, 4, 5]:
        await test_batch(c)
        await asyncio.sleep(1)

asyncio.run(main())
