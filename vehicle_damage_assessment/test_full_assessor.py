import asyncio
import json
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents import vehicle_prior_agent, photo_locator_agent, damage_assessor_agent
from config import PHOTO_LOCATOR_BATCH_SIZE, MAX_CONCURRENT_API_CALLS

async def main():
    vehicle_info = {"brand": "蔚来", "model": "ES8", "year": "2019"}
    vehicle_prior = await vehicle_prior_agent(vehicle_info)

    folder = "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868"
    import os
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    photos = [{"id": f, "path": f"{folder}/{f}", "name": f, "url": f"http://localhost:8000/uploads/test/{f}"} for f in files]

    batches = [photos[i:i+PHOTO_LOCATOR_BATCH_SIZE] for i in range(0, len(photos), PHOTO_LOCATOR_BATCH_SIZE)]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

    async def locate_batch(batch):
        async with semaphore:
            return await photo_locator_agent(batch, vehicle_prior)

    results = await asyncio.gather(*[locate_batch(b) for b in batches])
    all_locations = []
    for r in results:
        if isinstance(r, list):
            all_locations.extend(r)

    damage_result = await damage_assessor_agent(photos, all_locations, vehicle_prior)
    print("Damage result type:", type(damage_result))
    print("Damage result:")
    print(json.dumps(damage_result, ensure_ascii=False, indent=2)[:3000])
    with open("debug_damage.json", "w", encoding="utf-8") as f:
        json.dump(damage_result, f, ensure_ascii=False, indent=2)

asyncio.run(main())
