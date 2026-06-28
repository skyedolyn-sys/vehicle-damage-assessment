import asyncio
import json
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents import vehicle_prior_agent, photo_locator_agent
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

    async def locate_batch(batch, idx):
        async with semaphore:
            try:
                return await photo_locator_agent(batch, vehicle_prior)
            except Exception as e:
                print(f"\nBatch {idx+1} failed: {e}")
                import traceback
                traceback.print_exc()
                return None

    results = await asyncio.gather(*[locate_batch(b, i) for i, b in enumerate(batches)])
    for i, r in enumerate(results):
        status = "OK" if isinstance(r, list) else "FAIL"
        print(f"Batch {i+1}: {status}")

asyncio.run(main())
