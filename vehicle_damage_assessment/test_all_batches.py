import asyncio
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from config import PHOTO_LOCATOR_BATCH_SIZE, MAX_CONCURRENT_API_CALLS
from agents.photo_locator import photo_locator_agent

async def main():
    vp = {'vehicle': '蔚来 ES8 2019款', 'topology': {}, 'key_anchors': {}}
    folder = '/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868'
    import os
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    photos = [{'id': f, 'path': f'{folder}/{f}', 'url': f'http://localhost:8000/uploads/test/{f}'} for f in files]
    print(f"Total photos: {len(photos)}, batch size: {PHOTO_LOCATOR_BATCH_SIZE}, max concurrent: {MAX_CONCURRENT_API_CALLS}")

    batches = [photos[i:i+PHOTO_LOCATOR_BATCH_SIZE] for i in range(0, len(photos), PHOTO_LOCATOR_BATCH_SIZE)]
    for i, batch in enumerate(batches):
        print(f"\n--- Batch {i+1}/{len(batches)}: {[p['id'] for p in batch]} ---")
        try:
            r = await photo_locator_agent(batch, vp)
            print(f"OK, got {len(r)} results")
        except Exception as e:
            import traceback
            print(f"FAILED: {e}")
            traceback.print_exc()
        await asyncio.sleep(1)

asyncio.run(main())
