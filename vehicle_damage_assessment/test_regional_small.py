import asyncio
import json
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents.regional_worker import regional_damage_worker
from agents.synthesizer import synthesizer_agent
from agents import vehicle_prior_agent

async def main():
    vehicle_info = {"brand": "蔚来", "model": "ES8", "year": "2019"}
    vehicle_prior = await vehicle_prior_agent(vehicle_info)

    folder = "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868"
    import os
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])

    # 模拟按位置分组
    groups = {
        "车头": [{"id": "蔚来ES8_02.png", "path": f"{folder}/蔚来ES8_02.png", "detail": "车头", "confidence": "high"}],
        "车尾": [{"id": "蔚来ES8_03.png", "path": f"{folder}/蔚来ES8_03.png", "detail": "车尾", "confidence": "high"}],
        "左侧": [{"id": "蔚来ES8_05.png", "path": f"{folder}/蔚来ES8_05.png", "detail": "左侧", "confidence": "high"}],
        "右侧": [{"id": "蔚来ES8_06.png", "path": f"{folder}/蔚来ES8_06.png", "detail": "右侧", "confidence": "high"}],
    }

    region_results = []
    for location, photos in groups.items():
        print(f"\n--- Worker: {location} ---")
        try:
            result = await regional_damage_worker(location, photos, vehicle_prior)
            print(f"OK, parts: {len(result.get('parts', []))}")
            print(json.dumps(result, ensure_ascii=False, indent=2)[:1000])
            region_results.append(result)
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\n--- Synthesizer ({len(region_results)} regions) ---")
    final = synthesizer_agent(region_results, vehicle_prior)
    print(f"Final parts: {len(final.get('parts', []))}")
    print(json.dumps(final, ensure_ascii=False, indent=2)[:2000])

asyncio.run(main())
