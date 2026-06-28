import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents import assessment_orchestrator

FOLDER = '/Users/sky/Downloads/车顶闸调试样本_20260622/lead_167111'
VEHICLE_INFO = {"brand": "蔚来", "model": "ES8", "year": "2019"}

def to_native(obj):
    """Recursively convert custom objects to JSON-serializable native types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if hasattr(obj, 'value') and hasattr(obj, 'name'):
        # Enum-like object (e.g., PartActualState)
        return obj.value
    if hasattr(obj, '__dict__'):
        return to_native(obj.__dict__)
    return obj

async def main():
    files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith('.png')])
    photos = [{"id": f, "path": f"{FOLDER}/{f}", "name": f} for f in files]
    start = time.time()
    result = await assessment_orchestrator(photos, VEHICLE_INFO)
    elapsed = time.time() - start
    # Convert to native types before JSON serialization
    result_native = to_native(result)
    out_path = '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/stability_reports/team_alpha_run_1.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result_native, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}, elapsed={elapsed:.1f}s")
    # summary
    parts = result_native.get('parts', [])
    summary = []
    for p in parts:
        summary.append({
            'part_id': p.get('part_id'),
            'part_name': p.get('part_name'),
            'status': p.get('status'),
            'damage_level': p.get('damage_level'),
            'confidence': p.get('confidence'),
            'evidence_photo': p.get('evidence_photo', ''),
        })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Total parts: {len(parts)}, uncertain_items: {len(result_native.get('uncertain_items', []))}")

asyncio.run(main())
