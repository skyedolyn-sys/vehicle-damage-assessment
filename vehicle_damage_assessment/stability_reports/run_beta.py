import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents import assessment_orchestrator

FOLDER = '/Users/sky/Downloads/车顶闸调试样本_20260622/lead_167111'
VEHICLE_INFO = {"brand": "蔚来", "model": "ES8", "year": "2019"}

def make_serializable(obj):
    """递归地将对象转换为 JSON 可序列化的结构。"""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [make_serializable(item) for item in obj]
    if isinstance(obj, set):
        return [make_serializable(item) for item in obj]
    # Handle Enum objects (including StrEnum)
    import enum
    if isinstance(obj, enum.Enum):
        return obj.value
    # Handle any object with a .value attribute that isn't a basic type
    if hasattr(obj, 'value') and not isinstance(obj, (int, float, bool, str, type(None))):
        return obj.value
    # Handle objects with __dict__ (dataclasses, etc.)
    if hasattr(obj, '__dict__') and not isinstance(obj, (int, float, bool, str, type(None))):
        return make_serializable(obj.__dict__)
    return obj

async def main():
    files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith('.png')])
    photos = [{"id": f, "path": f"{FOLDER}/{f}", "name": f} for f in files]
    start = time.time()
    result = await assessment_orchestrator(photos, VEHICLE_INFO)
    elapsed = time.time() - start
    # make result JSON serializable
    result = make_serializable(result)
    out_path = '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/stability_reports/team_beta_run_2.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}, elapsed={elapsed:.1f}s")
    # summary
    parts = result.get('parts', [])
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
    print(f"Total parts: {len(parts)}, uncertain_items: {len(result.get('uncertain_items', []))}")

asyncio.run(main())
