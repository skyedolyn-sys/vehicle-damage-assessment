import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents import assessment_orchestrator

FOLDER = '/Users/sky/Downloads/车顶闸调试样本_20260622/lead_167111'
VEHICLE_INFO = {"brand": "蔚来", "model": "ES8", "year": "2019"}

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle PartActualState
        if hasattr(obj, 'to_legacy_dict'):
            return obj.to_legacy_dict()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        # Handle Enum types
        if hasattr(obj, 'value'):
            return obj.value
        # Handle set
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

async def main():
    files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith('.png')])
    photos = [{"id": f, "path": f"{FOLDER}/{f}", "name": f} for f in files]
    start = time.time()
    result = await assessment_orchestrator(photos, VEHICLE_INFO)
    elapsed = time.time() - start

    out_path = '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/stability_reports/team_run_6.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, cls=CustomEncoder)

    plan = result.get('plan', {})
    covered_views = [v for v, g in plan.get('view_groups', {}).items() if g and v in {
        'front','front_left_45','front_right_45','rear','rear_left_45','rear_right_45','left_90','right_90','top'
    }]
    subagent_count = len(result.get('subagent_results', []))
    parts = result.get('parts', [])
    uncertain_parts = [p for p in parts if p.get('status') == 'uncertain']

    summary = {
        "run": 6,
        "elapsed": elapsed,
        "photo_count": len(photos),
        "covered_views": covered_views,
        "covered_view_count": len(covered_views),
        "subagent_count": subagent_count,
        "total_parts": len(parts),
        "uncertain_count": len(uncertain_parts),
        "uncertain_part_ids": [p['part_id'] for p in uncertain_parts],
        "status_counts": {},
        "parts": [{"part_id": p['part_id'], "part_name": p['part_name'], "status": p['status'], "damage_level": p['damage_level'], "confidence": p['confidence']} for p in parts]
    }
    from collections import Counter
    summary["status_counts"] = dict(Counter(p['status'] for p in parts))

    sum_path = '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/stability_reports/team_run_6_summary.json'
    with open(sum_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Run 6 done: elapsed={elapsed:.1f}s, covered_views={covered_views}, subagent_count={subagent_count}, uncertain={len(uncertain_parts)}/25")
    print(f"Saved: {out_path}, {sum_path}")

asyncio.run(main())
