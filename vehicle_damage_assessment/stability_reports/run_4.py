import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents import assessment_orchestrator

FOLDER = '/Users/sky/Downloads/车顶闸调试样本_20260622/lead_167111'
VEHICLE_INFO = {"brand": "蔚来", "model": "ES8", "year": "2019"}


def make_serializable(obj):
    """Recursively convert objects to JSON-serializable dicts."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if hasattr(obj, 'to_dict'):
        return make_serializable(obj.to_dict())
    if hasattr(obj, 'to_legacy_dict'):
        return make_serializable(obj.to_legacy_dict())
    if hasattr(obj, '__dict__'):
        return make_serializable(obj.__dict__)
    if hasattr(obj, 'value'):
        return obj.value
    return obj


async def main():
    files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith('.png')])
    photos = [{"id": f, "path": f"{FOLDER}/{f}", "name": f} for f in files]
    start = time.time()
    result = await assessment_orchestrator(photos, VEHICLE_INFO)
    elapsed = time.time() - start

    serializable_result = make_serializable(result)

    out_path = '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/stability_reports/team_run_4.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)

    plan = result.get('plan', {})
    covered_views = [v for v, g in plan.get('view_groups', {}).items() if g and v in {
        'front','front_left_45','front_right_45','rear','rear_left_45','rear_right_45','left_90','right_90','top'
    }]
    subagent_count = len(result.get('subagent_results', []))
    parts = result.get('parts', [])
    uncertain_parts = [p for p in parts if p.get('status') == 'uncertain']

    summary = {
        "run": 4,
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

    sum_path = '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/stability_reports/team_run_4_summary.json'
    with open(sum_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Run 4 done: elapsed={elapsed:.1f}s, covered_views={covered_views}, subagent_count={subagent_count}, uncertain={len(uncertain_parts)}/25")
    print(f"Saved: {out_path}, {sum_path}")

asyncio.run(main())
