import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents import assessment_orchestrator

FOLDER = '/Users/sky/Downloads/车顶闸调试样本_20260622/lead_167111'
VEHICLE_INFO = {"brand": "蔚来", "model": "ES8", "year": "2019"}
OUT_DIR = '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/stability_reports'


def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if hasattr(obj, 'value') and hasattr(obj, 'name'):
        return obj.value
    if hasattr(obj, '__dict__'):
        return to_native(obj.__dict__)
    return obj


def build_summary(result_native: dict, elapsed: float) -> dict:
    plan = result_native.get('plan', {})
    parts = result_native.get('parts', [])
    uncertain_part_ids = [p['part_id'] for p in parts if p.get('status') == 'uncertain']
    status_counts = {}
    for p in parts:
        status_counts[p.get('status', 'unknown')] = status_counts.get(p.get('status', 'unknown'), 0) + 1

    view_groups = plan.get('view_groups', {})
    covered_views = [v for v, g in view_groups.items() if g and v in (
        'front', 'front_left_45', 'front_right_45', 'rear', 'rear_left_45', 'rear_right_45', 'left_90', 'right_90', 'top'
    )]
    return {
        "elapsed": elapsed,
        "photo_count": len(plan.get('photo_views', [])),
        "covered_views": covered_views,
        "covered_view_count": len(covered_views),
        "subagent_count": len([v for v, g in view_groups.items() if g and v in (
            'front', 'front_left_45', 'front_right_45', 'rear', 'rear_left_45', 'rear_right_45', 'left_90', 'right_90', 'top'
        )]),
        "total_parts": len(parts),
        "uncertain_count": len(uncertain_part_ids),
        "uncertain_part_ids": uncertain_part_ids,
        "status_counts": status_counts,
        "parts": [
            {
                "part_id": p.get('part_id'),
                "part_name": p.get('part_name'),
                "status": p.get('status'),
                "damage_level": p.get('damage_level'),
                "confidence": p.get('confidence'),
                "notes": p.get('notes', ''),
            }
            for p in parts
        ],
    }


async def run_once(run_id: int):
    files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith('.png')])
    photos = [{"id": f, "path": f"{FOLDER}/{f}", "name": f} for f in files]
    start = time.time()
    result = await assessment_orchestrator(photos, VEHICLE_INFO)
    elapsed = time.time() - start
    result_native = to_native(result)

    out_path = os.path.join(OUT_DIR, f'team_v2_run_{run_id}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result_native, f, ensure_ascii=False, indent=2)

    summary = build_summary(result_native, elapsed)
    summary_path = os.path.join(OUT_DIR, f'team_v2_run_{run_id}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({"run": run_id, **summary}, f, ensure_ascii=False, indent=2)

    print(f"Run {run_id}: saved to {out_path}, elapsed={elapsed:.1f}s, uncertain={summary['uncertain_count']}")
    return summary


if __name__ == '__main__':
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    asyncio.run(run_once(run_id))
