import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents import assessment_orchestrator

FOLDER = '/Users/sky/Downloads/车顶闸调试样本_20260622/lead_167111'
VEHICLE_INFO = {"brand": "蔚来", "model": "ES8", "year": "2019"}

async def main():
    files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith('.png')])
    photos = [{"id": f, "path": f"{FOLDER}/{f}", "name": f} for f in files]
    start = time.time()
    result = await assessment_orchestrator(photos, VEHICLE_INFO)
    elapsed = time.time() - start
    out_path = '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/stability_reports/team_gamma_run_3.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}, elapsed={elapsed:.1f}s")

    # 结构稳定性检查
    parts = result.get('parts', [])
    required_fields = {'part_id','part_name','part_category','side','status','damage_level','damage_type','confidence','evidence_photo','notes'}
    missing_fields_counts = {}
    type_issues = []
    for p in parts:
        missing = required_fields - set(p.keys())
        if missing:
            missing_fields_counts[p.get('part_id','?')] = list(missing)
        for field in ['damage_type', 'evidence_photo']:
            val = p.get(field)
            if val is not None and not isinstance(val, (str, list)):
                type_issues.append({p.get('part_id'): {field: type(val).__name__}})

    print(f"Total parts: {len(parts)}")
    print(f"Required fields missing: {len(missing_fields_counts)}")
    if missing_fields_counts:
        print(json.dumps(missing_fields_counts, ensure_ascii=False, indent=2))
    print(f"Type issues: {len(type_issues)}")
    if type_issues:
        print(json.dumps(type_issues, ensure_ascii=False, indent=2))
    print(f"Uncertain items count: {len(result.get('uncertain_items', []))}")
    if result.get('uncertain_items'):
        print(json.dumps(result['uncertain_items'][:5], ensure_ascii=False, indent=2))

    # 25部件摘要
    summary = []
    for p in parts:
        summary.append({
            'part_id': p.get('part_id'),
            'part_name': p.get('part_name'),
            'status': p.get('status'),
            'damage_level': p.get('damage_level'),
            'confidence': p.get('confidence'),
        })
    print(json.dumps(summary, ensure_ascii=False, indent=2))

asyncio.run(main())
