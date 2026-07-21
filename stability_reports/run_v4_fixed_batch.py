import asyncio
import json
import os
import sys
import time
import traceback

sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents import assessment_orchestrator

FOLDER = '/Users/sky/Downloads/车顶闸调试样本_20260622/lead_167111'
VEHICLE_INFO = {"brand": "蔚来", "model": "ES8", "year": "2019"}
OUT_DIR = '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/stability_reports'


# Import helpers from run_v4
sys.path.insert(0, OUT_DIR)
from run_v4 import build_summary, to_native


async def run_fixed_plan_once(run_id: int, plan: dict):
    files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith('.png')])
    photos = [{"id": f, "path": f"{FOLDER}/{f}", "name": f} for f in files]
    start = time.time()
    result = await assessment_orchestrator(photos, VEHICLE_INFO, plan=plan)
    elapsed = time.time() - start
    result_native = to_native(result)

    out_path = os.path.join(OUT_DIR, f'team_v4_fixed_run_{run_id}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result_native, f, ensure_ascii=False, indent=2)

    summary = build_summary(result_native, elapsed)
    summary_path = os.path.join(OUT_DIR, f'team_v4_fixed_run_{run_id}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({"run": run_id, **summary}, f, ensure_ascii=False, indent=2)

    print(f"Fixed run {run_id}: saved to {out_path}, elapsed={elapsed:.1f}s, uncertain={summary['uncertain_count']}, views={summary['covered_view_count']}")
    return summary


async def run_fixed_plan_batch(plan_path: str, start_id: int, count: int, max_retries: int = 2):
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)

    summaries = []
    failures = []

    for i in range(count):
        run_id = start_id + i
        attempt = 0
        while True:
            try:
                print(f"\n[FixedBatch] Starting run {run_id} (attempt {attempt + 1}/{max_retries + 1})")
                summary = await run_fixed_plan_once(run_id, plan)
                summaries.append({"run_id": run_id, "status": "ok", **summary})
                break
            except Exception as exc:
                attempt += 1
                print(f"[FixedBatch] Run {run_id} failed: {exc}")
                traceback.print_exc()
                if attempt > max_retries:
                    failures.append({"run_id": run_id, "error": str(exc)})
                    break
                wait = 10 * attempt
                print(f"[FixedBatch] Retrying run {run_id} in {wait}s...")
                await asyncio.sleep(wait)

        if i < count - 1:
            await asyncio.sleep(5)

    batch_summary = {
        "plan_path": plan_path,
        "start_id": start_id,
        "count": count,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "successful_runs": [s["run_id"] for s in summaries],
        "failed_runs": failures,
        "summaries": summaries,
    }

    summary_path = os.path.join(OUT_DIR, f'team_v4_fixed_batch_{start_id}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    print(f"\n[FixedBatch] Saved summary to {summary_path}")
    print(f"[FixedBatch] Success: {len(summaries)}/{count}, Failures: {len(failures)}")
    return batch_summary


if __name__ == '__main__':
    plan_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DIR, 'team_v4_fixed_plan_run_13.json')
    start_id = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    asyncio.run(run_fixed_plan_batch(plan_path, start_id, count))
