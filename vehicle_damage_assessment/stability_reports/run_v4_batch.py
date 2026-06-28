import asyncio
import json
import os
import sys
import time
import traceback

sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')

from run_v4 import run_once, build_summary, to_native, OUT_DIR

FOLDER = '/Users/sky/Downloads/车顶闸调试样本_20260622/lead_167111'
VEHICLE_INFO = {"brand": "蔚来", "model": "ES8", "year": "2019"}


async def run_batch(start_id: int, count: int, max_retries: int = 2):
    summaries = []
    failures = []

    for i in range(count):
        run_id = start_id + i
        attempt = 0
        while True:
            try:
                print(f"\n[Batch] Starting run {run_id} (attempt {attempt + 1}/{max_retries + 1})")
                summary = await run_once(run_id)
                summaries.append({"run_id": run_id, "status": "ok", **summary})
                break
            except Exception as exc:
                attempt += 1
                print(f"[Batch] Run {run_id} failed: {exc}")
                traceback.print_exc()
                if attempt > max_retries:
                    failures.append({"run_id": run_id, "error": str(exc)})
                    break
                wait = 10 * attempt
                print(f"[Batch] Retrying run {run_id} in {wait}s...")
                await asyncio.sleep(wait)

        # Small cooldown between runs to reduce API pressure.
        if i < count - 1:
            await asyncio.sleep(5)

    batch_summary = {
        "start_id": start_id,
        "count": count,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "successful_runs": [s["run_id"] for s in summaries],
        "failed_runs": failures,
        "summaries": summaries,
    }

    summary_path = os.path.join(OUT_DIR, f'team_v4_batch_{start_id}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    print(f"\n[Batch] Saved batch summary to {summary_path}")
    print(f"[Batch] Success: {len(summaries)}/{count}, Failures: {len(failures)}")
    return batch_summary


if __name__ == '__main__':
    start_id = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(run_batch(start_id, count))
