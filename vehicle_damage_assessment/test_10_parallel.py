"""Run 10 parallel vehicle damage assessments on lead_167111.

This script uses the current assessment_orchestrator (with the latest
synthesizer/vision/planner changes) and runs 10 assessments concurrently,
writing both per-run detailed JSON and a summary JSON/HTML report.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, "/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment")

from agents import assessment_orchestrator, vehicle_prior_agent
from agents.topology_builder import build_vehicle_topology
from agents.topology_comparator import compare_topology


FOLDER = "/Users/sky/Downloads/车顶闸调试样本_20260622/lead_167111"
VEHICLE_INFO = {"brand": "蔚来", "model": "ES8", "year": "2019"}
OUTPUT_DIR = Path("/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/team_parallel_results")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
RUN_COUNT = 10
MAX_CONCURRENCY = 5


@dataclass
class RunResult:
    run_id: int
    status: str = "pending"
    duration_seconds: float = 0.0
    error: str = ""
    parts: List[Dict[str, Any]] = field(default_factory=list)
    damaged_part_ids: List[str] = field(default_factory=list)
    missing_part_ids: List[str] = field(default_factory=list)
    uncertain_part_ids: List[str] = field(default_factory=list)
    intact_part_ids: List[str] = field(default_factory=list)
    coverage_summary: Dict[str, Any] = field(default_factory=dict)
    subagent_views: List[str] = field(default_factory=list)
    overall_severity: str = ""
    structural_damage_flag: bool = False


def _collect_photos() -> List[Dict[str, Any]]:
    photos: List[Dict[str, Any]] = []
    for path in sorted(Path(FOLDER).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            photos.append({
                "id": path.name,
                "path": str(path),
                "name": path.name,
                "url": f"file://{path}",
            })
    return photos


def _extract_part_lists(parts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    result = {"damaged": [], "missing": [], "intact": [], "uncertain": []}
    for part in parts:
        status = part.get("status", "uncertain")
        part_id = part.get("part_id", "")
        result.get(status, result["uncertain"]).append(part_id)
    return result


async def run_single(run_id: int, semaphore: asyncio.Semaphore) -> RunResult:
    result = RunResult(run_id=run_id)
    async with semaphore:
        start = time.perf_counter()
        try:
            photos = _collect_photos()
            orchestrator_result = await assessment_orchestrator(photos, VEHICLE_INFO)
            parts = orchestrator_result.get("parts", [])
            result.parts = parts

            lists = _extract_part_lists(parts)
            result.damaged_part_ids = lists["damaged"]
            result.missing_part_ids = lists["missing"]
            result.uncertain_part_ids = lists["uncertain"]
            result.intact_part_ids = lists["intact"]

            plan = orchestrator_result.get("plan", {})
            result.coverage_summary = {
                "covered_views": plan.get("workflow_plan", {}).get("priority_views", []),
                "missing_critical_views": plan.get("workflow_plan", {}).get("missing_critical_views", []),
                "coverage_gaps": plan.get("coverage_gaps", []),
            }
            result.subagent_views = [r.get("view_id", "") for r in orchestrator_result.get("subagent_results", [])]

            vehicle_prior = orchestrator_result.get("vehicle_prior", {})
            topology = build_vehicle_topology(VEHICLE_INFO, vehicle_prior)
            from models.part_state import PartActualState
            flat_states = [PartActualState.from_legacy_dict(p) for p in parts]
            damage_assessment = compare_topology(topology, flat_states)
            result.overall_severity = damage_assessment.overall_severity
            result.structural_damage_flag = damage_assessment.structural_damage_flag

            result.status = "success"
        except Exception as exc:
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        result.duration_seconds = round(time.perf_counter() - start, 2)
    return result


def _write_reports(results: List[RunResult]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: Dict[str, Any] = {"timestamp": timestamp, "runs": []}

    for result in results:
        run_file = OUTPUT_DIR / f"167111_run{result.run_id}_{timestamp}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2, default=str)

        summary["runs"].append({
            "run_id": result.run_id,
            "status": result.status,
            "duration_seconds": result.duration_seconds,
            "overall_severity": result.overall_severity,
            "structural_damage_flag": result.structural_damage_flag,
            "damaged_count": len(result.damaged_part_ids),
            "missing_count": len(result.missing_part_ids),
            "uncertain_count": len(result.uncertain_part_ids),
            "intact_count": len(result.intact_part_ids),
            "damaged_parts": result.damaged_part_ids,
            "missing_parts": result.missing_part_ids,
            "uncertain_parts": result.uncertain_part_ids,
            "subagent_views": result.subagent_views,
            "coverage_summary": result.coverage_summary,
            "error": result.error[:500] if result.error else "",
        })

    summary_file = OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    html_file = OUTPUT_DIR / f"summary_{timestamp}.html"
    _write_html_report(html_file, summary)

    print(f"\nReports written to: {OUTPUT_DIR}")
    print(f"  Summary JSON: {summary_file}")
    print(f"  Summary HTML: {html_file}")


def _write_html_report(path: Path, summary: Dict[str, Any]) -> None:
    rows = ""
    for run in summary["runs"]:
        rows += f"""
        <tr>
          <td>{run['run_id']}</td>
          <td class="{run['status']}">{run['status']}</td>
          <td>{run['duration_seconds']}</td>
          <td>{run['overall_severity']}</td>
          <td>{'是' if run['structural_damage_flag'] else '否'}</td>
          <td>{run['damaged_count']}</td>
          <td>{run['missing_count']}</td>
          <td>{run['uncertain_count']}</td>
          <td>{', '.join(run['subagent_views'])}</td>
          <td>{', '.join(run['damaged_parts'])}</td>
          <td>{', '.join(run['uncertain_parts'])}</td>
          <td style="max-width:300px;white-space:pre-wrap">{run['error']}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Lead 167111 — 10 Parallel Runs Summary</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .success {{ color: green; font-weight: bold; }}
  .error {{ color: red; font-weight: bold; }}
</style>
</head>
<body>
<h1>Lead 167111 — 10 Parallel Runs Summary ({summary['timestamp']})</h1>
<table>
  <tr>
    <th>Run</th>
    <th>Status</th>
    <th>Time(s)</th>
    <th>Severity</th>
    <th>Structural</th>
    <th>Damaged</th>
    <th>Missing</th>
    <th>Uncertain</th>
    <th>Views</th>
    <th>Damaged Parts</th>
    <th>Uncertain Parts</th>
    <th>Error</th>
  </tr>
  {rows}
</table>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


async def main() -> None:
    print(f"Starting {RUN_COUNT} parallel assessments on {FOLDER}")
    print(f"Output directory: {OUTPUT_DIR}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [asyncio.create_task(run_single(i, semaphore), name=f"run_{i}") for i in range(1, RUN_COUNT + 1)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    normalized: List[RunResult] = []
    for result in results:
        if isinstance(result, Exception):
            placeholder = RunResult(run_id=0, status="error", error=f"{type(result).__name__}: {result}\n{traceback.format_exc()}")
            normalized.append(placeholder)
        else:
            normalized.append(result)

    _write_reports(normalized)

    print("\n" + "=" * 100)
    print(f"{'Run':<6} {'Status':<8} {'Time(s)':<9} {'Severity':<9} {'Structural':<10} {'Damaged':<8} {'Missing':<8} {'Uncertain':<10}")
    print("-" * 100)
    for r in normalized:
        print(
            f"{r.run_id:<6} {r.status:<8} {r.duration_seconds:<9} "
            f"{r.overall_severity:<9} {'是' if r.structural_damage_flag else '否':<10} "
            f"{len(r.damaged_part_ids):<8} {len(r.missing_part_ids):<8} {len(r.uncertain_part_ids):<10}"
        )
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
