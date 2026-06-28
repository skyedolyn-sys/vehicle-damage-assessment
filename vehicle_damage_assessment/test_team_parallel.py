"""Team parallel test — run 5 vehicle damage assessments concurrently.

This script exercises the new assessment_orchestrator against 5 different
vehicle photo sets and writes per-lead JSON reports plus a summary HTML/JSON
comparison.  It is intentionally self-contained so it can be re-run without
touching the Django API layer.
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_ROOT = Path("/Users/sky/Downloads/车顶闸调试样本_20260622")

# Run the same lead 5 times to measure stability / variance.
_BASE_LEAD = {"lead": "167111", "brand": "蔚来", "model": "ES8", "year": "2019"}
SELECTED_LEADS = [{**_BASE_LEAD, "run_id": i} for i in range(1, 6)]

OUTPUT_DIR = Path("/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/team_parallel_results")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ---------------------------------------------------------------------------
# Data classes for reporting
# ---------------------------------------------------------------------------

@dataclass
class LeadResult:
    lead: str
    run_id: int
    vehicle_info: Dict[str, str]
    photo_count: int
    status: str = "pending"
    duration_seconds: float = 0.0
    error: str = ""
    parts: List[Dict[str, Any]] = field(default_factory=list)
    uncertain_items: List[Dict[str, Any]] = field(default_factory=list)
    damaged_part_ids: List[str] = field(default_factory=list)
    missing_part_ids: List[str] = field(default_factory=list)
    intact_part_ids: List[str] = field(default_factory=list)
    uncertain_part_ids: List[str] = field(default_factory=list)
    coverage_summary: Dict[str, Any] = field(default_factory=dict)
    subagent_views: List[str] = field(default_factory=list)
    overall_severity: str = ""
    structural_damage_flag: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_photos(lead_dir: Path) -> List[Dict[str, Any]]:
    photos: List[Dict[str, Any]] = []
    for path in sorted(lead_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            photos.append(
                {
                    "id": path.name,
                    "path": str(path),
                    "name": path.name,
                    "url": f"file://{path}",
                }
            )
    return photos


def _extract_part_lists(parts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    result = {
        "damaged": [],
        "missing": [],
        "intact": [],
        "uncertain": [],
    }
    for part in parts:
        status = part.get("status", "uncertain")
        part_id = part.get("part_id", "")
        if status == "damaged":
            result["damaged"].append(part_id)
        elif status == "missing":
            result["missing"].append(part_id)
        elif status == "intact":
            result["intact"].append(part_id)
        else:
            result["uncertain"].append(part_id)
    return result


# ---------------------------------------------------------------------------
# Single lead runner
# ---------------------------------------------------------------------------

async def run_lead(config: Dict[str, Any]) -> LeadResult:
    lead = config["lead"]
    run_id = config["run_id"]
    lead_dir = SAMPLE_ROOT / f"lead_{lead}"
    vehicle_info = {
        "brand": config["brand"],
        "model": config["model"],
        "year": config["year"],
    }

    result = LeadResult(
        lead=lead,
        run_id=run_id,
        vehicle_info=vehicle_info,
        photo_count=0,
    )

    start = time.perf_counter()
    try:
        photos = _collect_photos(lead_dir)
        result.photo_count = len(photos)
        if not photos:
            raise FileNotFoundError(f"No photos found in {lead_dir}")

        orchestrator_result = await assessment_orchestrator(photos, vehicle_info)
        parts = orchestrator_result.get("parts", [])
        result.parts = parts
        result.uncertain_items = orchestrator_result.get("uncertain_items", [])

        lists = _extract_part_lists(parts)
        result.damaged_part_ids = lists["damaged"]
        result.missing_part_ids = lists["missing"]
        result.intact_part_ids = lists["intact"]
        result.uncertain_part_ids = lists["uncertain"]

        plan = orchestrator_result.get("plan", {})
        covered = plan.get("workflow_plan", {}).get("priority_views", [])
        result.coverage_summary = {
            "covered_views": covered,
            "missing_critical_views": plan.get("workflow_plan", {}).get("missing_critical_views", []),
            "coverage_gaps": plan.get("coverage_gaps", []),
        }
        result.subagent_views = [r.get("view_id", "") for r in orchestrator_result.get("subagent_results", [])]

        # Run topology comparator to get severity / structural flag.
        vehicle_prior = orchestrator_result.get("vehicle_prior", {})
        topology = build_vehicle_topology(vehicle_info, vehicle_prior)
        states = [orchestrator_result.get("part_actual_states", [])]
        # Flatten PartActualState objects if present.
        flat_states = []
        for item in orchestrator_result.get("part_actual_states", []):
            if hasattr(item, "part_id"):
                flat_states.append(item)
        if not flat_states:
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _write_reports(results: List[LeadResult]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "leads": [],
    }

    for result in results:
        # Per-lead detailed JSON
        lead_file = OUTPUT_DIR / f"{result.lead}_run{result.run_id}_{timestamp}.json"
        with open(lead_file, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2, default=str)

        summary["leads"].append(
            {
                "lead": result.lead,
                "status": result.status,
                "photos": result.photo_count,
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
            }
        )

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
    for lead in summary["leads"]:
        rows += f"""
        <tr>
          <td>{lead['lead']}</td>
          <td class="{lead['status']}">{lead['status']}</td>
          <td>{lead['photos']}</td>
          <td>{lead['duration_seconds']}</td>
          <td>{lead['overall_severity']}</td>
          <td>{'是' if lead['structural_damage_flag'] else '否'}</td>
          <td>{lead['damaged_count']}</td>
          <td>{lead['missing_count']}</td>
          <td>{lead['uncertain_count']}</td>
          <td>{', '.join(lead['subagent_views'])}</td>
          <td>{', '.join(lead['damaged_parts'])}</td>
          <td style="max-width:300px;white-space:pre-wrap">{lead['error']}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Team Parallel Test Summary</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .success {{ color: green; font-weight: bold; }}
  .error {{ color: red; font-weight: bold; }}
</style>
</head>
<body>
<h1>Team Parallel Test Summary ({summary['timestamp']})</h1>
<table>
  <tr>
    <th>Lead</th>
    <th>Status</th>
    <th>Photos</th>
    <th>Duration(s)</th>
    <th>Severity</th>
    <th>Structural</th>
    <th>Damaged</th>
    <th>Missing</th>
    <th>Uncertain</th>
    <th>Views</th>
    <th>Damaged Parts</th>
    <th>Error</th>
  </tr>
  {rows}
</table>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"Starting team parallel test at {datetime.now().isoformat()}")
    print(f"Output directory: {OUTPUT_DIR}")

    tasks = [asyncio.create_task(run_lead(cfg), name=cfg["lead"]) for cfg in SELECTED_LEADS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    normalized: List[LeadResult] = []
    for result in results:
        if isinstance(result, Exception):
            placeholder = LeadResult(
                lead="unknown",
                vehicle_info={},
                photo_count=0,
                status="error",
                error=f"{type(result).__name__}: {result}\n{traceback.format_exc()}",
            )
            normalized.append(placeholder)
        else:
            normalized.append(result)

    _write_reports(normalized)

    print("\n" + "=" * 80)
    print(f"{'Lead':<10} {'Status':<8} {'Photos':<7} {'Time(s)':<9} {'Severity':<9} {'Structural':<10} {'Damaged':<8} {'Missing':<8} {'Uncertain':<10}")
    print("-" * 80)
    for r in normalized:
        print(
            f"{r.lead}_run{r.run_id:<4} {r.status:<8} {r.photo_count:<7} {r.duration_seconds:<9} "
            f"{r.overall_severity:<9} {'是' if r.structural_damage_flag else '否':<10} "
            f"{len(r.damaged_part_ids):<8} {len(r.missing_part_ids):<8} {len(r.uncertain_part_ids):<10}"
        )
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
