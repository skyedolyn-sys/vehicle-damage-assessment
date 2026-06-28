"""Reviewer subagent — checks coverage gaps, conflicts and low-confidence items.

The reviewer is invoked after all vision subagents have returned.  It looks at
all region results, the original plan, and the vehicle topology, then decides:

- Which parts need re-evaluation because of conflicting subagent outputs.
- Which regions are truly uncovered and require re-photography.
- Whether a low-confidence but structurally important part should be marked
  uncertain rather than trusted.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agents.minimax_client import call_minimax, build_image_content, extract_json
from agents.view_mapping import get_display_name
from config import PARTS_BY_ID
from models import PartActualState, Status, DamageLevel


_SYSTEM_PROMPT = """你是车辆损伤评估复核专家。你的任务是在所有视角子Agent识别完成后，检查覆盖缺口、冲突结论和低置信度项目，并给出最终复核意见。

输入包含：
1. 车型先验信息
2. 每张照片的视角分配
3. 各视角子Agent识别的部件状态
4. 覆盖缺口（缺少哪些视角）

输出 JSON 格式：
{
  "reviewed_parts": [
    {
      "part_id": "door_rear_left",
      "part_name": "左后门",
      "status": "uncertain",
      "damage_level": "unknown",
      "damage_type": [],
      "confidence": "low",
      "evidence_photo": [],
      "notes": "车头左侧照片无法覆盖左后门，应标记为无法判断",
      "action": "keep|revise|needs_rephoto"
    }
  ],
  "added_uncertain_items": [
    {
      "item": "右后翼子板状态",
      "reason": "缺少右侧或车尾右侧照片",
      "suggested_action": "补拍右侧照片"
    }
  ],
  "needs_rephotography": [
    {
      "view": "right",
      "display_name": "车辆右侧",
      "impacted_parts": ["右前门", "右后门", "右后视镜", "右后翼子板"],
      "reason": "右侧无任何照片覆盖"
    }
  ],
  "summary": "右侧和车顶完全缺失，车尾损伤可信度高，车头完好"
}

复核规则：
1. 如果某部件在不同视角子Agent中有冲突结论，取最保守结论（损伤程度往高报，置信度往低调），并在 notes 中说明冲突。
2. 如果某区域完全没有照片覆盖，该区域所有部件标记为 uncertain，并在 needs_rephotography 中列出。
3. 如果某部件虽然被覆盖但 confidence=low，且属于安全关键部件（大灯/尾灯/后视镜/结构件），优先标记为 uncertain。
4. action 含义：keep=维持原结论；revise=按复核意见修改；needs_rephoto=必须补拍。
5. 只输出 JSON，不要额外文字。
"""


def _build_conflict_prompt(
    all_results: List[Dict[str, Any]],
    plan: Dict[str, Any],
    vehicle_prior: Dict[str, Any],
) -> str:
    """Build the user prompt for the reviewer."""
    vehicle_name = vehicle_prior.get("vehicle", "该车")

    lines: List[str] = [f"车型：{vehicle_name}", "", "【照片视角分配】"]
    for entry in plan.get("photo_views", []):
        lines.append(
            f"- {entry['photo_id']}: {entry.get('view_id', 'unknown')} "
            f"(confidence={entry.get('confidence', 'low')}, reason={entry.get('reason', '')})"
        )

    lines.extend(["", "【覆盖缺口】"])
    for gap in plan.get("coverage_gaps", []):
        lines.append(
            f"- 缺少 {gap.get('missing_view', '')}，影响区域: "
            f"{', '.join(gap.get('impacted_regions', []))}，建议: {gap.get('suggested_action', '')}"
        )

    lines.extend(["", "【各视角识别结果】"])
    for result in all_results:
        view_id = result.get("view_id", "unknown")
        lines.append(f"\n视角: {view_id}")
        for state in result.get("part_actual_states", []):
            lines.append(
                f"  - {state.part_name}({state.part_id}): {state.status.value} "
                f"level={state.damage_level.value} confidence={state.confidence} "
                f"photos={state.evidence_photos}"
            )

    return "\n".join(lines)


def _parse_reviewed_parts(data: List[Dict[str, Any]]) -> List[PartActualState]:
    """Parse reviewed part dicts into PartActualState objects."""
    states: List[PartActualState] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        status_str = item.get("status", "uncertain")
        try:
            status = Status(status_str)
        except ValueError:
            status = Status.UNCERTAIN

        level_str = item.get("damage_level", "unknown")
        try:
            damage_level = DamageLevel(level_str)
        except ValueError:
            damage_level = DamageLevel.UNKNOWN

        damage_types = item.get("damage_type", [])
        if isinstance(damage_types, str):
            if damage_types and damage_types != "none":
                damage_types = [t.strip() for t in damage_types.split(",") if t.strip()]
            else:
                damage_types = []

        evidence_photos = item.get("evidence_photo", [])
        if isinstance(evidence_photos, str):
            if evidence_photos:
                evidence_photos = [p.strip() for p in evidence_photos.split(",") if p.strip()]
            else:
                evidence_photos = []

        part_info = PARTS_BY_ID.get(item.get("part_id", ""), {})
        states.append(
            PartActualState(
                part_id=item.get("part_id", ""),
                part_name=item.get("part_name", part_info.get("part_name", "")),
                region=item.get("region", part_info.get("part_category", "")),
                side=item.get("side", part_info.get("side", "")),
                status=status,
                damage_level=damage_level,
                damage_types=damage_types,
                confidence=item.get("confidence", "low"),
                evidence_photos=evidence_photos,
                notes=item.get("notes", ""),
            )
        )
    return states


async def reviewer_subagent(
    all_results: List[Dict[str, Any]],
    plan: Dict[str, Any],
    vehicle_prior: Dict[str, Any],
) -> Dict[str, Any]:
    """Review all vision subagent results and produce corrections.

    Parameters
    ----------
    all_results:
        List of dicts returned by ``vision_subagent`` for each view.
    plan:
        Planner output with photo views and coverage gaps.
    vehicle_prior:
        Output from ``vehicle_prior_agent``.

    Returns
    -------
    dict
        ``{"reviewed_parts": [...], "added_uncertain_items": [...], "needs_rephotography": [...], "summary": "..."}``
    """
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": _SYSTEM_PROMPT},
        {"type": "text", "text": _build_conflict_prompt(all_results, plan, vehicle_prior)},
    ]

    messages = [{"role": "user", "content": content}]
    raw = await call_minimax(messages, temperature=0.1, max_tokens=3000)
    result = extract_json(raw) or {}

    if not isinstance(result, dict):
        result = {}

    reviewed_parts = _parse_reviewed_parts(result.get("reviewed_parts", []))

    return {
        "reviewed_parts": [p.to_legacy_dict() for p in reviewed_parts],
        "reviewed_part_actual_states": reviewed_parts,
        "added_uncertain_items": [
            item for item in result.get("added_uncertain_items", [])
            if isinstance(item, dict)
        ],
        "needs_rephotography": [
            item for item in result.get("needs_rephotography", [])
            if isinstance(item, dict)
        ],
        "summary": result.get("summary", ""),
    }
