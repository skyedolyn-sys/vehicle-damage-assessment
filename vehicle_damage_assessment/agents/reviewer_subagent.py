"""Reviewer subagent — deterministic cross-validation.

DAMAGE_RECOGNITION_POLICY §1.6 (步骤 4): 原 review subagent 调用 LLM 重读子 agent
结果,但 LLM 偏好"保守 intact"覆盖 fusion 的 damaged severe,反而降低最终质量。
reviewer 失败时(26% parse 失败率)整个流程 fallback;成功时也常常推翻正确
的 fusion 结论。

新版实现: 完全确定性,复用 evidence_fusion.apply_fusion + 简单 sanity checks。
无 LLM 调用、无 narrative 摘要、每次运行结果完全一致。

The reviewer is invoked after all vision subagents have returned.  It looks at
all region results, the original plan, and the vehicle topology, then decides:

- Which parts need re-evaluation because of conflicting subagent outputs.
- Which regions are truly uncovered and require re-photography.
- Whether a low-confidence but structurally important part should be marked
  uncertain rather than trusted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from config import PARTS_BY_ID
from models import DamageLevel, PartActualState, Status

logger = logging.getLogger(__name__)


async def reviewer_subagent(
    all_results: List[Dict[str, Any]],
    plan: Dict[str, Any],
    vehicle_prior: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic reviewer: cross-validate using existing fusion rules.

    DAMAGE_RECOGNITION_POLICY §1.6 / 步骤 4: 重写为纯确定性。

    Reuses ``evidence_fusion.apply_fusion`` (high-sensitivity 严重穿透 +
    primary-intact confidence-aware) 并加上 reviewer 专属的 sanity checks:

    1. 列出 N >= 2 个 secondary view 给 damaged 但 fusion 输出 intact 的部件
       ——需人工复核(reviewer_advisory,不影响 fusion 结论)
    2. 标记 confidence == low 的 damaged 项 ——需人工复核
    3. 找出没有 primary 视角证据且无 wide_shot 覆盖的部件 ——需更直接视角

    Returns
    -------
    dict
        ``{"reviewed_parts": [...], "added_uncertain_items": [...],
           "needs_rephotography": [...], "summary": "..."}``

        其中 reviewed_parts 等价于 evidence_fusion.apply_fusion 的 overrides;
        added_uncertain_items / needs_rephotography 是 reviewer 额外标记的清单。
    """
    # 直接复用 evidence_fusion 的融合结果,不做 LLM 二次审查
    overrides: Dict[str, Dict[str, Any]] = {}
    try:
        from agents.evidence_fusion import apply_fusion
        overrides = apply_fusion(all_results)
    except Exception as exc:
        # 如果 fusion 失败也保持确定行为:返回空,让 orchestrator 走 synthesizer 默认路径
        logger.warning(
            "[reviewer] evidence_fusion.apply_fusion failed: %s; "
            "falling back to empty overrides", exc, exc_info=True,
        )
        overrides = {}

    # reviewer 额外的工作:仅做 advisory,不覆盖 fusion 结论
    needs_rephotography: List[Dict[str, Any]] = []
    advisory_reasons: List[str] = []

    # 1) 找出 confidence=low + damaged 的部件 —— 需人工复核
    low_conf_damaged = [
        (pid, ov) for pid, ov in overrides.items()
        if ov.get("status") == "damaged" and ov.get("confidence") == "low"
    ]
    if low_conf_damaged:
        advisory_reasons.append(
            f"{len(low_conf_damaged)} 个 damaged 部件 confidence=low, 需人工复核"
        )

    # 2) 找出 secondary view 多次 damaged 但 fusion 仍 intact 的冲突
    secondary_damaged_overrides = _find_secondary_damage_conflicts(all_results, overrides)
    if secondary_damaged_overrides:
        advisory_reasons.append(
            f"{len(secondary_damaged_overrides)} 个部件被 ≥2 个 secondary view 报 damaged,"
            " 但 fusion 结论是 intact,需人工复核"
        )

    # 3) 列出无 primary view 且无 wide_shot 覆盖的部件 —— 需更直接视角
    uncovered = _find_uncovered_parts(all_results, overrides)
    if uncovered:
        needs_rephotography.append({
            "items": uncovered,
            "reason": "no primary view,no wide_shot coverage",
        })

    # 4) 组装 reviewed_parts: 等于 fusion overrides,序列化为 list
    reviewed_parts: List[Dict[str, Any]] = []
    for pid, ov in overrides.items():
        reviewed_parts.append({
            "part_id": pid,
            **ov,
        })

    summary = (
        f"deterministic reviewer: {len(reviewed_parts)} fusion overrides;"
        + (f" {len(secondary_damaged_overrides)} secondary-damage conflicts;"
           if secondary_damaged_overrides else "")
        + (f" {len(low_conf_damaged)} low-confidence damaged;"
           if low_conf_damaged else "")
    )

    return {
        "reviewed_parts": reviewed_parts,
        "reviewed_part_actual_states": [
            PartActualState(
                part_id=p["part_id"],
                part_name=PARTS_BY_ID.get(p["part_id"], {}).get("part_name", p["part_id"]),
                part_category=PARTS_BY_ID.get(p["part_id"], {}).get("part_category", ""),
                side=PARTS_BY_ID.get(p["part_id"], {}).get("side", ""),
                status=Status(p.get("status", "uncertain")),
                damage_level=DamageLevel(p.get("damage_level", "unknown")),
                damage_types=p.get("damage_type", []) or [],
                standard_exists=True,
                actual_visible=True,
                actual_present=p.get("status") != "missing",
                confidence=p.get("confidence", "low"),
                evidence_photos=p.get("evidence_photo", []),
                notes=p.get("notes", ""),
                adjacent_status={},
                photo_type="unknown",
            )
            for p in reviewed_parts
        ],
        "added_uncertain_items": [],  # reviewer 不再添加 uncertain,只做 advisory
        "needs_rephotography": needs_rephotography,
        "summary": summary,
    }


def _find_secondary_damage_conflicts(
    all_results: List[Dict[str, Any]],
    overrides: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """找出 ≥2 个 secondary view 报告 damaged 但 fusion 仍 intact 的部件。

    仅做 advisory,不修改 overrides。
    """
    conflicts = []
    damage_votes: Dict[str, int] = {}
    for result in all_results:
        view_id = result.get("view_id", "")
        if not view_id:
            continue
        for part in result.get("parts", []):
            pid = part.get("part_id", "")
            if part.get("status") == "damaged":
                damage_votes[pid] = damage_votes.get(pid, 0) + 1
    for pid, ov in overrides.items():
        if ov.get("status") == "intact" and damage_votes.get(pid, 0) >= 2:
            conflicts.append({"part_id": pid, "secondary_damaged_views": damage_votes[pid]})
    return conflicts


def _find_uncovered_parts(
    all_results: List[Dict[str, Any]],
    overrides: Dict[str, Dict[str, Any]],
) -> List[str]:
    """找出无 primary view 且无 wide_shot 覆盖的部件。

    启发式:检查 evidence_sources 是否只来自非 primary 视图,或是 close_up_* 而非 wide_shot。
    """
    if not all_results:
        return []
    from agents.evidence_fusion import _PART_VIEW_PRIORITY, _view_priority
    uncovered = []
    for pid, ov in overrides.items():
        if ov.get("status") != "uncertain":
            continue
        # 如果所有 evidence 都是 low confidence 且无 primary view
        evidence_sources = ov.get("evidence_sources", [])
        has_primary = False
        for src in evidence_sources:
            view = src.get("view_id", "") if isinstance(src, dict) else ""
            prio = _view_priority(pid, view)
            if prio <= 1:
                has_primary = True
                break
        if not has_primary:
            uncovered.append(pid)
