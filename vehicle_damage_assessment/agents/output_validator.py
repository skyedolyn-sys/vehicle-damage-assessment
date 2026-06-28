import copy
from typing import Dict, Any, List, Optional

from config import PARTS_BY_ID
from models.topology import VehicleTopology
from models.part_state import PartActualState, Status
from models.assessment import DamageAssessment
from agents.topology_comparator import compare_topology


def validate_and_enrich(
    raw_result: Dict[str, Any],
    topology: Optional[VehicleTopology] = None,
) -> Dict[str, Any]:
    """
    输出校验与增强
    1. 确保所有部件都有结论
    2. 计算 summary
    3. 触发 structural_damage_flag
    4. 校验字段完整性

    When *topology* is provided, the new topology-aware path is used
    (topology comparator + structural pattern detection).  Otherwise the
    legacy heuristic path is kept for backward compatibility.
    """
    if topology is not None:
        return _validate_and_enrich_with_topology(raw_result, topology)

    # Legacy path — unchanged behaviour
    result = copy.deepcopy(raw_result)
    parts = result.get("parts", [])

    # 用 part_id 索引已有结果
    parts_by_id = {p.get("part_id"): p for p in parts if p.get("part_id")}

    # 补全缺失部件
    complete_parts = []
    for part_id, base_info in PARTS_BY_ID.items():
        if part_id in parts_by_id:
            p = parts_by_id[part_id]
        else:
            p = {
                "part_id": part_id,
                "part_name": base_info["part_name"],
                "part_category": base_info["part_category"],
                "side": base_info["side"],
                "status": "uncertain",
                "damage_level": "unknown",
                "damage_type": [],
                "confidence": "low",
                "evidence_photo": [],
                "notes": "模型未输出该部件结论，默认无法判断",
            }
        # 补齐字段
        p.setdefault("part_name", base_info["part_name"])
        p.setdefault("part_category", base_info["part_category"])
        p.setdefault("side", base_info["side"])
        p.setdefault("status", "uncertain")
        p.setdefault("damage_level", "unknown")
        p.setdefault("damage_type", [])
        p.setdefault("confidence", "low")
        p.setdefault("evidence_photo", [])
        p.setdefault("notes", "")
        complete_parts.append(p)

    result["parts"] = complete_parts

    # 计算 summary
    damaged_count = sum(1 for p in complete_parts if p["status"] == "damaged")
    intact_count = sum(1 for p in complete_parts if p["status"] == "intact")
    uncertain_count = sum(1 for p in complete_parts if p["status"] == "uncertain")
    total = len(complete_parts)

    # 判断主要损伤区域
    region_severity = {}
    for p in complete_parts:
        if p["status"] == "damaged":
            cat = p["part_category"]
            level = {"light": 1, "moderate": 2, "severe": 3, "unknown": 0}.get(p["damage_level"], 0)
            region_severity[cat] = region_severity.get(cat, 0) + level

    primary_zone = "multiple"
    if region_severity:
        primary_zone = max(region_severity, key=region_severity.get)

    # 整体严重等级
    severe_count = sum(1 for p in complete_parts if p["damage_level"] == "severe")
    moderate_count = sum(1 for p in complete_parts if p["damage_level"] == "moderate")
    if severe_count >= 3:
        overall_severity = "severe"
    elif severe_count >= 1 or moderate_count >= 3:
        overall_severity = "moderate"
    elif moderate_count >= 1:
        overall_severity = "light"
    else:
        overall_severity = "none"

    # 触发结构性事故规则
    structural_flag, structural_reasons = check_structural_damage(complete_parts)

    result["assessment_summary"] = {
        "overall_severity": overall_severity,
        "primary_damage_zone": primary_zone,
        "structural_damage_flag": structural_flag,
        "total_parts": total,
        "damaged_parts_count": damaged_count,
        "intact_parts_count": intact_count,
        "uncertain_parts_count": uncertain_count,
    }

    result["structural_damage_reasoning"] = {
        "triggered": structural_flag,
        "rules_matched": structural_reasons,
        "description": (
            "触发整车结构性事故标记。即使车顶等部分区域完好，因关键区域/结构件严重受损，不宜按普通外观损伤处理。"
            if structural_flag else "未触发整车结构性事故标记。"
        ),
    }

    result.setdefault("uncertain_items", [])

    return result


# ---------------------------------------------------------------------------
# Topology-aware path
# ---------------------------------------------------------------------------

def _validate_and_enrich_with_topology(
    raw_result: Dict[str, Any],
    topology: VehicleTopology,
) -> Dict[str, Any]:
    """Validate and enrich using the topology comparator.

    Converts legacy ``parts`` dicts to :class:`PartActualState`, runs the
    topology comparator, and returns a result that is backward-compatible
    with the old format while adding new topology keys.
    """
    result = copy.deepcopy(raw_result)

    # 1. Convert legacy parts dicts to PartActualState objects
    legacy_parts = result.get("parts", [])
    actual_states: List[PartActualState] = []
    for p in legacy_parts:
        if isinstance(p, dict) and p.get("part_id"):
            actual_states.append(PartActualState.from_legacy_dict(p))

    # 2. Run topology comparison
    assessment = compare_topology(topology, actual_states)

    # 3. Use assessment.to_legacy_result() as the base result
    enriched = assessment.to_legacy_result()

    # 4. Ensure backward-compatible keys exist
    enriched.setdefault("uncertain_items", result.get("uncertain_items", []))

    # Ensure structural_damage_reasoning dict exists (old format)
    structural_flag = enriched.get("structural_damage_flag", False)
    patterns = enriched.get("structural_patterns", [])
    reasoning_rules = []
    for pat in patterns:
        reasoning_rules.append(
            f"{pat['pattern_name']}: {pat['description']} (severity={pat['severity']}, confidence={pat['confidence']})"
        )
    enriched["structural_damage_reasoning"] = {
        "triggered": structural_flag,
        "rules_matched": reasoning_rules,
        "description": (
            "触发整车结构性事故标记。基于拓扑模式识别检测到结构性损伤模式。"
            if structural_flag else "未触发整车结构性事故标记。"
        ),
    }

    # 5. Ensure assessment_summary has old-style counts
    parts_list = enriched.get("parts", [])
    enriched["assessment_summary"] = {
        "overall_severity": enriched.get("overall_severity", ""),
        "primary_damage_zone": enriched.get("primary_damage_zone", ""),
        "structural_damage_flag": structural_flag,
        "total_parts": len(parts_list),
        "damaged_parts_count": len(enriched.get("damaged_parts", [])),
        "intact_parts_count": len(enriched.get("intact_parts", [])),
        "uncertain_parts_count": len(enriched.get("uncertain_parts", [])),
        "missing_parts_count": len(enriched.get("missing_parts", [])),
    }

    return enriched


def check_structural_damage(parts: List[Dict[str, Any]]) -> (bool, List[str]):
    """检查是否触发整车结构性事故"""
    reasons = []

    # 规则 A：任一区域同时有 ≥3 个 severe
    region_counts = {}
    for p in parts:
        if p["status"] == "damaged" and p["damage_level"] == "severe":
            cat = p["part_category"]
            region_counts[cat] = region_counts.get(cat, 0) + 1
    for cat, count in region_counts.items():
        if count >= 3:
            reasons.append(f"规则A：{cat}区域同时有{count}个部件重度损坏")

    # 规则 B：引擎盖 + 任一车门 + 任一翼子板同时 severe
    hood_severe = any(p["part_id"] == "hood" and p["damage_level"] == "severe" for p in parts)
    door_severe = any(
        p["part_id"].startswith("door_") and p["damage_level"] == "severe" for p in parts
    )
    fender_severe = any(
        p["part_id"].startswith("fender_") and p["damage_level"] == "severe" for p in parts
    )
    if hood_severe and door_severe and fender_severe:
        reasons.append("规则B：引擎盖 + 车门 + 翼子板同时重度损坏")

    # 规则 C：结构件受损
    structural_parts = ["pillar_a_left", "pillar_a_right", "roof_front", "roof_middle", "roof_rear"]
    for p in parts:
        if p["part_id"] in structural_parts and p["status"] == "damaged":
            if p["damage_level"] in ["moderate", "severe"]:
                reasons.append(f"规则C：结构件 {p['part_name']} 出现{p['damage_level']}损伤")

    return len(reasons) > 0, reasons
