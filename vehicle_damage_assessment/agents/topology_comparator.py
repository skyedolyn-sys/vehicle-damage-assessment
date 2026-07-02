"""Topology comparator — compare standard VehicleTopology against actual PartActualState list.

Produces a DamageAssessment with structural damage pattern detection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set

from agents.rules import load_part_profile, load_region_units, load_view_weights
from agents.view_mapping import canonicalize_view_id
from config import PARTS_BY_ID
from models.assessment import DamageAssessment, StructuralDamagePattern
from models.part_state import DamageLevel, PartActualState, Status
from models.topology import VehicleTopology


# Damage level weights for primary_damage_zone calculation.
_DAMAGE_WEIGHTS: Dict[str, int] = {
    DamageLevel.NONE.value: 0,
    DamageLevel.LIGHT.value: 1,
    DamageLevel.MODERATE.value: 2,
    DamageLevel.SEVERE.value: 3,
    DamageLevel.UNKNOWN.value: 1,
}

# Symmetric part pairs for symmetric_damage detection.
_SYMMETRIC_PAIRS: List[tuple[str, str]] = [
    ("headlight_front_left", "headlight_front_right"),
    ("taillight_rear_left", "taillight_rear_right"),
    ("fender_front_left", "fender_front_right"),
    ("fender_rear_left", "fender_rear_right"),
    ("door_front_left", "door_front_right"),
    ("door_rear_left", "door_rear_right"),
    ("mirror_left", "mirror_right"),
]

# Safety-critical parts for safety_critical_missing detection.
_SAFETY_CRITICAL_PARTS: Set[str] = {
    "headlight_front_left",
    "headlight_front_right",
    "taillight_rear_left",
    "taillight_rear_right",
    "mirror_left",
    "mirror_right",
}

# Pure-topology consistency rules are enforced here instead of the synthesizer
# so that structural pattern recognition stays close to the topology graph.
_ROOF_PARTS: Set[str] = load_part_profile("roof")
_REGION_UNITS: Dict[str, Set[str]] = load_region_units()

_VIEW_WEIGHTS_CONFIG = load_view_weights()
_VIEW_WEIGHTS: Dict[str, Any] = _VIEW_WEIGHTS_CONFIG.get("view_weights", {})

_STATUS_PRIORITY: Dict[Status, int] = {
    Status.MISSING: 4,
    Status.DAMAGED: 3,
    Status.UNCERTAIN: 2,
    Status.INTACT: 1,
    Status.NOT_APPLICABLE: 0,
}

_LEVEL_PRIORITY: Dict[DamageLevel, int] = {
    DamageLevel.SEVERE: 4,
    DamageLevel.MODERATE: 3,
    DamageLevel.LIGHT: 2,
    DamageLevel.NONE: 1,
    DamageLevel.UNKNOWN: 0,
}


def _append_note(part: PartActualState, note: str) -> PartActualState:
    """Return a new PartActualState with *note* appended to its notes field."""
    existing = part.notes or ""
    new_notes = f"{existing}；{note}" if existing else note
    return replace(part, notes=new_notes)


def _set_damaged_severe(part: PartActualState, damage_type: str, note: str) -> PartActualState:
    """Return a new PartActualState marked damaged severe with a note."""
    new_part = replace(
        part,
        status=Status.DAMAGED,
        damage_level=DamageLevel.SEVERE,
        damage_types=[damage_type],
        confidence="low",
    )
    return _append_note(new_part, note)


def _severe_neighbors(
    neighbors: List[PartActualState],
    allowed: Set[str],
) -> List[str]:
    """Return ids of adjacent parts that are damaged/missing and severe."""
    return [
        n.part_id for n in neighbors
        if n.part_id in allowed
        and n.status in (Status.DAMAGED, Status.MISSING)
        and n.damage_level == DamageLevel.SEVERE
    ]


def _has_source_status(part: PartActualState, status: str, regions: Set[str]) -> bool:
    """Return True if part has an evidence source with the given status in one of the regions."""
    canonical_regions = {canonicalize_view_id(r) for r in regions}
    return any(
        src.get("status") == status and canonicalize_view_id(src.get("region", "")) in canonical_regions
        for src in part.evidence_sources
    )


class TopologyConsistencyEnforcer:
    """Enforce geometric/consistency rules that depend only on topology.

    These rules used to live in the synthesizer. They are moved here because
    they need no evidence-source metadata beyond the resolved part state and
    the adjacency graph.
    """

    def __init__(self, topology: VehicleTopology) -> None:
        self.topology = topology

    def enforce(self, parts: List[PartActualState]) -> List[PartActualState]:
        """Run all topology-only consistency rules and return updated parts."""
        by_id = {p.part_id: p for p in parts}

        # 1. Region-unit unification (physically connected groups).
        by_id = self._unify_region_units(by_id)

        # 2. Infer missing roof parts from intact roof neighbors.
        by_id = self._infer_missing_roof_parts(by_id)

        # 3. Adjacency rules that need only topology + resolved status/level.
        updated: List[PartActualState] = []
        for part in parts:
            state = by_id.get(part.part_id, part)
            updated.append(self._apply_adjacency_rules(state, by_id))

        return updated

    # ------------------------------------------------------------------
    # Region units
    # ------------------------------------------------------------------

    def _unify_region_units(
        self, by_id: Dict[str, PartActualState]
    ) -> Dict[str, PartActualState]:
        """Force unified conclusions within physically connected region units.

        Worst status wins, but damaged overrides missing when both are present.
        """
        result = dict(by_id)
        for unit_name, members in _REGION_UNITS.items():
            present = [result[m] for m in members if m in result]
            if not present:
                continue

            statuses = [p.status for p in present]
            levels = [p.damage_level for p in present]

            unit_status = (
                Status.DAMAGED
                if Status.DAMAGED in statuses
                else max(statuses, key=lambda s: _STATUS_PRIORITY.get(s, 0))
            )
            unit_level = (
                DamageLevel.SEVERE
                if DamageLevel.SEVERE in levels
                else max(levels, key=lambda lvl: _LEVEL_PRIORITY.get(lvl, 0))
            )

            note = f"区域单元统一结论（{unit_name}）"
            for member in present:
                new_part = replace(
                    member,
                    status=unit_status,
                    damage_level=unit_level,
                )
                if not (member.notes or "").startswith(note):
                    new_part = _append_note(new_part, note)
                result[member.part_id] = new_part

        return result

    # ------------------------------------------------------------------
    # Missing-roof inference
    # ------------------------------------------------------------------

    def _infer_missing_roof_parts(
        self, by_id: Dict[str, PartActualState]
    ) -> Dict[str, PartActualState]:
        """Infer an intact roof part when it is uncertain and roof neighbours are intact."""
        result = dict(by_id)

        for part_id, state in list(result.items()):
            if state.status != Status.UNCERTAIN or part_id not in _ROOF_PARTS:
                continue

            node = self.topology.get_node(part_id)
            if node is None:
                continue

            # DAMAGE_RECOGNITION_POLICY §3.2: 相邻 intact 必须是 medium+ confidence 且至少
            # 有一个来自 top / close_up_damage 视角直接观察,否则不允许推断 intact。
            intact_roof_neighbors = []
            has_direct_observation = False
            for adj in self.topology.get_adjacent(part_id):
                if adj.part_id not in _ROOF_PARTS:
                    continue
                neighbor = result.get(adj.part_id)
                if neighbor is None or neighbor.status != Status.INTACT:
                    continue
                if neighbor.confidence not in ("medium", "high"):
                    continue
                intact_roof_neighbors.append(adj.part_id)
                if any(
                    src.get("view_id") in ("top",) or src.get("photo_type") == "close_up_damage"
                    for src in getattr(neighbor, "evidence_sources", [])
                ):
                    has_direct_observation = True
            if not intact_roof_neighbors or not has_direct_observation:
                continue

            result[part_id] = replace(
                state,
                status=Status.INTACT,
                damage_level=DamageLevel.NONE,
                confidence="low",
                notes=(
                    f"该部件无直接视角覆盖，从相邻 intact 车顶部件推断："
                    f"{', '.join(intact_roof_neighbors)}"
                ),
            )

        return result

    # ------------------------------------------------------------------
    # Adjacency rules
    # ------------------------------------------------------------------

    def _apply_adjacency_rules(
        self,
        part: PartActualState,
        by_id: Dict[str, PartActualState],
    ) -> PartActualState:
        """Apply pure-topology adjacency rules to a single part."""
        node = self.topology.get_node(part.part_id)
        if node is None:
            return part

        adjacents = self.topology.get_adjacent(part.part_id)
        neighbors = [by_id.get(adj.part_id) for adj in adjacents]
        neighbors = [n for n in neighbors if n is not None]

        new_part = part

        # Rule 1: intact next to damaged/missing -> cap confidence at medium.
        if new_part.status == Status.INTACT and new_part.confidence == "high":
            damaged = [
                n.part_id
                for n in neighbors
                if n.status in (Status.DAMAGED, Status.MISSING)
            ]
            if damaged:
                new_part = replace(new_part, confidence="medium")
                new_part = _append_note(
                    new_part, f"相邻部件存在损伤，降低置信度：{', '.join(damaged)}"
                )

        # Rule 2: sunroof level should not be lower than adjacent damaged roof.
        if part.part_id == "sunroof_glass" and new_part.status == Status.DAMAGED:
            for neighbor in neighbors:
                if (
                    neighbor.part_id in _ROOF_PARTS
                    and neighbor.status == Status.DAMAGED
                    and _LEVEL_PRIORITY.get(neighbor.damage_level, 0)
                    > _LEVEL_PRIORITY.get(new_part.damage_level, 0)
                ):
                    new_part = replace(
                        new_part, damage_level=neighbor.damage_level
                    )
                    new_part = _append_note(
                        new_part,
                        f"相邻车顶部件 {neighbor.part_id} 为 {neighbor.damage_level.value}，天窗级别同步上调",
                    )

        # Rule 3: door next to severely damaged fender/bumper cannot stay intact.
        if part.part_id.startswith("door_") and new_part.status == Status.INTACT:
            severe = [
                n.part_id
                for n in neighbors
                if n.status == Status.DAMAGED
                and n.damage_level == DamageLevel.SEVERE
            ]
            if severe:
                new_part = replace(new_part, confidence="low")
                new_part = _append_note(
                    new_part,
                    f"相邻部件严重受损，降低车门置信度：{', '.join(severe)}",
                )

        # Rule 9: Pillar-to-roof/structural propagation.
        # Pillars are structural safety components. If a pillar is uncertain or
        # missing and any adjacent structural part (same-side roof, windshield,
        # or neighboring pillar) is damaged severe, the pillar is likely damaged
        # too due to structural continuity.  Only propagate to pillars that have
        # at least one evidence source (i.e. were actually observed/assessed).
        if (
            part.part_id.startswith("pillar_")
            and new_part.status in (Status.UNCERTAIN, Status.MISSING)
            and new_part.evidence_sources
        ):
            side = part.part_id.split("_")[-1]
            allowed = {
                "roof_front",
                "roof_middle",
                "roof_rear",
                "windshield_front",
                "windshield_rear",
                f"pillar_a_{side}",
                f"pillar_b_{side}",
                f"pillar_c_{side}",
            }
            allowed = allowed - {part.part_id}
            severe_neighbors = _severe_neighbors(neighbors, allowed)
            if severe_neighbors:
                primary_regions = {canonicalize_view_id(v) for v in _VIEW_WEIGHTS.get(part.part_id, {}).get("primary", [])}
                if not _has_source_status(new_part, "intact", primary_regions):
                    new_part = _set_damaged_severe(
                        new_part,
                        "deformation",
                        f"推断：相邻结构件严重损伤（{', '.join(severe_neighbors)}），立柱同步受损",
                    )

        # Rule 10: Roof-front edge propagation.
        # If roof_front is intact or uncertain, and a front corner view reports
        # severe damage to the windshield_front or same-side pillar_a, and there
        # is no top-view intact evidence that clearly refutes it, infer roof_front
        # as damaged severe. Front collisions often crumple the roof-front edge
        # where the A-pillars meet the roof rail.
        if (
            part.part_id == "roof_front"
            and new_part.status in (Status.INTACT, Status.UNCERTAIN)
            and new_part.evidence_sources
        ):
            severe_front_structural = []
            for side in ("left", "right"):
                pillar_id = f"pillar_a_{side}"
                neighbor = by_id.get(pillar_id)
                if neighbor and neighbor.status in (Status.DAMAGED, Status.MISSING) and neighbor.damage_level == DamageLevel.SEVERE:
                    severe_front_structural.append(pillar_id)
            windshield = by_id.get("windshield_front")
            if windshield and windshield.status in (Status.DAMAGED, Status.MISSING) and windshield.damage_level == DamageLevel.SEVERE:
                severe_front_structural.append("windshield_front")
            if severe_front_structural and not _has_source_status(new_part, "intact", {"top"}):
                new_part = _set_damaged_severe(
                    new_part,
                    "deformation",
                    f"前部结构件严重受损（{', '.join(severe_front_structural)}），推断车顶前缘同步受损",
                )

        # Rule 11: Roof-middle/rear propagation.
        # If roof_middle or roof_rear is uncertain and adjacent roof/pillar parts
        # are damaged severe, infer damaged severe. This handles cases where side
        # views show roof rail deformation but the top view is missing or unclear.
        if (
            part.part_id in ("roof_middle", "roof_rear")
            and new_part.status == Status.UNCERTAIN
            and new_part.evidence_sources
        ):
            if part.part_id == "roof_middle":
                allowed = {"roof_front", "roof_rear", "sunroof_glass", "roof_rack"}
            else:
                allowed = {"roof_middle", "windshield_rear", "trunk_lid", "tailgate"}
            for side in ("left", "right"):
                allowed.add(f"pillar_b_{side}")
                if part.part_id == "roof_rear":
                    allowed.add(f"pillar_c_{side}")
            severe_neighbors = _severe_neighbors(neighbors, allowed)
            if severe_neighbors and not _has_source_status(new_part, "intact", {"top"}):
                new_part = _set_damaged_severe(
                    new_part,
                    "deformation",
                    f"相邻车顶/立柱结构严重受损（{', '.join(severe_neighbors)}），推断该部位同步受损",
                )

        return new_part


class TopologyComparator:
    """Compare standard topology against actual part states.

    Steps:
    1. Iterate all topology nodes.
    2. Look up actual state per node; synthesise UNCERTAIN when missing.
    3. Classify each node as intact / damaged / missing / uncertain.
    4. Detect structural damage patterns (regional mass, cross-region,
       structural component, symmetric, safety-critical missing).
    5. Compute overall severity and primary damage zone.
    6. Build summary counts and return DamageAssessment.
    """

    def __init__(self, topology: VehicleTopology) -> None:
        self.topology = topology

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self, actual_states: List[PartActualState]) -> DamageAssessment:
        """Run full comparison and return a DamageAssessment."""
        actual_by_id: Dict[str, PartActualState] = {
            s.part_id: s for s in actual_states
        }

        parts: List[PartActualState] = []
        missing_parts: List[str] = []
        damaged_parts: List[str] = []
        intact_parts: List[str] = []
        uncertain_parts: List[str] = []

        # 1. Walk every topology node.
        for node_id, node in self.topology.nodes.items():
            state = self._resolve_state(node_id, node, actual_by_id)
            parts.append(state)

        # 2. Enforce topology-only consistency rules.
        parts = TopologyConsistencyEnforcer(self.topology).enforce(parts)

        # Recompute classified lists after enforcement.
        missing_parts = [p.part_id for p in parts if p.status == Status.MISSING]
        damaged_parts = [p.part_id for p in parts if p.status == Status.DAMAGED]
        intact_parts = [p.part_id for p in parts if p.status == Status.INTACT]
        uncertain_parts = [p.part_id for p in parts if p.status == Status.UNCERTAIN]

        # 3. Structural pattern detection.
        damaged_node_ids: Set[str] = set(missing_parts) | set(damaged_parts)
        patterns = self._detect_patterns(damaged_node_ids, parts)

        # 3. Severity rollup.
        overall_severity = self._compute_overall_severity(patterns)

        # 4. Primary damage zone.
        primary_zone = self._compute_primary_zone(parts)

        # 5. Summary dict (old output_validator style).
        summary = {
            "total_parts": len(parts),
            "missing_count": len(missing_parts),
            "damaged_count": len(damaged_parts),
            "intact_count": len(intact_parts),
            "uncertain_count": len(uncertain_parts),
            "structural_pattern_count": len(patterns),
            "pattern_ids": [p.pattern_id for p in patterns],
        }

        # 6. structural_damage_flag: True if any severe pattern exists.
        structural_flag = any(p.severity == "severe" for p in patterns)

        return DamageAssessment(
            vehicle_info={
                "vehicle_id": self.topology.vehicle_id,
                "vehicle_name": self.topology.vehicle_name,
            },
            topology_model=self.topology.to_dict(),
            parts=parts,
            missing_parts=missing_parts,
            damaged_parts=damaged_parts,
            intact_parts=intact_parts,
            uncertain_parts=uncertain_parts,
            structural_patterns=patterns,
            structural_damage_flag=structural_flag,
            overall_severity=overall_severity,
            primary_damage_zone=primary_zone,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # State resolution
    # ------------------------------------------------------------------

    def _resolve_state(
        self,
        node_id: str,
        node,
        actual_by_id: Dict[str, PartActualState],
    ) -> PartActualState:
        """Return the actual state for a topology node.

        If the node is absent from *actual_states*, synthesise an UNCERTAIN
        state with standard_exists=True and actual_visible=False.
        """
        if node_id in actual_by_id:
            return actual_by_id[node_id]

        # Fallback: build a synthetic UNCERTAIN state.
        part_info = PARTS_BY_ID.get(node_id, {})
        return PartActualState(
            part_id=node_id,
            part_name=part_info.get("part_name", node.node_name),
            region=node.region,
            side=node.side,
            status=Status.UNCERTAIN,
            damage_level=DamageLevel.UNKNOWN,
            standard_exists=True,
            actual_visible=False,
            actual_present=False,
            confidence="low",
            notes="No actual state provided; defaulted to uncertain.",
        )

    # ------------------------------------------------------------------
    # Structural pattern detection
    # ------------------------------------------------------------------

    def _detect_patterns(
        self,
        damaged_node_ids: Set[str],
        parts: List[PartActualState],
    ) -> List[StructuralDamagePattern]:
        """Detect all 5 structural damage patterns."""
        patterns: List[StructuralDamagePattern] = []

        # a) regional_mass_damage
        regional_counts: Dict[str, int] = {}
        for node_id in damaged_node_ids:
            node = self.topology.nodes.get(node_id)
            if node is None:
                continue
            regional_counts[node.region] = regional_counts.get(node.region, 0) + 1

        for region, count in regional_counts.items():
            if count >= 3:
                region_nodes = [
                    nid
                    for nid in damaged_node_ids
                    if self.topology.nodes.get(nid) is not None
                    and self.topology.nodes[nid].region == region
                ]
                patterns.append(
                    StructuralDamagePattern(
                        pattern_id="regional_mass_damage",
                        pattern_name="区域大面积损伤",
                        description=f"Region '{region}' has {count} damaged/missing parts",
                        matched_nodes=region_nodes,
                        severity="severe",
                        confidence="high",
                    )
                )

        # b) cross_region_penetration
        clusters = self._bfs_clusters(damaged_node_ids)
        for cluster in clusters:
            regions_in_cluster: Set[str] = {
                self.topology.nodes[nid].region
                for nid in cluster
                if nid in self.topology.nodes
            }
            if len(regions_in_cluster) >= 2:
                patterns.append(
                    StructuralDamagePattern(
                        pattern_id="cross_region_penetration",
                        pattern_name="跨区域穿透损伤",
                        description=(
                            f"Connected cluster spans {len(regions_in_cluster)} regions: "
                            f"{', '.join(sorted(regions_in_cluster))}"
                        ),
                        matched_nodes=sorted(cluster),
                        severity="severe",
                        confidence="high",
                    )
                )

        # c) structural_component_damage
        structural_damaged: List[str] = []
        for node_id in damaged_node_ids:
            node = self.topology.nodes.get(node_id)
            if node is not None and node.node_type == "structural":
                structural_damaged.append(node_id)
        if structural_damaged:
            patterns.append(
                StructuralDamagePattern(
                    pattern_id="structural_component_damage",
                    pattern_name="结构件受损",
                    description="One or more structural components are damaged/missing",
                    matched_nodes=structural_damaged,
                    severity="severe",
                    confidence="high",
                )
            )

        # d) symmetric_damage
        symmetric_damaged_pairs: List[tuple[str, str]] = []
        for left, right in _SYMMETRIC_PAIRS:
            if left in damaged_node_ids and right in damaged_node_ids:
                symmetric_damaged_pairs.append((left, right))
        if len(symmetric_damaged_pairs) >= 2:
            matched = []
            for left, right in symmetric_damaged_pairs:
                matched.extend([left, right])
            patterns.append(
                StructuralDamagePattern(
                    pattern_id="symmetric_damage",
                    pattern_name="对称损伤",
                    description=(
                        f"{len(symmetric_damaged_pairs)} symmetric pairs "
                        "are simultaneously damaged/missing"
                    ),
                    matched_nodes=sorted(set(matched)),
                    severity="moderate",
                    confidence="medium",
                )
            )

        # e) safety_critical_missing
        safety_missing: List[str] = []
        for part in parts:
            if part.part_id in _SAFETY_CRITICAL_PARTS and part.status == Status.MISSING:
                safety_missing.append(part.part_id)
        if safety_missing:
            patterns.append(
                StructuralDamagePattern(
                    pattern_id="safety_critical_missing",
                    pattern_name="安全关键部件缺失",
                    description="One or more safety-critical parts (headlight/taillight/mirror) are missing",
                    matched_nodes=safety_missing,
                    severity="severe",
                    confidence="high",
                )
            )

        return patterns

    # ------------------------------------------------------------------
    # BFS cluster analysis
    # ------------------------------------------------------------------

    def _bfs_clusters(self, damaged_node_ids: Set[str]) -> List[Set[str]]:
        """Return connected components of damaged nodes via topology adjacency.

        Only edges between nodes that are both in *damaged_node_ids* are traversed.
        """
        if not damaged_node_ids:
            return []

        remaining: Set[str] = set(damaged_node_ids)
        clusters: List[Set[str]] = []

        while remaining:
            start = remaining.pop()
            cluster: Set[str] = {start}
            queue: deque[str] = deque([start])

            while queue:
                current = queue.popleft()
                node = self.topology.nodes.get(current)
                if node is None:
                    continue
                for adj in node.adjacent_nodes:
                    if adj in remaining and adj in damaged_node_ids:
                        remaining.remove(adj)
                        cluster.add(adj)
                        queue.append(adj)

            clusters.append(cluster)

        return clusters

    # ------------------------------------------------------------------
    # Severity computation
    # ------------------------------------------------------------------

    def _compute_overall_severity(
        self, patterns: List[StructuralDamagePattern]
    ) -> str:
        """Roll up pattern severities into an overall severity string.

        severe count >= 3          -> severe
        severe >= 1 or moderate >= 3 -> moderate
        moderate >= 1              -> light
        else                       -> none
        """
        severe_count = sum(1 for p in patterns if p.severity == "severe")
        moderate_count = sum(1 for p in patterns if p.severity == "moderate")

        if severe_count >= 3:
            return "severe"
        if severe_count >= 1 or moderate_count >= 3:
            return "moderate"
        if moderate_count >= 1:
            return "light"
        return "none"

    # ------------------------------------------------------------------
    # Primary damage zone
    # ------------------------------------------------------------------

    def _compute_primary_zone(self, parts: List[PartActualState]) -> str:
        """Return the region with the highest weighted damage score.

        Uses _DAMAGE_WEIGHTS to score each part's damage_level.  Ties or
        zero scores return "multiple".
        """
        region_scores: Dict[str, int] = {}
        for part in parts:
            weight = _DAMAGE_WEIGHTS.get(part.damage_level.value, 0)
            if weight == 0:
                continue
            region_scores[part.region] = region_scores.get(part.region, 0) + weight

        if not region_scores:
            return "multiple"

        max_score = max(region_scores.values())
        top_regions = [r for r, s in region_scores.items() if s == max_score]

        if len(top_regions) > 1 or max_score == 0:
            return "multiple"
        return top_regions[0]


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def compare_topology(
    topology: VehicleTopology, actual_states: List[PartActualState]
) -> DamageAssessment:
    """Compare *topology* against *actual_states* and return a DamageAssessment.

    This is a thin wrapper around TopologyComparator.compare().
    """
    comparator = TopologyComparator(topology)
    return comparator.compare(actual_states)
