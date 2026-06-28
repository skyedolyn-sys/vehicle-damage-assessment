"""Topology comparator — compare standard VehicleTopology against actual PartActualState list.

Produces a DamageAssessment with structural damage pattern detection.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set

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

            if state.status == Status.MISSING:
                missing_parts.append(node_id)
            elif state.status == Status.DAMAGED:
                damaged_parts.append(node_id)
            elif state.status == Status.INTACT:
                intact_parts.append(node_id)
            elif state.status == Status.UNCERTAIN:
                uncertain_parts.append(node_id)

        # 2. Structural pattern detection.
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
