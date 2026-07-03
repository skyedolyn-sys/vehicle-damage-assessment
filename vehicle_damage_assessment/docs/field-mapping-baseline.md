# Field Mapping Baseline

> Vehicle damage assessment codebase, current state on `main` as of 2026-07-03.
> This is the **factual** field map — every entry was grep-verified against the
> actual source files. Use it as the single source of truth when deciding
> "should field X live in YAML or stay in Python" / "what's the canonical
> enum value for Y".
>
> Policy-level ("how agents *should* use these fields") lives in
> [`agents/DAMAGE_RECOGNITION_POLICY.md`](../agents/DAMAGE_RECOGNITION_POLICY.md).
> This file is the implementation baseline — it documents what the code
> actually does, not what it ought to do.

## 1. Enums & canonical string values

### 1.1 `Status` (comparison status of a part)

Defined in `models/part_state.py` as `class Status(str, Enum)`.

| Value | Meaning |
| --- | --- |
| `intact` | Standard exists, actual exists and normal |
| `damaged` | Standard exists, actual exists but abnormal |
| `missing` | Standard exists, actual does not exist |
| `uncertain` | Cannot determine |
| `na` | Not applicable (standard does not exist) |

**Source of truth:** `models/part_state.py:11`. `Status.value` is what goes
into JSON. The YAML at `agents/rules/config/priorities.yaml` only lists
`damaged/uncertain/intact/missing` because `na` is treated as 0 priority
in `_merge_two_states` (`assessment_orchestrator.py:392`). So the YAML is
a subset of the enum; that is **intentional** but worth flagging.

### 1.2 `DamageLevel` (severity of damage)

Defined in `models/part_state.py` as `class DamageLevel(str, Enum)`.

| Value | Numeric priority (`priorities.yaml#level`) |
| --- | --- |
| `none` | 1 |
| `light` | 2 |
| `moderate` | 3 |
| `severe` | 4 |
| `unknown` | 0 |

The numeric priority is what `_merge_two_states` uses to keep the higher
severity between two candidates (`assessment_orchestrator.py:395`).

### 1.3 `confidence`

3-value enum, defined by usage (not as a Python enum):

| Value | Numeric priority |
| --- | --- |
| `low` | 0 |
| `medium` | 1 |
| `high` | 2 |

Sourced from `priorities.yaml#confidence`.

### 1.4 `damage_type` (per-damage list)

Free-form string list. Examples actually used in vision_subagent prompts
(see `agents/rules/templates/`):
`crack`, `deformation`, `tear`, `missing`, `paint_damage`, `glass_breakage`,
`sunroof_damage`, `airbag_deployed`, `none`.

There is **no central enum** — agents can introduce new types freely.
This is a known gap; for now validate against a soft allow-list at the
synthesizer layer if needed.

### 1.5 `view_id`

Defined in `agents/view_mapping.py`:

```
STANDARD_VIEWS = [
    "front",
    "front_left_45", "front_right_45",
    "rear",
    "rear_left_45", "rear_right_45",
    "left_90", "right_90",
    "top",
    "interior", "auxiliary", "unknown",
    "scene_intake",
]

EXTERIOR_VIEWS = {"front", "front_left_45", "front_right_45",
                  "rear", "rear_left_45", "rear_right_45",
                  "left_90", "right_90", "top"}

NON_EXTERIOR_VIEWS = {"interior", "auxiliary", "unknown", "scene_intake"}
```

13 view ids total. Vision subagents are dispatched for **all** of these
except `unknown` (`assessment_orchestrator.py:106`).

### 1.6 `photo_type`

5-value enum, also from `agents/view_mapping.py`:

```
PHOTO_TYPE_CATEGORIES = {"exterior", "interior", "auxiliary", "unknown", "scene_intake"}
```

Produced by `_classify_photo_by_signals` in `planner_agent.py` (filename +
aspect ratio heuristics, **no LLM**). Per `DAMAGE_RECOGNITION_POLICY §1.6`,
this is fully deterministic since the 26%-failure LLM path was removed.

### 1.7 `part_category` / `region`

`config.PARTS_CATALOG` defines 5 regions:
`front`, `rear`, `left`, `right`, `roof`.

In `models/part_state.py` this is stored as the `region` attribute. When
serialized via `to_dict()`, the key is `part_category` (UI format).
The `side` field is one of:
`center`, `front_left`, `front_right`, `rear_left`, `rear_right`.

## 2. Field lifecycle — what flows where

```
                            ┌──────────────────┐
                            │ planner_agent    │
                            │  produces:       │
                            │  photo_views[]   │
                            │  view_groups{}   │
                            │  workflow_plan{} │
                            │  coverage_gaps[] │
                            └─────────┬────────┘
                                      │ view_id per photo
                                      ▼
                            ┌──────────────────┐
                            │ assessment_      │
                            │ orchestrator     │  (per view group, parallel)
                            │   dispatches     │
                            └─────────┬────────┘
                                      │ photos + view_id
                                      ▼
                ┌──────────────────────────────────┐
                │ vision_subagent(view_id, …)      │
                │  produces per part:              │
                │    part_id                       │
                │    status        (enum §1.1)     │
                │    damage_level  (enum §1.2)     │
                │    damage_type[] (free-form)     │
                │    confidence    (enum §1.3)     │
                │    evidence_photo[]              │
                │    notes                         │
                │    + view_id (echoed in result)  │
                └─────────────────┬────────────────┘
                                  │ result["parts"][], result["view_id"]
                                  ▼
                ┌──────────────────────────────────┐
                │ evidence_fusion.apply_fusion     │
                │  internal _origin_view tagged    │
                │  on every candidate              │
                └─────────────────┬────────────────┘
                                  │ overrides{part_id: {status, damage_level, confidence, evidence_photo, …}}
                                  ▼
                ┌──────────────────────────────────┐
                │ reviewer_subagent                │
                │  reviewed_parts[]                │
                │  reviewed_part_actual_states[]   │
                │  needs_rephotography[]           │
                └─────────────────┬────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │ synthesizer_agent                │
                │  applies view_weights.primary/   │
                │  secondary + adjacency rules     │
                │  + roof-specific rules           │
                │  → new_part dict per part_id     │
                └─────────────────┬────────────────┘
                                  │ list of PartActualState (via to_legacy_dict / to_dict)
                                  ▼
                ┌──────────────────────────────────┐
                │ assessment_orchestrator          │
                │  _merge_two_states per part_id   │
                │  (uses STATUS_PRIORITY,           │
                │   LEVEL_PRIORITY,                │
                │   CONFIDENCE_PRIORITY from       │
                │   priorities.yaml)               │
                └─────────────────┬────────────────┘
                                  │ DamageAssessment
                                  ▼
                ┌──────────────────────────────────┐
                │ topology_comparator              │
                │  compare_topology(topology,      │
                │                   actual_states) │
                │  → DamageAssessment              │
                │    .to_legacy_result()           │
                │    .parts[]                      │
                │    .structural_damage_flag       │
                └─────────────────┬────────────────┘
                                  │ assessment_result dict (frontend-compatible)
                                  ▼
                                API / JSON
```

## 3. Per-field reference (who reads / writes what)

### `status`
- **Writes:** `vision_subagent._llm_dict_to_part_actual_state`,
  `evidence_fusion.fuse_evidence` (in overrides),
  `synthesizer_agent._resolve_status_*`,
  `topology_comparator.compare_topology`,
  `_merge_two_states` (`assessment_orchestrator.py:386`).
- **Reads:** every downstream consumer + `_filter_uncertain_items`
  (`agents/output_validator.py`).
- **Allowed values:** see §1.1.

### `damage_level`
- **Writes:** `vision_subagent`, `synthesizer_agent` (sometimes downgrades
  via `Rule 5` and `Rule 6` in `synthesizer.py:529-580`),
  `evidence_fusion.fuse_evidence`.
- **Reads:** `_merge_two_states`, all UI layers.
- **Allowed values:** see §1.2.

### `damage_type` (list of strings)
- **Writes:** `vision_subagent` (LLM output), `_enforce_positive_anchor`
  (downgrades to `["none"]`).
- **Reads:** `output_validator._filter_uncertain_items`,
  `synthesizer_agent` (for adjacency patterns).
- **Default when empty:** `["none"]` (in `PartActualState.to_dict`).
- **Legacy string format:** comma-separated in `to_legacy_dict`.

### `confidence`
- **Writes:** `vision_subagent`, `synthesizer_agent` (downgrades),
  `evidence_fusion.fuse_evidence` (preserves max).
- **Reads:** `_merge_two_states`, all UI layers.
- **Allowed values:** see §1.3.

### `evidence_photo` / `evidence_photos` / `evidence_sources`
- **Three different shapes coexist:**

  | Field | Type | Where |
  | --- | --- | --- |
  | `evidence_photo` | list[str] | LLM output, `PartActualState.to_dict` |
  | `evidence_photo` | csv string | `PartActualState.to_legacy_dict` |
  | `evidence_photos` | list[str] | `PartActualState` dataclass attribute |
  | `evidence_sources` | list[dict] | `PartActualState` dataclass attribute (full provenance) |

  `_as_photo_list` (`agents/evidence_fusion.py:146`) normalises both list
  and csv-string forms. **When changing, watch all four call sites.**

### `view_id`
- **Writes:** `planner_agent._clean_view_entries` (per photo), `vision_subagent` (echoed into result).
- **Reads:** orchestrator dispatch, `evidence_fusion` (via `_origin_view`),
  `synthesizer` (`VIEW_WEIGHTS.get(part_id).get("primary", [])`),
  `topology_comparator`.
- **Allowed values:** see §1.5.

### `part_id`
- **Writes:** `vision_subagent._normalize_part_id` (after LLM emits an alias),
  `synthesizer_agent` (passthrough).
- **Catalog:** 32 part ids, defined in `config.PARTS_CATALOG` (also
  resolvable via `PARTS_BY_ID`). Aliases resolved via
  `agents.rules.resolve_part_alias`.

### `part_category` vs `region`
- `region` is the dataclass attribute name (`PartActualState.region`).
- `part_category` is the JSON key emitted by `to_dict()` / `to_legacy_dict()`.
- Both hold the same 5-value enum (`front|rear|left|right|roof`).
- **When writing new code:** stick to whichever the surrounding file uses
  — the codebase is inconsistent. The dataclass uses `region`, the wire
  format uses `part_category`.

### `photo_type`
- **Writes:** planner's `_classify_photo_by_signals` (per photo).
- **Reads:** `vision_subagent._resolve_photo_type` (decorates each part).
- **Allowed values:** see §1.6.

### `evidence_sources` (provenance)
- **Writes:** synthesizer adds `{view_id, status, damage_level,
  evidence_photo, confidence}` per source.
- **Reads:** synthesizer's adjacency rules (Rules 5/6/9-11),
  `evidence_fusion._find_secondary_damage_conflicts`,
  `reviewer_subagent._find_uncovered_parts`.

## 4. View authority — two competing sources

This is the field-mapping conflict that motivated this baseline.

### 4.1 YAML: `agents/rules/config/view_weights.yaml`
Has 4 keys (`primary_view`, `view_weights`, `roof_primary_regions`,
`roof_secondary_regions`). Loaded via `load_view_weights()`.

- `primary_view[part_id] → [view_id, …]` — single best view per part.
  Used in `planner_agent` (coverage-gap impact estimation).
- `view_weights[part_id] → {primary: [...], secondary: [...]}` — used by
  `synthesizer_agent` for every adjacency rule (§3-§11 of the policy).
- `roof_primary_regions`, `roof_secondary_regions` — used by
  `synthesizer_agent._resolve_status_roof`.

### 4.2 Python: `_PART_VIEW_PRIORITY` in `agents/evidence_fusion.py:82`
A hardcoded `Dict[str, Dict[str, int]]` with priority 0/1/2/99 per
`(part_id, view_id)`. Used by `fuse_evidence` to decide if a view is
primary/secondary for a given part.

### 4.3 Discrepancy

| Aspect | YAML `view_weights.yaml` | Python `_PART_VIEW_PRIORITY` |
| --- | --- | --- |
| Format | `primary: [view_ids], secondary: [view_ids]` | `{view_id: priority_int}` (0/1/2/99) |
| Roof coverage | primary_view has roof_front→[top], roof_middle→[top], roof_rear→[top] | roof_front→{front:0, front_left_45:1, front_right_45:1}, roof_middle→{left_90:0, right_90:0, top:0, front:1, rear:1}, roof_rear→{rear:0, rear_left_45:1, rear_right_45:1} |
| Part coverage | full PARTS_CATALOG (32 parts) | 17 critical parts only |
| Source | editable YAML, hot-reloadable | frozen in code |

**They are NOT the same data and they are NOT kept in sync.**

Concrete example: `roof_front`. In YAML `primary_view`, the best view for
`roof_front` is `[top]`. In Python `_PART_VIEW_PRIORITY`, `roof_front` has
priority 0 on `front` and priority 1 on `front_left_45`/`front_right_45`,
but **no entry for `top`** — so `top` defaults to priority 5 (neutral).

In real terms, the synthesizer will trust a top view as primary for the
roof, but `evidence_fusion` will not. For the 172852 sample this means a
top-view report of "roof front: damaged" gets through `evidence_fusion`
only with neutral authority, not as primary. That's why "车顶前部" lands at
`damaged severe` for that sample via synthesizer + reviewer, but the same
view authority would rank differently under evidence_fusion's rules.

### 4.4 The `part_view_priority` key

`agents/rules/loader.py:365` defines `load_part_view_priority()` and
`agents/rules/__init__.py` re-exports it, **but**:
- `view_weights.yaml` does **not** contain a `part_view_priority` key.
- `load_part_view_priority()` therefore returns `{}`.
- **No production code calls `load_part_view_priority()`.**
- The `tests/test_part_view_priority_consistency.py` test would fail
  if run, because it's checking consistency against a structure that
  doesn't exist yet.

This is a **dead-but-intended feature**. Most likely the worktree in
`.claude/worktrees/agent-...` (per `git diff` history) intended to add the
key to the YAML and have `evidence_fusion` consume it. It never landed.

## 5. Priority maps — already centralised

`priorities.yaml` is the source of truth for the 4 priority maps:
`status`, `level`, `confidence`, `uncertain_status`. Used by
`_merge_two_states` (`assessment_orchestrator.py:382-397`) and the
synthesizer (`synthesizer.py:16-20`).

This is the **one place** field-mapping consolidation is already
complete. The 32-part catalog + per-part `view_weights.primary/secondary`
is **not** consolidated yet.

## 6. Adjacent catalogues that should be unified next

If the goal is "single source of truth per concept", here's what's
duplicated or scattered:

| Concept | Currently in | Should be in |
| --- | --- | --- |
| Per-part best view (priority 0) | `evidence_fusion._PART_VIEW_PRIORITY` | YAML `view_weights.yaml#part_view_priority` (key already designed, not populated) |
| Per-part secondary views (priority 1) | YAML `view_weights.yaml#view_weights.{part}.secondary` + Python `_PART_VIEW_PRIORITY` | YAML only |
| Per-part profile sets (`conservative`, `roof`, `front_false_damage`, …) | YAML `part_profiles.yaml` | already centralised |
| Photo type heuristics | YAML `filename_heuristics.yaml` | already centralised |
| Region ↔ view mapping | Python `VIEW_TO_REGIONS` (`view_mapping.py:55`) | could move to YAML, low priority |
| Side ↔ display name | `models.topology.TopologyNode.side` | already centralised |
| Damage types (free-form) | nowhere | gap — needs `damage_types.yaml` if we want to constrain |
| Threshold values | YAML `thresholds.yaml` | already centralised |

## 7. Quick checklist for "field X is unified"

For any field you want to declare "unified", verify:

- [ ] Single definition site (Python enum OR YAML key — not both)
- [ ] All writes go through one helper (e.g. `_as_photo_list` for lists)
- [ ] All reads accept the same shape (string | list | csv) without manual
      unwrapping at every call site
- [ ] The wire format (JSON keys) matches the dataclass attribute names
      OR has explicit serialiser/deserialiser methods
- [ ] Tests assert the canonical values, not hard-coded strings

`priorities.yaml` passes all five. The roof view authority fails 1, 2, and 5.