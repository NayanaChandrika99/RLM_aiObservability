# Contracts (Artifacts)

## Purpose

Define the canonical **schemas and examples** for all Plan1 artifacts.

These contracts are the integration boundary between agents, deterministic tools, and the Digital Twin.
They are designed to be:

- **AI-parsable** (clear field names, enums, and examples)
- **machine-validated** (Pydantic/JSON schema)
- **replayable** (stable IDs and deterministic fields)

## Non-goals

- Not a full API spec for a web service (MVP is CLI/module-first).
- Not a full database schema (MVP uses file-backed JSON fixtures + artifacts).

## Conventions

- **Key naming**: `snake_case`.
- **IDs**: strings (stable; avoid random UUIDs in goldens unless seeded).
- **Timestamps**: ISO-8601 strings (UTC recommended). In tests, timestamps must be frozen or normalized.
- **Numeric scales**:
  - `confidence`: `0.0..1.0`
  - `risk_score`: `0.0..1.0`
- **Coverage flags**: `FULL | PARTIAL | NONE`.
- **Artifacts can be partial**: on early exit, later-stage artifacts are absent.

## Enums

### RunOutcome

`RunRecord.outcome` is one of:

- `EXTRACT_FAILED`
- `LOW_CONFIDENCE`
- `NO_IMPACT`
- `LOW_RISK`
- `NO_VIABLE_PLAN`
- `REJECTED`
- `EXECUTION_FAILED`
- `APPROVED`
- `AUTO_APPROVED`

### RiskLevel

- `HIGH` (score ≥ 0.60)
- `MEDIUM` (0.45–0.59)
- `LOW` (score < 0.45)

### KGResultExpectation (used in scenario fixtures)

This is not the same thing as `RunOutcome`.

- `PATHS_FOUND`
- `NO_PATHS`
- `LOW_COVERAGE`

## Artifact schemas

Below, “Required” means required for that artifact when it exists.

### 1) IncidentTicket

**Purpose:** structured representation of a disruption.

Fields (required unless noted):

- `incident_id`: string
- `source_type`: `"scenario" | "manual"`
- `scenario_id`: string (optional; required when `source_type="scenario"`)
- `timestamp`: ISO string (optional; tests may omit and rely on normalization)
- `disruption_type`: string (e.g., `"geopolitical"`, `"natural_disaster"`)
- `summary`: string
- `entities`: list of objects:
  - `entity_type`: `"country" | "company" | "industry" | "product"`
  - `name`: string
- `diagnostic_questions`: list of strings (optional; deterministic “questions to ask” when incident details are incomplete)
- `evidence`: list of objects (optional):
  - `url`: string (optional)
  - `snippet`: string (optional)
- `confidence`: float `0..1`

Example:

```json
{
  "incident_id": "inc_geopol_001",
  "source_type": "scenario",
  "scenario_id": "scn_001_embargo",
  "timestamp": "2026-01-20T10:00:00Z",
  "disruption_type": "geopolitical",
  "summary": "Trade embargo impacts aluminum exports from Country X.",
  "entities": [
    {"entity_type": "country", "name": "Country X"},
    {"entity_type": "product", "name": "Aluminum Can"}
  ],
  "diagnostic_questions": ["Which supplier/company is affected (Tier-1 name if known)?"],
  "evidence": [{"url": "https://example.com/article", "snippet": "Embargo announced..."}],
  "confidence": 0.82
}
```

### 2) KGImpactMap

**Purpose:** multi-tier exposure mapping result.

Fields:

- `coverage_flag`: `FULL | PARTIAL | NONE`
- `uncertainty_flags`: list of strings (optional; default empty)
  - stable ordering: sort ascending
  - pinned flag strings for MVP:
    - `unmapped_country` (at least one `IncidentTicket` country entity did not resolve)
    - `unmapped_company` (at least one company entity did not resolve)
    - `unmapped_industry` (at least one industry entity did not resolve)
    - `unmapped_product` (at least one product entity did not resolve)
    - `no_entities` (the incident ticket contained no entities)
- `query`: object (optional): the normalized query intent (countries/industries/products)
- `disrupted_nodes`: list of node IDs (strings)
- `disrupted_edges`: list of objects:
  - `from_id`: string
  - `to_id`: string
  - `edge_type`: string (e.g., `"SUPPLIES_TO"`)
- `tier_paths`: list of paths, each path:
  - `path_id`: string
  - `nodes`: list of objects: `{ "node_id": string, "tier": int }`

Example:

```json
{
  "coverage_flag": "FULL",
  "uncertainty_flags": [],
  "disrupted_nodes": ["c_tier3_aluminum_mine_01"],
  "disrupted_edges": [{"from_id": "c_tier3_aluminum_mine_01", "to_id": "c_tier2_sheet_02", "edge_type": "SUPPLIES_TO"}],
  "tier_paths": [
    {
      "path_id": "p_001",
      "nodes": [
        {"node_id": "c_pepsico", "tier": 0},
        {"node_id": "c_tier1_can_supplier_01", "tier": 1},
        {"node_id": "c_tier2_sheet_02", "tier": 2},
        {"node_id": "c_tier3_aluminum_mine_01", "tier": 3}
      ]
    }
  ]
}
```

### 3) Tier1RiskTable

**Purpose:** deterministic Tier-1 risk ranking and breakdown.

Fields:

- `risks`: list of entries (sorted descending by `risk_score`):
  - `tier1_supplier_id`: string
  - `tier1_supplier_name`: string
  - `risk_score`: float `0..1`
  - `risk_level`: `HIGH | MEDIUM | LOW`
  - `components`: object (floats `0..1`):
    - `exposure_breadth`
    - `dependency_ratio`
    - `downstream_criticality`
    - `tier1_centrality`
    - `exposure_depth`

Example:

```json
{
  "risks": [
    {
      "tier1_supplier_id": "c_tier1_can_supplier_01",
      "tier1_supplier_name": "Can Supplier A",
      "risk_score": 0.71,
      "risk_level": "HIGH",
      "components": {
        "exposure_breadth": 0.8,
        "dependency_ratio": 0.6,
        "downstream_criticality": 0.7,
        "tier1_centrality": 0.4,
        "exposure_depth": 0.75
      }
    }
  ]
}
```

### 4) PlanOption

**Purpose:** a single mitigation option proposed by CSCO/Architect.

Fields:

- `plan_option_id`: string
- `action_type`: string (e.g., `"reroute"`, `"expedite"`, `"shift_production"`, `"alt_supplier"`, `"monitor"`)
- `title`: string
- `description`: string
- `assumptions`: list of strings
- `parameters`: object (schema depends on `action_type`, but must be JSON-serializable and stable)
  - Canonical `action_type` taxonomy + required/optional parameters are pinned in `specs/plan-options.md`.

Example:

```json
{
  "plan_option_id": "opt_2",
  "action_type": "reroute",
  "title": "Reroute aluminum shipments via alternate lane",
  "description": "Switch Tier-2 sheet supply from Lane A to Lane B for 14 days.",
  "assumptions": ["Lane B has spare capacity", "Cost premium is acceptable"],
  "parameters": {
    "time_window_days": 14,
    "alt_lane_id": "lane_alt_01",
    "max_units": 5000,
    "max_lead_time_days": 7
  }
}
```

### 5) ExecutiveActionPlan

**Purpose:** executive-facing, directly actionable summary of what to do for each candidate plan option.

This artifact is derived from the final, validated `PlanOption[]` set and is intended to be:

- deterministic (no additional LLM calls required),
- easy to read in `run_record.json` without opening code,
- distinct from `PlanOption.description` (focused on action steps + rollback hints).

Fields:

- `options`: list of entries (one per `PlanOption`):
  - `plan_option_id`: string
  - `action_type`: string
  - `title`: string
  - `steps`: list of strings (short, directly actionable)
  - `rollback_hint`: string (informational; rollback is not executed in MVP)

Example:

```json
{
  "options": [
    {
      "plan_option_id": "opt_2",
      "action_type": "reroute",
      "title": "Reroute aluminum shipments via alternate lane",
      "steps": ["Reroute shipments using alt_lane_id=lane_alt_01 for 14 days (max_units=5000)."],
      "rollback_hint": "Revert routing to the pre-run lane configuration."
    }
  ]
}
```

### 6) TwinResult

**Purpose:** feasibility + KPI estimates for a single `PlanOption`.

Fields:

- `plan_option_id`: string
- `feasible`: boolean
- `constraint_violations`: list of strings
- `optimizer`: object:
  - `status`: string (e.g., `"OPTIMAL"`, `"INFEASIBLE"`)
  - `objective_value`: number (optional)
- `kpis`: object (optional for infeasible):
  - `cost`
  - `cost_breakdown` (optional; object `dict[str, number]`)
  - `lead_time_p50_days` (optional)
  - `lead_time_p95_days`
  - `fill_rate`
  - `backlog_days` (optional; integer)
  - `emissions_kg_co2`

Example:

```json
{
  "plan_option_id": "opt_2",
  "feasible": true,
  "constraint_violations": [],
  "optimizer": {"status": "OPTIMAL", "objective_value": 1234.5},
  "kpis": {
    "cost": 12000,
    "cost_breakdown": {"transport": 7000, "production": 5000, "total": 12000},
    "lead_time_p50_days": 6.0,
    "lead_time_p95_days": 9.0,
    "fill_rate": 0.97,
    "backlog_days": 2,
    "emissions_kg_co2": 4200
  }
}
```

### 7) Scorecard

**Purpose:** ranking over `TwinResult[]` and a chosen recommendation.

Fields:

- `ranked_options`: list (best first):
  - `plan_option_id`
  - `score`: number
  - `rationale`: string
- `recommended_plan_option_id`: string (optional; may be absent if all infeasible)

Example:

```json
{
  "ranked_options": [
    {"plan_option_id": "opt_2", "score": 0.82, "rationale": "Feasible with good service and moderate cost."}
  ],
  "recommended_plan_option_id": "opt_2"
}
```

### 8) ApprovalRecord

**Purpose:** explicit HITL/auto decision record.

Fields:

- `approval_required`: boolean
- `policy_triggers`: list of strings
- `decision`: `"APPROVED" | "REJECTED" | "AUTO_APPROVED"`
- `decided_by`: string (optional; required if human)
- `timestamp`: ISO string (optional)
- `notes`: string (optional)

Example:

```json
{
  "approval_required": true,
  "policy_triggers": ["cost_delta_gt_threshold"],
  "decision": "APPROVED",
  "decided_by": "nainy",
  "timestamp": "2026-01-20T10:05:00Z"
}
```

### 9) ExecutionReceipt

**Purpose:** auditable record of attempted side effects (even in dry-run).

Fields:

- `receipt_id`: string
- `plan_option_id`: string
- `adapter`: string (e.g., `"erp_stub"`)
- `dry_run`: boolean
- `idempotency_key`: string
- `status`: `"SUCCESS" | "FAILED"`
- `external_refs`: object (optional; IDs from external systems or stub IDs)
- `error`: string (optional)
- `compensation_hint`: string (optional)

Example:

```json
{
  "receipt_id": "rcpt_001",
  "plan_option_id": "opt_2",
  "adapter": "erp_stub",
  "dry_run": true,
  "idempotency_key": "run_001:opt_2:erp_stub",
  "status": "SUCCESS",
  "external_refs": {"stub_order_id": "stub_123"}
}
```

### 10) RunRecord

**Purpose:** top-level stitched run artifact, always produced.

Fields:

- `run_id`: string
- `scenario_id`: string (optional)
- `outcome`: `RunOutcome`
- `reason`: string (required for early exits/failures; optional otherwise)
- `coverage_flag`: `FULL | PARTIAL | NONE` (KG coverage summary)
- `artifacts`: object (optional; may be partial):
  - `incident_ticket` (optional)
  - `kg_impact_map` (optional)
  - `tier1_risk_table` (optional)
  - `plan_options` (optional)
  - `executive_action_plan` (optional)
  - `twin_results` (optional)
  - `scorecard` (optional)
  - `approval_record` (optional)
  - `execution_receipts` (optional)
- `trace_ids`: object (optional): `{ "trace_id": "...", "span_ids": {...} }`
- `timeline`: object (optional): step timestamps (ISO strings)
  - always includes: `start`, `end`
  - may include per-stage keys when those stages were reached:
    - `sentinel`
    - `kg_query`
    - `risk_manager`
    - `csco`
    - `digital_twin`
    - `simulate_top_k`
    - `executive_gate`
    - `exec_gate_interrupt`
    - `execute`
    - `finalize`
  - stage keys appear only for stages reached before a terminal `outcome` is set, plus `finalize`

Example:

```json
{
  "run_id": "run_001",
  "scenario_id": "scn_001_embargo",
  "outcome": "AUTO_APPROVED",
  "reason": "Low-impact plan below approval thresholds.",
  "coverage_flag": "FULL",
  "artifacts": {
    "incident_ticket": {"incident_id": "inc_geopol_001", "source_type": "scenario", "scenario_id": "scn_001_embargo", "disruption_type": "geopolitical", "summary": "Embargo impacts aluminum.", "entities": [{"entity_type": "country", "name": "Country X"}], "confidence": 0.82},
    "execution_receipts": [{"receipt_id": "rcpt_001", "plan_option_id": "opt_2", "adapter": "erp_stub", "dry_run": true, "idempotency_key": "run_001:opt_2:erp_stub", "status": "SUCCESS"}]
  }
}
```

## Determinism requirements

- All IDs must be stable in fixtures and goldens.
- `RunRecord` must be produced even when intermediate artifacts are missing.
- Timestamp handling must be test-friendly (freeze or normalize).

## Acceptance checks

Once implemented:

- Validate that every produced artifact is JSON-serializable and schema-valid.
- Run a scenario end-to-end and confirm a `RunRecord` exists and references artifacts correctly.

## Keywords

contracts, schemas, artifacts, json, pydantic, run record, receipts, approvals, determinism
