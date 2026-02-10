# Phoenix-RLM Specs Pin Index

This is the formal spec pin for Phoenix-RLM Investigator.
Use it as the first lookup when implementing or reviewing work.

Rules:
- Keep this file updated in the same change when a spec doc changes.
- Keep Purpose and Keywords retrieval-friendly (include synonyms).
- Link to repo-relative code locations once implementation exists.

Recommended table format: `Spec | Code | Purpose | Keywords`.

## Core

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [`AGENTS.md`](../AGENTS.md) | `apps/`, `investigator/`, `connectors/` | project rules, phase order, operating constraints, canonical local run path | rules, constraints, phase plan, phoenix serve, no docker |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | `apps/`, `investigator/`, `configs/`, `datasets/`, `connectors/` | system architecture, boundaries, components, dataflow, trust model | architecture, components, dataflow, trust boundary, observability |
| [`API.md`](../API.md) | `investigator/inspection_api/` | read-only Trace Inspection API contracts and evidence ID rules | trace inspection api, spans, artifact_id, retrieval chunk, tool io |
| [`DESIGN.md`](../DESIGN.md) | `investigator/rca/`, `investigator/compliance/`, `investigator/incident/`, `investigator/schemas/` | evaluator design, RCA/compliance/dossier output schemas, write-back strategy | RCA, compliance, dossier, evaluator, annotations, phoenix evals |
| [`implementation_journey.md`](./implementation_journey.md) | `specs/`, `execplan/`, `apps/`, `investigator/` | codified implementation playbook from spec to validated delivery for this repo | implementation journey, spec-first, contract-first, deterministic loop |
| [`rlm_runtime_contract.md`](./rlm_runtime_contract.md) | `investigator/runtime/` | runtime-level contract for recursion loop, sandboxing, tool access, budgets, and failure handling | rlm runtime, sandbox, recursion, budgets, tool contract |
| [`rlm_engines.md`](./rlm_engines.md) | `investigator/rca/`, `investigator/compliance/`, `investigator/incident/` | source-of-truth detailed behavior spec for the three independent RLM engines | rlm engines, rca engine, compliance engine, incident engine |
| [`formal_contracts.md`](./formal_contracts.md) | `investigator/schemas/`, `controls/library/`, `artifacts/investigator_runs/` | canonical formal schemas and compatibility rules for run/output/annotation/control artifacts | formal contracts, schema, runrecord, compliance report, incident dossier |

## Execution Artifacts

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [`DESIGN.md`](../DESIGN.md) | `artifacts/investigator_runs/` | RunRecord-equivalent contract required for every RCA/compliance/dossier run | run record, reproducibility, audit trail, evaluator run |
| [`formal_contracts.md`](./formal_contracts.md) | `artifacts/investigator_runs/` | required field-level contract for RunRecord and evaluator outputs | schema validation, required fields, artifact contracts |

## Phase Status

- Phases 1-5 are active in scope.
- RLM is the core implementation path for Phase 2 (RCA), Phase 3 (policy compliance), and Phase 4 (incident dossier).
- Phase 5 is hardening and expansion of the same RLM runtime, with these pinned constraints:
  - read-only Inspection API boundary
  - RCA/compliance/dossier output schema compatibility
  - sandbox policy (no network/filesystem; allowlisted tool APIs only)

## External RLM References

- `rlm/2512.24601v2.pdf` (primary recursive-programming reference)
- `https://kmad.ai/Recursive-Language-Models-Security-Audit` (security-oriented implementation reference)
