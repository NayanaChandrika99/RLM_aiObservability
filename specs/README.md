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

## Phase 10 — RLM-RCA System

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [`RLM_RCA_ARCHITECTURE.md`](../execplan/phase10/RLM_RCA_ARCHITECTURE.md) | `investigator/rca/`, `investigator/runtime/` | RLM-based RCA system architecture, REPL-primary execution, per-hypothesis recursion, subprocess sandbox (diagram-rich reference) | rlm rca, repl harness, per-hypothesis, subprocess sandbox, architecture |
| [`RLM_RCA_ARCHITECTURE_EXECPLAN.md`](../execplan/phase10/RLM_RCA_ARCHITECTURE_EXECPLAN.md) | `investigator/rca/`, `investigator/runtime/` | PLANS.md-compliant architecture ExecPlan with decision log, validation steps, and novice orientation | rlm rca, architecture, execplan, decision log |
| [`RLM_RCA_IMPLEMENTATION_PLAN.md`](../execplan/phase10/RLM_RCA_IMPLEMENTATION_PLAN.md) | `apps/demo_agent/`, `investigator/rca/`, `investigator/runtime/` | 6-phase implementation plan for autonomous RCA via RLM (table-rich reference) | implementation plan, fault injector, cli, evaluation, phase 10 |
| [`RLM_RCA_IMPLEMENTATION_EXECPLAN.md`](../execplan/phase10/RLM_RCA_IMPLEMENTATION_EXECPLAN.md) | `apps/demo_agent/`, `investigator/rca/`, `investigator/runtime/` | PLANS.md-compliant implementation ExecPlan with milestones, concrete steps, and acceptance criteria | implementation, execplan, milestones, fault injector, cli, evaluation |

## Phase Status

- Phases 1-5 are active in scope.
- **Phase 10** (RLM-RCA system) is design-locked and ready for implementation.
- RLM is the core implementation path for Phase 2 (RCA), Phase 3 (policy compliance), and Phase 4 (incident dossier).
- Phase 5 is hardening and expansion of the same RLM runtime, with these pinned constraints:
  - read-only Inspection API boundary
  - RCA/compliance/dossier output schema compatibility
  - sandbox policy (no network/filesystem; subprocess sandbox with import blocklist)

## External RLM References

- `rlm/2512.24601v2.pdf` (primary recursive-programming reference)
- `https://kmad.ai/Recursive-Language-Models-Security-Audit` (security-oriented implementation reference)
- `https://www.primeintellect.ai/blog/rlm` (Prime Intellect RLM implementation — tools, sub-LLMs, sandboxing)
- `https://alexzhang13.github.io/blog/2025/rlm/` (RLM explainer)
- `https://www.dbreunig.com/2026/02/09/the-potential-of-rlms.html` (RLM potential and patterns)
