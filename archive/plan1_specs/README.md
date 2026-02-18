# Specs Pin Index

This is the lookup table (“pin”) for agents and humans.
It exists to improve retrieval (search hit-rate) and reduce invention.

Rules:

- Add/update entries here whenever you add/change a spec in `specs/`.
- Keep **Purpose** and **Keywords** rich in synonyms.
- Link to repo-relative **Code** locations once code exists (use `—` until then).
- Avoid dangling spec links: only list spec files that exist.

Recommended table format: `Spec | Code | Purpose | Keywords`.

## Core

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [vision.md](./vision.md) | — | what/why, MVP scope, success criteria, user stories | vision, PRD, goals, scope, success |
| [architecture.md](./architecture.md) | `plan1/scenarios.py`, `plan1/kg/query.py`, `plan1/risk.py`, `plan1/digital_twin.py`, `plan1/scorecard.py` | high-level design, components, dataflow, invariants, outcomes | architecture, system design, workflow, outcomes |
| [knowledge-graph.md](./knowledge-graph.md) | `plan1/kg/generate.py`, `plan1/kg/graph.py`, `plan1/kg/query.py` | synthetic supply network fixtures: schema, realism targets, invariants, determinism | knowledge graph, KG, nodes.json, edges.json, tiers, Tier-4, generator, invariants, determinism |
| [product-mapping.md](./product-mapping.md) | `plan1/product_catalog.py` | single-product scope as Pepsi SKU family + deterministic static BOM lookup | product mapping, BOM, bill of materials, sku family, pepsi |
| [risk.md](./risk.md) | `plan1/risk.py` | deterministic Tier-1 risk scoring model: paper-aligned formula, metric definitions, scaling, replay requirements | risk scoring, risk model, Tier-1, pagerank, centrality, normalization, deterministic, replay |

## Orchestration

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [workflow.md](./workflow.md) | `plan1/workflow/graph.py`, `plan1/workflow/nodes.py`, `plan1/workflow/state.py`, `plan1/app.py` | LangGraph StateGraph wiring, interrupts, resume semantics, checkpointing | workflow, orchestration, LangGraph, stategraph, interrupt, HITL, resume, checkpoint, negative cases, LOW_COVERAGE, NO_PATHS |
| [app.md](./app.md) | — | CLI entrypoint for the LangGraph runtime (`python -m plan1.app`) | app, CLI, run, resume, manual incident, thread_id, run_id |

## LLM + Agents

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [llm.md](./llm.md) | — | LLM provider abstraction + record/replay cache for determinism (OpenAI-first) | LLM, openai, OpenAI, GPT, gpt-4o-mini, record/replay, cache, JSON, schema, prompts |
| [sentinel.md](./sentinel.md) | `plan1/agents/sentinel.py`, `plan1/agents/sentinel_runtime.py`, `tests/unit/test_sentinel_runtime.py` | Sentinel Agent: incident text → `IncidentTicket` (LLM-backed) + deterministic diagnostic questions | sentinel, disruption, extraction, classification, IncidentTicket, entities, diagnostic questions, confidence |
| [csco.md](./csco.md) | `plan1/agents/csco.py`, `plan1/workflow/nodes.py`, `tests/unit/test_csco_node.py` | CSCO Agent: risk/exposure → `PlanOption[]` + `ExecutiveActionPlan` | CSCO, architect, plan proposer, mitigations, PlanOption, planning, executive action plan |

## Contracts

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [contracts.md](./contracts.md) | `plan1/contracts.py`, `data/fixtures/contracts/`, `plan1/contracts_schema.py` | artifact schemas + examples: IncidentTicket, KGImpactMap, Tier1RiskTable, PlanOption, TwinResult, Scorecard, ApprovalRecord, ExecutionReceipt, RunRecord | contracts, schema, json, pydantic, artifacts, coverage_flag, uncertainty_flags |
| [plan-options.md](./plan-options.md) | — | deterministic `PlanOption[]` generation rules + `action_type` / `parameters` contract for the IR builder | plan options, planoption, mitigations, actions, action_type, deterministic ids, ordering |

## Digital Twin

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [twin-state.md](./twin-state.md) | `plan1/digital_twin.py`, `plan1/twin_state_validator.py`, `plan1/kg/generate.py`, `data/fixtures/kg/twin_state.json` | canonical schema + invariants for `twin_state.json` snapshot consumed by the Digital Twin | twin_state, snapshot, schema, facilities, lanes, demand, capacity, lead time |
| [optimization-ir.md](./optimization-ir.md) | `plan1/optimization_ir.py`, `plan1/optimization_ir_builder.py`, `plan1/optimization_ir_compiler.py`, `plan1/optimization_ir_validator.py`, `plan1/digital_twin.py` | Optimization IR schema + constraint library + validation rules (OR-Tools) | optimization, OR-Tools, constraints, IR, feasibility |
| [scorecard.md](./scorecard.md) | `plan1/scorecard.py`, `plan1/digital_twin.py`, `tests/unit/test_scorecard.py` | deterministic ranking + recommendation over `TwinResult[]` | scorecard, ranking, recommendation, decisioning, tie-breaker, scoring |
| [simulation.md](./simulation.md) | `plan1/simulation.py`, `plan1/workflow/nodes.py`, `tests/unit/test_simulate_top_k.py` | SimPy discrete-event simulation for plan verification + KPI definitions | simulation, SimPy, discrete-event, KPIs, fill rate, backlog, lead time, emissions |

## Execution

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [execution-policy.md](./execution-policy.md) | — | approval gate rules + policy triggers + when to attempt execution | execution policy, approval, governance, triggers, HITL |
| [executive-gate.md](./executive-gate.md) | `plan1/workflow/nodes.py`, `plan1/workflow/graph.py`, `plan1/app.py` | Executive Gate responsibilities: approval gating, audit artifacts, verifier-first execution preconditions | executive gate, approval gate, HITL, interrupt, approval_record, verifier-first |
| [adapters.md](./adapters.md) | `plan1/adapters/base.py`, `plan1/adapters/registry.py`, `plan1/adapters/erp_stub.py`, `plan1/adapters/monitor_stub.py`, `plan1/executor.py` | adapter boundary for side effects; `dry_run` semantics; receipt/idempotency rules | adapters, execution, receipts, idempotency, dry-run |

## Observability

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [observability.md](./observability.md) | — | tracing spans and artifact linkage; Phoenix/OpenTelemetry exporter rules | observability, tracing, OpenTelemetry, OTEL, Phoenix, spans, trace_ids |

## Reporting

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [report.md](./report.md) | — | deterministic per-run report artifact (`report.json`) derived from `RunRecord` | report, report.json, run summary, audit, reporting, artifacts |

## Testing

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [testing.md](./testing.md) | `tests/integration/test_golden_suite.py`, `tests/golden/` | golden runs, determinism rules, pytest assertions, test layout | testing, pytest, golden, regression, determinism |

## Runbook

| Spec | Code | Purpose | Keywords |
|------|------|---------|----------|
| [runbook.md](./runbook.md) | `plan1/demo.py`, `plan1/runner.py`, `plan1/kg/generate.py` | developer run commands: generate fixtures, run scenarios, resume HITL, run tests | runbook, commands, demo, checklist, how-to |

## Guardrails (Non-Negotiables)

These constraints must hold for the MVP:

- No live RSS/web ingestion in MVP (scenario/manual inputs only).
- No local model inference required (LLM usage must be API-hosted).
- No code generation execution (optimization uses schema-bound Optimization IR).
- Side effects only through adapters; adapters support `dry_run` and always emit receipts.
- Every run emits a `RunRecord` (including early exits/failures).
