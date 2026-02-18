# Architecture

Phoenix-RLM Investigator is an offline "investigation plane" that sits next to Arize Phoenix.

Status: this document is active for Phases 1-5. From first evaluator implementation onward, RLM is the core runtime for three separate engines under one shared contract.

## Scope (Phases 1-5)

- Python-only demo agent + investigators.
- Phoenix is run locally via `phoenix serve` (no Docker required).
- “Distributed traces” means multi-span within one process for now (LLM/tool/retriever/chain spans). True multi-service tracing is deferred.
- RCA, policy compliance, and incident dossier evaluation are RLM-driven (recursive Inspection API exploration) from initial implementation.

## Non-goals (until later phases)

- Multi-service trace propagation across deployments.
- Always-on production incident response runtime. Investigation is offline/asynchronous.
- Logs/metrics correlation backends beyond simple staged connectors.

## Components

- **Phoenix (server + UI)**: stores spans/traces and renders them in the UI.
- **Instrumented app(s)**: demo agent(s) that emit OpenTelemetry traces using OpenInference semantic attributes.
- **Trace export**: pulls spans from Phoenix into a local dataset for evaluation.
- **Evaluators**:
  - Trace RCA engine (RLM-driven, per-trace root cause analysis)
  - Policy-to-Trace Compliance engine (RLM-driven, control-based verdicts with evidence pointers)
  - Incident dossier engine (RLM-driven, per-incident window correlation)
- **Shared RLM runtime contract**: common sandbox, recursion, budget, and run-artifact behavior used by all three engines (`specs/rlm_runtime_contract.md`).
- **Run artifact store**: per-invocation RunRecord-equivalent JSON artifacts for RCA/compliance/dossier runs, including failures.
- **Inspection API** (`API.md`): read-only interface evaluators use to inspect traces and artifacts.
- **Policy control library**: versioned control definitions and required evidence rules used by the compliance auditor.
- **Context sources (staged)**:
  - V1: traces + repo-stored deploy/config snapshots and diffs
  - V2: logs + metrics connectors (separate backends/tools)
- **RLM runtime (core)**: recursive inspector runtime shared by all three engines and connected only to read-only APIs.

## Directory Layout (target)

- `apps/`
  - `demo_agent/` runnable agent harness used to generate traces and seeded failures
- `investigator/`
  - `runtime/` shared runtime components used by all three engines
  - `inspection_api/` Phoenix-backed read-only trace access layer (see `API.md`)
  - `rca/` Trace RCA engine
  - `compliance/` Policy-to-Trace Compliance engine
  - `incident/` Incident dossier engine
  - `schemas/` JSON schemas for outputs
- `controls/`
  - `library/` structured policy controls (YAML/JSON)
- `datasets/`
  - `seeded_failures/` exported traces (Parquet) + external ground-truth manifest
- `configs/`
  - `snapshots/<tag>/` versioned deploy/config snapshots for Phase 3 V1
- `artifacts/`
  - `investigator_runs/<run_id>/run_record.json` reproducibility and audit artifact for each evaluator run
- `connectors/` staged logs/metrics connectors (V2)

## Dataflow

1. Run Phoenix locally.
2. Run a tutorial agent that emits traces to the Phoenix collector endpoint.
3. Confirm traces are visible, navigable, and filterable in the Phoenix UI.
4. Export spans/traces from Phoenix as a dataset.
5. Run three offline RLM engines (RCA, policy compliance, incident dossier) against the exported dataset through the shared runtime contract.
6. Emit one RunRecord-equivalent artifact for each evaluator invocation (success or failure).
7. Write results back into Phoenix as annotations/evals so they appear alongside the original traces.

## Investigation Plane vs Data Plane

- Data plane: the agent run itself (LLM calls, tool calls, retrieval, orchestration). It emits traces.
- Investigation plane: offline jobs that read traces, compute RCA/compliance/dossiers, and write results back as annotations.

The investigation plane must be safe:
- no side effects against the production system
- read-only access to traces/config/log/metric backends
- deterministic steps before LLM synthesis
- durable run audit trail via per-run RunRecord-equivalent artifacts

## OpenTelemetry Signals (why connectors exist)

OpenTelemetry separates signals:
- **Traces**: span timelines and causal relationships (Phoenix-native).
- **Logs**: unstructured/structured events, often high-volume (not the same data path as traces).
- **Metrics**: time-series aggregates (latency histograms, error rates, saturation).

Incident “rich context” requires correlating multiple signals. V1 intentionally stays trace-first and repo-local (deploy/config diffs). V2 adds connectors for logs/metrics and correlates them by time window, service identity, and (when possible) `trace_id`.

For compliance auditing, traces remain the primary evidence source and policy controls are a second required input. The evaluator must map controls -> required evidence -> verdicts with clickable trace/artifact pointers.

## Trust Boundaries

- Evaluators must be **read-only** over observability data.
- The RLM execution environment must be sandboxed: no network/filesystem access, only calls to our Inspection/connector APIs.

## Phoenix Integration Points

- Trace ingestion: OpenTelemetry exporter configured by `phoenix.otel.register(...)`.
- Trace inspection/export: Phoenix `Client` span export/query APIs (see `API.md`).
- Write-back: log annotations/evals keyed by `context.span_id` so results render inside the trace UI.

## CI Reproducibility Stance (Phase 1-4)

- CI does not require full LLM replay-mode enforcement.
- Reproducibility comes from:
  - fixed exported datasets (Parquet)
  - pinned evaluator config + prompt/template hashes
  - mandatory per-run RunRecord-equivalent artifacts
