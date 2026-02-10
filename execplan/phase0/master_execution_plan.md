# Phoenix-RLM Investigator Master Execution Plan

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

The goal of this project is to deliver three production-relevant RLM investigation capabilities on top of Phoenix traces: Trace RCA, Policy-to-Trace Compliance, and Incident Dossier generation. After completion, a reviewer should be able to pick a trace or incident window in Phoenix, run the corresponding engine, and inspect structured, evidence-linked results that are reproducible and auditable through run artifacts.

The first visible proof of value is not model sophistication. It is operational reliability: repeatable trace generation, repeatable datasets, repeatable evaluator runs, and a strict evidence chain from output back to concrete trace spans and artifacts.

## Progress

- [x] (2026-02-10 00:00Z) Established project contract documents (`AGENTS.md`, `API.md`, `ARCHITECTURE.md`, `DESIGN.md`) with three-engine topology and shared runtime contract.
- [x] (2026-02-10 00:00Z) Validated Phase 1 baseline trace generation locally with Phoenix and tutorial agent.
- [x] (2026-02-10 00:00Z) Created `investigator/` scaffold with shared runtime and three stub engines.
- [x] (2026-02-10 00:00Z) Implemented Phoenix-backed Inspection API module with deterministic ordering and error handling.
- [x] (2026-02-10 00:00Z) Implemented RunRecord persistence to `artifacts/investigator_runs/<run_id>/run_record.json` for every invocation.
- [x] (2026-02-10 00:00Z) Implemented RCA deterministic hot-span narrowing (`ERROR` -> exception signal -> latency -> span_id) with evidence-linked label selection.
- [x] (2026-02-10 00:00Z) Extended RCA to recursive branch inspection with parent-to-child traversal and richer multi-span evidence synthesis.
- [x] (2026-02-10 00:00Z) Implemented Compliance Phase 4A control scoping with deterministic severity ordering and controls-version traceability.
- [x] (2026-02-10 00:00Z) Implemented Compliance Phase 4B required-evidence sufficiency checks with explicit per-control `missing_evidence` contracts.
- [x] (2026-02-10 00:00Z) Added Compliance Phoenix write-back scaffolding and workflow-level RunRecord write-back persistence.
- [x] (2026-02-10 00:00Z) Completed Compliance engine v1 with true pass/fail verdict logic, missing-evidence contracts, and live Phoenix write-back integration coverage.
- [x] (2026-02-10 00:00Z) Implemented Incident Phase 5A deterministic representative trace selection (`error -> latency -> cluster`) with signature dedupe and override support.
- [x] (2026-02-10 00:00Z) Implemented Incident Phase 5B config snapshot correlation with deterministic latest-pair diff evidence and gap handling.
- [x] (2026-02-10 00:00Z) Implemented Incident Phase 5C dossier write-back integration for root and timeline evidence annotations with compatibility fallback.
- [x] (2026-02-10 00:00Z) Implemented RCA Phoenix write-back for root and evidence annotations plus workflow-level RunRecord write-back persistence.
- [x] (2026-02-10 00:00Z) Added runtime sandbox-violation fail-fast and recursion/wall-time budget termination checks in shared `run_engine` orchestration.
- [x] (2026-02-10 00:00Z) Added Phase 6B runtime schema/evidence validation gates with explicit failure taxonomy and workflow write-back error-path coverage.
- [x] (2026-02-10 00:00Z) Added Phase 6C replay-equivalence acceptance tests across RCA, compliance, and incident workflows with deterministic output and write-back assertions.

## Surprises & Discoveries

- Observation: Local Phoenix imports may attempt to use `~/.phoenix`, which can fail under restricted environments.
  Evidence: Earlier runs required `PHOENIX_WORKING_DIR=$PWD/.phoenix_data` to avoid permission errors.
- Observation: Trace export and annotation workflows are stable enough to support iterative evaluator development before full runtime hardening.
  Evidence: Phase 1 tutorial trace run produced consistent spans and exported artifacts.
- Observation: Installed Phoenix package uses deprecated `phoenix.session.client.Client` methods in this environment, but still supports required Phase 2 inspection and evaluation workflows.
  Evidence: Phase 2 implementation and tests pass using `Client.get_spans_dataframe` and evaluator contracts.
- Observation: Deterministic RCA now runs without live Phoenix dependency when a custom inspection API is injected for tests.
  Evidence: Phase 3A RCA tests pass with fake inspection API scenarios for tool/upstream/retrieval classifications.
- Observation: RCA write-back must support current deprecated eval endpoints while preserving annotation naming and annotator-kind semantics for audits.
  Evidence: Added RCA write-back adapter that logs `rca.primary` and `rca.evidence` evaluations and persists write-back metadata in run artifacts.
- Observation: Compliance control scoping quality depends on span-derived app/tool profile and a deterministic override merge path.
  Evidence: Phase 4A tests validate inferred `app_type=agentic`, inferred `tools_used`, and severity-ordered merged control list.
- Observation: Compliance required-evidence evaluation is more reliable when evidence is tracked as requirement-specific pointers rather than boolean signal flags.
  Evidence: Phase 4B implementation now maps each requirement to concrete evidence refs and computes `missing_evidence` per control deterministically.
- Observation: Phoenix annotation APIs vary by endpoint availability across versions and environments; write-back needs a compatibility path.
  Evidence: Phase 4C live integration test initially returned 404 on `/v1/span_annotations` in one run path and succeeded after compatibility fallback handling.
- Observation: Deterministic representative selection quality depends on explicit tie-break policy and override behavior, not just bucket quotas.
  Evidence: Phase 5A tests required stable error-bucket tie-breaks and explicit inclusion of remaining `trace_ids_override` traces.
- Observation: Config-diff correlation quality depends on stable snapshot pair selection and explicit insufficiency gaps when snapshots are unavailable.
  Evidence: Phase 5B tests validate latest-pair selection (`base_snapshot_id`, `target_snapshot_id`) and enforce a `gaps[]` path for missing snapshots.
- Observation: Incident write-back compatibility must handle Phoenix client surface differences between span-annotation APIs and evaluation logging.
  Evidence: Phase 5C write-back uses span-annotation path first and falls back to evaluation logging; unit and incident-focused suites pass.
- Observation: Runtime guardrails require a stable engine-to-runtime signal channel for deterministic budget checks without tightly coupling engine internals.
  Evidence: Phase 6A adds `get_runtime_signals()` support in `run_engine` and validates iteration/depth/tool/subcall/token/wall-time enforcement via unit tests.
- Observation: Schema and evidence validation should run before write-back to prevent persisting invalid evaluator artifacts.
  Evidence: Phase 6B adds runtime validators and fails with `SCHEMA_VALIDATION_FAILED`/`EVIDENCE_VALIDATION_FAILED` before output/write-back on malformed payloads.
- Observation: Write-back failure behavior must be verified per engine workflow, not only at runtime-core level.
  Evidence: Phase 6B adds workflow tests confirming `RCA_WRITEBACK_FAILED`, `COMPLIANCE_WRITEBACK_FAILED`, and `INCIDENT_WRITEBACK_FAILED` persist as `partial` runs.
- Observation: Replay equivalence is strongest when validated at workflow level (runtime + engine + write-back metadata), not engine-only unit boundaries.
  Evidence: Phase 6C runs each workflow twice with fixed inputs and confirms identical output artifacts plus stable annotation-name contracts.

## Decision Log

- Decision: Keep three capabilities as separate engines from day one and share one runtime contract.
  Rationale: Capabilities have different narrowing logic and acceptance behavior; shared runtime avoids duplicated safety and audit code.
  Date/Author: 2026-02-10 / Codex
- Decision: Keep project Docker-optional and Python-first through Phases 1-3.
  Rationale: Faster local iteration and direct alignment with Phoenix Python workflows.
  Date/Author: 2026-02-10 / Codex
- Decision: Use Parquet as canonical exported dataset format in early phases.
  Rationale: Stable schema, fast filtering, deterministic reruns.
  Date/Author: 2026-02-10 / Codex
- Decision: Keep Inspection API adapter compatible with current installed Phoenix client while isolating it behind `PhoenixInspectionAPI` for future client upgrade.
  Rationale: Enables immediate progress on evaluator engines without blocking on SDK migration.
  Date/Author: 2026-02-10 / Codex
- Decision: Ship deterministic RCA narrowing first, then layer recursive branch inspection in a separate subphase.
  Rationale: Deterministic hot-span selection is required for reproducibility and provides immediate RCA quality gains.
  Date/Author: 2026-02-10 / Codex
- Decision: Use a compatibility-first RCA write-back path via Phoenix evaluations while encoding `annotator_kind` and `run_id` in explanation payloads.
  Rationale: Current installed Phoenix client supports eval logging reliably across environments; metadata remains explicit for audit and replay.
  Date/Author: 2026-02-10 / Codex
- Decision: Phase 4A compliance engine uses deterministic control scoping first (inferred profile + overrides) before deeper recursive evidence logic.
  Rationale: This makes control coverage and controls-version traceability testable now while preserving a clear path to Phase 4B evidence sufficiency rules.
  Date/Author: 2026-02-10 / Codex
- Decision: Compliance write-back uses Phoenix span-annotation APIs when available and falls back to evaluation logging to preserve compatibility.
  Rationale: Current environments differ in Phoenix client surface; compatibility-first write-back keeps annotation naming and audit metadata stable.
  Date/Author: 2026-02-10 / Codex
- Decision: Compliance Phase 4C verdict logic is deterministic-first and control-driven (`violation_patterns`, `max_error_spans`) after evidence sufficiency checks.
  Rationale: It provides explicit fail semantics without requiring opaque model behavior and keeps reruns comparable under fixed inputs.
  Date/Author: 2026-02-10 / Codex
- Decision: Incident Phase 5A uses deterministic bucket selection with error tie-break by `trace_id` and signature dedupe across buckets.
  Rationale: This preserves reproducibility while preventing over-representation of near-duplicate traces in representative sets.
  Date/Author: 2026-02-10 / Codex
- Decision: Incident Phase 5B correlates config diffs using the two latest snapshots by `(created_at, snapshot_id)` and only asserts change types when `diff_ref` is present.
  Rationale: This keeps incident suspected-change claims deterministic and evidence-resolvable under the shared evidence pointer contract.
  Date/Author: 2026-02-10 / Codex
- Decision: Incident Phase 5C write-back logs `incident.dossier` as `LLM` and `incident.timeline.evidence` as `CODE` while preserving `run_id` in explanation payloads.
  Rationale: This preserves filterable reviewer UX in Phoenix and keeps deterministic timeline evidence auditable per run.
  Date/Author: 2026-02-10 / Codex
- Decision: Phase 6A runtime hardening enforces guardrails in shared orchestration by combining wall-time measurement with engine-reported runtime signals and explicit error taxonomy.
  Rationale: It delivers budget/sandbox contract enforcement immediately across all engines with minimal coupling and consistent RunRecord persistence semantics.
  Date/Author: 2026-02-10 / Codex
- Decision: Phase 6B introduces shared runtime output validators (`schema` then `evidence`) and maps validation failures to contract error codes before write-back.
  Rationale: This centralizes contract enforcement and prevents invalid reports from entering downstream annotations or artifacts.
  Date/Author: 2026-02-10 / Codex
- Decision: Phase 6C acceptance uses deterministic fake inspection environments and workflow-level replay checks as the baseline reproducibility gate for all three engines.
  Rationale: It verifies contract-level determinism end-to-end without requiring live-service variability during routine development.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

The project now has a stable documentation baseline and an executable scaffold for runtime and engine modules. The next outcome target is moving from contract-only stubs to real Phoenix-backed inspection and write-back behavior, while preserving strict run artifact guarantees.

## Context and Orientation

This repository has two active development tracks:

1. `apps/demo_agent/` creates traces and seeded failure datasets.
2. `investigator/` executes offline evaluation engines against traces and writes back results.

The source-of-truth contracts are:

- `API.md` for read-only Inspection API function signatures and evidence reference shape.
- `specs/rlm_runtime_contract.md` for runtime safety, recursion, and budgets.
- `specs/formal_contracts.md` for JSON output and RunRecord shape.
- `specs/rlm_engines.md` for per-engine behavior and acceptance.

Reference repositories in `phoenix/` and `rlm/` are read-only in normal execution unless explicitly requested otherwise.

## Plan of Work

### Phase 1: Trace and Dataset Foundation

This phase guarantees reliable raw material for all engines. The team runs Phoenix locally, runs a tutorial agent, verifies trace visibility, and generates seeded failures with an external ground-truth manifest. Subphase 1A is tutorial trace validation. Subphase 1B is seeded failure generation with deterministic manifest IDs. Subphase 1C is dataset export to Parquet and reproducibility checks.

Acceptance for Phase 1 is that traces are visible and filterable in Phoenix, seeded failures include 30-100 trace runs, and the dataset can be regenerated with equivalent manifest structure.

### Phase 2: Shared Runtime and Inspection API Implementation

This phase replaces stubs with real execution plumbing. Subphase 2A implements Phoenix-backed Inspection API adapters (`list_spans`, `get_span`, `get_tool_io`, `get_retrieval_chunks`, control APIs, and config snapshot/diff APIs). Subphase 2B persists one RunRecord artifact per invocation, including failures. Subphase 2C adds deterministic ordering and explicit budget counters.

Acceptance for Phase 2 is that each engine invocation creates a schema-valid RunRecord with non-empty runtime metadata, deterministic list ordering, and stable evidence references.

### Phase 3: Trace RCA Engine v1

This phase delivers the first real capability. Subphase 3A implements deterministic hot-span narrowing (errors, exception events, latency). Subphase 3B adds recursive branch inspection and evidence accumulation. Subphase 3C emits RCA JSON and writes root plus span evidence annotations back into Phoenix.

Acceptance for Phase 3 is majority match against seeded RCA labels and clickable evidence pointers for each non-trivial RCA claim.

### Phase 4: Policy-to-Trace Compliance Engine v1

This phase delivers governance-grade approvals. Subphase 4A implements control scoping with `controls_version` and override handling. Subphase 4B implements required-evidence checks and `insufficient_evidence` behavior with explicit `missing_evidence`. Subphase 4C emits compliance reports and writes control-linked annotations to Phoenix.

Acceptance for Phase 4 is one verdict per applicable control, controls-version traceability in outputs and write-back, and reviewer-clickable evidence links for failing controls.

### Phase 5: Incident Dossier Engine v1

This phase delivers incident triage and hypothesis synthesis. Subphase 5A implements deterministic representative trace selection (error, latency, dedupe rules). Subphase 5B correlates trace evidence with config snapshot diffs. Subphase 5C emits dossier JSON and trace-linked annotations.

Acceptance for Phase 5 is coherent dossier output with timeline, hypotheses, actions, and evidence links, plus a documented selection rationale for representative traces.

### Phase 6: Runtime Hardening and End-to-End Validation

This phase hardens safety and reproducibility. Subphase 6A enforces sandbox restrictions and recursion budget termination behavior. Subphase 6B adds failure-path tests for schema validation and write-back errors. Subphase 6C runs end-to-end acceptance across all engines on fixed datasets and compares rerun equivalence.

Acceptance for Phase 6 is passing runtime contract checks from `specs/rlm_runtime_contract.md` and repeatable outputs under fixed seeds and budgets.

## Concrete Steps

All commands run from repository root unless noted.

1. Prepare environment and Phoenix service.
   - `uv sync`
   - `PHOENIX_WORKING_DIR=$PWD/.phoenix_data uv run phoenix serve`
2. Validate trace generation.
   - `PHOENIX_WORKING_DIR=$PWD/.phoenix_data uv run python -m apps.demo_agent.phase1_tutorial_run`
3. Generate seeded dataset.
   - `PHOENIX_WORKING_DIR=$PWD/.phoenix_data uv run python -m apps.demo_agent.phase1_seeded_failures`
4. Run unit tests during runtime and engine implementation.
   - `uv run pytest tests/unit -q`
5. Run contract-focused tests as they are added per phase.
   - `uv run pytest tests/unit/test_investigator_runtime_scaffold.py -q`

Expected observable outcomes are listed in each phase acceptance section and must be confirmed before marking a phase complete in `Progress`.

## Validation and Acceptance

Each phase must satisfy both behavioral and artifact acceptance:

- Behavioral acceptance means the feature is observable through Phoenix UI, CLI output, or deterministic test output.
- Artifact acceptance means outputs are schema-valid and a RunRecord artifact exists for each invocation.

For every major completion checkpoint:

- Run unit tests relevant to changed modules.
- Execute one happy-path run and one failure-path run.
- Confirm RunRecord contains status, budget, input refs, and output refs or error block.
- Confirm write-back payloads carry `run_id`, `name`, and `annotator_kind`.

## Idempotence and Recovery

The plan is intentionally idempotent. Re-running data generation or evaluator commands should create new run IDs without corrupting prior artifacts. If a phase fails midway, the recovery path is to rerun the failed subphase after fixing root cause, without deleting prior datasets or artifacts. If a schema contract changes, bump schema version and document migration behavior in `Decision Log` before continuing.

## Artifacts and Notes

Primary artifact paths:

- Seeded dataset manifest: `datasets/seeded_failures/manifest.json`
- Exported trace tables: `datasets/seeded_failures/exports/`
- Evaluator runs: `artifacts/investigator_runs/<run_id>/run_record.json`

Supporting references:

- Runtime contract: `specs/rlm_runtime_contract.md`
- Engine behavior source of truth: `specs/rlm_engines.md`
- Formal schemas: `specs/formal_contracts.md`

## Interfaces and Dependencies

The implementation depends on Python-only modules and Phoenix-compatible tooling.

- Phoenix service and client APIs for trace ingestion, inspection, and annotation write-back.
- OpenInference/OpenTelemetry span semantics for stable span kind behavior.
- OpenAI models (`gpt-5-mini` default, `gpt-5.2` optional override) under explicit budget caps.
- Shared runtime module in `investigator/runtime/` that all engines call.
- Engine modules in `investigator/rca/`, `investigator/compliance/`, and `investigator/incident/`.

The required runtime interfaces are defined in `API.md` and must remain read-only for evidence sources.

Revision Note (2026-02-10): Initial master execution plan created to provide one living phase/subphase guide for full project delivery and tracking.
Revision Note (2026-02-10): Updated progress after completing Phase 2A and 2B implementation and tests.
Revision Note (2026-02-10): Updated progress after Phase 3A deterministic RCA narrowing implementation and tests.
Revision Note (2026-02-10): Updated progress after Phase 3B recursive branch inspection and Phase 3C RCA write-back workflow implementation.
Revision Note (2026-02-10): Updated progress after Phase 4A compliance control scoping and controls-version traceability implementation.
Revision Note (2026-02-10): Updated progress after Phase 4B compliance evidence sufficiency and compliance write-back scaffolding implementation.
Revision Note (2026-02-10): Updated progress after Phase 4C compliance true verdict logic and live Phoenix write-back integration tests.
Revision Note (2026-02-10): Updated progress after Phase 5A incident representative trace selection and seeded integration tests.
Revision Note (2026-02-10): Updated progress after Phase 5B incident config snapshot diff correlation and evidence-linking tests.
Revision Note (2026-02-10): Updated progress after Phase 5C incident write-back workflow integration and live-gated roundtrip test coverage.
Revision Note (2026-02-10): Updated progress after Phase 6A runtime sandbox/budget guardrail enforcement and runtime contract tests.
Revision Note (2026-02-10): Updated progress after Phase 6B schema/evidence validation failures and write-back error-path test coverage.
Revision Note (2026-02-10): Updated progress after Phase 6C replay-equivalence acceptance coverage across all three workflows.
