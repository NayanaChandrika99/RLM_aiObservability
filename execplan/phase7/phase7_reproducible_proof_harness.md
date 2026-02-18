# Phase 7 Reproducible Proof Harness (Dataset + Baseline vs RLM)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

After this phase, Nainy can run one deterministic pipeline that: (1) generates traces, (2) freezes them into a dataset, and (3) runs baseline and RLM-style evaluators on the exact same frozen dataset for RCA, policy compliance, and incident dossier. The user-visible output is a single proof artifact that reports side-by-side metrics and references the exact run outputs so claims are auditable and rerunnable.

## Progress

- [x] (2026-02-10 08:43Z) Reviewed Phase 7 references and existing runtime/engine/dataset code.
- [x] (2026-02-10 08:58Z) Implemented `ParquetInspectionAPI` and manifest trace-link helper; added unit tests.
- [x] (2026-02-10 09:03Z) Added deterministic proof benchmark runner for RCA/compliance/incident baseline-vs-RLM comparisons; added unit tests.
- [x] (2026-02-10 09:06Z) Added `investigator.proof.run_phase7_proof` with live generation flow + frozen dataset proof artifact writer.
- [x] (2026-02-10 09:08Z) Executed full proof flow and generated `artifacts/proof_runs/phase7-proof-20260210T140743Z/proof_report.json`.
- [x] (2026-02-10 09:09Z) Ran full test suite: `60 passed, 2 skipped`.

## Surprises & Discoveries

- Observation: Current seeded manifest stores `run_id` and expected labels but leaves `trace_id` null.
  Evidence: `datasets/seeded_failures/manifest.json` currently has `trace_id: null` entries.
- Observation: Existing engines already run against injected inspection APIs, which makes a dataset-backed adapter low-risk.
  Evidence: Unit tests for all three engines pass fake inspection implementations in `tests/unit/`.
- Observation: Running tutorial trace generation and seeded trace generation in one Python process triggered `Overriding of current TracerProvider is not allowed`.
  Evidence: First end-to-end proof run produced `seeded_export_rows: 0` and the OpenTelemetry override warning.
- Observation: Isolating trace-generation steps in subprocesses resolved the tracer-provider conflict.
  Evidence: Second end-to-end proof run produced `seeded_export_rows: 103` and dataset `trace_count: 30`.

## Decision Log

- Decision: Treat the existing engine implementations as the current "RLM-style" system under test and compare them against simpler deterministic baselines in this phase.
  Rationale: This allows immediate measurable comparison using current code while deferring deeper runtime recursion redesign.
  Date/Author: 2026-02-10 / Codex
- Decision: Build Parquet-backed inspection as an `InspectionAPI` implementation rather than adding dataset-specific logic into engines.
  Rationale: Keeps engine code unchanged and preserves the read-only API boundary.
  Date/Author: 2026-02-10 / Codex
- Decision: Use root-only deterministic baseline for RCA to ensure a simple, reproducible baseline floor.
  Rationale: A weaker but explicit baseline is easier to reason about than partially overlapping heuristics when measuring lift.
  Date/Author: 2026-02-10 / Codex
- Decision: Define incident score as overlap@k against deterministic expected representatives derived from error-count/latency sorting.
  Rationale: Gives a concrete numeric comparison without requiring manual incident labels.
  Date/Author: 2026-02-10 / Codex
- Decision: Execute `apps.demo_agent.phase1_tutorial_run` and `apps.demo_agent.phase1_seeded_failures` in subprocesses inside the proof runner.
  Rationale: Avoids global tracer-provider reuse issues that occur when both run in the same interpreter process.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

Phase completed with a reproducible proof artifact and passing tests. The generated artifact contains dataset hash, side-by-side capability metrics, and linked evaluator run IDs. Observed metrics show RCA lift over baseline, no compliance delta in this dataset, and negative incident delta under current expected-representative scoring. Remaining gap: incident benchmark expectations and scoring need refinement before claiming end-to-end lift across all three capabilities.

## References

The following files were reviewed fully before drafting this plan; listed behaviors will be reused:

- `apps/demo_agent/phase1_seeded_failures.py`
  Behavior reused: deterministic seeded failure generation and Parquet export path conventions.
- `apps/demo_agent/phase1_tutorial_run.py`
  Behavior reused: local Phoenix + OpenAI agent trace generation workflow.
- `investigator/inspection_api/protocol.py`
  Behavior reused: all evaluator data access must remain through the shared read-only interface.
- `investigator/inspection_api/phoenix_client.py`
  Behavior reused: stable span ordering, tool/retrieval/control/config helper semantics.
- `investigator/rca/engine.py`
  Behavior reused: existing recursive branch inspection path as comparison candidate.
- `investigator/compliance/engine.py`
  Behavior reused: control scoping + evidence sufficiency pipeline.
- `investigator/incident/engine.py`
  Behavior reused: deterministic representative trace selection and config diff correlation.
- `investigator/runtime/runner.py`
  Behavior reused: run artifact persistence and validation gates.
- `tests/unit/test_phase6_replay_acceptance.py`
  Behavior reused: replay-equivalence expectations and workflow-level verification style.

## Context and Orientation

The repository already has three evaluator engines and one Phoenix-backed inspection adapter. The missing proof piece is a deterministic offline evaluation path where both baseline and RLM-style methods read from the same frozen dataset instead of live state. In this repo, "frozen dataset" means a Parquet span table plus external manifest under `datasets/seeded_failures/`.

## Plan of Work

Phase 7A introduces `ParquetInspectionAPI` under `investigator/inspection_api/` and keeps the same method signatures as `InspectionAPI`. The adapter reads Parquet once, supports deterministic filtering, and reuses existing controls/config lookup behavior.

Phase 7B adds benchmark code under `investigator/proof/` with three baseline evaluators and one comparison runner. The runner executes baseline and current RLM-style engines on identical trace sets and computes measurable metrics per capability.

Phase 7C updates dataset generation so manifest entries are linked to emitted `trace_id` values, making RCA accuracy computation explicit and reproducible.

Phase 7D adds a top-level command script that performs the full flow: ensure Phoenix reachable, run tutorial trace generation, run seeded generation/export, execute baseline-vs-RLM comparisons from frozen dataset, and write a single proof artifact JSON.

## Concrete Steps

All commands run from repository root.

1. Red test for Parquet-backed inspection adapter:
   - `uv run pytest tests/unit/test_parquet_inspection_api_phase7.py -q`
2. Implement adapter and rerun test.
3. Red test for proof benchmark runner:
   - `uv run pytest tests/unit/test_proof_benchmark_phase7.py -q`
4. Implement proof runner and rerun test.
5. End-to-end run command:
   - `PHOENIX_WORKING_DIR=.phoenix_data uv run python -m investigator.proof.run_phase7_proof`
6. Full regression:
   - `uv run pytest tests/ -q -rs`

## Validation and Acceptance

Acceptance requires all of the following:

- `ParquetInspectionAPI` can satisfy engine reads for RCA/compliance/incident without Phoenix access.
- Proof runner generates one artifact under `artifacts/proof_runs/<proof_run_id>/proof_report.json` with:
  - dataset reference (`manifest_path`, `spans_parquet_path`, dataset hash)
  - baseline metrics and RLM-style metrics for RCA/compliance/incident
  - per-capability delta summaries
  - links to evaluator run artifacts generated during the proof run
- Tests pass and include at least one new unit test that would fail before these changes.

## Idempotence and Recovery

The proof runner must be idempotent: each execution creates a new proof run directory and does not overwrite prior artifacts. If Phoenix is unavailable, the runner should fail with an explicit error before benchmark execution. If seeded generation succeeds but benchmark fails, rerunning should regenerate artifacts with a new proof run id.

## Artifacts and Notes

Primary new artifacts for this phase:

- `artifacts/proof_runs/<proof_run_id>/proof_report.json`
- Existing evaluator run artifacts under `artifacts/investigator_runs/<run_id>/`

## Interfaces and Dependencies

Required interfaces after this phase:

- `investigator.inspection_api.parquet_client.ParquetInspectionAPI`
  - Implements `InspectionAPI` methods used by all three engines.
- `investigator.proof.baselines`
  - Exposes deterministic baseline evaluators for RCA/compliance/incident.
- `investigator.proof.run_phase7_proof`
  - Entry point that orchestrates dataset-linked benchmark runs and writes proof artifact.

Revision Note (2026-02-10): Initial Phase 7 ExecPlan created to deliver reproducible baseline-vs-RLM proof runs on frozen datasets.
Revision Note (2026-02-10): Updated Progress, discoveries, decisions, and outcomes after implementing Phase 7 and running the full proof pipeline.
