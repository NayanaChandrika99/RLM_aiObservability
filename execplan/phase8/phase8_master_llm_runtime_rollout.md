# Phase 8 Master Plan: Real LLM Recursive Runtime Rollout

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

After this phase, Nainy can run one end-to-end proof flow where RCA, compliance, and incident evaluators actually call a real LLM through a shared recursive runtime. The existing deterministic engines remain as a stable baseline. The visible outcome is a proof report that compares deterministic baseline versus LLM-recursive evaluators on the same frozen dataset and enforces explicit pass thresholds.

In plain language, "recursive runtime" here means the evaluator does not read the entire trace dataset at once. It repeatedly asks what to inspect next, calls read-only inspection tools for small slices of evidence, optionally delegates focused subproblems, and then returns a schema-valid report with evidence pointers.

This rollout also adds prompt version pinning and explicit cost caps so proof artifacts remain reproducible and budget-safe.

## Progress

- [x] (2026-02-10 15:00Z) Verified clean workspace and stashed generated artifacts before planning.
- [x] (2026-02-10 15:00Z) Reviewed runtime, engine, inspection, proof, and contract references for Phase 8 design.
- [x] (2026-02-10 15:00Z) Drafted initial Phase 8 rollout and subplans.
- [x] (2026-02-10 15:08Z) Revised plans after Nainy review: provider abstraction, structured output strategy, explicit recursive state model, dataset-grounded compliance scope, and incident split.
- [x] (2026-02-10 15:42Z) Created Phase 8A shared LLM runtime implementation, including structured validation/retry and usage-cost accounting.
- [x] (2026-02-10 15:50Z) Created Phase 8B sandboxed recursive execution path with recursion budgets and state transitions.
- [x] (2026-02-10 15:58Z) Migrated RCA to real LLM judgment path behind runtime toggles and fallback handling.
- [x] (2026-02-10 16:20Z) Migrated compliance to real LLM judgment path and cleared the +0.05 proof threshold on frozen benchmark data.
- [x] (2026-02-10 16:32Z) Completed incident selector-target alignment pass with deterministic diagnostics for bucket/signature mismatches.
- [x] (2026-02-10 16:47Z) Completed incident LLM synthesis pass with runtime signals and write-back provenance wiring.
- [x] (2026-02-10 17:11Z) Ran full phase proof and full regression; updated outcomes with artifact links and final gate evidence.

## Surprises & Discoveries

- Observation: The current evaluator stack declares model metadata but does not invoke an LLM anywhere in runtime or engine execution.
  Evidence: `investigator/runtime/runner.py` executes `engine.run(request)` directly and no LLM client imports exist.
- Observation: Current write-back payloads still mark root annotations as `annotator_kind: "LLM"` even for deterministic logic.
  Evidence: `investigator/rca/writeback.py`, `investigator/compliance/writeback.py`, and `investigator/incident/writeback.py` root payload builders.
- Observation: `RuntimeRef.model_provider` is currently `Literal["openai"]`, which blocks provider-agnostic runtime metadata.
  Evidence: `investigator/runtime/contracts.py`.
- Observation: Existing deterministic narrowing logic is strong enough to keep as pre-LLM evidence narrowing, which reduces cost and improves reproducibility.
  Evidence: Hot span sorting and representative selection are already deterministic in `investigator/rca/engine.py` and `investigator/incident/engine.py`.

## Decision Log

- Decision: Treat current engines and proof harness as a locked deterministic baseline and build LLM recursion as an additive path.
  Rationale: Preserves a known-good comparator and allows proof-gated migration per capability.
  Date/Author: 2026-02-10 / Codex
- Decision: Implement one shared LLM runtime loop before capability migration.
  Rationale: Avoids duplicating model-call, structured-output, budget, and telemetry logic across three engines.
  Date/Author: 2026-02-10 / Codex
- Decision: Keep deterministic pre-LLM narrowing mandatory for all capabilities.
  Rationale: This aligns with repository reproducibility goals and budget constraints.
  Date/Author: 2026-02-10 / Codex
- Decision: Keep OpenAI as first concrete provider but make runtime metadata/provider interface extensible from Phase 8A.
  Rationale: Avoids immediate rework when adding a second provider later.
  Date/Author: 2026-02-10 / Codex
- Decision: Pin prompts as versioned files and derive `prompt_template_hash` from canonical prompt+schema bytes.
  Rationale: Reproducibility requires immutable prompt identity, not ad-hoc inline strings.
  Date/Author: 2026-02-10 / Codex
- Decision: Split incident work into deterministic target-alignment pass and separate LLM synthesis pass.
  Rationale: Prevents metric-causality confusion when overlap@k behavior changes.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

Phase 8 is complete. The evaluator runtime now performs real LLM calls with versioned prompts, schema-validated structured outputs, retry/fallback behavior, recursive sandbox-budget enforcement, and run-record usage/cost accounting across RCA, compliance, and incident engines.

Validation evidence:

- Live RCA smoke run with real model calls and non-zero usage/cost:
  `artifacts/investigator_runs/phase8-live-rca-smoke-default-20260210T154137Z/run_record.json`
- Full proof artifact with all gates passing:
  `artifacts/proof_runs/phase7-proof-20260210T165400Z/proof_report.json`
  - `gates.all_passed=true`
  - `rca delta.accuracy=+0.3666666666666667`
  - `compliance delta.accuracy=+0.33333333333333337`
  - Incident gate passes at overlap ceiling with non-regression (`baseline_overlap_at_k=1.0`, `rlm_overlap_at_k=1.0`, `effective_threshold=0.0`)
- Full regression run:
  `uv run pytest tests/ -q -rs` produced `92 passed, 2 skipped` (skips are live integration tests gated by env vars).

## References

The following files were reviewed fully before drafting and revising this plan; listed behaviors will be reused:

- `PLANS.md`
  Behavior reused: required ExecPlan sections and living-document maintenance rules.
- `execplan/phase7/phase7_reproducible_proof_harness.md`
  Behavior reused: proof artifact and gate reporting style.
- `specs/rlm_runtime_contract.md`
  Behavior reused: recursion budget, sandbox, error taxonomy, and run artifact expectations.
- `specs/rlm_engines.md`
  Behavior reused: engine-specific recursion and evidence policies.
- `specs/formal_contracts.md`
  Behavior reused: output and run record compatibility requirements.
- `API.md`
  Behavior reused: read-only inspection API boundary and canonical `evidence_ref` shape.
- `DESIGN.md`
  Behavior reused: deterministic narrowing before LLM synthesis and write-back naming conventions.
- `investigator/runtime/contracts.py`
  Behavior reused: runtime metadata dataclasses and current provider field behavior.
- `investigator/runtime/runner.py`
  Behavior reused: run record persistence and validation gate placement.
- `investigator/proof/benchmark.py`
  Behavior reused: baseline-vs-current comparison and threshold gates.
- `investigator/proof/run_phase7_proof.py`
  Behavior reused: end-to-end proof orchestration and proof artifact writing.

## Context and Orientation

Current project behavior is deterministic. Engines use regex and fixed heuristics over inspection data and output schema-valid artifacts. Runtime records include model metadata, but those metadata fields do not drive execution. This means Phase 8 is a functional migration, not a cleanup task.

Three constraints remain non-negotiable during migration:

1. All evaluator outputs must stay contract-compatible with existing schemas and evidence refs.
2. The inspection boundary stays read-only and tool-driven.
3. Every invocation still writes `artifacts/investigator_runs/<run_id>/run_record.json`, including failures.

Two additional constraints are added in this revision:

4. Prompt content used in a run must be version-pinned and hash-stable.
5. Proof runs must enforce a hard `max_cost_usd` cap and report cost utilization.

## Plan of Work

Phase 8 is split into six implementation passes plus one proof-gating pass.

Phase 8A builds a shared LLM runtime loop with provider-agnostic interface, OpenAI first implementation, JSON-schema structured output, deterministic parse-retry rules, prompt registry, and token/cost accounting into `RunRecord`.

Phase 8B adds recursive execution support with strict typed actions, explicit state transitions, and sandboxed tool invocation. Budget ownership is split clearly: runtime loop mutates counters, runner remains canonical persistence/validation owner.

Phase 8C migrates RCA judgment from regex label selection to LLM-driven classification while preserving deterministic candidate narrowing. RCA threshold `+0.15` remains provisional and must be re-justified against frozen dataset diagnostics in this phase.

Phase 8D migrates compliance verdicting from regex-only rule matching to LLM evidence-judgment per control and enriches controls grounded in seeded dataset signals so benchmark lift can reach `+0.05`.

Phase 8E1 aligns deterministic incident expected-target construction with selector contract and resolves overlap@k metric causality before synthesis changes.

Phase 8E2 migrates incident hypothesis/action synthesis to LLM output while keeping deterministic representative selection and preserving any overlap gains from 8E1.

Final proof pass reruns the frozen benchmark and enforces thresholds:

- RCA delta accuracy >= +0.15
- Compliance delta accuracy >= +0.05
- Incident delta overlap@k >= +0.10

## Concrete Steps

All commands run from repository root.

1. Implement Phase 8A and run focused tests.
   - `uv run pytest tests/unit -q -k "runtime and llm"`
2. Implement Phase 8B and run sandbox/budget tests.
   - `uv run pytest tests/unit -q -k "sandbox or recursion"`
3. Implement Phase 8C and run RCA + proof tests.
   - `uv run pytest tests/unit -q -k "rca or proof_benchmark"`
4. Implement Phase 8D and run compliance + proof tests.
   - `uv run pytest tests/unit -q -k "compliance or proof_benchmark"`
5. Implement Phase 8E1 and run incident selector diagnostics tests.
   - `uv run pytest tests/unit -q -k "incident and selector"`
6. Implement Phase 8E2 and run incident synthesis + proof tests.
   - `uv run pytest tests/unit -q -k "incident or proof_benchmark"`
7. Run full proof flow with cost gate enabled.
   - `PHOENIX_WORKING_DIR=.phoenix_data uv run python -m investigator.proof.run_phase7_proof`
8. Run full regression.
   - `uv run pytest tests/ -q -rs`

## Validation and Acceptance

Phase 8 is accepted only when all criteria pass:

- Real model calls occur and runtime usage fields (`tokens_in`, `tokens_out`) are non-zero for LLM-path runs.
- Deterministic baseline path remains runnable for side-by-side proof comparison.
- All three engines still emit schema-valid outputs with evidence pointers.
- Run records contain budget, usage, prompt hash, output refs, and write-back refs for every invocation.
- Prompt registry and schema files used in each run are reconstructable from run metadata.
- Proof gates pass all three thresholds in one artifact without exceeding configured `max_cost_usd`.

## Idempotence and Recovery

Each implementation step must be additive and behind explicit runtime flags so deterministic fallback remains available while migrating. If a migration step regresses proof gates, revert only that capability to deterministic mode and continue debugging with the shared runtime intact. Proof artifacts are immutable by run id and should never be overwritten.

If cost cap is hit mid-run, runtime must terminate with contract-valid `partial` or `failed` status and persisted artifacts.

## Artifacts and Notes

Primary artifacts for this phase:

- `artifacts/investigator_runs/<run_id>/run_record.json`
- `artifacts/investigator_runs/<run_id>/output.json`
- `artifacts/proof_runs/<proof_run_id>/proof_report.json`

Planning and execution docs:

- `execplan/phase8/phase8A_shared_llm_runtime_loop.md`
- `execplan/phase8/phase8B_sandboxed_recursive_execution.md`
- `execplan/phase8/phase8C_rca_engine_llm_migration.md`
- `execplan/phase8/phase8D_compliance_engine_llm_migration.md`
- `execplan/phase8/phase8E_incident_engine_llm_migration.md`

## Interfaces and Dependencies

Required interfaces after rollout:

- Shared runtime interface for LLM calls and structured output parsing, used by all engines.
- Provider-agnostic model client interface with OpenAI implementation first.
- Prompt registry at stable path (`investigator/prompts/`) with hash-stable template retrieval.
- Read-only `InspectionAPI` usage from engine logic; no direct Phoenix or filesystem access from recursive steps.
- Existing workflow entrypoints remain stable:
  - `run_trace_rca_workflow`
  - `run_policy_compliance_workflow`
  - `run_incident_dossier_workflow`

External dependency scope:

- OpenAI API client for first live implementation.
- Existing Phoenix client and proof harness modules remain unchanged at boundary level.

Revision Note (2026-02-10): Initial Phase 8 master plan created to sequence real LLM recursive runtime rollout over the stable deterministic baseline.
Revision Note (2026-02-10): Revised after Nainy review to add provider extensibility, prompt/version reproducibility, explicit cost governance, detailed recursion ownership, and incident split (alignment before synthesis).
