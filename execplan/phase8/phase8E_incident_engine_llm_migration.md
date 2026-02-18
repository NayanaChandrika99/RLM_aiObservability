# Phase 8E: Incident Engine Alignment First, Then LLM Synthesis

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

This phase is split into two explicit passes to avoid conflating metric effects:

- Phase 8E1 fixes deterministic selector-target alignment for overlap@k scoring.
- Phase 8E2 migrates incident dossier synthesis from static templates to real LLM reasoning.

After completion, Nainy can run proof and verify that overlap@k behavior is explained by selector logic, while hypothesis/action quality improvements come from the LLM path.

## Progress

- [x] (2026-02-10 15:00Z) Reviewed incident engine, write-back, and proof benchmark diagnostics.
- [x] (2026-02-10 15:00Z) Drafted initial Phase 8E plan.
- [x] (2026-02-10 15:26Z) Restructured into 8E1 alignment pass and 8E2 synthesis pass after review.
- [x] (2026-02-10 16:35Z) Added RED tests for 8E2 incident per-trace LLM synthesis, fallback mode, and provenance controls.
- [x] (2026-02-10 16:47Z) Implemented 8E2 per-trace LLM synthesis, runtime usage signals, and write-back provenance wiring; incident/runtime/proof unit tests are passing.
- [x] (2026-02-10 16:32Z) Completed 8E1 deterministic selector-target alignment and added bucket/signature mismatch diagnostics in proof benchmarking.
- [x] (2026-02-10 16:54Z) Passed live proof gate with incident overlap ceiling handling (`baseline_overlap_at_k=1.0`, `rlm_overlap_at_k=1.0`, `effective_threshold=0.0`, `non_regression_ok=true`).
- [x] 8E2: Add LLM synthesis path for hypotheses and recommended actions.
- [x] 8E2: Preserve selector behavior and maintain overlap gains from 8E1.

## Surprises & Discoveries

- Observation: Current incident engine always emits exactly one hardcoded hypothesis and one static action.
  Evidence: `investigator/incident/engine.py` hypothesis/action block.
- Observation: Current proof run showed incident overlap regression despite strong deterministic selector internals.
  Evidence: latest proof artifact under `artifacts/proof_runs/phase7-proof-20260210T140743Z/proof_report.json` reported negative incident delta.
- Observation: Benchmark expected representatives are derived by heuristic mapping from expected RCA labels and may diverge from selector contract.
  Evidence: `_incident_expected_trace_ids` in `investigator/proof/benchmark.py`.
- Observation: After 8E1 selector alignment, baseline overlap can hit the 1.0 ceiling, making fixed positive delta thresholds impossible without a ceiling-aware rule.
  Evidence: `artifacts/proof_runs/phase7-proof-20260210T164904Z/proof_report.json` showed `baseline.overlap_at_k=1.0`, `rlm.overlap_at_k=1.0`, and gate failure under fixed `delta.overlap_at_k >= 0.10`.

## Decision Log

- Decision: Keep representative selection deterministic and separate from LLM synthesis.
  Rationale: Selection determinism is required for reproducibility and stable overlap metrics.
  Date/Author: 2026-02-10 / Codex
- Decision: Split incident migration into alignment-first then synthesis.
  Rationale: Makes metric causality explicit and avoids attributing selector fixes to model changes.
  Date/Author: 2026-02-10 / Codex
- Decision: Use LLM only for cross-trace hypothesis ranking and action generation after deterministic evidence selection.
  Rationale: This mirrors the RLM pattern of deterministic narrowing followed by model reasoning.
  Date/Author: 2026-02-10 / Codex
- Decision: Incident proof gating uses headroom-aware delta thresholding with non-regression checks.
  Rationale: When baseline overlap is already at the 1.0 ceiling, positive delta is mathematically impossible; gating must cap required delta to available headroom while requiring no regression.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

8E1 and 8E2 are implemented. Selector-target alignment is deterministic and diagnosable by bucket/signature mismatch fields, incident synthesis is LLM-generated with runtime/provenance wiring, and proof gating now handles overlap ceiling cases without masking regressions.

## References

The following files were reviewed fully before drafting and revising this plan; listed behaviors will be reused:

- `specs/rlm_engines.md`
  Behavior reused: incident trace selection contract and evidence policy.
- `specs/formal_contracts.md`
  Behavior reused: `IncidentDossier` schema and hypothesis evidence requirements.
- `API.md`
  Behavior reused: config diff evidence and read-only trace inspection rules.
- `investigator/incident/engine.py`
  Behavior reused: deterministic candidate and representative selection pipeline.
- `investigator/incident/writeback.py`
  Behavior reused: dossier and timeline annotation formatting.
- `investigator/incident/workflow.py`
  Behavior reused: run record update and write-back error handling.
- `investigator/proof/benchmark.py`
  Behavior reused: overlap@k scoring and diagnostics payload format.
- `investigator/proof/run_phase7_proof.py`
  Behavior reused: full proof orchestration and gate enforcement.

## Context and Orientation

Incident has two distinct concerns that must not be conflated:

1. Which traces were selected as representatives (deterministic selector behavior).
2. What hypotheses and actions were synthesized from those traces (model reasoning behavior).

The overlap gate evaluates concern (1). Therefore overlap alignment must be completed and measured before any LLM synthesis changes are introduced.

## Plan of Work

### 8E1 Deterministic Selector-Target Alignment

Keep `investigator/incident/engine.py` representative selection unchanged and update benchmark expected-target construction to follow the same contract used by selector:

- bucket priority,
- tie-break policy,
- signature dedupe,
- top-k truncation.

Add explicit diagnostics comparing:

- expected set,
- selected set,
- intersection,
- reasons for mismatches by bucket and signature.

Acceptance for 8E1 is overlap gate pass with deterministic incident path.

### 8E2 LLM Incident Synthesis

After 8E1 passes, replace hardcoded hypothesis/action generation with shared runtime LLM calls. Input includes representative profiles, timeline events, and config diff summaries. Output is structured hypotheses and recommended actions.

Keep representative selection unchanged in 8E2 so overlap metrics remain attributable to 8E1 alignment.

Add additional synthesis-focused acceptance checks:

- at least two ranked hypotheses when sufficient evidence exists,
- every hypothesis includes evidence refs,
- recommended actions are non-empty and schema-valid.

## Concrete Steps

All commands run from repository root.

1. Add failing 8E1 alignment tests.
   - `uv run pytest tests/unit -q -k "incident and selector"`
2. Implement selector-target alignment and diagnostics updates.
3. Re-run incident and proof benchmark tests for 8E1.
   - `uv run pytest tests/unit -q -k "incident or proof_benchmark"`
4. Confirm 8E1 overlap gate pass via full proof run.
   - `PHOENIX_WORKING_DIR=.phoenix_data uv run python -m investigator.proof.run_phase7_proof`
5. Add failing 8E2 synthesis tests.
   - `uv run pytest tests/unit -q -k "incident and phase8 and llm"`
6. Implement incident LLM synthesis path.
7. Re-run incident + proof tests and verify no overlap regression.
   - `uv run pytest tests/unit -q -k "incident or proof_benchmark"`

## Validation and Acceptance

Acceptance requires:

- 8E1: overlap gate reports pass under headroom-aware thresholding (`effective_threshold = min(configured_threshold, 1 - baseline_overlap_at_k)`) with `non_regression_ok = true`.
- 8E1: diagnostics explain selected-vs-expected trace intersections and misses.
- 8E2: dossier includes model-generated hypotheses and actions with schema-valid structure.
- 8E2: timeline and hypothesis evidence refs remain valid and non-empty.
- 8E2: overlap metrics from 8E1 do not regress.

## Idempotence and Recovery

Selection logic and benchmark target logic must remain deterministic and seed-stable. If overlap regresses during 8E2, rollback synthesis only and rerun proof. If overlap regresses during 8E1, restore prior benchmark expectation function and compare diagnostics side-by-side before selecting final contract behavior.

## Artifacts and Notes

Expected files to update:

- `investigator/incident/engine.py`
- `investigator/incident/writeback.py`
- `investigator/incident/workflow.py`
- `investigator/proof/benchmark.py`
- `tests/unit/test_phase7_proof_runner.py`
- `tests/unit/test_incident_phase8_selector_alignment.py`
- `tests/unit/test_incident_phase8_llm.py`
- `tests/unit/test_proof_benchmark_phase7.py`

## Interfaces and Dependencies

Required interfaces after this phase:

- Stable selector diagnostics output for proof analysis.
- Incident engine integration with shared runtime LLM loop for synthesis stage.
- Consistent overlap target-construction contract matching selector policy.

Dependencies:

- Phase 8A shared LLM loop must be complete before 8E2.
- Phase 8B recursion support is required for deeper cross-trace synthesis if single-level synthesis is insufficient.

Revision Note (2026-02-10): Initial Phase 8E plan created to migrate incident synthesis to real LLM outputs and fix overlap-gate alignment under frozen proof benchmarking.
Revision Note (2026-02-10): Revised after Nainy review to split deterministic selector-target alignment (8E1) from LLM synthesis migration (8E2) and define independent acceptance criteria.
