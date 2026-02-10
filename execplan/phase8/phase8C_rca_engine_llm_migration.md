# Phase 8C: RCA Engine Migration to Real LLM Judgment

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

This phase migrates RCA from deterministic regex label selection to LLM-based evidence judgment, while preserving deterministic hot-span narrowing and branch inspection. After completion, Nainy can inspect RCA outputs where the primary label and summary come from real model reasoning over selected evidence and still pass schema/evidence validation.

The user-visible result is improved RCA lift in proof runs with auditable evidence links and accurate `annotator_kind` metadata.

## Progress

- [x] (2026-02-10 15:00Z) Reviewed RCA engine, workflow, writeback, and proof benchmark behavior.
- [x] (2026-02-10 15:00Z) Drafted initial Phase 8C plan.
- [x] (2026-02-10 15:19Z) Revised Phase 8C with explicit threshold rationale and calibration criteria.
- [ ] Add LLM prompt+schema path for RCA label selection.
- [ ] Keep deterministic narrowing as pre-LLM stage.
- [ ] Update write-back metadata to distinguish LLM vs deterministic fallback.
- [ ] Add RCA-specific proof and regression tests.

## Surprises & Discoveries

- Observation: `_detect_label()` and `_confidence_from_evidence()` are deterministic and currently own RCA judgment.
  Evidence: `investigator/rca/engine.py` label and confidence helpers.
- Observation: Current root RCA write-back always records `annotator_kind: "LLM"` even without model invocation.
  Evidence: `_build_primary_evaluation` in `investigator/rca/writeback.py`.
- Observation: RCA threshold `+0.15` is already encoded in benchmark defaults and should be treated as provisional policy tied to this dataset.
  Evidence: `DEFAULT_DELTA_THRESHOLDS["rca"]` in `investigator/proof/benchmark.py`.

## Decision Log

- Decision: Preserve candidate generation and recursive branch collection exactly as deterministic pre-LLM narrowing.
  Rationale: This keeps evidence collection reproducible and controls cost.
  Date/Author: 2026-02-10 / Codex
- Decision: Replace only the final label/confidence/summary decision with structured LLM output.
  Rationale: Smallest migration that yields real model behavior without rewriting proven narrowing logic.
  Date/Author: 2026-02-10 / Codex
- Decision: Add explicit fallback mode that marks root annotation as `CODE` when LLM path is disabled or fails.
  Rationale: Prevents mislabeled provenance in Phoenix annotations.
  Date/Author: 2026-02-10 / Codex
- Decision: Keep `+0.15` RCA gate for this phase but require threshold calibration note if dataset hash changes.
  Rationale: Current threshold is policy from existing benchmark config, not universal truth.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

Phase not implemented yet. Success is RCA output generation that can run in LLM mode and deterministic fallback mode, with provenance reflected correctly in write-back and run artifacts.

## References

The following files were reviewed fully before drafting and revising this plan; listed behaviors will be reused:

- `specs/rlm_engines.md`
  Behavior reused: RCA evidence policy and acceptance criteria.
- `specs/formal_contracts.md`
  Behavior reused: `RCAReport` and annotation payload constraints.
- `API.md`
  Behavior reused: canonical evidence pointer requirements.
- `investigator/rca/engine.py`
  Behavior reused: hot-span sorting, branch inspection, evidence extraction.
- `investigator/rca/workflow.py`
  Behavior reused: run lifecycle and write-back update into run records.
- `investigator/rca/writeback.py`
  Behavior reused: annotation names and evidence row shaping.
- `investigator/runtime/runner.py`
  Behavior reused: shared validation and run artifact persistence.
- `investigator/proof/benchmark.py`
  Behavior reused: RCA baseline-vs-current accuracy computation and threshold definitions.

## Context and Orientation

RCA currently does two separable jobs:

1. deterministic evidence narrowing,
2. deterministic label judgment.

Phase 8C changes only the second job.

A "structured RCA judgment" in this phase means the model returns a JSON object that maps to RCA contract fields (`primary_label`, `summary`, `confidence`, optional remediation notes), and the engine reuses existing evidence refs rather than allowing ungrounded model citations.

The `+0.15` gate means the LLM path should improve by about five correct labels on a 30-trace dataset. That target is provisional and must be revisited only if dataset hash or label distribution changes.

## Plan of Work

Add an RCA-specific prompt builder in `investigator/rca/engine.py` (or a nearby helper module) that takes selected hot candidates and compact evidence summaries. Send that payload to the shared runtime LLM loop from Phase 8A using schema-constrained output.

Parse the model output into a strict RCA judgment object. Validate that:

- `primary_label` is in taxonomy,
- `confidence` is numeric in [0,1],
- summary is non-empty.

Build final `RCAReport` using existing evidence refs and branch-derived gaps. If LLM output is invalid, follow runtime failure taxonomy. If fallback mode is enabled, run deterministic label logic and mark provenance as deterministic.

Update `investigator/rca/writeback.py` so root annotation `annotator_kind` is driven by run provenance (`LLM` for model path, `CODE` for deterministic fallback), while `rca.evidence` remains `CODE` unless later synthesis explicitly changes.

Add a calibration check in proof diagnostics:

- compare RCA delta against threshold,
- record dataset hash and per-label confusion,
- if threshold miss occurs, include whether miss is concentrated in one label family.

Update tests to cover:

- valid LLM output path,
- invalid LLM output failure path,
- fallback deterministic path,
- write-back annotator kind correctness,
- threshold-calibration diagnostics fields present.

## Concrete Steps

All commands run from repository root.

1. Add failing RCA migration and diagnostics tests.
   - `uv run pytest tests/unit -q -k "rca and phase8"`
2. Implement RCA LLM judgment path and fallback behavior.
3. Update RCA write-back provenance handling.
4. Add/adjust proof diagnostics for RCA calibration context.
5. Re-run RCA tests.
   - `uv run pytest tests/unit -q -k "rca and phase8"`
6. Run proof benchmark tests.
   - `uv run pytest tests/unit/test_proof_benchmark_phase7.py -q`

## Validation and Acceptance

Acceptance requires:

- RCA run in LLM mode produces schema-valid report with evidence refs and non-zero token usage.
- Deterministic fallback remains available and clearly marked as `CODE` provenance.
- Proof benchmark RCA delta remains >= +0.15 on unchanged dataset hash.
- If dataset hash changes, threshold rationale is re-recorded in Decision Log before enforcing gate.
- No regression in run artifact persistence or write-back metadata.

## Idempotence and Recovery

Migration must remain toggleable per run so regressions can be isolated without blocking other capabilities. If LLM outputs invalid schema, engine should fail cleanly with persisted run record rather than silently degrading evidence quality.

## Artifacts and Notes

Expected files to update:

- `investigator/rca/engine.py`
- `investigator/rca/writeback.py`
- `investigator/rca/workflow.py`
- `investigator/proof/benchmark.py`
- `tests/unit/test_trace_rca_phase8_llm.py`
- `tests/unit/test_proof_benchmark_phase7.py`

## Interfaces and Dependencies

Required interfaces after this phase:

- RCA engine hook into shared runtime LLM loop.
- RCA write-back provenance field derived from runtime mode.
- Optional RCA request/runtime flag to select LLM mode vs deterministic fallback.

Dependencies:

- Phase 8A shared LLM loop must be complete.
- Phase 8B recursion shell can remain optional for first RCA LLM cut as long as evidence narrowing stays deterministic.

Revision Note (2026-02-10): Initial Phase 8C plan created to migrate RCA label judgment to real LLM output while preserving deterministic evidence narrowing and proof comparability.
Revision Note (2026-02-10): Revised after Nainy review to clarify threshold rationale, provisional calibration rules, and required RCA diagnostics for gate interpretation.
