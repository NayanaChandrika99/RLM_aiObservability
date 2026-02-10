# Phase 8D: Compliance Engine LLM Migration and Control Enrichment

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

This phase makes compliance the first capability that must clear a hard quality gate: `delta.accuracy >= +0.05` in the frozen proof benchmark. To achieve that, the work has two parts: migrate control verdicting to real LLM evidence judgment and enrich control definitions so the evaluator can detect policy-relevant behavior that baseline error-count logic misses.

After completion, Nainy can rerun proof and observe measurable compliance lift with auditable per-control evidence links.

## Progress

- [x] (2026-02-10 15:00Z) Reviewed compliance engine, controls library, write-back, and benchmark expectations.
- [x] (2026-02-10 15:00Z) Drafted initial Phase 8D plan.
- [x] (2026-02-10 15:22Z) Revised Phase 8D with dataset-grounded control enrichment scope and explicit manifest-driven coverage checks.
- [ ] Add LLM per-control verdict path using shared runtime loop.
- [ ] Enrich controls library using observable seeded dataset signals.
- [ ] Align benchmark compliance expectations with control semantics and evidence rules.
- [ ] Pass compliance proof gate (`delta.accuracy >= +0.05`) on frozen dataset.

## Surprises & Discoveries

- Observation: Current control set is very small and mostly mirrors error-span detection, which limits measurable lift over baseline.
  Evidence: `controls/library/controls_v1.json` has only two controls, one of which is hard-failure pattern matching.
- Observation: Baseline compliance in benchmark uses only "has any ERROR span" logic.
  Evidence: `_baseline_compliance_verdict` in `investigator/proof/benchmark.py`.
- Observation: Seeded dataset has only failure profiles with skewed label distribution, so enrichment must target present signals.
  Evidence: `datasets/seeded_failures/manifest.json` label counts: tool 9, retrieval 8, data_schema 7, upstream 4, instruction 2.
- Observation: Seeded trace generator emits specific phase attributes that can support deterministic evidence extraction.
  Evidence: `apps/demo_agent/phase1_seeded_failures.py` sets `phase1.step`, `phase1.retrieval.relevance`, `phase1.output.format`, and `http.status_code`.

## Decision Log

- Decision: Keep deterministic evidence collection in compliance engine and migrate only control judgment to LLM first.
  Rationale: Evidence collection is already contract-aligned and reusable; judgment is the missing model-dependent step.
  Date/Author: 2026-02-10 / Codex
- Decision: Introduce richer controls as additive entries, not by deleting existing controls.
  Rationale: Additive controls preserve backward compatibility and improve traceability.
  Date/Author: 2026-02-10 / Codex
- Decision: Ground enrichment only in signals present in seeded traces for Phase 8 proof gating.
  Rationale: Controls that cannot be evidenced in dataset cannot move benchmark metrics.
  Date/Author: 2026-02-10 / Codex
- Decision: Tie acceptance to benchmark gate in the same phase, not as a later cleanup.
  Rationale: User goal is explicit measurable lift, not only architectural migration.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

Phase not implemented yet. Success is compliance LLM path plus dataset-grounded control enrichment producing >= +0.05 lift while preserving per-control evidence contracts and write-back traceability.

## References

The following files were reviewed fully before drafting and revising this plan; listed behaviors will be reused:

- `specs/rlm_engines.md`
  Behavior reused: per-control evidence policy and governance write-back expectations.
- `specs/formal_contracts.md`
  Behavior reused: `ComplianceReport` and finding-level required fields.
- `API.md`
  Behavior reused: `required_evidence` and canonical evidence refs.
- `investigator/compliance/engine.py`
  Behavior reused: control scoping, evidence cataloging, and finding assembly flow.
- `investigator/compliance/writeback.py`
  Behavior reused: root + evidence annotation naming and metadata persistence.
- `investigator/compliance/workflow.py`
  Behavior reused: workflow-level run record write-back update path.
- `investigator/proof/benchmark.py`
  Behavior reused: compliance baseline/current metric computation and gate evaluation.
- `controls/library/controls_v1.json`
  Behavior reused: current control schema shape and versioning conventions.
- `datasets/seeded_failures/manifest.json`
  Behavior reused: label distribution and run set for enrichment targeting.
- `apps/demo_agent/phase1_seeded_failures.py`
  Behavior reused: available trace attributes for control evidence extraction.

## Context and Orientation

Compliance currently blends deterministic evidence gathering and deterministic verdicting in one function. For Phase 8D, verdicting moves to shared LLM runtime while evidence gathering remains deterministic. This split keeps auditability and improves model effectiveness because prompts receive pre-filtered evidence bundles per control.

"Control enrichment" in this phase means adding controls that can be evaluated on the current seeded dataset, not speculative future controls.

Dataset-grounded enrichment scope for this phase:

- upstream behavior (`http.status_code=503`, status message containing upstream failure),
- schema mismatch behavior (`tool.parse`, status text `schema mismatch`),
- instruction format drift (`phase1.output.format=unexpected`, status text `format drift`),
- retrieval quality signal (`phase1.retrieval.relevance` low values).

## Plan of Work

Add a compliance judgment helper that packages per-control evidence into a compact structured prompt for the shared LLM loop. The model must return one of `pass|fail|not_applicable|insufficient_evidence`, confidence, and short remediation rationale.

Retain deterministic guardrails before model output is accepted:

- if required evidence is missing, force `insufficient_evidence` regardless of model output,
- if model output enum is invalid, fail with runtime validation error,
- preserve deterministic fallback mode for debugging.

Enrich controls library with additive controls that map to seeded signal families listed above. Each new control must declare:

- `controls_version`,
- `required_evidence` keys that are resolvable from current inspection data,
- deterministic applicability rules that can trigger on seeded spans.

Add a manifest-coverage check in proof tests:

- verify each seeded `fault_profile` maps to at least one applicable control,
- report uncovered profiles before running compliance gate.

Update proof benchmark compliance expectation logic only when control semantics demand it. Any expectation change must be documented with control-behavior rationale and kept deterministic.

Add tests for:

- per-control LLM verdict parsing,
- forced insufficient-evidence precedence,
- dataset-profile-to-control coverage,
- compliance gate pass on representative fixture data.

## Concrete Steps

All commands run from repository root.

1. Add failing compliance LLM, coverage, and control-enrichment tests.
   - `uv run pytest tests/unit -q -k "compliance and phase8"`
2. Implement compliance LLM verdict path.
3. Add enriched controls under `controls/library/` and update loading/version selection.
4. Implement manifest-profile coverage diagnostics in proof benchmark tests.
5. Update benchmark expectations only if contract-alignment requires it.
6. Re-run compliance and benchmark tests.
   - `uv run pytest tests/unit -q -k "compliance or proof_benchmark"`
7. Run full proof command and verify compliance gate.
   - `PHOENIX_WORKING_DIR=.phoenix_data uv run python -m investigator.proof.run_phase7_proof`

## Validation and Acceptance

Acceptance requires all of the following:

- Compliance engine in LLM mode returns valid `ComplianceReport` with evidence refs for non-`not_applicable` findings.
- Finding provenance and root annotation provenance reflect true source (`LLM` vs `CODE`).
- Coverage diagnostic shows every seeded `fault_profile` is evaluable by at least one control.
- Proof report compliance gate shows `actual >= threshold` for threshold `+0.05`.
- No regressions in RCA or incident benchmark sections while improving compliance.

## Idempotence and Recovery

Control enrichment must be versioned and additive to avoid breaking existing runs. If gate fails after a control change, revert only the specific control additions or benchmark expectation updates and rerun proof with identical dataset hash for comparable diagnostics.

## Artifacts and Notes

Expected files to update:

- `investigator/compliance/engine.py`
- `investigator/compliance/writeback.py`
- `investigator/compliance/workflow.py`
- `controls/library/controls_v1.json` or new versioned control file under `controls/library/`
- `investigator/proof/benchmark.py`
- `tests/unit/test_compliance_phase8_llm.py`
- `tests/unit/test_proof_benchmark_phase7.py`

## Interfaces and Dependencies

Required interfaces after this phase:

- Compliance engine hook into shared runtime LLM loop for per-control judgments.
- Deterministic policy for precedence of missing-evidence outcomes.
- Stable controls loader behavior by `controls_version`.
- Coverage-diagnostic helper mapping seeded profiles to applicable controls.

Dependencies:

- Phase 8A shared LLM loop must be complete.
- Phase 8B recursion shell can be used for multi-control subcalls when needed, but initial migration may use bounded single-level calls.

Revision Note (2026-02-10): Initial Phase 8D plan created to deliver compliance LLM migration and control enrichment with explicit +0.05 proof gate acceptance.
Revision Note (2026-02-10): Revised after Nainy review to ground control enrichment in actual seeded dataset signals and add manifest-profile coverage diagnostics.
