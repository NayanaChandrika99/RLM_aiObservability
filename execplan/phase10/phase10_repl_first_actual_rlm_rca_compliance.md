# Phase 10 Plan: REPL-First RLM for RCA and Policy Compliance

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

Phase 9 and 9.1 delivered planner-driven recursive actions, pooled budgets, and planner-driven child subcalls. However, they are still action-schema loops, not an actual REPL-centric Recursive Language Model loop where the model writes code, inspects environment state iteratively, and submits final outputs.

Phase 10 makes RCA and Policy Compliance truly REPL-first RLM engines for long-context observability tasks:

- `TraceRCAEngine`: model inspects narrowed span/tool/retrieval context through iterative code execution and recursive sub-LLM calls as a primary semantic operator, then emits evidence-linked RCA output.
- `PolicyComplianceEngine`: model runs per-control recursive code loops to gather required evidence incrementally and produce auditable control findings.

Incident dossier is explicitly out of scope for this phase and remains on the Phase 9.1 architecture.

Visible outcome: Nainy can run RCA and compliance with `use_repl_runtime=True` and inspect run records that show real REPL trajectory (`reasoning`, `code`, `output`) plus bounded recursion/tool/sub-LLM usage.

## Progress

- [x] (2026-02-10 21:40Z) Drafted Phase 10 ExecPlan with REPL-first RLM scope limited to RCA + Policy Compliance.
- [x] (2026-02-10 22:15Z) Added RED runtime tests for REPL loop semantics (`SUBMIT`, iteration bounds, usage accounting, sandbox rejection, malformed output fallback).
- [x] (2026-02-10 22:16Z) Implemented shared REPL runtime module and prompt/schema contracts.
- [x] (2026-02-10 22:16Z) Wired RCA to REPL runtime behind `use_repl_runtime` and validated unit/regression slices.
- [x] (2026-02-10 22:16Z) Wired Policy Compliance to REPL runtime behind `use_repl_runtime` and validated unit/regression slices.
- [x] (2026-02-10 22:44Z) Ran high-budget live RCA + compliance reruns; both reached `succeeded` with non-zero `llm_subcalls`.
- [ ] (2026-02-10) Run proof benchmark for RCA + compliance comparison vs baseline and document deltas.
- [x] (2026-02-10 22:45Z) Updated Outcomes with current Phase 10 implementation verdict and remaining follow-up.
- [x] (2026-02-10 22:45Z) Full regression passed: `127 passed, 2 skipped` (live writeback integration remains opt-in).

## Surprises & Discoveries

- Observation: Current recursive runtime is action-typed (`tool_call`, `delegate_subcall`, `synthesize`, `finalize`) and does not execute model-generated code.
  Evidence: `investigator/runtime/recursive_loop.py`.
- Observation: Budget pooling and child planner delegation are now stable and should be retained as control rails for REPL execution as well.
  Evidence: `investigator/runtime/budget_pool.py`, `execplan/phase9/phase9_1_budget_pool_and_child_planners.md`.
- Observation: DSPy RLM uses iterative code-generation + sandbox execution + `SUBMIT(...)` + capped `llm_query` calls, which maps directly to the missing gap.
  Evidence: `/tmp/dspy/docs/docs/api/modules/RLM.md`, `/tmp/dspy/dspy/predict/rlm.py`.
- Observation: For compliance, deterministic `insufficient_evidence` precedence is a hard invariant and must remain stronger than model preference.
  Evidence: `investigator/compliance/engine.py`, `specs/formal_contracts.md`.
- Observation: `gpt-5-mini` text subqueries can return `status=incomplete` with reasoning-only output unless low reasoning effort is requested.
  Evidence: live sanity checks against `OpenAI.responses.create(...)` and failing REPL `llm_query` outputs before patch.
- Observation: Missing `isinstance`/`repr`/`hasattr` in REPL safe builtins created avoidable code-step failures and slowed convergence.
  Evidence: live run trajectory errors in `phase10-live-rca-repl-highbudget-20260210T222403Z`.
- Observation: Explicit submit-deadline context and mode-specific `SUBMIT` field guidance materially improved convergence.
  Evidence: RCA/compliance high-budget reruns reached `succeeded` after prompt/runtime-context update.

## Decision Log

- Decision: Scope Phase 10 to RCA + Policy Compliance only; keep Incident on existing Phase 9.1 runtime.
  Rationale: RCA/compliance are the immediate priority, and incident introduces extra multi-trace complexity that would dilute Phase 10 validation.
  Date/Author: 2026-02-10 / Codex
- Decision: Keep deterministic narrowing/scoping as pre-REPL seed context for both engines.
  Rationale: Reduces search space and cost; aligns with existing reproducibility guarantees.
  Date/Author: 2026-02-10 / Codex
- Decision: Preserve pooled budget and runtime-state contracts; extend them for REPL trajectory rather than replacing them.
  Rationale: Avoids contract churn in runner/writeback while adding true RLM behavior.
  Date/Author: 2026-02-10 / Codex
- Decision: Keep fallback path (`use_llm_judgment=True`, `use_repl_runtime=False`) until proof gate passes with REPL mode.
  Rationale: Safe rollback and controlled migration.
  Date/Author: 2026-02-10 / Codex
- Decision: Treat sub-LLM recursion as required behavior in REPL mode for non-trivial RCA/compliance runs.
  Rationale: DSPy/paper RLM framing relies on recursive sub-queries for semantic decomposition; REPL-only regex/heuristics path is an ablation baseline, not target behavior.
  Date/Author: 2026-02-10 / Codex
- Decision: Set `reasoning.effort=minimal` for text subqueries on GPT-5 models in runtime client.
  Rationale: Prevents reasoning-token exhaustion and restores deterministic text output for `llm_query`.
  Date/Author: 2026-02-10 / Codex
- Decision: Add explicit remaining-budget context and submit-deadline instructions in REPL planner prompt.
  Rationale: Forces bounded convergence instead of iterative drift into budget termination.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

Phase 10 implementation outcome (current):

1. REPL-first runtime is active for RCA/compliance behind `use_repl_runtime=True`, with iterative code trajectory persisted in run records.
2. Runtime usage now includes `llm_subcalls` and `repl_trajectory`; sandbox and schema contracts remain enforced.
3. `llm_query` now uses text generation with GPT-5 low-reasoning configuration, and fallback structured mode remains available for test doubles.
4. High-budget live convergence evidence:
   - RCA succeeded: `phase10-live-rca-repl-highbudget-rerun3-20260210T224201Z` (`iterations=4`, `llm_subcalls=2`, `cost_usd=0.0144145`).
   - Compliance succeeded: `phase10-live-compliance-repl-highbudget-rerun-20260210T224354Z` (`iterations=9`, `llm_subcalls=2`, `cost_usd=0.03354175`).
5. Regression status: `127 passed, 2 skipped`.

Remaining follow-up:

- Run proof benchmark comparison explicitly configured for REPL mode (RCA/compliance) and record baseline-vs-REPL deltas in this document.

## Context and Orientation

Current state after Phase 9.1:

- Recursive control loop exists with planner-generated typed actions.
- Child planner delegation and shared budget pool are implemented.
- RCA/compliance can run recursive mode but still rely on action-schema planning instead of code REPL reasoning.

Target state for Phase 10:

- Shared REPL runtime where the model returns:
  - `reasoning` (text),
  - `code` (Python snippet),
  - explicit finalization via `SUBMIT({...})`.
- Runtime executes code in sandboxed interpreter with read-only tools and bounded sub-LLM calls.
- Engine outputs remain on existing contracts (`RCAReport`, `ComplianceReport`) and runner/writeback behavior remains unchanged.

## Plan of Work

### Phase 10A: Shared REPL Runtime (RED -> GREEN)

Add a new runtime path for REPL-centric RLM execution:

- `investigator/runtime/repl_loop.py`:
  - iterative loop over `max_iterations`,
  - captures trajectory entries (`reasoning`, `code`, `output`),
  - supports `SUBMIT(output_dict)` early termination,
  - applies budget checks before and after each model/interpreter step,
  - maps limit hits to `terminated_budget` with best-effort output.
- `investigator/runtime/repl_interpreter.py`:
  - sandboxed code execution interface,
  - explicit allowlist of built-ins and runtime helper functions,
  - no filesystem/network/subprocess access.
- Runtime helper functions exposed to code:
  - `call_tool(tool_name: str, **kwargs)` (backed by `ToolRegistry`, read-only tools only),
  - `llm_query(prompt: str)` (required) and `llm_query_batched(prompts: list[str])` (optional batching helper) with per-run call caps,
  - `SUBMIT(**fields)` finalizer.
- Prompt/schema assets:
  - `investigator/prompts/runtime/repl_runtime_step_v1.md`,
  - `investigator/prompts/runtime/repl_runtime_step_v1.schema.json`,
  - register prompt id/hash in `prompt_registry`.

Acceptance:

- RED tests fail first for:
  - no-`SUBMIT` max-iteration fallback,
  - invalid `SUBMIT` fields,
  - sandbox violations (`import os`, filesystem open, subprocess),
  - LLM subcall limit enforcement,
  - non-trivial objective policy violation when no `llm_query`/`llm_query_batched` call is made,
  - run-record usage/state propagation.
- GREEN tests pass with non-zero tokens/cost and non-empty REPL trajectory.

### Phase 10B: RCA REPL Wiring

Wire `TraceRCAEngine` to REPL runtime behind `use_repl_runtime` (requires `use_llm_judgment=True`):

- keep deterministic hot-span + branch narrowing as seed context,
- expose seed artifacts as structured REPL variables (`candidate_summaries`, `evidence_seed`, `allowed_labels`),
- allow focused delegated subcalls for label-specific investigation,
- finalize with schema-valid RCA fields and evidence pointers.

Acceptance:

- RCA REPL runs emit trajectory and subcall metadata.
- RCA REPL succeeded runs on non-trivial traces show `llm_subcalls > 0` in runtime usage/signals.
- At least one high-budget live run ends `succeeded` without depth-limit partial.
- RCA proof metric is non-regression vs baseline on seeded failures.

### Phase 10C: Compliance REPL Wiring

Wire `PolicyComplianceEngine` to REPL runtime behind `use_repl_runtime`:

- maintain deterministic control ordering and applicability scoping,
- run per-control REPL loops with shared `RuntimeBudgetPool`,
- preserve hard rule: missing required evidence forces `insufficient_evidence`,
- support delegated subcalls for targeted missing-evidence resolution.

Acceptance:

- Compliance REPL runs emit trajectory and subcall metadata.
- Compliance REPL succeeded runs on non-trivial controls show `llm_subcalls > 0` in runtime usage/signals.
- At least one high-budget live run ends `succeeded`.
- Compliance proof metric is non-regression vs baseline and no control fail/pass without evidence refs.

### Phase 10D: Benchmark and Proof Gates (RCA + Compliance)

Run benchmark/proof validation focused on RCA and compliance:

- baseline mode: current non-REPL path,
- REPL mode: new `use_repl_runtime=True` path.

Track and compare:

- quality: RCA label match / compliance agreement metrics already used in proof benchmark,
- runtime: tokens, cost, tool calls, iterations, subcalls,
- convergence: `succeeded` vs `partial`/`failed` with root error code.

Acceptance:

- RCA and compliance both pass proof gate thresholds.
- REPL path is non-regression on quality and within declared cost caps.

### Phase 10E: Rollout Guardrails

- Keep REPL runtime off by default until proof gate passes.
- Expose per-engine toggles:
  - `use_llm_judgment`,
  - `use_recursive_runtime`,
  - `use_repl_runtime`.
- Document precedence:
  - `use_repl_runtime=True` selects REPL path,
  - else existing recursive typed-action path,
  - else single-turn path.

## Concrete Steps

All commands run from repository root (`/Users/nainy/Documents/Personal/rlm_observability`).

1. Add RED REPL runtime tests.
   `uv run pytest tests/unit -q -k "runtime and repl and phase10"`
2. Implement shared REPL runtime modules + prompt/schema registration.
3. Run focused runtime regression.
   `uv run pytest tests/unit -q -k "runtime and (repl or recursive_loop or sandbox)"`
4. Add RED RCA REPL wiring tests.
   `uv run pytest tests/unit -q -k "trace_rca and phase10 and repl"`
5. Implement RCA REPL wiring and run focused RCA tests.
   `uv run pytest tests/unit -q -k "trace_rca and (phase9 or phase10)"`
6. Add RED compliance REPL wiring tests.
   `uv run pytest tests/unit -q -k "compliance and phase10 and repl"`
7. Implement compliance REPL wiring and run focused compliance tests.
   `uv run pytest tests/unit -q -k "compliance and (phase9 or phase10)"`
8. Run full regression.
   `uv run pytest tests/ -q -rs`
9. Run one high-budget live RCA run and one high-budget live compliance run with REPL mode.
10. Run proof benchmark gate for RCA + compliance comparison.
    `PHOENIX_WORKING_DIR=.phoenix_data uv run python -m investigator.proof.run_phase7_proof`

## Validation and Acceptance

Phase 10 is accepted when all are true:

- REPL runtime tests pass and show real code-execution trajectory.
- RCA + compliance run records include:
  - non-empty `state_trajectory`,
  - non-empty `repl_trajectory` (new),
  - non-zero `llm_subcalls` for non-trivial succeeded runs,
  - non-zero `tokens_in`, `tokens_out`, `cost_usd`,
  - contract-valid output artifacts.
- At least one high-budget live run succeeds for each of RCA and compliance.
- Proof benchmark shows non-regression or improvement for RCA and compliance versus baseline.
- No sandbox escape paths are observed in tests (filesystem/network/subprocess blocked).

## Idempotence and Recovery

- All migration is behind explicit toggles; fallback paths remain available.
- If REPL path regresses:
  - disable `use_repl_runtime` per engine,
  - retain existing recursive typed-action path for continuity.
- Keep failing run artifacts; do not delete partial/failed evidence.

## Artifacts and Notes

Primary artifacts:

- `artifacts/investigator_runs/<run_id>/run_record.json`
- `artifacts/investigator_runs/<run_id>/output.json`
- `artifacts/proof_runs/<proof_run_id>/proof_report.json`

Expected new runtime artifact section:

- `runtime_ref.repl_trajectory` (or artifact path reference) with bounded per-step logs.

## Interfaces and Dependencies

Likely touched modules:

- `investigator/runtime/recursive_loop.py`
- `investigator/runtime/budget_pool.py`
- `investigator/runtime/runner.py`
- `investigator/runtime/llm_client.py`
- `investigator/runtime/prompt_registry.py`
- `investigator/runtime/sandbox.py`
- `investigator/runtime/tool_registry.py`
- `investigator/rca/engine.py`
- `investigator/compliance/engine.py`

Likely new modules:

- `investigator/runtime/repl_loop.py`
- `investigator/runtime/repl_interpreter.py`

## References

Reviewed fully before drafting this plan, with behaviors to reuse:

- `PLANS.md`
  Behavior reused: living ExecPlan structure and required sections.
- `execplan/phase9/phase9_recursive_tool_driven_rlm.md`
  Behavior reused: recursive migration sequencing and proof-gate discipline.
- `execplan/phase9/phase9_1_budget_pool_and_child_planners.md`
  Behavior reused: shared budget pool and planner-driven child delegation constraints.
- `specs/rlm_runtime_contract.md`
  Behavior reused: runtime states, sandbox contract, budget/error taxonomy, run artifact requirements.
- `specs/rlm_engines.md`
  Behavior reused: RCA and compliance recursion/evidence expectations and writeback names.
- `specs/formal_contracts.md`
  Behavior reused: RCA/Compliance/RunRecord schema and evidence invariants.
- `investigator/runtime/recursive_loop.py`
  Behavior reused: state-machine transitions, budget enforcement order, subcall metadata handling.
- `investigator/runtime/budget_pool.py`
  Behavior reused: fair-share sibling allocation and global limit checks.
- `investigator/runtime/recursive_planner.py`
  Behavior reused: structured generation envelope and usage accounting pattern.
- `investigator/runtime/sandbox.py`
  Behavior reused: action/tool restriction model and violation handling semantics.
- `investigator/runtime/tool_registry.py`
  Behavior reused: read-only tool invocation boundary and deterministic response hashing.
- `investigator/runtime/runner.py`
  Behavior reused: run status mapping, run_record persistence, and writeback integration.
- `investigator/rca/engine.py`
  Behavior reused: deterministic hot-span narrowing, branch inspection seed logic, RCA signal propagation.
- `investigator/compliance/engine.py`
  Behavior reused: control scoping/order, required evidence semantics, insufficient-evidence precedence.
- `tests/unit/test_trace_rca_engine_phase9_recursive.py`
  Behavior reused: recursive runtime trajectory/subcall assertions and budget-limit partial mapping checks.
- `tests/unit/test_compliance_phase9_recursive.py`
  Behavior reused: per-control recursive assertions and budget-limit partial mapping checks.
- `tests/unit/test_runtime_recursive_planner_scenarios_phase9.py`
  Behavior reused: representative runtime scenario harness pattern.
- `/tmp/dspy/docs/docs/api/modules/RLM.md`
  Behavior reused: iterative REPL loop (`reasoning` + `code` + output), `llm_query`, `SUBMIT`, and iteration/subcall cap model.
- `/tmp/dspy/dspy/predict/rlm.py`
  Behavior reused: action-generation loop structure, fallback on max iterations, and trajectory capture design.
- `/tmp/dspy/dspy/primitives/repl_types.py`
  Behavior reused: structured REPL history/variable metadata format.
- `/tmp/dspy/dspy/primitives/python_interpreter.py`
  Behavior reused: sandboxed interpreter lifecycle and host-side tool registration pattern.

Revision Note (2026-02-10): Initial Phase 10 plan drafted to migrate RCA and Policy Compliance from typed-action recursion to REPL-first actual RLM execution with preserved runtime contracts.
Revision Note (2026-02-10): Updated to treat sub-LLM recursion as required in REPL mode for non-trivial RCA/compliance runs, matching DSPy/paper RLM intent.
