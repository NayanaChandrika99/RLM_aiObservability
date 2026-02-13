# Phase 9.1 Plan: Shared Budget Pool and Planner-Driven Child Subcalls

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

Phase 9 wired recursive planner execution into all three engines. Phase 9.1 closes two critical runtime gaps: child subcalls must be able to run their own planner loops (`use_planner=true`), and multi-scope engines must share one global budget pool so one branch cannot consume unbounded resources while siblings starve.

After this work, Nainy can verify in live run records that RCA, compliance, and incident each emit real planner-driven child subcalls and that pooled budgets enforce bounded execution across sibling loops.

## Progress

- [x] (2026-02-10 20:16Z) Added missing Phase 9.1 ExecPlan document under `execplan/phase9/`.
- [x] (2026-02-10 20:25Z) Simplified planner-usage parsing in `investigator/runtime/recursive_loop.py` to single-parse reuse for local usage and budget-pool consumption.
- [x] (2026-02-10 20:26Z) Ran focused runtime regressions after parser simplification: `19 passed`.
- [x] (2026-02-10 20:29Z) Ran one higher-budget live run per engine with explicit cost caps and collected run records:
  - `phase91-live-rca-highbudget-20260210T202612Z`
  - `phase91-live-compliance-highbudget-20260210T202612Z`
  - `phase91-live-incident-highbudget-20260210T202612Z`
- [x] (2026-02-10 20:29Z) Recorded convergence outcome: all three runs remained `partial` with `RECURSION_LIMIT_REACHED`, but with materially higher depth/usage than smoke runs.
- [x] (2026-02-10 20:31Z) Ran full regression after updates: `119 passed`, `2 skipped`.
- [x] (2026-02-10 21:03Z) Added runtime depth-stop rule in `recursive_loop.py` + planner prompt context updates (`depth_stop_rule`) and validated focused recursive/runtime slices (`27 passed`).
- [x] (2026-02-10 21:08Z) Re-ran one higher-budget live run per engine after depth-stop updates:
  - `phase91-live-rca-highbudget-final-20260210T210325Z` (`succeeded`)
  - `phase91-live-compliance-highbudget-final-20260210T210520Z` (`succeeded`)
  - `phase91-live-incident-highbudget-final-20260210T210710Z` (`partial`, shared pool wall-time limit)
- [x] (2026-02-10 21:08Z) Success target met: at least one high-budget engine run now finishes `succeeded` (RCA and compliance both succeeded).
- [x] (2026-02-10 21:13Z) Executed a fresh post-depth-stop high-budget rerun set with sourced `.env` credentials:
  - `phase91-live-rca-highbudget-rerun5-20260210T211322Z` (`partial`, wall-time limit)
  - `phase91-live-compliance-highbudget-rerun5-20260210T211322Z` (`succeeded`)
  - `phase91-live-incident-highbudget-rerun3-20260210T211322Z` (`partial`, wall-time limit)
- [x] (2026-02-10 21:13Z) Reconfirmed success target on fresh reruns: at least one engine `succeeded` (compliance succeeded).

## Surprises & Discoveries

- Observation: A strict OpenAI schema contract for planner actions is stricter than local validator assumptions, especially around object property constraints.
  Evidence: live planner call failures until schema/runtime handling was adjusted.
- Observation: Incident per-trace recursion did not initially receive `delegation_policy`, so planner behavior depended only on generic prompt guidance.
  Evidence: per-trace planner contexts lacked policy fields until engine seed wiring was updated.
- Observation: High-budget runs still hit recursive depth ceilings through repeated planner-driven delegation chains.
  Evidence: high-budget run records for RCA/compliance/incident all include `error.code=RECURSION_LIMIT_REACHED` with depth-limit messages while showing increased tokens/cost and subcall counts.
- Observation: After adding depth-stop and near-wall-time finalize guards, RCA and compliance converge to `succeeded` under high-budget runs; incident still hits shared pool wall-time in multi-scope execution.
  Evidence: final run records `phase91-live-rca-highbudget-final-20260210T210325Z`, `phase91-live-compliance-highbudget-final-20260210T210520Z`, and `phase91-live-incident-highbudget-final-20260210T210710Z`.

## Decision Log

- Decision: Keep Phase 9.1 as an incremental extension of Phase 9 architecture instead of introducing a separate runtime surface.
  Rationale: budget pooling and planner-driven children are contract-level improvements on the same `RecursiveLoop` interface.
  Date/Author: 2026-02-10 / Codex
- Decision: Use targeted live runs with explicit run IDs and run-record inspection as proof, not only unit tests.
  Rationale: Phase 9.1 goals are runtime-behavioral and must be validated against real model trajectories and usage signals.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

Phase 9.1 implementation is complete for the requested scope:

- The missing execution plan document now exists at `execplan/phase9/phase9_1_budget_pool_and_child_planners.md`.
- Planner-usage parsing in `investigator/runtime/recursive_loop.py` was simplified without changing behavior.
- Higher-budget live runs were executed for all three engines and verified from run records.

Convergence result from higher-budget runs:

- `phase91-live-rca-highbudget-20260210T202612Z`: `partial`, `RECURSION_LIMIT_REACHED`, planner-driven subcalls present (`subcalls=4`), `tokens_in=17382`, `cost_usd=0.0122095`.
- `phase91-live-compliance-highbudget-20260210T202612Z`: `partial`, `RECURSION_LIMIT_REACHED`, planner-driven subcalls present (`subcalls=4`), `tokens_in=12879`, `cost_usd=0.00852375`.
- `phase91-live-incident-highbudget-20260210T202612Z`: `partial`, `RECURSION_LIMIT_REACHED`, planner-driven subcalls present (`subcalls=16`), `tokens_in=67944`, `cost_usd=0.04249`.

Interpretation: budgets were materially higher than smoke runs and produced deeper trajectories plus higher evidence-gathering cost, but current planner behavior still tends to recurse until depth limits rather than converging earlier.

Final rerun result after depth-stop updates:

- `phase91-live-rca-highbudget-final-20260210T210325Z`: `succeeded`, planner-driven subcalls present (`subcalls=4`), `tokens_in=21478`, `cost_usd=0.0155275`.
- `phase91-live-compliance-highbudget-final-20260210T210520Z`: `succeeded`, planner-driven subcalls present (`subcalls=4`), `tokens_in=22362`, `cost_usd=0.0174345`.
- `phase91-live-incident-highbudget-final-20260210T210710Z`: `partial`, `RECURSION_LIMIT_REACHED` via shared budget-pool wall-time, planner-driven subcalls present (`subcalls=4`), `tokens_in=30907`, `cost_usd=0.02022275`.
- `phase91-live-rca-highbudget-rerun5-20260210T211322Z`: `partial`, `RECURSION_LIMIT_REACHED` via wall-time, planner-driven subcalls present (`subcalls=4`), `tokens_in=24943`, `cost_usd=0.01824975`.
- `phase91-live-compliance-highbudget-rerun5-20260210T211322Z`: `succeeded`, planner-driven subcalls present (`subcalls=4`), `tokens_in=23207`, `cost_usd=0.01748175`.
- `phase91-live-incident-highbudget-rerun3-20260210T211322Z`: `partial`, `RECURSION_LIMIT_REACHED` via wall-time, planner-driven subcalls present (`subcalls=4`), `tokens_in=27234`, `cost_usd=0.0184185`.

Success-target check: satisfied (`>=1` succeeded; actual `2` succeeded).

## Context and Orientation

The runtime loop lives in `investigator/runtime/recursive_loop.py`. It executes typed actions (`tool_call`, `synthesize`, `delegate_subcall`, `finalize`) and writes state trajectories and subcall metadata consumed by `investigator/runtime/runner.py`.

The shared budget pool implementation is in `investigator/runtime/budget_pool.py`. Compliance and incident engines both create a `RuntimeBudgetPool` and allocate per-scope loop budgets through `allocate_run_budget(...)`. RCA remains single-scope and intentionally does not use pooled sibling allocation.

Planner-driven child subcalls are represented by `delegate_subcall` actions carrying `use_planner=true`. The parent loop must thread its planner callable into the child run and annotate subcall metadata with planner-driven provenance.

## Plan of Work

First, simplify the planner-usage parsing block in `investigator/runtime/recursive_loop.py` so planner usage values (`tokens_in`, `tokens_out`, `cost_usd`) are parsed once, then reused for both local usage and budget-pool consumption. This is a readability and maintainability improvement; behavior must remain unchanged.

Second, execute one higher-budget live run per engine using `ParquetInspectionAPI` and real model calls with `use_llm_judgment=True` and `use_recursive_runtime=True`. The run budget must be materially larger than smoke budgets and include explicit `max_cost_usd` safety caps. Capture run IDs and verify status, error code, usage, and child subcall provenance.

Third, update this plan with concrete outcomes and evidence references from the run artifacts.

## Concrete Steps

From repository root:

1. Run focused runtime tests after parser simplification:
   `uv run pytest tests/unit/test_runtime_recursive_loop_phase8.py tests/unit/test_runtime_recursive_planner_phase9.py tests/unit/test_runtime_recursive_planner_scenarios_phase9.py -q`
2. Run one higher-budget live RCA run and inspect generated run record in `artifacts/investigator_runs/<run_id>/run_record.json`.
3. Run one higher-budget live compliance run (`controls_version=controls-v1`) and inspect generated run record.
4. Run one higher-budget live incident run and inspect generated run record.
5. Run full regression:
   `uv run pytest tests/ -q -rs`

## Validation and Acceptance

This phase is accepted when:

- Runtime parser simplification preserves existing unit behavior (focused runtime slices pass).
- Three higher-budget live runs (one per engine) complete with valid run records.
- For each live run, run record includes non-zero usage (`tokens_in`, `tokens_out`, `cost_usd`) and explicit child-subcall provenance (`subcall_metadata[*].planner_driven`).
- Convergence status is explicitly documented: either `succeeded` or `partial/failed` with precise `error.code` and `error.message` from run record.

## Idempotence and Recovery

All edits are additive and can be reapplied safely. Live run commands create new run directories keyed by run ID and do not overwrite prior artifacts. If a live run fails, keep the failing run artifact and rerun with a new run ID and adjusted budget; do not delete prior evidence.

## Artifacts and Notes

Primary proof artifacts for this phase:

- `artifacts/investigator_runs/<run_id>/run_record.json`
- `artifacts/investigator_runs/<run_id>/output.json`

## Interfaces and Dependencies

- `investigator/runtime/recursive_loop.py`
- `investigator/runtime/budget_pool.py`
- `investigator/rca/engine.py`
- `investigator/compliance/engine.py`
- `investigator/incident/engine.py`
- `investigator/runtime/llm_client.py`
- `investigator/runtime/runner.py`

## References

Reviewed fully and reused behaviors from:

- `PLANS.md`
- `execplan/phase9/phase9_recursive_tool_driven_rlm.md`
- `investigator/runtime/recursive_loop.py`
- `investigator/runtime/budget_pool.py`
- `investigator/runtime/runner.py`
- `investigator/runtime/llm_client.py`
- `investigator/rca/engine.py`
- `investigator/compliance/engine.py`
- `investigator/incident/engine.py`
- `tests/unit/test_runtime_budget_pool_phase91.py`
- `tests/unit/test_runtime_recursive_planner_phase9.py`
- `tests/unit/test_runtime_recursive_planner_scenarios_phase9.py`
- `tests/unit/test_runtime_recursive_loop_phase8.py`

Revision Note (2026-02-10): Created this missing Phase 9.1 execution plan to document runtime budget-pool + planner-driven-child implementation and required verification evidence.
Revision Note (2026-02-10): Updated after implementation to record parser simplification, higher-budget live run evidence, and convergence outcome.
Revision Note (2026-02-10): Updated with depth-stop runtime changes and final higher-budget rerun outcomes showing succeeded RCA/compliance convergence.
