# Testing (Goldens, determinism, and pytest)

## Purpose

Define the MVP test strategy so the system remains deterministic, debuggable, and safe to extend.

This spec answers:

- what we test,
- where tests live,
- what “golden scenarios” are,
- and how determinism is enforced.

## Non-goals

- Not a full CI/CD pipeline document.
- Not load/performance testing (post-MVP).
- Not a benchmarking/performance comparison between solvers.

## Test pyramid (MVP)

### 1) Unit tests (fast)

Target pure logic:

- risk scoring math + normalization
- graph traversal and tier annotation
- Optimization IR validation + compilation (small cases)
- adapter receipt formatting + idempotency behavior (stubbed)

### 2) Integration tests (medium)

Target pipeline wiring:

- run one scenario end-to-end and assert:
  - `RunRecord` exists
  - outcome is expected
  - artifacts are schema-valid

### 3) Golden scenarios (behavioral regression)

Run a suite of scenario fixtures (15–20) and assert stable outcomes and key properties.

## Golden scenarios

### Fixture shape

Each scenario fixture should include:

- input incident (`IncidentTicket` fixture or raw incident text)
- fixed `twin_state.json` snapshot reference
- in `data/fixtures/scenarios/*.json`, store this as `twin_state_path` (repo-relative path)
- expected KG result:
  - `PATHS_FOUND | NO_PATHS | LOW_COVERAGE`
- in `data/fixtures/scenarios/*.json`, store this as `expected.kg_result`
- **Requirement (pinned):** every scenario fixture MUST include `expected.kg_result`.
  - Scenarios that provide `plan_options` (and therefore may bypass KG-derived planning) MUST still
    pin `expected.kg_result` so negative cases remain regression-testable.
- optional plan options (for twin verification + scorecard regression):
  - store as top-level `plan_options` (list of `PlanOption` objects)
- expected run outcome (`RunOutcome` enum)
- expected properties (high-level assertions), e.g.:
  - “must require approval”
  - “must avoid Supplier A”
  - “must produce at least 2 PlanOptions”
  - “all TwinResult entries are feasible”

### Coverage targets

The suite should cover:

- positive cases (paths exist)
- negative cases (no paths)
- partial coverage cases
- infeasible plan cases (`NO_VIABLE_PLAN`)
- approval vs rejection outcomes
- adapter failures (`EXECUTION_FAILED`)

### Scenario library coverage assertions (pinned)

The golden scenario index (`data/fixtures/scenarios/index.json`) is the evaluation scenario library. It must
remain intentionally shaped so we do not accidentally lose coverage breadth as scenarios are added/removed.

The test suite MUST enforce:

- **Suite size band:** 15–20 scenarios in the index.
- **Expected KG result distribution** (by `expected.kg_result`):
  - `PATHS_FOUND >= 12` (true positives)
  - `NO_PATHS >= 3` (negative cases)
  - `LOW_COVERAGE >= 3` (edge cases / partial coverage)
- **Disruption type coverage** (by `incident_ticket.disruption_type`): at least one scenario for each of:
  - `geopolitical`
  - `trade_policy`
  - `natural_disaster`
  - `labor_strike`
  - `cybersecurity_incident`

The intent is to enforce minimum coverage breadth, not to overconstrain exact scenario counts. Additional
disruption types are allowed.

## Determinism rules

These are required for golden stability:

- **Fixed seeds** for synthetic KG generation and any randomized simulation.
- **Stable IDs** for nodes/edges and artifacts.
- **Timestamp handling**:
  - freeze time in tests, or
  - normalize timestamps out of snapshot comparisons.
- **Stable ordering**:
  - sorted lists where ordering matters (risk rankings, plan rankings).
- **No ambient randomness**:
  - avoid calling `uuid4()` or `random.random()` without a seeded RNG in test paths.
- **Serialization stability**:
  - JSON output must be stable (sorted keys, consistent formatting) to avoid noisy diffs.

### LLM determinism (app runtime)

The LangGraph app runtime uses LLM-backed agents. Tests MUST remain deterministic and offline.

Rules:

- Golden tests continue to use `plan1/scenarios.py` (no LLM, no LangGraph).
- Any tests that exercise the app runtime MUST run in LLM replay mode:
  - `PLAN1_LLM_MODE=replay`
  - cache fixtures live under `tests/fixtures/llm/` and are committed to the repo
- CI MUST fail fast on LLM cache misses (no network calls in CI).

## Test runner and layout

### Runner

- Use `pytest` as the canonical test runner.

### Recommended layout

- `tests/unit/` — pure logic tests
- `tests/integration/` — pipeline tests
- `tests/golden/` — golden scenario tests
- `data/fixtures/` — scenario fixtures and synthetic KG fixtures (committed, small)
- `artifacts/` — run outputs (not committed; tests should not write here)

## What pytest asserts (MVP)

Golden run assertions should include:

- `RunRecord` always exists.
- `RunRecord.outcome == expected_outcome`.
- `KGImpactMap` and `Tier1RiskTable` exist for every scenario run.
- If `expected_kg_result == NO_PATHS`, then `KGImpactMap.disrupted_nodes == []` and `Tier1RiskTable.risks == []`.
- If `expected_kg_result == LOW_COVERAGE`, then:
  - `RunRecord.outcome == LOW_CONFIDENCE`, and
  - `RunRecord.reason` is present and follows the pinned reason string rules in `specs/workflow.md`.
- If `RunRecord.outcome == NO_IMPACT`, then `Tier1RiskTable` is empty.
- If `RunRecord.outcome == NO_VIABLE_PLAN`, then all `TwinResult.feasible == false`.
- Risk scoring is deterministic for a fixed KG snapshot.
- Optimization IR validation passes for generated IR.

## Failure handling in tests

Tests should make early-exit and failure branches explicit, and assert:

- `RunRecord.reason` is present and meaningful for early exits and failures.
- `ExecutionReceipt[]` exists whenever execution is attempted, even with `dry_run=true`.
- Adapter failures surface as `EXECUTION_FAILED` with receipts indicating `status="FAILED"`.

## Golden update workflow

Golden snapshots must only be updated intentionally. Tests should respect an explicit environment
variable, `PLAN1_UPDATE_GOLDENS=1`, which enables snapshot regeneration. When the variable is not set,
tests must fail on mismatches and display a focused diff for review. Golden snapshots should be written
under `tests/golden/<scenario_id>/` and tests must use `tmp_path` for run outputs to keep the working
tree clean.

To reduce accidental golden churn, tests MAY support scoping via `PLAN1_GOLDEN_SCENARIOS` (a comma-separated
list of scenario IDs). When set, golden update behavior should update only the scoped scenarios; otherwise
golden updates should default to creating missing snapshots only.

## Acceptance checks

Once implemented:

- Running the test suite produces deterministic results on repeated runs.
- Adding a new scenario requires:
  - a fixture file,
  - an entry in a scenario index (if used),
  - and expected-outcome assertions.

## Keywords

testing, pytest, goldens, regression, determinism, fixtures, integration tests, unit tests
