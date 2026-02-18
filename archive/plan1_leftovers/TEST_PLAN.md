ABOUTME: Test plan for validating the Plan1 supply-chain MVP end-to-end.
ABOUTME: Focuses on determinism, artifacts, and verifier-first execution.

# Plan1 MVP Test Plan

This repo is in a “plan-first” stage. This test plan defines *what we must be able to validate* once the implementation exists.

Primary references:

- `TEST_PLAN.md` (this file)
- `specs/README.md` (spec index / “pin”)
- `specs/runbook.md` (canonical commands)
- `specs/testing.md` (testing strategy + determinism rules)
- `specs/contracts.md` (artifact schemas)

## Goals

- Validate the end-to-end pipeline produces a `RunRecord` for every run (including failures / early exits).
- Validate outcomes and artifacts match the contracts in `specs/contracts.md`.
- Enforce determinism for “golden scenarios” (fixed seed → stable results).
- Ensure “side effects go through adapters”, defaulting to `dry_run=true`, and always emitting receipts when execution is attempted.

## Non-goals (MVP)

- Performance/load testing.
- Live web/RSS ingestion.
- Real ERP writes (unless explicitly enabled and gated by approval policy).

## Prerequisites

- Python environment managed by `uv` (preferred).
- `pytest` as test runner (once tests exist).
- A small committed fixture set under `data/fixtures/`.

## Quick smoke checks (manual)

These are the fastest checks to run locally and confirm the system is wired correctly.

### 1) Install dependencies

Preferred:

    uv sync

Fallback:

    python3 -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt

### 2) Generate synthetic fixtures (deterministic)

    uv run python -m plan1.kg.generate --seed 42 --out data/fixtures/kg

Expected files:

- `data/fixtures/kg/nodes.json`
- `data/fixtures/kg/edges.json`
- `data/fixtures/kg/twin_state.json`

### 3) Run a scenario (dry-run)

    uv run python -m plan1.demo --scenario scn_012_paths_found --dry-run --out artifacts/runs

Expected:

- `artifacts/runs/<run_id>/run_record.json` exists
- Any additional artifacts written are consistent with the `RunRecord` outcome

### 4) Run a manual incident (dry-run)

    uv run python -m plan1.demo --incident-text "Embargo affects aluminum exports from Country X" --dry-run --out artifacts/runs

Expected:

- A `RunRecord` exists even if outcome is `LOW_CONFIDENCE` or `NO_IMPACT`.

### 5) Resume a HITL run (approval gate)

If a run pauses for approval, resume it:

    uv run python -m plan1.runner resume --run-id <run_id> --decision APPROVED --decided-by nainy

Expected:

- A new `RunRecord` (or an updated run record) reflects the approval decision.
- If execution is attempted (even in dry-run), an `ExecutionReceipt` is present.

## Determinism checks (golden stability)

Run the same scenario twice with the same seed and confirm stable outputs.

Baseline procedure:

1. Run scenario A with fixed seed.
2. Run scenario A again with the same seed.
3. Compare key artifacts (at minimum `run_record.json`).

Notes:

- If `RunRecord` contains timestamps, tests must either freeze time or normalize timestamps out of comparisons (per `specs/testing.md`).
- Ordering-sensitive lists must be stable (sorted) where order affects outputs (plan ranking, risk ranking).

## Contract-level acceptance checks

These are the behaviors that should be enforced by automated tests (once code exists).

### A) RunRecord always exists

For every run attempt (success, early exit, error):

- `run_record.json` exists.
- `RunRecord.outcome` is present and valid.
- Failure branches are explicit (example outcomes mentioned in specs: `LOW_CONFIDENCE`, `NO_IMPACT`, `NO_VIABLE_PLAN`).

### B) Outcome ↔ artifact consistency

High-value invariants from `specs/testing.md`:

- If outcome is `NO_IMPACT`, `Tier1RiskTable` is empty.
- If outcome is `NO_VIABLE_PLAN`, all `TwinResult.feasible == false`.
- If expected KG result is `NO_PATHS`, then `KGImpactMap.disrupted_nodes == []`.

### C) Verifier-first behavior

For each `PlanOption` considered:

- A corresponding `TwinResult` exists.
- `TwinResult` indicates feasibility and provides violations when infeasible.
- Scorecard/ranking logic is deterministic for a fixed input snapshot.

### D) Adapter + receipts

When execution is attempted:

- All side effects go through adapter interfaces.
- `dry_run=true` is the default.
- Receipts exist for attempted actions (even in dry-run), and are attached to the run record.

## Test suite structure (pytest)

Once implementation starts, prefer this layout (per `specs/testing.md`):

- `tests/unit/` — pure logic (risk scoring, graph traversal, IR validation)
- `tests/integration/` — one end-to-end scenario; assert artifacts exist and are schema-valid
- “goldens” — a suite of 15–20 deterministic scenario fixtures with stable assertions

## Update log (keep this current)

When we implement or validate a feature, add a short entry here:

- Date:
- What was validated:
- How validated (`pytest`, manual command, both):
- Notes / follow-ups:

- Date: 2026-01-20
  What was validated: Golden determinism harness (scenario replay + golden snapshot comparison) and baseline contract/IR unit tests.
  How validated (`pytest`, manual command, both): `uv run pytest` (plus lint: `uv run ruff check plan1 tests`).
  Notes / follow-ups: Added opt-in golden update workflow via `PLAN1_UPDATE_GOLDENS=1`.

- Date: 2026-01-20
  What was validated: Optimization IR builder/compiler tests, full golden suite (15 scenarios), contract example fixtures, and runbook CLI integration test.
  How validated (`pytest`, manual command, both): `uv run pytest`.
  Notes / follow-ups: Pytest emits OR-Tools deprecation warnings from SWIG bindings.

- Date: 2026-01-20
  What was validated: KG query + Tier-1 risk artifacts (`KGImpactMap`, `Tier1RiskTable`) emitted in scenario runs, plus golden-suite invariants for `NO_PATHS`.
  How validated (`pytest`, manual command, both): `uv run pytest`.
  Notes / follow-ups: Updated goldens for `scn_011_no_paths` and `scn_012_paths_found`; added committed synthetic KG fixtures under `data/fixtures/kg/`.

- Date: 2026-01-20
  What was validated: Digital Twin optimization stage (PlanOption → Optimization IR → OR-Tools solve) emitting `TwinResult` + `Scorecard` in scenario runs.
  How validated (`pytest`, manual command, both): `uv run pytest`.
  Notes / follow-ups: Updated goldens for `scn_005_no_viable_plan` and `scn_014_alt_supplier_approved`; added unit tests for scorecard ordering and twin evaluation.

- Date: 2026-01-20
  What was validated: Twin state snapshot guardrails (`twin_state.json` schema models + deterministic validator + solver preflight returning infeasible-with-violations).
  How validated (`pytest`, manual command, both): `uv run pytest` and `uv run python -m plan1.twin_state_validator --path data/fixtures/kg/twin_state.json`.
  Notes / follow-ups: Added unit tests pinning `data/fixtures/kg/twin_state.json`; documented validator usage in `specs/runbook.md` and `specs/twin-state.md`.

- Date: 2026-01-21
  What was validated: Deterministic PlanOption generation (incident/KG/risk → computed `PlanOption[]`) and IR validation for every generated option against the canonical twin state snapshot.
  How validated (`pytest`, manual command, both): `uv run pytest -q`.
  Notes / follow-ups: Added a golden scenario that omits fixture `plan_options` and still produces twin artifacts; pinned required parameters per `action_type` in unit tests.

- Date: 2026-01-21
  What was validated: Execution policy + adapters wired into scenario runs: policy triggers in `ApprovalRecord` and adapter-produced `ExecutionReceipt[]` (including deterministic failure injection).
  How validated (`pytest`, manual command, both): `uv run pytest -q` (golden suite updated via `.venv/bin/python -m pytest` with `PLAN1_UPDATE_GOLDENS=1` due to sandboxed uv cache restrictions).
  Notes / follow-ups: Added unit tests for policy trigger determinism and idempotency key format; updated scenarios to include plan options for approval/execution outcomes.

- Date: 2026-01-21
  What was validated: LangGraph app runtime skeleton (`python -m plan1.app`) with SQLite checkpointing and HITL pause/resume (no RunRecord on pause; RunRecord on resume).
  How validated (`pytest`, manual command, both): `uv run pytest` (new integration test: `tests/integration/test_app_hitl_pause_resume.py`).
  Notes / follow-ups: Resume uses LangGraph `Command(update=...)` and streams to detect interrupts.

- Date: 2026-01-21
  What was validated: OpenAI-first LLM boundary with record/replay cache (offline-safe replay, deterministic cache keys, fail-fast on cache miss).
  How validated (`pytest`, manual command, both): `uv run pytest` (unit tests: `tests/unit/test_llm_cache.py`, `tests/unit/test_llm_client_replay.py`).
  Notes / follow-ups: Env vars are shell-exported (no implicit `.env` loading); real fixture recording is a local-dev step.

- Date: 2026-01-21
  What was validated: Sentinel LLM agent (incident text → `IncidentTicket`) in replay mode, including LOW_CONFIDENCE and EXTRACT_FAILED mapping.
  How validated (`pytest`, manual command, both): `uv run pytest` (fixtures in `tests/fixtures/llm/`; tests: `tests/unit/test_sentinel_runtime.py`, `tests/integration/test_app_sentinel_replay.py`).
  Notes / follow-ups: Sentinel is now wired into `plan1.app` and is offline-safe under `PLAN1_LLM_MODE=replay`.

- Date: 2026-01-21
  What was validated: CSCO LLM plan proposer node (LLM draft → deterministic normalization → Optimization IR validation gate → deterministic fallback) wired into `plan1.app` and offline-safe in replay mode.
  How validated (`pytest`, manual command, both): `uv run pytest` (fixtures in `tests/fixtures/llm/`; tests: `tests/unit/test_csco_node.py`, `tests/integration/test_app_csco_replay.py`).
  Notes / follow-ups: Resume RunRecord currently does not guarantee inclusion of pre-interrupt artifacts; CSCO unit tests assert plan options behavior directly.

- Date: 2026-01-21
  What was validated: Phoenix/OpenTelemetry tracing for `plan1.app` (root + stage spans; `RunRecord.trace_ids` populated when enabled).
  How validated (`pytest`, manual command, both): `uv run ruff check plan1 tests` and `uv run pytest` (integration test: `tests/integration/test_observability_tracing.py`).
  Notes / follow-ups: Tracing is opt-in via env vars; tests use a local dummy OTLP/HTTP endpoint (no Phoenix collector required).

- Date: 2026-01-21
  What was validated: Golden suite is generated via the LangGraph workflow graph (same path as `plan1.app`), and simulation KPIs are attached to top-ranked feasible options in golden run records.
  How validated (`pytest`, manual command, both): `PLAN1_UPDATE_GOLDENS=1 uv run pytest tests/integration/test_golden_suite.py -q`, then `uv run pytest -q`.
  Notes / follow-ups: Verified `lead_time_p50_days`/`lead_time_p95_days` appear in `tests/golden/*/run_record.normalized.json` for scenarios with feasible top-ranked options.

- Date: 2026-01-21
  What was validated: Scenario fixture expectations are pinned for KG negative cases (`expected.kg_result` required; `NO_PATHS` and `LOW_COVERAGE` outcomes/reasons asserted), and the Tier‑4 KG generator + invariant validator are green for seed 42.
  How validated (`pytest`, manual command, both): `UV_CACHE_DIR=./.uv-cache uv run pytest -q` and `UV_CACHE_DIR=./.uv-cache uv run python -m plan1.kg.validate --dir data/fixtures/kg`.
  Notes / follow-ups: `uv run` may require `UV_CACHE_DIR=./.uv-cache` in sandboxed environments; approval-policy threshold triggers and report artifacts remain open gaps.

- Date: 2026-01-22
  What was validated: App runtime infers insufficient KG coverage from incident entities (no fixture hints) and terminates as `LOW_CONFIDENCE`, emitting `KGImpactMap.uncertainty_flags`.
  How validated (`pytest`, manual command, both): `uv run pytest -q` (including `-k runtime_low_coverage_inference` and `-k golden_suite` during TDD).
  Notes / follow-ups: Entity resolution now normalizes simple case/whitespace so Sentinel replay fixture output like product `"aluminum"` resolves deterministically.

- Date: 2026-01-22
  What was validated: `RunRecord.timeline` contains per-step stage keys for LangGraph runs (separately asserted from goldens) and golden snapshots were regenerated to include the normalized timeline keys.
  How validated (`pytest`, manual command, both): `UV_CACHE_DIR=./.uv-cache uv run pytest -q`.
  Notes / follow-ups: In sandboxed environments, `uv run` may require `UV_CACHE_DIR=./.uv-cache` to avoid permission errors in `~/.cache/uv`.

- Date: 2026-01-22
  What was validated: Closed parity-audit gaps for twin modeling (dock capacity constraint; reliability + inventory semantics), CSCO output (`ExecutiveActionPlan`), and adapter receipts (`compensation_hint`); goldens updated for impacted scenarios.
  How validated (`pytest`, manual command, both): `UV_CACHE_DIR=./.uv-cache uv run pytest -q`.
  Notes / follow-ups: OR-Tools SWIG deprecation warnings remain (pytest output is otherwise clean).

- Date: 2026-01-22
  What was validated: Single-product scope is explicit via a pinned “Pepsi” SKU family BOM; BOM component names are validated against committed KG product fixtures.
  How validated (`pytest`, manual command, both): `UV_CACHE_DIR=./.uv-cache uv run pytest -q -k product_bom`.
  Notes / follow-ups: This is a static MVP lookup (not a general product search/enrichment system).

- Date: 2026-01-22
  What was validated: Sentinel emits deterministic diagnostic questions for network tracing; CSCO replay remains offline-stable by excluding diagnostic questions from the CSCO prompt payload.
  How validated (`pytest`, manual command, both): `UV_CACHE_DIR=./.uv-cache uv run pytest -q -k diagnostic_questions` and `UV_CACHE_DIR=./.uv-cache uv run pytest -q -k app_csco_replay`.
  Notes / follow-ups: Diagnostic questions are post-LLM deterministic helpers; no additional LLM calls are introduced.
