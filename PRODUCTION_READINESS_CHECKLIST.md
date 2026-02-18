ABOUTME: Concrete production readiness checklist for Phoenix-RLM Investigator launch decisions.
ABOUTME: Defines measurable gates, evidence artifacts, and go/no-go criteria for RCA, compliance, and incident.

# Production Readiness Checklist

## How To Use

1. Treat each item as pass/fail.
2. Do not mark an item complete without linked evidence.
3. Run this checklist separately for:
   - `RCA`
   - `Policy Compliance`
   - `Incident Dossier`
4. Launch only capabilities that pass all `P0` gates.

## Launch Posture (Current)

- `RCA`: candidate for shadow deployment after P0 gates pass.
- `Policy Compliance`: candidate for shadow deployment after P0 gates pass.
- `Incident Dossier`: keep experimental until incident-specific gates pass.

## P0 Gates (Must Pass)

### 1) Contract Alignment

- [ ] `specs/formal_contracts.md` matches runtime models in `investigator/runtime/contracts.py`.
- [ ] Runtime validation in `investigator/runtime/validation.py` enforces all required fields and rules from formal contracts.
- [ ] No schema drift between spec, runtime dataclasses, and prompt output schemas.

Evidence:
- Updated files with matching fields/enums.
- Unit tests proving required-field enforcement for each output type.

Validation:
- `uv run pytest tests/unit/test_runtime_validation_phase6.py -q`

### 2) Evidence Integrity

- [ ] Every output claim has valid evidence pointers (`trace_id`, `span_id`, `kind`, `ref`, `excerpt_hash`).
- [ ] Evidence refs are resolvable in inspected context (not only shape-valid).
- [ ] Runs fail with explicit validation error when evidence is invalid.

Evidence:
- Failing test for unresolved evidence pointer.
- Passing test for valid evidence pointer resolution.

### 3) Reliability And Failure Rates

For a shadow sample of at least `100` runs per capability:

- [ ] `RCA` failed rate <= `2%`.
- [ ] `RCA` partial rate <= `10%`.
- [ ] `Policy Compliance` failed rate <= `1%`.
- [ ] `Policy Compliance` partial rate <= `8%`.
- [ ] No `UNEXPECTED_RUNTIME_ERROR` in the final shadow sample.

Evidence:
- Aggregated `artifacts/investigator_runs/*/run_record.json` report.

### 4) Budget And Latency Guardrails

- [ ] Wall-time caps are respected without frequent partials.
- [ ] Recursion/token/cost caps are tuned so quality remains acceptable.
- [ ] Per-run cost budget is documented and enforced by runtime config.

Evidence:
- Run-record aggregates showing budget errors trending down.
- Final configured budget profiles per engine.

### 5) Phoenix Writeback Correctness

- [ ] Root annotations are written with correct names:
  - `rca.primary`
  - `compliance.overall`
  - `incident.dossier` (when incident is enabled)
- [ ] Span-level evidence annotations are present and parseable.
- [ ] Writeback payload includes `run_id`, `schema_version`, and explanation JSON.

Evidence:
- Live Phoenix test results and captured annotation payloads.

Validation:
- `PHASE4C_LIVE=1 uv run pytest tests/integration/test_policy_compliance_live_writeback_phase4.py -q`
- `PHASE5C_LIVE=1 uv run pytest tests/integration/test_incident_dossier_live_writeback_phase5.py -q`

### 6) Sandbox And Runtime Safety

- [ ] Runtime blocks filesystem/network/subprocess access from model-generated code paths.
- [ ] Only allowlisted read-only inspection tools are callable.
- [ ] Sandbox violation paths reliably produce `SANDBOX_VIOLATION` run errors.

Evidence:
- Unit tests for blocked operations and forbidden tool calls.

Validation:
- `uv run pytest tests/unit/test_runtime_sandbox_phase8.py -q`
- `uv run pytest tests/unit/test_runtime_repl_loop_phase10.py -q`

### 7) Reproducibility Inputs

- [ ] Seeded dataset manifest is present and valid: `datasets/seeded_failures/manifest.json`.
- [ ] Exported spans dataset is present for replay: `datasets/seeded_failures/exports/spans.parquet`.
- [ ] Run records contain dataset reference (`dataset_id` or dataset hash).

Evidence:
- Replay of benchmark/canary on fixed dataset with stable results.

## Capability-Specific Gates

### RCA

- [ ] Accuracy delta vs baseline is positive on current seeded dataset.
- [ ] Label quality is stable across two consecutive benchmark runs.
- [ ] High-severity mislabels are reviewed and documented.

Suggested validation:
- `uv run pytest tests/unit/test_trace_rca_engine_phase3.py -q`
- `uv run pytest tests/unit/test_trace_rca_engine_phase9_recursive.py -q`

### Policy Compliance

- [ ] Every applicable control yields exactly one verdict.
- [ ] `insufficient_evidence` is used when required evidence is missing.
- [ ] Failing verdicts always include evidence pointers.

Suggested validation:
- `uv run pytest tests/unit/test_policy_compliance_engine_phase4.py -q`
- `uv run pytest tests/unit/test_compliance_phase10_repl.py -q`

### Incident Dossier (Do Not Launch Until Passed)

- [ ] No synthetic fallback traces (remove `trace-stub` behavior in live execution path).
- [ ] Representative trace selection is reproducible and benchmarked.
- [ ] Incident overlap/selection quality is non-regressing versus baseline on seeded datasets.
- [ ] Change-correlation evidence (`CONFIG_DIFF`) is present when asserting suspected change.

Suggested validation:
- `uv run pytest tests/unit/test_incident_dossier_engine_phase5.py -q`
- `uv run pytest tests/integration/test_incident_dossier_seeded_selection_phase5.py -q`

## CI Gates

- [ ] Unit tests required in CI.
- [ ] Integration writeback tests required in CI (against controlled live Phoenix environment).
- [ ] Failing gate blocks merge/release.

Minimum command:
- `uv run pytest -q`

## Go / No-Go Record

- Date:
- Reviewer:
- Dataset hash:
- Runtime versions:
  - RCA:
  - Compliance:
  - Incident:
- Gate summary:
  - P0 passed:
  - RCA launch decision:
  - Compliance launch decision:
  - Incident decision:
- Notes / Risks:

## Immediate Next Actions

- [ ] Align formal contracts and runtime validation strictly.
- [ ] Restore/create reproducible parquet export under `datasets/seeded_failures/exports/`.
- [ ] Re-run benchmark and canary reports on current code and record gate results.
- [ ] Keep incident off production path until incident-specific gates pass.
