# Phase 9 Master Plan: Tool-Driven Recursive RLM Engines

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

After this phase, Nainy can run RCA, compliance, and incident investigations where the model does not receive one pre-bundled evidence blob and answer once. Instead, each engine will run an explicit recursive investigation loop: plan the next inspection step, call read-only tools, delegate focused sub-investigations when needed, and finalize only after evidence is sufficient or budgets end the run.

Visible outcome: one proof run where all three capabilities still pass schema/evidence gates, and runtime artifacts show real recursive state trajectories, tool-call traces, and subcall metadata for each engine.

## Progress

- [x] (2026-02-10 17:11Z) Cleaned workspace for new implementation: committed Phase 8 master-plan closeout and stashed generated artifacts/manifest churn.
- [x] (2026-02-10 17:12Z) Created working branch `wip/phase9-aspirational-recursive-rlm`.
- [x] (2026-02-10 17:22Z) Reviewed runtime contract/spec references and current runtime/engine implementations to map aspirational-vs-current gaps.
- [ ] Add shared recursive planner protocol that produces typed runtime actions from model output.
- [ ] Wire RCA engine to execute through recursive runtime loop (tool-call planning + optional delegated label hypotheses).
- [ ] Wire compliance engine to run per-control recursive evidence collection instead of pre-bulk evidence cataloging.
- [ ] Wire incident engine to run per-trace recursive drilldown plus explicit cross-trace synthesis recursion.
- [ ] Run full unit/integration-proof validation and update outcomes with artifact links.

## Surprises & Discoveries

- Observation: `RecursiveLoop`/`SandboxGuard`/`ToolRegistry` are implemented and tested, but no production engine currently invokes `RecursiveLoop.run(...)`.
  Evidence: no runtime usage references from engine modules to `investigator.runtime.recursive_loop`; engines currently call `run_structured_generation_loop(...)` directly.
- Observation: Current engine “recursion” is deterministic in-engine traversal logic, not model-directed action planning.
  Evidence: `TraceRCAEngine._collect_branch_span_ids(...)` BFS traversal and compliance/incident deterministic evidence/profile construction.
- Observation: Current runtime contracts already carry fields needed for recursive audit (`state_trajectory`, `subcall_metadata`), reducing schema churn risk.
  Evidence: `investigator/runtime/contracts.py` and `investigator/runtime/runner.py`.

## Decision Log

- Decision: Keep deterministic narrowing/selection as the first stage in all engines, then layer model-driven recursive tool use on top.
  Rationale: preserves reproducibility and avoids cost blowups from unconstrained exploratory calls.
  Date/Author: 2026-02-10 / Codex
- Decision: Use one shared recursive planner interface for all engines rather than three bespoke planner loops.
  Rationale: sandbox, budget, and audit behavior must remain uniform under `specs/rlm_runtime_contract.md`.
  Date/Author: 2026-02-10 / Codex
- Decision: Migrate in engine order RCA -> compliance -> incident.
  Rationale: RCA has smallest scope and is best place to validate planner/action loop semantics before applying per-control and cross-trace recursion.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

Implementation not started yet. Phase starts from a clean branch with Phase 8 complete and validated. The main risk is regressions in proof thresholds during migration from single-turn judgment to iterative tool-driven recursion.

## Context and Orientation

Current runtime has two separate paths:

1. Shared single-turn structured generation (`investigator/runtime/llm_loop.py`) used by all three engines for final judgment.
2. Recursive action runtime (`investigator/runtime/recursive_loop.py`) with sandbox and tool allowlist, currently tested but not used by engines.

Current engine behavior:

- `investigator/rca/engine.py`: deterministic hot-span/branch narrowing, then optional one-shot LLM label judgment.
- `investigator/compliance/engine.py`: deterministic span/control scoping and bulk evidence cataloging, then optional one-shot per-control LLM verdict.
- `investigator/incident/engine.py`: deterministic representative selection and profile construction, then optional one-shot per-representative LLM synthesis.

To reach aspirational behavior from specs:

- model must choose iterative tool calls and delegation steps under runtime budgets;
- evidence should be gathered incrementally by objective, not fully preloaded;
- incident engine must include explicit per-trace recursive drilldown and cross-trace synthesis step.

## Plan of Work

### Phase 9A: Shared Recursive Planner Runtime

Add a planner layer that turns model output into typed actions (`tool_call`, `delegate_subcall`, `synthesize`, `finalize`) on each loop turn. The planner prompt/schema must be versioned in `investigator/prompts/` and hashed via `prompt_registry`.

Implementation focus:

- extend runtime loop execution to support planner-driven iterations instead of only pre-supplied action lists;
- preserve current sandbox and tool allowlist enforcement from `sandbox.py` and `tool_registry.py`;
- emit comprehensive runtime signals: iterations, depth, tool calls, tokens/cost, runtime state, budget reason, trajectory, subcalls.

Acceptance:

- new runtime unit tests prove planner loop can perform multiple tool actions before finalize;
- forced budget and sandbox violations still produce contract-valid `partial`/`failed` run records.

### Phase 9B: RCA Recursive Wiring

Refactor `TraceRCAEngine.run(...)` to drive recursive tool inspection through the shared planner runtime. Keep deterministic hot-root narrowing as initial seed set, then let planner choose targeted span/tool/retrieval lookups and optional delegated label subcalls.

Add an explicit “hypothesis competition” mode:

- one delegated subcall objective per candidate label family;
- parent synthesis step ranks candidates by evidence support and selects final `primary_label`.

Acceptance:

- RCA run records include non-empty `state_trajectory` and `subcall_metadata` when competition mode is enabled;
- output schema/evidence validation and writeback behavior remain intact.

### Phase 9C: Compliance Recursive Wiring

Replace bulk prefetch evidence cataloging with control-centric recursive evidence loops:

- for each scoped control, planner requests only required evidence paths (`required_evidence(...)` plus selective tool calls);
- loop stops when control evidence checklist is satisfied, insufficient, or budget terminated.

Preserve deterministic guardrails:

- missing required evidence must still force `insufficient_evidence` without hallucinated pass/fail.

Acceptance:

- per-control recursion metadata present in run artifacts;
- compliance gates remain above threshold on frozen proof data.

### Phase 9D: Incident Hierarchical Recursive Wiring

Keep deterministic representative selection contract from Phase 8E1, then add two recursive levels:

1. Per-representative trace drilldown subcalls (mini RCA-like evidence enrichment).
2. Parent cross-trace synthesis recursion that clusters repeated signatures/patterns and ranks hypotheses/actions.

Config diff correlation remains an explicit evidence tool call in the recursive process.

Acceptance:

- dossier hypotheses include cross-trace evidence groups, not only per-trace deduped statements;
- run artifacts show representative drilldown subcalls and top-level synthesis steps.

### Phase 9E: Proof and Hardening

Run targeted and full validations, then execute full proof flow:

- all runtime/engine tests pass;
- proof report passes gates with recursive paths enabled;
- run records demonstrate recursive behavior for all three engines.

## Concrete Steps

All commands run from repository root (`/Users/nainy/Documents/Personal/rlm_observability`).

1. Add RED tests for planner-driven recursive loop behavior.
   `uv run pytest tests/unit -q -k "runtime and recursive and planner"`
2. Implement shared recursive planner runtime changes.
3. Add RED tests for RCA recursion wiring and hypothesis subcalls.
   `uv run pytest tests/unit -q -k "trace_rca and phase9 and recursive"`
4. Implement RCA recursive wiring and verify.
   `uv run pytest tests/unit -q -k "trace_rca or runtime"`
5. Add RED tests for compliance per-control recursion.
   `uv run pytest tests/unit -q -k "compliance and phase9 and recursive"`
6. Implement compliance recursive wiring and verify.
   `uv run pytest tests/unit -q -k "compliance or runtime"`
7. Add RED tests for incident per-trace drilldown + cross-trace synthesis recursion.
   `uv run pytest tests/unit -q -k "incident and phase9 and recursive"`
8. Implement incident recursive wiring and verify.
   `uv run pytest tests/unit -q -k "incident or runtime or proof_benchmark"`
9. Run full regression.
   `uv run pytest tests/ -q -rs`
10. Run proof gate.
   `PHOENIX_WORKING_DIR=.phoenix_data uv run python -m investigator.proof.run_phase7_proof`

## Validation and Acceptance

Phase 9 is complete when:

- All three engines execute with planner-driven recursive tool usage (observable in `state_trajectory` and `subcall_metadata`).
- Engine outputs remain schema-valid and evidence-linked under `specs/formal_contracts.md`.
- Sandbox and budget constraints are enforced during recursive execution.
- Proof report passes all gates in one run artifact with recursive paths enabled.
- Full regression test suite passes (except explicit opt-in live skips).

## Idempotence and Recovery

Runtime changes will be introduced behind explicit engine-level toggles (`use_recursive_runtime` style flags) during migration so fallback remains available. If a capability regresses, disable recursive path only for that engine and continue validating others.

Generated artifacts remain outside committed source by default; unexpected proof churn should be stashed before continuing iterative development.

## Artifacts and Notes

Primary artifacts to track during implementation:

- `artifacts/investigator_runs/<run_id>/run_record.json`
- `artifacts/investigator_runs/<run_id>/output.json`
- `artifacts/proof_runs/<proof_run_id>/proof_report.json`

Pre-phase cleanup record:

- Commit: `a989245` (`phase8: mark master rollout plan complete`)
- Stash: `stash@{0}` (`pre-aspirational-rlm-cleanup`)

## Interfaces and Dependencies

New/updated interfaces expected:

- Shared recursive planner protocol in runtime module (planner request/response schema tied to typed actions).
- Engine adapters translating engine-specific objectives/context into planner inputs and synthesis outputs.
- Existing read-only inspection API surface from `investigator/inspection_api/protocol.py` remains the only runtime tool boundary.

Dependencies:

- OpenAI structured generation path via `investigator/runtime/llm_client.py`.
- Prompt registry/versioning via `investigator/runtime/prompt_registry.py`.
- Runtime contracts and schemas from `investigator/runtime/contracts.py` and `specs/formal_contracts.md`.

## References

Reviewed fully before drafting this plan, with behaviors to reuse:

- `PLANS.md`
  Behavior reused: ExecPlan structure and living-document requirements.
- `specs/rlm_runtime_contract.md`
  Behavior reused: sandbox, recursion limits, state transitions, run artifact requirements.
- `specs/rlm_engines.md`
  Behavior reused: aspirational engine recursion strategies and evidence policies.
- `specs/formal_contracts.md`
  Behavior reused: output and run record schema invariants.
- `API.md`
  Behavior reused: read-only inspection tool boundary and canonical evidence refs.
- `investigator/runtime/recursive_loop.py`
  Behavior reused: typed action execution, state machine, budget termination, subcall metadata.
- `investigator/runtime/sandbox.py`
  Behavior reused: action/tool validation and JSON-safe payload enforcement.
- `investigator/runtime/tool_registry.py`
  Behavior reused: allowlisted Inspection API invocation and hash logging.
- `investigator/runtime/runner.py`
  Behavior reused: run-record persistence and failure/status mapping.
- `investigator/runtime/llm_loop.py`
  Behavior reused: structured output retry loop and `MODEL_OUTPUT_INVALID` semantics.
- `investigator/runtime/llm_client.py`
  Behavior reused: model invocation and usage/cost accounting.
- `investigator/runtime/prompt_registry.py`
  Behavior reused: prompt/schema version pinning and hash derivation.
- `investigator/inspection_api/protocol.py`
  Behavior reused: canonical read-only inspection API signatures.
- `investigator/inspection_api/phoenix_client.py`
  Behavior reused: deterministic trace/span/tool/config retrieval semantics.
- `investigator/inspection_api/parquet_client.py`
  Behavior reused: deterministic offline proof dataset behavior.
- `investigator/rca/engine.py`
  Behavior reused: deterministic hot-span narrowing and branch-root traversal seed logic.
- `investigator/compliance/engine.py`
  Behavior reused: control scoping, required-evidence semantics, deterministic insufficient-evidence precedence.
- `investigator/incident/engine.py`
  Behavior reused: deterministic representative selection contract and config diff correlation.
- `investigator/rca/workflow.py`
  Behavior reused: run/writeback orchestration and annotator-kind mapping.
- `investigator/compliance/workflow.py`
  Behavior reused: run/writeback orchestration and partial-on-writeback-failure handling.
- `investigator/incident/workflow.py`
  Behavior reused: run/writeback orchestration and annotator-kind mapping.
- `tests/unit/test_runtime_recursive_loop_phase8.py`
  Behavior reused: recursion termination and runner partial mapping tests.
- `tests/unit/test_runtime_sandbox_phase8.py`
  Behavior reused: sandbox rejection coverage.
- `tests/unit/test_runtime_state_machine_phase8.py`
  Behavior reused: state transition invariants.
- `tests/unit/test_runtime_runner_llm_usage_phase8.py`
  Behavior reused: runtime usage/cost persistence checks.

Revision Note (2026-02-10): Initial Phase 9 plan drafted to migrate from single-turn LLM judgments to tool-driven recursive engine execution aligned with aspirational RLM behavior.
