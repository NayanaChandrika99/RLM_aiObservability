# Phase 8B: Sandboxed Recursive Inspection Execution

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

This phase adds the real recursive runtime behavior. After completion, Nainy can run an evaluator that performs bounded multi-step inspection loops: choose what to inspect, call read-only tools, optionally create focused subcalls, and synthesize output. The user-visible difference is richer evidence trails and explicit recursion metadata in run artifacts.

In this repository, "sandboxed recursive execution" means model-directed inspection that is strictly confined to allowlisted read-only tools and bounded by depth, tool-call, token, subcall, cost, and wall-time budgets.

## Progress

- [x] (2026-02-10 15:00Z) Reviewed recursion and sandbox contract requirements in runtime specs.
- [x] (2026-02-10 15:00Z) Drafted initial Phase 8B plan.
- [x] (2026-02-10 15:16Z) Revised Phase 8B with explicit action types, state machine, and budget ownership.
- [x] (2026-02-10 15:50Z) Added runtime recursion state machine and subcall metadata tracking in `recursive_loop.py` and `run_record.runtime_ref`.
- [x] (2026-02-10 15:50Z) Added sandbox guard layer that rejects unknown actions and forbidden tools.
- [x] (2026-02-10 15:50Z) Added allowlisted tool invocation registry bound to `InspectionAPI`.
- [x] (2026-02-10 15:50Z) Added and passed budget-stop/state/sandbox tests for recursive loop and runner partial mapping.

## Surprises & Discoveries

- Observation: Current runtime enforces budget limits after engine completion, not during recursive decision-making.
  Evidence: `investigator/runtime/runner.py` checks limits from aggregated signals after `engine.run(request)` returns.
- Observation: Existing engine logic already isolates read access through `InspectionAPI` methods.
  Evidence: `investigator/rca/engine.py`, `investigator/compliance/engine.py`, and `investigator/incident/engine.py` call inspection APIs rather than direct Phoenix reads in core logic.
- Observation: Runner needs explicit runtime-state input (`runtime_state=terminated_budget`) to emit partial status when counters do not exceed post-hoc thresholds.
  Evidence: `tests/unit/test_runtime_recursive_loop_phase8.py` requires `run_engine` to map terminated budget to `status=partial` using runtime signals.

## Decision Log

- Decision: Implement recursion as explicit runtime steps with typed actions, not free-form Python execution.
  Rationale: Typed actions are easier to validate, safer to sandbox, and align with deterministic replay goals.
  Date/Author: 2026-02-10 / Codex
- Decision: Keep recursion depth defaults from contract and expose all budget knobs per run.
  Rationale: Ensures direct compatibility with `specs/rlm_runtime_contract.md` and proof reproducibility.
  Date/Author: 2026-02-10 / Codex
- Decision: Make `runner.py` the canonical budget and artifact owner; recursive loop updates shared counters but does not persist artifacts itself.
  Rationale: Prevents conflicting sources of truth for status/usage.
  Date/Author: 2026-02-10 / Codex
- Decision: Implement tool registry as a thin allowlist/normalization layer on top of `InspectionAPI`, not a second abstraction hierarchy.
  Rationale: Avoids duplicate interfaces while preserving sandbox control.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

Phase implemented with typed recursive loop execution, sandbox validation, allowlisted tool registry calls, and runner mapping for `terminated_budget -> partial` artifacts.

## References

The following files were reviewed fully before drafting and revising this plan; listed behaviors will be reused:

- `specs/rlm_runtime_contract.md`
  Behavior reused: runtime state machine, recursion limits, sandbox restrictions, and error codes.
- `specs/rlm_engines.md`
  Behavior reused: engine-specific recursion styles and stopping criteria.
- `API.md`
  Behavior reused: canonical read-only tool signatures.
- `investigator/inspection_api/protocol.py`
  Behavior reused: stable Python protocol for tool-call allowlist.
- `investigator/runtime/runner.py`
  Behavior reused: run record lifecycle and error emission.
- `investigator/runtime/contracts.py`
  Behavior reused: budget and usage dataclasses, run record serialization.
- `investigator/runtime/validation.py`
  Behavior reused: final schema/evidence gate after recursive synthesis.
- `investigator/inspection_api/phoenix_client.py`
  Behavior reused: deterministic sorting and read-only access semantics.

## Context and Orientation

The runtime today treats each engine as a single opaque function call. That design cannot express recursive investigation trajectories, parent/child call relationships, or mid-run budget enforcement. This phase introduces a reusable recursion shell that engines can drive with objective-specific prompts and tool policies.

A "subcall" in this plan means a focused recursive investigation step that inherits budget context, increases depth, and returns a normalized result object to the parent step.

## Plan of Work

Add `investigator/runtime/recursive_loop.py` with explicit typed actions. Initial action set:

- `tool_call` with allowlisted `tool_name` and validated arguments.
- `delegate_subcall` with child objective and bounded context reference set.
- `synthesize` to build/update a structured draft output from collected evidence.
- `finalize` to terminate loop with final structured output.

Add `investigator/runtime/sandbox.py` that validates action objects before execution. Any unknown action, forbidden tool, or disallowed argument shape raises `SANDBOX_VIOLATION`.

Add `investigator/runtime/tool_registry.py` as thin allowlist wrapper over `InspectionAPI` methods. Registry responsibilities are only:

- action/tool-name allowlisting,
- deterministic argument normalization,
- response normalization and hashing for audit logs.

State machine definition for the recursive loop:

- `initialized -> planning`
- `planning -> acting` (tool call), `planning -> delegating` (subcall), `planning -> finalizing`
- `acting -> planning`
- `delegating -> planning`
- `finalizing -> completed`
- any state -> `terminated_budget` or `failed`

Subcall merge contract:

- child returns `{summary, evidence_refs, gaps, status}`.
- parent merges `evidence_refs` by `(kind, ref)` dedupe key.
- parent appends child `gaps` with `subcall:<id>` prefix.
- child `failed` status is recorded in parent trajectory and may continue unless fatal policy says stop.

Budget ownership split:

- `runner.py` creates canonical budget and counter objects.
- `recursive_loop.py` increments shared counters per action and checks limits pre/post step.
- `runner.py` performs final status mapping, validation gates, and artifact persistence.

Add tests in one RED/GREEN loop per behavior:

- invalid state transition rejection,
- recursion depth enforcement,
- tool-call quota enforcement,
- cost/wall-time cap enforcement,
- sandbox violation fail-fast,
- terminated-budget -> partial output behavior,
- deterministic action ordering under fixed seed.

## Concrete Steps

All commands run from repository root.

1. Write failing recursion, state-machine, and sandbox tests.
   - `uv run pytest tests/unit -q -k "phase8 and recursion"`
   - `uv run pytest tests/unit -q -k "phase8 and sandbox"`
2. Implement `recursive_loop.py`, `tool_registry.py`, and `sandbox.py`.
3. Wire runtime runner integration with shared budget counters.
4. Re-run focused tests.
   - `uv run pytest tests/unit -q -k "phase8 and (recursion or sandbox)"`
5. Run runtime regression tests.
   - `uv run pytest tests/unit -q -k "runtime"`

## Validation and Acceptance

Acceptance requires:

- Recursive runs emit measurable depth/tool/subcall counters into run records.
- Forced over-budget runs terminate as `partial` or `failed`, never hang.
- Sandbox violations return `SANDBOX_VIOLATION` and persist error detail.
- All tool calls in recursion path route only through allowlisted read-only API methods.
- State transitions in trajectory logs match the declared state machine.

## Idempotence and Recovery

Recursive state changes must remain in-memory per run and never mutate source datasets. If a recursion step fails, runtime should return the best available partial output with explicit gaps when possible; otherwise fail with preserved trajectory metadata. Re-running with same dataset and seed should remain comparable.

## Artifacts and Notes

Expected new/updated files:

- `investigator/runtime/recursive_loop.py`
- `investigator/runtime/tool_registry.py`
- `investigator/runtime/sandbox.py`
- `investigator/runtime/runner.py`
- `tests/unit/test_runtime_recursive_loop_phase8.py`
- `tests/unit/test_runtime_sandbox_phase8.py`
- `tests/unit/test_runtime_state_machine_phase8.py`

## Interfaces and Dependencies

Required interfaces after this phase:

- `RecursiveLoop.run(...)` with depth, budget tracker, and action parser.
- `ToolRegistry.call(action_name, args)` returning normalized tool results.
- `SandboxGuard.validate(action)` raising runtime-standard violations.
- `SubcallResult` dataclass for parent-child merge behavior.

Dependencies:

- Phase 8A LLM loop interface must be available before full recursion integration.
- Existing `InspectionAPI` implementations remain unchanged at signature level.

Revision Note (2026-02-10): Initial Phase 8B plan created to add contract-compliant recursive execution and sandbox enforcement on top of Phase 8A shared model loop.
Revision Note (2026-02-10): Revised after Nainy review to define typed actions, explicit state transitions, subcall merge rules, and canonical budget ownership between recursive loop and runner.
