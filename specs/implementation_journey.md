ABOUTME: Defines the implementation workflow for Phoenix-RLM Investigator from spec to validated code.
ABOUTME: Captures the repeatable journey pattern from archive and binds it to current RLM specs.

# Implementation Journey

## Purpose

Codify how this repo is implemented end-to-end so work remains deterministic, auditable, and spec-driven.

This is the project execution playbook for Nainy + Codex.

## Non-goals

- Not a feature spec for any single engine.
- Not a replacement for `PLANS.md` ExecPlan format.
- Not a long-form architecture doc.

## Source-of-Truth Stack (top to bottom)

1. `AGENTS.md` (operating rules and project constraints)
2. `specs/README.md` (pin index)
3. Core design and interfaces:
   - `ARCHITECTURE.md`
   - `API.md`
   - `DESIGN.md`
4. Normative runtime and schema contracts:
   - `specs/rlm_runtime_contract.md`
   - `specs/rlm_engines.md`
   - `specs/formal_contracts.md`
5. Execution plan format and workflow:
   - `PLANS.md`

When any lower-level doc conflicts with a higher-level doc, update the lower-level doc in the same change.

## Journey Pattern (from archive, codified)

1. Pin-first
- Start by updating/confirming `specs/README.md`.
- Every new/changed spec must appear in the pin in the same change.

2. Spec-first
- Define boundaries, invariants, and acceptance criteria before implementation.
- Specs constrain solution space; they do not become step-by-step checklists.

3. Contract-first
- Formalize artifacts before writing engine logic.
- Required outputs are schema-bound and machine-validatable.

4. ExecPlan-driven implementation
- For complex work, author an ExecPlan in `execplan/` per `PLANS.md`.
- Use atomic, verifiable loop steps with explicit validation commands.

5. Deterministic build loops
- Prefer deterministic narrowing and stable ordering before model synthesis.
- Record seeds, budget knobs, hashes, and versions.

6. Run-artifact-first validation
- Every evaluator invocation emits a RunRecord-equivalent artifact.
- Failed/partial runs are first-class and persisted.

7. Write-back with auditability
- Phoenix annotations include `name`, `annotator_kind`, `label`, `score`, `explanation`.
- Explanations must preserve resolvable evidence pointers and run linkage.

8. Testing as backpressure
- Unit + integration + deterministic regression checks.
- Contract validation and evidence-pointer validation are mandatory gates.

## Cross-links to Active RLM Specs

- Runtime contract: `specs/rlm_runtime_contract.md`
- Three-engine source of truth: `specs/rlm_engines.md`
- Formal schemas/contracts: `specs/formal_contracts.md`
- Inspection API: `API.md`
- Architecture/dataflow: `ARCHITECTURE.md`
- Evaluator design and phase requirements: `DESIGN.md`

## Implementation Guardrails

- Python-only for Phases 1-4.
- `uv`-first command surface.
- Phoenix runs locally via `phoenix serve` (no Docker requirement).
- OpenAI models for evaluator synthesis:
  - default `gpt-4o-mini` (all calls — root and sub-calls)
  - optional upgrade for root synthesis: `gpt-4o` or `gpt-5.2`
- Trace exports are Parquet-first.
- Ground truth labels stay in external manifests, not production-like trace metadata.

## Phase Execution Contract

### Phase 1 (must execute before engine coding)

Objective:
- establish repeatable trace generation and dataset export loop.

Required outcomes:
- Phoenix reachable locally.
- Chosen tutorial path runnable.
- Traces visible in Phoenix with expected span structure.
- Seeded failures generated and exported with manifest.

Blocking policy:
- If required credentials are missing, execute all non-blocked setup and record explicit blocker in run artifacts/notes.

### Phase 2-4

Implement engines in this order:
1. Trace RCA engine
2. Policy-to-Trace Compliance engine
3. Incident Dossier engine

Each engine must:
- conform to runtime contract and formal contracts
- emit RunRecord-equivalent artifact
- write back Phoenix annotations with required fields

### Phase 5

Harden runtime:
- sandbox tests
- recursion/budget tests
- adversarial/tool-output safety checks
- reproducibility checks

### Phase 10 — RLM-RCA System

Execution path: `execplan/phase10/`

Design decisions locked:
- Custom REPL harness (not DSPy), REPL-primary execution mode
- Local subprocess sandbox with import blocklist
- CLI trigger (`python -m investigator.rca.cli`)
- `gpt-4o-mini` for all calls (upgrade root model later)
- Per-hypothesis recursive decomposition

Architecture doc: `execplan/phase10/RLM_RCA_ARCHITECTURE.md`
Implementation plan: `execplan/phase10/RLM_RCA_IMPLEMENTATION_PLAN.md`

## Definition of “Done” for Any Implementation Slice

A slice is done only if all are true:

1. Pin updated (if any spec changed).
2. Spec/contract references included in code change context.
3. Smallest relevant checks executed and recorded.
4. Run artifact emitted (if evaluator/runtime executed).
5. Phoenix write-back payloads conform to formal annotation contract.
6. Determinism requirements documented for the slice.

## Execution Cadence

- Make the smallest reasonable change per loop.
- Validate immediately.
- Record what was validated and how.
- Continue until phase acceptance criteria are satisfied.

## Acceptance Checks

1. A new contributor can follow this doc + pin to implement work without guessing.
2. Work performed using this journey produces schema-valid outputs and run artifacts.
3. Re-running on the same dataset/config produces comparable structured results.

## Keywords

implementation journey, pin-first, spec-first, contract-first, deterministic loop, run artifact, phoenix write-back, rlm engines
