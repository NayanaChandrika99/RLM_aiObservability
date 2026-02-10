# RLM Runtime Contract

## Purpose

Define the single runtime contract that every RLM engine in this project must obey.

This contract is the binding implementation spec for:
- sandbox behavior
- recursive execution behavior
- tool access behavior
- reproducibility and audit behavior

## Scope

In scope:
- runtime loop and state machine
- recursion limits and quotas
- read-only tool interface for trace/policy/config evidence
- output handoff and run artifact requirements

Out of scope:
- model training and fine-tuning strategy
- UI design
- external connector implementation details

## Normative References

- `rlm/2512.24601v2.pdf`
- `https://kmad.ai/Recursive-Language-Models-Security-Audit`
- `API.md`
- `DESIGN.md`
- `specs/formal_contracts.md`
- `specs/rlm_engines.md`

## Runtime Model

The runtime is a scaffold around a base model and must execute as:

1. Initialize runtime state with input payload and metadata only.
2. Provide bounded read-only tools to the model.
3. Run iterative reasoning turns.
4. Allow recursive sub-calls under explicit depth and quota limits.
5. Build final output as a structured artifact variable.
6. Validate schema.
7. Persist run artifact and write Phoenix annotations.

The runtime must treat trace/context/policy content as external environment data, not as one monolithic prompt blob.

## Runtime State

Minimum runtime state object:

```json
{
  "run_id": "string",
  "engine_type": "rca|policy_compliance|incident_dossier",
  "status": "initialized|running|succeeded|failed|partial|terminated_budget",
  "input_ref": {
    "project_name": "string|null",
    "trace_ids": ["string"],
    "time_window": {"start": "RFC3339|null", "end": "RFC3339|null"},
    "filter_expr": "string|null",
    "controls_version": "string|null"
  },
  "budget": {
    "max_iterations": "integer",
    "max_depth": "integer",
    "max_tool_calls": "integer",
    "max_subcalls": "integer",
    "max_tokens_total": "integer",
    "max_cost_usd": "number|null",
    "sampling_seed": "integer|null",
    "max_wall_time_sec": "integer"
  },
  "counters": {
    "iteration": "integer",
    "depth": "integer",
    "tool_calls_total": "integer",
    "tokens_total": "integer"
  },
  "artifacts": {
    "intermediate_evidence_ids": ["string"],
    "final_output": {"type": "object|null"}
  }
}
```

## Runtime State Machine

Allowed transitions:

- `initialized -> running`
- `running -> succeeded`
- `running -> failed`
- `running -> partial`
- `running -> terminated_budget`
- `terminated_budget -> partial`
- `terminated_budget -> failed`

Invalid transitions must hard-fail the run.

## Sandbox Contract

Required restrictions:

- No network access from runtime code execution.
- No filesystem read/write from runtime code execution.
- No subprocess execution.
- No dynamic imports outside allowlisted runtime modules.
- No mutable side-effect tools.

Allowed behavior:

- Call allowlisted read-only tool functions from `API.md`.
- Store intermediate variables in runtime memory.
- Emit bounded logs and bounded explanations.

If a sandbox violation occurs:

- Set status to `failed`.
- Emit standardized error code `SANDBOX_VIOLATION`.
- Persist run artifact with violation details.

## Tool Contract

The runtime must only expose read-only tools.

Required tool groups:

- Trace tools:
  - `list_traces`
  - `get_spans`/`list_spans`
  - `get_span`
  - `get_children`
  - `get_messages`
  - `search_trace`
- Evidence tools:
  - `get_tool_io`
  - `get_retrieval_chunks`
  - `search`
- Policy tools:
  - `list_controls`
  - `required_evidence`

Rules:

- Every tool call must be logged in runtime trace with arguments hash and response hash.
- Tool responses must be normalized to stable ordering where defined.
- Tool errors must never be silently swallowed.

## Recursion Contract

The runtime must support recursive sub-calls with explicit limits.

Required limits:

- `max_depth`: default 2
- `max_iterations`: default 40
- `max_tool_calls`: default 200
- `max_subcalls`: default 80
- `max_tokens_total`: default 300000
- `max_cost_usd`: optional hard cap
- `max_wall_time_sec`: default 180

If any limit is reached:

- runtime enters `terminated_budget`
- engine attempts best-effort finalization
- final status must be `partial` or `failed`

Required recursion metadata per sub-call:

```json
{
  "parent_call_id": "string",
  "call_id": "string",
  "depth": "integer",
  "objective": "string",
  "input_ref_hash": "string",
  "started_at": "RFC3339",
  "completed_at": "RFC3339",
  "status": "succeeded|failed|terminated_budget"
}
```

## Determinism and Reproducibility

The runtime must produce replayable outcomes on a fixed dataset and config.

Required controls:

- stable sort rules for all list-like outputs
- fixed model parameters for production runs
- recorded prompt/template hash
- recorded tool argument hashes
- recorded controls library version
- recorded dataset id/hash
- recorded budget knobs (including seed/cost caps)

Non-deterministic fields must be isolated:

- timestamps
- run ids

## Output and Validation Contract

The runtime must output only schema-valid artifacts:

- RCA: `RCAReport`
- Policy compliance: `ComplianceReport`
- Incident dossier: `IncidentDossier`

Before completion:

1. validate engine-specific output schema
2. validate evidence pointers exist in inspected context
3. validate run artifact schema

If validation fails:

- set status `failed` or `partial`
- emit `SCHEMA_VALIDATION_FAILED` or `EVIDENCE_VALIDATION_FAILED`

## Run Artifact Contract

Every invocation must persist:

- `artifacts/investigator_runs/<run_id>/run_record.json`

The artifact must include:

- runtime budget and counters
- execution status
- output ref or error block
- write-back refs
- sub-call summary

Contract shape is defined in `specs/formal_contracts.md`.

## Phoenix Write-back Contract

The runtime must write back:

- one root annotation for overall result
- span-level annotations for evidence-linked findings

Each annotation payload must include:

- `run_id`
- `engine_type`
- `schema_version`
- `name`
- `annotator_kind`

## Error Taxonomy

Standard runtime-level error codes:

- `SANDBOX_VIOLATION`
- `TOOL_CALL_FAILED`
- `TOOL_RESPONSE_INVALID`
- `RECURSION_LIMIT_REACHED`
- `WALL_TIME_LIMIT_REACHED`
- `SCHEMA_VALIDATION_FAILED`
- `EVIDENCE_VALIDATION_FAILED`
- `MODEL_OUTPUT_INVALID`
- `WRITEBACK_FAILED`
- `UNEXPECTED_RUNTIME_ERROR`

All failures must be persisted with:

- `error.code`
- `error.message`
- `error.stage`
- `error.retryable`

## Required Metrics

Each run must collect:

- total latency
- model latency
- tool latency (aggregate and p95)
- tool call count
- recursion depth reached
- tokens in/out
- write-back success/failure count

## Security Requirements

- Prompt injection resistance:
  - treat tool output as untrusted data
  - never execute tool output as code
- Data minimization:
  - pass only required fields to sub-calls
- Evidence integrity:
  - persist excerpt hashes, not mutable raw snippets, in final evidence bundle

## Acceptance Checks

Must pass before runtime is considered compliant:

1. Sandbox test: filesystem/network calls are blocked.
2. Budget test: forced recursion overflow yields `partial` or `failed`, never hang.
3. Determinism test: two runs on same dataset/config produce equivalent structured output.
4. Validation test: malformed model output fails schema gate.
5. Audit test: run artifact includes required fields and tool/sub-call summary.

## Keywords

rlm runtime, recursive loop, sandbox, tool contract, recursion budget, run record, deterministic evaluation, evidence audit
