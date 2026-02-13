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
- `https://www.primeintellect.ai/blog/rlm`
- `https://www.dbreunig.com/2026/02/09/the-potential-of-rlms.html`
- `API.md`
- `DESIGN.md`
- `specs/formal_contracts.md`
- `specs/rlm_engines.md`
- `execplan/phase10/RLM_RCA_ARCHITECTURE.md`

## Runtime Model

The runtime is a **REPL-primary** scaffold around a base model. The REPL (Read-Eval-Print Loop)
is the default and primary execution mode for all engines. Deterministic logic (hot-span narrowing,
pattern matching) runs as a pre-filter step **inside** the REPL context, not as a separate mode.

Execution sequence:

1. Initialize runtime state with input payload and metadata only.
2. Run deterministic pre-filter (hot-span narrowing, error/exception/latency sorting) and package
   results as the REPL's initial context.
3. Start the REPL loop: provide the model with persistent execution state, bounded read-only tools,
   and a Python analysis sandbox.
4. The model iteratively writes code to explore, calls tools, and accumulates evidence.
5. Allow recursive sub-calls under explicit depth and quota limits (per-hypothesis decomposition
   for RCA; per-control for compliance; per-trace for incident).
6. Build final output as a structured artifact variable (via SUBMIT action).
7. Validate output against engine-specific schema.
8. Persist run artifact and write Phoenix annotations.

The runtime must treat trace/context/policy content as external environment data, not as one
monolithic prompt blob. Large context lives as program state outside the model's token window;
the model selectively pulls slices into tokens via code and tool calls.

### Model Configuration

- Default model for all calls (root and sub-calls): `gpt-4o-mini`
- Optional upgrade for root synthesis: `gpt-4o` or `gpt-5.2`
- Temperature: `0.0` (maximize determinism)
- Output format: JSON schema mode (structured generation)
- Reasoning effort: `minimal` for gpt-5 family models

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

### Execution Model

Model-generated code runs in a **local subprocess** with restricted imports. The subprocess
is the security boundary between the RLM's generated code and the host system.

- The REPL harness spawns a Python subprocess per execution turn.
- The subprocess has a custom import hook that blocks dangerous modules.
- Tool calls from generated code are proxied through the parent process via `ToolRegistry`.
- REPL output (stdout/stderr) is captured and **truncated to 8192 characters per turn**.
- Each code execution turn has a **30-second timeout**. Runaway code is killed.

### Import Blocklist

The following modules are **blocked** in the subprocess (attempting to import raises `ImportError`):

```
os, subprocess, socket, http, urllib, pathlib, shutil, signal,
ctypes, importlib, sys, multiprocessing, threading
```

### Allowed Analysis Modules

The following modules are **allowed** for data analysis inside the REPL:

```
json, re, math, statistics, collections, itertools, functools,
operator, datetime, dataclasses, typing, copy, textwrap, hashlib
```

### Required Restrictions

- No network access from runtime code execution.
- No filesystem read/write from runtime code execution.
- No spawning child processes from generated code (the subprocess itself is the sandbox;
  code within it cannot escape to spawn further processes).
- No imports outside the allowed analysis modules.
- No mutable side-effect tools.

### Allowed Behavior

- Call allowlisted read-only tool functions from `API.md` (routed through `ToolRegistry`).
- Write Python code to filter, aggregate, group, and transform tool results.
- Store intermediate variables in runtime memory.
- Emit bounded logs and bounded explanations (truncated per turn).

### Violation Handling

If a sandbox violation occurs:

- Set status to `failed`.
- Emit standardized error code `SANDBOX_VIOLATION`.
- Persist run artifact with violation details.
- No partial output emitted for sandbox violations.

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

### Budget Defaults

These are **contract-level upper bounds**. Engines may set tighter per-engine defaults
(see `specs/rlm_engines.md`), but must not exceed contract limits.

| Knob | Contract max | RCA default | Compliance default | Incident default |
|------|-------------|-------------|-------------------|-----------------|
| `max_depth` | 3 | 2 | 2 | 3 |
| `max_iterations` | 60 | 40 | 40 | 60 |
| `max_tool_calls` | 220 | 120 | 160 | 220 |
| `max_subcalls` | 90 | 40 | 60 | 90 |
| `max_tokens_total` | 320000 | 200000 | 260000 | 320000 |
| `max_cost_usd` | optional | optional | optional | optional |
| `max_wall_time_sec` | 300 | 180 | 180 | 300 |

Override policy:
- CLI invocations may override any knob via flags.
- Overrides must not exceed contract-level maximums.
- All active budget values (including overrides) must be recorded in the RunRecord.

### Recursion Strategy

The default recursion strategy is **per-hypothesis decomposition** (for RCA):

1. Root REPL identifies 1–4 candidate failure hypotheses from hot-span analysis.
2. Root spawns one `delegate_subcall` per hypothesis, each with:
   - a focused objective (the hypothesis statement)
   - a filtered context (only relevant span IDs and their branches)
   - shared global budget counters
3. Each sub-call explores evidence independently via tools + analysis code.
4. Each sub-call returns: `{label, confidence, evidence_refs, gaps}`.
5. Root synthesizes sub-call results into the final output.

Other engines use different decomposition strategies (per-control for compliance,
per-trace for incident) but the budget and depth mechanics are identical.

### Budget Enforcement

If any limit is reached:

- runtime enters `terminated_budget`
- engine attempts best-effort finalization with evidence gathered so far
- final status must be `partial` or `failed`
- at 90% of `max_wall_time_sec`, the REPL forces finalization

### Required Recursion Metadata

Per sub-call:

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

1. Sandbox test: filesystem/network calls are blocked (import blocklist enforced).
2. Sandbox test: blocked modules (`os`, `subprocess`, `socket`, etc.) raise `ImportError`.
3. Sandbox test: REPL output truncated at 8192 chars per turn.
4. Sandbox test: per-turn timeout kills runaway code after 30s.
5. Budget test: forced recursion overflow yields `partial` or `failed`, never hang.
6. Budget test: 90% wall-time triggers forced finalization.
7. Determinism test: two runs on same dataset/config produce equivalent structured output.
8. Validation test: malformed model output fails schema gate.
9. Audit test: run artifact includes required fields and tool/sub-call summary.
10. REPL test: tool calls routed through ToolRegistry (not direct API access).

## REPL Execution Model

The REPL is the primary execution environment for all engines. It is a custom-built Python
loop (not DSPy) that gives the model a persistent execution state, tool access, and data
analysis capabilities.

### What the Model Can Do Per REPL Turn

1. **Write Python analysis code**: filter dataframes, compute statistics, group spans by
   attribute, find patterns, format evidence summaries. Executed in the subprocess sandbox.
2. **Call Inspection API tools**: routed through `ToolRegistry` which enforces the allowlist
   and logs argument/response hashes.
3. **Spawn recursive sub-calls**: `delegate_subcall(objective="...", context={...})` creates
   an isolated sub-instance at `depth+1`.
4. **Update working state**: store intermediate findings, accumulate evidence pointers.
5. **SUBMIT**: end the loop and return the final structured output.

### REPL State (Persistent Across Turns)

- `variables`: dict of named values (tool results, filtered slices, intermediate computations)
- `evidence_refs`: accumulated evidence pointers
- `hypothesis_candidates`: list of candidate hypotheses (for per-hypothesis recursion)
- `tool_trace`: call log with argument hashes and response hashes
- `budget_counters`: global counters shared across root + sub-calls

### Fallback Behavior

- If the REPL fails or budget is exhausted before SUBMIT, the engine falls back to a
  deterministic report using the pre-filter results and any evidence gathered so far.
- The fallback report has `annotator_kind=CODE` (not `LLM`).

## Keywords

rlm runtime, repl-primary, recursive loop, subprocess sandbox, import blocklist, tool contract, recursion budget, per-hypothesis, run record, deterministic evaluation, evidence audit
