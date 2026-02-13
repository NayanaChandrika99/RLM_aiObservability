# RLM-RCA System Architecture

> Autonomous Root Cause Analysis for AI observability using Recursive Language Models

**Status**: Design-locked. Ready for implementation.
**Scope**: Trace RCA engine running against Phoenix traces from a LlamaIndex agent.
**Normative companions**: `specs/rlm_runtime_contract.md`, `API.md`, `DESIGN.md`, `ARCHITECTURE.md`

---

## 1. What is an RLM and Why Use It for RCA

A Recursive Language Model (RLM) is a deployment pattern where an LLM operates inside a persistent
execution environment (typically a Python REPL) with access to tools, and can spawn recursive
sub-calls of itself over partitioned subsets of the problem.

Key properties (from Prime Intellect, DSPy, and the RLM paper):

- **Large context stays external**: trace data, logs, metrics live as program state outside the
  model's token window. The model selectively pulls slices into tokens via code and tool calls.
- **Code as the reasoning medium**: the model writes Python to filter, aggregate, and transform
  data, then uses tool calls to fetch more detail on interesting subsets.
- **Recursive decomposition**: the model spawns sub-instances of itself, each with a focused
  slice of the problem and an isolated context. Sub-calls return structured, compact results.
- **Bounded execution**: iteration caps, recursion depth limits, wall-time budgets, and tool-call
  quotas prevent runaway cost and latency.

**Why RLM fits RCA specifically**:
- RCA over traces is fundamentally an exploration problem: the agent must decide *what* to
  inspect and *how deeply*, rather than being given a fixed prompt with all context.
- Trace datasets can be large (hundreds of spans, kilobytes of tool I/O per span). An RLM
  can programmatically narrow to hot spans before using tokens for synthesis.
- Competing hypotheses benefit from parallel, independent investigation — the natural unit
  of recursion is "one hypothesis, one sub-call."

### References
- [Prime Intellect RLM blog](https://www.primeintellect.ai/blog/rlm)
- [Alex Zhang RLM explainer](https://alexzhang13.github.io/blog/2025/rlm/)
- [Kevin Madura — RLM Security Audit (DSPy)](https://kmad.ai/Recursive-Language-Models-Security-Audit)
- [Drew Breunig — The Potential of RLMs](https://www.dbreunig.com/2026/02/09/the-potential-of-rlms.html)

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Phoenix-RLM Investigator                      │
│                                                                       │
│  ┌──────────────┐     ┌───────────────────────────────────────────┐  │
│  │  LlamaIndex   │     │              RLM-RCA Engine               │  │
│  │  Agent + Fault│     │                                           │  │
│  │  Injector     │     │  ┌────────────────┐  ┌────────────────┐  │  │
│  │               │     │  │ Deterministic   │  │ REPL Harness   │  │  │
│  │  (generates   │     │  │ Pre-filter      │  │ (Python exec   │  │  │
│  │   traces)     │     │  │ (hot-span       │  │  + tool calls  │  │  │
│  └──────┬───────┘     │  │  narrowing)     │  │  + data        │  │  │
│         │              │  └───────┬────────┘  │  analysis)      │  │  │
│         │ OTLP/HTTP    │          │            └───────┬────────┘  │  │
│         ▼              │          ▼                    │            │  │
│  ┌──────────────┐     │  ┌────────────────┐          │            │  │
│  │              │     │  │ Hypothesis      │◄─────────┘            │  │
│  │   Phoenix    │◄────┤  │ Generator       │                       │  │
│  │   Server     │     │  └───────┬────────┘                       │  │
│  │              │     │          │                                  │  │
│  │  (traces +   │     │          ▼  spawn per hypothesis           │  │
│  │   evals/     │     │  ┌────────────────┐                       │  │
│  │   annots)    │     │  │ Sub-call Pool   │                       │  │
│  │              │     │  │ ┌────┐ ┌────┐   │                       │  │
│  └──────────────┘     │  │ │ H1 │ │ H2 │   │  (each gets filtered │  │
│         ▲              │  │ └────┘ └────┘   │   span slice +       │  │
│         │              │  │ ┌────┐          │   hypothesis to       │  │
│         │ writeback    │  │ │ H3 │          │   investigate)        │  │
│         │ (annots)     │  │ └────┘          │                       │  │
│         │              │  └───────┬────────┘                       │  │
│         │              │          │                                  │  │
│         │              │          ▼                                  │  │
│         │              │  ┌────────────────┐                       │  │
│         │              │  │ Root Synthesis  │                       │  │
│         │              │  │ (pick winner,   │                       │  │
│         │              │  │  cite evidence, │                       │  │
│         │              │  │  emit RCAReport)│                       │  │
│         │              │  └───────┬────────┘                       │  │
│         │              │          │                                  │  │
│         │              │          ▼                                  │  │
│         │              │  ┌────────────────┐  ┌────────────────┐  │  │
│         └──────────────┤  │ Writeback       │  │ RunRecord      │  │  │
│                        │  │ (Phoenix annots)│  │ (JSON artifact)│  │  │
│                        │  └────────────────┘  └────────────────┘  │  │
│                        └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Trace Generation Layer

**Component**: `apps/demo_agent/` (LlamaIndex tutorial agent + fault injector)

Generates OpenTelemetry traces and sends them to Phoenix via OTLP/HTTP.

| Aspect | Detail |
|--------|--------|
| Agent framework | LlamaIndex (from `phoenix/tutorials/tracing/llama_index_openai_agent_tracing_tutorial.ipynb`) |
| Trace protocol | OTLP/HTTP via `phoenix.otel.register(protocol="http/protobuf")` |
| Collector endpoint | `PHOENIX_COLLECTOR_ENDPOINT=http://127.0.0.1:6006` |
| OTLP traces path | `http://127.0.0.1:6006/v1/traces` |
| Fault injection | Programmatic injection of 5 failure classes into agent runs |
| Output | 30 traces with known failure labels, mapped in `datasets/seeded_failures/manifest.json` |

**Failure injection profiles** (one per RCA taxonomy label):

| Profile | Injection Method | Expected Label |
|---------|-----------------|----------------|
| `profile_tool_failure` | Force tool exception, timeout, or wrong tool selection | `tool_failure` |
| `profile_retrieval_failure` | Return irrelevant documents, empty retrieval, or wrong index | `retrieval_failure` |
| `profile_instruction_failure` | Corrupt system prompt, inject format drift, break schema compliance | `instruction_failure` |
| `profile_upstream_dependency_failure` | Mock API 500/429/timeout on external calls | `upstream_dependency_failure` |
| `profile_data_schema_mismatch` | Return malformed JSON from tools, break expected output schema | `data_schema_mismatch` |

### 3.2 Inspection API (Read Layer)

**Component**: `investigator/inspection_api/`

Read-only interface between the RLM engine and Phoenix trace data. All trace access goes
through this layer — the RLM never queries Phoenix directly.

**Implementations**:
- `PhoenixInspectionAPI` — live Phoenix server via `phoenix.session.Client`
- `ParquetClient` — offline Parquet files for reproducible evaluation

**Tool surface exposed to the RLM** (16 functions):

| Tool | Purpose | Return type |
|------|---------|-------------|
| `list_traces(project, time_window, filter)` | Find traces matching criteria | `list[TraceSummary]` |
| `list_spans(trace_id)` / `get_spans(trace_id, type)` | All spans in a trace | `list[SpanSummary]` |
| `get_span(span_id)` | Full span detail with attributes + events | `SpanDetail` |
| `get_children(span_id)` | Child spans of a parent | `list[SpanSummary]` |
| `get_messages(span_id)` | LLM messages (input/output) for a span | `list[Message]` |
| `get_tool_io(span_id)` | Tool name + input/output for a tool span | `ToolIO` |
| `get_retrieval_chunks(span_id)` | Retrieved documents with scores | `list[RetrievalChunk]` |
| `search_trace(trace_id, pattern)` | Regex search across span attributes | `list[SearchHit]` |
| `search(text, pattern)` | Regex search over arbitrary text/chunks | `list[str]` |
| `list_controls(version, ...)` | Policy controls (compliance engine) | `list[PolicyControl]` |
| `required_evidence(control_id, version)` | Evidence rules for a control | `list[str]` |
| `list_config_snapshots(project, ...)` | Config snapshots (incident engine) | `list[ConfigSnapshot]` |
| `get_config_snapshot(snapshot_id)` | Single config snapshot | `ConfigSnapshot` |
| `get_config_diff(base, target)` | Diff between two snapshots | `ConfigDiff` |

**Determinism guarantees**:
- All list outputs sorted by `start_time` then `span_id`
- Tool arguments and responses are SHA256-hashed for audit trail
- Argument sanitization: only parameters in the method signature are passed through

### 3.3 RLM-RCA Engine (Core)

**Component**: `investigator/rca/engine.py` + `investigator/runtime/`

This is the core autonomous investigation system. A single CLI invocation triggers a
multi-step, tool-using, recursion-capable analysis of a trace.

#### Execution Flow

```
CLI invocation (trace_id or batch)
       │
       ▼
┌─────────────────────────────────┐
│  1. DETERMINISTIC PRE-FILTER    │  No LLM calls. Pure code.
│                                  │
│  Sort all spans by priority:    │
│    1st: status_code == ERROR    │
│    2nd: exception events        │
│    3rd: highest latency         │
│    Tie-break: span_id asc       │
│                                  │
│  Select top-K "hot spans"       │
│  (default K=5)                  │
│                                  │
│  For each hot span, collect     │
│  branch (BFS, depth≤2,         │
│  nodes≤30): parent/child/       │
│  sibling spans                  │
└─────────┬───────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│  2. RLM REPL LOOP (root)       │  LLM + code execution.
│                                  │
│  Model receives:                │
│    - Hot-span summary           │
│    - Tool function descriptions │
│    - Analysis sandbox           │
│    - Instruction: "investigate  │
│      this trace, identify       │
│      failure hypotheses"        │
│                                  │
│  Model can:                     │
│    - Write Python to analyze    │
│      (filter, group, compute)   │
│    - Call Inspection API tools  │
│    - Read tool I/O, messages,   │
│      retrieval chunks           │
│                                  │
│  Loop until:                    │
│    - Model identifies candidate │
│      hypotheses, OR             │
│    - Budget exhausted           │
└─────────┬───────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│  3. PER-HYPOTHESIS SUB-CALLS   │  Recursive RLM invocations.
│                                  │
│  For each candidate hypothesis: │
│    - Spawn sub-call at depth+1  │
│    - Sub-call gets:             │
│      • filtered span slice      │
│      • hypothesis statement     │
│      • evidence-gathering goal  │
│    - Sub-call explores via      │
│      tools + analysis code      │
│    - Sub-call returns:          │
│      • label + confidence       │
│      • evidence_refs[]          │
│      • gaps[]                   │
│                                  │
│  Budget is shared across all    │
│  sub-calls (global counters).   │
└─────────┬───────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│  4. ROOT SYNTHESIS              │  LLM judgment on sub-results.
│                                  │
│  Root model receives sub-call   │
│  results and:                   │
│    - Picks primary_label        │
│      (highest evidence support) │
│    - Records competing          │
│      hypotheses + why rejected  │
│    - Computes confidence        │
│      (base + evidence bonus)    │
│    - Generates remediation      │
│    - Lists gaps                 │
│                                  │
│  Emits: RCAReport (structured   │
│  JSON, schema-validated)        │
└─────────┬───────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│  5. OUTPUT + WRITEBACK          │
│                                  │
│  a) Persist run_record.json     │
│     (artifacts/investigator_    │
│      runs/<run_id>/)            │
│                                  │
│  b) Write to Phoenix:           │
│     - Trace-level: rca.primary  │
│       (label + confidence)      │
│     - Span-level: rca.evidence  │
│       (per evidence pointer)    │
│                                  │
│  c) Include run_id in all       │
│     annotations for traceability│
└─────────────────────────────────┘
```

### 3.4 REPL Harness (Custom)

**Component**: `investigator/runtime/repl_loop.py`

The REPL harness is the core execution environment for the RLM. It is a custom-built
Python loop (not DSPy, not a framework) that gives the model a persistent execution
state, tool access, and data analysis capabilities.

**Design** (inspired by Prime Intellect's RLM implementation):

```
┌──────────────────────────────────────────────────┐
│                REPL Harness                       │
│                                                    │
│  ┌──────────────┐    ┌──────────────────────┐    │
│  │  Model Call   │    │  Code Execution       │    │
│  │  (structured  │───▶│  (subprocess with     │    │
│  │   generation) │    │   import blocklist)   │    │
│  └──────┬───────┘    └──────────┬───────────┘    │
│         │                       │                  │
│         │ reasoning +           │ stdout/stderr    │
│         │ code block            │ (capped)         │
│         │                       │                  │
│         ▼                       ▼                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Persistent State                         │    │
│  │  - variables dict (tool results, slices)  │    │
│  │  - evidence_refs[] (accumulated)          │    │
│  │  - hypothesis_candidates[]                │    │
│  │  - tool_trace[] (call log with hashes)    │    │
│  │  - budget counters                        │    │
│  └──────────────────────────────────────────┘    │
│                                                    │
│  Loop terminates when:                             │
│  - Model calls SUBMIT(final_output)                │
│  - Budget limit reached (iterations/tools/time)    │
│  - Unrecoverable error                             │
└──────────────────────────────────────────────────┘
```

**What the model can do in each REPL turn**:

1. **Write Python analysis code**: filter dataframes, compute statistics, group spans by
   attribute, find patterns, format evidence summaries. Executed in a subprocess with
   restricted imports.

2. **Call Inspection API tools**: `get_span(span_id)`, `get_tool_io(span_id)`,
   `search_trace(trace_id, "error")`, etc. Routed through `ToolRegistry` which enforces
   the allowlist and logs argument/response hashes.

3. **Spawn recursive sub-calls**: `delegate_subcall(objective="...", context={...})`.
   Creates an isolated sub-instance at `depth+1` with its own REPL loop and a filtered
   slice of the trace data.

4. **Update working state**: store intermediate findings, accumulate evidence pointers,
   refine hypotheses.

5. **SUBMIT**: end the loop and return the final structured output.

**What the model cannot do** (enforced by sandbox):

- Import `os`, `subprocess`, `socket`, `http`, `urllib`, `pathlib`, `shutil`, `signal`
- Access the filesystem or network
- Execute shell commands
- Import packages not in the allowlist
- Call tools not in `DEFAULT_ALLOWED_TOOLS`

### 3.5 Sandbox and Security

**Component**: `investigator/runtime/sandbox.py`

The sandbox enforces the security boundary between the RLM's generated code and the
host system.

**Layers of protection**:

| Layer | Mechanism | What it prevents |
|-------|-----------|------------------|
| Import blocklist | Block `os`, `subprocess`, `socket`, `http`, `urllib`, `pathlib`, `shutil`, `signal`, `ctypes` | File/network/process access |
| Action type allowlist | Only `tool_call`, `delegate_subcall`, `synthesize`, `finalize` | Arbitrary action injection |
| Tool allowlist | Only 16 Inspection API functions | Unauthorized data access |
| JSON-safety validation | All action args must be JSON-serializable primitives | Code injection via complex objects |
| Argument sanitization | `ToolRegistry` strips unknown kwargs | Parameter injection |
| Output cap | REPL stdout/stderr truncated per turn | Token flooding |

**Sandbox violation handling**:
- Raises `SandboxViolationError`
- Run status set to `failed`
- Error code: `SANDBOX_VIOLATION`
- Run record persisted with violation details
- No partial output emitted

### 3.6 Budget and Resource Management

**Component**: `investigator/runtime/contracts.py` (RuntimeBudget, RuntimeUsage)

Every RLM run operates under a hard budget. Budgets are global — shared across the root
loop and all recursive sub-calls.

**Default budget knobs**:

| Knob | Default | Purpose |
|------|---------|---------|
| `max_iterations` | 40 | Total REPL turns (root + sub-calls) |
| `max_depth` | 2 | Maximum recursion depth |
| `max_tool_calls` | 120 | Total Inspection API calls |
| `max_subcalls` | 40 | Total recursive sub-call spawns |
| `max_tokens_total` | 200,000 | Total LLM tokens (in + out) |
| `max_cost_usd` | None (soft $5/day) | Optional hard cost cap |
| `max_wall_time_sec` | 180 | Wall-clock timeout |

**Budget enforcement behavior**:
- At 90% of `max_wall_time_sec`, the REPL forces finalization
- When any limit is reached, status transitions to `terminated_budget`
- Engine attempts best-effort synthesis with evidence gathered so far
- Final status is `partial` (usable output) or `failed` (insufficient evidence)
- Run record always persisted, including budget exhaustion reason

**Cost estimation** (per model):

| Model | Input ($/M tokens) | Output ($/M tokens) | Typical RCA run cost |
|-------|--------------------|--------------------|---------------------|
| gpt-4o-mini | $0.25 | $2.00 | ~$0.01–0.05 |
| gpt-4o | $2.00 | $8.00 | ~$0.05–0.30 |

### 3.7 Recursion Strategy: Per-Hypothesis Decomposition

The RLM uses **per-hypothesis recursion**: the root model identifies candidate failure
modes, then spawns one sub-call per hypothesis to gather evidence independently.

```
Root RLM
├── Explores hot spans, identifies 2-3 candidate hypotheses
├── Hypothesis 1: "tool_failure — the search tool returned a timeout"
│   └── Sub-call 1 (depth=1)
│       ├── Gets: tool spans only + parent/child context
│       ├── Calls: get_tool_io, get_span, get_children
│       └── Returns: {label, confidence, evidence_refs, gaps}
├── Hypothesis 2: "retrieval_failure — irrelevant documents returned"
│   └── Sub-call 2 (depth=1)
│       ├── Gets: retriever spans + retrieval chunks
│       ├── Calls: get_retrieval_chunks, get_span, search_trace
│       └── Returns: {label, confidence, evidence_refs, gaps}
└── Synthesis: compare sub-call results, pick winner, emit RCAReport
```

**Why per-hypothesis (vs per-service or on-low-confidence)**:
- The tutorial agent is single-service, so per-service decomposition doesn't help
- Per-hypothesis gives independent, parallel evidence gathering for competing explanations
- Matches Prime Intellect's "parallel deep dives" pattern
- Each sub-call has a focused objective, reducing wasted tool calls

**Sub-call metadata** (recorded per sub-call):

```json
{
  "parent_call_id": "root",
  "call_id": "subcall_001",
  "depth": 1,
  "objective": "Investigate tool_failure hypothesis: search tool timeout",
  "input_ref_hash": "sha256:...",
  "started_at": "2026-02-12T10:00:00Z",
  "completed_at": "2026-02-12T10:00:12Z",
  "status": "succeeded"
}
```

### 3.8 Evidence Model

Every claim in an RCA report must cite evidence using the canonical `evidence_ref` shape:

```json
{
  "trace_id": "abc123",
  "span_id": "span_456",
  "kind": "TOOL_IO",
  "ref": "tool:span_456",
  "excerpt_hash": "sha256:...",
  "ts": "2026-02-12T10:00:00Z"
}
```

**Evidence kinds**:

| Kind | What it points to | When used |
|------|-------------------|-----------|
| `SPAN` | A span summary (status, latency, name) | Hot span identification |
| `TOOL_IO` | Tool input/output content | Tool failure analysis |
| `RETRIEVAL_CHUNK` | Retrieved document with score | Retrieval failure analysis |
| `MESSAGE` | LLM input/output message | Instruction failure analysis |
| `CONFIG_DIFF` | Config change between snapshots | Incident/deploy correlation |

**Evidence policy**:
- Minimum 1 evidence ref for low confidence (<0.5)
- Minimum 2 independent evidence refs for medium/high confidence (>=0.5)
- `excerpt_hash` is a SHA256 of the evidence text — ensures integrity without storing
  sensitive content in the report
- All evidence refs are validated against the inspected trace before the report is emitted

### 3.9 Output Contracts

**RCAReport** (per-trace):

```json
{
  "schema_version": "1.0.0",
  "trace_id": "...",
  "primary_label": "tool_failure",
  "summary": "The search tool timed out after 30s...",
  "confidence": 0.76,
  "evidence_refs": [...],
  "remediation": ["Add retry with exponential backoff to search tool", ...],
  "gaps": ["Could not inspect upstream service logs"]
}
```

**RunRecord** (per-invocation):

```json
{
  "run_id": "...",
  "run_type": "rca",
  "status": "succeeded",
  "started_at": "...",
  "completed_at": "...",
  "dataset_ref": { "dataset_id": "seeded_failures_v1", "dataset_hash": "..." },
  "input_ref": { "trace_ids": ["..."], ... },
  "runtime_ref": {
    "engine_version": "0.2.0",
    "model_name": "gpt-4o-mini",
    "prompt_template_hash": "sha256:...",
    "budget": { ... },
    "usage": { "iterations": 12, "tool_calls": 34, "depth_reached": 1, ... },
    "subcall_metadata": [...]
  },
  "output_ref": { "artifact_path": "...", "schema_version": "1.0.0" },
  "writeback_ref": { "annotation_names": ["rca.primary", "rca.evidence"], ... }
}
```

### 3.10 Writeback to Phoenix

Results are written back as Phoenix annotations so they appear in the trace UI:

| Annotation | Level | Content | `annotator_kind` |
|------------|-------|---------|-------------------|
| `rca.primary` | Trace (root span) | label + confidence + full report JSON | `LLM` (or `CODE` if deterministic-only) |
| `rca.evidence` | Span (per evidence span) | evidence kind + weight + pointer | `CODE` |

Both annotations include `run_id` so the UI can link back to the full run record.

---

## 4. Model Configuration

**Locked decisions**:

| Setting | Value | Rationale |
|---------|-------|-----------|
| Model (all calls) | `gpt-4o-mini` | Cost-efficient for dev/eval. Single model everywhere simplifies debugging. |
| Temperature | 0.0 | Maximize determinism for reproducible RCA |
| Structured output | JSON schema mode | Guarantees parseable output |
| Upgrade path | Swap root model to `gpt-4o` or `gpt-5.2` | One config change when workflow is stable |

---

## 5. Directory Layout

```
rlm_observability/
├── apps/
│   └── demo_agent/
│       ├── phase1_langgraph_runner.py    # Existing agent runner
│       ├── phase1_tutorial_run.py        # Existing tutorial run
│       └── fault_injector.py             # NEW: failure injection harness
├── investigator/
│   ├── inspection_api/
│   │   ├── protocol.py                   # Stable read-only interface
│   │   └── phoenix_client.py             # Phoenix-backed implementation
│   ├── runtime/
│   │   ├── contracts.py                  # Dataclasses: RCAReport, RunRecord, etc.
│   │   ├── repl_loop.py                  # REPL harness (iterative code exec)
│   │   ├── recursive_loop.py             # Recursive action executor
│   │   ├── recursive_planner.py          # Action planner for recursive mode
│   │   ├── sandbox.py                    # Sandbox enforcement
│   │   ├── tool_registry.py             # Allowlisted tool dispatch
│   │   ├── llm_client.py                # OpenAI structured generation client
│   │   └── prompt_registry.py           # Prompt template loader + hashing
│   ├── rca/
│   │   ├── engine.py                     # TraceRCAEngine (core)
│   │   ├── workflow.py                   # End-to-end orchestration
│   │   ├── writeback.py                  # Phoenix annotation writer
│   │   └── cli.py                        # NEW: CLI entrypoint
│   ├── prompts/
│   │   ├── rca/                          # RCA prompt templates
│   │   └── runtime/                      # REPL/recursive prompt templates
│   └── schemas/                          # JSON schemas for validation
├── datasets/
│   └── seeded_failures/
│       ├── manifest.json                 # Ground-truth labels (30 cases)
│       └── exports/                      # Parquet trace exports (gitignored)
├── artifacts/
│   └── investigator_runs/
│       └── <run_id>/
│           └── run_record.json           # Per-invocation audit artifact
├── specs/
│   ├── rlm_runtime_contract.md           # Runtime behavior contract
│   ├── rlm_engines.md                    # Engine-level behavior spec
│   └── formal_contracts.md               # Schema contracts
└── execplan/
    └── phase10/
        ├── RLM_RCA_ARCHITECTURE.md       # This document
        └── RLM_RCA_IMPLEMENTATION_PLAN.md
```

---

## 6. Data Flow (End to End)

```
1. TRACE GENERATION
   LlamaIndex agent + fault injector
       │
       │  OTLP/HTTP (http://127.0.0.1:6006/v1/traces)
       ▼
   Phoenix server (http://127.0.0.1:6006)
       │
       │  Stores spans with OpenInference attributes
       ▼
   manifest.json updated with trace_id → expected_label mapping

2. RCA INVESTIGATION
   CLI: python -m investigator.rca.cli --trace-id <id>
       │
       │  PhoenixInspectionAPI (or ParquetClient for offline)
       ▼
   Deterministic pre-filter → hot spans
       │
       ▼
   RLM REPL loop (root)
       │  Tool calls + Python analysis
       ▼
   Hypothesis generation → per-hypothesis sub-calls
       │  Each sub-call: focused tool exploration
       ▼
   Root synthesis → RCAReport (JSON)
       │
       ├──▶ run_record.json (artifacts/)
       └──▶ Phoenix annotations (rca.primary + rca.evidence)

3. EVALUATION
   Compare RCAReport.primary_label against manifest.json.expected_label
       │
       ▼
   Accuracy metrics: top-1 accuracy, per-label precision/recall
```

---

## 7. Trust Boundaries

```
┌─────────────────────────────────────────────┐
│  TRUSTED ZONE                                │
│  - CLI invocation                            │
│  - Phoenix server                            │
│  - Inspection API implementation             │
│  - ToolRegistry (allowlist enforcement)      │
│  - SandboxGuard (action validation)          │
│  - RunRecord persistence                     │
│  - Writeback to Phoenix                      │
├─────────────────────────────────────────────┤
│  UNTRUSTED ZONE (sandboxed)                  │
│  - LLM-generated code (REPL execution)       │
│  - LLM-generated actions (tool_call, etc)    │
│  - LLM-generated hypotheses and evidence     │
│  - Sub-call objectives and context           │
│                                               │
│  Everything from LLM is treated as untrusted │
│  input. Validated before execution.          │
└─────────────────────────────────────────────┘
```

**Key invariant**: The RLM can only *observe* trace data and *produce* structured output.
It cannot modify Phoenix data, access the filesystem, make network calls, or execute
arbitrary system commands. All tool outputs are treated as untrusted data — never executed
as code by the harness.

---

## 8. Comparison with Traditional Arize RCA

| Aspect | Traditional Arize RCA | RLM-RCA (this system) |
|--------|----------------------|----------------------|
| Driven by | Human analyst clicking through UI | Autonomous RLM with tool access |
| Exploration strategy | Manual drill-down: monitor → slice → inspect | Programmatic: hot-span pre-filter → hypothesis → sub-call |
| Context handling | Fits in analyst's working memory | Large context as external state; selective token loading |
| Depth of analysis | Limited by analyst time/attention | Bounded by budget (iterations, tools, cost) |
| Output | Mental model + ad-hoc notes | Structured JSON with evidence pointers |
| Reproducibility | Not reproducible | Deterministic pre-filter + recorded run artifacts |
| Coverage | Whatever the analyst chooses to look at | Systematic: all hot spans examined, multiple hypotheses tested |
| Write-back | Manual annotations (if any) | Automatic Phoenix annotations with run traceability |

---

## 9. Future Extensions

These are explicitly **out of scope** for the current implementation but designed-for
in the architecture:

1. **Multi-model routing**: Strong root model + cheap sub-model. Config change only.
2. **Webhook trigger**: Phoenix alert → HTTP POST → RCA run. Requires a listener service.
3. **Batch/scheduled mode**: Cron job scanning recent traces. Uses same CLI under the hood.
4. **Logs/metrics correlation**: Add `get_logs()` and `get_metrics()` to Inspection API.
5. **Remote sandbox (E2B/Modal)**: Swap subprocess executor for remote sandbox client.
6. **Compliance and incident engines**: Same architecture, different prompt templates and
   tool usage patterns. Share runtime, contracts, and writeback.
