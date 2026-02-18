# RLM-RCA Implementation Plan

> Step-by-step plan to build an autonomous RCA system using Recursive Language Models
> against real Phoenix traces from a LlamaIndex agent.

**Companion document**: `RLM_RCA_ARCHITECTURE.md` (same directory)
**Current state**: ~70% of components exist. Major gaps are failure injection, RLM-primary
execution path, per-hypothesis recursion, CLI, and end-to-end validation.

---

## Design Decisions (Locked)

These were decided in the design session and must not be revisited during implementation:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RLM framework | Custom REPL harness | Full control, no framework coupling. Matches Prime Intellect. |
| Sandbox | Local subprocess + import blocklist | Simple, fast, full CPython. Single-user local dev. |
| Trigger mode | CLI command | Manual control for dev/eval. `python -m investigator.rca.cli` |
| Model routing | Same model everywhere (gpt-4o-mini) | Cost-efficient, simpler debugging. Upgrade root model later. |
| Default mode | RLM-primary | REPL loop always runs. Deterministic narrowing is a pre-filter inside it. |
| REPL scope | Tool calls + Python data analysis | Model can call tools AND write analysis code. Import blocklist enforced. |
| Recursion trigger | Per failure hypothesis | Root identifies hypotheses, spawns one sub-call per hypothesis. |
| Trace generation | Modify LlamaIndex tutorial agent | Add fault injection to generate 30 realistic failure traces. |

---

## Prerequisites

Before starting implementation:

- [ ] Phoenix installed and runnable locally (`phoenix serve`)
- [ ] OpenAI API key configured (`OPENAI_API_KEY` env var)
- [ ] `uv` installed for dependency management
- [ ] LlamaIndex tutorial notebook verified runnable
  (`phoenix/tutorials/tracing/llama_index_openai_agent_tracing_tutorial.ipynb`)

---

## Implementation Phases

The plan is organized into 6 sequential phases. Each phase produces a testable artifact.
Phases are ordered by dependency — each builds on the previous.

---

### Phase A: Fault Injector for LlamaIndex Agent

**Goal**: Generate 30 traces with known failure modes in Phoenix.

**Why first**: Everything downstream (RCA engine, evaluation, writeback) needs real traces
to work against. The manifest has 30 cases with `trace_id: null` — we need to fill those.

#### What exists
- `apps/demo_agent/phase1_langgraph_runner.py` — agent runner with Phoenix tracing setup
- `apps/demo_agent/phase1_tutorial_run.py` — tutorial execution with OTel registration
- `datasets/seeded_failures/manifest.json` — 30 cases with expected labels, no trace IDs

#### What to build

**File**: `apps/demo_agent/fault_injector.py`

A Python module that wraps the LlamaIndex agent from the tutorial and injects failures
programmatically. Each run produces one trace in Phoenix with a known failure mode.

**Fault injection strategies** (one per RCA taxonomy label):

| Profile | Implementation |
|---------|---------------|
| `profile_tool_failure` | Monkey-patch tool functions to raise exceptions, return after forced delay (timeout simulation), or select wrong tool via modified tool descriptions |
| `profile_retrieval_failure` | Replace retriever with one that returns irrelevant documents, empty results, or documents from wrong index/collection |
| `profile_instruction_failure` | Modify system prompt mid-conversation to introduce format drift, inject contradictory instructions, or break expected output schema |
| `profile_upstream_dependency_failure` | Mock external API responses with HTTP 500, 429, connection timeout, or malformed response body |
| `profile_data_schema_mismatch` | Modify tool output format to return unexpected JSON structure, wrong field names, or unparseable strings where JSON is expected |

**Injection harness interface**:

```python
# apps/demo_agent/fault_injector.py

def run_with_fault(
    *,
    fault_profile: str,           # e.g., "profile_tool_failure"
    run_id: str,                   # from manifest, e.g., "seed_run_0000"
    phoenix_endpoint: str = "http://127.0.0.1:6006",
) -> str:
    """
    Run the LlamaIndex agent with a specific fault injected.
    Returns the trace_id of the resulting trace in Phoenix.
    """
    ...

def run_all_seeded_failures(
    *,
    manifest_path: str = "datasets/seeded_failures/manifest.json",
    phoenix_endpoint: str = "http://127.0.0.1:6006",
) -> dict[str, str]:
    """
    Run all 30 cases from the manifest.
    Updates manifest.json with trace_ids.
    Returns mapping of run_id → trace_id.
    """
    ...
```

**File**: `apps/demo_agent/run_seeded_failures.py`

CLI script to execute the full seeded failure generation:

```bash
python -m apps.demo_agent.run_seeded_failures \
    --manifest datasets/seeded_failures/manifest.json \
    --phoenix-endpoint http://127.0.0.1:6006
```

#### Acceptance criteria
- [ ] 30 traces visible in Phoenix UI
- [ ] Each trace has spans reflecting the injected failure mode
- [ ] `manifest.json` updated with non-null `trace_id` for all 30 cases
- [ ] Distribution: ~9 tool_failure, ~8 retrieval_failure, ~2 instruction_failure,
      ~4 upstream_dependency_failure, ~7 data_schema_mismatch (matching manifest)
- [ ] Traces exportable as Parquet to `datasets/seeded_failures/exports/`

#### Estimated effort
Medium. Requires understanding the LlamaIndex agent internals from the tutorial
and designing 5 distinct injection strategies.

---

### Phase B: Make REPL Loop the Primary Execution Path

**Goal**: Refactor the RCA engine so the REPL loop is the default mode, with deterministic
narrowing as a pre-filter step inside the REPL.

**Why second**: The engine exists but has 4 modes (deterministic, LLM judgment, REPL,
recursive). We need to unify around REPL-primary per the design decision.

#### What exists
- `investigator/rca/engine.py` — `TraceRCAEngine` with `use_repl_runtime` flag (off by default)
- `investigator/runtime/repl_loop.py` — `ReplLoop` class (functional)
- `investigator/runtime/recursive_loop.py` — `RecursiveLoop` class (functional)

#### What to change

**File**: `investigator/rca/engine.py`

1. **Change default**: `use_repl_runtime=True` (was `False`)
2. **Merge deterministic pre-filter into REPL context**: Instead of the engine running
   hot-span narrowing as a separate step that produces a report independently, run it
   as the first step of the REPL loop's initial context. The REPL model receives the
   pre-filtered hot spans as its starting state.
3. **Remove the pure-deterministic path as a standalone mode**: Deterministic logic stays
   as a pre-filter and as a fallback when budget is exhausted. It is no longer an
   alternative "mode."
4. **Keep LLM judgment as a sub-component**: The structured generation call used in
   "LLM judgment" mode becomes a tool the REPL can use (`synthesize` action) rather than
   a separate execution path.

**Specific changes to `engine.py`**:

```python
# Before: engine.run() has if/elif branches for each mode
# After: engine.run() always does:
#   1. Deterministic pre-filter (hot spans, branches)
#   2. Package pre-filter results as REPL initial context
#   3. Run REPL loop with tool access + analysis sandbox
#   4. If REPL produces output → use it
#   5. If REPL fails/budget-exceeded → fall back to deterministic report
```

**File**: `investigator/runtime/repl_loop.py`

1. **Ensure initial context includes hot-span data**: The `run()` method should accept
   a `pre_filter_context` dict containing hot spans, branch spans, and preliminary
   pattern-match results.
2. **System prompt update**: Include the pre-filter results in the system prompt so the
   model starts from a narrowed view, not from scratch.

#### Acceptance criteria
- [ ] `TraceRCAEngine(use_repl_runtime=True)` is the default (no flag needed)
- [ ] Running `engine.run(request)` always enters the REPL loop
- [ ] Deterministic pre-filter results are visible to the REPL model as initial context
- [ ] Budget exhaustion falls back to deterministic report (not a crash)
- [ ] All existing tests still pass (backward compatibility for test harnesses that
      use deterministic mode)

#### Estimated effort
Medium. Mostly refactoring existing code paths, not writing new logic.

---

### Phase C: Per-Hypothesis Recursive Sub-calls

**Goal**: The REPL loop identifies candidate failure hypotheses and spawns one sub-call
per hypothesis to gather evidence independently.

**Why third**: Requires the REPL loop (Phase B) to be the primary path. Recursion is
the core value-add of the RLM approach.

#### What exists
- `investigator/runtime/recursive_loop.py` — `RecursiveLoop` with `delegate_subcall` action
- `investigator/runtime/recursive_planner.py` — `StructuredActionPlanner`
- `investigator/runtime/sandbox.py` — `SandboxGuard` validates `delegate_subcall` actions

#### What to build

**Hypothesis decomposition strategy** (new logic in REPL loop):

The root REPL's prompt instructs the model to:
1. Analyze hot spans and identify 1–4 candidate failure hypotheses
2. For each hypothesis, specify:
   - `hypothesis_label`: one of the 5 RCA taxonomy labels
   - `hypothesis_statement`: one-sentence description
   - `relevant_span_ids`: which spans to investigate
   - `investigation_tools`: which Inspection API tools to use
3. Spawn a `delegate_subcall` for each hypothesis

**Sub-call execution** (modify `recursive_loop.py`):

Each sub-call receives:
- `objective`: the hypothesis statement
- `context`: filtered span data (only `relevant_span_ids` and their branches)
- Tool access: same Inspection API tools
- Budget: shared global counters (not independent budgets)

Each sub-call must return:
```json
{
  "label": "tool_failure",
  "confidence": 0.72,
  "evidence_refs": [...],
  "supporting_facts": ["tool X returned timeout after 30s", ...],
  "gaps": ["could not inspect tool X's internal retry logic"]
}
```

**Root synthesis** (new logic after sub-calls complete):

The root model receives all sub-call results and:
1. Ranks hypotheses by evidence strength (count + quality of evidence_refs)
2. Picks `primary_label` from the winning hypothesis
3. Records rejected hypotheses in `summary` ("Considered retrieval_failure but...")
4. Merges all evidence_refs (deduplicating by span_id + kind)
5. Computes final confidence (evidence bonus rules from spec)
6. Emits `RCAReport`

**Files to modify**:

| File | Change |
|------|--------|
| `investigator/runtime/repl_loop.py` | Add hypothesis extraction step before sub-call delegation |
| `investigator/runtime/recursive_loop.py` | Ensure sub-calls return structured hypothesis results |
| `investigator/rca/engine.py` | Wire hypothesis decomposition into the main `run()` flow |
| `investigator/prompts/runtime/recursive_runtime_action_v1.md` | Update prompt to instruct hypothesis-based delegation |
| `investigator/prompts/rca/trace_rca_judgment_v1.md` | Update to include sub-call synthesis instructions |

#### Acceptance criteria
- [ ] Root REPL identifies 1–4 hypotheses per trace
- [ ] One sub-call spawned per hypothesis
- [ ] Sub-calls return structured results with evidence_refs
- [ ] Root synthesizes sub-call results into final RCAReport
- [ ] Total budget (tool calls, tokens) is shared across root + sub-calls
- [ ] Depth limit enforced (max 2)
- [ ] If all sub-calls fail, falls back to deterministic report

#### Estimated effort
High. This is the core RLM innovation. Requires prompt engineering for hypothesis
decomposition and synthesis, plus integration of existing recursive_loop machinery.

---

### Phase D: Import Blocklist and Subprocess Sandbox

**Goal**: Enforce security restrictions on the REPL's code execution environment.

**Why fourth**: The REPL (Phase B) and recursion (Phase C) must work before we restrict
what code can run. Adding restrictions too early blocks development.

#### What exists
- `investigator/runtime/sandbox.py` — `SandboxGuard` for action-level validation
- No import-level blocking in the REPL interpreter

#### What to build

**File**: `investigator/runtime/repl_interpreter.py` (new or modify existing)

A restricted Python execution environment:

```python
BLOCKED_MODULES = {
    "os", "subprocess", "socket", "http", "urllib", "pathlib",
    "shutil", "signal", "ctypes", "importlib", "sys",
    "multiprocessing", "threading",
}

ALLOWED_ANALYSIS_MODULES = {
    "json", "re", "math", "statistics", "collections",
    "itertools", "functools", "operator", "datetime",
    "dataclasses", "typing", "copy", "textwrap",
    "hashlib",  # for evidence hashing
}
```

**Execution model**:
- Code runs in a subprocess (not in the main process)
- Subprocess has a custom import hook that blocks `BLOCKED_MODULES`
- REPL output (stdout/stderr) is captured and truncated to N chars per turn
  (default 8192, matching Prime Intellect's approach)
- Subprocess has a per-turn timeout (default 30s)
- Tool calls are proxied through the parent process via a simple protocol
  (subprocess writes JSON request → parent calls ToolRegistry → parent writes
  JSON response)

**Fallback**: If subprocess setup fails (platform issues), fall back to in-process
execution with import hook only (less isolated but functional for dev).

#### Acceptance criteria
- [ ] `import os` in REPL code raises `ImportError`
- [ ] `import subprocess` in REPL code raises `ImportError`
- [ ] `import socket` in REPL code raises `ImportError`
- [ ] Allowed modules (`json`, `re`, `math`, etc.) work normally
- [ ] REPL output truncated at 8192 chars per turn
- [ ] Per-turn timeout kills runaway code after 30s
- [ ] Tool calls from REPL code routed through ToolRegistry (not direct API access)
- [ ] Sandbox violation logged in run record with `SANDBOX_VIOLATION` error code

#### Estimated effort
Medium. Subprocess execution with import hooks is well-understood Python.
The tricky part is the tool-call proxy protocol.

---

### Phase E: CLI Entrypoint

**Goal**: Single command to run RCA on a trace and see results.

**Why fifth**: Requires all core components (REPL, recursion, sandbox) to be working.

#### What to build

**File**: `investigator/rca/cli.py`

```bash
# Single trace
python -m investigator.rca.cli --trace-id <trace_id>

# Batch (all traces in manifest)
python -m investigator.rca.cli --manifest datasets/seeded_failures/manifest.json

# With custom budget
python -m investigator.rca.cli --trace-id <id> --max-iterations 20 --max-tool-calls 60

# Output control
python -m investigator.rca.cli --trace-id <id> --output-dir artifacts/investigator_runs

# Phoenix endpoint override
python -m investigator.rca.cli --trace-id <id> --phoenix-endpoint http://127.0.0.1:6006

# Skip writeback (analysis only)
python -m investigator.rca.cli --trace-id <id> --no-writeback

# Offline mode (Parquet)
python -m investigator.rca.cli --trace-id <id> --parquet datasets/seeded_failures/exports/spans.parquet
```

**CLI arguments**:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--trace-id` | One of trace-id or manifest | — | Single trace to analyze |
| `--manifest` | One of trace-id or manifest | — | Path to manifest.json for batch |
| `--phoenix-endpoint` | No | `http://127.0.0.1:6006` | Phoenix server URL |
| `--parquet` | No | — | Offline Parquet file (skip Phoenix) |
| `--output-dir` | No | `artifacts/investigator_runs` | Where to write run records |
| `--max-iterations` | No | 40 | Budget: max REPL turns |
| `--max-tool-calls` | No | 120 | Budget: max tool calls |
| `--max-depth` | No | 2 | Budget: max recursion depth |
| `--max-wall-time` | No | 180 | Budget: max seconds |
| `--model` | No | `gpt-4o-mini` | Model for all LLM calls |
| `--no-writeback` | No | False | Skip Phoenix annotation writeback |
| `--verbose` | No | False | Print REPL trajectory to stdout |

**Output**:
- Prints `RCAReport` JSON to stdout
- Writes `run_record.json` to output directory
- Writes Phoenix annotations (unless `--no-writeback`)
- Exit code 0 for succeeded/partial, 1 for failed

**Batch mode** (`--manifest`):
- Iterates over all cases in manifest
- Runs RCA for each trace_id (skips null trace_ids)
- Prints summary table at the end: `run_id | trace_id | label | expected | match?`
- Writes all run records to output directory

#### Acceptance criteria
- [ ] `python -m investigator.rca.cli --trace-id <id>` produces an RCAReport on stdout
- [ ] Run record written to artifacts directory
- [ ] `--manifest` mode processes all 30 cases
- [ ] `--no-writeback` skips Phoenix annotation
- [ ] `--verbose` prints REPL turns (reasoning + code + output)
- [ ] Budget arguments override defaults
- [ ] Helpful error messages for missing Phoenix server, invalid trace_id, etc.

#### Estimated effort
Low–Medium. Mostly argument parsing and wiring existing components.

---

### Phase F: End-to-End Validation and Evaluation

**Goal**: Run RCA on all 30 seeded failure traces and measure accuracy.

**Why last**: Requires all components to be working end-to-end.

#### What to build

**File**: `investigator/rca/evaluate.py`

Evaluation script that:
1. Loads `manifest.json` (ground truth)
2. Loads all `run_record.json` files from a batch run
3. Computes metrics:
   - **Top-1 accuracy**: % of traces where `primary_label == expected_label`
   - **Per-label precision/recall**: for each of the 5 labels
   - **Average confidence**: for correct vs incorrect predictions
   - **Evidence quality**: average evidence_refs count for correct vs incorrect
   - **Cost**: average tokens and USD per RCA run
   - **Latency**: average wall-time per RCA run
   - **Budget utilization**: average % of budget consumed
4. Prints summary table
5. Writes evaluation report to `artifacts/evaluation/eval_report.json`

**Evaluation invocation**:

```bash
# Run all 30 cases
python -m investigator.rca.cli --manifest datasets/seeded_failures/manifest.json

# Evaluate results
python -m investigator.rca.evaluate \
    --manifest datasets/seeded_failures/manifest.json \
    --runs-dir artifacts/investigator_runs
```

**Expected output**:

```
RLM-RCA Evaluation Report
═══════════════════════════════════════════════════
Dataset: seeded_failures_v1 (30 cases)
Model: gpt-4o-mini

Top-1 Accuracy: 23/30 (76.7%)

Per-Label Results:
  tool_failure            : P=0.88  R=0.78  F1=0.82  (9 cases)
  retrieval_failure       : P=0.75  R=0.75  F1=0.75  (8 cases)
  instruction_failure     : P=0.50  R=0.50  F1=0.50  (2 cases)
  upstream_dep_failure    : P=0.80  R=1.00  F1=0.89  (4 cases)
  data_schema_mismatch    : P=0.71  R=0.71  F1=0.71  (7 cases)

Cost:  avg $0.03/run  total $0.90
Time:  avg 24s/run    total 12m
Tokens: avg 8,400/run

Evidence Quality:
  Correct predictions:   avg 3.2 evidence_refs
  Incorrect predictions: avg 1.4 evidence_refs
```

#### Additional validation checks

**Smoke test** (single trace, end-to-end):
```bash
# 1. Verify Phoenix is running
curl -s http://127.0.0.1:6006/healthz

# 2. Run RCA on one trace
python -m investigator.rca.cli --trace-id <first_trace_id> --verbose

# 3. Verify outputs
# - RCAReport printed to stdout (valid JSON)
# - run_record.json exists in artifacts/
# - Annotations visible in Phoenix UI on that trace
```

**Contract validation** (for every run):
- [ ] RCAReport passes JSON schema validation
- [ ] All evidence_refs have valid span_ids (exist in the trace)
- [ ] RunRecord has all required fields
- [ ] No `SANDBOX_VIOLATION` errors
- [ ] No `SCHEMA_VALIDATION_FAILED` errors
- [ ] Budget counters are within limits

**Regression baseline**:
- Record evaluation metrics as the baseline
- Future changes to prompts, models, or logic must not regress top-1 accuracy by >5%
- Store baseline in `artifacts/evaluation/baseline_v1.json`

#### Acceptance criteria
- [ ] All 30 seeded failures run through RCA without crashes
- [ ] Top-1 accuracy ≥ 60% (baseline; expect 70-80% with gpt-4o-mini)
- [ ] Every run produces a valid RunRecord
- [ ] Every run produces Phoenix annotations (when writeback enabled)
- [ ] Evaluation report generated with per-label metrics
- [ ] No sandbox violations in any run
- [ ] Total cost for 30 runs < $2.00

#### Estimated effort
Medium. Evaluation script is straightforward; the hard part is debugging
failures and tuning prompts when accuracy is low.

---

## Phase Dependency Graph

```
Phase A: Fault Injector
    │
    │ (produces 30 traces with trace_ids in manifest)
    ▼
Phase B: REPL-Primary Execution
    │
    │ (REPL loop is the default engine mode)
    ▼
Phase C: Per-Hypothesis Recursion
    │
    │ (sub-calls work, evidence gathered per hypothesis)
    ▼
Phase D: Subprocess Sandbox
    │
    │ (REPL code execution is restricted)
    ▼
Phase E: CLI Entrypoint
    │
    │ (single command to run RCA)
    ▼
Phase F: End-to-End Validation
```

Note: Phases B, C, D can be partially parallelized — B and D are independent of each
other, and C depends only on B. However, the recommended order above minimizes rework.

---

## Files Changed per Phase (Summary)

| Phase | New files | Modified files |
|-------|-----------|---------------|
| A | `apps/demo_agent/fault_injector.py`, `apps/demo_agent/run_seeded_failures.py` | `datasets/seeded_failures/manifest.json` |
| B | — | `investigator/rca/engine.py`, `investigator/runtime/repl_loop.py` |
| C | — | `investigator/runtime/repl_loop.py`, `investigator/runtime/recursive_loop.py`, `investigator/rca/engine.py`, prompts |
| D | `investigator/runtime/repl_interpreter.py` (new or refactor) | `investigator/runtime/repl_loop.py`, `investigator/runtime/sandbox.py` |
| E | `investigator/rca/cli.py` | — |
| F | `investigator/rca/evaluate.py` | — |

**Total new files**: 4–5
**Total modified files**: ~8

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LlamaIndex tutorial agent doesn't support fault injection cleanly | Medium | High (blocks Phase A) | Fall back to synthetic trace builder as escape hatch |
| gpt-4o-mini produces poor hypotheses (low RCA accuracy) | Medium | Medium | Tune prompts first; swap to gpt-4o for root model if needed |
| REPL code execution is slow (subprocess overhead per turn) | Low | Low | Use in-process execution with import hooks for dev; subprocess for production |
| Recursive sub-calls consume budget too fast | Medium | Medium | Start with max 2 sub-calls per run; tune up based on accuracy |
| Phoenix API changes break Inspection API | Low | High | Pin Phoenix version in `pyproject.toml`; ParquetClient as offline fallback |
| Model generates code that bypasses import blocklist | Low | Medium | Defense in depth: blocklist + subprocess isolation + no network |

---

## Success Criteria (Overall)

The RLM-RCA system is "done" when:

1. **Functional**: `python -m investigator.rca.cli --trace-id <id>` produces a correct,
   evidence-linked RCA report for a real trace in Phoenix
2. **Accurate**: ≥60% top-1 accuracy on the 30 seeded failure traces
3. **Auditable**: Every run produces a RunRecord with tool trace, sub-call metadata,
   and budget usage
4. **Observable**: Results are visible in Phoenix UI as annotations on the original traces
5. **Safe**: REPL code execution is sandboxed; no filesystem/network access from generated code
6. **Bounded**: No run exceeds budget limits; budget exhaustion produces partial results
7. **Reproducible**: Same trace + same config → same deterministic pre-filter → same
   REPL exploration (modulo LLM non-determinism at temperature 0.0)
