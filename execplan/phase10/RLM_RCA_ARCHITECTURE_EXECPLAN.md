# RLM-RCA System Architecture: Design Reference ExecPlan

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must be kept up to date as work proceeds. This document must be maintained in accordance with PLANS.md at the repository root.

## Purpose / Big Picture

After reading this document, a contributor with no prior knowledge of this repository will understand every component of the RLM-RCA system, how they connect, and why each design choice was made. They will be able to navigate every file in the project, explain the end-to-end data flow from trace generation through autonomous root cause analysis to Phoenix annotation writeback, and validate that the architecture is internally consistent. This document is the single source of architectural truth for Phase 10 of Phoenix-RLM Investigator.

The system solves one problem: given a trace from an AI agent (recorded in Arize Phoenix), automatically determine what went wrong and produce a structured, evidence-linked Root Cause Analysis report. It does this using a Recursive Language Model, which is a deployment pattern where an LLM runs inside a persistent Python environment, uses tools to inspect trace data, and spawns recursive sub-calls of itself to investigate competing failure hypotheses independently.

To see this working after implementation: run the CLI command below from the repository root and observe the structured JSON output, the run record artifact, and the annotations in the Phoenix trace UI.

    python -m investigator.rca.cli --trace-id <some_trace_id> --verbose

## Progress

- [x] (2026-02-12) Architecture design session completed, all decisions locked.
- [x] (2026-02-12) Architecture reference document written (RLM_RCA_ARCHITECTURE.md).
- [x] (2026-02-12) Specs aligned across 9 files (rlm_runtime_contract.md, rlm_engines.md, formal_contracts.md, implementation_journey.md, README.md, CLAUDE.md, AGENTS.md, DESIGN.md, API.md).
- [ ] Architecture validated against implementation (pending Phase 10 implementation).

## Surprises & Discoveries

- The existing codebase already implements roughly 70% of the required components. The engine, contracts, inspection API, tool registry, REPL loop, recursive loop, writeback, and run record persistence all exist. The primary gaps are the fault injector, making the REPL the primary execution path, per-hypothesis recursion, the subprocess sandbox, and the CLI.
- The specs had diverged from the code in two places: the RCAReport used a nested evidence structure in specs/formal_contracts.md while the code at investigator/runtime/contracts.py used a flat evidence_refs list, and the RunRecord in the spec was missing three metadata fields (state_trajectory, subcall_metadata, repl_trajectory) that the code tracks.
- The model defaults across the spec stack were inconsistent: some docs said gpt-5-mini, others gpt-5.2, while the design session locked gpt-4o-mini. This required a 9-file alignment pass.

## Decision Log

- Decision: Use a custom REPL harness, not DSPy's dspy.RLM() primitive.
  Rationale: Full control over tool dispatch, recursion, budgets, and sandbox. No framework coupling. Matches Prime Intellect's approach most closely. DSPy would give scaffolding for free but would limit control over internals.
  Date/Author: 2026-02-12 / Nainy

- Decision: Local subprocess with import blocklist as the sandbox model.
  Rationale: Simple, fast, full CPython support. The system is single-user local dev, not production multi-tenant. Subprocess provides basic isolation; import hooks block dangerous modules. WASM (Pyodide) was rejected because it limits package support. Remote sandboxes (E2B/Modal) add latency and cost.
  Date/Author: 2026-02-12 / Nainy

- Decision: CLI command as the trigger mode.
  Rationale: Manual control for dev/eval phase. The operator decides which traces to analyze. Webhook and scheduled batch triggers are future extensions that use the same core RCA logic.
  Date/Author: 2026-02-12 / Nainy

- Decision: gpt-4o-mini for all calls (root and sub-calls), single model everywhere.
  Rationale: Cost-efficient at roughly $0.01-0.05 per RCA run. Simplifies debugging because there is only one model config. When the workflow is stable, the root model can be upgraded to gpt-4o or gpt-5.2 for better synthesis quality. This is a one-line config change.
  Date/Author: 2026-02-12 / Nainy

- Decision: REPL-primary execution mode (not deterministic-primary).
  Rationale: The RLM REPL loop always runs as the main mode. Deterministic narrowing (hot-span sorting, pattern matching) runs as a pre-filter step inside the REPL context, not as a standalone alternative mode. This matches the Prime Intellect approach where the model always has access to code execution and tools.
  Date/Author: 2026-02-12 / Nainy

- Decision: Tool calls plus Python data analysis in the REPL scope.
  Rationale: The model can call Inspection API tools and also write Python code to filter, aggregate, group, and transform tool results. This is more capable than tool-calls-only and matches what Prime Intellect does. The import blocklist prevents dangerous operations while allowing analysis libraries like json, re, math, statistics, and collections.
  Date/Author: 2026-02-12 / Nainy

- Decision: Per-hypothesis recursive decomposition as the recursion trigger.
  Rationale: The root model identifies candidate failure modes and spawns one sub-call per hypothesis. Each sub-call gets a filtered span slice and investigates independently. This was chosen over per-service decomposition (the tutorial agent is single-service, so that does not help) and over on-low-confidence recursion (which would miss competing hypotheses). It matches Prime Intellect's "parallel deep dives" pattern.
  Date/Author: 2026-02-12 / Nainy

- Decision: Modify the LlamaIndex tutorial agent to generate seeded failure traces.
  Rationale: Wrapping the real agent with programmatic fault injection produces the most realistic traces. Synthetic trace builders produce traces that do not look like real agent runs. Manual fault injection is too slow for 30 cases. The tutorial agent is already runnable locally via Phoenix.
  Date/Author: 2026-02-12 / Nainy

## Outcomes & Retrospective

This section will be filled after Phase 10 implementation is complete.

## Context and Orientation

This section defines every concept and component a novice needs to understand.

A "Recursive Language Model" (RLM) is not a new model architecture. It is a deployment pattern where a standard LLM (like gpt-4o-mini) operates inside a persistent execution environment, typically a Python REPL, where it can write and run code, call tool functions, and spawn recursive instances of itself over subsets of the problem. The key insight is that large context (like hundreds of trace spans) stays as external program state outside the model's token window, and the model decides what to pull into tokens by writing code and calling tools. This idea comes from the RLM paper (rlm/2512.24601v2.pdf in this repository), Prime Intellect's implementation, and DSPy's RLM primitive.

"Arize Phoenix" is an open-source AI observability platform. It stores traces (collections of spans representing LLM calls, tool invocations, retrieval steps, and orchestration events) and provides a web UI to explore them. In this project, Phoenix runs locally via the command "phoenix serve" and listens on http://127.0.0.1:6006. Agents send traces to Phoenix using OpenTelemetry (OTLP/HTTP protocol). The investigator reads traces from Phoenix through its Client API and writes results back as annotations that appear in the trace UI.

"Root Cause Analysis" (RCA) means: given a trace where something went wrong (a tool errored, retrieval returned irrelevant documents, the model ignored instructions, an upstream API timed out, or output did not match expected schema), determine the most likely failure class, cite the specific spans and artifacts that prove it, and suggest remediation. The five failure classes (called the "RCA taxonomy") are: retrieval_failure, tool_failure, instruction_failure, upstream_dependency_failure, and data_schema_mismatch.

"REPL" stands for Read-Eval-Print Loop. In this system it means: the model receives a prompt with context and tool descriptions, writes Python code and tool calls, the harness executes them and returns results, and the loop repeats until the model calls SUBMIT or a budget limit is reached. The REPL is implemented in investigator/runtime/repl_loop.py.

"Hot spans" are the spans most likely to contain the root cause. They are selected by a deterministic pre-filter that prioritizes spans with ERROR status, then spans with exception events, then spans with the highest latency, with ties broken by span_id ascending. This pre-filter runs before the REPL loop starts and its output becomes the REPL's initial context.

"Per-hypothesis recursion" means: the root REPL identifies candidate failure modes (for example, "the search tool timed out" and "retrieval returned wrong documents"), then spawns one sub-call per hypothesis. Each sub-call gets a filtered slice of the trace (only the spans relevant to that hypothesis) and investigates independently using the same tools and analysis capabilities. Sub-calls return structured results (label, confidence, evidence_refs, gaps). The root model then synthesizes these results into a final RCAReport.

"Evidence refs" are canonical pointer objects that link every claim in an RCA report to a specific span, tool input/output, retrieval chunk, message, or config diff. They have a fixed shape: trace_id, span_id, kind, ref, excerpt_hash, ts. The excerpt_hash is a SHA256 of the relevant text, ensuring integrity without storing sensitive content. Evidence refs are the audit trail that makes RCA reports verifiable.

The project repository is organized as follows. The apps/demo_agent/ directory contains the agent harness that generates traces. The investigator/ directory contains the three RLM engines (rca/, compliance/, incident/), the shared runtime (runtime/), and the read-only Inspection API (inspection_api/). The datasets/seeded_failures/ directory holds the ground-truth manifest and exported traces. The artifacts/investigator_runs/ directory stores per-invocation run records. The specs/ directory contains normative contracts. The execplan/phase10/ directory (where this document lives) contains Phase 10 design and implementation docs.

Key files and what they do:

investigator/rca/engine.py is the TraceRCAEngine class. It runs deterministic hot-span narrowing, then the REPL loop, then per-hypothesis sub-calls, then synthesis. Its run() method takes a TraceRCARequest and returns an RCAReport.

investigator/runtime/repl_loop.py is the ReplLoop class. It manages the iterative code-execution loop: model generates reasoning plus code, harness executes code in a sandbox, results are fed back. It enforces budgets and tracks state trajectory.

investigator/runtime/recursive_loop.py is the RecursiveLoop class. It executes typed actions (tool_call, delegate_subcall, synthesize, finalize) with bounded budgets and explicit state transitions. Sub-calls create nested instances.

investigator/runtime/contracts.py defines all data contracts: EvidenceRef, RCAReport, RuntimeBudget, RuntimeUsage, RunRecord, and their serialization.

investigator/runtime/tool_registry.py provides the ToolRegistry class that enforces the tool allowlist, sanitizes arguments, normalizes results for determinism, and logs argument/response hashes.

investigator/runtime/sandbox.py provides the SandboxGuard class that validates every action before execution. It checks action types, tool names, JSON safety, and argument structure.

investigator/runtime/llm_client.py provides the OpenAIModelClient class for structured generation. It calls the OpenAI Responses API with JSON schema mode, tracks token usage and cost, and handles model-specific quirks (gpt-5 family does not support temperature).

investigator/inspection_api/protocol.py defines the InspectionAPI protocol (Python Protocol class) with all 16 tool function signatures.

investigator/inspection_api/phoenix_client.py implements PhoenixInspectionAPI backed by the Phoenix Client. It provides deterministic span access, message extraction, tool I/O extraction, retrieval chunk extraction, controls loading, config snapshot management, and regex search.

investigator/rca/workflow.py orchestrates a complete RCA run: calls the engine, writes results to Phoenix, persists the run record.

investigator/rca/writeback.py builds Phoenix-compatible annotation payloads (TraceEvaluations for rca.primary, SpanEvaluations for rca.evidence) and logs them via the Phoenix Client.

datasets/seeded_failures/manifest.json contains 30 test cases mapping run_id to expected_label, with trace_id fields currently null (to be filled by the fault injector).

## Plan of Work

This document is a reference architecture, not an implementation plan. The implementation plan is in RLM_RCA_IMPLEMENTATION_EXECPLAN.md in the same directory. This document's "work" is: keep the architecture description accurate as implementation proceeds, and update the Decision Log when new decisions are made.

The architecture describes a five-step execution flow that happens when the CLI is invoked with a trace_id. First, deterministic pre-filtering selects the top five hot spans using a stable sort by error status, exception events, latency, and span_id. Second, the REPL loop starts with the hot-span summary as initial context, and the model iteratively writes Python code and calls Inspection API tools to explore the trace. Third, the model identifies one to four candidate failure hypotheses and spawns a recursive sub-call per hypothesis, each receiving a filtered span slice and an investigation objective. Fourth, the root model synthesizes sub-call results by comparing evidence strength, picking a primary label, recording rejected hypotheses, and computing confidence. Fifth, the system emits an RCAReport, persists a RunRecord, and writes annotations back to Phoenix.

The sandbox operates as follows. Model-generated code runs in a local subprocess with a custom import hook. The import hook blocks os, subprocess, socket, http, urllib, pathlib, shutil, signal, ctypes, importlib, sys, multiprocessing, and threading. It allows json, re, math, statistics, collections, itertools, functools, operator, datetime, dataclasses, typing, copy, textwrap, and hashlib. REPL output is truncated to 8192 characters per turn. Each code execution turn has a 30-second timeout. Tool calls from generated code are proxied through the parent process via the ToolRegistry.

The budget system enforces global limits shared across the root loop and all sub-calls. The RCA engine defaults are: 40 max iterations, depth 2, 120 tool calls, 40 sub-calls, 200000 total tokens, and 180 seconds wall time. When any limit is reached, the runtime enters terminated_budget status and attempts best-effort finalization. At 90% of wall time, the REPL forces finalization. The run record always records budget usage regardless of outcome.

The evidence model requires every claim to cite evidence using canonical evidence_ref objects. Minimum one evidence ref for low confidence, minimum two independent refs for medium or high confidence. Evidence kinds are SPAN, TOOL_IO, RETRIEVAL_CHUNK, MESSAGE, and CONFIG_DIFF. Evidence integrity is ensured via excerpt_hash (SHA256 of the text). All evidence refs are validated against the inspected trace before the report is emitted.

Results are written back to Phoenix as two annotation types. The rca.primary annotation is trace-level, containing the primary label, confidence, and full RCAReport JSON. The rca.evidence annotation is span-level, one per evidence span, containing the evidence kind, weight, and pointer. Both include run_id for traceability.

## Concrete Steps

To validate the architecture against the implementation, a contributor should perform these checks from the repository root working directory.

Verify that the InspectionAPI protocol matches API.md by reading both files and confirming all 16 function signatures are present:

    uv run python -c "from investigator.inspection_api.protocol import InspectionAPI; print([m for m in dir(InspectionAPI) if not m.startswith('_')])"

Verify that the contracts module defines all required data classes:

    uv run python -c "from investigator.runtime.contracts import EvidenceRef, RCAReport, RunRecord, RuntimeBudget, RuntimeUsage; print('All contracts importable')"

Verify that the tool registry exposes the expected 16 tools:

    uv run python -c "from investigator.runtime.tool_registry import DEFAULT_ALLOWED_TOOLS; print(f'{len(DEFAULT_ALLOWED_TOOLS)} tools:', sorted(DEFAULT_ALLOWED_TOOLS))"

Verify that the sandbox guard blocks disallowed action types:

    uv run python -c "
    from investigator.runtime.sandbox import SandboxGuard, SandboxViolationError
    guard = SandboxGuard(allowed_tools={'get_span'})
    try:
        guard.validate_action({'type': 'shell_exec', 'command': 'ls'})
        print('FAIL: should have raised')
    except SandboxViolationError:
        print('PASS: disallowed action type blocked')
    "

## Validation and Acceptance

The architecture is considered validated when all of the following are true. The InspectionAPI protocol has all 16 function signatures matching API.md. The contracts module defines EvidenceRef, RCAReport, ComplianceReport, IncidentDossier, RunRecord, RuntimeBudget, and RuntimeUsage with to_dict serialization. The ToolRegistry exposes exactly the 16 tools listed in DEFAULT_ALLOWED_TOOLS. The SandboxGuard blocks action types not in the allowlist and rejects tool calls to tools not in the allowed set. The REPL loop, recursive loop, and recursive planner are all importable and their constructors accept the expected parameters. The writeback module produces TraceEvaluations and SpanEvaluations with the correct annotation names (rca.primary and rca.evidence).

These checks can be run without a live Phoenix server or OpenAI API key.

## Idempotence and Recovery

This document is a reference. Reading it and running the validation commands has no side effects. The commands can be run repeatedly. If any import fails, it indicates a code-spec misalignment that should be resolved by updating either the code or this document.

## Artifacts and Notes

The architecture reference document (non-ExecPlan format) is preserved at execplan/phase10/RLM_RCA_ARCHITECTURE.md for diagram rendering, since indented blocks do not render ASCII diagrams well. That file contains the system overview diagram, execution flow diagram, REPL harness diagram, trust boundary diagram, and data flow diagram. Consult it for visual orientation.

## Interfaces and Dependencies

The system depends on the following external packages, all managed via uv:

arize-phoenix provides the Phoenix server, Client API, TraceEvaluations, SpanEvaluations, and span query DSL. It is the trace storage and annotation backend.

openai provides the OpenAI Python SDK for structured generation via the Responses API. The model is gpt-4o-mini by default.

pandas is used internally by the PhoenixInspectionAPI for span dataframe operations.

The internal interface surface consists of:

InspectionAPI (investigator/inspection_api/protocol.py) is a Python Protocol with 16 methods. Every read operation on trace data goes through this interface.

TraceRCAEngine (investigator/rca/engine.py) is the main engine class. Its run method takes a TraceRCARequest (containing trace_id and an InspectionAPI instance) and returns an RCAReport.

ReplLoop (investigator/runtime/repl_loop.py) manages iterative code execution. Its run method takes a system prompt, initial context, tool registry, and budget, and returns a ReplLoopResult.

RecursiveLoop (investigator/runtime/recursive_loop.py) manages recursive action execution. Its run method takes an action queue, tool registry, and budget, and returns a RecursiveLoopResult.

ToolRegistry (investigator/runtime/tool_registry.py) wraps an InspectionAPI instance and exposes a call method that validates tool names, sanitizes args, invokes the tool, normalizes results, and returns a dict with argument and response hashes.

SandboxGuard (investigator/runtime/sandbox.py) validates action dicts before execution. Its validate_action method raises SandboxViolationError for invalid actions.

RunRecord (investigator/runtime/contracts.py) is the per-invocation audit artifact. Every RCA run produces exactly one RunRecord persisted as JSON.
