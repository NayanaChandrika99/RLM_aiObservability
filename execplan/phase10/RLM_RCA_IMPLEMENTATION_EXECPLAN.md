# RLM-RCA Implementation: Build Autonomous Trace Root Cause Analysis

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must be kept up to date as work proceeds. This document must be maintained in accordance with PLANS.md at the repository root.

The companion architecture reference is at execplan/phase10/RLM_RCA_ARCHITECTURE_EXECPLAN.md. That document defines every concept, component, and design decision. This document tells you how to build and validate each piece.

## Purpose / Big Picture

After this plan is complete, a user can run a single CLI command and get an autonomous, evidence-linked Root Cause Analysis of any AI agent trace stored in Phoenix. The system uses a Recursive Language Model (an LLM running inside a Python REPL with tool access and recursive sub-calls) to investigate failure hypotheses, gather evidence from trace spans, and produce a structured report with remediation guidance. The report is written back to Phoenix as annotations so it appears alongside the original trace in the UI.

To see the final result working, start Phoenix locally, generate 30 traces with known failure modes, then run:

    phoenix serve
    python -m apps.demo_agent.run_seeded_failures --manifest datasets/seeded_failures/manifest.json
    python -m investigator.rca.cli --manifest datasets/seeded_failures/manifest.json
    python -m investigator.rca.evaluate --manifest datasets/seeded_failures/manifest.json --runs-dir artifacts/investigator_runs

The evaluation command prints a report showing top-1 accuracy across all 30 traces, per-label precision and recall, cost, and latency. The target is at least 60% top-1 accuracy with total cost under $2.00 for 30 runs.

## Progress

- [x] (2026-02-12) Design session completed, all architectural decisions locked.
- [x] (2026-02-12) Architecture document and implementation plan written.
- [x] (2026-02-12) Specs aligned across 9 files to match new decisions.
- [x] (2026-02-13) Milestone 1: Fault injector and seeded trace generation (Phase A) completed end-to-end (live LlamaIndex path + deterministic fallback + manifest update CLI + unit tests + live Phoenix smoke validation).
- [x] (2026-02-13) Milestone 2: REPL-primary engine refactor (Phase B) completed (REPL default path + deterministic pre-filter context + deterministic fallback wiring + Phase 10 RED/GREEN coverage).
- [x] (2026-02-13) Milestone 3: Per-hypothesis recursive sub-calls (Phase C) completed (hypothesis extraction payload path + per-hypothesis recursive sub-calls + synthesis selection + merged evidence refs + regression coverage).
- [x] (2026-02-13) Milestone 4: Subprocess sandbox with import blocklist (Phase D) completed (blocked-module import guard + subprocess execution timeout/output caps + JSON request/response proxy for tool and semantic subquery helpers + in-process fallback path).
- [x] (2026-02-13) Milestone 5: CLI entrypoint (Phase E) completed (`investigator.rca.cli` supports single-trace and manifest batch execution, parquet mode, budget/model flags, optional writeback suppression, and verbose trajectory output).
- [ ] Milestone 6: End-to-end evaluation (Phase F) is implementation-complete (`investigator.rca.evaluate` + tests), with full 30-trace acceptance run pending refreshed non-null `trace_id` values in the manifest.

## Surprises & Discoveries

This section will be populated as implementation proceeds. Placeholder entries based on design research:

- Observation: The existing engine at investigator/rca/engine.py already has four modes (deterministic, LLM judgment, REPL runtime, recursive runtime) controlled by boolean flags. The REPL and recursive modes are functional but opt-in. The refactor to REPL-primary is mostly eliminating the branching, not building new logic.

- Observation: The gpt-5-mini model (previously the default) rejects the temperature parameter on the OpenAI Responses API. The client at investigator/runtime/llm_client.py already handles this via the _supports_temperature check. The new default gpt-4o-mini does support temperature, so this quirk may not apply, but the guard should stay.

- Observation: The manifest at datasets/seeded_failures/manifest.json has 30 cases with trace_id set to null. The fault_profile field maps directly to injection strategy names. The distribution is: 9 tool_failure, 8 retrieval_failure, 2 instruction_failure, 4 upstream_dependency_failure, 7 data_schema_mismatch.

- Observation: A fast Milestone 1 baseline can be implemented by wrapping the existing phase1 seeded span emitter and resolving trace_id by run_id from Phoenix, then upgraded incrementally to live LlamaIndex execution.

- Observation: Phoenix exposes custom span attributes for this path under a nested column (`attributes.phase1`) rather than a flattened `attributes.phase1.run_id` column. Trace resolution by run_id must parse the nested dict; otherwise live traces are generated but lookup fails.

- Observation: During live instrumentation, OpenTelemetry may emit "Overriding of current TracerProvider is not allowed" in local runs. The warning is non-fatal in this setup; spans are still ingested and queryable when run_id extraction is correct.

- Observation: Recursive sub-call outputs must preserve structured fields (`label`, `confidence`, `supporting_facts`, `evidence_refs`, `gaps`) in `RecursiveLoop` output rather than only merged evidence/gaps. Without this structure, the RCA engine cannot rank hypotheses deterministically.

- Observation: Subprocess sandbox event messages must bypass redirected user stdout/stderr. Using `sys.__stdout__`/`sys.__stdin__` for protocol I/O avoids deadlocks when model code runs under captured output streams.

- Observation: `run_trace_rca_workflow` currently always performs writeback, so CLI-level `--no-writeback` is best implemented with a no-op writeback client to preserve run-record persistence without Phoenix annotation side effects.

- Observation: The current checked-in manifest (`datasets/seeded_failures/manifest.json`) still has null `trace_id` values, so evaluation CLI execution reports 0-case metrics until traces are regenerated or attached.

## Decision Log

All design decisions are recorded in the architecture ExecPlan (RLM_RCA_ARCHITECTURE_EXECPLAN.md, Decision Log section). This implementation plan inherits those decisions without modification. Key decisions summarized here for quick reference: custom REPL harness, local subprocess sandbox, CLI trigger, gpt-4o-mini everywhere, REPL-primary mode, tool calls plus data analysis, per-hypothesis recursion, modify LlamaIndex tutorial for fault injection.

Implementation-specific decisions will be recorded here as they arise during coding.

- Decision: Implement Milestone 1 in two steps: first ship deterministic seeded-failure orchestration (fault_injector.py + run_seeded_failures.py + tests), then upgrade to live LlamaIndex fault injection in a follow-up patch.
  Rationale: Keeps forward progress with testable manifest/trace wiring while isolating the higher-risk LlamaIndex runtime patching work.
  Date/Author: 2026-02-13 / Codex

- Decision: Keep deterministic seeded emission as an explicit fallback path when live LlamaIndex dependencies or credentials are unavailable.
  Rationale: Preserves deterministic local progress and avoids blocking batch dataset generation in constrained environments.
  Date/Author: 2026-02-13 / Codex

- Decision: Resolve trace_id from Phoenix using both flattened run_id columns and nested attributes.phase1 dict payloads.
  Rationale: Live traces write run_id inside nested phase1 attributes in this environment; strict flattened lookup is brittle and caused false negatives.
  Date/Author: 2026-02-13 / Codex

- Decision: Run per-hypothesis recursive investigations from `TraceRCAEngine` immediately after REPL hypothesis emission, using a shared `RuntimeBudgetPool` and one recursive call per hypothesis.
  Rationale: This keeps hypothesis execution deterministic, budget-aware, and isolated while reusing the existing recursive runtime contract.
  Date/Author: 2026-02-13 / Codex

- Decision: Implement sandbox helper calls (`call_tool`, `llm_query`, `llm_query_batched`) through a newline-delimited JSON event protocol between parent runtime and subprocess worker.
  Rationale: Preserves existing REPL helper semantics while moving code execution into an isolated subprocess.
  Date/Author: 2026-02-13 / Codex

- Decision: Keep CLI execution routed through `run_trace_rca_workflow` for both single and batch paths, and derive batch summary rows from workflow outputs/run-record errors.
  Rationale: Reuses the existing artifact/writeback lifecycle instead of creating a parallel runtime path in the CLI layer.
  Date/Author: 2026-02-13 / Codex

- Decision: In RCA evaluation, select the latest run per `trace_id` (by `completed_at`/`started_at`) when multiple run records exist in the runs directory.
  Rationale: Makes repeated batch runs idempotent for metric aggregation without requiring manual run-directory cleanup.
  Date/Author: 2026-02-13 / Codex

## Outcomes & Retrospective

This section will be filled at major milestones and at completion.

- Milestone completion (2026-02-13): Milestone 1 is fully complete. run_with_fault now attempts live LlamaIndex execution first (instrumented to Phoenix), injects profile-specific failure markers, resolves run_id -> trace_id, and falls back deterministically when live mode is unavailable. run_all_seeded_failures updates manifest trace IDs and exports Parquet.

- Validation evidence (2026-02-13): live smoke run returned trace_id 36d4170af1ffe77acbc08c58ee372810 for project phase10-live-smoke, with span names including RetrieverQueryEngine.query, VectorIndexRetriever.retrieve, OpenAI.chat, and tool.call.

- Milestone completion (2026-02-13): Milestone 2 is complete. `TraceRCAEngine` now treats REPL as the primary default path, always computes deterministic pre-filter context before REPL execution, and falls back deterministically when REPL judgment fails or budgets out.

- Milestone completion (2026-02-13): Milestone 3 is complete. REPL hypothesis payloads are normalized in the RCA engine, each hypothesis is investigated via recursive sub-calls, and synthesis chooses the strongest supported hypothesis while merging deduplicated evidence refs into the final report.

- Validation evidence (2026-02-13): `uv run pytest tests/unit/test_trace_rca_engine_phase10_repl.py tests/unit/test_runtime_repl_loop_phase10.py tests/unit/test_runtime_recursive_loop_phase8.py tests/unit/test_trace_rca_engine_phase9_recursive.py tests/unit/test_trace_rca_engine_phase8_llm.py tests/unit/test_phase6_replay_acceptance.py tests/unit/test_investigator_runtime_scaffold.py -q` passed with 39 tests.

- Milestone completion (2026-02-13): Milestone 4 is complete. `execute_in_sandbox` now runs model-generated code in a subprocess with blocked-module import guardrails, output truncation, timeout enforcement, protocol-proxied helper requests, and an in-process guarded fallback when subprocess setup fails.

- Validation evidence (2026-02-13): `uv run pytest tests/unit/test_runtime_repl_interpreter_phase10.py tests/unit/test_runtime_repl_loop_phase10.py tests/unit/test_trace_rca_engine_phase10_repl.py tests/unit/test_compliance_phase10_repl.py tests/unit/test_runtime_recursive_loop_phase8.py tests/unit/test_trace_rca_engine_phase9_recursive.py tests/unit/test_trace_rca_engine_phase8_llm.py tests/unit/test_phase6_replay_acceptance.py tests/unit/test_investigator_runtime_scaffold.py -q` passed with 46 tests.

- Milestone completion (2026-02-13): Milestone 5 is complete. `investigator/rca/cli.py` now exposes `python -m investigator.rca.cli` with trace/manifest modes, parquet support, runtime budget flags, model override, optional writeback suppression, and batch summary reporting.

- Validation evidence (2026-02-13): `uv run pytest tests/unit/test_trace_rca_cli_phase10.py tests/unit/test_runtime_repl_interpreter_phase10.py tests/unit/test_runtime_repl_loop_phase10.py tests/unit/test_trace_rca_engine_phase10_repl.py tests/unit/test_compliance_phase10_repl.py tests/unit/test_runtime_recursive_loop_phase8.py tests/unit/test_trace_rca_engine_phase9_recursive.py tests/unit/test_trace_rca_engine_phase8_llm.py tests/unit/test_phase6_replay_acceptance.py tests/unit/test_investigator_runtime_scaffold.py -q` passed with 50 tests.

- Milestone completion (2026-02-13): Milestone 6 evaluator implementation is complete. `investigator/rca/evaluate.py` now computes top-1 accuracy, per-label precision/recall/F1, confidence/evidence quality splits, runtime cost/time/token metrics, and budget utilization from run records, then writes `artifacts/evaluation/eval_report.json`.

- Validation evidence (2026-02-13): `uv run pytest tests/unit/test_trace_rca_evaluate_phase10.py tests/unit/test_trace_rca_cli_phase10.py tests/unit/test_runtime_repl_interpreter_phase10.py tests/unit/test_runtime_repl_loop_phase10.py tests/unit/test_trace_rca_engine_phase10_repl.py tests/unit/test_compliance_phase10_repl.py tests/unit/test_runtime_recursive_loop_phase8.py tests/unit/test_trace_rca_engine_phase9_recursive.py tests/unit/test_trace_rca_engine_phase8_llm.py tests/unit/test_phase6_replay_acceptance.py tests/unit/test_investigator_runtime_scaffold.py -q` passed with 53 tests.

- Validation evidence (2026-02-13): `uv run python -m investigator.rca.evaluate --manifest datasets/seeded_failures/manifest.json --runs-dir artifacts/investigator_runs` executed successfully and wrote a report; output currently reflects 0 manifest cases with non-null `trace_id`.

## Context and Orientation

This section explains what exists today and what needs to change.

Phoenix-RLM Investigator is a project that builds an offline investigation layer on top of Arize Phoenix, an AI observability platform. Phoenix stores traces (timelines of LLM calls, tool invocations, retrieval steps) and this project adds automated analysis engines that read those traces, determine what went wrong, and write results back as annotations.

The project runs locally without Docker. Phoenix is started with "phoenix serve" on http://127.0.0.1:6006. Agents send traces via OTLP/HTTP to http://127.0.0.1:6006/v1/traces. The investigator reads from Phoenix via the Client API and writes annotations back.

The repository uses uv for dependency management. All Python commands are run with "uv run python" to ensure the correct environment.

The current state of the codebase has approximately 70% of the RCA components implemented. Here is what exists and what is missing, file by file.

Existing and complete: investigator/inspection_api/protocol.py defines the 16-method InspectionAPI protocol. investigator/inspection_api/phoenix_client.py implements PhoenixInspectionAPI with full span access, message extraction, tool I/O, retrieval chunks, controls, config snapshots, and search. investigator/runtime/contracts.py defines all dataclasses (EvidenceRef, RCAReport, RunRecord, RuntimeBudget, etc.) with serialization. investigator/runtime/tool_registry.py provides allowlisted tool dispatch with argument sanitization and response hashing. investigator/runtime/sandbox.py validates actions against type and tool allowlists. investigator/runtime/llm_client.py provides OpenAI structured generation with usage tracking. investigator/runtime/prompt_registry.py loads prompt templates and computes hashes. investigator/runtime/repl_loop.py implements the iterative REPL with budget enforcement. investigator/runtime/recursive_loop.py implements recursive action execution with state machine. investigator/runtime/recursive_planner.py produces typed actions from structured model responses. investigator/rca/engine.py implements TraceRCAEngine with hot-span narrowing, branch collection, pattern-based label detection, and four execution modes. investigator/rca/workflow.py orchestrates end-to-end RCA with writeback and run record persistence. investigator/rca/writeback.py builds Phoenix annotation payloads. apps/demo_agent/phase1_langgraph_runner.py and apps/demo_agent/phase1_tutorial_run.py set up Phoenix tracing and run the tutorial agent. apps/demo_agent/fault_injector.py and apps/demo_agent/run_seeded_failures.py now provide the full Milestone 1 orchestration path (live LlamaIndex fault execution with deterministic fallback, trace_id resolution, and manifest update CLI).

Missing and needed: investigator/rca/cli.py (CLI for RCA execution), investigator/rca/evaluate.py (evaluation metrics), investigator/runtime/repl_interpreter.py (subprocess sandbox with import hooks).

Needs modification: investigator/rca/engine.py (make REPL the primary mode, wire per-hypothesis recursion), investigator/runtime/repl_loop.py (accept pre-filter context, hypothesis extraction), investigator/runtime/recursive_loop.py (structured hypothesis results from sub-calls), prompt templates in investigator/prompts/ (hypothesis decomposition and synthesis instructions).

## Plan of Work

The work is organized into six milestones, executed sequentially because each builds on the previous. The dependency chain is: fault injector produces traces, REPL-primary refactor makes the engine work, per-hypothesis recursion adds the RLM's core capability, the sandbox adds security, the CLI makes it runnable, and evaluation proves it works.

### Milestone 1: Fault Injector and Seeded Trace Generation

This milestone produces 30 traces in Phoenix with known failure modes. Everything downstream needs real traces to operate on.

The LlamaIndex tutorial at phoenix/tutorials/tracing/llama_index_openai_agent_tracing_tutorial.ipynb shows how to build an agent with tools and retrieval. The implemented live path runs an instrumented LlamaIndex query engine over profile-specific documents and injects profile failure markers in the same trace to preserve deterministic label mapping for seeded cases.

Create apps/demo_agent/fault_injector.py. This module defines two functions. The first, run_with_fault, takes a fault_profile string (one of profile_tool_failure, profile_retrieval_failure, profile_instruction_failure, profile_upstream_dependency_failure, profile_data_schema_mismatch), a run_id string, and a phoenix_endpoint URL. It configures Phoenix tracing, sets up the LlamaIndex agent from the tutorial, applies the fault injection for the given profile, runs the agent on a standard query, and returns the trace_id of the resulting trace. The second, run_all_seeded_failures, loads the manifest at datasets/seeded_failures/manifest.json, iterates over all 30 cases, calls run_with_fault for each, updates the manifest with the resulting trace_ids, and writes the updated manifest back.

The fault injection strategies are as follows. For profile_tool_failure, monkey-patch one of the agent's tool functions to raise an exception (like ConnectionError or TimeoutError), return after a forced 30-second sleep, or modify the tool description so the agent selects the wrong tool. For profile_retrieval_failure, replace the retriever with one that returns documents from an unrelated topic, returns empty results, or returns documents with very low relevance scores. For profile_instruction_failure, modify the system prompt to include contradictory instructions ("always respond in JSON" conflicting with "respond in plain English"), remove format requirements, or inject a mid-conversation system message that changes the task. For profile_upstream_dependency_failure, mock the OpenAI API response to return HTTP 500, HTTP 429, a connection timeout, or a malformed JSON body. For profile_data_schema_mismatch, modify tool output to return a string where the agent expects JSON, change field names in the output schema, or return a nested structure where a flat one is expected.

Create apps/demo_agent/run_seeded_failures.py as a CLI script that parses --manifest and --phoenix-endpoint arguments and calls run_all_seeded_failures.

To validate this milestone, start Phoenix, run the script, and verify:

    phoenix serve &
    uv run python -m apps.demo_agent.run_seeded_failures --manifest datasets/seeded_failures/manifest.json
    uv run python -c "import json; m = json.load(open('datasets/seeded_failures/manifest.json')); nulls = [c for c in m['cases'] if c['trace_id'] is None]; print(f'{len(nulls)} traces still null out of {len(m[\"cases\"])}')"

The expected output of the last command is "0 traces still null out of 30". Additionally, open http://127.0.0.1:6006 in a browser and confirm that 30 traces are visible with spans reflecting the injected failures (error status codes, exception events, retrieval spans with irrelevant content, etc.).

### Milestone 2: REPL-Primary Engine Refactor

This milestone makes the REPL loop the default execution path in the RCA engine. Currently the engine has four modes controlled by boolean flags (use_llm_judgment, use_repl_runtime, use_recursive_runtime). After this milestone, the engine always runs through the REPL.

Edit investigator/rca/engine.py. Change the default for use_repl_runtime from False to True. Then restructure the run method so it always executes: (1) deterministic pre-filter to produce hot spans and branch context, (2) package pre-filter results as a dict containing hot_spans, branch_span_ids, and preliminary pattern-match labels, (3) pass this dict to the REPL loop as initial context, (4) if the REPL produces a valid RCAReport, use it, (5) if the REPL fails or budget is exhausted, fall back to a deterministic report built from the pre-filter results alone. Remove the if/elif branching that selects between modes. Keep the deterministic logic as helper methods (_sort_hot_spans, _collect_branch_span_ids, _detect_label, _confidence_from_evidence) since they are used by both the pre-filter and the fallback.

Edit investigator/runtime/repl_loop.py. Update the run method to accept a pre_filter_context dict parameter. Incorporate this context into the system prompt so the model starts with a narrowed view of the trace rather than exploring from scratch. The system prompt should present the hot spans, their status codes, latencies, and any preliminary pattern matches, and instruct the model to investigate further using tools and analysis code.

To validate, run the existing test suite to confirm nothing is broken, then run a manual test against one trace:

    uv run python -m pytest tests/ -x -q
    uv run python -c "
    from investigator.rca.engine import TraceRCAEngine
    engine = TraceRCAEngine()
    print(f'use_repl_runtime default: {engine.use_repl_runtime}')
    assert engine.use_repl_runtime is True, 'REPL should be default'
    print('PASS')
    "

### Milestone 3: Per-Hypothesis Recursive Sub-calls

This milestone is the core RLM innovation. The root REPL identifies candidate failure hypotheses and spawns one sub-call per hypothesis to gather evidence independently.

This requires changes in three areas. First, the REPL prompt must instruct the model to analyze hot spans and identify one to four candidate failure hypotheses, each with a label from the taxonomy (retrieval_failure, tool_failure, instruction_failure, upstream_dependency_failure, data_schema_mismatch), a one-sentence statement, a list of relevant span IDs, and a list of Inspection API tools to use for investigation. Second, the REPL loop must extract these hypotheses and spawn a delegate_subcall for each one. The sub-call receives an objective (the hypothesis statement), a context (filtered span data containing only the relevant span IDs and their branches), and access to the same Inspection API tools. Third, each sub-call must return a structured result containing label, confidence, evidence_refs, supporting_facts, and gaps. The root model then receives all sub-call results and synthesizes them into a final RCAReport by ranking hypotheses by evidence strength, picking the primary label from the winner, recording rejected hypotheses in the summary, merging and deduplicating evidence_refs, computing confidence with evidence bonus rules, and generating remediation.

Edit investigator/prompts/runtime/recursive_runtime_action_v1.md to include instructions for hypothesis-based delegation. The prompt should explain that the model's goal is to identify competing hypotheses and spawn independent investigations, not to try to solve everything in one pass.

Edit investigator/prompts/rca/trace_rca_judgment_v1.md to include synthesis instructions for when the model receives sub-call results. The prompt should explain how to compare evidence strength, handle conflicting hypotheses, and produce a final verdict.

Edit investigator/runtime/repl_loop.py to add a hypothesis extraction step. After the model generates hypotheses, the loop should create delegate_subcall actions and pass them to the recursive loop.

Edit investigator/runtime/recursive_loop.py to ensure sub-calls return structured hypothesis results. The sub-call's finalize action should produce a dict with the required fields (label, confidence, evidence_refs, supporting_facts, gaps).

Edit investigator/rca/engine.py to wire the hypothesis decomposition into the main run flow. After the REPL generates hypotheses and sub-calls complete, the engine should invoke synthesis and produce the final RCAReport.

To validate, run a single-trace RCA with verbose output and inspect the REPL trajectory for hypothesis generation and sub-call results:

    uv run python -c "
    from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
    from investigator.inspection_api.phoenix_client import PhoenixInspectionAPI
    api = PhoenixInspectionAPI(endpoint='http://127.0.0.1:6006')
    engine = TraceRCAEngine()
    # Use a trace_id from the manifest after Milestone 1
    import json
    manifest = json.load(open('datasets/seeded_failures/manifest.json'))
    first_case = [c for c in manifest['cases'] if c['trace_id'] is not None][0]
    request = TraceRCARequest(trace_id=first_case['trace_id'], inspection_api=api)
    report = engine.run(request)
    print(f'Label: {report.primary_label}')
    print(f'Expected: {first_case[\"expected_label\"]}')
    print(f'Confidence: {report.confidence}')
    print(f'Evidence refs: {len(report.evidence_refs)}')
    print(f'Match: {report.primary_label == first_case[\"expected_label\"]}')
    "

The expected output should show a label, confidence above 0.3, at least one evidence ref, and ideally a match with the expected label.

### Milestone 4: Subprocess Sandbox with Import Blocklist

This milestone adds security restrictions to the REPL's code execution. Model-generated Python code runs in a subprocess with a custom import hook that blocks dangerous modules.

Create or modify investigator/runtime/repl_interpreter.py. Define BLOCKED_MODULES as a set containing os, subprocess, socket, http, urllib, pathlib, shutil, signal, ctypes, importlib, sys, multiprocessing, and threading. Define ALLOWED_ANALYSIS_MODULES as a set containing json, re, math, statistics, collections, itertools, functools, operator, datetime, dataclasses, typing, copy, textwrap, and hashlib. Implement a custom import hook (a class with find_module and load_module methods, or using importlib.abc.MetaPathFinder) that raises ImportError for any module in BLOCKED_MODULES. Implement a function execute_in_sandbox that spawns a subprocess, installs the import hook, executes the model's code string, captures stdout and stderr (truncated to 8192 characters), and returns the result. The subprocess should have a 30-second timeout enforced via subprocess.run with timeout parameter. Tool calls from the generated code should be proxied through a simple JSON protocol: the subprocess writes a tool request as a JSON line to stdout, the parent reads it, calls ToolRegistry, and writes the response as a JSON line to the subprocess's stdin. As a fallback, if subprocess setup fails, execute in-process with only the import hook (less isolated but functional for development).

Edit investigator/runtime/repl_loop.py to use the new sandbox for code execution instead of any existing in-process execution.

To validate, run the sandbox in isolation:

    uv run python -c "
    from investigator.runtime.repl_interpreter import execute_in_sandbox
    # Test 1: blocked module
    result = execute_in_sandbox('import os; print(os.getcwd())')
    assert 'ImportError' in result.stderr or result.returncode != 0, 'os should be blocked'
    print('PASS: os blocked')

    # Test 2: allowed module
    result = execute_in_sandbox('import json; print(json.dumps({\"a\": 1}))')
    assert result.returncode == 0, 'json should work'
    print('PASS: json allowed')

    # Test 3: output truncation
    result = execute_in_sandbox('print(\"x\" * 20000)')
    assert len(result.stdout) <= 8192, 'output should be truncated'
    print('PASS: output truncated')

    # Test 4: timeout
    import time
    start = time.time()
    result = execute_in_sandbox('import time; time.sleep(60)')
    elapsed = time.time() - start
    assert elapsed < 35, 'should timeout at 30s'
    print('PASS: timeout enforced')
    "

Note: the exact API of execute_in_sandbox may differ during implementation. Adjust the validation script to match. The key behaviors to verify are: blocked modules raise errors, allowed modules work, output is truncated, and long-running code is killed.

### Milestone 5: CLI Entrypoint

This milestone creates a single command to run RCA. All core components (REPL, recursion, sandbox) must be working.

Create investigator/rca/cli.py. Use Python's argparse module. The CLI accepts --trace-id for single-trace analysis or --manifest for batch analysis (these are mutually exclusive and one is required). Additional optional arguments: --phoenix-endpoint (default http://127.0.0.1:6006), --parquet (offline Parquet file path, skips Phoenix), --output-dir (default artifacts/investigator_runs), --max-iterations (default 40), --max-tool-calls (default 120), --max-depth (default 2), --max-wall-time (default 180), --model (default gpt-4o-mini), --no-writeback (flag to skip Phoenix annotations), --verbose (flag to print REPL trajectory).

For single-trace mode, the CLI creates a PhoenixInspectionAPI (or ParquetClient if --parquet is given), creates a TraceRCAEngine with the specified model and budget, runs the workflow via run_trace_rca_workflow, and prints the RCAReport JSON to stdout. Exit code is 0 for succeeded or partial, 1 for failed.

For batch mode, the CLI loads the manifest, iterates over all cases with non-null trace_ids, runs RCA for each, collects results, and prints a summary table showing run_id, trace_id, predicted label, expected label, and whether they match. It writes all run records to the output directory.

Add a __main__.py or if __name__ == "__main__" block so the CLI is runnable via "python -m investigator.rca.cli".

To validate:

    # Single trace (replace TRACE_ID with a real trace_id from manifest)
    uv run python -m investigator.rca.cli --trace-id TRACE_ID --verbose --no-writeback

    # Expected: RCAReport JSON printed to stdout, run_record.json in artifacts/

    # Batch mode
    uv run python -m investigator.rca.cli --manifest datasets/seeded_failures/manifest.json --no-writeback

    # Expected: summary table printed, 30 run records in artifacts/

    # Help text
    uv run python -m investigator.rca.cli --help

    # Expected: usage information with all arguments listed

### Milestone 6: End-to-End Evaluation

This milestone runs RCA on all 30 seeded failure traces and measures accuracy.

Create investigator/rca/evaluate.py. This script accepts --manifest (path to manifest.json) and --runs-dir (path to directory containing run record subdirectories). It loads the manifest to get the ground-truth expected_label for each case. It loads each run_record.json from the runs directory and extracts the primary_label from the output. It computes: top-1 accuracy (percentage where primary_label equals expected_label), per-label precision, recall, and F1 (treating each of the five taxonomy labels as a class), average confidence for correct versus incorrect predictions, average evidence_refs count for correct versus incorrect predictions, average cost in USD per run, average wall-time per run, and average budget utilization (percentage of max_tool_calls and max_iterations consumed). It prints a formatted report to stdout and writes the full evaluation to artifacts/evaluation/eval_report.json.

To run the full evaluation:

    # Step 1: Ensure Phoenix is running and traces are generated (Milestone 1)
    phoenix serve &

    # Step 2: Run RCA on all 30 traces
    uv run python -m investigator.rca.cli --manifest datasets/seeded_failures/manifest.json

    # Step 3: Evaluate
    uv run python -m investigator.rca.evaluate --manifest datasets/seeded_failures/manifest.json --runs-dir artifacts/investigator_runs

Expected output format:

    RLM-RCA Evaluation Report
    Dataset: seeded_failures_v1 (30 cases)
    Model: gpt-4o-mini

    Top-1 Accuracy: XX/30 (XX.X%)

    Per-Label Results:
      tool_failure            : P=X.XX  R=X.XX  F1=X.XX  (9 cases)
      retrieval_failure       : P=X.XX  R=X.XX  F1=X.XX  (8 cases)
      instruction_failure     : P=X.XX  R=X.XX  F1=X.XX  (2 cases)
      upstream_dep_failure    : P=X.XX  R=X.XX  F1=X.XX  (4 cases)
      data_schema_mismatch    : P=X.XX  R=X.XX  F1=X.XX  (7 cases)

    Cost:  avg $X.XX/run  total $X.XX
    Time:  avg Xs/run     total Xm
    Tokens: avg X,XXX/run

    Evidence Quality:
      Correct predictions:   avg X.X evidence_refs
      Incorrect predictions: avg X.X evidence_refs

The smoke test for single-trace end-to-end validation is:

    # Get first trace_id from manifest
    TRACE_ID=$(uv run python -c "import json; m=json.load(open('datasets/seeded_failures/manifest.json')); print([c['trace_id'] for c in m['cases'] if c['trace_id']][0])")

    # Run RCA
    uv run python -m investigator.rca.cli --trace-id $TRACE_ID --verbose

    # Verify run record exists
    ls artifacts/investigator_runs/*/run_record.json | head -1

    # Verify Phoenix annotations (open UI and check the trace)
    echo "Open http://127.0.0.1:6006 and navigate to trace $TRACE_ID to see rca.primary and rca.evidence annotations"

## Validation and Acceptance

The system is accepted when all of the following behaviors are observable:

Running "python -m investigator.rca.cli --trace-id TRACE_ID" produces a valid RCAReport JSON on stdout with a primary_label from the five-label taxonomy, confidence between 0.0 and 1.0, at least one evidence_ref, and a non-empty summary.

Running "python -m investigator.rca.cli --manifest datasets/seeded_failures/manifest.json" processes all 30 cases without crashes and produces 30 run_record.json files in the artifacts directory.

The evaluation report shows top-1 accuracy of at least 60% (18 out of 30 correct), with a target of 70-80%.

The total cost for 30 runs is under $2.00 USD.

Every run_record.json has all required fields: run_id, run_type, status, started_at, completed_at, runtime_ref with budget and usage, and either output_ref or error.

No run produces a SANDBOX_VIOLATION error, confirming that the import blocklist and subprocess isolation work correctly.

Phoenix annotations (rca.primary and rca.evidence) are visible in the Phoenix UI on analyzed traces, with labels and scores matching the RCAReport.

Running "import os" inside REPL-generated code produces an ImportError, confirming the sandbox.

The evaluation report is written to artifacts/evaluation/eval_report.json with all metrics.

## Idempotence and Recovery

All milestones are designed to be re-runnable. The fault injector can be run multiple times; it updates the manifest with new trace_ids (previous ones are overwritten). RCA runs produce new run records with unique run_ids (UUID-based), so re-running does not overwrite previous results. The evaluation script reads from whatever run records exist in the runs directory. If a milestone fails partway through, fix the issue and re-run the same commands.

If Phoenix is not running when the CLI is invoked, the error message should say "Could not connect to Phoenix at http://127.0.0.1:6006" and exit with code 1. Start Phoenix with "phoenix serve" and retry.

If the OpenAI API key is not set, the error should say "OPENAI_API_KEY environment variable is not set." Set it with "export OPENAI_API_KEY=sk-..." and retry.

If a REPL run fails due to budget exhaustion, the run record is still persisted with status "partial" or "terminated_budget" and whatever evidence was gathered is included. This is expected behavior, not an error.

## Artifacts and Notes

Expected artifacts after full completion:

    datasets/seeded_failures/manifest.json              # Updated with 30 non-null trace_ids
    datasets/seeded_failures/exports/spans.parquet       # Optional Parquet export for offline use
    artifacts/investigator_runs/<run_id>/run_record.json # One per RCA run (30 total for batch)
    artifacts/evaluation/eval_report.json                # Evaluation metrics
    artifacts/evaluation/baseline_v1.json                # Baseline for regression testing

New source files:

    apps/demo_agent/fault_injector.py                    # Fault injection harness
    apps/demo_agent/run_seeded_failures.py               # CLI for batch fault injection
    investigator/rca/cli.py                              # RCA CLI entrypoint
    investigator/rca/evaluate.py                         # Evaluation metrics script
    investigator/runtime/repl_interpreter.py             # Subprocess sandbox with import hooks

Modified source files:

    investigator/rca/engine.py                           # REPL-primary, per-hypothesis recursion
    investigator/runtime/repl_loop.py                    # Pre-filter context, hypothesis extraction
    investigator/runtime/recursive_loop.py               # Structured hypothesis results
    investigator/prompts/runtime/recursive_runtime_action_v1.md   # Hypothesis delegation prompt
    investigator/prompts/rca/trace_rca_judgment_v1.md    # Synthesis prompt

## Interfaces and Dependencies

External dependencies (managed via uv in pyproject.toml):

arize-phoenix is the trace storage and annotation backend. Pin a specific version to avoid API breakage. The key imports are phoenix.session.client.Client, phoenix.trace.SpanEvaluations, phoenix.trace.TraceEvaluations, and phoenix.otel.register.

openai is the LLM client. The system uses the Responses API with JSON schema mode for structured generation. The key class is openai.OpenAI.

llama-index is the agent framework used by the tutorial. The fault injector needs llama-index-core, llama-index-llms-openai, and llama-index-tools-* packages matching the tutorial.

pandas is used by PhoenixInspectionAPI for span dataframe operations.

Internal interfaces (all existing, documented here for the implementer):

TraceRCARequest is a dataclass or dict containing trace_id (string) and inspection_api (an InspectionAPI instance). It is the input to TraceRCAEngine.run().

TraceRCAEngine.run(request) returns an RCAReport dataclass with fields: trace_id, primary_label, summary, confidence, evidence_refs (list of EvidenceRef), remediation (list of str), gaps (list of str).

run_trace_rca_workflow(request, engine, ...) orchestrates a full run: calls engine.run(), writes results to Phoenix, persists run_record.json, and returns a tuple of (report, run_record).

ReplLoop.run(system_prompt, user_prompt, ...) runs the iterative REPL and returns a ReplLoopResult with status, output, usage, state_trajectory, and repl_trajectory.

RecursiveLoop.run(initial_actions, tool_registry, ...) executes recursive actions and returns a RecursiveLoopResult with status, output, usage, subcall_metadata, and state_trajectory.

ToolRegistry.call(tool_name, args) validates the tool is in the allowlist, sanitizes args against the method signature, invokes the InspectionAPI method, normalizes the result, and returns a dict with tool_name, normalized_args, args_hash, result, and response_hash.

SandboxGuard.validate_action(action_dict) raises SandboxViolationError if the action type is not in the allowlist, the tool name is not allowed, or the args are not JSON-safe.
