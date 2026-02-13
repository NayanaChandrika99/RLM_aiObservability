# RLM AI Observability

Recursive Language Model (RLM) runtime for AI observability investigations.

This project automates long-context investigation workflows on AI traces, with two production-focused capabilities:
- Root Cause Analysis (RCA)
- Policy-to-Trace Compliance

It is designed to run on top of Arize Phoenix or a compatible trace/data platform through a read-only inspection interface.

## Why This Exists

AI observability incidents are usually not a single error line. Evidence is spread across:
- trace trees and span events
- tool input/output artifacts
- retrieval context
- policy controls and required evidence

A one-shot prompt over all evidence is expensive and brittle for long-context analysis.  
This runtime uses bounded recursion and tool-driven evidence collection to keep runs auditable, controllable, and low-cost.

## What The System Does

For each investigation run:
1. Deterministically narrows candidate evidence (for example hot spans, scoped controls).
2. Executes a recursive or REPL-like runtime loop.
3. Uses allowlisted inspection tools to fetch only relevant evidence.
4. Optionally delegates focused sub-investigations.
5. Synthesizes and finalizes into structured JSON contracts.
6. Emits run artifacts with usage, trajectory, and evidence references.

All findings are evidence-linked (`trace_id`, `span_id`, `artifact_id`) and reproducible on the same dataset.

## Current Capabilities

### 1) Trace RCA
- Labels primary failure mode (for example retrieval failure, tool failure, schema mismatch, upstream dependency).
- Returns remediation guidance and evidence pointers.
- Supports deterministic mode, single-turn LLM mode, recursive planner mode, and REPL mode.

### 2) Policy Compliance
- Evaluates controls against trace behavior and evidence requirements.
- Produces per-control and overall verdicts (`pass`, `fail`, `needs_review`, `insufficient_evidence`).
- Returns missing evidence lists and confidence/severity context.

## Why This Is Different From A Generic Tool Agent

- Scoped subcalls: child investigations run with focused objective/context, not one growing chat transcript.
- Typed actions and typed outputs: runtime validates planner actions and schema outputs.
- Runtime budgets: depth, iterations, tool calls, tokens, cost, and wall-time are enforced.
- Sandbox and allowlist: only approved inspection actions can execute.
- Audit trail: each run records trajectory, subcall metadata, usage/cost, and outputs.

## Proof Snapshot (Current Evidence)

Evidence files in this repo:
- `artifacts/proof_runs/phase10-rca-only-canary-5-retrieverfix-20260211T021018Z/rca_only_report.json`
- `artifacts/proof_runs/phase10-compliance-only-canary-5-20260211T023031Z/compliance_only_report.json`

Measured on 5-trace canaries:
- RCA: baseline accuracy `0.0` -> RLM `0.8` (delta `+0.8`), succeeded `5/5`, wall-time partials `0`
- Compliance: baseline accuracy `0.8` -> RLM `1.0` (delta `+0.2`), succeeded `5/5`, partial rate `0`, failed `0`
- Approximate per-trace cost: RCA `~$0.0043`, Compliance `~$0.0094`

## Repository Layout

- `investigator/rca/` - RCA engine, workflow, write-back logic
- `investigator/compliance/` - compliance engine, workflow, write-back logic
- `investigator/runtime/` - shared runtime (planner, recursive loop, REPL loop, sandbox, budgets, tool registry)
- `investigator/inspection_api/` - read-only inspection clients/protocol
- `investigator/prompts/` - runtime and capability prompt templates + schemas
- `investigator/proof/` - benchmark/canary runners
- `tests/unit/` and `tests/integration/` - validation coverage
- `controls/library/controls_v1.json` - compliance control set

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (`OPENAI_API_KEY`)
- Trace source (Phoenix or parquet export)

Recommended packages:
- `openai`
- `pandas`
- `pyarrow`
- `python-dotenv`
- `pytest`
- `requests` (integration/live tests)

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install openai pandas pyarrow python-dotenv pytest requests
```

### Run Unit Tests

```bash
pytest tests/unit
```

### Run REPL Canary Proof (RCA + Compliance)

```bash
python -m investigator.proof.repl_canary \
  --proof-run-id phase10-repl-canary-local \
  --manifest-path datasets/seeded_failures/manifest.json \
  --spans-parquet-path datasets/seeded_failures/exports/spans.parquet \
  --controls-dir controls/library \
  --trace-limit 5
```

Note: dataset exports are intentionally excluded from this repo snapshot. Use your own parquet export and manifest to run full proofs.

## Runtime Outputs

Each evaluator invocation writes artifacts under:
- `artifacts/investigator_runs/<run_id>/run_record.json`
- `artifacts/investigator_runs/<run_id>/output.json`

Run records include:
- status/state trajectory
- usage and cost
- subcall metadata
- error blocks (for failed/partial runs)
- references needed for reproducibility

## Safety Model

- Read-only inspection API access from runtime tools
- Strict allowlist and argument validation
- Budget termination paths (including wall-time and recursion limits)
- Deterministic fallback behavior near budget exhaustion

## Scope and Next Steps

Current focus is RCA + policy compliance, with incident investigation path available for expansion.

Primary scale-up path:
- richer trace/log/metric connectors
- tighter benchmark gates by domain
- gradual rollout with budget policies and human review routing for low-confidence/high-severity runs

