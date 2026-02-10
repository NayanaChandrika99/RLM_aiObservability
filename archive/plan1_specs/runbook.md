# Runbook (Commands and how to run the MVP)

## Purpose

Define the **canonical commands** for running the MVP locally:

- generate fixtures,
- run scenarios,
- resume HITL,
- run tests,
- and inspect artifacts.

This is a “commands” spec, intended to guide both humans and coding agents.

## Non-goals

- Not a production deployment guide.
- Not cloud infrastructure documentation.

## Assumptions (MVP)

- Python-first stack.
- In-process graph traversal (NetworkX).
- Digital Twin uses OR-Tools + SimPy.
- LLM usage (if needed) is API-hosted only.
- Default execution is `dry_run=true`.

## Environment setup

Preferred (recommended):

- Use `uv` for dependency management and running commands.

### Local environment variables (recommended: direnv)

Plan1 reads configuration from environment variables (it does not automatically load `.env`).

Recommended local setup uses `direnv` so variables are exported automatically when you `cd` into the repo:

1) Install `direnv` and hook it into your shell.
2) Create a local `.env` file from the template:

    cp .env.example .env

3) Create a local `.envrc` that loads `.env`:

    echo "dotenv .env" > .envrc

4) Allow it once:

    direnv allow

Notes:

- `.env` and `.envrc` are git-ignored (do not commit secrets).
- CI should set variables explicitly and run with `PLAN1_LLM_MODE=replay`.

Fallback:

- Use `python -m venv .venv` + `pip install -r requirements.txt`.

## Canonical commands (targets)

These commands are the **intended stable interface** once implemented.

### 1) Install dependencies

Preferred:

    uv sync

Fallback:

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt

### 2) Generate synthetic fixtures

Generate a synthetic KG and a twin state snapshot:

    uv run python -m plan1.kg.generate --seed 42 --out data/fixtures/kg

Expected outputs (files):

- `data/fixtures/kg/nodes.json`
- `data/fixtures/kg/edges.json`
- `data/fixtures/kg/twin_state.json`

Optional: validate the generated twin state snapshot:

    uv run python -m plan1.twin_state_validator --path data/fixtures/kg/twin_state.json

Expected behavior:

- No output on success
- Exit code `0` when valid, `1` when invalid

### 3) Run a scenario (dry-run)

Run a named scenario fixture and write artifacts (default is `dry_run=true`; use `--no-dry-run` to disable):

    uv run python -m plan1.demo --scenario scn_012_paths_found --dry-run --out artifacts/runs

Optional flags:

- `--seed <int>` (default `42`)
- `--now-iso <timestamp>` for deterministic time (e.g., `2026-01-20T10:00:00Z`)
- `--checkpoints <path>` to override the checkpointer path (default `artifacts/checkpoints.sqlite`)

Expected outputs:

- `artifacts/runs/<run_id>/run_record.json`
- optional per-artifact JSON files (implementation detail; but RunRecord must exist)
- for scenarios that declare `expected.kg_result` in their fixture, `run_record.json` includes `artifacts.kg_impact_map` and `artifacts.tier1_risk_table`
- for scenarios that include `plan_options` (and a `twin_state_path`) in their fixture, `run_record.json` includes `artifacts.plan_options`, `artifacts.twin_results`, and `artifacts.scorecard`

### 4) Run a manual incident (dry-run)

    uv run python -m plan1.demo --incident-text "Embargo affects aluminum exports from Country X" --dry-run --out artifacts/runs

Optional flags:

- `--seed <int>` (default `42`)
- `--now-iso <timestamp>` for deterministic time (e.g., `2026-01-20T10:00:00Z`)
- `--checkpoints <path>` to override the checkpointer path (default `artifacts/checkpoints.sqlite`)

Note: `plan1.demo --incident-text ...` is a minimal harness path. The long-term manual/demo surface is
`plan1.app` (LangGraph + LLM agents + HITL). See “App runtime (LangGraph + LLM agents)” below.

### 5) Resume a HITL run

If the run pauses for approval, resume it by providing the run/thread ID:

    uv run python -m plan1.runner resume --run-id <run_id> --decision APPROVED --decided-by nainy

Optional flags:

- `--runs-dir <path>` to override the runs directory (default `artifacts/runs`)
- `--checkpoints <path>` to override the checkpointer path (default `artifacts/checkpoints.sqlite`)

Checkpoint persistence (MVP):

- SQLite checkpointer stored at `artifacts/checkpoints.sqlite` (path may be configured, but must be documented).

### 6) Run tests

    uv run pytest

## App runtime (LangGraph + LLM agents)

These commands are the intended stable interface once implemented.

### 1) Run a manual incident through the agentic workflow

Environment (pinned initial values):

    export PLAN1_LLM_PROVIDER=openai
    export PLAN1_LLM_MODEL=gpt-4o-mini
    export OPENAI_API_KEY=...

Optional tracing to Phoenix:

    export PLAN1_OTEL_EXPORTER=phoenix
    export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006

Run:

    uv run python -m plan1.app run --incident-text "Flood in Taiwan semiconductor region" --dry-run --out artifacts/runs

Expected behavior:

- prints a `thread_id` if approval is required
- writes `run_record.json` only when the run completes (not when paused)

### 2) Resume a paused run (HITL)

    uv run python -m plan1.app resume --thread-id <thread_id> --decision APPROVED --decided-by nainy --out artifacts/runs

### 7) “Check all” (optional)

If a Makefile exists, prefer a single backpressure command:

    make check

## Artifact inspection checklist

For a completed run:

- Open `run_record.json` and verify:
  - `outcome` is present and valid,
  - `coverage_flag` is present,
  - artifacts are present/absent consistent with the outcome,
  - receipts exist if execution was attempted (even in dry-run).

## Acceptance checks

Once implemented:

- The commands above should work from repo root and produce the described outputs.
- The same scenario + seed should produce deterministic outcomes (modulo normalized timestamps).

## Keywords

runbook, commands, how to run, demo, fixtures, generate, pytest, resume, HITL, artifacts
