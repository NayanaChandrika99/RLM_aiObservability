# Phoenix-RLM Investigator

Recursive Language Model runtime for AI observability investigations on agent traces.

## Benchmark Claim (TRAIL, GAIA Joint Accuracy)

| Reference | Score |
| --- | --- |
| [TRAIL paper](https://arxiv.org/pdf/2505.08638) | `0.183` |
| [AgentCompass paper](https://arxiv.org/pdf/2509.14647) | `0.239` |
| This repo (full-GAIA live campaign mean) | `0.2427` |

## What This Means For Autonomous Error Tracing

In this project, “beating the benchmark” means the system can automatically analyze full agent trajectories and produce better joint error tracing quality on TRAIL:
- It identifies both **what** error happened (category) and **where** it happened (location).
- It links findings to concrete evidence (`trace_id`, `span_id`, `artifact_id`).
- It produces reproducible, structured outputs for offline auditing of agentic runs.

## Core Capabilities

- **Trace RCA**: classify primary failure mode, point to evidence, and propose remediation.
- **Policy-to-Trace Compliance**: evaluate controls against observed trace behavior with evidence-backed verdicts.

## Quick Start

Prerequisites:
- Python 3.10+
- `OPENAI_API_KEY`
- Phoenix or parquet trace export

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install openai pandas pyarrow python-dotenv pytest requests
```

Run unit tests:

```bash
pytest tests/unit
```

Run canary proof:

```bash
python -m investigator.proof.repl_canary \
  --proof-run-id phase10-repl-canary-local \
  --manifest-path datasets/seeded_failures/manifest.json \
  --spans-parquet-path datasets/seeded_failures/exports/spans.parquet \
  --controls-dir controls/library \
  --trace-limit 5
```

## Key Paths

- `investigator/rca/`
- `investigator/compliance/`
- `investigator/runtime/`
- `investigator/prompts/`
- `tests/unit/`
- `controls/library/controls_v1.json`

## Detailed Docs

- `API.md`
- `ARCHITECTURE.md`
- `DESIGN.md`
- `EVALS.md`
- `PLANS.md`

Note: generated run outputs are intentionally gitignored.
