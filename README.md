# RLM AI Observability

RLM-powered AI observability investigation layer that automates trace root-cause analysis (RCA) and policy compliance with bounded recursive execution.

## What it does
- Runs recursive, evidence-driven investigations over AI traces.
- Produces structured, evidence-linked outputs for RCA and policy compliance.
- Enforces runtime guardrails (sandbox, tool allowlist, depth/iteration/cost budgets, wall-time).
- Emits replayable run artifacts for auditability.

## Included in this public snapshot
- `investigator/`: runtime, engines, prompt schemas, and proof harness.
- `tests/unit/`: unit coverage for recursive runtime, REPL loop, RCA, and compliance.
- `controls/library/controls_v1.json`: control library used by compliance flows.
- `artifacts/proof_runs/...`: two proof reports used as evidence.

## Evidence reports
- `artifacts/proof_runs/phase10-rca-only-canary-5-retrieverfix-20260211T021018Z/rca_only_report.json`
- `artifacts/proof_runs/phase10-compliance-only-canary-5-20260211T023031Z/compliance_only_report.json`
