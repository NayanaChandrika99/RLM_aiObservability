# RLM Engines Source of Truth

## Purpose

Define the three distinct RLM engines as independent capability implementations:

1. Trace RCA engine
2. Policy-to-Trace Compliance engine
3. Incident Dossier engine

This document is the authoritative behavior spec for engine-level design and acceptance.

## Scope

In scope:
- engine responsibilities
- input/output contracts
- recursive execution strategy per engine
- stopping criteria
- write-back behavior
- evaluation acceptance criteria

Out of scope:
- runtime sandbox internals (see `specs/rlm_runtime_contract.md`)
- shared JSON object definitions (see `specs/formal_contracts.md`)

## Shared Engine Principles

All engines must:

- use read-only inspection tools only
- produce schema-valid structured JSON
- attach evidence pointers to each non-trivial claim
- emit one RunRecord-equivalent artifact per invocation
- write results back to Phoenix annotations/evals

RLM usage statement:
- Engines use recursive selective inspection over external trace/policy context.
- Engines must not assume hidden learned memory; any intermediate state must be explicit in runtime variables or persisted artifacts.

## Shared Evidence Pointer Contract

All engines must use a single evidence pointer shape:

```json
{
  "trace_id": "string",
  "span_id": "string",
  "kind": "SPAN|TOOL_IO|RETRIEVAL_CHUNK|MESSAGE|CONFIG_DIFF",
  "ref": "string",
  "excerpt_hash": "string",
  "ts": "RFC3339|null"
}
```

Rules:
- `ref` must be resolvable via `span_id` and/or `artifact_id` conventions from `API.md`.
- `excerpt_hash` is required when a textual snippet influences a verdict.
- engines must prefer hashed excerpts over storing raw sensitive text in final outputs.

## Shared Phoenix Write-back Contract

Phoenix write-back rows must include:

- `name`
- `annotator_kind` (`LLM|HUMAN|CODE`)
- `label`
- `score`
- `explanation`

Trace-level annotation naming:
- RCA: `name=rca.primary`
- Compliance: `name=compliance.overall`
- Incident: `name=incident.dossier`

Span-level evidence naming:
- RCA evidence: `name=rca.evidence`
- Compliance evidence: `name=compliance.control.<control_id>.evidence`
- Incident evidence: `name=incident.timeline.evidence`

`annotator_kind` rules:
- `LLM` for model-generated judgments
- `CODE` for deterministic heuristics and selectors
- `HUMAN` for manual overrides/reviews

## OpenInference Span Kind Assumptions

All engines must use OpenInference/OpenTelemetry span semantics rather than ad-hoc names.

Primary span kinds:
- `LLM`
- `TOOL`
- `RETRIEVER`
- `AGENT`
- `CHAIN`
- `RERANKER`
- `EMBEDDING`
- `EVALUATOR`
- `GUARDRAIL`

If span kind is missing, engines must mark a gap and continue with conservative confidence.

## Shared Budget and Determinism Knobs

Each engine invocation must record:

- `max_tool_calls`
- `max_subcalls`
- `max_tokens_total` or `max_cost_usd`
- `max_wall_time_sec`
- deterministic narrowing rules in effect
- sampling seed when stochastic selection is used

## Engine 1: Trace RCA

### Problem Definition

Given a trace, identify the most likely failure class, provide evidence-linked explanation, and suggest remediation.

### Primary Input

```json
{
  "trace_id": "string",
  "project_name": "string",
  "dataset_ref": {"dataset_id": "string|null", "dataset_hash": "string|null"}
}
```

### Output

`RCAReport` (see `specs/formal_contracts.md`).

### Core Algorithm (REPL-Primary)

The RCA engine uses a REPL-primary execution model. Deterministic narrowing is a
pre-filter step inside the REPL, not a separate mode.

1. **Deterministic pre-filter** (no LLM calls):
   - Build candidate hot-span set: `status_code == ERROR`, exception events, highest latency
   - Stable sort: error → exception → latency desc → span_id asc
   - Select top-K hot spans (default K=5)
   - Collect branch context per hot span (BFS, depth≤2, nodes≤30)
2. **REPL loop starts** (model receives hot-span summary + tool access + analysis sandbox):
   - Model writes Python code to analyze hot spans, call tools, find patterns
   - Model identifies 1–4 candidate failure hypotheses from taxonomy:
     - retrieval_failure
     - tool_failure
     - instruction_failure
     - upstream_dependency_failure
     - data_schema_mismatch
3. **Per-hypothesis sub-calls** (one sub-call per hypothesis):
   - Each sub-call gets filtered span slice + hypothesis statement
   - Each sub-call explores evidence via tools + analysis code
   - Each sub-call returns: `{label, confidence, evidence_refs, gaps}`
4. **Root synthesis** (model compares sub-call results):
   - Pick primary label (highest evidence support)
   - Record rejected hypotheses + reasoning
   - Compute final confidence (evidence bonus rules)
5. Produce evidence-linked RCAReport with remediation and gaps.

### Recursion Strategy

- **Per-hypothesis decomposition**: root identifies candidate failure modes, spawns one
  sub-call per hypothesis with a focused objective and filtered span slice.
- Budget is **shared globally** across root + all sub-calls (not independent budgets).
- Stop when:
  - all hypothesis sub-calls complete
  - budget threshold reached (terminate with best-effort synthesis)
  - evidence coverage passes confidence threshold

### Evidence Policy

- Minimum 1 evidence pointer for low confidence output.
- Minimum 2 independent evidence pointers for medium/high confidence output.
- No label may be emitted with zero evidence pointers.

Independent evidence definition:
- Evidence pointers are independent only if they come from different evidence kinds, or from different artifact sources with distinct `ref` values.

### Phoenix Write-back

- Root annotation:
  - `name=rca.primary`
  - `annotator_kind=LLM` (or `CODE` for deterministic-only fallback)
  - `label=rca.primary_label`
  - `score=confidence`
  - `explanation=RCAReport JSON string`
- Span annotations for each evidence span:
  - `name=rca.evidence`
  - `annotator_kind=CODE` for narrowing, `LLM` for synthesized evidence links
  - `label=hot_span|tool_error|retrieval_signal|schema_error`
  - `score=evidence_weight` (optional)
  - `explanation=tiny JSON string with evidence pointer and why`

### Determinism and Budget Knobs

- `max_tool_calls`: default 120
- `max_subcalls`: default 40
- `max_tokens_total`: default 200000
- deterministic narrowing order:
  - error status first
  - then exception event presence
  - then latency descending
  - tie-break: `span_id` ascending

### Acceptance Criteria

- Majority label match on seeded failures.
- Reviewer can click from RCA to supporting spans/artifacts.
- Re-running on fixed dataset yields comparable outputs.

## Engine 2: Policy-to-Trace Compliance

### Problem Definition

Given a trace and controls library, evaluate each applicable control with auditable evidence-backed verdicts.

### Primary Input

```json
{
  "trace_id": "string",
  "project_name": "string",
  "controls_version": "string",
  "control_scope_override": ["string"]
}
```

### Output

`ComplianceReport` (see `specs/formal_contracts.md`).

### Core Algorithm

1. Control scoping:
   - infer app profile from trace/tool/domain features
   - retrieve controls via `list_controls(...)`
   - merge explicit override controls
2. For each applicable control:
   - fetch required evidence checklist via `required_evidence(control_id)`
   - recursively gather only required evidence from spans/messages/chunks/tool I/O
   - evaluate pass/fail/not_applicable/insufficient_evidence
3. Aggregate:
   - overall verdict
   - top failing controls by severity
   - remediation list
   - missing evidence summary

### Recursion Strategy

- Control-centric recursion:
  - one recursive chain per control
  - optional shared evidence cache across controls
- Stop when:
  - required evidence checklist complete
  - control verdict confidence threshold met
  - budget exhausted

### Evidence Policy

For each control verdict:

- Must include `control_id`.
- Must include `pass_fail`.
- Must include at least one evidence pointer unless `not_applicable`.
- Must include `insufficient_evidence` when checklist cannot be satisfied.
- Must include `missing_evidence[]` when `pass_fail=insufficient_evidence`.

### Governance Mode

This engine supports evidence-based approvals:

- Governance reviewers can inspect each control finding.
- Each finding links directly to trace spans/artifacts.
- Findings are auditable by `controls_version` and `run_id`.
- `controls_version` must be present in the output artifact and in root write-back explanation payload.

### Phoenix Write-back

- Root annotation:
  - `name=compliance.overall`
  - `annotator_kind=LLM`
  - `label=compliance.overall_verdict`
  - `score=overall_confidence`
  - `explanation=ComplianceReport JSON string (must include controls_version)`
- Span annotations:
  - `name=compliance.control.<control_id>.evidence`
  - `annotator_kind=LLM` (or `CODE` for deterministic checks)
  - `label=control_evidence|control_violation|control_gap`
  - `score=control_confidence`
  - `explanation=tiny JSON string with control finding and evidence pointer`

### Determinism and Budget Knobs

- `max_tool_calls`: default 160
- `max_subcalls`: default 60
- `max_tokens_total`: default 260000
- deterministic control ordering:
  - sort by severity desc, then `control_id` asc
- deterministic evidence ordering:
  - sort by `span_id`, then `ref`

### Acceptance Criteria

- Every applicable control produces a verdict.
- No failing control without evidence pointers.
- Governance reviewer can trace every fail decision to exact span/artifact references.

## Engine 3: Incident Dossier

### Problem Definition

Given incident trigger inputs, produce a coherent, ranked investigation dossier that links evidence, suspected changes, and actions.

### Primary Input

```json
{
  "project_name": "string",
  "time_window": {"start": "RFC3339", "end": "RFC3339"},
  "filter_expr": "string|null",
  "trace_ids_override": ["string"]
}
```

### Output

`IncidentDossier` (see `specs/formal_contracts.md`).

### Core Algorithm

1. Candidate trace retrieval:
   - from window/filter or explicit trace list
2. Representative trace selection:
   - deterministic rule-based selection contract
3. Recursive investigation:
   - inspect selected traces
   - gather cross-trace evidence clusters
   - attach config/deploy diff evidence
4. Hypothesis ranking:
   - rank by evidence strength and breadth
5. Action synthesis:
   - immediate mitigations
   - follow-up fixes

### Recursion Strategy

- Multi-trace hierarchical recursion:
  - trace-level loops
  - intra-trace span-level loops
  - cross-trace synthesis pass
- Stop when:
  - representative coverage target reached
  - top hypotheses are stable
  - budget exhausted

### Trace Selection Contract

Given candidate traces and target size `N`:

1. Select up to `K_error` traces by highest error signal.
2. Select up to `K_latency` traces by highest root latency.
3. Select up to `M_cluster` semantic cluster representatives when embeddings exist.
4. Dedupe by signature key `(service, tool_name, error_type)`.
5. Final sort by:
   - priority bucket (`error`, `latency`, `cluster`)
   - score descending
   - `trace_id` ascending

Parameters:
- `N = K_error + K_latency + M_cluster`
- default `N=12`, `K_error=5`, `K_latency=5`, `M_cluster=2`
- `sampling_seed` required when clustering has randomness

### Evidence Policy

- Timeline events must include at least one evidence pointer.
- Each hypothesis must include evidence pointer list.
- Suspected change must reference a config/deploy diff evidence object if asserted.
- If no diff evidence exists, suspected change must remain a hypothesis, not an asserted change.

### Phoenix Write-back

- Root annotations on representative traces:
  - `name=incident.dossier`
  - `annotator_kind=LLM`
  - `label=incident_dossier`
  - `score=confidence`
  - `explanation=IncidentDossier JSON string`
- Optional span annotations for key timeline events.
  - `name=incident.timeline.evidence`
  - `annotator_kind=CODE` (selection) or `LLM` (synthesis)
  - `label=timeline_event|hypothesis_evidence|change_evidence`
  - `score=evidence_weight` (optional)
  - `explanation=tiny JSON string with timeline evidence`

### Determinism and Budget Knobs

- `max_tool_calls`: default 220
- `max_subcalls`: default 90
- `max_tokens_total`: default 320000
- `max_cost_usd`: optional hard cap
- `sampling_seed`: required whenever cluster representative selection is enabled

### Acceptance Criteria

- Dossier includes timeline, hypotheses, actions, and confidence/gaps.
- Reviewer can inspect why each representative trace was selected.
- Dossier evidence links resolve to valid trace/span/artifact ids.

## Engine Differences Matrix

| Dimension | Trace RCA | Policy Compliance | Incident Dossier |
|---|---|---|---|
| Unit of analysis | single trace | single trace + controls | trace set in incident window |
| Main objective | root cause label | control verdicts | multi-trace synthesis |
| Recursion style | span-branch | per-control evidence loops | hierarchical cross-trace |
| Primary output | `RCAReport` | `ComplianceReport` | `IncidentDossier` |
| Reviewer persona | SRE/ML engineer | governance/risk | incident commander |
| Key write-back name | `rca.primary` | `compliance.overall` | `incident.dossier` |

## Shared Runtime Dependencies

All engines depend on:

- `specs/rlm_runtime_contract.md`
- `API.md` read-only tools
- `specs/formal_contracts.md` schema contracts
- `artifacts/investigator_runs/<run_id>/run_record.json`

## Test Strategy by Engine

RCA:
- seeded failure trace label tests
- evidence pointer existence tests

Compliance:
- control coverage tests
- no-evidence fail prevention tests
- insufficient-evidence path tests

Incident:
- representative selection tests
- hypothesis ranking determinism tests
- config diff evidence linkage tests

## Keywords

rlm engines, trace rca, policy compliance, incident dossier, recursive evaluator, evidence-based approvals, phoenix annotations
