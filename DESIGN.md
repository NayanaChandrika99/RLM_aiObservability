# Design

This repo builds Phoenix-RLM Investigator in a trace-first, evaluator-first way:
- Phase 1 proves we can generate and export traces reliably.
- Phase 2 implements RLM Trace RCA and writes results back into Phoenix.
- Phase 3 implements the RLM Policy-to-Trace Compliance Auditor and writes results back into Phoenix.
- Phase 4 implements the RLM incident investigator and writes results back into Phoenix.
- Phase 5 hardens and expands the same RLM runtime without changing output contracts.

Status: RLM is in active scope across Phases 2-5. We run three separate engines over one shared runtime contract. Core interfaces are stable from the start:
- output contracts (RCA/compliance/dossier JSON)
- sandbox restrictions (no network/filesystem; API-only)
- Inspection API surface (`API.md`)

Normative internal specs:
- Runtime behavior and sandbox rules: `specs/rlm_runtime_contract.md`
- Engine-level source of truth: `specs/rlm_engines.md`
- Formal schema contracts: `specs/formal_contracts.md`

## Engine Topology (locked)

- Trace RCA engine: trace-level failure labeling with evidence-linked remediation.
- Policy-to-Trace Compliance engine: control-level compliance verdicts with auditable evidence bundles.
- Incident dossier engine: incident-window synthesis across representative traces and config context.
- Shared runtime contract: all three engines use the same recursion, budget, sandbox, and run-record rules defined in `specs/rlm_runtime_contract.md`.

## External References

- Recursive Language Models paper: `rlm/2512.24601v2.pdf`
- Practical RLM security audit example: `https://kmad.ai/Recursive-Language-Models-Security-Audit`

Patterns we reuse:
- Prompt-as-environment: treat traces and policy controls as external objects inspected via tools, not stuffed into one prompt.
- Recursive sub-calls: delegate focused subproblems and synthesize outputs into schema-bound artifacts.
- Persistent program state: keep intermediate evidence in runtime state and return structured final variables.
- Budgeted recursion: enforce max iterations, depth, and tool-call quotas to control cost and failure modes.
- Auditability: persist trajectory-linked evidence pointers and per-run artifacts for replay and governance review.

## Mandatory Evaluator Run Artifact

Every RCA/compliance/dossier evaluator invocation must produce one RunRecord-equivalent artifact, regardless of outcome.

Storage convention:
- `artifacts/investigator_runs/<run_id>/run_record.json`

Behavioral rules:
- `status=succeeded`: run record includes output references and Phoenix write-back references.
- `status=partial`: run record includes partial output references plus explicit `gaps`.
- `status=failed`: run record includes a non-empty `error` block and no silent drop.
- One invocation -> one run record.

## Output Contracts (stable)

All outputs are structured JSON and must include evidence pointers (`trace_id`, `span_id`, `kind`, `ref`, `excerpt_hash`, `ts`).
The canonical pointer object is `evidence_ref` from `API.md` and must be reused without field drift.

### RCA JSON (trace-level)

```json
{
  "schema_version": "1.0.0",
  "trace_id": "…",
  "primary_label": "retrieval_failure|tool_failure|instruction_failure|upstream_dependency_failure|data_schema_mismatch",
  "summary": "…",
  "evidence_refs": [
    {
      "trace_id": "…",
      "span_id": "…",
      "kind": "SPAN|TOOL_IO|RETRIEVAL_CHUNK|MESSAGE|CONFIG_DIFF",
      "ref": "…",
      "excerpt_hash": "…",
      "ts": "RFC3339|null"
    }
  ],
  "gaps": ["…"],
  "confidence": 0.0
}
```

### Incident dossier JSON (incident-level)

```json
{
  "schema_version": "1.0.0",
  "incident_summary": "…",
  "impacted_components": ["…"],
  "timeline": [
    {
      "timestamp": "…",
      "event": "…",
      "evidence_refs": [
        {
          "trace_id": "…",
          "span_id": "…",
          "kind": "SPAN|TOOL_IO|RETRIEVAL_CHUNK|MESSAGE|CONFIG_DIFF",
          "ref": "…",
          "excerpt_hash": "…",
          "ts": "RFC3339|null"
        }
      ]
    }
  ],
  "representative_traces": [
    {
      "trace_id": "…",
      "why_selected": "…",
      "evidence_refs": [
        {
          "trace_id": "…",
          "span_id": "…",
          "kind": "SPAN|TOOL_IO|RETRIEVAL_CHUNK|MESSAGE|CONFIG_DIFF",
          "ref": "…",
          "excerpt_hash": "…",
          "ts": "RFC3339|null"
        }
      ]
    }
  ],
  "suspected_change": {
    "change_type": "deploy|config|prompt|dependency|unknown",
    "change_ref": "…",
    "diff_ref": "configdiff:<sha256>|null",
    "summary": "…",
    "evidence_refs": [
      {
        "trace_id": "…",
        "span_id": "…",
        "kind": "CONFIG_DIFF",
        "ref": "configdiff:<sha256>",
        "excerpt_hash": "…",
        "ts": "RFC3339|null"
      }
    ]
  },
  "hypotheses": [
    {
      "rank": 1,
      "statement": "…",
      "evidence_refs": [
        {
          "trace_id": "…",
          "span_id": "…",
          "kind": "SPAN|TOOL_IO|RETRIEVAL_CHUNK|MESSAGE|CONFIG_DIFF",
          "ref": "…",
          "excerpt_hash": "…",
          "ts": "RFC3339|null"
        }
      ],
      "confidence": 0.0
    }
  ],
  "recommended_actions": [
    { "priority": "P0|P1|P2", "action": "…", "type": "mitigation|follow_up_fix" }
  ],
  "confidence": 0.0,
  "gaps": ["…"]
}
```

### Policy compliance JSON (trace-level)

```json
{
  "schema_version": "1.0.0",
  "trace_id": "…",
  "controls_version": "…",
  "controls_evaluated": [
    {
      "controls_version": "…",
      "control_id": "…",
      "pass_fail": "pass|fail|not_applicable|insufficient_evidence",
      "severity": "critical|high|medium|low",
      "confidence": 0.0,
      "evidence_refs": [
        {
          "trace_id": "…",
          "span_id": "…",
          "kind": "SPAN|TOOL_IO|RETRIEVAL_CHUNK|MESSAGE|CONFIG_DIFF",
          "ref": "…",
          "excerpt_hash": "…",
          "ts": "RFC3339|null"
        }
      ],
      "missing_evidence": ["…"],
      "remediation": "…"
    }
  ],
  "overall_verdict": "compliant|non_compliant|needs_review",
  "overall_confidence": 0.0,
  "gaps": ["…"]
}
```

## RCA Taxonomy (v1)

- `retrieval_failure`: wrong/irrelevant context retrieved or missing needed context.
- `tool_failure`: tool execution errors, timeouts, or wrong tool chosen.
- `instruction_failure`: prompt/system instruction drift, format/schema noncompliance, incorrect task framing.
- `upstream_dependency_failure`: API errors/timeouts/rate limits upstream of the agent.
- `data_schema_mismatch`: tool outputs that cannot be parsed/validated, or schema drift between components.

## Evaluator Recipes (RLM-first)

### Trace RCA evaluator

1. Deterministic narrowing:
   - pick “hot spans”: `status_code == ERROR`, exception events, retries (if present), slowest spans
2. Recursive evidence extraction:
   - tool spans: capture tool name + input/output refs
   - retriever spans: capture retrieved document IDs/chunk IDs + scores
3. RLM synthesis:
   - recursively inspect suspicious spans/chunks, then produce RCA JSON using stable IDs

### Incident dossier evaluator

1. Select representative traces for a time window (errors, p95 latency, or semantic clustering).
2. Recursively explore representative traces and drill into hot spans/chunks.
3. Attach deploy/config diffs as evidence objects (V1) via config snapshot and diff APIs.
4. Add logs/metrics correlation via connectors later (V2).

### Policy-to-Trace Compliance auditor

1. Scope controls:
   - map trace context (app type, tools used, data domains) to applicable controls
2. Recursive evidence gathering:
   - inspect relevant spans/messages/retrieval chunks/tool I/O for each control
   - use required evidence definitions to avoid over-scanning unrelated context
3. Control verdicts:
   - emit one structured finding per applicable control
4. Write-back:
   - annotate root span with overall verdict
   - annotate evidence spans with control-specific findings

## Phase 1: Seeded Failure Dataset Design

Goal: generate 30–100 traces with intentional failure cases and maintain deterministic scoring without contaminating trace metadata with labels.

### Principles

- Keep traces “production-like”: do not embed the expected RCA label in span attributes.
- Keep scoring deterministic: store ground truth externally in a manifest.
- Ensure each run is traceable: include a `run_id` in logs/artifacts (and optionally in trace metadata as a neutral correlation id).

### Manifest (external ground truth)

Store under `datasets/seeded_failures/manifest.json` (or `.yaml`).

Minimum fields:

```json
{
  "dataset_id": "seeded_failures_v1",
  "generator_version": "0.1.0",
  "cases": [
    {
      "run_id": "uuid-or-stable-id",
      "trace_id": "optional-if-known",
      "expected_label": "tool_failure",
      "notes": "short description of the injected failure"
    }
  ]
}
```

Failure classes to cover (v1):
- tool errors (forced exception / timeout)
- ambiguous entity resolution (two similar entities)
- prompt regression (format drift / schema noncompliance)
- retrieval mistakes (deferred until retrieval is present)

### Trace exports (Parquet-first)

Exported spans should be stored as Parquet to support:
- stable schemas
- fast filtering/groupby for evaluator pipelines
- reproducible re-runs without requiring a live Phoenix server

Conventions:
- store exports under `datasets/seeded_failures/exports/`
- prefer `spans.parquet` as the primary table (with Phoenix column naming)
- keep manifests and generator code committed; keep Parquet exports gitignored by default

## Phase 2: RCA Output and Write-back Strategy

We write RCA in two layers for usability:
- trace-level: annotate the root span with the RCA label + confidence for filtering.
- span-level: annotate each hot span with evidence pointers and short explanations.

Recommended annotation naming (v1):
- root/trace result: `rca.primary`
- root/trace score: `rca.primary` `score` field
- full payload: `rca.primary` `explanation` JSON string

Deterministic narrowing (hot spans) should be reproducible:
- prefer spans with `status_code == "ERROR"`
- then spans with exception events
- then top latency spans
- define stable tie-breakers (latency desc, then span_id asc)

RLM runtime settings (Phases 2-4):
- default model: `gpt-4o-mini` (all calls — root and sub-calls)
- optional upgrade for root synthesis: `gpt-4o` or `gpt-5.2`
- run at temperature 0.0; enforce JSON schema output; record prompt/template hash

## Phase 3: Policy-to-Trace Compliance Auditor Design (V1)

Primary input:
- `trace_id`
- `project_name`
- `controls_version` (control library tag/hash)

Optional input:
- explicit list of `control_id[]` overrides for targeted review

Control scoping (v1):
- infer candidate controls from tools used, data domains, and app profile
- include explicit control overrides even if scoping would skip them

Verdict rules (v1):
- every applicable control gets exactly one verdict object
- each verdict must include `evidence_refs[]` using the canonical `evidence_ref` object
- if required evidence is missing, emit `insufficient_evidence` instead of guessing

Write-back strategy:
- trace/root annotation: overall compliance verdict + confidence
- span-level annotations: per-control findings at evidence spans
- every finding and root verdict must include the same `controls_version`

## Phase 4: Incident Dossier Design (V1)

Primary input:
- `project_name`
- `time_window` (start/end)
- `filter_expr` (Phoenix span/trace filter DSL)

Optional override:
- explicit `trace_id[]`

Representative trace selection (v1):
- include traces with error spans
- include traces near p95 latency (from root span latency)
- cap N (e.g., 5–20) for cost control

Config/deploy context (V1):
- store snapshots under `configs/snapshots/<tag>/`
- compute diffs between “last known good” and “current”
- represent each diff as an evidence object with `artifact_id = configdiff:<sha256>`
- include git commit hash + file paths in the dossier for auditability
- access context through Inspection API contracts (`list_config_snapshots`, `get_config_snapshot`, `get_config_diff`)

## Writing Results Back Into Phoenix

Phoenix’s eval/annotation loop is the backbone:
- Export spans from Phoenix.
- Produce an annotations dataframe keyed by `context.span_id`.
- Log annotations back to Phoenix so results render in the UI.

Mapping guidance (v1):
- `name`: metric key (`rca.primary`, `compliance.overall`, `incident.dossier`, and evidence names)
- `annotator_kind`: `LLM|CODE|HUMAN`
- `label`: RCA primary label, control verdict label, or `incident_dossier` for dossier attachments.
- `score`: use `confidence` (0–1) or optional evidence weight.
- `explanation`: JSON string of the full RCA/compliance/dossier payload (include evidence IDs).

Recommended annotation names:
- trace-level:
  - `rca.primary`
  - `compliance.overall`
  - `incident.dossier`
- span-level evidence:
  - `rca.evidence`
  - `compliance.control.<control_id>.evidence`
  - `incident.timeline.evidence`

RunRecord linkage (required):
- Each Phoenix write-back payload must include `run_id` so UI annotations can be traced to the run artifact.
- Run records must include any Phoenix annotation/eval identifiers returned by write-back.

## Reproducibility Requirements

Every evaluator run must record enough metadata to reproduce results on the same dataset:
- `dataset_id` (or a deterministic hash of exported spans)
- evaluator version
- model name + parameters
- prompt/template hash
- run id (timestamped identifier is fine)
- invocation status + error metadata (when present)

Operational stance for CI:
- CI replay mode for LLM calls is optional for Phases 1-4.
- Mandatory guarantees remain: schema-valid outputs, reproducible run metadata, and deterministic pre-LLM narrowing.

## Cost and Safety Controls

- Prefer deterministic narrowing and sampling before LLM calls.
- Keep a default daily budget target (e.g., ~$5/day) while the harness is unstable.
- Avoid running evaluators on every span; evaluate:
  - root spans for trace-level classification
  - hot spans for evidence-level annotation
- Preserve one run record per invocation even when budget limits terminate the run early.

## Phase 5 Hardening Focus

Phase 5 extends the same RLM runtime already used in Phases 2-4:
- enforce recursion budgets and per-tool call quotas
- add stronger sandbox validation and adversarial tool-call tests
- add robustness checks for malformed spans/artifacts and partial data
- scale representative-trace selection and multi-trace synthesis while preserving schema compatibility
