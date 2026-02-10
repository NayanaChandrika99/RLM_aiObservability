# Trace Inspection API

This API is the read-only interface evaluators use to inspect Phoenix traces. It is the boundary that makes RCA, policy compliance, and incident dossiers auditable and reproducible.

Status: this spec is active for Phases 1-5 and is the primary runtime interface for RLM evaluators.

## Goals

- Read-only access to traces/spans/artifacts (no side effects).
- Stable evidence pointers: every claim can point to `trace_id`, `span_id`, and `artifact_id`.
- Deterministic behavior on a fixed trace dataset.
- Friendly to Phoenix export workflows: the API should be implementable using Phoenix’s span export/query surfaces.
- RLM-ready tool surface: function signatures are stable so recursive evaluators can call them directly.
- Policy-aware inspection: expose controls and required evidence definitions as read-only inputs.
- Shared runtime alignment: all three evaluator engines (RCA, compliance, incident) call this same API surface.

Normative companions:
- `specs/rlm_runtime_contract.md` for runtime behavior and sandbox constraints
- `specs/formal_contracts.md` for output and run artifact schema contracts

## IDs

- `trace_id`: Phoenix `context.trace_id` (string).
- `span_id`: Phoenix `context.span_id` (string).
- `artifact_id`: stable identifier for non-span evidence.

  Canonical `artifact_id` formats:
  - Tool I/O: `tool:<span_id>` (tool span id is stable and clickable).
  - Retrieval chunk: `retrieval:<span_id>:<document_position>:<document_id>` (append `:chunk=<chunk_id>` when available).
  - Config/deploy diff: `configdiff:<sha256>` (hash of diff bytes, normalized line endings).
  - Compliance evidence bundle: `compliance:<control_id>:<sha256>`

Canonical evidence pointer object (`evidence_ref`):

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
- `ref` must resolve to a known span/artifact identifier.
- `excerpt_hash` should be used for snippet integrity without storing full sensitive text.
- All evaluator outputs and Phoenix write-back payloads must use this exact `evidence_ref` shape.

## Data Structures

### `SpanSummary`

```json
{
  "trace_id": "string",
  "span_id": "string",
  "parent_id": "string|null",
  "name": "string",
  "span_kind": "LLM|TOOL|RETRIEVER|CHAIN|AGENT|RERANKER|EMBEDDING|EVALUATOR|GUARDRAIL|UNKNOWN",
  "status_code": "OK|ERROR|UNSET",
  "status_message": "string",
  "start_time": "RFC3339 timestamp",
  "end_time": "RFC3339 timestamp",
  "latency_ms": "number"
}
```

Span typing must follow OpenInference/OpenTelemetry conventions via `span_kind`, not ad-hoc span name matching.

### `SpanDetail`

```json
{
  "summary": { "$ref": "#/SpanSummary" },
  "attributes": { "type": "object" },
  "events": [
    { "name": "string", "timestamp": "RFC3339 timestamp", "attributes": { "type": "object" } }
  ]
}
```

### `ToolIO`

```json
{
  "trace_id": "string",
  "span_id": "string",
  "artifact_id": "string",
  "tool_name": "string",
  "input": "string|object|null",
  "output": "string|object|null",
  "status_code": "OK|ERROR|UNSET"
}
```

### `RetrievalChunk`

```json
{
  "trace_id": "string",
  "span_id": "string",
  "artifact_id": "string",
  "document_id": "string",
  "chunk_id": "string|null",
  "content": "string",
  "score": "number|null",
  "metadata": { "type": "object" }
}
```

### `SearchHit`

```json
{
  "trace_id": "string",
  "span_id": "string",
  "field": "string",
  "value_snippet": "string"
}
```

### `Message`

```json
{
  "trace_id": "string",
  "span_id": "string",
  "role": "system|user|assistant|tool",
  "content": "string",
  "metadata": { "type": "object" }
}
```

### `PolicyControl`

```json
{
  "controls_version": "string",
  "control_id": "string",
  "control_hash": "string",
  "title": "string",
  "description": "string",
  "severity": "critical|high|medium|low",
  "applies_when": { "type": "object" },
  "required_evidence": ["string"]
}
```

### `ComplianceFinding`

```json
{
  "controls_version": "string",
  "control_id": "string",
  "pass_fail": "pass|fail|not_applicable|insufficient_evidence",
  "severity": "critical|high|medium|low",
  "confidence": "number",
  "evidence_refs": [{ "$ref": "#/evidence_ref" }],
  "missing_evidence": ["string"],
  "remediation": "string"
}
```

### `ConfigSnapshot`

```json
{
  "snapshot_id": "string",
  "project_name": "string",
  "tag": "string",
  "created_at": "RFC3339 timestamp",
  "git_commit": "string|null",
  "paths": ["string"],
  "metadata": { "type": "object" }
}
```

### `ConfigDiff`

```json
{
  "project_name": "string",
  "base_snapshot_id": "string",
  "target_snapshot_id": "string",
  "artifact_id": "configdiff:<sha256>",
  "diff_ref": "string",
  "git_commit_base": "string|null",
  "git_commit_target": "string|null",
  "paths": ["string"],
  "summary": "string"
}
```

## Function Signatures (conceptual)

```python
def get_spans(trace_id: str, type: str | None = None) -> list[SpanSummary]: ...
def list_traces(
    project_name: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    filter_expr: str | None = None,
) -> list[dict]: ...

def list_spans(trace_id: str) -> list[SpanSummary]: ...
def get_span(span_id: str) -> SpanDetail: ...
def get_children(span_id: str) -> list[SpanSummary]: ...
def get_messages(span_id: str) -> list[Message]: ...

def get_tool_io(span_id: str) -> ToolIO | None: ...
def get_retrieval_chunks(span_id: str) -> list[RetrievalChunk]: ...
def list_controls(
    controls_version: str,
    app_type: str | None = None,
    tools_used: list[str] | None = None,
    data_domains: list[str] | None = None,
) -> list[PolicyControl]: ...
def get_control(control_id: str, controls_version: str) -> PolicyControl: ...
def required_evidence(control_id: str, controls_version: str) -> list[str]: ...

def list_config_snapshots(
    project_name: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    tag: str | None = None,
) -> list[ConfigSnapshot]: ...
def get_config_snapshot(snapshot_id: str) -> ConfigSnapshot: ...
def get_config_diff(
    base_snapshot_id: str,
    target_snapshot_id: str,
) -> ConfigDiff: ...

def search_trace(
    trace_id: str,
    pattern: str,
    fields: list[str] | None = None,
) -> list[SearchHit]: ...
def search(text_or_chunks: str | list[str], pattern: str) -> list[str]: ...
```

## Notes

- Evaluators must prefer evidence from `status_code == "ERROR"`, exceptions/events, and anomalous latencies before using an LLM judge.
- If required attributes for evidence are missing (e.g., no retrieval document IDs), the evaluator must surface a `gaps[]` entry rather than guessing.
- This API is read-only for trace inspection; evaluator run metadata is persisted separately as RunRecord-equivalent artifacts.
- `get_spans(...)` and `list_spans(...)` are equivalent; keep both names for compatibility with different evaluator toolchains.
- Compliance engine invocations must pass `controls_version`, and outputs must echo the same `controls_version` for auditability.

## Implementation Notes (Phoenix-backed)

The first implementation should be backed by Phoenix export APIs so it is easy to run locally:

- Phoenix server endpoint is the base URL (no path), e.g. `http://127.0.0.1:6006`.
- Tracing export endpoint for OTLP HTTP is `http://127.0.0.1:6006/v1/traces` (used by exporters, not by the Phoenix `Client` base URL).

Suggested building blocks:
- `phoenix.session.Client(endpoint=...)` (span export/query, annotations write-back).
- `phoenix.trace.dsl.SpanQuery` (filtering and selecting fields).

Determinism requirements:
- Stable ordering: when returning lists, sort by `start_time` then `span_id`.
- Stable selection: “top N hot spans” must define tie-breakers (latency desc, then span_id asc).

## RunRecord-equivalent Contract (required for evaluator runs)

Every RCA, policy-compliance, or incident-dossier evaluator invocation must emit exactly one run artifact, even when evaluation fails.

Path convention:
- `artifacts/investigator_runs/<run_id>/run_record.json`

Minimum object shape:

```json
{
  "run_id": "string",
  "run_type": "rca|policy_compliance|incident_dossier",
  "status": "succeeded|failed|partial|terminated_budget",
  "started_at": "RFC3339 timestamp",
  "completed_at": "RFC3339 timestamp",
  "dataset_ref": {
    "dataset_id": "string|null",
    "dataset_hash": "string|null"
  },
  "input_ref": {
    "project_name": "string",
    "time_window": {"start": "RFC3339|null", "end": "RFC3339|null"},
    "filter_expr": "string|null",
    "trace_ids": ["string"]
  },
  "model": {
    "provider": "openai",
    "name": "gpt-5-mini|gpt-5.2",
    "temperature": "number",
    "prompt_template_hash": "string",
    "evaluator_version": "string"
  },
  "budget": {
    "max_tool_calls": "integer",
    "max_subcalls": "integer",
    "max_tokens_total": "integer|null",
    "max_cost_usd": "number|null",
    "max_wall_time_sec": "integer",
    "sampling_seed": "integer|null"
  },
  "output_ref": {
    "schema_version": "string",
    "artifact_path": "string|null",
    "phoenix_annotation_ids": ["string"]
  },
  "error": {
    "code": "string",
    "message": "string"
  }
}
```

Rules:
- `error` is required when `status` is `failed` and optional otherwise.
- `output_ref.artifact_path` may be null for failed runs, but the run record itself must exist.
