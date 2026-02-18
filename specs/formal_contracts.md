# Formal Contracts

## Purpose

Define machine-oriented contract shapes for runtime artifacts, evaluator outputs, and policy controls.

This spec is normative for:
- schema fields and required keys
- enum vocabularies
- id formats
- compatibility/versioning rules

## Scope

In scope:
- `RunRecord`
- `RCAReport`
- `ComplianceReport`
- `IncidentDossier`
- `PolicyControl`
- annotation payload contract

Out of scope:
- database schema
- connector storage schema

## Global Conventions

- Key format: `snake_case`
- ID type: string
- Time format: RFC3339 UTC
- Confidence range: `0.0..1.0`
- Severity enum: `critical|high|medium|low`
- Pass/fail enum: `pass|fail|not_applicable|insufficient_evidence`
- Annotator kind enum: `LLM|HUMAN|CODE`

## Canonical ID Formats

- `trace_id`: Phoenix trace id string
- `span_id`: Phoenix span id string
- `artifact_id`:
  - `tool:<span_id>`
  - `retrieval:<span_id>:<document_position>:<document_id>[:chunk=<chunk_id>]`
  - `configdiff:<sha256>`
  - `compliance:<control_id>:<sha256>`

## Canonical EvidenceRef

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
- `ref` must be a resolvable identifier under canonical id formats.
- `excerpt_hash` is required when a text excerpt influences a verdict.

## Contract 1: RunRecord

### Purpose

Persist one auditable record for each evaluator invocation (success, partial, or failure).

### Required fields

```json
{
  "schema_version": "1.0.0",
  "run_id": "string",
  "run_type": "rca|policy_compliance|incident_dossier",
  "status": "succeeded|failed|partial|terminated_budget",
  "started_at": "RFC3339",
  "completed_at": "RFC3339",
  "dataset_ref": {
    "dataset_id": "string|null",
    "dataset_hash": "string|null"
  },
  "input_ref": {
    "project_name": "string|null",
    "trace_ids": ["string"],
    "time_window": {"start": "RFC3339|null", "end": "RFC3339|null"},
    "filter_expr": "string|null",
    "controls_version": "string|null"
  },
  "runtime_ref": {
    "engine_version": "string",
    "model_provider": "openai",
    "model_name": "string",
    "temperature": "number",
    "prompt_template_hash": "string",
    "budget": {
      "max_iterations": "integer",
      "max_depth": "integer",
      "max_tool_calls": "integer",
      "max_subcalls": "integer",
      "max_tokens_total": "integer|null",
      "max_cost_usd": "number|null",
      "sampling_seed": "integer|null",
      "max_wall_time_sec": "integer"
    },
    "usage": {
      "iterations": "integer",
      "depth_reached": "integer",
      "tool_calls": "integer",
      "llm_subcalls": "integer",
      "tokens_in": "integer",
      "tokens_out": "integer",
      "cost_usd": "number"
    },
    "state_trajectory": ["string"],
    "subcall_metadata": ["object"],
    "repl_trajectory": ["object"]
  },
  "output_ref": {
    "artifact_type": "RCAReport|ComplianceReport|IncidentDossier|null",
    "artifact_path": "string|null",
    "schema_version": "string|null"
  },
  "writeback_ref": {
    "phoenix_annotation_ids": ["string"],
    "writeback_status": "succeeded|partial|failed",
    "annotation_names": ["string"],
    "annotator_kinds": ["LLM|HUMAN|CODE"]
  },
  "error": {
    "code": "string",
    "message": "string",
    "stage": "string",
    "retryable": "boolean"
  }
}
```

### Rules

- `error` is required for `status=failed`.
- `output_ref.artifact_path` is required for `status=succeeded`.
- exactly one RunRecord per invocation.

## Contract 2: RCAReport

```json
{
  "schema_version": "1.0.0",
  "trace_id": "string",
  "primary_label": "retrieval_failure|tool_failure|instruction_failure|upstream_dependency_failure|data_schema_mismatch",
  "summary": "string",
  "confidence": "number",
  "evidence_refs": [
    {
      "trace_id": "string",
      "span_id": "string",
      "kind": "SPAN|TOOL_IO|RETRIEVAL_CHUNK|MESSAGE|CONFIG_DIFF",
      "ref": "string",
      "excerpt_hash": "string",
      "ts": "RFC3339|null"
    }
  ],
  "remediation": ["string"],
  "gaps": ["string"]
}
```

Note: the field is `evidence_refs` (flat list of canonical `EvidenceRef` objects), matching
`investigator/runtime/contracts.py`. The previous `evidence` array with nested
`artifact_id`/`artifact_type`/`why` is superseded.

Rules:
- `evidence_refs` must be non-empty.
- `confidence` must be in `0..1`.

## Contract 3: ComplianceReport

```json
{
  "schema_version": "1.0.0",
  "trace_id": "string",
  "controls_version": "string",
  "controls_evaluated": [
    {
      "control_id": "string",
      "pass_fail": "pass|fail|not_applicable|insufficient_evidence",
      "severity": "critical|high|medium|low",
      "confidence": "number",
      "evidence": {
        "trace_id": "string",
        "span_ids": ["string"],
        "artifact_ids": ["string"],
        "excerpt_hashes": ["string"],
        "evidence_refs": ["EvidenceRef"]
      },
      "missing_evidence": ["string"],
      "remediation": "string"
    }
  ],
  "overall_verdict": "compliant|non_compliant|needs_review",
  "overall_confidence": "number",
  "gaps": ["string"]
}
```

Rules:
- every applicable control must appear exactly once in `controls_evaluated`.
- `fail` verdict must include at least one evidence pointer.
- `insufficient_evidence` must be used when required evidence is missing.
- `insufficient_evidence` must include non-empty `missing_evidence`.

## Contract 4: IncidentDossier

```json
{
  "schema_version": "1.0.0",
  "incident_id": "string",
  "incident_summary": "string",
  "impacted_components": ["string"],
  "timeline": [
    {
      "timestamp": "RFC3339",
      "event": "string",
      "evidence": {
        "trace_ids": ["string"],
        "span_ids": ["string"],
        "artifact_ids": ["string"],
        "evidence_refs": ["EvidenceRef"]
      }
    }
  ],
  "representative_traces": [
    {
      "trace_id": "string",
      "why_selected": "string"
    }
  ],
  "suspected_change": {
    "change_type": "deploy|config|prompt|dependency|unknown",
    "change_ref": "string",
    "diff_ref": "EvidenceRef",
    "summary": "string"
  },
  "hypotheses": [
    {
      "rank": "integer",
      "statement": "string",
      "confidence": "number",
      "evidence": {
        "trace_ids": ["string"],
        "span_ids": ["string"],
        "artifact_ids": ["string"],
        "evidence_refs": ["EvidenceRef"]
      }
    }
  ],
  "recommended_actions": [
    {
      "priority": "P0|P1|P2",
      "type": "mitigation|follow_up_fix",
      "action": "string"
    }
  ],
  "confidence": "number",
  "gaps": ["string"]
}
```

Rules:
- `representative_traces` must be non-empty.
- each hypothesis must include evidence.
- timeline must be chronologically sorted ascending.
- if `suspected_change` is asserted, `diff_ref` must be present.

## Contract 5: PolicyControl

```json
{
  "control_id": "string",
  "version": "string",
  "title": "string",
  "description": "string",
  "severity": "critical|high|medium|low",
  "applies_when": {
    "app_types": ["string"],
    "tools": ["string"],
    "data_domains": ["string"]
  },
  "required_evidence": ["string"],
  "decision_logic_hint": "string",
  "remediation_template": "string"
}
```

Rules:
- `control_id + version` must be unique in control library.
- `required_evidence` must be non-empty for controls that can fail.

## Contract 6: Annotation Payload

```json
{
  "run_id": "string",
  "engine_type": "rca|policy_compliance|incident_dossier",
  "schema_version": "string",
  "name": "string",
  "annotator_kind": "LLM|HUMAN|CODE",
  "label": "string",
  "score": "number|null",
  "explanation_json": "string",
  "target_span_id": "string|null"
}
```

Rules:
- `explanation_json` must parse to one of the formal output contracts.
- `run_id` is mandatory for all write-back rows.
- `name` is mandatory and used as the Phoenix annotation metric key.
- `annotator_kind` is mandatory.

## Compatibility and Versioning

- Backward-compatible additions:
  - optional fields may be added
  - enum values may be extended only with explicit version bump
- Breaking changes:
  - removing required fields
  - changing enum semantics
  - changing id formats

Versioning rules:

- `schema_version` uses semantic versioning.
- major bump for breaking changes.
- minor bump for additive non-breaking changes.
- patch bump for clarifications with identical shape.

## Validation Requirements

Before persistence/write-back:

1. Validate against formal contract.
2. Validate evidence pointers reference known ids.
3. Validate confidence ranges and enum values.
4. Validate required sections are present for engine type.

## Acceptance Checks

1. Contract fixtures parse and validate.
2. Invalid enum values fail validation.
3. Missing required evidence fields fail validation.
4. Wrong id format fails validation.
5. RunRecord required-on-failure behavior is enforced.

## Keywords

formal schema, contract specification, rca report, compliance report, incident dossier, run record, annotations contract, policy controls
