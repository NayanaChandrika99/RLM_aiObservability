# ABOUTME: Defines contract-aligned runtime and output dataclasses for RLM investigator engines.
# ABOUTME: Provides stable serialization helpers used by runtime orchestration and engine stubs.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any, Literal


RUN_SCHEMA_VERSION = "1.0.0"

RunType = Literal["rca", "policy_compliance", "incident_dossier"]
RunStatus = Literal["succeeded", "failed", "partial", "terminated_budget"]


def utc_now_rfc3339() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_excerpt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class EvidenceRef:
    trace_id: str
    span_id: str
    kind: Literal["SPAN", "TOOL_IO", "RETRIEVAL_CHUNK", "MESSAGE", "CONFIG_DIFF"]
    ref: str
    excerpt_hash: str
    ts: str | None = None


@dataclass
class RuntimeBudget:
    max_iterations: int = 40
    max_depth: int = 2
    max_tool_calls: int = 200
    max_subcalls: int = 80
    max_tokens_total: int | None = 300000
    max_cost_usd: float | None = None
    sampling_seed: int | None = None
    max_wall_time_sec: int = 180


@dataclass
class RuntimeUsage:
    iterations: int = 0
    depth_reached: int = 0
    tool_calls: int = 0
    llm_subcalls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class DatasetRef:
    dataset_id: str | None = None
    dataset_hash: str | None = None


@dataclass
class TimeWindow:
    start: str | None = None
    end: str | None = None


@dataclass
class InputRef:
    project_name: str | None = None
    trace_ids: list[str] = field(default_factory=list)
    time_window: TimeWindow = field(default_factory=TimeWindow)
    filter_expr: str | None = None
    controls_version: str | None = None


@dataclass
class RuntimeRef:
    engine_version: str
    model_provider: str
    model_name: str
    temperature: float
    prompt_template_hash: str
    budget: RuntimeBudget
    usage: RuntimeUsage
    state_trajectory: list[str] = field(default_factory=list)
    subcall_metadata: list[dict[str, Any]] = field(default_factory=list)
    repl_trajectory: list[dict[str, Any]] = field(default_factory=list)
    scaffold: str | None = None


@dataclass
class OutputRef:
    artifact_type: Literal["RCAReport", "ComplianceReport", "IncidentDossier"] | None
    artifact_path: str | None
    schema_version: str | None


@dataclass
class WritebackRef:
    phoenix_annotation_ids: list[str] = field(default_factory=list)
    writeback_status: Literal["succeeded", "partial", "failed"] = "partial"
    annotation_names: list[str] = field(default_factory=list)
    annotator_kinds: list[Literal["LLM", "HUMAN", "CODE"]] = field(default_factory=list)


@dataclass
class RunError:
    code: str
    message: str
    stage: str
    retryable: bool


@dataclass
class RunRecord:
    run_id: str
    run_type: RunType
    status: RunStatus
    started_at: str
    completed_at: str
    dataset_ref: DatasetRef
    input_ref: InputRef
    runtime_ref: RuntimeRef
    output_ref: OutputRef
    writeback_ref: WritebackRef
    error: RunError | None = None
    scaffold: str | None = None
    schema_version: str = RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RCAReport:
    trace_id: str
    primary_label: Literal[
        "retrieval_failure",
        "tool_failure",
        "instruction_failure",
        "upstream_dependency_failure",
        "data_schema_mismatch",
    ]
    summary: str
    confidence: float
    evidence_refs: list[EvidenceRef]
    remediation: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    schema_version: str = RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComplianceFinding:
    controls_version: str
    control_id: str
    pass_fail: Literal["pass", "fail", "not_applicable", "insufficient_evidence"]
    severity: Literal["critical", "high", "medium", "low"]
    confidence: float
    evidence_refs: list[EvidenceRef]
    missing_evidence: list[str]
    remediation: str


@dataclass
class ComplianceReport:
    trace_id: str
    controls_version: str
    controls_evaluated: list[ComplianceFinding]
    overall_verdict: Literal["compliant", "non_compliant", "needs_review"]
    overall_confidence: float
    gaps: list[str] = field(default_factory=list)
    schema_version: str = RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IncidentTimelineEvent:
    timestamp: str
    event: str
    evidence_refs: list[EvidenceRef]


@dataclass
class RepresentativeTrace:
    trace_id: str
    why_selected: str
    evidence_refs: list[EvidenceRef]


@dataclass
class SuspectedChange:
    change_type: Literal["deploy", "config", "prompt", "dependency", "unknown"]
    change_ref: str
    diff_ref: str | None
    summary: str
    evidence_refs: list[EvidenceRef]


@dataclass
class IncidentHypothesis:
    rank: int
    statement: str
    evidence_refs: list[EvidenceRef]
    confidence: float


@dataclass
class RecommendedAction:
    priority: Literal["P0", "P1", "P2"]
    action: str
    type: Literal["mitigation", "follow_up_fix"]


@dataclass
class IncidentDossier:
    incident_summary: str
    impacted_components: list[str]
    timeline: list[IncidentTimelineEvent]
    representative_traces: list[RepresentativeTrace]
    suspected_change: SuspectedChange
    hypotheses: list[IncidentHypothesis]
    recommended_actions: list[RecommendedAction]
    confidence: float
    gaps: list[str] = field(default_factory=list)
    schema_version: str = RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
