# ABOUTME: Validates Phase 4A compliance engine control scoping and deterministic ordering behavior.
# ABOUTME: Ensures controls_version remains traceable across report findings and runtime input refs.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from investigator.compliance.engine import PolicyComplianceEngine, PolicyComplianceRequest
from investigator.runtime.runner import run_engine


@dataclass
class _Span:
    trace_id: str
    span_id: str
    name: str
    span_kind: str
    status_code: str = "OK"
    status_message: str = ""


class _FakeComplianceInspectionAPI:
    def __init__(
        self,
        *,
        spans: list[_Span],
        controls: list[dict[str, Any]],
        override_controls: dict[str, dict[str, Any]] | None = None,
        required_by_control: dict[str, list[str]] | None = None,
        tool_io_by_span: dict[str, dict[str, Any] | None] | None = None,
        retrieval_by_span: dict[str, list[dict[str, Any]]] | None = None,
        messages_by_span: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._spans = spans
        self._controls = controls
        self._override_controls = override_controls or {}
        self._required_by_control = required_by_control or {}
        self._tool_io_by_span = tool_io_by_span or {}
        self._retrieval_by_span = retrieval_by_span or {}
        self._messages_by_span = messages_by_span or {}
        self.list_controls_calls: list[dict[str, Any]] = []

    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [
            {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_id": None,
                "name": span.name,
                "span_kind": span.span_kind,
                "status_code": span.status_code,
                "status_message": span.status_message,
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 10.0,
            }
            for span in self._spans
            if span.trace_id == trace_id
        ]

    def list_controls(
        self,
        controls_version: str,
        app_type: str | None = None,
        tools_used: list[str] | None = None,
        data_domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.list_controls_calls.append(
            {
                "controls_version": controls_version,
                "app_type": app_type,
                "tools_used": tools_used,
                "data_domains": data_domains,
            }
        )
        return [control for control in self._controls if control["controls_version"] == controls_version]

    def get_control(self, control_id: str, controls_version: str) -> dict[str, Any]:
        control = self._override_controls.get(control_id)
        if control is None:
            raise KeyError(control_id)
        if control["controls_version"] != controls_version:
            raise KeyError(control_id)
        return control

    def required_evidence(self, control_id: str, controls_version: str) -> list[str]:
        return self._required_by_control.get(control_id, [])

    def get_tool_io(self, span_id: str) -> dict[str, Any] | None:
        return self._tool_io_by_span.get(span_id)

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        return self._retrieval_by_span.get(span_id, [])

    def get_messages(self, span_id: str) -> list[dict[str, Any]]:
        return self._messages_by_span.get(span_id, [])


def test_policy_compliance_scopes_controls_with_inferred_tools() -> None:
    spans = [
        _Span(trace_id="trace-1", span_id="root", name="agent.run", span_kind="AGENT"),
        _Span(trace_id="trace-1", span_id="tool-a", name="sql_db_query", span_kind="TOOL"),
    ]
    controls = [
        {
            "control_id": "control.tool.safety",
            "controls_version": "controls-v1",
            "severity": "high",
            "required_evidence": ["required_tool_io"],
        }
    ]
    api = _FakeComplianceInspectionAPI(
        spans=spans,
        controls=controls,
        required_by_control={"control.tool.safety": ["required_tool_io"]},
        tool_io_by_span={
            "tool-a": {
                "trace_id": "trace-1",
                "span_id": "tool-a",
                "artifact_id": "tool:tool-a",
                "status_code": "OK",
            }
        },
    )
    engine = PolicyComplianceEngine(inspection_api=api)
    request = PolicyComplianceRequest(
        trace_id="trace-1",
        project_name="phase4",
        controls_version="controls-v1",
    )
    report = engine.run(request)

    assert api.list_controls_calls
    first_call = api.list_controls_calls[0]
    assert first_call["controls_version"] == "controls-v1"
    assert first_call["app_type"] == "agentic"
    assert first_call["tools_used"] == ["sql_db_query"]
    assert report.controls_version == "controls-v1"
    assert all(item.controls_version == "controls-v1" for item in report.controls_evaluated)


def test_policy_compliance_merges_override_and_sorts_by_severity() -> None:
    spans = [_Span(trace_id="trace-2", span_id="root", name="agent.run", span_kind="AGENT")]
    controls = [
        {
            "control_id": "control.low",
            "controls_version": "controls-v1",
            "severity": "low",
            "required_evidence": [],
        }
    ]
    override = {
        "control.critical": {
            "control_id": "control.critical",
            "controls_version": "controls-v1",
            "severity": "critical",
            "required_evidence": [],
        }
    }
    api = _FakeComplianceInspectionAPI(spans=spans, controls=controls, override_controls=override)
    engine = PolicyComplianceEngine(inspection_api=api)
    request = PolicyComplianceRequest(
        trace_id="trace-2",
        project_name="phase4",
        controls_version="controls-v1",
        control_scope_override=["control.critical"],
    )
    report = engine.run(request)
    control_ids = [finding.control_id for finding in report.controls_evaluated]

    assert control_ids == ["control.critical", "control.low"]


def test_policy_compliance_input_ref_preserves_controls_version() -> None:
    spans = [_Span(trace_id="trace-3", span_id="root", name="agent.run", span_kind="AGENT")]
    controls = [
        {
            "control_id": "control.empty",
            "controls_version": "controls-v1",
            "severity": "medium",
            "required_evidence": [],
        }
    ]
    api = _FakeComplianceInspectionAPI(spans=spans, controls=controls)
    engine = PolicyComplianceEngine(inspection_api=api)
    request = PolicyComplianceRequest(
        trace_id="trace-3",
        project_name="phase4",
        controls_version="controls-v1",
    )
    _, run_record = run_engine(engine=engine, request=request)

    assert run_record.input_ref.controls_version == "controls-v1"


def test_policy_compliance_marks_missing_evidence_per_control() -> None:
    spans = [
        _Span(trace_id="trace-4", span_id="root", name="agent.run", span_kind="AGENT"),
        _Span(trace_id="trace-4", span_id="tool-a", name="db_query", span_kind="TOOL"),
    ]
    controls = [
        {
            "control_id": "control.retrieval.required",
            "controls_version": "controls-v1",
            "severity": "high",
            "required_evidence": ["required_tool_io", "required_retrieval_chunks"],
        }
    ]
    api = _FakeComplianceInspectionAPI(
        spans=spans,
        controls=controls,
        tool_io_by_span={
            "tool-a": {
                "trace_id": "trace-4",
                "span_id": "tool-a",
                "artifact_id": "tool:tool-a",
                "status_code": "OK",
            }
        },
        retrieval_by_span={},
    )
    engine = PolicyComplianceEngine(inspection_api=api)
    report = engine.run(
        PolicyComplianceRequest(
            trace_id="trace-4",
            project_name="phase4",
            controls_version="controls-v1",
        )
    )
    finding = report.controls_evaluated[0]

    assert finding.pass_fail == "insufficient_evidence"
    assert finding.missing_evidence == ["required_retrieval_chunks"]
    assert finding.evidence_refs


def test_policy_compliance_passes_when_required_evidence_present() -> None:
    spans = [
        _Span(trace_id="trace-5", span_id="root", name="agent.run", span_kind="AGENT"),
        _Span(trace_id="trace-5", span_id="tool-a", name="db_query", span_kind="TOOL"),
        _Span(trace_id="trace-5", span_id="retr-a", name="retrieve", span_kind="RETRIEVER"),
    ]
    controls = [
        {
            "control_id": "control.full.required",
            "controls_version": "controls-v1",
            "severity": "medium",
            "required_evidence": ["required_tool_io", "required_retrieval_chunks"],
        }
    ]
    api = _FakeComplianceInspectionAPI(
        spans=spans,
        controls=controls,
        tool_io_by_span={
            "tool-a": {
                "trace_id": "trace-5",
                "span_id": "tool-a",
                "artifact_id": "tool:tool-a",
                "status_code": "OK",
            }
        },
        retrieval_by_span={
            "retr-a": [
                {
                    "trace_id": "trace-5",
                    "span_id": "retr-a",
                    "artifact_id": "retrieval:retr-a:0:doc-1",
                    "document_id": "doc-1",
                    "content": "alpha",
                }
            ]
        },
    )
    engine = PolicyComplianceEngine(inspection_api=api)
    report = engine.run(
        PolicyComplianceRequest(
            trace_id="trace-5",
            project_name="phase4",
            controls_version="controls-v1",
        )
    )
    finding = report.controls_evaluated[0]

    assert finding.pass_fail == "pass"
    assert finding.missing_evidence == []


def test_policy_compliance_fails_on_violation_patterns() -> None:
    spans = [
        _Span(trace_id="trace-6", span_id="root", name="agent.run", span_kind="AGENT"),
    ]
    controls = [
        {
            "control_id": "control.no_secrets",
            "controls_version": "controls-v1",
            "severity": "critical",
            "required_evidence": ["required_messages"],
            "violation_patterns": ["password\\s*[:=]"],
        }
    ]
    api = _FakeComplianceInspectionAPI(
        spans=spans,
        controls=controls,
        messages_by_span={
            "root": [
                {
                    "trace_id": "trace-6",
                    "span_id": "root",
                    "role": "assistant",
                    "content": "password: hunter2",
                }
            ]
        },
    )
    engine = PolicyComplianceEngine(inspection_api=api)
    report = engine.run(
        PolicyComplianceRequest(
            trace_id="trace-6",
            project_name="phase4",
            controls_version="controls-v1",
        )
    )
    finding = report.controls_evaluated[0]

    assert finding.pass_fail == "fail"
    assert finding.missing_evidence == []
    assert report.overall_verdict == "non_compliant"
