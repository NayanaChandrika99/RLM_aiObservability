# ABOUTME: Validates Phase 3A Trace RCA deterministic narrowing and label selection behavior.
# ABOUTME: Ensures RCA output uses evidence refs derived from inspection API signals.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from investigator.rca.engine import TraceRCAEngine, TraceRCARequest


@dataclass
class _Span:
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    span_kind: str
    status_code: str
    status_message: str
    latency_ms: float
    events: list[dict[str, Any]]


class _FakeInspectionAPI:
    def __init__(
        self,
        *,
        spans: list[_Span],
        tool_io_by_span: dict[str, dict[str, Any] | None] | None = None,
        retrieval_chunks_by_span: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._spans = spans
        self._tool_io_by_span = tool_io_by_span or {}
        self._retrieval_chunks_by_span = retrieval_chunks_by_span or {}
        self.get_children_calls: list[str] = []

    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [
            {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "name": span.name,
                "span_kind": span.span_kind,
                "status_code": span.status_code,
                "status_message": span.status_message,
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": span.latency_ms,
            }
            for span in self._spans
            if span.trace_id == trace_id
        ]

    def get_children(self, span_id: str) -> list[dict[str, Any]]:
        self.get_children_calls.append(span_id)
        children = [span for span in self._spans if span.parent_id == span_id]
        return [
            {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "name": span.name,
                "span_kind": span.span_kind,
                "status_code": span.status_code,
                "status_message": span.status_message,
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": span.latency_ms,
            }
            for span in children
        ]

    def get_span(self, span_id: str) -> dict[str, Any]:
        span = next(item for item in self._spans if item.span_id == span_id)
        return {
            "summary": {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "name": span.name,
                "span_kind": span.span_kind,
                "status_code": span.status_code,
                "status_message": span.status_message,
                "latency_ms": span.latency_ms,
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
            },
            "attributes": {},
            "events": span.events,
        }

    def get_tool_io(self, span_id: str) -> dict[str, Any] | None:
        return self._tool_io_by_span.get(span_id)

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        return self._retrieval_chunks_by_span.get(span_id, [])


def test_trace_rca_prefers_error_then_exception_then_latency() -> None:
    spans = [
        _Span(
            span_id="s-latency",
            trace_id="trace-1",
            parent_id=None,
            name="slow.ok",
            span_kind="CHAIN",
            status_code="OK",
            status_message="",
            latency_ms=900.0,
            events=[],
        ),
        _Span(
            span_id="s-error-exception",
            trace_id="trace-1",
            parent_id=None,
            name="tool.call",
            span_kind="TOOL",
            status_code="ERROR",
            status_message="tool crashed",
            latency_ms=20.0,
            events=[{"name": "exception", "attributes": {"message": "boom"}}],
        ),
        _Span(
            span_id="s-error-no-exception",
            trace_id="trace-1",
            parent_id=None,
            name="tool.parse",
            span_kind="TOOL",
            status_code="ERROR",
            status_message="schema parse error",
            latency_ms=300.0,
            events=[],
        ),
    ]
    tool_io = {
        "s-error-exception": {
            "trace_id": "trace-1",
            "span_id": "s-error-exception",
            "artifact_id": "tool:s-error-exception",
            "tool_name": "lookup",
            "input": {"q": "x"},
            "output": "failed",
            "status_code": "ERROR",
        }
    }
    engine = TraceRCAEngine(inspection_api=_FakeInspectionAPI(spans=spans, tool_io_by_span=tool_io))
    report = engine.run(TraceRCARequest(trace_id="trace-1", project_name="phase3"))
    payload = report.to_dict()

    assert payload["primary_label"] == "tool_failure"
    assert payload["evidence_refs"][0]["span_id"] == "s-error-exception"
    assert any(evidence["ref"] == "tool:s-error-exception" for evidence in payload["evidence_refs"])


def test_trace_rca_detects_upstream_dependency_failure() -> None:
    spans = [
        _Span(
            span_id="s-upstream",
            trace_id="trace-2",
            parent_id=None,
            name="tool.http",
            span_kind="TOOL",
            status_code="ERROR",
            status_message="upstream timeout 503",
            latency_ms=1500.0,
            events=[],
        )
    ]
    engine = TraceRCAEngine(inspection_api=_FakeInspectionAPI(spans=spans))
    report = engine.run(TraceRCARequest(trace_id="trace-2", project_name="phase3"))
    assert report.primary_label == "upstream_dependency_failure"


def test_trace_rca_detects_retrieval_failure_from_empty_chunks() -> None:
    spans = [
        _Span(
            span_id="s-retriever",
            trace_id="trace-3",
            parent_id=None,
            name="retriever.fetch",
            span_kind="RETRIEVER",
            status_code="OK",
            status_message="",
            latency_ms=90.0,
            events=[],
        )
    ]
    engine = TraceRCAEngine(
        inspection_api=_FakeInspectionAPI(
            spans=spans,
            retrieval_chunks_by_span={"s-retriever": []},
        )
    )
    report = engine.run(TraceRCARequest(trace_id="trace-3", project_name="phase3"))
    assert report.primary_label == "retrieval_failure"


def test_trace_rca_inferrs_retriever_span_from_name_when_kind_unknown() -> None:
    spans = [
        _Span(
            span_id="s-retriever-unknown",
            trace_id="trace-3b",
            parent_id=None,
            name="retriever.fetch",
            span_kind="UNKNOWN",
            status_code="OK",
            status_message="",
            latency_ms=95.0,
            events=[],
        )
    ]
    engine = TraceRCAEngine(
        inspection_api=_FakeInspectionAPI(
            spans=spans,
            retrieval_chunks_by_span={"s-retriever-unknown": []},
        )
    )
    report = engine.run(TraceRCARequest(trace_id="trace-3b", project_name="phase3"))
    assert report.primary_label == "retrieval_failure"


def test_trace_rca_fallback_when_inspection_api_fails() -> None:
    class _BrokenAPI:
        def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
            raise RuntimeError("not available")

    engine = TraceRCAEngine(inspection_api=_BrokenAPI())
    report = engine.run(TraceRCARequest(trace_id="trace-4", project_name="phase3"))
    assert report.primary_label == "instruction_failure"
    assert report.gaps


def test_trace_rca_recursively_inspects_branch_children() -> None:
    spans = [
        _Span(
            span_id="root-error",
            trace_id="trace-5",
            parent_id=None,
            name="agent.run",
            span_kind="AGENT",
            status_code="ERROR",
            status_message="root failure",
            latency_ms=100.0,
            events=[],
        ),
        _Span(
            span_id="child-tool",
            trace_id="trace-5",
            parent_id="root-error",
            name="tool.lookup",
            span_kind="TOOL",
            status_code="ERROR",
            status_message="tool crashed",
            latency_ms=40.0,
            events=[{"name": "exception", "attributes": {"message": "boom"}}],
        ),
    ]
    tool_io = {
        "child-tool": {
            "trace_id": "trace-5",
            "span_id": "child-tool",
            "artifact_id": "tool:child-tool",
            "tool_name": "lookup",
            "input": {"q": "x"},
            "output": "error",
            "status_code": "ERROR",
        }
    }
    api = _FakeInspectionAPI(spans=spans, tool_io_by_span=tool_io)
    engine = TraceRCAEngine(inspection_api=api, max_hot_spans=1)
    report = engine.run(TraceRCARequest(trace_id="trace-5", project_name="phase3"))
    payload = report.to_dict()

    assert "root-error" in api.get_children_calls
    assert any(evidence["span_id"] == "child-tool" for evidence in payload["evidence_refs"])
    assert any(evidence["ref"] == "tool:child-tool" for evidence in payload["evidence_refs"])
