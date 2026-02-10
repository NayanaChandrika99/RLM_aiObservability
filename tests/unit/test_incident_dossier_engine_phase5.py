# ABOUTME: Validates Phase 5A incident representative trace selection determinism and ordering.
# ABOUTME: Ensures error/latency bucket selection and signature dedupe rules are applied consistently.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from investigator.incident.engine import IncidentDossierEngine, IncidentDossierRequest


@dataclass
class _TraceRow:
    trace_id: str
    latency_ms: float
    start_time: str
    end_time: str


@dataclass
class _Span:
    trace_id: str
    span_id: str
    name: str
    span_kind: str
    status_code: str
    status_message: str
    parent_id: str | None = None
    start_time: str = "2026-02-10T00:00:00Z"


@dataclass
class _SnapshotRow:
    snapshot_id: str
    created_at: str
    tag: str
    git_commit: str | None
    paths: list[str]


class _FakeIncidentInspectionAPI:
    def __init__(
        self,
        *,
        traces: list[_TraceRow],
        spans: list[_Span],
        snapshots: list[_SnapshotRow] | None = None,
        diff_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self._traces = traces
        self._spans = spans
        self._snapshots = snapshots or []
        self._diff_lookup = diff_lookup or {}
        self.list_traces_calls = 0
        self.last_diff_args: tuple[str, str] | None = None

    def list_traces(  # noqa: ANN201
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ):
        self.list_traces_calls += 1
        return [
            {
                "project_name": project_name,
                "trace_id": trace.trace_id,
                "span_count": sum(1 for span in self._spans if span.trace_id == trace.trace_id),
                "start_time": trace.start_time,
                "end_time": trace.end_time,
                "latency_ms": trace.latency_ms,
            }
            for trace in self._traces
        ]

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
                "start_time": span.start_time,
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 10.0,
            }
            for span in self._spans
            if span.trace_id == trace_id
        ]

    def list_config_snapshots(
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        del project_name, start_time, end_time
        rows = list(self._snapshots)
        if tag is not None:
            rows = [row for row in rows if row.tag == tag]
        return [
            {
                "snapshot_id": row.snapshot_id,
                "project_name": "phase5",
                "tag": row.tag,
                "created_at": row.created_at,
                "git_commit": row.git_commit,
                "paths": row.paths,
                "metadata": {},
            }
            for row in rows
        ]

    def get_config_diff(self, base_snapshot_id: str, target_snapshot_id: str) -> dict[str, Any]:
        self.last_diff_args = (base_snapshot_id, target_snapshot_id)
        key = (base_snapshot_id, target_snapshot_id)
        if key not in self._diff_lookup:
            raise KeyError(key)
        return self._diff_lookup[key]


def test_incident_selector_prioritizes_error_then_latency_with_signature_dedupe() -> None:
    traces = [
        _TraceRow("trace-a", 220.0, "2026-02-10T00:00:00Z", "2026-02-10T00:00:01Z"),
        _TraceRow("trace-b", 850.0, "2026-02-10T00:02:00Z", "2026-02-10T00:02:01Z"),
        _TraceRow("trace-c", 900.0, "2026-02-10T00:04:00Z", "2026-02-10T00:04:01Z"),
        _TraceRow("trace-d", 180.0, "2026-02-10T00:06:00Z", "2026-02-10T00:06:01Z"),
        _TraceRow("trace-e", 700.0, "2026-02-10T00:08:00Z", "2026-02-10T00:08:01Z"),
    ]
    spans = [
        _Span("trace-a", "a-root", "svc.order.agent", "AGENT", "ERROR", "timeout"),
        _Span("trace-a", "a-tool", "db_query", "TOOL", "ERROR", "timeout"),
        _Span("trace-b", "b-root", "svc.order.agent", "AGENT", "ERROR", "timeout"),
        _Span("trace-b", "b-tool", "db_query", "TOOL", "ERROR", "timeout"),
        _Span("trace-c", "c-root", "svc.search.agent", "AGENT", "OK", ""),
        _Span("trace-c", "c-tool", "retriever", "TOOL", "OK", ""),
        _Span("trace-d", "d-root", "svc.billing.agent", "AGENT", "ERROR", "schema"),
        _Span("trace-d", "d-tool", "ledger_lookup", "TOOL", "ERROR", "schema mismatch"),
        _Span("trace-e", "e-root", "svc.reco.agent", "AGENT", "OK", ""),
    ]
    api = _FakeIncidentInspectionAPI(traces=traces, spans=spans)
    engine = IncidentDossierEngine(
        inspection_api=api,
        max_representatives=4,
        error_quota=2,
        latency_quota=2,
        cluster_quota=0,
    )
    request = IncidentDossierRequest(
        project_name="phase5",
        time_window_start="2026-02-10T00:00:00Z",
        time_window_end="2026-02-10T01:00:00Z",
    )
    dossier = engine.run(request)
    selected_ids = [item.trace_id for item in dossier.representative_traces]
    reasons = [item.why_selected for item in dossier.representative_traces]

    assert selected_ids == ["trace-a", "trace-d", "trace-c", "trace-e"]
    assert reasons[0].startswith("error_bucket")
    assert reasons[1].startswith("error_bucket")
    assert reasons[2].startswith("latency_bucket")
    assert reasons[3].startswith("latency_bucket")


def test_incident_selector_uses_override_trace_ids_without_list_traces_query() -> None:
    traces = [_TraceRow("trace-x", 200.0, "2026-02-10T00:00:00Z", "2026-02-10T00:00:01Z")]
    spans = [
        _Span("trace-2", "t2-root", "svc.a.agent", "AGENT", "OK", ""),
        _Span("trace-1", "t1-root", "svc.b.agent", "AGENT", "ERROR", "timeout"),
    ]
    api = _FakeIncidentInspectionAPI(traces=traces, spans=spans)
    engine = IncidentDossierEngine(
        inspection_api=api,
        max_representatives=2,
        error_quota=2,
        latency_quota=0,
        cluster_quota=0,
    )
    request = IncidentDossierRequest(
        project_name="phase5",
        time_window_start="2026-02-10T00:00:00Z",
        time_window_end="2026-02-10T01:00:00Z",
        trace_ids_override=["trace-2", "trace-1"],
    )
    dossier = engine.run(request)

    assert api.list_traces_calls == 0
    assert [item.trace_id for item in dossier.representative_traces] == ["trace-1", "trace-2"]


def test_incident_selector_attaches_config_diff_evidence_when_snapshots_exist() -> None:
    traces = [
        _TraceRow("trace-a", 440.0, "2026-02-10T00:00:00Z", "2026-02-10T00:00:02Z"),
        _TraceRow("trace-b", 300.0, "2026-02-10T00:10:00Z", "2026-02-10T00:10:01Z"),
    ]
    spans = [
        _Span("trace-a", "a-root", "svc.order.agent", "AGENT", "ERROR", "timeout"),
        _Span("trace-a", "a-tool", "db_query", "TOOL", "ERROR", "timeout"),
        _Span("trace-b", "b-root", "svc.search.agent", "AGENT", "OK", ""),
    ]
    snapshots = [
        _SnapshotRow("s-001", "2026-02-10T00:00:00Z", "baseline", "abc111", ["agent.yaml"]),
        _SnapshotRow("s-002", "2026-02-10T00:20:00Z", "candidate", "abc222", ["agent.yaml"]),
        _SnapshotRow("s-003", "2026-02-10T00:40:00Z", "candidate", "abc333", ["prompts/investigator.txt"]),
    ]
    diff_lookup = {
        ("s-002", "s-003"): {
            "project_name": "phase5",
            "base_snapshot_id": "s-002",
            "target_snapshot_id": "s-003",
            "artifact_id": "configdiff:diff-123",
            "diff_ref": "configdiff:diff-123",
            "git_commit_base": "abc222",
            "git_commit_target": "abc333",
            "paths": ["prompts/investigator.txt"],
            "summary": "1 changed file(s)",
        }
    }
    api = _FakeIncidentInspectionAPI(
        traces=traces,
        spans=spans,
        snapshots=snapshots,
        diff_lookup=diff_lookup,
    )
    engine = IncidentDossierEngine(
        inspection_api=api,
        max_representatives=2,
        error_quota=1,
        latency_quota=1,
        cluster_quota=0,
    )
    request = IncidentDossierRequest(
        project_name="phase5",
        time_window_start="2026-02-10T00:00:00Z",
        time_window_end="2026-02-10T01:00:00Z",
    )

    dossier = engine.run(request)

    assert api.last_diff_args == ("s-002", "s-003")
    assert dossier.suspected_change.change_type == "prompt"
    assert dossier.suspected_change.diff_ref == "configdiff:diff-123"
    assert any(
        evidence.kind == "CONFIG_DIFF" and evidence.ref == "configdiff:diff-123"
        for evidence in dossier.suspected_change.evidence_refs
    )
    assert any("config diff" in event.event.lower() for event in dossier.timeline)


def test_incident_selector_records_gap_when_config_snapshots_missing() -> None:
    traces = [_TraceRow("trace-a", 200.0, "2026-02-10T00:00:00Z", "2026-02-10T00:00:01Z")]
    spans = [_Span("trace-a", "a-root", "svc.order.agent", "AGENT", "OK", "")]
    api = _FakeIncidentInspectionAPI(traces=traces, spans=spans, snapshots=[])
    engine = IncidentDossierEngine(
        inspection_api=api,
        max_representatives=1,
        error_quota=0,
        latency_quota=1,
        cluster_quota=0,
    )
    request = IncidentDossierRequest(
        project_name="phase5",
        time_window_start="2026-02-10T00:00:00Z",
        time_window_end="2026-02-10T01:00:00Z",
    )

    dossier = engine.run(request)

    assert dossier.suspected_change.diff_ref is None
    assert any("snapshot" in gap.lower() for gap in dossier.gaps)
