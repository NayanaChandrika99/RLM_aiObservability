# ABOUTME: Validates Phase 6C replay-equivalence acceptance across RCA, compliance, and incident workflows.
# ABOUTME: Ensures deterministic outputs and successful write-back metadata on repeated runs.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.compliance.engine import PolicyComplianceEngine, PolicyComplianceRequest
from investigator.compliance.workflow import run_policy_compliance_workflow
from investigator.incident.engine import IncidentDossierEngine, IncidentDossierRequest
from investigator.incident.workflow import run_incident_dossier_workflow
from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.rca.workflow import run_trace_rca_workflow


class _FakeEvaluationClient:
    def __init__(self) -> None:
        self.logged_evaluations = []

    def log_evaluations(self, *evaluations, **kwargs) -> None:  # noqa: ANN003
        self.logged_evaluations.extend(evaluations)


class _FakeSpansResource:
    def __init__(self) -> None:
        self.logged_annotations = []

    def log_span_annotations(self, *, span_annotations, sync: bool = False) -> None:  # noqa: ANN001
        self.logged_annotations.extend(span_annotations)


class _FakeAnnotationClient:
    def __init__(self) -> None:
        self.spans = _FakeSpansResource()


class _RCAInspectionAPI:
    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [
            {
                "trace_id": trace_id,
                "span_id": "rca-root",
                "parent_id": None,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "ERROR",
                "status_message": "tool timeout",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 100.0,
            },
            {
                "trace_id": trace_id,
                "span_id": "rca-tool",
                "parent_id": "rca-root",
                "name": "lookup",
                "span_kind": "TOOL",
                "status_code": "ERROR",
                "status_message": "tool timeout",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 80.0,
            },
        ]

    def get_span(self, span_id: str) -> dict[str, Any]:
        spans = {span["span_id"]: span for span in self.list_spans("trace-rca")}
        return {"summary": spans[span_id], "attributes": {}, "events": []}

    def get_children(self, span_id: str) -> list[dict[str, Any]]:
        return [span for span in self.list_spans("trace-rca") if span.get("parent_id") == span_id]

    def get_tool_io(self, span_id: str) -> dict[str, Any] | None:
        if span_id != "rca-tool":
            return None
        return {
            "trace_id": "trace-rca",
            "span_id": "rca-tool",
            "artifact_id": "tool:rca-tool",
            "tool_name": "lookup",
            "input": {"q": "x"},
            "output": "timeout",
            "status_code": "ERROR",
        }

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        del span_id
        return []


class _ComplianceInspectionAPI:
    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [
            {
                "trace_id": trace_id,
                "span_id": "cmp-root",
                "parent_id": None,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "OK",
                "status_message": "",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 90.0,
            },
            {
                "trace_id": trace_id,
                "span_id": "cmp-tool",
                "parent_id": "cmp-root",
                "name": "safe_tool",
                "span_kind": "TOOL",
                "status_code": "OK",
                "status_message": "",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 50.0,
            },
        ]

    def list_controls(
        self,
        controls_version: str,
        app_type: str | None = None,
        tools_used: list[str] | None = None,
        data_domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        del app_type, tools_used, data_domains
        return [
            {
                "control_id": "control.safe.tool",
                "controls_version": controls_version,
                "severity": "high",
                "required_evidence": ["required_tool_io", "required_messages"],
            }
        ]

    def get_control(self, control_id: str, controls_version: str) -> dict[str, Any]:
        return {
            "control_id": control_id,
            "controls_version": controls_version,
            "severity": "high",
            "required_evidence": ["required_tool_io", "required_messages"],
        }

    def required_evidence(self, control_id: str, controls_version: str) -> list[str]:  # noqa: ARG002
        return ["required_tool_io", "required_messages"]

    def get_tool_io(self, span_id: str) -> dict[str, Any] | None:
        if span_id != "cmp-tool":
            return None
        return {
            "trace_id": "trace-cmp",
            "span_id": "cmp-tool",
            "artifact_id": "tool:cmp-tool",
            "tool_name": "safe_tool",
            "input": {"q": "policy"},
            "output": {"ok": True},
            "status_code": "OK",
        }

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        del span_id
        return []

    def get_messages(self, span_id: str) -> list[dict[str, Any]]:
        if span_id != "cmp-root":
            return []
        return [
            {
                "trace_id": "trace-cmp",
                "span_id": "cmp-root",
                "role": "user",
                "content": "hello",
            }
        ]


class _IncidentInspectionAPI:
    def list_traces(
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        del project_name, start_time, end_time, filter_expr
        return [
            {
                "project_name": "phase6",
                "trace_id": "trace-inc-1",
                "span_count": 1,
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 300.0,
            },
            {
                "project_name": "phase6",
                "trace_id": "trace-inc-2",
                "span_count": 1,
                "start_time": "2026-02-10T00:01:00Z",
                "end_time": "2026-02-10T00:01:01Z",
                "latency_ms": 110.0,
            },
        ]

    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        if trace_id == "trace-inc-1":
            return [
                {
                    "trace_id": trace_id,
                    "span_id": "inc-root-1",
                    "parent_id": None,
                    "name": "svc.order.agent",
                    "span_kind": "AGENT",
                    "status_code": "ERROR",
                    "status_message": "timeout",
                    "start_time": "2026-02-10T00:00:00Z",
                    "end_time": "2026-02-10T00:00:01Z",
                    "latency_ms": 300.0,
                }
            ]
        return [
            {
                "trace_id": trace_id,
                "span_id": "inc-root-2",
                "parent_id": None,
                "name": "svc.search.agent",
                "span_kind": "AGENT",
                "status_code": "OK",
                "status_message": "",
                "start_time": "2026-02-10T00:01:00Z",
                "end_time": "2026-02-10T00:01:01Z",
                "latency_ms": 110.0,
            }
        ]

    def list_config_snapshots(
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        del project_name, start_time, end_time, tag
        return [
            {
                "snapshot_id": "snap-001",
                "project_name": "phase6",
                "tag": "baseline",
                "created_at": "2026-02-10T00:00:00Z",
                "git_commit": "abc111",
                "paths": ["agent.yaml"],
                "metadata": {},
            },
            {
                "snapshot_id": "snap-002",
                "project_name": "phase6",
                "tag": "candidate",
                "created_at": "2026-02-10T00:30:00Z",
                "git_commit": "abc222",
                "paths": ["prompts/investigator.txt"],
                "metadata": {},
            },
        ]

    def get_config_diff(self, base_snapshot_id: str, target_snapshot_id: str) -> dict[str, Any]:
        return {
            "project_name": "phase6",
            "base_snapshot_id": base_snapshot_id,
            "target_snapshot_id": target_snapshot_id,
            "artifact_id": "configdiff:phase6-replay",
            "diff_ref": "configdiff:phase6-replay",
            "git_commit_base": "abc111",
            "git_commit_target": "abc222",
            "paths": ["prompts/investigator.txt"],
            "summary": "1 changed file(s)",
        }


def _read_output_payload(artifacts_root: Path, run_id: str) -> dict[str, Any]:
    path = artifacts_root / run_id / "output.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase6c_trace_rca_replay_equivalence(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    request = TraceRCARequest(trace_id="trace-rca", project_name="phase6")

    report_a, run_a = run_trace_rca_workflow(
        request=request,
        engine=TraceRCAEngine(inspection_api=_RCAInspectionAPI(), max_hot_spans=2),
        run_id="run-phase6c-rca-a",
        artifacts_root=artifacts_root,
        writeback_client=_FakeEvaluationClient(),
    )
    report_b, run_b = run_trace_rca_workflow(
        request=request,
        engine=TraceRCAEngine(inspection_api=_RCAInspectionAPI(), max_hot_spans=2),
        run_id="run-phase6c-rca-b",
        artifacts_root=artifacts_root,
        writeback_client=_FakeEvaluationClient(),
    )

    assert report_a.to_dict() == report_b.to_dict()
    assert _read_output_payload(artifacts_root, "run-phase6c-rca-a") == _read_output_payload(
        artifacts_root, "run-phase6c-rca-b"
    )
    assert run_a.status == "succeeded"
    assert run_b.status == "succeeded"
    assert run_a.writeback_ref.annotation_names == ["rca.primary", "rca.evidence"]
    assert run_b.writeback_ref.annotation_names == ["rca.primary", "rca.evidence"]


def test_phase6c_compliance_replay_equivalence(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    request = PolicyComplianceRequest(
        trace_id="trace-cmp",
        project_name="phase6",
        controls_version="controls-v1",
    )

    report_a, run_a = run_policy_compliance_workflow(
        request=request,
        engine=PolicyComplianceEngine(inspection_api=_ComplianceInspectionAPI(), max_controls=5),
        run_id="run-phase6c-cmp-a",
        artifacts_root=artifacts_root,
        writeback_client=_FakeAnnotationClient(),
    )
    report_b, run_b = run_policy_compliance_workflow(
        request=request,
        engine=PolicyComplianceEngine(inspection_api=_ComplianceInspectionAPI(), max_controls=5),
        run_id="run-phase6c-cmp-b",
        artifacts_root=artifacts_root,
        writeback_client=_FakeAnnotationClient(),
    )

    assert report_a.to_dict() == report_b.to_dict()
    assert _read_output_payload(artifacts_root, "run-phase6c-cmp-a") == _read_output_payload(
        artifacts_root, "run-phase6c-cmp-b"
    )
    assert run_a.status == "succeeded"
    assert run_b.status == "succeeded"
    assert run_a.writeback_ref.annotation_names == [
        "compliance.overall",
        "compliance.control.control.safe.tool.evidence",
    ]
    assert run_b.writeback_ref.annotation_names == [
        "compliance.overall",
        "compliance.control.control.safe.tool.evidence",
    ]


def test_phase6c_incident_replay_equivalence(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    request = IncidentDossierRequest(
        project_name="phase6",
        time_window_start="2026-02-10T00:00:00Z",
        time_window_end="2026-02-10T01:00:00Z",
    )

    report_a, run_a = run_incident_dossier_workflow(
        request=request,
        engine=IncidentDossierEngine(inspection_api=_IncidentInspectionAPI(), max_representatives=2),
        run_id="run-phase6c-inc-a",
        artifacts_root=artifacts_root,
        writeback_client=_FakeAnnotationClient(),
    )
    report_b, run_b = run_incident_dossier_workflow(
        request=request,
        engine=IncidentDossierEngine(inspection_api=_IncidentInspectionAPI(), max_representatives=2),
        run_id="run-phase6c-inc-b",
        artifacts_root=artifacts_root,
        writeback_client=_FakeAnnotationClient(),
    )

    assert report_a.to_dict() == report_b.to_dict()
    assert _read_output_payload(artifacts_root, "run-phase6c-inc-a") == _read_output_payload(
        artifacts_root, "run-phase6c-inc-b"
    )
    assert run_a.status == "succeeded"
    assert run_b.status == "succeeded"
    assert run_a.writeback_ref.annotation_names == ["incident.dossier", "incident.timeline.evidence"]
    assert run_b.writeback_ref.annotation_names == ["incident.dossier", "incident.timeline.evidence"]
