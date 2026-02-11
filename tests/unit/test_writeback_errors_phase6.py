# ABOUTME: Validates Phase 6B workflow behavior when Phoenix write-back raises errors.
# ABOUTME: Ensures workflows persist partial RunRecord status with engine-specific write-back error codes.

from __future__ import annotations

import json
from pathlib import Path

from investigator.compliance.engine import PolicyComplianceEngine, PolicyComplianceRequest
from investigator.compliance.workflow import run_policy_compliance_workflow
from investigator.incident.engine import IncidentDossierEngine, IncidentDossierRequest
from investigator.incident.workflow import run_incident_dossier_workflow
from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.rca.workflow import run_trace_rca_workflow


class _RCAInspectionAPI:
    def list_spans(self, trace_id: str):  # noqa: ANN201
        return [
            {
                "trace_id": trace_id,
                "span_id": "rca-root",
                "parent_id": None,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "ERROR",
                "status_message": "tool failed",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 100.0,
            }
        ]

    def get_span(self, span_id: str):  # noqa: ANN201,ARG002
        return {"summary": self.list_spans("trace-rca")[0], "attributes": {}, "events": []}

    def get_children(self, span_id: str):  # noqa: ANN201,ARG002
        return []

    def get_tool_io(self, span_id: str):  # noqa: ANN201,ARG002
        return None

    def get_retrieval_chunks(self, span_id: str):  # noqa: ANN201,ARG002
        return []


class _ComplianceInspectionAPI:
    def list_spans(self, trace_id: str):  # noqa: ANN201
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
                "latency_ms": 120.0,
            }
        ]

    def list_controls(self, controls_version: str, app_type=None, tools_used=None, data_domains=None):  # noqa: ANN001,ANN201
        del app_type, tools_used, data_domains
        return [
            {
                "control_id": "control.alpha",
                "controls_version": controls_version,
                "severity": "high",
                "required_evidence": ["required_messages"],
            }
        ]

    def get_control(self, control_id: str, controls_version: str):  # noqa: ANN201
        return {
            "control_id": control_id,
            "controls_version": controls_version,
            "severity": "high",
            "required_evidence": ["required_messages"],
        }

    def required_evidence(self, control_id: str, controls_version: str):  # noqa: ANN201,ARG002
        return ["required_messages"]

    def get_tool_io(self, span_id: str):  # noqa: ANN201,ARG002
        return None

    def get_retrieval_chunks(self, span_id: str):  # noqa: ANN201,ARG002
        return []

    def get_messages(self, span_id: str):  # noqa: ANN201,ARG002
        return [{"trace_id": "trace-cmp", "span_id": "cmp-root", "role": "user", "content": "hello"}]


class _IncidentInspectionAPI:
    def list_traces(  # noqa: ANN201
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ):
        del project_name, start_time, end_time, filter_expr
        return [
            {
                "project_name": "phase6",
                "trace_id": "trace-inc",
                "span_count": 1,
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 200.0,
            }
        ]

    def list_spans(self, trace_id: str):  # noqa: ANN201
        return [
            {
                "trace_id": trace_id,
                "span_id": "inc-root",
                "parent_id": None,
                "name": "svc.order.agent",
                "span_kind": "AGENT",
                "status_code": "ERROR",
                "status_message": "timeout",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 200.0,
            }
        ]

    def list_config_snapshots(  # noqa: ANN201
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        tag: str | None = None,
    ):
        del project_name, start_time, end_time, tag
        return [
            {
                "snapshot_id": "snap-a",
                "project_name": "phase6",
                "tag": "baseline",
                "created_at": "2026-02-10T00:00:00Z",
                "git_commit": "abc111",
                "paths": ["agent.yaml"],
                "metadata": {},
            },
            {
                "snapshot_id": "snap-b",
                "project_name": "phase6",
                "tag": "candidate",
                "created_at": "2026-02-10T00:30:00Z",
                "git_commit": "abc222",
                "paths": ["prompts/investigator.txt"],
                "metadata": {},
            },
        ]

    def get_config_diff(self, base_snapshot_id: str, target_snapshot_id: str):  # noqa: ANN201,ARG002
        return {
            "project_name": "phase6",
            "base_snapshot_id": base_snapshot_id,
            "target_snapshot_id": target_snapshot_id,
            "artifact_id": "configdiff:phase6",
            "diff_ref": "configdiff:phase6",
            "git_commit_base": "abc111",
            "git_commit_target": "abc222",
            "paths": ["prompts/investigator.txt"],
            "summary": "1 changed file(s)",
        }


def test_trace_rca_workflow_sets_partial_on_writeback_error(monkeypatch, tmp_path: Path) -> None:
    def _raise_writeback(*args, **kwargs):  # noqa: ANN002,ANN003
        raise RuntimeError("simulated writeback failure")

    monkeypatch.setattr("investigator.rca.workflow.write_rca_to_phoenix", _raise_writeback)
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    engine = TraceRCAEngine(inspection_api=_RCAInspectionAPI(), max_hot_spans=1)
    request = TraceRCARequest(trace_id="trace-rca", project_name="phase6")

    _, run_record = run_trace_rca_workflow(
        request=request,
        engine=engine,
        run_id="run-phase6-rca-writeback",
        artifacts_root=artifacts_root,
    )

    assert run_record.status == "partial"
    assert run_record.error is not None
    assert run_record.error.code == "RCA_WRITEBACK_FAILED"
    persisted = json.loads((artifacts_root / "run-phase6-rca-writeback" / "run_record.json").read_text())
    assert persisted["status"] == "partial"
    assert persisted["error"]["code"] == "RCA_WRITEBACK_FAILED"


def test_compliance_workflow_sets_partial_on_writeback_error(monkeypatch, tmp_path: Path) -> None:
    def _raise_writeback(*args, **kwargs):  # noqa: ANN002,ANN003
        raise RuntimeError("simulated writeback failure")

    monkeypatch.setattr("investigator.compliance.workflow.write_compliance_to_phoenix", _raise_writeback)
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    engine = PolicyComplianceEngine(inspection_api=_ComplianceInspectionAPI(), max_controls=5)
    request = PolicyComplianceRequest(
        trace_id="trace-cmp",
        project_name="phase6",
        controls_version="controls-v1",
    )

    _, run_record = run_policy_compliance_workflow(
        request=request,
        engine=engine,
        run_id="run-phase6-cmp-writeback",
        artifacts_root=artifacts_root,
    )

    assert run_record.status == "partial"
    assert run_record.error is not None
    assert run_record.error.code == "COMPLIANCE_WRITEBACK_FAILED"
    persisted = json.loads((artifacts_root / "run-phase6-cmp-writeback" / "run_record.json").read_text())
    assert persisted["status"] == "partial"
    assert persisted["error"]["code"] == "COMPLIANCE_WRITEBACK_FAILED"


def test_incident_workflow_sets_partial_on_writeback_error(monkeypatch, tmp_path: Path) -> None:
    def _raise_writeback(*args, **kwargs):  # noqa: ANN002,ANN003
        raise RuntimeError("simulated writeback failure")

    monkeypatch.setattr("investigator.incident.workflow.write_incident_to_phoenix", _raise_writeback)
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    engine = IncidentDossierEngine(inspection_api=_IncidentInspectionAPI(), max_representatives=1)
    request = IncidentDossierRequest(
        project_name="phase6",
        time_window_start="2026-02-10T00:00:00Z",
        time_window_end="2026-02-10T01:00:00Z",
    )

    _, run_record = run_incident_dossier_workflow(
        request=request,
        engine=engine,
        run_id="run-phase6-inc-writeback",
        artifacts_root=artifacts_root,
    )

    assert run_record.status == "partial"
    assert run_record.error is not None
    assert run_record.error.code == "INCIDENT_WRITEBACK_FAILED"
    persisted = json.loads((artifacts_root / "run-phase6-inc-writeback" / "run_record.json").read_text())
    assert persisted["status"] == "partial"
    assert persisted["error"]["code"] == "INCIDENT_WRITEBACK_FAILED"
