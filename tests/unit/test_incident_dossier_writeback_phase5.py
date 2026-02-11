# ABOUTME: Validates incident dossier Phoenix write-back payload shape and workflow run-record persistence.
# ABOUTME: Ensures incident root and timeline evidence annotations are recorded with stable metadata.

from __future__ import annotations

import json
from pathlib import Path

from investigator.incident.engine import IncidentDossierEngine, IncidentDossierRequest
from investigator.incident.workflow import run_incident_dossier_workflow
from investigator.incident.writeback import write_incident_to_phoenix
from investigator.runtime.contracts import (
    EvidenceRef,
    IncidentDossier,
    IncidentHypothesis,
    IncidentTimelineEvent,
    RecommendedAction,
    RepresentativeTrace,
    SuspectedChange,
    hash_excerpt,
)


class _FakeSpansResource:
    def __init__(self) -> None:
        self.logged_annotations = []

    def log_span_annotations(self, *, span_annotations, sync: bool = False) -> None:  # noqa: ANN001
        self.logged_annotations.extend(span_annotations)


class _FakePhoenixClient:
    def __init__(self) -> None:
        self.spans = _FakeSpansResource()


class _SimpleIncidentInspectionAPI:
    def list_traces(  # noqa: ANN201
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ):
        return [
            {
                "project_name": project_name,
                "trace_id": "trace-inc-1",
                "span_count": 2,
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:02Z",
                "latency_ms": 200.0,
            },
            {
                "project_name": project_name,
                "trace_id": "trace-inc-2",
                "span_count": 1,
                "start_time": "2026-02-10T00:01:00Z",
                "end_time": "2026-02-10T00:01:01Z",
                "latency_ms": 80.0,
            },
        ]

    def list_spans(self, trace_id: str):  # noqa: ANN201
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
                    "end_time": "2026-02-10T00:00:02Z",
                    "latency_ms": 200.0,
                },
                {
                    "trace_id": trace_id,
                    "span_id": "inc-tool-1",
                    "parent_id": "inc-root-1",
                    "name": "lookup",
                    "span_kind": "TOOL",
                    "status_code": "ERROR",
                    "status_message": "timeout",
                    "start_time": "2026-02-10T00:00:01Z",
                    "end_time": "2026-02-10T00:00:02Z",
                    "latency_ms": 120.0,
                },
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
                "latency_ms": 80.0,
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
                "project_name": "phase5",
                "tag": "baseline",
                "created_at": "2026-02-10T00:00:00Z",
                "git_commit": "abc111",
                "paths": ["agent.yaml"],
                "metadata": {},
            },
            {
                "snapshot_id": "snap-b",
                "project_name": "phase5",
                "tag": "candidate",
                "created_at": "2026-02-10T00:30:00Z",
                "git_commit": "abc222",
                "paths": ["prompts/investigator.txt"],
                "metadata": {},
            },
        ]

    def get_config_diff(self, base_snapshot_id: str, target_snapshot_id: str):  # noqa: ANN201
        return {
            "project_name": "phase5",
            "base_snapshot_id": base_snapshot_id,
            "target_snapshot_id": target_snapshot_id,
            "artifact_id": "configdiff:phase5-1",
            "diff_ref": "configdiff:phase5-1",
            "git_commit_base": "abc111",
            "git_commit_target": "abc222",
            "paths": ["prompts/investigator.txt"],
            "summary": "1 changed file(s)",
        }


def test_write_incident_to_phoenix_logs_expected_annotation_rows() -> None:
    lead_evidence = EvidenceRef(
        trace_id="trace-inc-1",
        span_id="inc-root-1",
        kind="SPAN",
        ref="inc-root-1",
        excerpt_hash=hash_excerpt("lead"),
        ts="2026-02-10T00:00:00Z",
    )
    diff_evidence = EvidenceRef(
        trace_id="trace-inc-1",
        span_id="inc-root-1",
        kind="CONFIG_DIFF",
        ref="configdiff:phase5-1",
        excerpt_hash=hash_excerpt("diff"),
        ts="2026-02-10T00:30:00Z",
    )
    report = IncidentDossier(
        incident_summary="deterministic incident dossier",
        impacted_components=["svc.order.agent"],
        timeline=[
            IncidentTimelineEvent(
                timestamp="2026-02-10T00:00:00Z",
                event="Incident started.",
                evidence_refs=[lead_evidence],
            ),
            IncidentTimelineEvent(
                timestamp="2026-02-10T00:30:00Z",
                event="Config diff correlated.",
                evidence_refs=[lead_evidence, diff_evidence],
            ),
        ],
        representative_traces=[
            RepresentativeTrace(
                trace_id="trace-inc-1",
                why_selected="error_bucket(score=1)",
                evidence_refs=[lead_evidence],
            )
        ],
        suspected_change=SuspectedChange(
            change_type="prompt",
            change_ref="abc222",
            diff_ref="configdiff:phase5-1",
            summary="Prompt changed",
            evidence_refs=[lead_evidence, diff_evidence],
        ),
        hypotheses=[
            IncidentHypothesis(
                rank=1,
                statement="Likely prompt regression.",
                evidence_refs=[lead_evidence, diff_evidence],
                confidence=0.73,
            )
        ],
        recommended_actions=[
            RecommendedAction(
                priority="P1",
                type="follow_up_fix",
                action="Rollback prompt change.",
            )
        ],
        confidence=0.73,
    )
    client = _FakePhoenixClient()

    result = write_incident_to_phoenix(report=report, run_id="run-inc-1", client=client)

    assert result["annotation_names"] == ["incident.dossier", "incident.timeline.evidence"]
    assert result["annotator_kinds"] == ["LLM", "CODE"]
    assert len(client.spans.logged_annotations) >= 3

    root_annotation = next(
        item for item in client.spans.logged_annotations if item["name"] == "incident.dossier"
    )
    root_payload = json.loads(root_annotation["result"]["explanation"])
    assert root_payload["run_id"] == "run-inc-1"
    assert root_payload["annotator_kind"] == "LLM"
    assert root_payload["report"]["incident_summary"] == "deterministic incident dossier"


def test_run_incident_dossier_workflow_persists_writeback_metadata(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    engine = IncidentDossierEngine(inspection_api=_SimpleIncidentInspectionAPI(), max_representatives=2)
    request = IncidentDossierRequest(
        project_name="phase5",
        time_window_start="2026-02-10T00:00:00Z",
        time_window_end="2026-02-10T01:00:00Z",
    )
    client = _FakePhoenixClient()

    report, run_record = run_incident_dossier_workflow(
        request=request,
        engine=engine,
        run_id="run-inc-2",
        artifacts_root=artifacts_root,
        writeback_client=client,
    )

    assert report.incident_summary
    assert run_record.writeback_ref.writeback_status == "succeeded"
    assert run_record.writeback_ref.annotation_names == [
        "incident.dossier",
        "incident.timeline.evidence",
    ]

    persisted = json.loads((artifacts_root / "run-inc-2" / "run_record.json").read_text())
    assert persisted["writeback_ref"]["writeback_status"] == "succeeded"
    assert persisted["writeback_ref"]["annotation_names"] == [
        "incident.dossier",
        "incident.timeline.evidence",
    ]
