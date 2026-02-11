# ABOUTME: Validates compliance Phoenix write-back payload shape and workflow run-record persistence.
# ABOUTME: Ensures annotation names, annotator kinds, and controls-version traceability are recorded.

from __future__ import annotations

import json
from pathlib import Path

from investigator.compliance.engine import PolicyComplianceEngine, PolicyComplianceRequest
from investigator.compliance.workflow import run_policy_compliance_workflow
from investigator.compliance.writeback import write_compliance_to_phoenix
from investigator.runtime.contracts import ComplianceFinding, ComplianceReport, EvidenceRef, hash_excerpt


class _FakeSpansResource:
    def __init__(self) -> None:
        self.logged_annotations = []

    def log_span_annotations(self, *, span_annotations, sync: bool = False) -> None:  # noqa: ANN001
        self.logged_annotations.extend(span_annotations)


class _FakePhoenixClient:
    def __init__(self) -> None:
        self.spans = _FakeSpansResource()


class _SimpleInspectionAPI:
    def list_spans(self, trace_id: str):  # noqa: ANN201
        return [
            {
                "trace_id": trace_id,
                "span_id": "root-span",
                "parent_id": None,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "OK",
                "status_message": "",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 100.0,
            }
        ]

    def list_controls(self, controls_version: str, app_type=None, tools_used=None, data_domains=None):  # noqa: ANN001,ANN201
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
        return [{"trace_id": "trace-cmp", "span_id": "root-span", "role": "user", "content": "hi"}]


def test_write_compliance_to_phoenix_logs_expected_annotation_rows() -> None:
    report = ComplianceReport(
        trace_id="trace-cmp",
        controls_version="controls-v1",
        controls_evaluated=[
            ComplianceFinding(
                controls_version="controls-v1",
                control_id="control.alpha",
                pass_fail="insufficient_evidence",
                severity="high",
                confidence=0.41,
                evidence_refs=[
                    EvidenceRef(
                        trace_id="trace-cmp",
                        span_id="root-span",
                        kind="SPAN",
                        ref="root-span",
                        excerpt_hash=hash_excerpt("root"),
                        ts="2026-02-10T00:00:00Z",
                    )
                ],
                missing_evidence=["required_tool_io"],
                remediation="Collect required evidence.",
            )
        ],
        overall_verdict="needs_review",
        overall_confidence=0.41,
    )
    client = _FakePhoenixClient()
    result = write_compliance_to_phoenix(report=report, run_id="run-cmp-1", client=client)

    assert result["annotation_names"] == ["compliance.overall", "compliance.control.control.alpha.evidence"]
    assert result["annotator_kinds"] == ["CODE"]
    assert len(client.spans.logged_annotations) == 2

    root_annotation = next(
        item for item in client.spans.logged_annotations if item["name"] == "compliance.overall"
    )
    payload = json.loads(root_annotation["result"]["explanation"])
    assert payload["run_id"] == "run-cmp-1"
    assert payload["controls_version"] == "controls-v1"
    assert payload["annotator_kind"] == "CODE"


def test_write_compliance_to_phoenix_allows_llm_provenance_override() -> None:
    report = ComplianceReport(
        trace_id="trace-cmp",
        controls_version="controls-v1",
        controls_evaluated=[
            ComplianceFinding(
                controls_version="controls-v1",
                control_id="control.alpha",
                pass_fail="fail",
                severity="high",
                confidence=0.72,
                evidence_refs=[
                    EvidenceRef(
                        trace_id="trace-cmp",
                        span_id="root-span",
                        kind="SPAN",
                        ref="root-span",
                        excerpt_hash=hash_excerpt("root"),
                        ts="2026-02-10T00:00:00Z",
                    )
                ],
                missing_evidence=[],
                remediation="Address failure.",
            )
        ],
        overall_verdict="non_compliant",
        overall_confidence=0.72,
    )
    client = _FakePhoenixClient()
    result = write_compliance_to_phoenix(
        report=report,
        run_id="run-cmp-llm",
        client=client,
        primary_annotator_kind="LLM",
    )

    assert result["annotator_kinds"] == ["LLM"]
    root_annotation = next(
        item for item in client.spans.logged_annotations if item["name"] == "compliance.overall"
    )
    payload = json.loads(root_annotation["result"]["explanation"])
    assert payload["annotator_kind"] == "LLM"


def test_run_policy_compliance_workflow_persists_writeback_metadata(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    engine = PolicyComplianceEngine(inspection_api=_SimpleInspectionAPI(), max_controls=5)
    request = PolicyComplianceRequest(
        trace_id="trace-cmp",
        project_name="phase4",
        controls_version="controls-v1",
    )
    client = _FakePhoenixClient()

    report, run_record = run_policy_compliance_workflow(
        request=request,
        engine=engine,
        run_id="run-cmp-2",
        artifacts_root=artifacts_root,
        writeback_client=client,
    )

    assert report.controls_version == "controls-v1"
    assert run_record.writeback_ref.writeback_status == "succeeded"
    assert run_record.writeback_ref.annotation_names[0] == "compliance.overall"
    assert run_record.writeback_ref.annotator_kinds == ["CODE"]
    persisted = json.loads((artifacts_root / "run-cmp-2" / "run_record.json").read_text())
    assert persisted["writeback_ref"]["writeback_status"] == "succeeded"
    assert persisted["writeback_ref"]["annotator_kinds"] == ["CODE"]
