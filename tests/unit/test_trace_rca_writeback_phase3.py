# ABOUTME: Validates RCA Phoenix write-back payload shaping and workflow linkage.
# ABOUTME: Ensures root/evidence evaluations are logged and persisted in run_record metadata.

from __future__ import annotations

import json
from pathlib import Path

from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.rca.workflow import run_trace_rca_workflow
from investigator.rca.writeback import write_rca_to_phoenix
from investigator.runtime.contracts import EvidenceRef, RCAReport, hash_excerpt


class _FakePhoenixClient:
    def __init__(self) -> None:
        self.logged_evaluations = []

    def log_evaluations(self, *evaluations, **kwargs) -> None:  # noqa: ANN003
        self.logged_evaluations.extend(evaluations)


class _SimpleInspectionAPI:
    def list_spans(self, trace_id: str):  # noqa: ANN201
        return [
            {
                "trace_id": trace_id,
                "span_id": "root-span",
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

    def get_span(self, span_id: str):  # noqa: ANN201
        return {
            "summary": self.list_spans("trace-rca")[0],
            "attributes": {},
            "events": [],
        }

    def get_children(self, span_id: str):  # noqa: ANN201
        return []

    def get_tool_io(self, span_id: str):  # noqa: ANN201
        return {
            "trace_id": "trace-rca",
            "span_id": span_id,
            "artifact_id": f"tool:{span_id}",
            "tool_name": "lookup",
            "input": {"q": "x"},
            "output": "error",
            "status_code": "ERROR",
        }

    def get_retrieval_chunks(self, span_id: str):  # noqa: ANN201
        return []


def test_write_rca_to_phoenix_logs_root_and_evidence_rows() -> None:
    report = RCAReport(
        trace_id="trace-rca",
        primary_label="tool_failure",
        summary="deterministic result",
        confidence=0.71,
        evidence_refs=[
            EvidenceRef(
                trace_id="trace-rca",
                span_id="root-span",
                kind="SPAN",
                ref="root-span",
                excerpt_hash=hash_excerpt("root"),
                ts="2026-02-10T00:00:00Z",
            ),
            EvidenceRef(
                trace_id="trace-rca",
                span_id="child-tool",
                kind="TOOL_IO",
                ref="tool:child-tool",
                excerpt_hash=hash_excerpt("tool"),
                ts="2026-02-10T00:00:01Z",
            ),
        ],
    )
    client = _FakePhoenixClient()
    result = write_rca_to_phoenix(report=report, run_id="run-rca-1", client=client)

    assert len(client.logged_evaluations) == 2
    eval_names = {evaluation.eval_name for evaluation in client.logged_evaluations}
    assert eval_names == {"rca.primary", "rca.evidence"}
    assert result["annotation_names"] == ["rca.primary", "rca.evidence"]
    assert result["annotator_kinds"] == ["LLM", "CODE"]

    root_eval = next(evaluation for evaluation in client.logged_evaluations if evaluation.eval_name == "rca.primary")
    root_payload = json.loads(root_eval.dataframe.iloc[0]["explanation"])
    assert root_payload["run_id"] == "run-rca-1"
    assert root_payload["annotator_kind"] == "LLM"


def test_run_trace_rca_workflow_persists_writeback_metadata(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    request = TraceRCARequest(trace_id="trace-rca", project_name="phase3")
    engine = TraceRCAEngine(inspection_api=_SimpleInspectionAPI(), max_hot_spans=1)
    client = _FakePhoenixClient()

    report, run_record = run_trace_rca_workflow(
        request=request,
        engine=engine,
        run_id="run-rca-2",
        artifacts_root=artifacts_root,
        writeback_client=client,
    )

    assert report.trace_id == "trace-rca"
    assert run_record.writeback_ref.writeback_status == "succeeded"
    assert run_record.writeback_ref.annotation_names == ["rca.primary", "rca.evidence"]

    persisted = json.loads((artifacts_root / "run-rca-2" / "run_record.json").read_text())
    assert persisted["writeback_ref"]["writeback_status"] == "succeeded"
    assert persisted["writeback_ref"]["annotation_names"] == ["rca.primary", "rca.evidence"]
