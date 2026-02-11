# ABOUTME: Verifies Phase 2B run-record persistence for both successful and failed engine invocations.
# ABOUTME: Ensures every invocation writes artifacts/investigator_runs/<run_id>/run_record.json.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from investigator.runtime.contracts import EvidenceRef, InputRef, RCAReport, TimeWindow, hash_excerpt
from investigator.runtime.runner import run_engine


class _SuccessfulRCAEngine:
    engine_type = "rca"
    output_contract_name = "RCAReport"
    engine_version = "0.1.0-test"
    model_name = "gpt-5-mini"
    prompt_template_hash = "phase2b-test"
    temperature = 0.0

    def build_input_ref(self, request: dict[str, str]) -> InputRef:
        return InputRef(
            project_name=request["project_name"],
            trace_ids=[request["trace_id"]],
            time_window=TimeWindow(),
            filter_expr=None,
            controls_version=None,
        )

    def run(self, request: dict[str, str]) -> RCAReport:
        evidence = EvidenceRef(
            trace_id=request["trace_id"],
            span_id="root-span",
            kind="SPAN",
            ref="root-span",
            excerpt_hash=hash_excerpt("ok"),
            ts=None,
        )
        return RCAReport(
            trace_id=request["trace_id"],
            primary_label="instruction_failure",
            summary="ok",
            confidence=0.3,
            evidence_refs=[evidence],
        )


class _FailingRCAEngine(_SuccessfulRCAEngine):
    def run(self, request: dict[str, str]) -> RCAReport:
        raise RuntimeError("intentional failure for persistence test")


def test_run_engine_persists_run_record_and_output_on_success(tmp_path: Path) -> None:
    run_id = "run-success-1"
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    request = {"trace_id": "trace-1", "project_name": "phase2-project"}
    engine = _SuccessfulRCAEngine()

    report, run_record = run_engine(
        engine=engine,
        request=request,
        run_id=run_id,
        artifacts_root=artifacts_root,
    )

    assert report.trace_id == "trace-1"
    run_dir = artifacts_root / run_id
    run_record_path = run_dir / "run_record.json"
    output_path = run_dir / "output.json"
    assert run_record_path.exists()
    assert output_path.exists()

    persisted = json.loads(run_record_path.read_text())
    assert persisted["run_id"] == run_id
    assert persisted["status"] == "succeeded"
    assert persisted["output_ref"]["artifact_path"] == str(output_path)
    assert run_record.output_ref.artifact_path == str(output_path)


def test_run_engine_persists_failed_run_record(tmp_path: Path) -> None:
    run_id = "run-failed-1"
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    request = {"trace_id": "trace-2", "project_name": "phase2-project"}
    engine = _FailingRCAEngine()

    with pytest.raises(RuntimeError):
        run_engine(
            engine=engine,
            request=request,
            run_id=run_id,
            artifacts_root=artifacts_root,
        )

    run_record_path = artifacts_root / run_id / "run_record.json"
    assert run_record_path.exists()
    persisted = json.loads(run_record_path.read_text())
    assert persisted["run_id"] == run_id
    assert persisted["status"] == "failed"
    assert persisted["error"]["code"] == "UNEXPECTED_RUNTIME_ERROR"
