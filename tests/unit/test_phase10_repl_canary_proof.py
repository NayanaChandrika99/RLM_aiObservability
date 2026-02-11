# ABOUTME: Validates Phase 10 REPL canary proof runner checkpoint/resume behavior and delta gate reporting.
# ABOUTME: Ensures RCA/compliance canary metrics are persisted from deterministic fake executors.

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from investigator.runtime.contracts import RuntimeBudget
from investigator.proof.repl_canary import run_phase10_repl_canary_proof


class _FakeInspectionAPI:
    def __init__(self, spans_by_trace: dict[str, list[dict[str, Any]]]) -> None:
        self._spans_by_trace = spans_by_trace

    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self._spans_by_trace.get(trace_id, [])]


def _write_manifest(path: Path, trace_to_label: dict[str, str]) -> None:
    payload = {
        "dataset_id": "seeded_failures_v1",
        "generator_version": "0.1.0",
        "seed": 42,
        "cases": [
            {
                "run_id": f"seed_run_{index:04d}",
                "trace_id": trace_id,
                "expected_label": label,
            }
            for index, (trace_id, label) in enumerate(sorted(trace_to_label.items()), start=1)
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_phase10_repl_canary_writes_checkpoint_and_resumes_without_duplicate_runs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    trace_to_label = {
        "trace-a": "tool_failure",
        "trace-b": "retrieval_failure",
        "trace-c": "instruction_failure",
    }
    _write_manifest(manifest_path, trace_to_label)
    api = _FakeInspectionAPI(
        spans_by_trace={
            "trace-a": [{"trace_id": "trace-a", "span_id": "root-a", "name": "tool.call", "status_code": "ERROR"}],
            "trace-b": [{"trace_id": "trace-b", "span_id": "root-b", "name": "agent.run", "status_code": "UNSET"}],
            "trace-c": [{"trace_id": "trace-c", "span_id": "root-c", "name": "agent.run", "status_code": "UNSET"}],
        }
    )

    calls = {"rca": 0, "compliance": 0}

    def _rca_executor(*, trace_id: str, run_id: str, **_: Any) -> dict[str, Any]:
        calls["rca"] += 1
        return {
            "run_id": run_id,
            "status": "succeeded",
            "predicted_label": trace_to_label[trace_id],
            "error": None,
        }

    def _compliance_executor(*, trace_id: str, run_id: str, **_: Any) -> dict[str, Any]:
        calls["compliance"] += 1
        expected_label = trace_to_label[trace_id]
        expected_verdict = "needs_review" if expected_label in {"retrieval_failure", "instruction_failure"} else "non_compliant"
        return {
            "run_id": run_id,
            "status": "succeeded",
            "predicted_verdict": expected_verdict,
            "error": None,
        }

    first_report = run_phase10_repl_canary_proof(
        proof_run_id="phase10-canary-test",
        manifest_path=manifest_path,
        trace_limit=2,
        inspection_api=api,
        rca_executor=_rca_executor,
        compliance_executor=_compliance_executor,
        proof_artifacts_root=tmp_path / "proof_runs",
        evaluator_artifacts_root=tmp_path / "investigator_runs",
    )

    assert calls == {"rca": 2, "compliance": 2}
    assert first_report["dataset"]["trace_count"] == 2

    second_report = run_phase10_repl_canary_proof(
        proof_run_id="phase10-canary-test",
        manifest_path=manifest_path,
        trace_limit=2,
        inspection_api=api,
        rca_executor=_rca_executor,
        compliance_executor=_compliance_executor,
        proof_artifacts_root=tmp_path / "proof_runs",
        evaluator_artifacts_root=tmp_path / "investigator_runs",
    )

    assert calls == {"rca": 2, "compliance": 2}
    assert second_report["dataset"]["trace_count"] == 2
    checkpoint_path = tmp_path / "proof_runs" / "phase10-canary-test" / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert sorted((checkpoint.get("trace_results") or {}).keys()) == ["trace-a", "trace-b"]


def test_phase10_repl_canary_reports_status_counts_and_delta_gates(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    trace_to_label = {
        "trace-1": "tool_failure",
        "trace-2": "retrieval_failure",
    }
    _write_manifest(manifest_path, trace_to_label)

    api = _FakeInspectionAPI(
        spans_by_trace={
            "trace-1": [{"trace_id": "trace-1", "span_id": "root-1", "name": "tool.call", "status_code": "ERROR"}],
            "trace-2": [{"trace_id": "trace-2", "span_id": "root-2", "name": "agent.run", "status_code": "UNSET"}],
        }
    )

    def _rca_executor(*, trace_id: str, run_id: str, **_: Any) -> dict[str, Any]:
        predictions = {
            "trace-1": "tool_failure",
            "trace-2": "retrieval_failure",
        }
        status = "succeeded" if trace_id == "trace-1" else "partial"
        return {
            "run_id": run_id,
            "status": status,
            "predicted_label": predictions[trace_id],
            "error": None if status == "succeeded" else {"code": "RECURSION_LIMIT_REACHED"},
        }

    def _compliance_executor(*, trace_id: str, run_id: str, **_: Any) -> dict[str, Any]:
        verdicts = {
            "trace-1": "non_compliant",
            "trace-2": "needs_review",
        }
        return {
            "run_id": run_id,
            "status": "succeeded",
            "predicted_verdict": verdicts[trace_id],
            "error": None,
        }

    report = run_phase10_repl_canary_proof(
        proof_run_id="phase10-canary-gates",
        manifest_path=manifest_path,
        trace_limit=2,
        inspection_api=api,
        rca_executor=_rca_executor,
        compliance_executor=_compliance_executor,
        proof_artifacts_root=tmp_path / "proof_runs",
        evaluator_artifacts_root=tmp_path / "investigator_runs",
    )

    rca = report["capabilities"]["rca"]
    compliance = report["capabilities"]["compliance"]
    gates = report["gates"]

    assert rca["status_counts"]["succeeded"] == 1
    assert rca["status_counts"]["partial"] == 1
    assert compliance["status_counts"]["succeeded"] == 2
    assert rca["delta"]["accuracy"] >= 0.0
    assert compliance["delta"]["accuracy"] >= 0.0
    assert gates["results"]["rca"]["passed"] is True
    assert gates["results"]["compliance"]["passed"] is True
    assert gates["results"]["compliance_run_health"]["passed"] is True


def test_phase10_repl_canary_reports_compliance_run_health_gate_failures(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    trace_to_label = {
        "trace-1": "tool_failure",
        "trace-2": "retrieval_failure",
    }
    _write_manifest(manifest_path, trace_to_label)

    api = _FakeInspectionAPI(
        spans_by_trace={
            "trace-1": [{"trace_id": "trace-1", "span_id": "root-1", "name": "tool.call", "status_code": "ERROR"}],
            "trace-2": [{"trace_id": "trace-2", "span_id": "root-2", "name": "agent.run", "status_code": "UNSET"}],
        }
    )

    def _rca_executor(*, trace_id: str, run_id: str, **_: Any) -> dict[str, Any]:
        predictions = {
            "trace-1": "tool_failure",
            "trace-2": "retrieval_failure",
        }
        return {
            "run_id": run_id,
            "status": "succeeded",
            "predicted_label": predictions[trace_id],
            "error": None,
        }

    def _compliance_executor(*, trace_id: str, run_id: str, **_: Any) -> dict[str, Any]:
        if trace_id == "trace-1":
            return {
                "run_id": run_id,
                "status": "failed",
                "predicted_verdict": "non_compliant",
                "error": {"code": "MODEL_OUTPUT_INVALID"},
            }
        return {
            "run_id": run_id,
            "status": "partial",
            "predicted_verdict": "needs_review",
            "error": {"code": "RECURSION_LIMIT_REACHED"},
        }

    report = run_phase10_repl_canary_proof(
        proof_run_id="phase10-canary-health-gate",
        manifest_path=manifest_path,
        trace_limit=2,
        inspection_api=api,
        rca_executor=_rca_executor,
        compliance_executor=_compliance_executor,
        proof_artifacts_root=tmp_path / "proof_runs",
        evaluator_artifacts_root=tmp_path / "investigator_runs",
    )

    compliance_health = report["gates"]["results"]["compliance_run_health"]
    assert compliance_health["failed_count"] == 1
    assert compliance_health["failed_max"] == 0
    assert compliance_health["partial_rate"] == 0.5
    assert compliance_health["passed"] is False
    assert report["gates"]["all_passed"] is False


def test_phase10_repl_canary_uses_compliance_specific_budget_profile(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, {"trace-1": "tool_failure"})
    api = _FakeInspectionAPI(
        spans_by_trace={
            "trace-1": [{"trace_id": "trace-1", "span_id": "root-1", "name": "tool.call", "status_code": "ERROR"}]
        }
    )
    received_budgets: dict[str, int] = {"rca": -1, "compliance": -1}

    def _rca_executor(*, runtime_budget: RuntimeBudget, **_: Any) -> dict[str, Any]:
        received_budgets["rca"] = int(runtime_budget.max_iterations)
        return {
            "run_id": "rca-run",
            "status": "succeeded",
            "predicted_label": "tool_failure",
            "error": None,
        }

    def _compliance_executor(*, runtime_budget: RuntimeBudget, **_: Any) -> dict[str, Any]:
        received_budgets["compliance"] = int(runtime_budget.max_iterations)
        return {
            "run_id": "compliance-run",
            "status": "succeeded",
            "predicted_verdict": "non_compliant",
            "error": None,
        }

    report = run_phase10_repl_canary_proof(
        proof_run_id="phase10-canary-budget-profile",
        manifest_path=manifest_path,
        trace_limit=1,
        inspection_api=api,
        runtime_budget=RuntimeBudget(max_iterations=4),
        compliance_runtime_budget=RuntimeBudget(max_iterations=9),
        rca_executor=_rca_executor,
        compliance_executor=_compliance_executor,
        proof_artifacts_root=tmp_path / "proof_runs",
        evaluator_artifacts_root=tmp_path / "investigator_runs",
    )

    assert report["dataset"]["trace_count"] == 1
    assert received_budgets["rca"] == 4
    assert received_budgets["compliance"] == 9


def test_phase10_repl_canary_marks_executor_timeout_and_continues(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    trace_to_label = {
        "trace-1": "tool_failure",
        "trace-2": "retrieval_failure",
    }
    _write_manifest(manifest_path, trace_to_label)
    api = _FakeInspectionAPI(
        spans_by_trace={
            "trace-1": [{"trace_id": "trace-1", "span_id": "root-1", "name": "tool.call", "status_code": "ERROR"}],
            "trace-2": [{"trace_id": "trace-2", "span_id": "root-2", "name": "agent.run", "status_code": "UNSET"}],
        }
    )

    def _rca_executor(*, trace_id: str, run_id: str, **_: Any) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "status": "succeeded",
            "predicted_label": trace_to_label[trace_id],
            "error": None,
        }

    def _compliance_executor(*, trace_id: str, run_id: str, **_: Any) -> dict[str, Any]:
        if trace_id == "trace-1":
            time.sleep(0.2)
        verdict = "non_compliant" if trace_id == "trace-1" else "needs_review"
        return {
            "run_id": run_id,
            "status": "succeeded",
            "predicted_verdict": verdict,
            "error": None,
        }

    report = run_phase10_repl_canary_proof(
        proof_run_id="phase10-canary-timeout-guard",
        manifest_path=manifest_path,
        trace_limit=2,
        inspection_api=api,
        rca_executor=_rca_executor,
        compliance_executor=_compliance_executor,
        executor_timeout_sec=0.05,
        proof_artifacts_root=tmp_path / "proof_runs",
        evaluator_artifacts_root=tmp_path / "investigator_runs",
    )

    compliance_status = report["capabilities"]["compliance"]["status_counts"]
    assert compliance_status["failed_timeout"] == 1
    assert compliance_status["succeeded"] == 1
    assert report["gates"]["results"]["compliance_run_health"]["failed_count"] == 1
    assert report["gates"]["all_passed"] is False

    checkpoint_path = tmp_path / "proof_runs" / "phase10-canary-timeout-guard" / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    entry = checkpoint["trace_results"]["trace-1"]["compliance"]
    assert entry["status"] == "failed_timeout"
    assert entry["error"]["code"] == "EXECUTOR_TIMEOUT"
