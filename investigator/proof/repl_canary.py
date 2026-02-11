# ABOUTME: Runs a checkpointed Phase 10 REPL-mode canary proof for RCA and policy compliance on a trace subset.
# ABOUTME: Produces baseline-vs-REPL deltas, gate results, and resumable per-trace run artifacts.

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
from typing import Any, Callable

from dotenv import load_dotenv

from investigator.compliance.engine import PolicyComplianceEngine, PolicyComplianceRequest
from investigator.inspection_api import ParquetInspectionAPI
from investigator.proof.benchmark import (
    DEFAULT_DELTA_THRESHOLDS,
    _baseline_compliance_verdict,
    _baseline_rca_label,
    _expected_compliance_verdict,
    _resolved_delta_thresholds,
)
from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.runtime.contracts import RuntimeBudget
from investigator.runtime.runner import run_engine


DEFAULT_PROOF_PROJECT = "phase10-proof-repl"
DEFAULT_CONTROLS_VERSION = "controls-v1"

PROOF_FAST_RUNTIME_BUDGET = RuntimeBudget(
    max_iterations=5,
    max_depth=2,
    max_tool_calls=16,
    max_subcalls=6,
    max_tokens_total=35000,
    max_cost_usd=0.25,
    max_wall_time_sec=90,
)

PROOF_FAST_COMPLIANCE_RUNTIME_BUDGET = RuntimeBudget(
    max_iterations=8,
    max_depth=2,
    max_tool_calls=24,
    max_subcalls=8,
    max_tokens_total=50000,
    max_cost_usd=0.4,
    max_wall_time_sec=140,
)

DEFAULT_COMPLIANCE_RUN_HEALTH_THRESHOLDS = {
    "failed_max": 0.0,
    "partial_rate_max": 0.8,
}

RCAExecutor = Callable[..., dict[str, Any]]
ComplianceExecutor = Callable[..., dict[str, Any]]


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_expected_labels(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels: dict[str, str] = {}
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        trace_id = str(case.get("trace_id") or "").strip()
        expected_label = str(case.get("expected_label") or "").strip()
        if trace_id and expected_label:
            labels[trace_id] = expected_label
    return labels


def _runtime_budget_dict(budget: RuntimeBudget) -> dict[str, Any]:
    return {
        "max_iterations": int(budget.max_iterations),
        "max_depth": int(budget.max_depth),
        "max_tool_calls": int(budget.max_tool_calls),
        "max_subcalls": int(budget.max_subcalls),
        "max_tokens_total": budget.max_tokens_total,
        "max_cost_usd": budget.max_cost_usd,
        "max_wall_time_sec": int(budget.max_wall_time_sec),
    }


def _run_error_payload(error: Any) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "code": str(getattr(error, "code", "") or ""),
        "message": str(getattr(error, "message", "") or ""),
        "stage": str(getattr(error, "stage", "") or ""),
        "retryable": bool(getattr(error, "retryable", False)),
    }


class _ExecutorTimeoutError(TimeoutError):
    pass


def _execute_with_timeout(
    *,
    fn: Callable[[], dict[str, Any]],
    timeout_sec: float,
) -> dict[str, Any]:
    if timeout_sec <= 0.0:
        return fn()

    def _alarm_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise _ExecutorTimeoutError("Executor call timed out.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _invoke_executor(
    *,
    fn: Callable[[], dict[str, Any]],
    timeout_sec: float | None,
    run_id: str,
    stage: str,
    prediction_key: str,
) -> dict[str, Any]:
    try:
        if timeout_sec is None:
            return fn()
        return _execute_with_timeout(fn=fn, timeout_sec=float(timeout_sec))
    except _ExecutorTimeoutError:
        return {
            "run_id": run_id,
            "status": "failed_timeout",
            prediction_key: None,
            "error": {
                "code": "EXECUTOR_TIMEOUT",
                "message": f"Executor timed out after {float(timeout_sec):.3f}s.",
                "stage": stage,
                "retryable": True,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "run_id": run_id,
            "status": "failed_exception",
            prediction_key: None,
            "error": {
                "code": "EXECUTOR_EXCEPTION",
                "message": str(exc),
                "stage": stage,
                "retryable": False,
            },
        }


def _default_rca_executor(
    *,
    trace_id: str,
    run_id: str,
    project_name: str,
    inspection_api: Any,
    runtime_budget: RuntimeBudget,
    evaluator_artifacts_root: Path,
) -> dict[str, Any]:
    engine = TraceRCAEngine(
        inspection_api=inspection_api,
        use_llm_judgment=True,
        use_repl_runtime=True,
        recursive_budget=runtime_budget,
    )
    try:
        report, run_record = run_engine(
            engine=engine,
            request=TraceRCARequest(trace_id=trace_id, project_name=project_name),
            run_id=run_id,
            budget=runtime_budget,
            artifacts_root=evaluator_artifacts_root,
        )
        return {
            "run_id": run_record.run_id,
            "status": run_record.status,
            "predicted_label": str(report.primary_label),
            "error": _run_error_payload(run_record.error),
        }
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, RuntimeError) and exc.args and isinstance(exc.args[0], dict):
            payload = exc.args[0]
            return {
                "run_id": str(payload.get("run_id") or run_id),
                "status": str(payload.get("status") or "failed"),
                "predicted_label": None,
                "error": payload.get("error"),
            }
        return {
            "run_id": run_id,
            "status": "failed_exception",
            "predicted_label": None,
            "error": {"code": "EXCEPTION", "message": str(exc), "stage": "rca_executor"},
        }


def _default_compliance_executor(
    *,
    trace_id: str,
    run_id: str,
    project_name: str,
    controls_version: str,
    inspection_api: Any,
    runtime_budget: RuntimeBudget,
    evaluator_artifacts_root: Path,
) -> dict[str, Any]:
    engine = PolicyComplianceEngine(
        inspection_api=inspection_api,
        use_llm_judgment=True,
        use_repl_runtime=True,
        recursive_budget=runtime_budget,
    )
    try:
        report, run_record = run_engine(
            engine=engine,
            request=PolicyComplianceRequest(
                trace_id=trace_id,
                project_name=project_name,
                controls_version=controls_version,
            ),
            run_id=run_id,
            budget=runtime_budget,
            artifacts_root=evaluator_artifacts_root,
        )
        return {
            "run_id": run_record.run_id,
            "status": run_record.status,
            "predicted_verdict": str(report.overall_verdict),
            "error": _run_error_payload(run_record.error),
        }
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, RuntimeError) and exc.args and isinstance(exc.args[0], dict):
            payload = exc.args[0]
            return {
                "run_id": str(payload.get("run_id") or run_id),
                "status": str(payload.get("status") or "failed"),
                "predicted_verdict": None,
                "error": payload.get("error"),
            }
        return {
            "run_id": run_id,
            "status": "failed_exception",
            "predicted_verdict": None,
            "error": {"code": "EXCEPTION", "message": str(exc), "stage": "compliance_executor"},
        }


def _new_checkpoint(
    *,
    proof_run_id: str,
    trace_limit: int,
    selected_trace_ids: list[str],
    runtime_budget: RuntimeBudget,
    compliance_runtime_budget: RuntimeBudget,
) -> dict[str, Any]:
    return {
        "proof_run_id": proof_run_id,
        "trace_limit": int(trace_limit),
        "selected_trace_ids": [str(item) for item in selected_trace_ids],
        "runtime_budget": _runtime_budget_dict(runtime_budget),
        "compliance_runtime_budget": _runtime_budget_dict(compliance_runtime_budget),
        "trace_results": {},
    }


def _resolved_compliance_run_health_thresholds(
    thresholds: dict[str, float] | None,
) -> dict[str, float]:
    resolved = dict(DEFAULT_COMPLIANCE_RUN_HEALTH_THRESHOLDS)
    if thresholds:
        if "failed_max" in thresholds:
            resolved["failed_max"] = max(0.0, float(thresholds["failed_max"]))
        if "partial_rate_max" in thresholds:
            value = float(thresholds["partial_rate_max"])
            resolved["partial_rate_max"] = max(0.0, min(1.0, value))
    return resolved


def _compute_report(
    *,
    proof_run_id: str,
    checkpoint: dict[str, Any],
    manifest_path: Path,
    spans_parquet_path: Path | None,
    thresholds: dict[str, float],
    compliance_run_health_thresholds: dict[str, float],
) -> dict[str, Any]:
    trace_ids = [str(item) for item in (checkpoint.get("selected_trace_ids") or []) if str(item)]
    trace_results = checkpoint.get("trace_results") or {}

    rca_baseline_correct = 0
    rca_repl_correct = 0
    compliance_baseline_correct = 0
    compliance_repl_correct = 0
    rca_status_counts: dict[str, int] = {}
    compliance_status_counts: dict[str, int] = {}
    rca_failures: list[dict[str, Any]] = []
    compliance_failures: list[dict[str, Any]] = []
    rca_run_ids: list[str] = []
    compliance_run_ids: list[str] = []

    for trace_id in trace_ids:
        entry = trace_results.get(trace_id) or {}
        expected_label = str(entry.get("expected_label") or "")
        expected_compliance_verdict = str(entry.get("expected_compliance_verdict") or "")
        baseline_label = str(entry.get("baseline_rca_label") or "")
        baseline_compliance_verdict = str(entry.get("baseline_compliance_verdict") or "")

        if baseline_label == expected_label:
            rca_baseline_correct += 1
        if baseline_compliance_verdict == expected_compliance_verdict:
            compliance_baseline_correct += 1

        rca_payload = entry.get("rca") or {}
        rca_status = str(rca_payload.get("status") or "missing")
        rca_status_counts[rca_status] = rca_status_counts.get(rca_status, 0) + 1
        rca_prediction = str(rca_payload.get("predicted_label") or "")
        if rca_prediction == expected_label:
            rca_repl_correct += 1
        rca_run_id = str(rca_payload.get("run_id") or "")
        if rca_run_id:
            rca_run_ids.append(rca_run_id)
        if rca_status not in {"succeeded", "partial"}:
            rca_failures.append(
                {
                    "trace_id": trace_id,
                    "status": rca_status,
                    "error": rca_payload.get("error"),
                }
            )

        compliance_payload = entry.get("compliance") or {}
        compliance_status = str(compliance_payload.get("status") or "missing")
        compliance_status_counts[compliance_status] = (
            compliance_status_counts.get(compliance_status, 0) + 1
        )
        compliance_prediction = str(compliance_payload.get("predicted_verdict") or "")
        if compliance_prediction == expected_compliance_verdict:
            compliance_repl_correct += 1
        compliance_run_id = str(compliance_payload.get("run_id") or "")
        if compliance_run_id:
            compliance_run_ids.append(compliance_run_id)
        if compliance_status not in {"succeeded", "partial"}:
            compliance_failures.append(
                {
                    "trace_id": trace_id,
                    "status": compliance_status,
                    "error": compliance_payload.get("error"),
                }
            )

    denominator = len(trace_ids) if trace_ids else 1
    rca_baseline_accuracy = rca_baseline_correct / denominator
    rca_repl_accuracy = rca_repl_correct / denominator
    compliance_baseline_accuracy = compliance_baseline_correct / denominator
    compliance_repl_accuracy = compliance_repl_correct / denominator

    rca_delta = rca_repl_accuracy - rca_baseline_accuracy
    compliance_delta = compliance_repl_accuracy - compliance_baseline_accuracy
    compliance_failed_count = len(compliance_failures)
    compliance_partial_count = int(compliance_status_counts.get("partial", 0))
    compliance_partial_rate = compliance_partial_count / denominator

    gate_results = {
        "rca": {
            "metric": "delta.accuracy",
            "threshold": float(thresholds["rca"]),
            "actual": rca_delta,
            "passed": rca_delta >= float(thresholds["rca"]),
        },
        "compliance": {
            "metric": "delta.accuracy",
            "threshold": float(thresholds["compliance"]),
            "actual": compliance_delta,
            "passed": compliance_delta >= float(thresholds["compliance"]),
        },
        "compliance_run_health": {
            "metric": "status.failed_count_and_partial_rate",
            "failed_max": int(compliance_run_health_thresholds["failed_max"]),
            "partial_rate_max": float(compliance_run_health_thresholds["partial_rate_max"]),
            "failed_count": int(compliance_failed_count),
            "partial_rate": compliance_partial_rate,
            "passed": (
                compliance_failed_count <= int(compliance_run_health_thresholds["failed_max"])
                and compliance_partial_rate <= float(compliance_run_health_thresholds["partial_rate_max"])
            ),
        },
    }

    return {
        "proof_run_id": proof_run_id,
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "manifest_path": str(manifest_path),
            "spans_parquet_path": str(spans_parquet_path) if spans_parquet_path is not None else None,
            "dataset_hash": _file_sha256(spans_parquet_path),
            "trace_count": len(trace_ids),
            "selected_trace_ids": trace_ids,
        },
        "runtime": {
            "mode": "repl_canary",
            "budget": checkpoint.get("runtime_budget") or {},
            "compliance_budget": checkpoint.get("compliance_runtime_budget") or {},
        },
        "capabilities": {
            "rca": {
                "sample_count": len(trace_ids),
                "baseline": {"accuracy": rca_baseline_accuracy},
                "repl": {"accuracy": rca_repl_accuracy},
                "delta": {"accuracy": rca_delta},
                "status_counts": rca_status_counts,
                "failures": rca_failures,
            },
            "compliance": {
                "sample_count": len(trace_ids),
                "baseline": {"accuracy": compliance_baseline_accuracy},
                "repl": {"accuracy": compliance_repl_accuracy},
                "delta": {"accuracy": compliance_delta},
                "status_counts": compliance_status_counts,
                "failures": compliance_failures,
            },
        },
        "gates": {
            "thresholds": {
                "rca": float(thresholds["rca"]),
                "compliance": float(thresholds["compliance"]),
                "compliance_run_health": {
                    "failed_max": int(compliance_run_health_thresholds["failed_max"]),
                    "partial_rate_max": float(compliance_run_health_thresholds["partial_rate_max"]),
                },
            },
            "results": gate_results,
            "all_passed": bool(
                gate_results["rca"]["passed"]
                and gate_results["compliance"]["passed"]
                and gate_results["compliance_run_health"]["passed"]
            ),
        },
        "run_artifacts": {
            "rca": rca_run_ids,
            "compliance": compliance_run_ids,
        },
    }


def run_phase10_repl_canary_proof(
    *,
    proof_run_id: str,
    manifest_path: str | Path,
    trace_limit: int = 5,
    inspection_api: Any | None = None,
    spans_parquet_path: str | Path | None = None,
    project_name: str = DEFAULT_PROOF_PROJECT,
    controls_version: str = DEFAULT_CONTROLS_VERSION,
    controls_dir: str | Path | None = None,
    snapshots_dir: str | Path | None = None,
    proof_artifacts_root: str | Path = "artifacts/proof_runs",
    evaluator_artifacts_root: str | Path = "artifacts/investigator_runs",
    runtime_budget: RuntimeBudget | None = None,
    compliance_runtime_budget: RuntimeBudget | None = None,
    delta_thresholds: dict[str, float] | None = None,
    compliance_run_health_thresholds: dict[str, float] | None = None,
    executor_timeout_sec: float | None = 240.0,
    resume: bool = True,
    rca_executor: RCAExecutor | None = None,
    compliance_executor: ComplianceExecutor | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    labels = _load_expected_labels(manifest_file)
    selected_trace_ids = sorted(labels.keys())[: max(0, int(trace_limit))]
    active_budget = runtime_budget or PROOF_FAST_RUNTIME_BUDGET
    active_compliance_budget = (
        compliance_runtime_budget
        or runtime_budget
        or PROOF_FAST_COMPLIANCE_RUNTIME_BUDGET
    )

    active_api = inspection_api
    active_parquet_path = Path(spans_parquet_path) if spans_parquet_path is not None else None
    if active_api is None:
        if active_parquet_path is None:
            raise ValueError("spans_parquet_path is required when inspection_api is not provided.")
        if controls_dir is None or snapshots_dir is None:
            raise ValueError(
                "controls_dir and snapshots_dir are required when inspection_api is not provided."
            )
        active_api = ParquetInspectionAPI(
            parquet_path=active_parquet_path,
            project_name=project_name,
            controls_dir=controls_dir,
            snapshots_dir=snapshots_dir,
        )
        active_api.attach_manifest_trace_ids(manifest_path=manifest_file)

    proof_root = Path(proof_artifacts_root) / proof_run_id
    checkpoint_path = proof_root / "checkpoint.json"

    if resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    else:
        checkpoint = _new_checkpoint(
            proof_run_id=proof_run_id,
            trace_limit=trace_limit,
            selected_trace_ids=selected_trace_ids,
            runtime_budget=active_budget,
            compliance_runtime_budget=active_compliance_budget,
        )
        _ensure_json(checkpoint_path, checkpoint)

    trace_results = checkpoint.setdefault("trace_results", {})
    active_rca_executor = rca_executor or _default_rca_executor
    active_compliance_executor = compliance_executor or _default_compliance_executor
    evaluator_root = Path(evaluator_artifacts_root)

    for index, trace_id in enumerate(selected_trace_ids, start=1):
        entry = trace_results.get(trace_id) if isinstance(trace_results, dict) else None
        if not isinstance(entry, dict):
            entry = {}
            trace_results[trace_id] = entry

        if "expected_label" not in entry:
            entry["expected_label"] = labels[trace_id]
        if "expected_compliance_verdict" not in entry:
            entry["expected_compliance_verdict"] = _expected_compliance_verdict(labels[trace_id])
        spans = active_api.list_spans(trace_id)
        if "baseline_rca_label" not in entry:
            entry["baseline_rca_label"] = _baseline_rca_label(spans)
        if "baseline_compliance_verdict" not in entry:
            entry["baseline_compliance_verdict"] = _baseline_compliance_verdict(spans)

        if not isinstance(entry.get("rca"), dict):
            rca_run_id = f"{proof_run_id}-rca-{index:04d}"
            entry["rca"] = _invoke_executor(
                fn=lambda: active_rca_executor(
                    trace_id=trace_id,
                    run_id=rca_run_id,
                    project_name=project_name,
                    inspection_api=active_api,
                    runtime_budget=active_budget,
                    evaluator_artifacts_root=evaluator_root,
                ),
                timeout_sec=executor_timeout_sec,
                run_id=rca_run_id,
                stage="rca_executor",
                prediction_key="predicted_label",
            )
            _ensure_json(checkpoint_path, checkpoint)

        if not isinstance(entry.get("compliance"), dict):
            compliance_run_id = f"{proof_run_id}-compliance-{index:04d}"
            entry["compliance"] = _invoke_executor(
                fn=lambda: active_compliance_executor(
                    trace_id=trace_id,
                    run_id=compliance_run_id,
                    project_name=project_name,
                    controls_version=controls_version,
                    inspection_api=active_api,
                    runtime_budget=active_compliance_budget,
                    evaluator_artifacts_root=evaluator_root,
                ),
                timeout_sec=executor_timeout_sec,
                run_id=compliance_run_id,
                stage="compliance_executor",
                prediction_key="predicted_verdict",
            )
            _ensure_json(checkpoint_path, checkpoint)

    thresholds = _resolved_delta_thresholds(delta_thresholds)
    resolved_compliance_run_health_thresholds = _resolved_compliance_run_health_thresholds(
        compliance_run_health_thresholds
    )
    report = _compute_report(
        proof_run_id=proof_run_id,
        checkpoint=checkpoint,
        manifest_path=manifest_file,
        spans_parquet_path=active_parquet_path,
        thresholds=thresholds,
        compliance_run_health_thresholds=resolved_compliance_run_health_thresholds,
    )
    report_path = proof_root / "proof_report.json"
    _ensure_json(report_path, report)
    report["checkpoint_path"] = str(checkpoint_path)
    report["proof_report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 10 REPL canary proof (RCA + compliance).")
    parser.add_argument("--proof-run-id", default=f"phase10-repl-canary-{_utc_timestamp()}")
    parser.add_argument("--manifest-path", default="datasets/seeded_failures/manifest.json")
    parser.add_argument("--spans-parquet-path", default="datasets/seeded_failures/exports/spans.parquet")
    parser.add_argument("--controls-dir", default="controls/library")
    parser.add_argument("--snapshots-dir", default="configs/snapshots")
    parser.add_argument("--project-name", default=DEFAULT_PROOF_PROJECT)
    parser.add_argument("--controls-version", default=DEFAULT_CONTROLS_VERSION)
    parser.add_argument("--trace-limit", type=int, default=5)
    parser.add_argument("--proof-artifacts-root", default="artifacts/proof_runs")
    parser.add_argument("--evaluator-artifacts-root", default="artifacts/investigator_runs")
    parser.add_argument("--executor-timeout-sec", type=float, default=240.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    repo_root = Path.cwd()
    load_dotenv(repo_root / ".env", override=False)

    report = run_phase10_repl_canary_proof(
        proof_run_id=str(args.proof_run_id),
        manifest_path=Path(args.manifest_path),
        spans_parquet_path=Path(args.spans_parquet_path),
        project_name=str(args.project_name),
        controls_version=str(args.controls_version),
        controls_dir=Path(args.controls_dir),
        snapshots_dir=Path(args.snapshots_dir),
        trace_limit=int(args.trace_limit),
        proof_artifacts_root=Path(args.proof_artifacts_root),
        evaluator_artifacts_root=Path(args.evaluator_artifacts_root),
        executor_timeout_sec=float(args.executor_timeout_sec),
        resume=bool(args.resume),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
