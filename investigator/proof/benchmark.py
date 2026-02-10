# ABOUTME: Runs deterministic baseline-vs-RLM comparisons for RCA, compliance, and incident capabilities.
# ABOUTME: Produces reproducible benchmark metrics and links evaluator run artifacts for proof reports.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from investigator.compliance.engine import PolicyComplianceEngine, PolicyComplianceRequest
from investigator.compliance.workflow import run_policy_compliance_workflow
from investigator.incident.engine import IncidentDossierEngine, IncidentDossierRequest
from investigator.incident.workflow import run_incident_dossier_workflow
from investigator.inspection_api import ParquetInspectionAPI
from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.rca.workflow import run_trace_rca_workflow


DEFAULT_DELTA_THRESHOLDS = {
    "rca": 0.15,
    "compliance": 0.05,
    "incident": 0.10,
}


class _FakeEvaluationClient:
    def log_evaluations(self, *evaluations, **kwargs) -> None:  # noqa: ANN002, ANN003
        del evaluations, kwargs


class _FakeSpansResource:
    def log_span_annotations(self, *, span_annotations, sync: bool = False) -> list[dict[str, str]]:  # noqa: ANN001
        del sync
        return [{"id": f"fake-{index}"} for index, _ in enumerate(span_annotations)]


class _FakeAnnotationClient:
    def __init__(self) -> None:
        self.spans = _FakeSpansResource()


def _file_sha256(path: Path) -> str:
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
        trace_id = str(case.get("trace_id") or "")
        label = str(case.get("expected_label") or "")
        if not trace_id or not label:
            continue
        labels[trace_id] = label
    return labels


def _baseline_rca_label(spans: list[dict[str, Any]]) -> str:
    ordered = sorted(
        spans,
        key=lambda span: (
            str(span.get("start_time") or ""),
            str(span.get("span_id") or ""),
        ),
    )
    root = next(
        (
            span
            for span in ordered
            if not str(span.get("parent_id") or "")
        ),
        ordered[0] if ordered else {},
    )
    status_text = str(root.get("status_message") or "").lower()
    name = str(root.get("name") or "")
    if "upstream" in status_text or "503" in status_text or "timeout" in status_text:
        return "upstream_dependency_failure"
    if "retriever" in name:
        return "retrieval_failure"
    if "tool.parse" in name:
        return "data_schema_mismatch"
    if "tool.call" in name:
        return "tool_failure"
    return "instruction_failure"


def _baseline_compliance_verdict(spans: list[dict[str, Any]]) -> str:
    has_error = any(str(span.get("status_code") or "") == "ERROR" for span in spans)
    return "non_compliant" if has_error else "compliant"


def _expected_compliance_verdict(label: str) -> str:
    if label in {"tool_failure", "upstream_dependency_failure", "data_schema_mismatch"}:
        return "non_compliant"
    if label == "retrieval_failure":
        return "compliant"
    return "needs_review"


def _incident_signature(spans: list[dict[str, Any]]) -> tuple[str, str, str]:
    ordered = sorted(
        spans,
        key=lambda span: (
            str(span.get("start_time") or ""),
            str(span.get("span_id") or ""),
        ),
    )
    root = next(
        (
            span
            for span in ordered
            if not str(span.get("parent_id") or "")
        ),
        ordered[0] if ordered else {},
    )
    service = str(root.get("name") or "unknown_service")
    tool_name = "none"
    for span in ordered:
        if str(span.get("span_kind") or "") == "TOOL":
            tool_name = str(span.get("name") or "unknown_tool")
            break
    error_type = "none"
    for span in ordered:
        if str(span.get("status_code") or "") == "ERROR":
            error_type = str(span.get("status_message") or "error").strip().lower() or "error"
            break
    return (service, tool_name, error_type)


def _incident_profiles(
    *,
    api: ParquetInspectionAPI,
    trace_ids: list[str],
    trace_latency_map: dict[str, float],
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for trace_id in trace_ids:
        if not trace_id:
            continue
        spans = api.list_spans(trace_id)
        error_count = sum(1 for span in spans if str(span.get("status_code") or "") == "ERROR")
        latency = float(trace_latency_map.get(trace_id) or 0.0)
        if latency <= 0.0:
            latency = max(
                [float(span.get("latency_ms") or 0.0) for span in spans] or [0.0]
            )
        profiles.append(
            {
                "trace_id": trace_id,
                "error_spans": error_count,
                "latency_ms": latency,
                "signature": _incident_signature(spans),
            }
        )
    return profiles


def _incident_expected_trace_ids(
    *,
    profiles: list[dict[str, Any]],
    expected_labels: dict[str, str],
    top_k: int,
) -> list[str]:
    severe_labels = {"tool_failure", "upstream_dependency_failure", "data_schema_mismatch"}
    severe_candidates = [
        profile
        for profile in profiles
        if str(expected_labels.get(str(profile.get("trace_id") or ""), "")) in severe_labels
    ]
    ordered = sorted(
        severe_candidates,
        key=lambda profile: (
            -int(profile.get("error_spans") or 0),
            -float(profile.get("latency_ms") or 0.0),
            str(profile.get("trace_id") or ""),
        ),
    )
    selected: list[str] = []
    seen_signatures: set[tuple[str, str, str]] = set()
    for profile in ordered:
        if len(selected) >= top_k:
            break
        signature = profile.get("signature")
        if isinstance(signature, tuple) and signature in seen_signatures:
            continue
        trace_id = str(profile.get("trace_id") or "")
        if not trace_id:
            continue
        selected.append(trace_id)
        if isinstance(signature, tuple):
            seen_signatures.add(signature)

    if len(selected) >= top_k:
        return selected[:top_k]

    fallback = sorted(
        profiles,
        key=lambda profile: (
            -int(profile.get("error_spans") or 0),
            -float(profile.get("latency_ms") or 0.0),
            str(profile.get("trace_id") or ""),
        ),
    )
    for profile in fallback:
        if len(selected) >= top_k:
            break
        trace_id = str(profile.get("trace_id") or "")
        if not trace_id or trace_id in selected:
            continue
        selected.append(trace_id)
    return selected[:top_k]


def _incident_baseline_trace_ids(
    *,
    profiles: list[dict[str, Any]],
    top_k: int,
) -> list[str]:
    ordered = sorted(
        profiles,
        key=lambda profile: (
            -float(profile.get("latency_ms") or 0.0),
            str(profile.get("trace_id") or ""),
        ),
    )
    return [
        str(profile.get("trace_id") or "")
        for profile in ordered[:top_k]
        if str(profile.get("trace_id") or "")
    ]


def _overlap_at_k(*, expected: list[str], predicted: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    expected_set = set(expected[:k])
    predicted_set = set(predicted[:k])
    return len(expected_set.intersection(predicted_set)) / float(k)


def _incident_diagnostics(*, expected: list[str], predicted: list[str], k: int) -> dict[str, Any]:
    expected_at_k = expected[:k]
    predicted_at_k = predicted[:k]
    predicted_set = set(predicted_at_k)
    intersection = [trace_id for trace_id in expected_at_k if trace_id in predicted_set]
    rows: list[dict[str, Any]] = []
    for trace_id in sorted(set(expected_at_k).union(predicted_at_k)):
        rows.append(
            {
                "trace_id": trace_id,
                "expected_rank": (expected_at_k.index(trace_id) + 1) if trace_id in expected_at_k else None,
                "predicted_rank": (predicted_at_k.index(trace_id) + 1) if trace_id in predicted_at_k else None,
                "in_intersection": trace_id in intersection,
            }
        )
    return {
        "k": k,
        "expected_ids": expected_at_k,
        "predicted_ids": predicted_at_k,
        "intersection": intersection,
        "rows": rows,
    }


def _resolved_delta_thresholds(delta_thresholds: dict[str, float] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_DELTA_THRESHOLDS)
    for capability, value in (delta_thresholds or {}).items():
        if capability not in resolved:
            continue
        resolved[capability] = float(value)
    return resolved


def run_dataset_benchmark(
    *,
    spans_parquet_path: str | Path,
    manifest_path: str | Path,
    controls_version: str,
    controls_dir: str | Path,
    snapshots_dir: str | Path,
    project_name: str,
    artifacts_root: str | Path = "artifacts/investigator_runs",
    delta_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    parquet_path = Path(spans_parquet_path)
    manifest_file = Path(manifest_path)
    api = ParquetInspectionAPI(
        parquet_path=parquet_path,
        project_name=project_name,
        controls_dir=controls_dir,
        snapshots_dir=snapshots_dir,
    )
    api.attach_manifest_trace_ids(manifest_path=manifest_file)
    expected_labels = _load_expected_labels(manifest_file)
    trace_ids = sorted(expected_labels.keys())

    rca_baseline_correct = 0
    rca_rlm_correct = 0
    rca_run_ids: list[str] = []
    for trace_id in trace_ids:
        spans = api.list_spans(trace_id)
        baseline_label = _baseline_rca_label(spans)
        if baseline_label == expected_labels[trace_id]:
            rca_baseline_correct += 1

        _, run_record = run_trace_rca_workflow(
            request=TraceRCARequest(trace_id=trace_id, project_name=project_name),
            engine=TraceRCAEngine(inspection_api=api),
            artifacts_root=artifacts_root,
            writeback_client=_FakeEvaluationClient(),
        )
        rca_run_ids.append(run_record.run_id)
        output_path = Path(run_record.output_ref.artifact_path or "")
        output_payload = json.loads(output_path.read_text(encoding="utf-8"))
        if str(output_payload.get("primary_label") or "") == expected_labels[trace_id]:
            rca_rlm_correct += 1

    denominator = len(trace_ids) if trace_ids else 1
    rca_baseline_accuracy = rca_baseline_correct / denominator
    rca_rlm_accuracy = rca_rlm_correct / denominator

    compliance_baseline_correct = 0
    compliance_rlm_correct = 0
    compliance_run_ids: list[str] = []
    for trace_id in trace_ids:
        spans = api.list_spans(trace_id)
        expected_verdict = _expected_compliance_verdict(expected_labels[trace_id])
        baseline_verdict = _baseline_compliance_verdict(spans)
        if baseline_verdict == expected_verdict:
            compliance_baseline_correct += 1

        _, run_record = run_policy_compliance_workflow(
            request=PolicyComplianceRequest(
                trace_id=trace_id,
                project_name=project_name,
                controls_version=controls_version,
            ),
            engine=PolicyComplianceEngine(inspection_api=api),
            artifacts_root=artifacts_root,
            writeback_client=_FakeAnnotationClient(),
        )
        compliance_run_ids.append(run_record.run_id)
        output_path = Path(run_record.output_ref.artifact_path or "")
        output_payload = json.loads(output_path.read_text(encoding="utf-8"))
        if str(output_payload.get("overall_verdict") or "") == expected_verdict:
            compliance_rlm_correct += 1

    compliance_baseline_accuracy = compliance_baseline_correct / denominator
    compliance_rlm_accuracy = compliance_rlm_correct / denominator

    traces = api.list_traces(project_name)
    traces_by_id = {
        str(trace.get("trace_id") or ""): trace
        for trace in traces
        if str(trace.get("trace_id") or "")
    }
    benchmark_trace_ids = [trace_id for trace_id in trace_ids if trace_id in traces_by_id]
    if not benchmark_trace_ids:
        benchmark_trace_ids = sorted(traces_by_id.keys())
    benchmark_traces = [traces_by_id[trace_id] for trace_id in benchmark_trace_ids if trace_id in traces_by_id]
    if benchmark_traces:
        start_time = min(str(trace.get("start_time") or "") for trace in benchmark_traces)
        end_time = max(str(trace.get("end_time") or "") for trace in benchmark_traces)
    else:
        start_time = "2026-02-10T00:00:00Z"
        end_time = "2026-02-10T01:00:00Z"
    top_k = max(1, min(3, len(benchmark_trace_ids)))
    trace_latency_map = {
        trace_id: float((traces_by_id.get(trace_id) or {}).get("latency_ms") or 0.0)
        for trace_id in benchmark_trace_ids
    }
    incident_profiles = _incident_profiles(
        api=api,
        trace_ids=benchmark_trace_ids,
        trace_latency_map=trace_latency_map,
    )
    expected_incident = _incident_expected_trace_ids(
        profiles=incident_profiles,
        expected_labels=expected_labels,
        top_k=top_k,
    )
    baseline_incident = _incident_baseline_trace_ids(
        profiles=incident_profiles,
        top_k=top_k,
    )
    baseline_overlap = _overlap_at_k(expected=expected_incident, predicted=baseline_incident, k=top_k)
    baseline_diagnostics = _incident_diagnostics(
        expected=expected_incident,
        predicted=baseline_incident,
        k=top_k,
    )

    incident_report, incident_run_record = run_incident_dossier_workflow(
        request=IncidentDossierRequest(
            project_name=project_name,
            time_window_start=start_time,
            time_window_end=end_time,
            trace_ids_override=benchmark_trace_ids,
        ),
        engine=IncidentDossierEngine(
            inspection_api=api,
            max_representatives=top_k,
            error_quota=top_k,
            latency_quota=0,
            cluster_quota=0,
        ),
        artifacts_root=artifacts_root,
        writeback_client=_FakeAnnotationClient(),
    )
    incident_trace_ids = [item.trace_id for item in incident_report.representative_traces]
    rlm_overlap = _overlap_at_k(expected=expected_incident, predicted=incident_trace_ids, k=top_k)
    rlm_diagnostics = _incident_diagnostics(
        expected=expected_incident,
        predicted=incident_trace_ids,
        k=top_k,
    )

    rca_delta = rca_rlm_accuracy - rca_baseline_accuracy
    compliance_delta = compliance_rlm_accuracy - compliance_baseline_accuracy
    incident_delta = rlm_overlap - baseline_overlap
    thresholds = _resolved_delta_thresholds(delta_thresholds)
    gate_results = {
        "rca": {
            "metric": "delta.accuracy",
            "threshold": thresholds["rca"],
            "actual": rca_delta,
            "passed": rca_delta >= thresholds["rca"],
        },
        "compliance": {
            "metric": "delta.accuracy",
            "threshold": thresholds["compliance"],
            "actual": compliance_delta,
            "passed": compliance_delta >= thresholds["compliance"],
        },
        "incident": {
            "metric": "delta.overlap_at_k",
            "threshold": thresholds["incident"],
            "actual": incident_delta,
            "passed": incident_delta >= thresholds["incident"],
        },
    }

    return {
        "dataset": {
            "manifest_path": str(manifest_file),
            "spans_parquet_path": str(parquet_path),
            "dataset_hash": _file_sha256(parquet_path),
            "trace_count": len(trace_ids),
        },
        "capabilities": {
            "rca": {
                "sample_count": len(trace_ids),
                "baseline": {"accuracy": rca_baseline_accuracy},
                "rlm": {"accuracy": rca_rlm_accuracy},
                "delta": {"accuracy": rca_delta},
            },
            "compliance": {
                "sample_count": len(trace_ids),
                "baseline": {"accuracy": compliance_baseline_accuracy},
                "rlm": {"accuracy": compliance_rlm_accuracy},
                "delta": {"accuracy": compliance_delta},
            },
            "incident": {
                "sample_count": len(benchmark_trace_ids),
                "baseline": {
                    "overlap_at_k": baseline_overlap,
                    "selected_trace_ids": baseline_incident,
                },
                "rlm": {
                    "overlap_at_k": rlm_overlap,
                    "selected_trace_ids": incident_trace_ids,
                },
                "delta": {"overlap_at_k": incident_delta},
                "diagnostics": {
                    "k": top_k,
                    "baseline": baseline_diagnostics,
                    "rlm": rlm_diagnostics,
                },
            },
        },
        "gates": {
            "thresholds": thresholds,
            "results": gate_results,
            "all_passed": all(result["passed"] for result in gate_results.values()),
        },
        "run_artifacts": {
            "rca": rca_run_ids,
            "compliance": compliance_run_ids,
            "incident": [incident_run_record.run_id],
        },
    }
