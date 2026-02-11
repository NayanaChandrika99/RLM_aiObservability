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


def _load_fault_profiles(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    profiles: dict[str, str] = {}
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        trace_id = str(case.get("trace_id") or "")
        if not trace_id:
            continue
        profile = str(case.get("fault_profile") or "").strip()
        if not profile:
            label = str(case.get("expected_label") or "").strip()
            profile = f"profile_{label}" if label else "profile_unknown"
        profiles[trace_id] = profile
    return profiles


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
        return "needs_review"
    return "needs_review"


def _compliance_profile_coverage(
    *,
    api: ParquetInspectionAPI,
    trace_ids: list[str],
    expected_labels: dict[str, str],
    fault_profiles: dict[str, str],
    controls_version: str,
) -> dict[str, Any]:
    profile_rows: dict[str, dict[str, Any]] = {}
    for trace_id in trace_ids:
        if not trace_id:
            continue
        spans = api.list_spans(trace_id)
        app_type = PolicyComplianceEngine._infer_app_type(spans)
        tools_used = PolicyComplianceEngine._infer_tools_used(spans)
        controls = api.list_controls(
            controls_version=controls_version,
            app_type=app_type,
            tools_used=tools_used or None,
            data_domains=None,
        ) or []
        control_ids = sorted(
            {
                str(control.get("control_id") or "")
                for control in controls
                if str(control.get("control_id") or "")
            }
        )
        profile = str(fault_profiles.get(trace_id) or "").strip()
        if not profile:
            label = str(expected_labels.get(trace_id) or "").strip()
            profile = f"profile_{label}" if label else "profile_unknown"
        row = profile_rows.setdefault(
            profile,
            {
                "profile": profile,
                "sample_count": 0,
                "expected_labels": set(),
                "control_ids": set(),
                "trace_ids": [],
            },
        )
        row["sample_count"] = int(row["sample_count"]) + 1
        row["trace_ids"].append(trace_id)
        label = str(expected_labels.get(trace_id) or "").strip()
        if label:
            row["expected_labels"].add(label)
        row["control_ids"].update(control_ids)

    normalized_profiles: dict[str, dict[str, Any]] = {}
    uncovered_profiles: list[str] = []
    for profile in sorted(profile_rows.keys()):
        row = profile_rows[profile]
        control_ids = sorted(str(item) for item in row["control_ids"] if str(item))
        expected_labels_sorted = sorted(
            str(item) for item in row["expected_labels"] if str(item)
        )
        trace_ids_sorted = sorted(str(item) for item in row["trace_ids"] if str(item))
        covered = bool(control_ids)
        normalized_profiles[profile] = {
            "profile": profile,
            "sample_count": int(row["sample_count"]),
            "expected_labels": expected_labels_sorted,
            "control_ids": control_ids,
            "trace_ids": trace_ids_sorted,
            "covered": covered,
        }
        if not covered:
            uncovered_profiles.append(profile)
    return {
        "profiles": normalized_profiles,
        "uncovered_profiles": uncovered_profiles,
        "all_profiles_covered": len(uncovered_profiles) == 0,
    }


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


def _incident_selector_contract(top_k: int) -> dict[str, int]:
    return {
        "max_representatives": top_k,
        "error_quota": top_k,
        "latency_quota": 0,
        "cluster_quota": 0,
    }


def _incident_expected_profiles(
    *,
    profiles: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    selector_contract = _incident_selector_contract(top_k)
    selector = IncidentDossierEngine(
        inspection_api=None,
        max_representatives=selector_contract["max_representatives"],
        error_quota=selector_contract["error_quota"],
        latency_quota=selector_contract["latency_quota"],
        cluster_quota=selector_contract["cluster_quota"],
    )
    expected = selector._choose_representative_profiles(
        [dict(profile) for profile in profiles],
        override_trace_ids=None,
    )
    return expected[:top_k]


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


def _signature_key(signature: tuple[str, str, str] | None) -> str:
    if not isinstance(signature, tuple) or len(signature) != 3:
        return "unknown|unknown|unknown"
    return "|".join(str(item) for item in signature)


def _parse_bucket_from_why_selected(text: str) -> str:
    value = str(text or "").strip()
    if "_bucket" not in value:
        return "unknown"
    bucket = value.split("_bucket", 1)[0].strip()
    return bucket or "unknown"


def _incident_representative_buckets(representative_traces: list[Any]) -> dict[str, str]:
    buckets: dict[str, str] = {}
    for representative in representative_traces:
        trace_id = str(getattr(representative, "trace_id", "") or "")
        why_selected = str(getattr(representative, "why_selected", "") or "")
        if not trace_id:
            continue
        buckets[trace_id] = _parse_bucket_from_why_selected(why_selected)
    return buckets


def _incident_diagnostics(
    *,
    expected_profiles: list[dict[str, Any]],
    predicted_ids: list[str],
    k: int,
    profile_by_trace_id: dict[str, dict[str, Any]],
    predicted_buckets: dict[str, str] | None = None,
) -> dict[str, Any]:
    expected_at_k_profiles = expected_profiles[:k]
    expected_at_k = [str(profile.get("trace_id") or "") for profile in expected_at_k_profiles if str(profile.get("trace_id") or "")]
    predicted_at_k = [str(trace_id) for trace_id in predicted_ids[:k] if str(trace_id)]
    predicted_set = set(predicted_at_k)
    intersection = [trace_id for trace_id in expected_at_k if trace_id in predicted_set]

    expected_profile_by_id = {
        str(profile.get("trace_id") or ""): profile
        for profile in expected_at_k_profiles
        if str(profile.get("trace_id") or "")
    }
    expected_bucket_by_id = {
        trace_id: str(profile.get("bucket") or "unknown")
        for trace_id, profile in expected_profile_by_id.items()
    }
    expected_signature_by_id = {
        trace_id: profile.get("signature") if isinstance(profile.get("signature"), tuple) else None
        for trace_id, profile in expected_profile_by_id.items()
    }

    predicted_bucket_by_id: dict[str, str] = {}
    for trace_id in predicted_at_k:
        if predicted_buckets and trace_id in predicted_buckets:
            predicted_bucket_by_id[trace_id] = str(predicted_buckets[trace_id] or "unknown")
            continue
        predicted_bucket_by_id[trace_id] = str(
            (profile_by_trace_id.get(trace_id) or {}).get("bucket") or "unknown"
        )

    predicted_signature_by_id: dict[str, tuple[str, str, str] | None] = {}
    for trace_id in predicted_at_k:
        signature = (profile_by_trace_id.get(trace_id) or {}).get("signature")
        predicted_signature_by_id[trace_id] = signature if isinstance(signature, tuple) else None

    rows: list[dict[str, Any]] = []
    for trace_id in sorted(set(expected_at_k).union(predicted_at_k)):
        in_expected = trace_id in expected_profile_by_id
        in_predicted = trace_id in predicted_set
        mismatch: list[str] = []
        if in_expected and not in_predicted:
            mismatch.append("missing_from_predicted")
        if in_predicted and not in_expected:
            mismatch.append("unexpected_in_predicted")
        if in_expected and in_predicted:
            expected_bucket = expected_bucket_by_id.get(trace_id, "unknown")
            predicted_bucket = predicted_bucket_by_id.get(trace_id, "unknown")
            if expected_bucket != predicted_bucket:
                mismatch.append("bucket_mismatch")
        rows.append(
            {
                "trace_id": trace_id,
                "expected_rank": (expected_at_k.index(trace_id) + 1) if trace_id in expected_at_k else None,
                "predicted_rank": (predicted_at_k.index(trace_id) + 1) if trace_id in predicted_at_k else None,
                "in_intersection": trace_id in intersection,
                "expected_bucket": expected_bucket_by_id.get(trace_id),
                "predicted_bucket": predicted_bucket_by_id.get(trace_id),
                "expected_signature": _signature_key(expected_signature_by_id.get(trace_id)),
                "predicted_signature": _signature_key(predicted_signature_by_id.get(trace_id)),
                "mismatch_reasons": mismatch,
            }
        )

    missing_by_bucket: dict[str, int] = {}
    unexpected_by_bucket: dict[str, int] = {}
    for trace_id in expected_at_k:
        if trace_id in predicted_set:
            continue
        bucket = expected_bucket_by_id.get(trace_id, "unknown")
        missing_by_bucket[bucket] = missing_by_bucket.get(bucket, 0) + 1
    for trace_id in predicted_at_k:
        if trace_id in expected_profile_by_id:
            continue
        bucket = predicted_bucket_by_id.get(trace_id, "unknown")
        unexpected_by_bucket[bucket] = unexpected_by_bucket.get(bucket, 0) + 1

    expected_signatures: dict[str, list[str]] = {}
    predicted_signatures: dict[str, list[str]] = {}
    for trace_id in expected_at_k:
        signature_key = _signature_key(expected_signature_by_id.get(trace_id))
        expected_signatures.setdefault(signature_key, []).append(trace_id)
    for trace_id in predicted_at_k:
        signature_key = _signature_key(predicted_signature_by_id.get(trace_id))
        predicted_signatures.setdefault(signature_key, []).append(trace_id)

    missing_signatures = sorted(
        signature
        for signature in expected_signatures
        if signature not in predicted_signatures
    )
    unexpected_signatures = sorted(
        signature
        for signature in predicted_signatures
        if signature not in expected_signatures
    )
    substituted_signatures: list[dict[str, Any]] = []
    for signature in sorted(set(expected_signatures).intersection(predicted_signatures)):
        expected_ids = sorted(expected_signatures[signature])
        predicted_ids_for_signature = sorted(predicted_signatures[signature])
        if expected_ids != predicted_ids_for_signature:
            substituted_signatures.append(
                {
                    "signature": signature,
                    "expected_trace_ids": expected_ids,
                    "predicted_trace_ids": predicted_ids_for_signature,
                }
            )

    return {
        "k": k,
        "expected_ids": expected_at_k,
        "predicted_ids": predicted_at_k,
        "intersection": intersection,
        "expected_profiles": [
            {
                "trace_id": str(profile.get("trace_id") or ""),
                "bucket": str(profile.get("bucket") or ""),
                "error_spans": int(profile.get("error_spans") or 0),
                "latency_ms": float(profile.get("latency_ms") or 0.0),
                "signature": _signature_key(
                    profile.get("signature") if isinstance(profile.get("signature"), tuple) else None
                ),
            }
            for profile in expected_at_k_profiles
        ],
        "rows": rows,
        "mismatch_reasons": {
            "by_bucket": {
                "missing_from_predicted": missing_by_bucket,
                "unexpected_in_predicted": unexpected_by_bucket,
            },
            "by_signature": {
                "missing_signatures": missing_signatures,
                "unexpected_signatures": unexpected_signatures,
                "substituted_signatures": substituted_signatures,
            },
        },
    }


def _resolved_delta_thresholds(delta_thresholds: dict[str, float] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_DELTA_THRESHOLDS)
    for capability, value in (delta_thresholds or {}).items():
        if capability not in resolved:
            continue
        resolved[capability] = float(value)
    return resolved


def _incident_gate_result(
    *,
    threshold: float,
    baseline_overlap: float,
    rlm_overlap: float,
) -> dict[str, Any]:
    max_headroom = max(0.0, 1.0 - baseline_overlap)
    effective_threshold = min(float(threshold), max_headroom)
    delta = rlm_overlap - baseline_overlap
    non_regression_ok = rlm_overlap + 1e-12 >= baseline_overlap
    delta_ok = delta + 1e-12 >= effective_threshold
    return {
        "metric": "delta.overlap_at_k",
        "threshold": float(threshold),
        "effective_threshold": effective_threshold,
        "actual": delta,
        "baseline_overlap_at_k": baseline_overlap,
        "rlm_overlap_at_k": rlm_overlap,
        "headroom": max_headroom,
        "non_regression_ok": non_regression_ok,
        "passed": non_regression_ok and delta_ok,
    }


def _rca_per_label_diagnostics(
    *,
    trace_ids: list[str],
    expected_labels: dict[str, str],
    baseline_predictions: dict[str, str],
    rlm_predictions: dict[str, str],
) -> dict[str, dict[str, Any]]:
    labels = sorted({str(expected_labels.get(trace_id) or "") for trace_id in trace_ids if str(expected_labels.get(trace_id) or "")})
    rows: dict[str, dict[str, Any]] = {}
    for label in labels:
        support = 0
        baseline_correct = 0
        rlm_correct = 0
        for trace_id in trace_ids:
            expected = str(expected_labels.get(trace_id) or "")
            if expected != label:
                continue
            support += 1
            if str(baseline_predictions.get(trace_id) or "") == expected:
                baseline_correct += 1
            if str(rlm_predictions.get(trace_id) or "") == expected:
                rlm_correct += 1
        if support > 0:
            baseline_accuracy = baseline_correct / support
            rlm_accuracy = rlm_correct / support
        else:
            baseline_accuracy = 0.0
            rlm_accuracy = 0.0
        rows[label] = {
            "label": label,
            "support": support,
            "baseline_correct": baseline_correct,
            "rlm_correct": rlm_correct,
            "baseline_accuracy": baseline_accuracy,
            "rlm_accuracy": rlm_accuracy,
            "delta_accuracy": rlm_accuracy - baseline_accuracy,
        }
    return rows


def _rca_threshold_calibration(
    *,
    trace_ids: list[str],
    expected_labels: dict[str, str],
    rlm_predictions: dict[str, str],
    rca_delta: float,
    threshold: float,
) -> dict[str, Any]:
    threshold_miss = rca_delta < threshold
    misses_by_label: dict[str, int] = {}
    total_rlm_misses = 0
    for trace_id in trace_ids:
        expected = str(expected_labels.get(trace_id) or "")
        predicted = str(rlm_predictions.get(trace_id) or "")
        if expected and predicted != expected:
            total_rlm_misses += 1
            misses_by_label[expected] = misses_by_label.get(expected, 0) + 1
    top_label = None
    top_count = 0
    if misses_by_label:
        top_label, top_count = sorted(
            misses_by_label.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
    top_share = (top_count / total_rlm_misses) if total_rlm_misses > 0 else 0.0
    return {
        "threshold": threshold,
        "threshold_miss": threshold_miss,
        "delta_accuracy": rca_delta,
        "total_rlm_misses": total_rlm_misses,
        "misses_by_expected_label": misses_by_label,
        "miss_concentration": {
            "top_label": top_label,
            "top_label_count": top_count,
            "top_label_share": top_share,
            "concentrated_in_label_family": top_share >= 0.6 and total_rlm_misses > 0,
        },
    }


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
    dataset_hash = _file_sha256(parquet_path)
    api = ParquetInspectionAPI(
        parquet_path=parquet_path,
        project_name=project_name,
        controls_dir=controls_dir,
        snapshots_dir=snapshots_dir,
    )
    api.attach_manifest_trace_ids(manifest_path=manifest_file)
    expected_labels = _load_expected_labels(manifest_file)
    fault_profiles = _load_fault_profiles(manifest_file)
    trace_ids = sorted(expected_labels.keys())

    rca_baseline_correct = 0
    rca_rlm_correct = 0
    rca_baseline_predictions: dict[str, str] = {}
    rca_rlm_predictions: dict[str, str] = {}
    rca_run_ids: list[str] = []
    for trace_id in trace_ids:
        spans = api.list_spans(trace_id)
        baseline_label = _baseline_rca_label(spans)
        rca_baseline_predictions[trace_id] = baseline_label
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
        rlm_label = str(output_payload.get("primary_label") or "")
        rca_rlm_predictions[trace_id] = rlm_label
        if rlm_label == expected_labels[trace_id]:
            rca_rlm_correct += 1

    denominator = len(trace_ids) if trace_ids else 1
    rca_baseline_accuracy = rca_baseline_correct / denominator
    rca_rlm_accuracy = rca_rlm_correct / denominator

    compliance_baseline_correct = 0
    compliance_rlm_correct = 0
    compliance_run_ids: list[str] = []
    compliance_profile_coverage = _compliance_profile_coverage(
        api=api,
        trace_ids=trace_ids,
        expected_labels=expected_labels,
        fault_profiles=fault_profiles,
        controls_version=controls_version,
    )
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
    expected_incident_profiles = _incident_expected_profiles(
        profiles=incident_profiles,
        top_k=top_k,
    )
    expected_incident = [
        str(profile.get("trace_id") or "")
        for profile in expected_incident_profiles
        if str(profile.get("trace_id") or "")
    ]
    incident_profile_by_trace = {
        str(profile.get("trace_id") or ""): dict(profile)
        for profile in incident_profiles
        if str(profile.get("trace_id") or "")
    }
    baseline_incident = _incident_baseline_trace_ids(
        profiles=incident_profiles,
        top_k=top_k,
    )
    baseline_overlap = _overlap_at_k(expected=expected_incident, predicted=baseline_incident, k=top_k)
    baseline_diagnostics = _incident_diagnostics(
        expected_profiles=expected_incident_profiles,
        predicted_ids=baseline_incident,
        k=top_k,
        profile_by_trace_id=incident_profile_by_trace,
        predicted_buckets=None,
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
    incident_predicted_buckets = _incident_representative_buckets(
        incident_report.representative_traces
    )
    rlm_overlap = _overlap_at_k(expected=expected_incident, predicted=incident_trace_ids, k=top_k)
    rlm_diagnostics = _incident_diagnostics(
        expected_profiles=expected_incident_profiles,
        predicted_ids=incident_trace_ids,
        k=top_k,
        profile_by_trace_id=incident_profile_by_trace,
        predicted_buckets=incident_predicted_buckets,
    )

    rca_delta = rca_rlm_accuracy - rca_baseline_accuracy
    compliance_delta = compliance_rlm_accuracy - compliance_baseline_accuracy
    incident_delta = rlm_overlap - baseline_overlap
    thresholds = _resolved_delta_thresholds(delta_thresholds)
    rca_per_label = _rca_per_label_diagnostics(
        trace_ids=trace_ids,
        expected_labels=expected_labels,
        baseline_predictions=rca_baseline_predictions,
        rlm_predictions=rca_rlm_predictions,
    )
    rca_threshold_calibration = _rca_threshold_calibration(
        trace_ids=trace_ids,
        expected_labels=expected_labels,
        rlm_predictions=rca_rlm_predictions,
        rca_delta=rca_delta,
        threshold=thresholds["rca"],
    )
    incident_gate = _incident_gate_result(
        threshold=thresholds["incident"],
        baseline_overlap=baseline_overlap,
        rlm_overlap=rlm_overlap,
    )
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
            "coverage_ok": bool(compliance_profile_coverage["all_profiles_covered"]),
            "uncovered_profiles": list(compliance_profile_coverage["uncovered_profiles"]),
            "passed": (
                compliance_delta >= thresholds["compliance"]
                and bool(compliance_profile_coverage["all_profiles_covered"])
            ),
        },
        "incident": incident_gate,
    }

    return {
        "dataset": {
            "manifest_path": str(manifest_file),
            "spans_parquet_path": str(parquet_path),
            "dataset_hash": dataset_hash,
            "trace_count": len(trace_ids),
        },
        "capabilities": {
            "rca": {
                "sample_count": len(trace_ids),
                "baseline": {"accuracy": rca_baseline_accuracy},
                "rlm": {"accuracy": rca_rlm_accuracy},
                "delta": {"accuracy": rca_delta},
                "diagnostics": {
                    "dataset_hash": dataset_hash,
                    "per_label": rca_per_label,
                    "threshold_calibration": rca_threshold_calibration,
                },
            },
            "compliance": {
                "sample_count": len(trace_ids),
                "baseline": {"accuracy": compliance_baseline_accuracy},
                "rlm": {"accuracy": compliance_rlm_accuracy},
                "delta": {"accuracy": compliance_delta},
                "diagnostics": {
                    "dataset_hash": dataset_hash,
                    "profile_coverage": compliance_profile_coverage,
                },
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
                    "selector_contract": _incident_selector_contract(top_k),
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
