# ABOUTME: Evaluates Trace RCA run artifacts against a manifest and computes dataset-level quality metrics.
# ABOUTME: Produces a human-readable summary plus a persisted JSON report for milestone acceptance gates.

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


RCA_LABELS = (
    "tool_failure",
    "retrieval_failure",
    "instruction_failure",
    "upstream_dependency_failure",
    "data_schema_mismatch",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _utc_now_rfc3339() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_rfc3339(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(path: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object.")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Manifest must include a list field named 'cases'.")
    return payload


def _find_output_payload(run_record: dict[str, Any], run_record_path: Path) -> dict[str, Any] | None:
    output_ref = run_record.get("output_ref")
    candidate_paths: list[Path] = []
    if isinstance(output_ref, dict):
        artifact_path = str(output_ref.get("artifact_path") or "").strip()
        if artifact_path:
            artifact = Path(artifact_path)
            if artifact.is_absolute():
                candidate_paths.append(artifact)
            else:
                candidate_paths.append(Path.cwd() / artifact)
                candidate_paths.append(run_record_path.parent / artifact)
    candidate_paths.append(run_record_path.parent / "output.json")

    for path in candidate_paths:
        if not path.exists() or not path.is_file():
            continue
        payload = _load_json(path)
        if isinstance(payload, dict):
            return payload
    return None


def _select_latest_runs_by_trace(runs_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(runs_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    selected: dict[str, dict[str, Any]] = {}
    run_record_paths = sorted(root.rglob("run_record.json"))
    for run_record_path in run_record_paths:
        run_record_payload = _load_json(run_record_path)
        if not isinstance(run_record_payload, dict):
            continue
        input_ref = run_record_payload.get("input_ref")
        trace_ids = input_ref.get("trace_ids") if isinstance(input_ref, dict) else None
        trace_id = ""
        if isinstance(trace_ids, list) and trace_ids:
            trace_id = str(trace_ids[0] or "").strip()
        output_payload = _find_output_payload(run_record_payload, run_record_path)
        if not trace_id and isinstance(output_payload, dict):
            trace_id = str(output_payload.get("trace_id") or "").strip()
        if not trace_id:
            continue

        completed_at = _parse_rfc3339(str(run_record_payload.get("completed_at") or ""))
        started_at = _parse_rfc3339(str(run_record_payload.get("started_at") or ""))
        run_id = str(run_record_payload.get("run_id") or run_record_path.parent.name)
        candidate = {
            "run_id": run_id,
            "trace_id": trace_id,
            "status": str(run_record_payload.get("status") or ""),
            "completed_at": completed_at,
            "started_at": started_at,
            "run_record": run_record_payload,
            "output": output_payload or {},
        }

        existing = selected.get(trace_id)
        if existing is None:
            selected[trace_id] = candidate
            continue
        existing_completed = existing.get("completed_at")
        existing_started = existing.get("started_at")
        existing_run_id = str(existing.get("run_id") or "")
        candidate_key = (
            completed_at or datetime.min.replace(tzinfo=timezone.utc),
            started_at or datetime.min.replace(tzinfo=timezone.utc),
            run_id,
        )
        existing_key = (
            existing_completed or datetime.min.replace(tzinfo=timezone.utc),
            existing_started or datetime.min.replace(tzinfo=timezone.utc),
            existing_run_id,
        )
        if candidate_key >= existing_key:
            selected[trace_id] = candidate
    return selected


def evaluate_rca_runs(
    *,
    manifest_path: str | Path,
    runs_dir: str | Path,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    dataset_id = str(manifest.get("dataset_id") or "")
    run_lookup = _select_latest_runs_by_trace(runs_dir)

    cases: list[dict[str, Any]] = []
    total_cases = 0
    correct_count = 0
    missing_run_count = 0
    confidence_correct: list[float] = []
    confidence_incorrect: list[float] = []
    evidence_correct: list[float] = []
    evidence_incorrect: list[float] = []
    costs: list[float] = []
    wall_times_sec: list[float] = []
    tokens_total: list[float] = []
    iteration_utilization_pct: list[float] = []
    tool_call_utilization_pct: list[float] = []
    model_names: set[str] = set()

    expected_by_label: dict[str, int] = {label: 0 for label in RCA_LABELS}
    predicted_by_label: dict[str, int] = {label: 0 for label in RCA_LABELS}
    true_positive_by_label: dict[str, int] = {label: 0 for label in RCA_LABELS}

    manifest_cases = manifest.get("cases") or []
    for case in manifest_cases:
        if not isinstance(case, dict):
            continue
        trace_id = str(case.get("trace_id") or "").strip()
        expected_label = str(case.get("expected_label") or "").strip()
        if not trace_id or not expected_label:
            continue
        total_cases += 1

        selected_run = run_lookup.get(trace_id)
        if selected_run is None:
            missing_run_count += 1
            evidence_incorrect.append(0.0)
            if expected_label in expected_by_label:
                expected_by_label[expected_label] += 1
            cases.append(
                {
                    "trace_id": trace_id,
                    "expected_label": expected_label,
                    "predicted_label": None,
                    "correct": False,
                    "run_id": None,
                    "run_status": "missing",
                }
            )
            continue

        run_id = str(selected_run.get("run_id") or "")
        run_status = str(selected_run.get("status") or "")
        run_record = selected_run.get("run_record")
        output_payload = selected_run.get("output") if isinstance(selected_run.get("output"), dict) else {}
        predicted_label = str(output_payload.get("primary_label") or "").strip()
        confidence = output_payload.get("confidence")
        confidence_value = _safe_float(confidence, default=0.0) if isinstance(confidence, (int, float)) else None
        evidence_refs_raw = output_payload.get("evidence_refs")
        evidence_count = len(evidence_refs_raw) if isinstance(evidence_refs_raw, list) else 0
        is_correct = predicted_label == expected_label
        if is_correct:
            correct_count += 1
            evidence_correct.append(float(evidence_count))
            if confidence_value is not None:
                confidence_correct.append(float(confidence_value))
        else:
            evidence_incorrect.append(float(evidence_count))
            if confidence_value is not None:
                confidence_incorrect.append(float(confidence_value))

        if expected_label in expected_by_label:
            expected_by_label[expected_label] += 1
        if predicted_label in predicted_by_label:
            predicted_by_label[predicted_label] += 1
        if is_correct and predicted_label in true_positive_by_label:
            true_positive_by_label[predicted_label] += 1

        runtime_ref = run_record.get("runtime_ref") if isinstance(run_record, dict) else None
        usage = runtime_ref.get("usage") if isinstance(runtime_ref, dict) else None
        budget = runtime_ref.get("budget") if isinstance(runtime_ref, dict) else None
        model_name = str(runtime_ref.get("model_name") or "").strip() if isinstance(runtime_ref, dict) else ""
        if model_name:
            model_names.add(model_name)

        cost = _safe_float(usage.get("cost_usd") if isinstance(usage, dict) else 0.0)
        tokens_in = _safe_int(usage.get("tokens_in") if isinstance(usage, dict) else 0)
        tokens_out = _safe_int(usage.get("tokens_out") if isinstance(usage, dict) else 0)
        iterations = _safe_int(usage.get("iterations") if isinstance(usage, dict) else 0)
        tool_calls = _safe_int(usage.get("tool_calls") if isinstance(usage, dict) else 0)
        max_iterations = _safe_int(budget.get("max_iterations") if isinstance(budget, dict) else 0)
        max_tool_calls = _safe_int(budget.get("max_tool_calls") if isinstance(budget, dict) else 0)
        costs.append(cost)
        tokens_total.append(float(tokens_in + tokens_out))

        started_at = selected_run.get("started_at")
        completed_at = selected_run.get("completed_at")
        if isinstance(started_at, datetime) and isinstance(completed_at, datetime):
            wall_times_sec.append(max(0.0, (completed_at - started_at).total_seconds()))
        else:
            wall_times_sec.append(0.0)

        if max_iterations > 0:
            iteration_utilization_pct.append((float(iterations) / float(max_iterations)) * 100.0)
        if max_tool_calls > 0:
            tool_call_utilization_pct.append((float(tool_calls) / float(max_tool_calls)) * 100.0)

        cases.append(
            {
                "trace_id": trace_id,
                "expected_label": expected_label,
                "predicted_label": predicted_label or None,
                "correct": bool(is_correct),
                "run_id": run_id or None,
                "run_status": run_status,
                "confidence": confidence_value,
                "evidence_refs_count": evidence_count,
            }
        )

    denominator = total_cases if total_cases > 0 else 1
    accuracy = float(correct_count) / float(denominator)

    per_label: dict[str, dict[str, Any]] = {}
    for label in RCA_LABELS:
        tp = int(true_positive_by_label[label])
        fp = int(predicted_by_label[label] - tp)
        fn = int(expected_by_label[label] - tp)
        precision = (float(tp) / float(tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (float(tp) / float(tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_label[label] = {
            "support": int(expected_by_label[label]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    model_name = "mixed"
    if len(model_names) == 1:
        model_name = sorted(model_names)[0]
    elif len(model_names) == 0:
        model_name = "unknown"

    return {
        "generated_at": _utc_now_rfc3339(),
        "dataset": {
            "dataset_id": dataset_id or None,
            "manifest_path": str(Path(manifest_path)),
            "runs_dir": str(Path(runs_dir)),
            "trace_case_count": int(total_cases),
            "missing_run_count": int(missing_run_count),
        },
        "model": {"name": model_name},
        "metrics": {
            "top1_accuracy": {
                "correct": int(correct_count),
                "total": int(total_cases),
                "score": accuracy,
            },
            "per_label": per_label,
            "confidence": {
                "avg_correct": _average(confidence_correct),
                "avg_incorrect": _average(confidence_incorrect),
            },
            "evidence_quality": {
                "avg_correct_evidence_refs": _average(evidence_correct),
                "avg_incorrect_evidence_refs": _average(evidence_incorrect),
            },
            "runtime": {
                "cost_total_usd": sum(costs),
                "cost_avg_usd": _average(costs),
                "wall_time_total_sec": sum(wall_times_sec),
                "wall_time_avg_sec": _average(wall_times_sec),
                "tokens_total": int(sum(tokens_total)),
                "tokens_avg": _average(tokens_total),
                "avg_iteration_utilization_pct": _average(iteration_utilization_pct),
                "avg_tool_call_utilization_pct": _average(tool_call_utilization_pct),
            },
        },
        "cases": cases,
    }


def _format_score(value: float) -> str:
    return f"{value:.2f}"


def _format_pct(value: float) -> str:
    return f"{(value * 100.0):.1f}%"


def format_evaluation_report(report: dict[str, Any]) -> str:
    dataset = report.get("dataset") or {}
    metrics = report.get("metrics") or {}
    top1 = metrics.get("top1_accuracy") or {}
    per_label = metrics.get("per_label") or {}
    runtime = metrics.get("runtime") or {}
    confidence = metrics.get("confidence") or {}
    evidence_quality = metrics.get("evidence_quality") or {}
    dataset_id = str(dataset.get("dataset_id") or "unknown_dataset")
    total = _safe_int(top1.get("total"), default=0)
    correct = _safe_int(top1.get("correct"), default=0)
    score = _safe_float(top1.get("score"), default=0.0)
    model_name = str((report.get("model") or {}).get("name") or "unknown")

    lines = [
        "RLM-RCA Evaluation Report",
        f"Dataset: {dataset_id} ({total} cases)",
        f"Model: {model_name}",
        "",
        f"Top-1 Accuracy: {correct}/{total} ({_format_pct(score)})",
        "",
        "Per-Label Results:",
    ]
    for label in RCA_LABELS:
        row = per_label.get(label) or {}
        lines.append(
            "  "
            + f"{label:<26}: "
            + f"P={_format_score(_safe_float(row.get('precision')))}  "
            + f"R={_format_score(_safe_float(row.get('recall')))}  "
            + f"F1={_format_score(_safe_float(row.get('f1')))}  "
            + f"({_safe_int(row.get('support'))} cases)"
        )
    lines.extend(
        [
            "",
            f"Cost:  avg ${_safe_float(runtime.get('cost_avg_usd')):.4f}/run  total ${_safe_float(runtime.get('cost_total_usd')):.4f}",
            f"Time:  avg {_safe_float(runtime.get('wall_time_avg_sec')):.2f}s/run  total {_safe_float(runtime.get('wall_time_total_sec')):.2f}s",
            f"Tokens: avg {_safe_float(runtime.get('tokens_avg')):.1f}/run  total {_safe_int(runtime.get('tokens_total'))}",
            "",
            "Evidence Quality:",
            f"  Correct predictions:   avg {_safe_float(evidence_quality.get('avg_correct_evidence_refs')):.2f} evidence_refs",
            f"  Incorrect predictions: avg {_safe_float(evidence_quality.get('avg_incorrect_evidence_refs')):.2f} evidence_refs",
            "",
            "Confidence:",
            f"  Correct predictions:   avg {_safe_float(confidence.get('avg_correct')):.2f}",
            f"  Incorrect predictions: avg {_safe_float(confidence.get('avg_incorrect')):.2f}",
            "",
            "Budget Utilization:",
            f"  Iterations: avg {_safe_float(runtime.get('avg_iteration_utilization_pct')):.1f}%",
            f"  Tool calls: avg {_safe_float(runtime.get('avg_tool_call_utilization_pct')):.1f}%",
        ]
    )
    return "\n".join(lines)


def evaluate_rca_runs_comparative(
    *,
    manifest_path: str | Path,
    scaffold_runs: dict[str, str | Path],
) -> dict[str, Any]:
    per_scaffold: dict[str, dict[str, Any]] = {}
    for scaffold_name, runs_dir in scaffold_runs.items():
        per_scaffold[scaffold_name] = evaluate_rca_runs(
            manifest_path=manifest_path,
            runs_dir=runs_dir,
        )

    heuristic_report = per_scaffold.get("heuristic")
    heuristic_accuracy = 0.0
    heuristic_cost = 0.0
    if heuristic_report is not None:
        heuristic_accuracy = _safe_float(
            (heuristic_report.get("metrics") or {}).get("top1_accuracy", {}).get("score")
        )
        heuristic_cost = _safe_float(
            (heuristic_report.get("metrics") or {}).get("runtime", {}).get("cost_total_usd")
        )

    scaffold_summaries: dict[str, dict[str, Any]] = {}
    for scaffold_name, report in per_scaffold.items():
        metrics = report.get("metrics") or {}
        top1 = metrics.get("top1_accuracy") or {}
        runtime = metrics.get("runtime") or {}
        per_label = metrics.get("per_label") or {}
        accuracy = _safe_float(top1.get("score"))
        cost_total = _safe_float(runtime.get("cost_total_usd"))

        per_label_f1: dict[str, float] = {}
        for label in RCA_LABELS:
            label_data = per_label.get(label) or {}
            per_label_f1[label] = _safe_float(label_data.get("f1"))

        scaffold_summaries[scaffold_name] = {
            "accuracy": accuracy,
            "correct": _safe_int(top1.get("correct")),
            "total": _safe_int(top1.get("total")),
            "cost_total_usd": cost_total,
            "cost_avg_usd": _safe_float(runtime.get("cost_avg_usd")),
            "wall_time_total_sec": _safe_float(runtime.get("wall_time_total_sec")),
            "wall_time_avg_sec": _safe_float(runtime.get("wall_time_avg_sec")),
            "tokens_total": _safe_int(runtime.get("tokens_total")),
            "tokens_avg": _safe_float(runtime.get("tokens_avg")),
            "per_label_f1": per_label_f1,
            "delta_vs_heuristic": {
                "accuracy_gain": accuracy - heuristic_accuracy,
                "cost_delta_usd": cost_total - heuristic_cost,
            },
        }

    return {
        "generated_at": _utc_now_rfc3339(),
        "manifest_path": str(Path(manifest_path)),
        "scaffolds": scaffold_summaries,
        "per_scaffold_full": per_scaffold,
    }


def format_comparative_report(report: dict[str, Any]) -> str:
    scaffolds = report.get("scaffolds") or {}
    lines: list[str] = [
        "RLM-RCA Comparative Report",
        f"Manifest: {report.get('manifest_path', 'unknown')}",
        "",
    ]

    header_cols = ["Scaffold", "Accuracy", "Cost ($)", "Time (s)", "Tokens", "Acc. Gain"]
    col_widths = [max(len(h), 16) for h in header_cols]
    lines.append("  ".join(h.ljust(w) for h, w in zip(header_cols, col_widths)))
    lines.append("  ".join("-" * w for w in col_widths))

    for scaffold_name in sorted(scaffolds.keys()):
        data = scaffolds[scaffold_name]
        delta = data.get("delta_vs_heuristic") or {}
        accuracy_str = f"{_safe_int(data.get('correct'))}/{_safe_int(data.get('total'))} ({_format_pct(_safe_float(data.get('accuracy')))})"
        cost_str = f"${_safe_float(data.get('cost_total_usd')):.4f}"
        time_str = f"{_safe_float(data.get('wall_time_total_sec')):.1f}"
        tokens_str = str(_safe_int(data.get("tokens_total")))
        gain_str = f"{_safe_float(delta.get('accuracy_gain')):+.1%}"
        row = [scaffold_name, accuracy_str, cost_str, time_str, tokens_str, gain_str]
        lines.append("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

    lines.extend(["", "Per-Label F1 Breakdown:"])
    for scaffold_name in sorted(scaffolds.keys()):
        data = scaffolds[scaffold_name]
        per_label_f1 = data.get("per_label_f1") or {}
        f1_parts = [f"{label}={_format_score(_safe_float(per_label_f1.get(label)))}" for label in RCA_LABELS]
        lines.append(f"  {scaffold_name}: {', '.join(f1_parts)}")

    return "\n".join(lines)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RCA runs against manifest labels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output", default="artifacts/evaluation/eval_report.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_rca_runs(manifest_path=args.manifest, runs_dir=args.runs_dir)
    _write_report(Path(args.output), report)
    print(format_evaluation_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
