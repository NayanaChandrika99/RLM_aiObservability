# ABOUTME: Reprocesses existing TRAIL prediction files with deterministic joint-recall boosts and semantic checks.
# ABOUTME: Produces scorer-ready outputs and optional metrics without issuing new model or Agentica calls.

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .trail_agent import (
        apply_joint_recall_boost_to_prediction,
        apply_trajectory_action_correlation_to_prediction,
    )
    from .trail_experiment import _score_outputs
    from .trail_semantic_checks import enforce_semantic_faithfulness
except ImportError:
    from trail_agent import (
        apply_joint_recall_boost_to_prediction,
        apply_trajectory_action_correlation_to_prediction,
    )
    from trail_experiment import _score_outputs
    from trail_semantic_checks import enforce_semantic_faithfulness


_TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")


def _json_loads_relaxed(raw_text: str) -> Any:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        repaired = _TRAILING_COMMA_PATTERN.sub(r"\1", raw_text)
        return json.loads(repaired)


def _load_json_file(path: Path) -> Any:
    return _json_loads_relaxed(path.read_text(encoding="utf-8"))


def reprocess_outputs(
    *,
    input_dir: Path,
    output_dir: Path,
    trail_data_dir: Path,
    split: str,
    semantic_checks: str,
    trajectory_action_correlation: bool = True,
    joint_recall_boost: bool = True,
    gold_dir: Path | None = None,
) -> dict[str, Any]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    split_dir = trail_data_dir / split
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"TRAIL split directory does not exist: {split_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_errors = 0
    kept_errors = 0
    dropped_errors = 0
    repair_actions = {
        "location_repaired": 0,
        "evidence_repaired": 0,
        "impact_repaired": 0,
        "description_repaired": 0,
    }
    drop_reasons = {
        "invalid_error_shape": 0,
        "missing_category": 0,
        "unrepairable_location": 0,
        "unrepairable_evidence": 0,
    }
    file_reports: list[dict[str, Any]] = []
    processed_trace_files: list[Path] = []

    for prediction_path in sorted(input_dir.glob("*.json")):
        trace_path = split_dir / prediction_path.name
        if not trace_path.exists():
            continue

        try:
            trace_payload = _load_json_file(trace_path)
            prediction = _load_json_file(prediction_path)
        except Exception as exc:
            file_reports.append(
                {
                    "trace_file": prediction_path.name,
                    "status": "failed_to_read_inputs",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:300],
                }
            )
            continue

        if not isinstance(prediction, dict):
            prediction = {}

        reprocessed = prediction
        if trajectory_action_correlation:
            reprocessed = apply_trajectory_action_correlation_to_prediction(trace_payload, reprocessed)
        if joint_recall_boost:
            reprocessed = apply_joint_recall_boost_to_prediction(reprocessed)

        reprocessed, semantic_report = enforce_semantic_faithfulness(
            trace_payload=trace_payload,
            prediction=reprocessed,
            mode=semantic_checks,
        )

        (output_dir / prediction_path.name).write_text(
            json.dumps(reprocessed, indent=2),
            encoding="utf-8",
        )
        processed_trace_files.append(trace_path)

        total_errors += int(semantic_report.get("total_errors", 0))
        kept_errors += int(semantic_report.get("kept_errors", 0))
        dropped_errors += int(semantic_report.get("dropped_errors", 0))
        for key in repair_actions:
            repair_actions[key] += int(semantic_report.get("repair_actions", {}).get(key, 0))
        for key in drop_reasons:
            drop_reasons[key] += int(semantic_report.get("drop_reasons", {}).get(key, 0))

        trace_report = dict(semantic_report)
        trace_report["trace_file"] = prediction_path.name
        trace_report["status"] = "ok"
        file_reports.append(trace_report)

    metrics = None
    if gold_dir is not None:
        metrics = _score_outputs(
            gold_dir=gold_dir,
            generated_dir=output_dir,
            trace_files=processed_trace_files,
        )

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "split": split,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "trajectory_action_correlation": trajectory_action_correlation,
        "joint_recall_boost": joint_recall_boost,
        "files_processed": len(processed_trace_files),
        "semantic": {
            "mode": semantic_checks,
            "totals": {
                "total_errors": total_errors,
                "kept_errors": kept_errors,
                "dropped_errors": dropped_errors,
                "grounded_evidence_rate": 1.0 if total_errors == 0 else kept_errors / total_errors,
            },
            "repair_actions": repair_actions,
            "drop_reasons": drop_reasons,
            "files": file_reports,
        },
        "metrics": metrics,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reprocess existing TRAIL outputs with deterministic joint-recall rules.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with existing output JSONs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write reprocessed JSONs.")
    parser.add_argument(
        "--trail-data-dir",
        type=Path,
        required=True,
        help="TRAIL data root containing split folders (for semantic checks).",
    )
    parser.add_argument("--split", type=str, default="GAIA", help="TRAIL split name.")
    parser.add_argument(
        "--semantic-checks",
        type=str,
        default="strict",
        choices=["strict", "off"],
        help="Semantic faithfulness mode.",
    )
    parser.add_argument(
        "--trajectory-action-correlation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply deterministic trajectory-action correlation before recall boosting.",
    )
    parser.add_argument(
        "--joint-recall-boost",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply deterministic joint-recall category expansion rules.",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=None,
        help="Optional gold annotation directory to compute metrics.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional summary JSON path (defaults to <output-dir>/reprocess_summary.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = reprocess_outputs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        trail_data_dir=args.trail_data_dir,
        split=args.split,
        semantic_checks=args.semantic_checks,
        trajectory_action_correlation=args.trajectory_action_correlation,
        joint_recall_boost=args.joint_recall_boost,
        gold_dir=args.gold_dir,
    )

    summary_path = args.summary_out or (args.output_dir / "reprocess_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote reprocessed outputs to: {args.output_dir}")
    print(f"Wrote summary report to: {summary_path}")
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        print(
            "Metrics: "
            f"F1={metrics.get('weighted_f1', 0):.4f}, "
            f"Location={metrics.get('location_accuracy', 0):.4f}, "
            f"Joint={metrics.get('joint_accuracy', 0):.4f}"
        )


if __name__ == "__main__":
    main()
