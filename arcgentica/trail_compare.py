# ABOUTME: Compares TRAIL baseline and candidate runs using scorer metrics and semantic faithfulness diagnostics.
# ABOUTME: Writes a machine-readable comparison report with acceptance gates for Phase 11 milestone tracking.

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_PATTERNS = {
    "weighted_f1": re.compile(r"Weighted F1:\s*([0-9]*\.?[0-9]+)"),
    "location_accuracy": re.compile(r"Average Location Accuracy:\s*([0-9]*\.?[0-9]+)"),
    "joint_accuracy": re.compile(
        r"Average Location-Category Joint Accuracy:\s*([0-9]*\.?[0-9]+)"
    ),
}


def _parse_float(text: str, pattern: re.Pattern[str], label: str) -> float:
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not parse `{label}` from metrics file content.")
    return float(match.group(1))


def find_metrics_file(results_dir: Path, split: str) -> Path:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Expected results directory, got: {results_dir}")

    candidates = sorted(results_dir.glob("*-metrics.txt"))
    if not candidates:
        raise FileNotFoundError(f"No metrics files found under: {results_dir}")

    split_matches = [path for path in candidates if path.name.endswith(f"-{split}-metrics.txt")]
    if split_matches:
        return split_matches[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not find split-specific metrics file for split `{split}` under: {results_dir}"
    )


def parse_metrics_file(metrics_path: Path) -> dict[str, float]:
    text = metrics_path.read_text(encoding="utf-8")
    return {
        "weighted_f1": _parse_float(text, METRIC_PATTERNS["weighted_f1"], "Weighted F1"),
        "location_accuracy": _parse_float(
            text, METRIC_PATTERNS["location_accuracy"], "Average Location Accuracy"
        ),
        "joint_accuracy": _parse_float(
            text,
            METRIC_PATTERNS["joint_accuracy"],
            "Average Location-Category Joint Accuracy",
        ),
    }


def _round_delta(value: float) -> float:
    return round(value, 6)


def compare_runs(
    baseline_dir: Path,
    candidate_dir: Path,
    semantic_report_path: Path,
    split: str = "GAIA",
    paper_joint_reference: float = 0.18,
    grounded_threshold: float = 0.95,
) -> dict[str, Any]:
    baseline_metrics_path = find_metrics_file(baseline_dir, split=split)
    candidate_metrics_path = find_metrics_file(candidate_dir, split=split)

    baseline_metrics = parse_metrics_file(baseline_metrics_path)
    candidate_metrics = parse_metrics_file(candidate_metrics_path)

    semantic_report = json.loads(semantic_report_path.read_text(encoding="utf-8"))
    semantic_totals = semantic_report.get("totals", {})
    grounded_rate = float(semantic_totals.get("grounded_evidence_rate", 0.0))
    dropped_errors = int(semantic_totals.get("dropped_errors", 0))

    deltas = {
        "weighted_f1": _round_delta(
            candidate_metrics["weighted_f1"] - baseline_metrics["weighted_f1"]
        ),
        "location_accuracy": _round_delta(
            candidate_metrics["location_accuracy"] - baseline_metrics["location_accuracy"]
        ),
        "joint_accuracy": _round_delta(
            candidate_metrics["joint_accuracy"] - baseline_metrics["joint_accuracy"]
        ),
    }

    acceptance = {
        "beats_baseline_joint_accuracy": candidate_metrics["joint_accuracy"]
        > baseline_metrics["joint_accuracy"],
        "beats_paper_joint_reference": candidate_metrics["joint_accuracy"] > paper_joint_reference,
        "meets_semantic_grounded_threshold": grounded_rate >= grounded_threshold,
        "has_no_semantic_drops": dropped_errors == 0,
    }
    acceptance["accepted"] = all(acceptance.values())

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "split": split,
        "paper_joint_reference": paper_joint_reference,
        "grounded_threshold": grounded_threshold,
        "baseline": {
            "metrics_path": str(baseline_metrics_path),
            **baseline_metrics,
        },
        "candidate": {
            "metrics_path": str(candidate_metrics_path),
            **candidate_metrics,
        },
        "deltas": deltas,
        "semantic": {
            "report_path": str(semantic_report_path),
            "mode": semantic_report.get("mode"),
            "traces_processed": int(semantic_totals.get("traces_processed", 0)),
            "grounded_evidence_rate": grounded_rate,
            "dropped_errors": dropped_errors,
            "location_repaired": int(semantic_totals.get("location_repaired", 0)),
            "evidence_repaired": int(semantic_totals.get("evidence_repaired", 0)),
        },
        "acceptance": acceptance,
    }


def write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TRAIL baseline vs candidate results.")
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Baseline results root.")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="Candidate results root.")
    parser.add_argument(
        "--semantic-report",
        type=Path,
        required=True,
        help="Semantic report JSON generated by trail_main.py.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output comparison report path.")
    parser.add_argument("--split", type=str, default="GAIA", help="Benchmark split name.")
    parser.add_argument(
        "--paper-joint-reference",
        type=float,
        default=0.18,
        help="Reference joint accuracy target from TRAIL paper.",
    )
    parser.add_argument(
        "--grounded-threshold",
        type=float,
        default=0.95,
        help="Minimum grounded evidence rate required for acceptance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_runs(
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        semantic_report_path=args.semantic_report,
        split=args.split,
        paper_joint_reference=args.paper_joint_reference,
        grounded_threshold=args.grounded_threshold,
    )
    write_report(report, args.out)

    print(f"Wrote comparison report: {args.out}")
    print(
        "Joint accuracy delta: "
        f"{report['deltas']['joint_accuracy']:+.4f} "
        f"(baseline={report['baseline']['joint_accuracy']:.4f}, "
        f"candidate={report['candidate']['joint_accuracy']:.4f})"
    )
    print(f"Accepted: {report['acceptance']['accepted']}")


if __name__ == "__main__":
    main()
