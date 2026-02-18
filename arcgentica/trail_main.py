# ABOUTME: Provides a CLI entrypoint to generate TRAIL benchmark outputs from GAIA or SWE trace files.
# ABOUTME: Writes one JSON prediction per trace using ARCgentica's deterministic phase11 analysis pipeline.

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from .trail_agent import analyze_trace
    from .trail_common import TrailRunConfig, iter_trace_files, run_output_dir
    from .trail_semantic_checks import enforce_semantic_faithfulness
except ImportError:
    from trail_agent import analyze_trace
    from trail_common import TrailRunConfig, iter_trace_files, run_output_dir
    from trail_semantic_checks import enforce_semantic_faithfulness

load_dotenv()


def _default_semantic_report_path() -> Path:
    return Path(__file__).resolve().parent / "output" / "trail_semantic_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TRAIL predictions with ARCgentica.")
    parser.add_argument(
        "--trail-data-dir",
        type=Path,
        required=True,
        help="Path to TRAIL benchmark data directory containing split folders.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="GAIA",
        choices=["GAIA", "SWE Bench"],
        help="TRAIL split to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5.2",
        help="Model id recorded in run metadata and used for future model-based analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory where scorer-compatible outputs will be written.",
    )
    parser.add_argument(
        "--semantic-checks",
        type=str,
        default="off",
        choices=["off", "strict"],
        help="Semantic checks mode for output gating.",
    )
    parser.add_argument(
        "--semantic-report-path",
        type=Path,
        default=_default_semantic_report_path(),
        help="Path for semantic faithfulness run report JSON artifact.",
    )
    parser.add_argument(
        "--agentic-mode",
        type=str,
        default="on",
        choices=["off", "on"],
        help="Enable recursive Agentica investigation for long traces.",
    )
    parser.add_argument(
        "--max-num-agents",
        type=int,
        default=6,
        help="Maximum number of recursively spawned agents per trace.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=6,
        help="Maximum number of span chunks to investigate per trace.",
    )
    parser.add_argument(
        "--max-spans-per-chunk",
        type=int,
        default=12,
        help="Maximum number of spans included in each delegated chunk task.",
    )
    parser.add_argument(
        "--max-span-text-chars",
        type=int,
        default=1200,
        help="Maximum number of characters retained per span snippet.",
    )
    return parser.parse_args()


def generate_split_outputs(
    config: TrailRunConfig,
    semantic_report_path: Path | None = None,
) -> Path:
    output_dir = run_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_files = iter_trace_files(config)
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

    for trace_file in trace_files:
        trace_payload = json.loads(trace_file.read_text(encoding="utf-8"))
        prediction = analyze_trace(
            trace_payload,
            model=config.model,
            agentic_mode=config.agentic_mode,
            max_num_agents=config.max_num_agents,
            max_chunks=config.max_chunks,
            max_spans_per_chunk=config.max_spans_per_chunk,
            max_span_text_chars=config.max_span_text_chars,
        )
        prediction, semantic_report = enforce_semantic_faithfulness(
            trace_payload=trace_payload,
            prediction=prediction,
            mode=config.semantic_checks,
        )
        total_errors += int(semantic_report.get("total_errors", 0))
        kept_errors += int(semantic_report.get("kept_errors", 0))
        dropped_errors += int(semantic_report.get("dropped_errors", 0))
        for key in repair_actions:
            repair_actions[key] += int(semantic_report.get("repair_actions", {}).get(key, 0))
        for key in drop_reasons:
            drop_reasons[key] += int(semantic_report.get("drop_reasons", {}).get(key, 0))

        trace_report = dict(semantic_report)
        trace_report["trace_file"] = trace_file.name
        file_reports.append(trace_report)

        out_path = output_dir / trace_file.name
        out_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")

    report_path = semantic_report_path or _default_semantic_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    run_report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": config.semantic_checks,
        "split": config.split,
        "model": config.model,
        "output_dir": str(output_dir),
        "totals": {
            "traces_processed": len(trace_files),
            "total_errors": total_errors,
            "kept_errors": kept_errors,
            "dropped_errors": dropped_errors,
            "grounded_evidence_rate": 1.0 if total_errors == 0 else kept_errors / total_errors,
            "location_repaired": repair_actions["location_repaired"],
            "evidence_repaired": repair_actions["evidence_repaired"],
        },
        "repair_actions": repair_actions,
        "drop_reasons": drop_reasons,
        "files": file_reports,
    }
    report_path.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    return output_dir


def main() -> None:
    args = parse_args()
    config = TrailRunConfig(
        trail_data_dir=args.trail_data_dir,
        split=args.split,
        model=args.model,
        output_dir=args.output_dir,
        semantic_checks=args.semantic_checks,
        agentic_mode=args.agentic_mode,
        max_num_agents=args.max_num_agents,
        max_chunks=args.max_chunks,
        max_spans_per_chunk=args.max_spans_per_chunk,
        max_span_text_chars=args.max_span_text_chars,
    )
    output_dir = generate_split_outputs(config, semantic_report_path=args.semantic_report_path)
    print(f"Wrote TRAIL outputs to: {output_dir}")
    print(f"Wrote semantic report to: {args.semantic_report_path}")


if __name__ == "__main__":
    main()
