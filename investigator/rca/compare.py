# ABOUTME: CLI entry point for running all RCA scaffold presets on the same manifest and producing a comparative report.
# ABOUTME: Orchestrates per-scaffold batch runs, collects results, and emits a side-by-side evaluation JSON.

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from investigator.rca.cli import (
    _build_budget,
    _build_inspection_api,
    _load_manifest,
    _NoWritebackClient,
    _run_one_trace,
)
from investigator.rca.cli import _build_engine
from investigator.rca.evaluate import (
    evaluate_rca_runs_comparative,
    format_comparative_report,
)
from investigator.rca.scaffolds import list_scaffold_names
from investigator.runtime.contracts import DatasetRef, RuntimeBudget


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all RCA scaffold presets on a manifest and produce a comparative report."
    )
    parser.add_argument("--manifest", required=True, help="Manifest JSON path.")
    parser.add_argument("--parquet", help="Offline spans parquet path.")
    parser.add_argument("--phoenix-endpoint", default="http://127.0.0.1:6006")
    parser.add_argument("--output-dir", default="artifacts/investigator_runs")
    parser.add_argument(
        "--report-output",
        default="artifacts/evaluation/comparison_report.json",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--scaffolds",
        nargs="+",
        default=None,
        choices=list_scaffold_names(),
        help="Which scaffolds to run (default: all).",
    )
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--max-tool-calls", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-wall-time", type=int, default=180)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _timestamp_tag() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    scaffold_names = args.scaffolds or list_scaffold_names()

    manifest_payload = _load_manifest(str(args.manifest))
    inspection_api = _build_inspection_api(args)
    if args.parquet and hasattr(inspection_api, "attach_manifest_trace_ids"):
        inspection_api.attach_manifest_trace_ids(manifest_path=str(args.manifest))
        manifest_payload = _load_manifest(str(args.manifest))

    budget = _build_budget(args)
    dataset_id_raw = manifest_payload.get("dataset_id")
    dataset_ref = DatasetRef(dataset_id=str(dataset_id_raw)) if dataset_id_raw else DatasetRef()
    writeback_client = _NoWritebackClient()

    comparison_tag = f"comparison_{_timestamp_tag()}"
    base_output_dir = Path(args.output_dir) / comparison_tag
    scaffold_runs: dict[str, str | Path] = {}

    for scaffold_name in scaffold_names:
        scaffold_output_dir = base_output_dir / scaffold_name
        print(f"\n=== Scaffold: {scaffold_name} ===", file=sys.stderr)

        engine = _build_engine(
            inspection_api=inspection_api,
            model_name=str(args.model),
            scaffold_name=scaffold_name,
        )

        for case in manifest_payload.get("cases", []):
            if not isinstance(case, dict):
                continue
            trace_id = str(case.get("trace_id") or "").strip()
            if not trace_id:
                continue
            report_payload, run_payload, exit_code = _run_one_trace(
                trace_id=trace_id,
                engine=engine,
                budget=budget,
                dataset_ref=dataset_ref,
                artifacts_root=str(scaffold_output_dir),
                writeback_client=writeback_client,
                verbose=bool(args.verbose),
                scaffold=scaffold_name,
            )
            predicted = str(report_payload.get("primary_label") or "failed")
            expected = str(case.get("expected_label") or "")
            match_str = "yes" if predicted == expected else "no"
            if args.verbose:
                print(
                    f"  {trace_id}: predicted={predicted} expected={expected} match={match_str}",
                    file=sys.stderr,
                )

        scaffold_runs[scaffold_name] = scaffold_output_dir

    comparative_report = evaluate_rca_runs_comparative(
        manifest_path=str(args.manifest),
        scaffold_runs=scaffold_runs,
    )

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(comparative_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n" + format_comparative_report(comparative_report))
    print(f"\nComparative report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
