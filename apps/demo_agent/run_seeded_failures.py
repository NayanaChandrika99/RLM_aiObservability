# ABOUTME: Provides a CLI entrypoint to generate seeded fault traces and update manifest trace IDs.
# ABOUTME: Runs Phase 10 Milestone 1 batch execution against a local or remote Phoenix endpoint.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apps.demo_agent.fault_injector import run_all_seeded_failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate seeded fault traces and update manifest trace IDs.")
    parser.add_argument(
        "--manifest",
        default="datasets/seeded_failures/manifest.json",
        help="Path to seeded-failures manifest JSON file.",
    )
    parser.add_argument(
        "--phoenix-endpoint",
        default="http://127.0.0.1:6006",
        help="Phoenix server endpoint (base URL or /v1/traces URL).",
    )
    parser.add_argument(
        "--project-name",
        default="phase1-seeded-failures",
        help="Phoenix project name used for generated traces.",
    )
    parser.add_argument(
        "--export-path",
        default="datasets/seeded_failures/exports/spans.parquet",
        help="Parquet output path for exported spans.",
    )
    parser.add_argument(
        "--lookup-limit",
        type=int,
        default=100000,
        help="Maximum number of spans to scan when resolving run_id to trace_id.",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Use only live LlamaIndex fault generation and fail fast if live execution is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = run_all_seeded_failures(
        manifest_path=args.manifest,
        phoenix_endpoint=args.phoenix_endpoint,
        project_name=args.project_name,
        export_path=Path(args.export_path),
        lookup_limit=args.lookup_limit,
        live_only=args.live_only,
    )
    print(
        json.dumps(
            {
                "manifest_path": args.manifest,
                "project_name": args.project_name,
                "phoenix_endpoint": args.phoenix_endpoint,
                "live_only": args.live_only,
                "cases_processed": len(mapping),
                "trace_ids": mapping,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
