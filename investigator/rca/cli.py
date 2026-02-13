# ABOUTME: Provides a command-line entrypoint for running Trace RCA on one trace or a manifest batch.
# ABOUTME: Wires inspection APIs, runtime budgets, workflow execution, and summary reporting.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from investigator.inspection_api import ParquetInspectionAPI, PhoenixInspectionAPI
from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.rca.workflow import run_trace_rca_workflow
from investigator.runtime.contracts import DatasetRef, RuntimeBudget


class _NoWritebackClient:
    def log_evaluations(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        del args, kwargs
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Trace RCA for one trace or a manifest batch.")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--trace-id", help="Trace id for one RCA run.")
    target_group.add_argument("--manifest", help="Manifest JSON path for batch RCA runs.")
    parser.add_argument("--phoenix-endpoint", default="http://127.0.0.1:6006")
    parser.add_argument("--parquet", help="Offline spans parquet path. Skips Phoenix when provided.")
    parser.add_argument("--output-dir", default="artifacts/investigator_runs")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--max-tool-calls", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-wall-time", type=int, default=180)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-writeback", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _build_budget(args: argparse.Namespace) -> RuntimeBudget:
    return RuntimeBudget(
        max_iterations=max(1, int(args.max_iterations)),
        max_depth=max(1, int(args.max_depth)),
        max_tool_calls=max(1, int(args.max_tool_calls)),
        max_wall_time_sec=max(1, int(args.max_wall_time)),
    )


def _build_inspection_api(args: argparse.Namespace) -> Any:
    if args.parquet:
        return ParquetInspectionAPI(parquet_path=args.parquet)
    return PhoenixInspectionAPI(endpoint=args.phoenix_endpoint)


def _build_engine(*, inspection_api: Any, model_name: str) -> TraceRCAEngine:
    engine = TraceRCAEngine(
        inspection_api=inspection_api,
        use_llm_judgment=True,
        use_repl_runtime=True,
        fallback_on_llm_error=True,
    )
    engine.model_name = str(model_name)
    return engine


def _load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object.")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Manifest must include a list field named 'cases'.")
    return payload


def _print_repl_trajectory(run_record: Any) -> None:
    runtime_ref = getattr(run_record, "runtime_ref", None)
    trajectory = getattr(runtime_ref, "repl_trajectory", []) if runtime_ref is not None else []
    if not isinstance(trajectory, list) or not trajectory:
        return
    print(f"REPL trajectory ({len(trajectory)} step(s)):", file=sys.stderr)
    for index, step in enumerate(trajectory, start=1):
        if not isinstance(step, dict):
            continue
        reasoning = str(step.get("reasoning") or "").strip()
        output = str(step.get("output") or "").strip()
        print(f"[step {index}] reasoning={reasoning}", file=sys.stderr)
        if output:
            print(f"[step {index}] output={output}", file=sys.stderr)


def _run_one_trace(
    *,
    trace_id: str,
    engine: TraceRCAEngine,
    budget: RuntimeBudget,
    dataset_ref: DatasetRef | None,
    artifacts_root: str | Path,
    writeback_client: Any | None,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    try:
        report, run_record = run_trace_rca_workflow(
            request=TraceRCARequest(trace_id=trace_id, project_name="phase10-cli"),
            engine=engine,
            budget=budget,
            dataset_ref=dataset_ref,
            artifacts_root=artifacts_root,
            writeback_client=writeback_client,
        )
    except RuntimeError as exc:
        payload = exc.args[0] if exc.args and isinstance(exc.args[0], dict) else {"error": str(exc)}
        return {}, payload, 1

    if verbose:
        _print_repl_trajectory(run_record)
    report_payload = report.to_dict() if hasattr(report, "to_dict") else {"report": str(report)}
    run_payload = run_record.to_dict() if hasattr(run_record, "to_dict") else {"status": "failed"}
    status = str(run_payload.get("status") or "")
    exit_code = 0 if status in {"succeeded", "partial"} else 1
    return report_payload, run_payload, exit_code


def _print_batch_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No runnable manifest cases were found.")
        return
    headers = ["run_id", "trace_id", "predicted_label", "expected_label", "match"]
    widths: dict[str, int] = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def _run_single(args: argparse.Namespace) -> int:
    inspection_api = _build_inspection_api(args)
    engine = _build_engine(inspection_api=inspection_api, model_name=str(args.model))
    budget = _build_budget(args)
    writeback_client = _NoWritebackClient() if args.no_writeback else None
    report_payload, run_payload, exit_code = _run_one_trace(
        trace_id=str(args.trace_id),
        engine=engine,
        budget=budget,
        dataset_ref=None,
        artifacts_root=args.output_dir,
        writeback_client=writeback_client,
        verbose=bool(args.verbose),
    )
    if report_payload:
        print(json.dumps(report_payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(run_payload, indent=2, sort_keys=True), file=sys.stderr)
    return exit_code


def _run_batch(args: argparse.Namespace) -> int:
    inspection_api = _build_inspection_api(args)
    manifest_payload = _load_manifest(str(args.manifest))
    if args.parquet and hasattr(inspection_api, "attach_manifest_trace_ids"):
        inspection_api.attach_manifest_trace_ids(manifest_path=str(args.manifest))
        manifest_payload = _load_manifest(str(args.manifest))
    engine = _build_engine(inspection_api=inspection_api, model_name=str(args.model))
    budget = _build_budget(args)
    writeback_client = _NoWritebackClient() if args.no_writeback else None
    dataset_id_raw = manifest_payload.get("dataset_id")
    dataset_ref = DatasetRef(dataset_id=str(dataset_id_raw)) if dataset_id_raw else DatasetRef()

    rows: list[dict[str, Any]] = []
    highest_exit_code = 0
    for case in manifest_payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        trace_id = str(case.get("trace_id") or "").strip()
        if not trace_id:
            continue
        expected_label = str(case.get("expected_label") or "").strip()
        report_payload, run_payload, exit_code = _run_one_trace(
            trace_id=trace_id,
            engine=engine,
            budget=budget,
            dataset_ref=dataset_ref,
            artifacts_root=args.output_dir,
            writeback_client=writeback_client,
            verbose=bool(args.verbose),
        )
        highest_exit_code = max(highest_exit_code, exit_code)
        predicted_label = str(report_payload.get("primary_label") or "")
        if not predicted_label:
            error_payload = run_payload.get("error")
            if isinstance(error_payload, dict):
                predicted_label = f"failed:{error_payload.get('code', 'unknown')}"
            else:
                predicted_label = "failed"
        run_id = str(run_payload.get("run_id") or "")
        matched = bool(predicted_label and expected_label and predicted_label == expected_label)
        rows.append(
            {
                "run_id": run_id,
                "trace_id": trace_id,
                "predicted_label": predicted_label,
                "expected_label": expected_label,
                "match": "yes" if matched else "no",
            }
        )

    _print_batch_table(rows)
    return highest_exit_code


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.trace_id:
        return _run_single(args)
    return _run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
