# ABOUTME: Runs seeded fault-injection trace generation and resolves resulting trace IDs from Phoenix.
# ABOUTME: Updates seeded-failure manifests with run_id-to-trace_id mappings for Phase 10 Milestone 1.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phoenix.session.client import Client

from apps.demo_agent.phase1_seeded_failures import emit_seeded_traces, export_spans_to_parquet


DEFAULT_PROJECT_NAME = "phase1-seeded-failures"
DEFAULT_LOOKUP_LIMIT = 100000
DEFAULT_EXPORT_PATH = Path("datasets/seeded_failures/exports/spans.parquet")


FAULT_PROFILE_TO_LABEL = {
    "profile_tool_failure": "tool_failure",
    "profile_retrieval_failure": "retrieval_failure",
    "profile_instruction_failure": "instruction_failure",
    "profile_upstream_dependency_failure": "upstream_dependency_failure",
    "profile_data_schema_mismatch": "data_schema_mismatch",
}


def _base_phoenix_endpoint(endpoint: str) -> str:
    value = str(endpoint or "http://127.0.0.1:6006").strip()
    if not value:
        return "http://127.0.0.1:6006"
    if value.endswith("/v1/traces"):
        value = value[: -len("/v1/traces")]
    return value.rstrip("/")


def _collector_traces_endpoint(endpoint: str) -> str:
    base = _base_phoenix_endpoint(endpoint)
    return f"{base}/v1/traces"


def _single_case_manifest(*, fault_profile: str, run_id: str) -> dict[str, Any]:
    if fault_profile not in FAULT_PROFILE_TO_LABEL:
        supported = ", ".join(sorted(FAULT_PROFILE_TO_LABEL))
        raise ValueError(f"Unsupported fault profile '{fault_profile}'. Supported: {supported}")
    return {
        "dataset_id": "seeded_failures_v1",
        "generator_version": "0.1.0",
        "seed": None,
        "cases": [
            {
                "run_id": run_id,
                "trace_id": None,
                "expected_label": FAULT_PROFILE_TO_LABEL[fault_profile],
                "fault_profile": fault_profile,
                "notes": f"seeded deterministic case for {run_id}",
            }
        ],
    }


def _resolve_run_id_column(columns: list[str]) -> str | None:
    preferred = [
        "attributes.phase1.run_id",
        "phase1.run_id",
        "run_id",
    ]
    for name in preferred:
        if name in columns:
            return name
    for name in columns:
        if "phase1.run_id" in name:
            return name
    for name in columns:
        if name.endswith("run_id"):
            return name
    return None


def _resolve_trace_id_column(columns: list[str]) -> str | None:
    if "context.trace_id" in columns:
        return "context.trace_id"
    if "trace_id" in columns:
        return "trace_id"
    for name in columns:
        if name.endswith("trace_id"):
            return name
    return None


def _resolve_sort_column(columns: list[str]) -> str | None:
    for candidate in ("start_time", "start_time_unix_nano", "start_time_utc"):
        if candidate in columns:
            return candidate
    return None


def _lookup_trace_ids_by_run_id(
    *,
    endpoint: str,
    project_name: str,
    run_ids: list[str],
    limit: int = DEFAULT_LOOKUP_LIMIT,
) -> dict[str, str]:
    if not run_ids:
        return {}

    client = Client(endpoint=_base_phoenix_endpoint(endpoint))
    dataframe = client.get_spans_dataframe(project_name=project_name, limit=limit)
    if dataframe is None or dataframe.empty:
        return {}

    run_id_column = _resolve_run_id_column(list(dataframe.columns))
    trace_id_column = _resolve_trace_id_column(list(dataframe.columns))
    if run_id_column is None or trace_id_column is None:
        return {}

    subset = dataframe[[run_id_column, trace_id_column]].copy()
    sort_column = _resolve_sort_column(list(dataframe.columns))
    if sort_column is not None:
        subset[sort_column] = dataframe[sort_column]
        subset = subset.sort_values(by=sort_column, kind="stable")
    subset = subset.dropna(subset=[run_id_column, trace_id_column])

    target_ids = {str(value) for value in run_ids}
    mapping: dict[str, str] = {}
    for _, row in subset.iterrows():
        run_id_value = str(row[run_id_column])
        if run_id_value not in target_ids:
            continue
        mapping[run_id_value] = str(row[trace_id_column])
    return mapping


def run_with_fault(
    *,
    fault_profile: str,
    run_id: str,
    phoenix_endpoint: str = "http://127.0.0.1:6006",
    project_name: str = DEFAULT_PROJECT_NAME,
    lookup_limit: int = DEFAULT_LOOKUP_LIMIT,
) -> str:
    manifest = _single_case_manifest(fault_profile=fault_profile, run_id=run_id)
    emit_seeded_traces(
        manifest,
        project_name=project_name,
        endpoint=_collector_traces_endpoint(phoenix_endpoint),
    )
    trace_id_by_run = _lookup_trace_ids_by_run_id(
        endpoint=phoenix_endpoint,
        project_name=project_name,
        run_ids=[run_id],
        limit=lookup_limit,
    )
    trace_id = trace_id_by_run.get(run_id)
    if not trace_id:
        raise RuntimeError(
            f"Trace ID could not be resolved for run_id '{run_id}' in project '{project_name}'."
        )
    return trace_id


def run_all_seeded_failures(
    *,
    manifest_path: str = "datasets/seeded_failures/manifest.json",
    phoenix_endpoint: str = "http://127.0.0.1:6006",
    project_name: str = DEFAULT_PROJECT_NAME,
    export_path: Path = DEFAULT_EXPORT_PATH,
    lookup_limit: int = DEFAULT_LOOKUP_LIMIT,
) -> dict[str, str]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    run_to_trace: dict[str, str] = {}
    for case in manifest.get("cases", []):
        run_id = str(case.get("run_id", "")).strip()
        fault_profile = str(case.get("fault_profile", "")).strip()
        if not run_id:
            raise ValueError("Each manifest case must include a non-empty run_id.")
        trace_id = run_with_fault(
            fault_profile=fault_profile,
            run_id=run_id,
            phoenix_endpoint=phoenix_endpoint,
            project_name=project_name,
            lookup_limit=lookup_limit,
        )
        case["trace_id"] = trace_id
        run_to_trace[run_id] = trace_id

    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    export_spans_to_parquet(
        endpoint=_base_phoenix_endpoint(phoenix_endpoint),
        project_name=project_name,
        output_path=Path(export_path),
        limit=lookup_limit,
    )
    return run_to_trace
