# ABOUTME: Builds and emits deterministic seeded-failure traces for Phase 1 dataset creation.
# ABOUTME: Exports Phoenix spans to Parquet and keeps expected labels in an external manifest file.

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.trace import Status, StatusCode

from phoenix.otel import register
from phoenix.session.client import Client

from apps.demo_agent.phase1_langgraph_runner import otlp_http_traces_endpoint


FAILURE_LABELS = (
    "tool_failure",
    "retrieval_failure",
    "instruction_failure",
    "upstream_dependency_failure",
    "data_schema_mismatch",
)


def _deterministic_failure_label(index: int, rng: random.Random) -> str:
    if index < len(FAILURE_LABELS):
        return FAILURE_LABELS[index]
    return FAILURE_LABELS[rng.randrange(len(FAILURE_LABELS))]


def build_seed_manifest(seed: int, num_traces: int, dataset_id: str) -> dict[str, Any]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    for index in range(num_traces):
        label = _deterministic_failure_label(index, rng)
        cases.append(
            {
                "run_id": f"seed_run_{index:04d}",
                "trace_id": None,
                "expected_label": label,
                "fault_profile": f"profile_{label}",
                "notes": f"seeded deterministic case {index}",
            }
        )
    return {
        "dataset_id": dataset_id,
        "generator_version": "0.1.0",
        "seed": seed,
        "cases": cases,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def normalize_dataframe_for_parquet(dataframe):
    normalized = dataframe.copy()
    object_columns = [column for column in normalized.columns if normalized[column].dtype == object]
    for column in object_columns:
        normalized[column] = normalized[column].map(
            lambda value: None
            if value is None or value != value
            else value
            if isinstance(value, str)
            else json.dumps(value, sort_keys=True, default=str)
        )
        normalized[column] = normalized[column].where(normalized[column].notna(), None)
    return normalized


def _emit_tool_failure(tracer: trace_api.Tracer, run_id: str) -> None:
    with tracer.start_as_current_span("tool.call") as span:
        span.set_attribute("phase1.run_id", run_id)
        span.set_attribute("phase1.step", "tool.call")
        span.set_status(Status(StatusCode.ERROR, "forced tool timeout"))


def _emit_retrieval_failure(tracer: trace_api.Tracer, run_id: str) -> None:
    with tracer.start_as_current_span("retriever.fetch") as span:
        span.set_attribute("phase1.run_id", run_id)
        span.set_attribute("phase1.step", "retriever.fetch")
        span.set_attribute("phase1.retrieval.relevance", 0.11)
        span.set_status(Status(StatusCode.OK))


def _emit_instruction_failure(tracer: trace_api.Tracer, run_id: str) -> None:
    with tracer.start_as_current_span("llm.generate") as span:
        span.set_attribute("phase1.run_id", run_id)
        span.set_attribute("phase1.step", "llm.generate")
        span.set_attribute("phase1.output.format", "unexpected")
        span.set_status(Status(StatusCode.ERROR, "format drift"))


def _emit_upstream_dependency_failure(tracer: trace_api.Tracer, run_id: str) -> None:
    with tracer.start_as_current_span("dependency.http") as span:
        span.set_attribute("phase1.run_id", run_id)
        span.set_attribute("phase1.step", "dependency.http")
        span.set_attribute("http.status_code", 503)
        span.set_status(Status(StatusCode.ERROR, "upstream unavailable"))


def _emit_data_schema_mismatch(tracer: trace_api.Tracer, run_id: str) -> None:
    with tracer.start_as_current_span("tool.parse") as span:
        span.set_attribute("phase1.run_id", run_id)
        span.set_attribute("phase1.step", "tool.parse")
        span.set_status(Status(StatusCode.ERROR, "schema mismatch"))


def emit_seeded_traces(
    manifest: dict[str, Any],
    *,
    project_name: str,
    endpoint: str | None = None,
) -> None:
    collector = (endpoint or otlp_http_traces_endpoint()).rstrip("/")
    tracer_provider = register(
        endpoint=collector,
        protocol="http/protobuf",
        project_name=project_name,
        batch=False,
        verbose=False,
    )
    tracer = trace_api.get_tracer("phase1.seeded_failures")
    handlers = {
        "tool_failure": _emit_tool_failure,
        "retrieval_failure": _emit_retrieval_failure,
        "instruction_failure": _emit_instruction_failure,
        "upstream_dependency_failure": _emit_upstream_dependency_failure,
        "data_schema_mismatch": _emit_data_schema_mismatch,
    }
    for case in manifest["cases"]:
        run_id = case["run_id"]
        with tracer.start_as_current_span("agent.run") as root:
            root.set_attribute("phase1.run_id", run_id)
            root.set_attribute("phase1.project", project_name)
            handlers[case["expected_label"]](tracer, run_id)
    tracer_provider.force_flush()


def export_spans_to_parquet(
    *,
    endpoint: str | None,
    project_name: str,
    output_path: Path,
    limit: int = 100000,
) -> int:
    client = Client(endpoint=(endpoint or "http://127.0.0.1:6006").rstrip("/"))
    dataframe = client.get_spans_dataframe(project_name=project_name, limit=limit)
    if dataframe is None or dataframe.empty:
        return 0
    dataframe = normalize_dataframe_for_parquet(dataframe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    return int(len(dataframe))


def main() -> None:
    dataset_id = "seeded_failures_v1"
    project_name = "phase1-seeded-failures"
    manifest_path = Path("datasets/seeded_failures/manifest.json")
    parquet_path = Path("datasets/seeded_failures/exports/spans.parquet")
    manifest = build_seed_manifest(seed=42, num_traces=30, dataset_id=dataset_id)
    write_manifest(manifest, manifest_path)
    emit_seeded_traces(manifest, project_name=project_name, endpoint=None)
    row_count = export_spans_to_parquet(
        endpoint=None, project_name=project_name, output_path=parquet_path
    )
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "parquet_path": str(parquet_path),
                "rows_exported": row_count,
                "project_name": project_name,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
