# ABOUTME: Validates Phase 7 Parquet-backed Inspection API behavior for deterministic offline evaluator runs.
# ABOUTME: Ensures trace/span/tool/retrieval reads work from frozen Parquet without live Phoenix dependency.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from investigator.inspection_api.parquet_client import ParquetInspectionAPI


def _write_sample_parquet(path: Path) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "name": "agent.run",
                "span_kind": "AGENT",
                "parent_id": None,
                "start_time": pd.Timestamp("2026-02-10T00:00:00Z"),
                "end_time": pd.Timestamp("2026-02-10T00:00:01Z"),
                "status_code": "ERROR",
                "status_message": "tool timeout",
                "events": [],
                "context.span_id": "root-a",
                "context.trace_id": "trace-a",
                "attributes.phase1": {"run_id": "seed_run_0000", "step": None},
                "attributes.http": None,
            },
            {
                "name": "tool.call",
                "span_kind": "TOOL",
                "parent_id": "root-a",
                "start_time": pd.Timestamp("2026-02-10T00:00:00Z"),
                "end_time": pd.Timestamp("2026-02-10T00:00:01Z"),
                "status_code": "ERROR",
                "status_message": "forced tool timeout",
                "events": [],
                "context.span_id": "tool-a",
                "context.trace_id": "trace-a",
                "attributes.phase1": {"run_id": "seed_run_0000", "step": "tool.call"},
                "attributes.http": None,
            },
            {
                "name": "retriever.fetch",
                "span_kind": "RETRIEVER",
                "parent_id": None,
                "start_time": pd.Timestamp("2026-02-10T00:01:00Z"),
                "end_time": pd.Timestamp("2026-02-10T00:01:01Z"),
                "status_code": "OK",
                "status_message": "",
                "events": [],
                "context.span_id": "retr-b",
                "context.trace_id": "trace-b",
                "attributes.phase1": {
                    "run_id": "seed_run_0001",
                    "step": "retriever.fetch",
                    "retrieval": {
                        "documents": [
                            {
                                "document_id": "doc-1",
                                "content": "alpha chunk",
                                "score": 0.9,
                            }
                        ]
                    },
                },
                "attributes.http": None,
            },
        ]
    )
    dataframe.to_parquet(path, index=False)


def test_parquet_inspection_api_reads_traces_spans_tool_and_retrieval(tmp_path: Path) -> None:
    parquet_path = tmp_path / "spans.parquet"
    _write_sample_parquet(parquet_path)

    api = ParquetInspectionAPI(parquet_path=parquet_path)

    traces = api.list_traces("phase7-proof")
    assert [trace["trace_id"] for trace in traces] == ["trace-a", "trace-b"]

    spans = api.list_spans("trace-a")
    assert [span["span_id"] for span in spans] == ["root-a", "tool-a"]

    tool_io = api.get_tool_io("tool-a")
    assert tool_io is not None
    assert tool_io["artifact_id"] == "tool:tool-a"

    retrieval_chunks = api.get_retrieval_chunks("retr-b")
    assert len(retrieval_chunks) == 1
    assert retrieval_chunks[0]["artifact_id"] == "retrieval:retr-b:0:doc-1"


def test_parquet_inspection_api_supports_seed_manifest_linking(tmp_path: Path) -> None:
    parquet_path = tmp_path / "spans.parquet"
    _write_sample_parquet(parquet_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "seeded_failures_v1",
                "generator_version": "0.1.0",
                "seed": 42,
                "cases": [
                    {
                        "run_id": "seed_run_0000",
                        "trace_id": None,
                        "expected_label": "tool_failure",
                    },
                    {
                        "run_id": "seed_run_0001",
                        "trace_id": None,
                        "expected_label": "retrieval_failure",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    api = ParquetInspectionAPI(parquet_path=parquet_path)
    updated_manifest = api.attach_manifest_trace_ids(manifest_path=manifest_path)

    mapping = {case["run_id"]: case["trace_id"] for case in updated_manifest["cases"]}
    assert mapping["seed_run_0000"] == "trace-a"
    assert mapping["seed_run_0001"] == "trace-b"
