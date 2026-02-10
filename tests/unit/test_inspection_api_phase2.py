# ABOUTME: Validates the Phase 2A Phoenix-backed inspection API adapter behavior.
# ABOUTME: Ensures deterministic span ordering and evidence extraction from span records.

from __future__ import annotations

import json

import pandas as pd

from investigator.inspection_api.phoenix_client import PhoenixInspectionAPI


def _sample_spans_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "context.trace_id": "trace-a",
                "context.span_id": "s2",
                "parent_id": "s1",
                "name": "sql_db_query",
                "span_kind": "TOOL",
                "status_code": "OK",
                "status_message": "",
                "start_time": pd.Timestamp("2026-02-10T00:00:02Z"),
                "end_time": pd.Timestamp("2026-02-10T00:00:03Z"),
                "attributes.tool.name": "sql_db_query",
                "attributes.input.value": '{"query":"SELECT 1"}',
                "attributes.output.value": "[(1,)]",
                "events": [],
            },
            {
                "context.trace_id": "trace-a",
                "context.span_id": "s1",
                "parent_id": None,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "ERROR",
                "status_message": "boom",
                "start_time": pd.Timestamp("2026-02-10T00:00:00Z"),
                "end_time": pd.Timestamp("2026-02-10T00:00:01Z"),
                "attributes.llm.input_messages": json.dumps(
                    [{"role": "user", "content": "run sql"}]
                ),
                "attributes.llm.output_messages": json.dumps(
                    [{"role": "assistant", "content": "calling tool"}]
                ),
                "events": [],
            },
            {
                "context.trace_id": "trace-b",
                "context.span_id": "s3",
                "parent_id": None,
                "name": "retriever.fetch",
                "span_kind": "RETRIEVER",
                "status_code": "OK",
                "status_message": "",
                "start_time": pd.Timestamp("2026-02-10T00:00:05Z"),
                "end_time": pd.Timestamp("2026-02-10T00:00:06Z"),
                "attributes.retrieval.documents": json.dumps(
                    [{"document_id": "doc-1", "content": "alpha chunk", "score": 0.9}]
                ),
                "events": [],
            },
        ]
    )


def test_list_spans_sorted_by_start_time_then_span_id() -> None:
    spans_df = _sample_spans_dataframe()
    api = PhoenixInspectionAPI(
        spans_dataframe_provider=lambda **_: spans_df,
    )

    spans = api.list_spans("trace-a")
    assert [span["span_id"] for span in spans] == ["s1", "s2"]


def test_get_tool_io_returns_stable_artifact_id() -> None:
    spans_df = _sample_spans_dataframe()
    api = PhoenixInspectionAPI(
        spans_dataframe_provider=lambda **_: spans_df,
    )

    tool_io = api.get_tool_io("s2")
    assert tool_io is not None
    assert tool_io["artifact_id"] == "tool:s2"
    assert tool_io["tool_name"] == "sql_db_query"


def test_get_messages_extracts_llm_messages() -> None:
    spans_df = _sample_spans_dataframe()
    api = PhoenixInspectionAPI(
        spans_dataframe_provider=lambda **_: spans_df,
    )

    messages = api.get_messages("s1")
    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_get_retrieval_chunks_uses_canonical_artifact_id() -> None:
    spans_df = _sample_spans_dataframe()
    api = PhoenixInspectionAPI(
        spans_dataframe_provider=lambda **_: spans_df,
    )

    chunks = api.get_retrieval_chunks("s3")
    assert len(chunks) == 1
    assert chunks[0]["artifact_id"] == "retrieval:s3:0:doc-1"
    assert chunks[0]["document_id"] == "doc-1"


def test_list_traces_returns_deterministic_order() -> None:
    spans_df = _sample_spans_dataframe()
    api = PhoenixInspectionAPI(
        spans_dataframe_provider=lambda **_: spans_df,
    )

    traces = api.list_traces("phase2-project")
    assert [trace["trace_id"] for trace in traces] == ["trace-a", "trace-b"]
    assert traces[0]["span_count"] == 2
