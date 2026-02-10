# ABOUTME: Validates Phase 8 Trace RCA engine integration with the shared structured-generation LLM loop.
# ABOUTME: Ensures runtime usage/cost signals are emitted and invalid model output maps to runtime failure taxonomy.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.runtime.llm_client import StructuredGenerationResult, StructuredGenerationUsage
from investigator.runtime.runner import run_engine


class _FakeInspectionAPI:
    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [
            {
                "trace_id": trace_id,
                "span_id": "root",
                "parent_id": None,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "ERROR",
                "status_message": "tool timeout",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 80.0,
            }
        ]

    def get_span(self, span_id: str) -> dict[str, Any]:
        return {
            "summary": {
                "trace_id": "trace-llm",
                "span_id": span_id,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "ERROR",
                "status_message": "tool timeout",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 80.0,
            },
            "attributes": {},
            "events": [],
        }

    def get_children(self, span_id: str) -> list[dict[str, Any]]:
        del span_id
        return []

    def get_tool_io(self, span_id: str) -> dict[str, Any] | None:
        del span_id
        return None

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        del span_id
        return []


class _FakeModelClient:
    model_provider = "openai"

    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    def generate_structured(self, request):  # noqa: ANN001, ANN201
        del request
        if not self._outputs:
            raise AssertionError("No fake outputs configured.")
        self.calls += 1
        payload = self._outputs.pop(0)
        return StructuredGenerationResult(
            output=payload,
            raw_text=json.dumps(payload, sort_keys=True),
            usage=StructuredGenerationUsage(tokens_in=111, tokens_out=22, cost_usd=0.05),
        )


def test_trace_rca_engine_llm_path_emits_runtime_model_usage(tmp_path: Path) -> None:
    model_client = _FakeModelClient(
        [
            {
                "primary_label": "upstream_dependency_failure",
                "summary": "Upstream timeout signatures dominate hot spans.",
                "confidence": 0.77,
                "remediation": ["Add timeout backoff for upstream API calls."],
                "gaps": [],
            }
        ]
    )
    engine = TraceRCAEngine(
        inspection_api=_FakeInspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
    )

    report, run_record = run_engine(
        engine=engine,
        request=TraceRCARequest(trace_id="trace-llm", project_name="phase8"),
        run_id="run-phase8-rca-llm",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert report.primary_label == "upstream_dependency_failure"
    assert model_client.calls == 1
    assert run_record.runtime_ref.usage.tokens_in == 111
    assert run_record.runtime_ref.usage.tokens_out == 22
    assert run_record.runtime_ref.usage.cost_usd == 0.05
    assert run_record.runtime_ref.model_provider == "openai"


def test_trace_rca_engine_llm_invalid_output_maps_to_runtime_failure(tmp_path: Path) -> None:
    model_client = _FakeModelClient(
        outputs=[
            {"summary": "missing required keys"},
            {"summary": "still invalid"},
        ]
    )
    engine = TraceRCAEngine(
        inspection_api=_FakeInspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_engine(
            engine=engine,
            request=TraceRCARequest(trace_id="trace-llm", project_name="phase8"),
            run_id="run-phase8-rca-llm-invalid",
            artifacts_root=tmp_path / "artifacts" / "investigator_runs",
        )

    payload = exc_info.value.args[0]
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "MODEL_OUTPUT_INVALID"
    assert payload["runtime_ref"]["usage"]["tokens_in"] == 222
    assert payload["runtime_ref"]["usage"]["cost_usd"] == pytest.approx(0.1)


def test_trace_rca_engine_can_fallback_to_deterministic_on_llm_error() -> None:
    model_client = _FakeModelClient(
        outputs=[
            {"summary": "missing required keys"},
            {"summary": "still invalid"},
        ]
    )
    engine = TraceRCAEngine(
        inspection_api=_FakeInspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
        fallback_on_llm_error=True,
    )

    report = engine.run(TraceRCARequest(trace_id="trace-llm", project_name="phase8"))
    signals = engine.get_runtime_signals()

    assert report.primary_label == "upstream_dependency_failure"
    assert model_client.calls == 2
    assert signals["rca_judgment_mode"] == "deterministic_fallback"
