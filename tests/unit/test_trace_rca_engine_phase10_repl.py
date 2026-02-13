# ABOUTME: Validates Phase 10 RCA engine wiring to the shared REPL runtime loop.
# ABOUTME: Ensures non-trivial REPL runs emit llm_subcalls and repl trajectory in run records.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.runtime.contracts import RuntimeBudget
from investigator.runtime.llm_client import StructuredGenerationResult, StructuredGenerationUsage
from investigator.runtime.runner import run_engine


class _InspectionAPI:
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
                "trace_id": "trace-rca-repl",
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

    def get_tool_io(self, span_id: str) -> dict[str, Any]:
        return {
            "trace_id": "trace-rca-repl",
            "span_id": span_id,
            "artifact_id": f"tool:{span_id}",
            "tool_name": "lookup",
            "status_code": "ERROR",
        }

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        del span_id
        return []


class _FakeModelClient:
    model_provider = "openai"

    def __init__(self, *, step_outputs: list[dict[str, Any]], subquery_outputs: list[str]) -> None:
        self._step_outputs = list(step_outputs)
        self._subquery_outputs = list(subquery_outputs)
        self.calls = 0

    def generate_structured(self, request):  # noqa: ANN001, ANN201
        self.calls += 1
        schema_name = str(getattr(request, "response_schema_name", ""))
        if schema_name == "repl_runtime_step_v1":
            payload = self._step_outputs.pop(0)
            usage = StructuredGenerationUsage(tokens_in=120, tokens_out=24, cost_usd=0.05)
            return StructuredGenerationResult(
                output=payload,
                raw_text=json.dumps(payload, sort_keys=True),
                usage=usage,
            )
        if schema_name == "repl_runtime_subquery_v1":
            answer = self._subquery_outputs.pop(0)
            payload = {"answer": answer}
            usage = StructuredGenerationUsage(tokens_in=60, tokens_out=12, cost_usd=0.02)
            return StructuredGenerationResult(
                output=payload,
                raw_text=json.dumps(payload, sort_keys=True),
                usage=usage,
            )
        raise AssertionError(f"Unexpected schema name: {schema_name}")


def test_trace_rca_repl_runtime_emits_llm_subcalls_and_repl_trajectory(tmp_path: Path) -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Inspect spans and use semantic subquery before final label.",
                "code": (
                    "spans_payload = call_tool('list_spans', trace_id=trace_id)\n"
                    "label_note = llm_query('Determine the strongest RCA label from hot spans and tool state.')\n"
                    "SUBMIT("
                    "primary_label='tool_failure',"
                    "summary=f'RCA via REPL: {label_note}',"
                    "confidence=0.8,"
                    "remediation=['Add retries and classify tool errors early.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=["tool_failure"],
    )
    engine = TraceRCAEngine(
        inspection_api=_InspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
        use_repl_runtime=True,
    )

    report, run_record = run_engine(
        engine=engine,
        request=TraceRCARequest(trace_id="trace-rca-repl", project_name="phase10"),
        run_id="run-phase10-rca-repl",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert report.primary_label == "tool_failure"
    assert run_record.runtime_ref.usage.llm_subcalls > 0
    assert run_record.runtime_ref.repl_trajectory
    assert run_record.runtime_ref.usage.tokens_in > 0
    assert run_record.runtime_ref.usage.cost_usd > 0.0


def test_trace_rca_repl_tightens_per_trace_budget() -> None:
    budget = RuntimeBudget(
        max_iterations=8,
        max_depth=3,
        max_tool_calls=30,
        max_subcalls=7,
        max_tokens_total=50000,
        max_cost_usd=0.5,
        max_wall_time_sec=120,
    )

    tightened = TraceRCAEngine._tighten_repl_trace_budget(budget)

    assert tightened.max_iterations == 1
    assert tightened.max_depth == 3
    assert tightened.max_tool_calls == 6
    assert tightened.max_subcalls == 2
    assert tightened.max_tokens_total == 18000
    assert tightened.max_cost_usd == 0.12
    assert tightened.max_wall_time_sec == 45


def test_trace_rca_uses_repl_runtime_by_default(tmp_path: Path) -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Use semantic subquery, then submit.",
                "code": (
                    "spans_payload = call_tool('list_spans', trace_id=trace_id)\n"
                    "label_note = llm_query('Return one RCA label token.')\n"
                    "SUBMIT("
                    "primary_label='tool_failure',"
                    "summary=f'Default REPL path label: {label_note}',"
                    "confidence=0.74,"
                    "remediation=['Add retries around tool calls.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=["tool_failure"],
    )
    engine = TraceRCAEngine(
        inspection_api=_InspectionAPI(),
        model_client=model_client,
    )

    report, run_record = run_engine(
        engine=engine,
        request=TraceRCARequest(trace_id="trace-rca-repl", project_name="phase10"),
        run_id="run-phase10-rca-default-repl",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert report.primary_label == "tool_failure"
    assert model_client.calls > 0
    assert run_record.runtime_ref.repl_trajectory


def test_trace_rca_falls_back_to_deterministic_when_repl_step_is_invalid() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Invalid output missing code.",
                "code": "",
            }
        ],
        subquery_outputs=[],
    )
    engine = TraceRCAEngine(
        inspection_api=_InspectionAPI(),
        model_client=model_client,
    )

    report = engine.run(TraceRCARequest(trace_id="trace-rca-repl", project_name="phase10"))
    runtime_signals = engine.get_runtime_signals()

    assert report.primary_label == "upstream_dependency_failure"
    assert runtime_signals.get("rca_judgment_mode") == "deterministic_fallback"
    assert any("REPL RCA judgment failed" in str(gap) for gap in report.gaps)
