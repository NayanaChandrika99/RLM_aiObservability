# ABOUTME: Validates Phase 9B Trace RCA recursive runtime wiring and runtime signal propagation.
# ABOUTME: Ensures recursive planner execution emits trajectory/subcalls and budget termination maps to partial runs.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.runtime.contracts import RuntimeBudget
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
                "trace_id": "trace-recursive",
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
        return {
            "trace_id": "trace-recursive",
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

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
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
            usage=StructuredGenerationUsage(tokens_in=90, tokens_out=21, cost_usd=0.04),
        )


def test_trace_rca_recursive_runtime_emits_trajectory_and_subcall_metadata(tmp_path: Path) -> None:
    model_client = _FakeModelClient(
        [
            {
                "action": {
                    "type": "delegate_subcall",
                    "objective": "evaluate tool_failure hypothesis",
                    "actions": [
                        {
                            "type": "synthesize",
                            "output": {
                                "evidence_refs": [
                                    {
                                        "trace_id": "trace-recursive",
                                        "span_id": "root",
                                        "kind": "SPAN",
                                        "ref": "root",
                                    }
                                ],
                                "gaps": [],
                            },
                        },
                        {"type": "finalize", "output": {"summary": "tool failure candidate supported"}},
                    ],
                }
            },
            {
                "action": {
                    "type": "synthesize",
                    "output": {
                        "primary_label": "tool_failure",
                        "confidence": 0.78,
                        "remediation": ["Add retries around tool invocation."],
                        "gaps": [],
                    },
                }
            },
            {"action": {"type": "finalize", "output": {"summary": "Recursive RCA selected tool_failure."}}},
        ]
    )
    engine = TraceRCAEngine(
        inspection_api=_FakeInspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
        use_recursive_runtime=True,
    )

    report, run_record = run_engine(
        engine=engine,
        request=TraceRCARequest(trace_id="trace-recursive", project_name="phase9"),
        run_id="run-phase9-rca-recursive",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert report.primary_label == "tool_failure"
    assert model_client.calls == 3
    assert run_record.runtime_ref.state_trajectory
    assert "delegating" in run_record.runtime_ref.state_trajectory
    assert run_record.runtime_ref.subcall_metadata
    assert run_record.runtime_ref.usage.tokens_in > 0
    assert run_record.runtime_ref.usage.cost_usd > 0.0


def test_trace_rca_recursive_runtime_budget_termination_maps_to_partial(tmp_path: Path) -> None:
    model_client = _FakeModelClient(
        outputs=[
            {"action": {"type": "tool_call", "tool_name": "list_spans", "args": {"trace_id": "trace-recursive"}}},
            {"action": {"type": "finalize", "output": {"summary": "late finalize"}}},
        ]
    )
    engine = TraceRCAEngine(
        inspection_api=_FakeInspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
        use_recursive_runtime=True,
        recursive_budget=RuntimeBudget(max_iterations=1),
        fallback_on_llm_error=True,
    )

    _, run_record = run_engine(
        engine=engine,
        request=TraceRCARequest(trace_id="trace-recursive", project_name="phase9"),
        run_id="run-phase9-rca-recursive-budget",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert run_record.status == "partial"
    assert run_record.error is not None
    assert run_record.error.code == "RECURSION_LIMIT_REACHED"
    assert "max_iterations" in run_record.error.message
