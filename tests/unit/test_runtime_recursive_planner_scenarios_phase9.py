# ABOUTME: Validates the Phase 9 shared model-backed planner protocol for recursive runtime actions.
# ABOUTME: Exercises five representative planner scenarios to ensure typed actions run without manual correction.

from __future__ import annotations

import json
from typing import Any

from investigator.runtime.contracts import RuntimeBudget
from investigator.runtime.llm_client import (
    StructuredGenerationRequest,
    StructuredGenerationResult,
    StructuredGenerationUsage,
)
from investigator.runtime.recursive_loop import RecursiveLoop
from investigator.runtime.recursive_planner import StructuredActionPlanner
from investigator.runtime.sandbox import SandboxGuard
from investigator.runtime.tool_registry import ToolRegistry


class _InspectionAPI:
    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [{"trace_id": trace_id, "span_id": "root"}]

    def required_evidence(self, control_id: str) -> dict[str, Any]:
        return {"control_id": control_id, "required": ["messages", "tool_io"]}

    def get_tool_io(self, span_id: str) -> dict[str, Any]:
        return {"span_id": span_id, "tool_name": "search", "status": "ERROR"}

    def get_children(self, span_id: str) -> list[dict[str, Any]]:
        return [{"span_id": f"{span_id}:child"}]


class _FakeModelClient:
    model_provider = "openai"

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self._outputs = list(outputs)
        self.requests: list[StructuredGenerationRequest] = []
        self.calls = 0

    def generate_structured(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        if not self._outputs:
            raise AssertionError("No planner output fixture remaining.")
        self.requests.append(request)
        self.calls += 1
        payload = self._outputs.pop(0)
        return StructuredGenerationResult(
            output=payload,
            raw_text=json.dumps(payload, sort_keys=True),
            usage=StructuredGenerationUsage(tokens_in=11, tokens_out=7, cost_usd=0.01),
        )


def _planner_context(objective: str) -> dict[str, Any]:
    return {
        "objective": objective,
        "parent_call_id": "root",
        "depth": 0,
        "input_ref_hash": "phase9-hash",
        "state": "planning",
        "allowed_tools": ["get_children", "get_tool_io", "list_spans", "required_evidence"],
        "budget": {"max_iterations": 20},
        "usage": {"iterations": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0},
        "subcall_count": 0,
        "draft_output": {},
    }


def test_structured_action_planner_returns_action_and_usage_envelope() -> None:
    client = _FakeModelClient(
        outputs=[
            {"action": {"type": "tool_call", "tool_name": "list_spans", "args": {"trace_id": "trace-1"}}},
        ]
    )
    planner = StructuredActionPlanner(client=client, model_name="gpt-5-mini", temperature=0.0)

    planned = planner(_planner_context("rca hot-span drilldown"))

    assert isinstance(planned, dict)
    assert planned.get("action", {}).get("type") == "tool_call"
    assert planned.get("usage", {}).get("tokens_in") == 11
    assert client.calls == 1
    assert client.requests[0].response_schema_name == "recursive_runtime_action_v1"


def test_phase9_planner_scenario_harness_executes_five_representative_sequences() -> None:
    scenario_outputs: dict[str, list[dict[str, Any]]] = {
        "rca_hot_span_drilldown": [
            {"action": {"type": "tool_call", "tool_name": "list_spans", "args": {"trace_id": "trace-rca"}}},
            {"action": {"type": "finalize", "output": {"summary": "rca drilldown complete"}}},
        ],
        "rca_hypothesis_subcall": [
            {
                "action": {
                    "type": "delegate_subcall",
                    "objective": "candidate tool_failure",
                    "actions": [
                        {"type": "synthesize", "output": {"evidence_refs": [], "gaps": []}},
                        {"type": "finalize", "output": {"summary": "candidate done"}},
                    ],
                }
            },
            {"action": {"type": "finalize", "output": {"summary": "rca hypothesis chosen"}}},
        ],
        "compliance_missing_evidence": [
            {
                "action": {
                    "type": "tool_call",
                    "tool_name": "required_evidence",
                    "args": {"control_id": "CTRL-1"},
                }
            },
            {
                "action": {
                    "type": "synthesize",
                    "output": {"missing_evidence": ["messages"], "gaps": ["messages unavailable"]},
                }
            },
            {"action": {"type": "finalize", "output": {"summary": "insufficient evidence"}}},
        ],
        "compliance_fail_case": [
            {"action": {"type": "tool_call", "tool_name": "get_tool_io", "args": {"span_id": "span-1"}}},
            {"action": {"type": "finalize", "output": {"summary": "control failed with evidence"}}},
        ],
        "incident_trace_drilldown": [
            {"action": {"type": "tool_call", "tool_name": "get_children", "args": {"span_id": "root"}}},
            {
                "action": {
                    "type": "delegate_subcall",
                    "objective": "trace drilldown child",
                    "actions": [
                        {"type": "tool_call", "tool_name": "get_children", "args": {"span_id": "root:child"}},
                        {"type": "finalize", "output": {"summary": "child drilldown done"}},
                    ],
                }
            },
            {"action": {"type": "finalize", "output": {"summary": "incident trace drilldown complete"}}},
        ],
    }
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)

    for scenario_name, outputs in scenario_outputs.items():
        client = _FakeModelClient(outputs=outputs)
        planner = StructuredActionPlanner(client=client, model_name="gpt-5-mini", temperature=0.0)
        loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

        result = loop.run(
            actions=[],
            planner=planner,
            budget=RuntimeBudget(max_iterations=20),
            objective=scenario_name,
        )

        assert result.status == "completed", scenario_name
        assert result.error_code is None, scenario_name
        assert result.usage.iterations >= 2, scenario_name
        assert result.usage.tokens_in > 0, scenario_name
        assert result.usage.tokens_out > 0, scenario_name
        assert result.usage.cost_usd > 0.0, scenario_name

