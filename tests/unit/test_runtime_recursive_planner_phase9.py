# ABOUTME: Validates Phase 9 planner-driven recursive runtime behavior and planner output validation.
# ABOUTME: Ensures the runtime can iterate actions from a planner and fails safely on invalid planner output.

from __future__ import annotations

import json
from typing import Any

import pytest

from investigator.runtime.contracts import RuntimeBudget
from investigator.runtime.recursive_loop import RecursiveLoop
from investigator.runtime.sandbox import SandboxGuard
from investigator.runtime.tool_registry import ToolRegistry


class _InspectionAPI:
    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [{"trace_id": trace_id, "span_id": "root"}]


class _PermissiveGuard:
    def validate_action(self, action: dict[str, Any]) -> None:
        del action


def test_recursive_loop_executes_planner_actions_until_finalize() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    planner_actions = [
        {"type": "tool_call", "tool_name": "list_spans", "args": {"trace_id": "trace-9"}},
        {
            "type": "synthesize",
            "output": {
                "evidence_refs": [
                    {
                        "trace_id": "trace-9",
                        "span_id": "root",
                        "kind": "SPAN",
                        "ref": "root",
                    }
                ],
                "gaps": [],
            },
        },
        {"type": "finalize", "output": {"summary": "planner finalized"}},
    ]
    planner_calls = {"count": 0}

    def _planner(_: dict[str, Any]) -> dict[str, Any]:
        action = planner_actions[planner_calls["count"]]
        planner_calls["count"] += 1
        return action

    result = loop.run(
        actions=[],
        planner=_planner,
        budget=RuntimeBudget(max_iterations=10),
        objective="phase9 planner root",
    )

    assert result.status == "completed"
    assert result.output is not None
    assert result.output.get("summary") == "planner finalized"
    assert len(result.output.get("tool_calls") or []) == 1
    assert planner_calls["count"] == 3
    assert result.usage.iterations == 3


def test_recursive_loop_fails_when_planner_returns_invalid_action() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    def _bad_planner(_: dict[str, Any]) -> dict[str, Any]:  # type: ignore[return-value]
        return "not-an-action"

    result = loop.run(
        actions=[],
        planner=_bad_planner,
        budget=RuntimeBudget(max_iterations=5),
        objective="phase9 invalid planner output",
    )

    assert result.status == "failed"
    assert result.error_code == "MODEL_OUTPUT_INVALID"


def test_recursive_loop_records_subcall_metadata_for_planner_delegate() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    planner_calls = {"count": 0}

    def _planner(_: dict[str, Any]) -> dict[str, Any]:
        if planner_calls["count"] == 0:
            planner_calls["count"] += 1
            return {
                "type": "delegate_subcall",
                "objective": "label candidate",
                "actions": [
                    {"type": "synthesize", "output": {"evidence_refs": [], "gaps": []}},
                    {"type": "finalize", "output": {"summary": "child done"}},
                ],
            }
        planner_calls["count"] += 1
        return {"type": "finalize", "output": {"summary": "root done"}}

    result = loop.run(
        actions=[],
        planner=_planner,
        budget=RuntimeBudget(max_iterations=10),
        objective="phase9 delegated planner root",
    )

    assert result.status == "completed"
    assert result.output is not None
    assert result.output.get("summary") == "root done"
    assert result.subcall_metadata


def test_recursive_loop_delegate_subcall_can_use_child_planner_and_context() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    planner_contexts: list[dict[str, Any]] = []
    root_calls = {"count": 0}
    child_calls = {"count": 0}

    def _planner(context: dict[str, Any]) -> dict[str, Any]:
        planner_contexts.append(json.loads(json.dumps(context)))
        objective = str(context.get("objective") or "")
        if objective == "phase91b root":
            if root_calls["count"] == 0:
                root_calls["count"] += 1
                return {
                    "type": "delegate_subcall",
                    "objective": "phase91b child",
                    "use_planner": True,
                    "context": {"candidate_label": "tool_failure"},
                }
            root_calls["count"] += 1
            return {"type": "finalize", "output": {"summary": "root done"}}
        if objective == "phase91b child":
            if child_calls["count"] == 0:
                child_calls["count"] += 1
                return {
                    "type": "tool_call",
                    "tool_name": "list_spans",
                    "args": {"trace_id": "trace-9"},
                }
            child_calls["count"] += 1
            return {
                "type": "finalize",
                "output": {
                    "summary": "child done",
                    "evidence_refs": [
                        {
                            "trace_id": "trace-9",
                            "span_id": "root",
                            "kind": "SPAN",
                            "ref": "root",
                        }
                    ],
                    "gaps": [],
                },
            }
        raise AssertionError(f"Unexpected planner objective: {objective}")

    result = loop.run(
        actions=[],
        planner=_planner,
        budget=RuntimeBudget(max_iterations=10),
        objective="phase91b root",
    )

    assert result.status == "completed"
    assert result.output is not None
    assert result.output.get("summary") == "root done"
    assert root_calls["count"] == 2
    assert child_calls["count"] == 2
    assert result.subcall_metadata
    assert bool(result.subcall_metadata[0].get("planner_driven")) is True
    child_contexts = [
        context
        for context in planner_contexts
        if str(context.get("objective") or "") == "phase91b child"
    ]
    assert child_contexts
    assert all(int(context.get("depth") or -1) == 1 for context in child_contexts)
    assert all(
        isinstance(context.get("delegation_context"), dict)
        and context["delegation_context"].get("candidate_label") == "tool_failure"
        for context in child_contexts
    )


def test_recursive_loop_fails_when_delegate_subcall_requests_child_planner_without_parent_planner() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    result = loop.run(
        actions=[
            {
                "type": "delegate_subcall",
                "objective": "phase91b child",
                "use_planner": True,
            }
        ],
        budget=RuntimeBudget(max_iterations=5),
    )

    assert result.status == "failed"
    assert result.error_code == "MODEL_OUTPUT_INVALID"
    assert result.error_message is not None
    assert "use_planner=true" in result.error_message


def test_recursive_loop_applies_planner_usage_and_enforces_cost_budget() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    def _planner(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": {"type": "tool_call", "tool_name": "list_spans", "args": {"trace_id": "trace-9"}},
            "usage": {"tokens_in": 120, "tokens_out": 18, "cost_usd": 0.06},
        }

    result = loop.run(
        actions=[],
        planner=_planner,
        budget=RuntimeBudget(max_iterations=10, max_cost_usd=0.05),
        objective="phase9 planner usage budget",
    )

    assert result.status == "terminated_budget"
    assert result.budget_reason is not None
    assert "max_cost_usd" in result.budget_reason
    assert result.usage.tokens_in == 120
    assert result.usage.tokens_out == 18
    assert result.usage.cost_usd == pytest.approx(0.06)
    assert result.usage.tool_calls == 0


def test_recursive_loop_exposes_planner_context_summary_across_turns() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)
    planner_contexts: list[dict[str, Any]] = []

    def _planner(context: dict[str, Any]) -> dict[str, Any]:
        planner_contexts.append(json.loads(json.dumps(context)))
        if len(planner_contexts) == 1:
            return {
                "type": "synthesize",
                "output": {
                    "evidence_refs": [
                        {
                            "trace_id": "trace-9",
                            "span_id": "root",
                            "kind": "SPAN",
                            "ref": "root",
                        }
                    ],
                    "gaps": ["need final synthesis"],
                },
            }
        return {"type": "finalize", "output": {"summary": "planner context complete"}}

    result = loop.run(
        actions=[],
        planner=_planner,
        budget=RuntimeBudget(max_iterations=10),
        objective="phase9 planner context summary",
    )

    assert result.status == "completed"
    assert len(planner_contexts) == 2
    assert planner_contexts[0]["draft_output_summary"]["evidence_ref_count"] == 0
    assert planner_contexts[0]["draft_output_summary"]["gap_count"] == 0
    assert planner_contexts[1]["draft_output_summary"]["evidence_ref_count"] == 1
    assert planner_contexts[1]["draft_output_summary"]["gap_count"] == 1
    assert planner_contexts[1]["remaining_budget"]["iterations"] < planner_contexts[0]["remaining_budget"]["iterations"]


def test_recursive_loop_fails_fast_for_unknown_action_type_when_validation_is_permissive() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=_PermissiveGuard())  # type: ignore[arg-type]

    result = loop.run(
        actions=[{"type": "future_action"}],
        budget=RuntimeBudget(max_iterations=3),
        objective="phase9 unknown action guard",
    )

    assert result.status == "failed"
    assert result.error_code == "MODEL_OUTPUT_INVALID"
