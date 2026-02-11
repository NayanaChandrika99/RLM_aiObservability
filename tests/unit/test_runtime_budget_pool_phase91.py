# ABOUTME: Validates Phase 9.1A shared runtime budget pooling and fair sibling budget allocation.
# ABOUTME: Ensures pooled limits are enforced across multiple loop runs and delegated recursive children.

from __future__ import annotations

from typing import Any

from investigator.runtime.budget_pool import RuntimeBudgetPool
from investigator.runtime.contracts import RuntimeBudget
from investigator.runtime.recursive_loop import RecursiveLoop
from investigator.runtime.sandbox import SandboxGuard
from investigator.runtime.tool_registry import ToolRegistry


class _InspectionAPI:
    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [{"trace_id": trace_id, "span_id": "root"}]


def test_runtime_budget_pool_allocates_fair_sibling_slices() -> None:
    pool = RuntimeBudgetPool(
        budget=RuntimeBudget(
            max_iterations=5,
            max_tool_calls=10,
            max_subcalls=6,
            max_tokens_total=100,
            max_cost_usd=1.0,
        )
    )

    first = pool.allocate_run_budget(sibling_count=2)
    assert first.max_iterations == 2
    assert first.max_tool_calls == 5
    assert first.max_subcalls == 3
    assert first.max_tokens_total == 50
    assert first.max_cost_usd is not None
    assert first.max_cost_usd == 0.5

    pool.consume(iterations=2, tool_calls=1, tokens_in=20, tokens_out=10, cost_usd=0.2, subcalls=1)

    second = pool.allocate_run_budget(sibling_count=1)
    assert second.max_iterations == 3
    assert second.max_tool_calls == 9
    assert second.max_subcalls == 5
    assert second.max_tokens_total == 70
    assert second.max_cost_usd is not None
    assert second.max_cost_usd == 0.8


def test_recursive_loop_uses_shared_pool_to_enforce_global_iteration_cap() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)
    pool = RuntimeBudgetPool(budget=RuntimeBudget(max_iterations=3, max_tool_calls=10))

    first_result = loop.run(
        actions=[
            {"type": "synthesize", "output": {"summary": "first"}},
            {"type": "finalize", "output": {"summary": "first done"}},
        ],
        budget=pool.allocate_run_budget(sibling_count=1),
        budget_pool=pool,
        objective="phase91-first",
    )
    assert first_result.status == "completed"

    second_result = loop.run(
        actions=[
            {"type": "synthesize", "output": {"summary": "second"}},
            {"type": "finalize", "output": {"summary": "second done"}},
        ],
        budget=pool.allocate_run_budget(sibling_count=1),
        budget_pool=pool,
        objective="phase91-second",
    )
    assert second_result.status == "terminated_budget"
    assert second_result.budget_reason is not None
    assert "max_iterations" in second_result.budget_reason


def test_recursive_loop_child_delegation_draws_from_shared_pool() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)
    pool = RuntimeBudgetPool(budget=RuntimeBudget(max_iterations=4, max_subcalls=2))

    result = loop.run(
        actions=[
            {
                "type": "delegate_subcall",
                "objective": "first child",
                "actions": [
                    {"type": "synthesize", "output": {"summary": "child-one"}},
                    {"type": "finalize", "output": {"summary": "child-one-done"}},
                ],
            },
            {
                "type": "delegate_subcall",
                "objective": "second child",
                "actions": [
                    {"type": "synthesize", "output": {"summary": "child-two"}},
                    {"type": "finalize", "output": {"summary": "child-two-done"}},
                ],
            },
            {"type": "finalize", "output": {"summary": "root"}},
        ],
        budget=pool.allocate_run_budget(sibling_count=1),
        budget_pool=pool,
        objective="phase91-delegate",
    )

    assert result.status == "terminated_budget"
    assert result.budget_reason is not None
    assert "max_iterations" in result.budget_reason
