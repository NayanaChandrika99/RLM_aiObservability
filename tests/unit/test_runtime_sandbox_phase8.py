# ABOUTME: Validates Phase 8B sandbox guard enforcement for recursive runtime actions.
# ABOUTME: Ensures unknown actions and forbidden tool calls fail with sandbox violations.

from __future__ import annotations

import pytest

from investigator.runtime.sandbox import SandboxGuard, SandboxViolationError


def test_sandbox_rejects_unknown_action_type() -> None:
    guard = SandboxGuard(allowed_tools={"list_spans"})

    with pytest.raises(SandboxViolationError):
        guard.validate_action({"type": "exec_code", "code": "print('hi')"})


def test_sandbox_rejects_forbidden_tool_call() -> None:
    guard = SandboxGuard(allowed_tools={"list_spans"})

    with pytest.raises(SandboxViolationError):
        guard.validate_action(
            {
                "type": "tool_call",
                "tool_name": "delete_trace",
                "args": {"trace_id": "trace-1"},
            }
        )


def test_sandbox_rejects_non_mapping_tool_args() -> None:
    guard = SandboxGuard(allowed_tools={"list_spans"})

    with pytest.raises(SandboxViolationError):
        guard.validate_action(
            {
                "type": "tool_call",
                "tool_name": "list_spans",
                "args": ["not", "a", "mapping"],
            }
        )


def test_sandbox_allows_delegate_subcall_with_use_planner_and_context() -> None:
    guard = SandboxGuard(allowed_tools={"list_spans"})

    guard.validate_action(
        {
            "type": "delegate_subcall",
            "objective": "child investigation",
            "use_planner": True,
            "context": {"candidate_label": "tool_failure"},
        }
    )


def test_sandbox_rejects_delegate_subcall_without_actions_when_not_planner_driven() -> None:
    guard = SandboxGuard(allowed_tools={"list_spans"})

    with pytest.raises(SandboxViolationError):
        guard.validate_action(
            {
                "type": "delegate_subcall",
                "objective": "child investigation",
            }
        )
