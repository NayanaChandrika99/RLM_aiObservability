# ABOUTME: Enforces sandbox rules for recursive runtime actions before execution.
# ABOUTME: Blocks unknown actions, forbidden tools, and unsafe argument payload shapes.

from __future__ import annotations

from typing import Any


class SandboxViolationError(RuntimeError):
    pass


ALLOWED_ACTION_TYPES = {"tool_call", "delegate_subcall", "synthesize", "finalize"}


def _is_json_safe(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_safe(item) for key, item in value.items())
    return False


class SandboxGuard:
    def __init__(self, *, allowed_tools: set[str]) -> None:
        self._allowed_tools = set(allowed_tools)

    @property
    def allowed_tools(self) -> set[str]:
        return set(self._allowed_tools)

    def validate_action(self, action: dict[str, Any]) -> None:
        if not isinstance(action, dict):
            raise SandboxViolationError("Action must be an object.")
        action_type = str(action.get("type") or "")
        if action_type not in ALLOWED_ACTION_TYPES:
            raise SandboxViolationError(f"Unknown action type: {action_type or '<empty>'}.")

        if action_type == "tool_call":
            tool_name = str(action.get("tool_name") or "")
            if not tool_name:
                raise SandboxViolationError("tool_call requires non-empty tool_name.")
            if tool_name not in self._allowed_tools:
                raise SandboxViolationError(f"Tool not allowlisted: {tool_name}.")
            args = action.get("args", {})
            if not isinstance(args, dict):
                raise SandboxViolationError("tool_call args must be an object.")
            if not _is_json_safe(args):
                raise SandboxViolationError("tool_call args contain unsupported value types.")
            return

        if action_type == "delegate_subcall":
            objective = str(action.get("objective") or "")
            if not objective:
                raise SandboxViolationError("delegate_subcall requires objective.")
            use_planner = action.get("use_planner", False)
            if not isinstance(use_planner, bool):
                raise SandboxViolationError("delegate_subcall use_planner must be boolean when provided.")
            actions = action.get("actions")
            if actions is None and not use_planner:
                raise SandboxViolationError(
                    "delegate_subcall requires actions list when use_planner is false."
                )
            if actions is not None:
                if not isinstance(actions, list):
                    raise SandboxViolationError("delegate_subcall actions must be a list when provided.")
                for nested in actions:
                    if not isinstance(nested, dict):
                        raise SandboxViolationError("delegate_subcall actions must contain action objects.")
            context = action.get("context")
            if context is not None and not isinstance(context, dict):
                raise SandboxViolationError("delegate_subcall context must be an object when provided.")
            if context is not None and not _is_json_safe(context):
                raise SandboxViolationError("delegate_subcall context contains unsupported value types.")
            return

        if action_type in {"synthesize", "finalize"}:
            output = action.get("output")
            if output is not None and not isinstance(output, dict):
                raise SandboxViolationError(f"{action_type} output must be an object when provided.")
            if output is not None and not _is_json_safe(output):
                raise SandboxViolationError(f"{action_type} output contains unsupported value types.")
            return
