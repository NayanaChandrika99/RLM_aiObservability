# ABOUTME: Executes typed recursive runtime actions with bounded budgets and explicit state transitions.
# ABOUTME: Produces deterministic loop metadata for budget termination, sandbox failures, and subcall tracking.

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Literal
from uuid import uuid4

from investigator.runtime.contracts import RuntimeBudget, RuntimeUsage, utc_now_rfc3339
from investigator.runtime.sandbox import SandboxGuard, SandboxViolationError
from investigator.runtime.tool_registry import ToolRegistry


LoopState = Literal[
    "initialized",
    "running",
    "planning",
    "acting",
    "delegating",
    "finalizing",
    "completed",
    "terminated_budget",
    "partial",
    "failed",
]


_ALLOWED_TRANSITIONS: dict[LoopState, set[LoopState]] = {
    "initialized": {"running"},
    "running": {"planning", "terminated_budget", "failed"},
    "planning": {"acting", "delegating", "finalizing", "terminated_budget", "failed"},
    "acting": {"planning", "terminated_budget", "failed"},
    "delegating": {"planning", "terminated_budget", "failed"},
    "finalizing": {"completed", "failed"},
    "completed": set(),
    "terminated_budget": {"partial", "failed"},
    "partial": set(),
    "failed": set(),
}


class StateTransitionError(RuntimeError):
    pass


class RuntimeStateMachine:
    def __init__(self) -> None:
        self._state: LoopState = "initialized"
        self._trajectory: list[LoopState] = ["initialized"]

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def trajectory(self) -> list[LoopState]:
        return list(self._trajectory)

    def transition(self, next_state: LoopState) -> None:
        allowed = _ALLOWED_TRANSITIONS[self._state]
        if next_state not in allowed:
            raise StateTransitionError(f"Invalid transition: {self._state} -> {next_state}.")
        self._state = next_state
        self._trajectory.append(next_state)


def _parse_non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer >= 0.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be an integer >= 0.")
    return parsed


def _parse_non_negative_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number >= 0.")
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be a number >= 0.")
    return parsed


def _count_list_items(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def _draft_output_summary(draft_output: dict[str, Any]) -> dict[str, Any]:
    tool_calls = draft_output.get("tool_calls")
    latest_tool_call: dict[str, Any] | None = None
    if isinstance(tool_calls, list) and tool_calls:
        last = tool_calls[-1]
        if isinstance(last, dict):
            latest_tool_call = {
                "tool_name": str(last.get("tool_name") or ""),
                "args_hash": str(last.get("args_hash") or ""),
                "response_hash": str(last.get("response_hash") or ""),
            }
    return {
        "tool_call_count": _count_list_items(tool_calls),
        "evidence_ref_count": _count_list_items(draft_output.get("evidence_refs")),
        "gap_count": _count_list_items(draft_output.get("gaps")),
        "has_summary": bool(str(draft_output.get("summary") or "").strip()),
        "latest_tool_call": latest_tool_call,
    }


@dataclass
class SubcallResult:
    summary: str
    evidence_refs: list[dict[str, Any]]
    gaps: list[str]
    status: Literal["succeeded", "failed", "terminated_budget"]


@dataclass
class RecursiveLoopResult:
    status: Literal["completed", "terminated_budget", "failed"]
    output: dict[str, Any] | None
    usage: RuntimeUsage
    subcall_metadata: list[dict[str, Any]] = field(default_factory=list)
    state_trajectory: list[str] = field(default_factory=list)
    budget_reason: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class RecursiveLoop:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        sandbox_guard: SandboxGuard,
    ) -> None:
        self._tool_registry = tool_registry
        self._sandbox_guard = sandbox_guard

    def _budget_reason(
        self,
        *,
        budget: RuntimeBudget,
        usage: RuntimeUsage,
        subcall_count: int,
        depth: int,
        start_monotonic: float,
    ) -> str | None:
        if depth > budget.max_depth:
            return f"max_depth reached: depth={depth} max_depth={budget.max_depth}"
        if usage.iterations >= budget.max_iterations:
            return (
                f"max_iterations reached: iterations={usage.iterations} "
                f"max_iterations={budget.max_iterations}"
            )
        if usage.tool_calls >= budget.max_tool_calls:
            return f"max_tool_calls reached: tool_calls={usage.tool_calls} max_tool_calls={budget.max_tool_calls}"
        if subcall_count >= budget.max_subcalls:
            return f"max_subcalls reached: subcalls={subcall_count} max_subcalls={budget.max_subcalls}"
        if budget.max_tokens_total is not None:
            tokens_total = usage.tokens_in + usage.tokens_out
            if tokens_total >= budget.max_tokens_total:
                return (
                    f"max_tokens_total reached: tokens_total={tokens_total} "
                    f"max_tokens_total={budget.max_tokens_total}"
                )
        if budget.max_cost_usd is not None and usage.cost_usd >= budget.max_cost_usd:
            return (
                f"max_cost_usd reached: cost_usd={usage.cost_usd:.6f} "
                f"max_cost_usd={budget.max_cost_usd}"
            )
        elapsed = time.monotonic() - start_monotonic
        if elapsed >= float(budget.max_wall_time_sec):
            return (
                f"max_wall_time_sec reached: elapsed={elapsed:.3f} "
                f"max_wall_time_sec={budget.max_wall_time_sec}"
            )
        return None

    def run(
        self,
        *,
        actions: list[dict[str, Any]],
        budget: RuntimeBudget,
        planner: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        depth: int = 0,
        objective: str = "root",
        parent_call_id: str = "root",
        input_ref_hash: str = "",
        start_monotonic: float | None = None,
    ) -> RecursiveLoopResult:
        started = start_monotonic if start_monotonic is not None else time.monotonic()
        usage = RuntimeUsage(iterations=0, depth_reached=depth, tool_calls=0, tokens_in=0, tokens_out=0, cost_usd=0.0)
        subcall_metadata: list[dict[str, Any]] = []
        draft_output: dict[str, Any] = {}
        machine = RuntimeStateMachine()
        machine.transition("running")
        machine.transition("planning")
        queue = [dict(action) for action in actions]

        while queue or planner is not None:
            reason = self._budget_reason(
                budget=budget,
                usage=usage,
                subcall_count=len(subcall_metadata),
                depth=depth,
                start_monotonic=started,
            )
            if reason:
                machine.transition("terminated_budget")
                return RecursiveLoopResult(
                    status="terminated_budget",
                    output=draft_output or None,
                    usage=usage,
                    subcall_metadata=subcall_metadata,
                    state_trajectory=machine.trajectory,
                    budget_reason=reason,
                )

            if not queue and planner is not None:
                planner_context = {
                    "objective": objective,
                    "parent_call_id": parent_call_id,
                    "depth": depth,
                    "input_ref_hash": input_ref_hash,
                    "state": machine.state,
                    "allowed_tools": sorted(self._tool_registry.allowed_tools),
                    "budget": {
                        "max_iterations": budget.max_iterations,
                        "max_depth": budget.max_depth,
                        "max_tool_calls": budget.max_tool_calls,
                        "max_subcalls": budget.max_subcalls,
                        "max_tokens_total": budget.max_tokens_total,
                        "max_cost_usd": budget.max_cost_usd,
                        "sampling_seed": budget.sampling_seed,
                        "max_wall_time_sec": budget.max_wall_time_sec,
                    },
                    "usage": {
                        "iterations": usage.iterations,
                        "depth_reached": usage.depth_reached,
                        "tool_calls": usage.tool_calls,
                        "tokens_in": usage.tokens_in,
                        "tokens_out": usage.tokens_out,
                        "cost_usd": usage.cost_usd,
                    },
                    "remaining_budget": {
                        "iterations": max(0, budget.max_iterations - usage.iterations),
                        "depth": max(0, budget.max_depth - depth),
                        "tool_calls": max(0, budget.max_tool_calls - usage.tool_calls),
                        "subcalls": max(0, budget.max_subcalls - len(subcall_metadata)),
                        "tokens_total": (
                            None
                            if budget.max_tokens_total is None
                            else max(0, budget.max_tokens_total - (usage.tokens_in + usage.tokens_out))
                        ),
                        "cost_usd": (
                            None
                            if budget.max_cost_usd is None
                            else max(0.0, budget.max_cost_usd - usage.cost_usd)
                        ),
                    },
                    "subcall_count": len(subcall_metadata),
                    "draft_output": dict(draft_output),
                    "draft_output_summary": _draft_output_summary(draft_output),
                }
                try:
                    planned_action = planner(planner_context)
                except Exception as exc:
                    machine.transition("failed")
                    return RecursiveLoopResult(
                        status="failed",
                        output=draft_output or None,
                        usage=usage,
                        subcall_metadata=subcall_metadata,
                        state_trajectory=machine.trajectory,
                        error_code="MODEL_OUTPUT_INVALID",
                        error_message=f"planner action generation failed: {exc}",
                    )
                planner_usage: dict[str, Any] | None = None
                if isinstance(planned_action, dict) and "action" in planned_action:
                    envelope_usage = planned_action.get("usage")
                    if envelope_usage is not None and not isinstance(envelope_usage, dict):
                        machine.transition("failed")
                        return RecursiveLoopResult(
                            status="failed",
                            output=draft_output or None,
                            usage=usage,
                            subcall_metadata=subcall_metadata,
                            state_trajectory=machine.trajectory,
                            error_code="MODEL_OUTPUT_INVALID",
                            error_message="planner usage must be an object when provided.",
                        )
                    planned_action = planned_action.get("action")
                    planner_usage = envelope_usage
                if planner_usage is not None:
                    try:
                        usage.tokens_in += _parse_non_negative_int(
                            planner_usage.get("tokens_in", 0),
                            field_name="planner usage tokens_in",
                        )
                        usage.tokens_out += _parse_non_negative_int(
                            planner_usage.get("tokens_out", 0),
                            field_name="planner usage tokens_out",
                        )
                        usage.cost_usd += _parse_non_negative_float(
                            planner_usage.get("cost_usd", 0.0),
                            field_name="planner usage cost_usd",
                        )
                    except (TypeError, ValueError) as exc:
                        machine.transition("failed")
                        return RecursiveLoopResult(
                            status="failed",
                            output=draft_output or None,
                            usage=usage,
                            subcall_metadata=subcall_metadata,
                            state_trajectory=machine.trajectory,
                            error_code="MODEL_OUTPUT_INVALID",
                            error_message=f"planner usage invalid: {exc}",
                        )
                    reason_after_planning = self._budget_reason(
                        budget=budget,
                        usage=usage,
                        subcall_count=len(subcall_metadata),
                        depth=depth,
                        start_monotonic=started,
                    )
                    if reason_after_planning:
                        machine.transition("terminated_budget")
                        return RecursiveLoopResult(
                            status="terminated_budget",
                            output=draft_output or None,
                            usage=usage,
                            subcall_metadata=subcall_metadata,
                            state_trajectory=machine.trajectory,
                            budget_reason=reason_after_planning,
                        )
                if planned_action is None:
                    machine.transition("finalizing")
                    machine.transition("completed")
                    return RecursiveLoopResult(
                        status="completed",
                        output=draft_output or {},
                        usage=usage,
                        subcall_metadata=subcall_metadata,
                        state_trajectory=machine.trajectory,
                    )
                if not isinstance(planned_action, dict):
                    machine.transition("failed")
                    return RecursiveLoopResult(
                        status="failed",
                        output=draft_output or None,
                        usage=usage,
                        subcall_metadata=subcall_metadata,
                        state_trajectory=machine.trajectory,
                        error_code="MODEL_OUTPUT_INVALID",
                        error_message="planner must return an action object.",
                    )
                queue.append(dict(planned_action))

            action = queue.pop(0)
            try:
                self._sandbox_guard.validate_action(action)
            except SandboxViolationError as exc:
                machine.transition("failed")
                return RecursiveLoopResult(
                    status="failed",
                    output=draft_output or None,
                    usage=usage,
                    subcall_metadata=subcall_metadata,
                    state_trajectory=machine.trajectory,
                    error_code="SANDBOX_VIOLATION",
                    error_message=str(exc),
                )

            action_type = str(action.get("type") or "")
            if action_type == "tool_call":
                machine.transition("acting")
                usage.iterations += 1
                usage.tool_calls += 1
                call_result = self._tool_registry.call(
                    str(action.get("tool_name")),
                    dict(action.get("args") or {}),
                )
                draft_output.setdefault("tool_calls", []).append(call_result)
                machine.transition("planning")
                continue

            if action_type == "synthesize":
                machine.transition("acting")
                usage.iterations += 1
                output_patch = action.get("output")
                if isinstance(output_patch, dict):
                    draft_output.update(output_patch)
                machine.transition("planning")
                continue

            if action_type == "delegate_subcall":
                machine.transition("delegating")
                usage.iterations += 1
                sub_depth = depth + 1
                subcall_reason = self._budget_reason(
                    budget=budget,
                    usage=usage,
                    subcall_count=len(subcall_metadata),
                    depth=sub_depth,
                    start_monotonic=started,
                )
                if subcall_reason:
                    machine.transition("terminated_budget")
                    return RecursiveLoopResult(
                        status="terminated_budget",
                        output=draft_output or None,
                        usage=usage,
                        subcall_metadata=subcall_metadata,
                        state_trajectory=machine.trajectory,
                        budget_reason=subcall_reason,
                    )

                subcall_actions = action.get("actions") or []
                subcall_objective = str(action.get("objective") or f"subcall-{uuid4()}")
                subcall_id = str(uuid4())
                sub_started_at = utc_now_rfc3339()
                child_result = self.run(
                    actions=[dict(item) for item in subcall_actions if isinstance(item, dict)],
                    budget=budget,
                    depth=sub_depth,
                    objective=subcall_objective,
                    parent_call_id=subcall_id,
                    input_ref_hash=input_ref_hash,
                    start_monotonic=started,
                )
                sub_completed_at = utc_now_rfc3339()
                usage.iterations += child_result.usage.iterations
                usage.depth_reached = max(usage.depth_reached, child_result.usage.depth_reached)
                usage.tool_calls += child_result.usage.tool_calls
                usage.tokens_in += child_result.usage.tokens_in
                usage.tokens_out += child_result.usage.tokens_out
                usage.cost_usd += child_result.usage.cost_usd

                child_status = "succeeded"
                if child_result.status == "failed":
                    child_status = "failed"
                elif child_result.status == "terminated_budget":
                    child_status = "terminated_budget"
                subcall_metadata.append(
                    {
                        "parent_call_id": parent_call_id,
                        "call_id": subcall_id,
                        "depth": sub_depth,
                        "objective": subcall_objective,
                        "input_ref_hash": input_ref_hash,
                        "started_at": sub_started_at,
                        "completed_at": sub_completed_at,
                        "status": child_status,
                    }
                )
                subcall_metadata.extend(child_result.subcall_metadata)

                if isinstance(child_result.output, dict):
                    child_evidence = child_result.output.get("evidence_refs")
                    child_gaps = child_result.output.get("gaps")
                    if isinstance(child_evidence, list):
                        merged = draft_output.setdefault("evidence_refs", [])
                        if isinstance(merged, list):
                            seen = {(str(item.get("kind")), str(item.get("ref"))) for item in merged if isinstance(item, dict)}
                            for evidence in child_evidence:
                                if not isinstance(evidence, dict):
                                    continue
                                key = (str(evidence.get("kind")), str(evidence.get("ref")))
                                if key in seen:
                                    continue
                                merged.append(evidence)
                                seen.add(key)
                    if isinstance(child_gaps, list):
                        merged_gaps = draft_output.setdefault("gaps", [])
                        if isinstance(merged_gaps, list):
                            for gap in child_gaps:
                                if not isinstance(gap, str) or not gap.strip():
                                    continue
                                merged_gaps.append(f"subcall:{subcall_id}:{gap}")

                if child_result.status == "terminated_budget":
                    machine.transition("terminated_budget")
                    return RecursiveLoopResult(
                        status="terminated_budget",
                        output=draft_output or None,
                        usage=usage,
                        subcall_metadata=subcall_metadata,
                        state_trajectory=machine.trajectory,
                        budget_reason=child_result.budget_reason or "subcall reached runtime budget",
                    )
                if child_result.status == "failed" and bool(action.get("fatal", False)):
                    machine.transition("failed")
                    return RecursiveLoopResult(
                        status="failed",
                        output=draft_output or None,
                        usage=usage,
                        subcall_metadata=subcall_metadata,
                        state_trajectory=machine.trajectory,
                        error_code=child_result.error_code or "UNEXPECTED_RUNTIME_ERROR",
                        error_message=child_result.error_message or "subcall failed",
                    )
                machine.transition("planning")
                continue

            if action_type == "finalize":
                machine.transition("finalizing")
                usage.iterations += 1
                output_payload = action.get("output")
                if isinstance(output_payload, dict):
                    draft_output.update(output_payload)
                machine.transition("completed")
                return RecursiveLoopResult(
                    status="completed",
                    output=draft_output or {},
                    usage=usage,
                    subcall_metadata=subcall_metadata,
                    state_trajectory=machine.trajectory,
                )

            machine.transition("failed")
            return RecursiveLoopResult(
                status="failed",
                output=draft_output or None,
                usage=usage,
                subcall_metadata=subcall_metadata,
                state_trajectory=machine.trajectory,
                error_code="MODEL_OUTPUT_INVALID",
                error_message=f"Unsupported action type at runtime: {action_type or '<empty>'}.",
            )

        machine.transition("finalizing")
        machine.transition("completed")
        return RecursiveLoopResult(
            status="completed",
            output=draft_output or {},
            usage=usage,
            subcall_metadata=subcall_metadata,
            state_trajectory=machine.trajectory,
        )
