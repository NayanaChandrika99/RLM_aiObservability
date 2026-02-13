# ABOUTME: Runs an iterative model-driven REPL loop where each turn emits reasoning and executable Python code.
# ABOUTME: Enforces runtime budgets and sandbox restrictions while collecting trajectory and usage metadata.

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Literal

from investigator.runtime.contracts import RuntimeBudget, RuntimeUsage
from investigator.runtime.llm_client import (
    ModelOutputInvalidError,
    RuntimeModelClient,
    StructuredGenerationRequest,
)
from investigator.runtime.llm_loop import run_structured_generation_loop
from investigator.runtime.prompt_registry import get_prompt_definition
from investigator.runtime.repl_interpreter import ReplInterpreter
from investigator.runtime.sandbox import SandboxViolationError
from investigator.runtime.tool_registry import ToolRegistry


_REPL_PROMPT_ID = "repl_runtime_step_v1"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _build_user_prompt(context: dict[str, Any]) -> str:
    payload = json.dumps(_json_safe(context), ensure_ascii=False, sort_keys=True)
    return (
        "Runtime REPL context JSON:\n"
        f"{payload}\n\n"
        "Return JSON with reasoning and one Python code snippet only."
    )


def _extract_execution_error(output: str) -> str | None:
    for raw_line in str(output or "").splitlines():
        line = raw_line.strip()
        if line.startswith("[Error]"):
            return line.removeprefix("[Error]").strip()
    return None


@dataclass
class ReplLoopResult:
    status: Literal["completed", "terminated_budget", "failed"]
    output: dict[str, Any] | None
    usage: RuntimeUsage
    state_trajectory: list[str] = field(default_factory=list)
    repl_trajectory: list[dict[str, Any]] = field(default_factory=list)
    budget_reason: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class ReplLoop:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        model_client: RuntimeModelClient,
        model_name: str,
        temperature: float | None,
    ) -> None:
        self._tool_registry = tool_registry
        self._model_client = model_client
        self._model_name = model_name
        self._temperature = temperature
        self._prompt = get_prompt_definition(_REPL_PROMPT_ID)

    def _budget_reason(
        self,
        *,
        budget: RuntimeBudget,
        usage: RuntimeUsage,
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
        if usage.llm_subcalls >= budget.max_subcalls:
            return f"max_subcalls reached: subcalls={usage.llm_subcalls} max_subcalls={budget.max_subcalls}"
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

    @staticmethod
    def _build_recovery_submit_output(
        *,
        objective: str,
        input_vars: dict[str, Any],
        error_message: str,
    ) -> dict[str, Any]:
        allowed_labels_raw = input_vars.get("allowed_labels")
        deterministic_hint = str(input_vars.get("deterministic_label_hint") or "").strip()
        if isinstance(allowed_labels_raw, list):
            allowed_labels = [str(item) for item in allowed_labels_raw if str(item)]
            chosen_label = deterministic_hint if deterministic_hint in allowed_labels else ""
            if not chosen_label:
                chosen_label = allowed_labels[0] if allowed_labels else "instruction_failure"
            return {
                "primary_label": chosen_label,
                "summary": (
                    "Deterministic fallback SUBMIT applied after REPL execution error "
                    "near iteration budget exhaustion."
                ),
                "confidence": 0.25,
                "remediation": [
                    "Retry with one focused semantic synthesis step before finalization.",
                    "Tighten generated code to avoid undefined variables and invalid indexing.",
                ],
                "gaps": [str(error_message or "Unknown execution error.")],
            }

        required_evidence_raw = input_vars.get("required_evidence")
        missing_evidence: list[str] = []
        if isinstance(required_evidence_raw, list):
            missing_evidence = [str(item) for item in required_evidence_raw if str(item)]
        elif isinstance(required_evidence_raw, str) and required_evidence_raw.strip():
            missing_evidence = [required_evidence_raw.strip()]

        default_evidence = input_vars.get("default_evidence")
        evidence_refs: list[dict[str, Any]] = []
        if isinstance(default_evidence, dict):
            evidence_refs = [dict(default_evidence)]
        elif isinstance(default_evidence, list):
            evidence_refs = [dict(item) for item in default_evidence if isinstance(item, dict)]

        return {
            "pass_fail": "insufficient_evidence",
            "confidence": 0.25,
            "rationale": (
                "Deterministic fallback SUBMIT applied after REPL execution error near "
                "iteration budget exhaustion."
            ),
            "covered_requirements": [],
            "missing_evidence": missing_evidence,
            "evidence_refs": evidence_refs,
            "gaps": [
                str(error_message or "Unknown execution error."),
                f"Objective: {objective}",
            ],
            "remediation": (
                "Re-run with one stabilized synthesis step and finalization after collecting "
                "the required evidence items."
            ),
        }

    def run(
        self,
        *,
        objective: str,
        input_vars: dict[str, Any],
        pre_filter_context: dict[str, Any] | None = None,
        budget: RuntimeBudget,
        depth: int = 0,
        require_subquery_for_non_trivial: bool = False,
        start_monotonic: float | None = None,
    ) -> ReplLoopResult:
        started = start_monotonic if start_monotonic is not None else time.monotonic()
        usage = RuntimeUsage(iterations=0, depth_reached=depth, tool_calls=0, tokens_in=0, tokens_out=0, cost_usd=0.0, llm_subcalls=0)
        state_trajectory: list[str] = ["initialized", "running", "planning"]
        repl_trajectory: list[dict[str, Any]] = []

        interpreter = ReplInterpreter(
            tool_registry=self._tool_registry,
            model_client=self._model_client,
            model_name=self._model_name,
            temperature=self._temperature,
            max_llm_subcalls=budget.max_subcalls,
            sandbox_timeout_sec=30,
            sandbox_max_output_chars=8192,
        )
        normalized_pre_filter_context: dict[str, Any] = {}
        if isinstance(pre_filter_context, dict):
            normalized_pre_filter_context = {
                str(key): _json_safe(pre_filter_context[key]) for key in pre_filter_context
            }
        runtime_vars = dict(input_vars)
        if normalized_pre_filter_context:
            runtime_vars["pre_filter_context"] = normalized_pre_filter_context
        runtime_vars["available_tools"] = sorted(self._tool_registry.allowed_tools)
        runtime_vars["tool_signatures"] = self._tool_registry.describe_tools()

        while True:
            budget_reason = self._budget_reason(
                budget=budget,
                usage=usage,
                depth=depth,
                start_monotonic=started,
            )
            if budget_reason:
                if usage.iterations > 0 or repl_trajectory:
                    if repl_trajectory:
                        prior_output = str(repl_trajectory[-1].get("output") or "").strip()
                        recovery_line = (
                            "[Recovery] Deterministic fallback SUBMIT applied due to "
                            f"{budget_reason}"
                        )
                        repl_trajectory[-1]["output"] = (
                            f"{prior_output}\n{recovery_line}" if prior_output else recovery_line
                        )
                    state_trajectory.extend(["finalizing", "completed"])
                    return ReplLoopResult(
                        status="completed",
                        output=self._build_recovery_submit_output(
                            objective=objective,
                            input_vars=runtime_vars,
                            error_message=budget_reason,
                        ),
                        usage=usage,
                        state_trajectory=state_trajectory,
                        repl_trajectory=repl_trajectory,
                    )
                state_trajectory.append("terminated_budget")
                return ReplLoopResult(
                    status="terminated_budget",
                    output=None,
                    usage=usage,
                    state_trajectory=state_trajectory,
                    repl_trajectory=repl_trajectory,
                    budget_reason=budget_reason,
                )

            repl_history = [
                {
                    "reasoning": str(step.get("reasoning") or ""),
                    "code": str(step.get("code") or ""),
                    "output": str(step.get("output") or ""),
                }
                for step in repl_trajectory
            ]
            remaining_iterations = max(0, int(budget.max_iterations) - int(usage.iterations))
            remaining_subcalls = max(0, int(budget.max_subcalls) - int(usage.llm_subcalls))
            remaining_tool_calls = max(0, int(budget.max_tool_calls) - int(usage.tool_calls))
            remaining_tokens_total = None
            if budget.max_tokens_total is not None:
                remaining_tokens_total = max(
                    0,
                    int(budget.max_tokens_total) - int(usage.tokens_in + usage.tokens_out),
                )
            remaining_cost_usd = None
            if budget.max_cost_usd is not None:
                remaining_cost_usd = max(0.0, float(budget.max_cost_usd) - float(usage.cost_usd))
            submit_deadline_iterations_remaining = 2
            if require_subquery_for_non_trivial:
                submit_deadline_iterations_remaining = 1
            request = StructuredGenerationRequest(
                model_provider=str(getattr(self._model_client, "model_provider", "openai")),
                model_name=self._model_name,
                temperature=self._temperature,
                system_prompt=self._prompt.prompt_text,
                user_prompt=_build_user_prompt(
                    {
                        "objective": objective,
                        "iteration": usage.iterations + 1,
                        "budget": {
                            "max_iterations": budget.max_iterations,
                            "max_depth": budget.max_depth,
                            "max_tool_calls": budget.max_tool_calls,
                            "max_subcalls": budget.max_subcalls,
                            "max_tokens_total": budget.max_tokens_total,
                            "max_cost_usd": budget.max_cost_usd,
                            "max_wall_time_sec": budget.max_wall_time_sec,
                        },
                        "usage": {
                            "iterations": usage.iterations,
                            "depth_reached": usage.depth_reached,
                            "tool_calls": usage.tool_calls,
                            "llm_subcalls": usage.llm_subcalls,
                            "tokens_in": usage.tokens_in,
                            "tokens_out": usage.tokens_out,
                            "cost_usd": usage.cost_usd,
                        },
                        "remaining": {
                            "iterations": remaining_iterations,
                            "tool_calls": remaining_tool_calls,
                            "subcalls": remaining_subcalls,
                            "tokens_total": remaining_tokens_total,
                            "cost_usd": remaining_cost_usd,
                        },
                        "submit_deadline_iterations_remaining": submit_deadline_iterations_remaining,
                        "allowed_tools": sorted(self._tool_registry.allowed_tools),
                        "tool_signatures": runtime_vars.get("tool_signatures") or {},
                        "pre_filter_context": normalized_pre_filter_context,
                        "variables": sorted(interpreter.variables.keys()),
                        "history": repl_history,
                    }
                ),
                response_schema_name=self._prompt.prompt_id,
                response_schema=self._prompt.response_schema,
            )

            try:
                plan_result = run_structured_generation_loop(
                    client=self._model_client,
                    request=request,
                    max_attempts=2,
                )
            except ModelOutputInvalidError as exc:
                state_trajectory.append("failed")
                return ReplLoopResult(
                    status="failed",
                    output=None,
                    usage=usage,
                    state_trajectory=state_trajectory,
                    repl_trajectory=repl_trajectory,
                    error_code="MODEL_OUTPUT_INVALID",
                    error_message=str(exc),
                )

            usage.tokens_in += int(plan_result.usage.tokens_in)
            usage.tokens_out += int(plan_result.usage.tokens_out)
            usage.cost_usd += float(plan_result.usage.cost_usd)

            planning_budget_reason = self._budget_reason(
                budget=budget,
                usage=usage,
                depth=depth,
                start_monotonic=started,
            )
            if planning_budget_reason:
                if usage.iterations > 0 or repl_trajectory:
                    if repl_trajectory:
                        prior_output = str(repl_trajectory[-1].get("output") or "").strip()
                        recovery_line = (
                            "[Recovery] Deterministic fallback SUBMIT applied due to "
                            f"{planning_budget_reason}"
                        )
                        repl_trajectory[-1]["output"] = (
                            f"{prior_output}\n{recovery_line}" if prior_output else recovery_line
                        )
                    state_trajectory.extend(["finalizing", "completed"])
                    return ReplLoopResult(
                        status="completed",
                        output=self._build_recovery_submit_output(
                            objective=objective,
                            input_vars=runtime_vars,
                            error_message=planning_budget_reason,
                        ),
                        usage=usage,
                        state_trajectory=state_trajectory,
                        repl_trajectory=repl_trajectory,
                    )
                state_trajectory.append("terminated_budget")
                return ReplLoopResult(
                    status="terminated_budget",
                    output=None,
                    usage=usage,
                    state_trajectory=state_trajectory,
                    repl_trajectory=repl_trajectory,
                    budget_reason=planning_budget_reason,
                )

            reasoning = str(plan_result.output.get("reasoning") or "").strip()
            code = str(plan_result.output.get("code") or "")
            if not code.strip():
                state_trajectory.append("failed")
                return ReplLoopResult(
                    status="failed",
                    output=None,
                    usage=usage,
                    state_trajectory=state_trajectory,
                    repl_trajectory=repl_trajectory,
                    error_code="MODEL_OUTPUT_INVALID",
                    error_message="REPL step output must include non-empty code.",
                )

            state_trajectory.append("acting")
            try:
                exec_result = interpreter.execute(code=code, input_vars=runtime_vars)
            except SandboxViolationError as exc:
                state_trajectory.append("failed")
                return ReplLoopResult(
                    status="failed",
                    output=None,
                    usage=usage,
                    state_trajectory=state_trajectory,
                    repl_trajectory=repl_trajectory,
                    error_code="SANDBOX_VIOLATION",
                    error_message=str(exc),
                )
            except ModelOutputInvalidError as exc:
                state_trajectory.append("failed")
                return ReplLoopResult(
                    status="failed",
                    output=None,
                    usage=usage,
                    state_trajectory=state_trajectory,
                    repl_trajectory=repl_trajectory,
                    error_code="MODEL_OUTPUT_INVALID",
                    error_message=str(exc),
                )

            usage.iterations += 1
            usage.tool_calls += int(exec_result.usage.tool_calls)
            usage.llm_subcalls += int(exec_result.usage.llm_subcalls)
            usage.tokens_in += int(exec_result.usage.tokens_in)
            usage.tokens_out += int(exec_result.usage.tokens_out)
            usage.cost_usd += float(exec_result.usage.cost_usd)

            repl_trajectory.append(
                {
                    "reasoning": reasoning,
                    "code": code,
                    "output": exec_result.stdout,
                    "tool_trace": [dict(item) for item in exec_result.tool_trace],
                    "subquery_trace": [dict(item) for item in exec_result.subquery_trace],
                }
            )

            execution_error = _extract_execution_error(exec_result.stdout)
            remaining_iterations_after_step = max(0, int(budget.max_iterations) - int(usage.iterations))
            if (
                exec_result.submitted_output is None
                and remaining_iterations_after_step <= 0
            ):
                if repl_trajectory:
                    prior_output = str(repl_trajectory[-1].get("output") or "").strip()
                    recovery_reason = (
                        execution_error
                        if execution_error
                        else "submit deadline reached without SUBMIT."
                    )
                    recovery_line = (
                        "[Recovery] Deterministic fallback SUBMIT applied due to "
                        f"{recovery_reason}"
                    )
                    repl_trajectory[-1]["output"] = (
                        f"{prior_output}\n{recovery_line}" if prior_output else recovery_line
                    )
                state_trajectory.extend(["finalizing", "completed"])
                return ReplLoopResult(
                    status="completed",
                    output=self._build_recovery_submit_output(
                        objective=objective,
                        input_vars=runtime_vars,
                        error_message=(
                            execution_error
                            if execution_error
                            else "submit deadline reached without SUBMIT."
                        ),
                    ),
                    usage=usage,
                    state_trajectory=state_trajectory,
                    repl_trajectory=repl_trajectory,
                )

            if isinstance(exec_result.submitted_output, dict):
                if require_subquery_for_non_trivial and usage.llm_subcalls <= 0:
                    guardrail_message = (
                        "SUBMIT blocked: non-trivial objective requires "
                        "llm_query/llm_query_batched before SUBMIT."
                    )
                    if repl_trajectory:
                        prior_output = str(repl_trajectory[-1].get("output") or "").strip()
                        if prior_output:
                            repl_trajectory[-1]["output"] = (
                                f"{prior_output}\n[Guardrail] {guardrail_message}"
                            )
                        else:
                            repl_trajectory[-1]["output"] = f"[Guardrail] {guardrail_message}"
                    state_trajectory.append("planning")
                    continue
                state_trajectory.extend(["finalizing", "completed"])
                return ReplLoopResult(
                    status="completed",
                    output=dict(exec_result.submitted_output),
                    usage=usage,
                    state_trajectory=state_trajectory,
                    repl_trajectory=repl_trajectory,
                )

            state_trajectory.append("planning")
