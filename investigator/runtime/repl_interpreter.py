# ABOUTME: Executes model-generated Python snippets in a restricted REPL with read-only tool and sub-LLM helpers.
# ABOUTME: Captures stdout, helper usage accounting, and SUBMIT finalization payloads for bounded runtime loops.

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
import math
import re
from typing import Any

from investigator.runtime.llm_client import (
    ModelOutputInvalidError,
    RuntimeModelClient,
    StructuredGenerationRequest,
)
from investigator.runtime.llm_loop import run_structured_generation_loop
from investigator.runtime.sandbox import SandboxViolationError
from investigator.runtime.tool_registry import ToolRegistry


_REPL_SUBQUERY_SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string", "minLength": 1}},
    "additionalProperties": False,
}


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


def _clip_text(text: str, *, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...(truncated)"


def _preview_payload(value: Any, *, max_chars: int = 2000) -> str:
    safe_value = _json_safe(value)
    try:
        serialized = json.dumps(safe_value, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        serialized = str(safe_value)
    return _clip_text(serialized, max_chars=max_chars)


def _blocked_import(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    module_name = str(args[0]) if args else ""
    if module_name in {"json", "re", "math"}:
        return {"json": json, "re": re, "math": math}[module_name]
    del kwargs
    raise RuntimeError("Import statements are blocked in REPL runtime.")


def _blocked_callable(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    del args, kwargs
    raise SandboxViolationError("This callable is blocked in REPL runtime.")


_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "print": print,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "RuntimeError": RuntimeError,
    "__import__": _blocked_import,
    "open": _blocked_callable,
    "eval": _blocked_callable,
    "exec": _blocked_callable,
    "compile": _blocked_callable,
    "input": _blocked_callable,
}


class _SubmitSignal(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("SUBMIT called")
        self.payload = payload


@dataclass
class ReplExecutionUsage:
    tool_calls: int = 0
    llm_subcalls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class ReplExecutionResult:
    stdout: str
    submitted_output: dict[str, Any] | None
    usage: ReplExecutionUsage
    tool_trace: list[dict[str, Any]]
    subquery_trace: list[dict[str, Any]]


class ReplInterpreter:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        model_client: RuntimeModelClient,
        model_name: str,
        temperature: float | None,
        max_llm_subcalls: int,
    ) -> None:
        self._tool_registry = tool_registry
        self._model_client = model_client
        self._model_name = model_name
        self._temperature = temperature
        self._max_llm_subcalls = max(0, int(max_llm_subcalls))
        self._locals: dict[str, Any] = {}
        self._subcall_count = 0

    @property
    def variables(self) -> dict[str, Any]:
        return dict(self._locals)

    def execute(
        self,
        *,
        code: str,
        input_vars: dict[str, Any],
    ) -> ReplExecutionResult:
        if not isinstance(code, str) or not code.strip():
            raise ModelOutputInvalidError("REPL code must be a non-empty string.")

        usage = ReplExecutionUsage()
        tool_trace: list[dict[str, Any]] = []
        subquery_trace: list[dict[str, Any]] = []
        for key, value in input_vars.items():
            if key not in self._locals:
                self._locals[key] = value

        def _call_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise SandboxViolationError("call_tool requires a non-empty tool name.")
            usage.tool_calls += 1
            call_id = int(usage.tool_calls)
            tool_name_str = str(tool_name)
            normalized_kwargs = dict(kwargs)
            if "control" in normalized_kwargs and "control_id" not in normalized_kwargs:
                normalized_kwargs["control_id"] = normalized_kwargs.pop("control")
            control_id_value = normalized_kwargs.get("control_id")
            if isinstance(control_id_value, dict):
                nested_control_id = control_id_value.get("control_id")
                if isinstance(nested_control_id, str) and nested_control_id.strip():
                    normalized_kwargs["control_id"] = nested_control_id
            if tool_name_str in {"get_control", "required_evidence", "list_controls"}:
                controls_version = self._locals.get("controls_version")
                if (
                    "controls_version" not in normalized_kwargs
                    and isinstance(controls_version, str)
                    and controls_version.strip()
                ):
                    normalized_kwargs["controls_version"] = controls_version
            trace_entry: dict[str, Any] = {
                "call_id": call_id,
                "tool_name": tool_name_str,
                "args": _json_safe(normalized_kwargs),
                "status": "pending",
            }
            try:
                result = self._tool_registry.call(tool_name_str, dict(normalized_kwargs))
                trace_entry["status"] = "ok"
                trace_entry["result_preview"] = _preview_payload(result)
                if isinstance(result, dict):
                    trace_entry["result_keys"] = sorted([str(key) for key in result.keys()])
                tool_trace.append(trace_entry)
                return result
            except SandboxViolationError as exc:
                trace_entry["status"] = "error"
                trace_entry["error"] = str(exc)
                tool_trace.append(trace_entry)
                raise RuntimeError(f"[ToolError] {exc}") from exc

        def _llm_query(prompt: str) -> str:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ModelOutputInvalidError("llm_query prompt must be a non-empty string.")
            if self._subcall_count >= self._max_llm_subcalls:
                raise ModelOutputInvalidError(
                    "llm_query call limit reached for this REPL runtime execution."
                )
            subcall_id = int(usage.llm_subcalls) + 1
            prompt_text = str(prompt)
            text_generator = getattr(self._model_client, "generate_text", None)
            if callable(text_generator):
                text_result = text_generator(
                    model_name=self._model_name,
                    temperature=self._temperature,
                    system_prompt=(
                        "You are a semantic subquery assistant. Return concise factual output only."
                    ),
                    user_prompt=prompt_text,
                    max_output_tokens=400,
                )
                answer = str(getattr(text_result, "text", "") or "").strip()
                if not answer:
                    raise ModelOutputInvalidError("llm_query returned empty answer.")
                usage_payload = getattr(text_result, "usage", None)
                self._subcall_count += 1
                usage.llm_subcalls += 1
                usage.tokens_in += int(getattr(usage_payload, "tokens_in", 0))
                usage.tokens_out += int(getattr(usage_payload, "tokens_out", 0))
                usage.cost_usd += float(getattr(usage_payload, "cost_usd", 0.0))
                subquery_trace.append(
                    {
                        "subcall_id": subcall_id,
                        "mode": "text",
                        "prompt": _clip_text(prompt_text),
                        "answer": _clip_text(answer),
                        "tokens_in": int(getattr(usage_payload, "tokens_in", 0)),
                        "tokens_out": int(getattr(usage_payload, "tokens_out", 0)),
                        "cost_usd": float(getattr(usage_payload, "cost_usd", 0.0)),
                    }
                )
                return answer

            request = StructuredGenerationRequest(
                model_provider=str(getattr(self._model_client, "model_provider", "openai")),
                model_name=self._model_name,
                temperature=self._temperature,
                system_prompt=(
                    "You are a semantic subquery assistant. Return concise factual output only."
                ),
                user_prompt=prompt_text,
                response_schema_name="repl_runtime_subquery_v1",
                response_schema=_REPL_SUBQUERY_SCHEMA,
                max_output_tokens=400,
            )
            result = run_structured_generation_loop(
                client=self._model_client,
                request=request,
                max_attempts=2,
            )
            answer = result.output.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ModelOutputInvalidError("llm_query returned empty answer.")
            self._subcall_count += 1
            usage.llm_subcalls += 1
            usage.tokens_in += int(result.usage.tokens_in)
            usage.tokens_out += int(result.usage.tokens_out)
            usage.cost_usd += float(result.usage.cost_usd)
            subquery_trace.append(
                {
                    "subcall_id": subcall_id,
                    "mode": "structured",
                    "prompt": _clip_text(prompt_text),
                    "answer": _clip_text(answer),
                    "tokens_in": int(result.usage.tokens_in),
                    "tokens_out": int(result.usage.tokens_out),
                    "cost_usd": float(result.usage.cost_usd),
                }
            )
            return answer

        def _llm_query_batched(prompts: list[str]) -> list[str]:
            if not isinstance(prompts, list):
                raise ModelOutputInvalidError("llm_query_batched expects a list of prompts.")
            answers: list[str] = []
            for prompt in prompts:
                answers.append(_llm_query(str(prompt)))
            return answers

        def _submit(**fields: Any) -> None:
            raise _SubmitSignal(dict(fields))

        globals_env: dict[str, Any] = {
            "__builtins__": dict(_SAFE_BUILTINS),
            "call_tool": _call_tool,
            "llm_query": _llm_query,
            "llm_query_batched": _llm_query_batched,
            "SUBMIT": _submit,
            "json": json,
            "re": re,
            "math": math,
        }
        globals_env["__builtins__"]["globals"] = lambda: dict(globals_env)

        stdout_buffer = io.StringIO()
        submitted_output: dict[str, Any] | None = None
        with redirect_stdout(stdout_buffer):
            try:
                exec(code, globals_env, self._locals)
            except _SubmitSignal as submit_signal:
                submitted_output = dict(submit_signal.payload)
            except SandboxViolationError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"[Error] {exc}")

        return ReplExecutionResult(
            stdout=stdout_buffer.getvalue().strip(),
            submitted_output=submitted_output,
            usage=usage,
            tool_trace=tool_trace,
            subquery_trace=subquery_trace,
        )
