# ABOUTME: Executes model-generated Python snippets in a restricted REPL with read-only tool and sub-LLM helpers.
# ABOUTME: Captures stdout, helper usage accounting, and SUBMIT finalization payloads for bounded runtime loops.

from __future__ import annotations

import builtins
import collections
from contextlib import redirect_stderr, redirect_stdout
import copy
import dataclasses
import datetime
from dataclasses import dataclass
import functools
import hashlib
import io
import itertools
import json
import math
import operator
import re
import select
import statistics
import subprocess
import sys
import textwrap
import time
import typing
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

BLOCKED_MODULES = {
    "os",
    "subprocess",
    "socket",
    "http",
    "urllib",
    "pathlib",
    "shutil",
    "signal",
    "ctypes",
    "importlib",
    "sys",
    "multiprocessing",
    "threading",
}

ALLOWED_ANALYSIS_MODULES = {
    "json",
    "re",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "operator",
    "datetime",
    "dataclasses",
    "typing",
    "copy",
    "textwrap",
    "hashlib",
}

_ORIGINAL_IMPORT = builtins.__import__


def _root_module_name(module_name: str) -> str:
    return str(module_name).split(".", 1)[0].strip()


def _guarded_import(
    name: str,
    globals: dict[str, Any] | None = None,  # noqa: A002
    locals: dict[str, Any] | None = None,  # noqa: A002
    fromlist: tuple[Any, ...] | list[Any] = (),
    level: int = 0,
) -> Any:
    root = _root_module_name(name)
    if root in BLOCKED_MODULES:
        raise ImportError("Import statements are blocked in REPL runtime.")
    if root not in ALLOWED_ANALYSIS_MODULES:
        raise ImportError("Import statements are blocked in REPL runtime.")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


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


def _truncate_output(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


@dataclass
class SandboxExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    mode: str
    submitted_output: dict[str, Any] | None = None
    locals_snapshot: dict[str, Any] | None = None


_SUBPROCESS_SANDBOX_TEMPLATE = textwrap.dedent(
    """
    import builtins
    import collections
    import copy
    import dataclasses
    import datetime
    import functools
    import hashlib
    import io
    import itertools
    import json
    import math
    import operator
    import re
    import statistics
    import textwrap
    import typing
    from contextlib import redirect_stderr, redirect_stdout
    import sys

    BLOCKED_MODULES = set(__BLOCKED_MODULES__)
    ALLOWED_ANALYSIS_MODULES = set(__ALLOWED_MODULES__)
    _ORIGINAL_IMPORT = builtins.__import__
    _MAX_OUTPUT_CHARS = int(__MAX_OUTPUT_CHARS__)
    _EVENT_OUT = sys.__stdout__
    _EVENT_IN = sys.__stdin__

    def _root_module_name(module_name):
        return str(module_name).split(".", 1)[0].strip()

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = _root_module_name(name)
        if root in BLOCKED_MODULES:
            raise ImportError("Import statements are blocked in REPL runtime.")
        if root not in ALLOWED_ANALYSIS_MODULES:
            raise ImportError("Import statements are blocked in REPL runtime.")
        return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [_json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        return str(value)

    def _emit_event(payload):
        _EVENT_OUT.write(json.dumps(payload, ensure_ascii=False) + "\\n")
        _EVENT_OUT.flush()

    def _request(payload):
        _emit_event(payload)
        response_line = _EVENT_IN.readline()
        if not response_line:
            raise RuntimeError("Sandbox parent did not return a response.")
        response = json.loads(response_line)
        if not isinstance(response, dict):
            raise RuntimeError("Sandbox parent returned invalid response payload.")
        if not bool(response.get("ok")):
            raise RuntimeError(str(response.get("error") or "Sandbox parent handler failed."))
        return response.get("result")

    class _SubmitSignal(RuntimeError):
        def __init__(self, payload):
            super().__init__("SUBMIT called")
            self.payload = payload

    def _blocked_callable(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("[SandboxViolation] This callable is blocked in REPL runtime.")

    def _call_tool(tool_name, **kwargs):
        return _request(
            {
                "event": "tool_request",
                "tool_name": str(tool_name),
                "args": _json_safe(kwargs),
            }
        )

    def _llm_query(prompt):
        answer = _request({"event": "llm_query_request", "prompt": str(prompt)})
        if not isinstance(answer, str):
            raise RuntimeError("llm_query returned non-string answer.")
        return answer

    def _llm_query_batched(prompts):
        if not isinstance(prompts, list):
            raise RuntimeError("llm_query_batched expects a list of prompts.")
        answers = _request(
            {
                "event": "llm_query_batched_request",
                "prompts": [str(item) for item in prompts],
            }
        )
        if not isinstance(answers, list):
            raise RuntimeError("llm_query_batched returned non-list answers.")
        return [str(item) for item in answers]

    def _submit(**fields):
        raise _SubmitSignal(dict(fields))

    safe_builtins = {
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
        "__import__": _guarded_import,
        "open": _blocked_callable,
        "eval": _blocked_callable,
        "exec": _blocked_callable,
        "compile": _blocked_callable,
        "input": _blocked_callable,
    }
    init_payload = {}
    init_line = _EVENT_IN.readline()
    if init_line:
        try:
            init_payload = json.loads(init_line)
        except Exception:
            init_payload = {}
    code = str(init_payload.get("code") or "")
    locals_env = init_payload.get("locals")
    if not isinstance(locals_env, dict):
        locals_env = {}
    globals_env = {
        "__builtins__": safe_builtins,
        "call_tool": _call_tool,
        "llm_query": _llm_query,
        "llm_query_batched": _llm_query_batched,
        "SUBMIT": _submit,
        "json": json,
        "re": re,
        "math": math,
        "statistics": statistics,
        "collections": collections,
        "itertools": itertools,
        "functools": functools,
        "operator": operator,
        "datetime": datetime,
        "dataclasses": dataclasses,
        "typing": typing,
        "copy": copy,
        "textwrap": textwrap,
        "hashlib": hashlib,
    }
    globals_env["__builtins__"]["globals"] = lambda: dict(globals_env)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    submitted_output = None
    returncode = 0
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        try:
            exec(code, globals_env, locals_env)
        except _SubmitSignal as submit_signal:
            submitted_output = _json_safe(submit_signal.payload)
        except Exception as exc:
            returncode = 1
            print(f"[Error] {exc}")
    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    if len(stdout_text) > _MAX_OUTPUT_CHARS:
        stdout_text = stdout_text[:_MAX_OUTPUT_CHARS]
    if len(stderr_text) > _MAX_OUTPUT_CHARS:
        stderr_text = stderr_text[:_MAX_OUTPUT_CHARS]
    _emit_event(
        {
            "event": "result",
            "returncode": int(returncode),
            "stdout": stdout_text,
            "stderr": stderr_text,
            "submitted_output": submitted_output,
            "locals_snapshot": _json_safe(locals_env),
        }
    )
    raise SystemExit(0)
    """
).strip()


def _run_in_process_sandbox(
    *,
    code: str,
    globals_env: dict[str, Any] | None,
    locals_env: dict[str, Any] | None,
    submit_signal_type: type[BaseException] | None,
    max_output_chars: int,
) -> SandboxExecutionResult:
    runtime_globals = dict(globals_env) if isinstance(globals_env, dict) else {}
    builtins_payload = runtime_globals.get("__builtins__")
    if isinstance(builtins_payload, dict):
        safe_builtins = dict(builtins_payload)
    else:
        safe_builtins = dict(_SAFE_BUILTINS)
    safe_builtins["__import__"] = _guarded_import
    runtime_globals["__builtins__"] = safe_builtins
    runtime_globals.setdefault("json", json)
    runtime_globals.setdefault("re", re)
    runtime_globals.setdefault("math", math)
    runtime_globals.setdefault("statistics", statistics)
    runtime_globals.setdefault("collections", collections)
    runtime_globals.setdefault("itertools", itertools)
    runtime_globals.setdefault("functools", functools)
    runtime_globals.setdefault("operator", operator)
    runtime_globals.setdefault("datetime", datetime)
    runtime_globals.setdefault("dataclasses", dataclasses)
    runtime_globals.setdefault("typing", typing)
    runtime_globals.setdefault("copy", copy)
    runtime_globals.setdefault("textwrap", textwrap)
    runtime_globals.setdefault("hashlib", hashlib)
    runtime_locals: dict[str, Any] = locals_env if isinstance(locals_env, dict) else {}

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    submitted_output: dict[str, Any] | None = None
    returncode = 0
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        try:
            exec(code, runtime_globals, runtime_locals)
        except SandboxViolationError:
            raise
        except Exception as exc:  # noqa: BLE001
            if submit_signal_type is not None and isinstance(exc, submit_signal_type):
                payload = getattr(exc, "payload", None)
                if isinstance(payload, dict):
                    submitted_output = dict(payload)
                else:
                    submitted_output = {}
            else:
                returncode = 1
                print(f"[Error] {exc}")
    return SandboxExecutionResult(
        stdout=_truncate_output(stdout_buffer.getvalue().strip(), max_chars=max_output_chars),
        stderr=_truncate_output(stderr_buffer.getvalue().strip(), max_chars=max_output_chars),
        returncode=returncode,
        timed_out=False,
        mode="in_process",
        submitted_output=submitted_output,
        locals_snapshot=_json_safe(runtime_locals),
    )


def _run_subprocess_sandbox(
    *,
    code: str,
    timeout_sec: int,
    max_output_chars: int,
    locals_env: dict[str, Any] | None,
    tool_handler: Any,
    llm_query_handler: Any,
    llm_query_batched_handler: Any,
) -> SandboxExecutionResult:
    sandbox_script = (
        _SUBPROCESS_SANDBOX_TEMPLATE
        .replace("__BLOCKED_MODULES__", repr(sorted(BLOCKED_MODULES)))
        .replace("__ALLOWED_MODULES__", repr(sorted(ALLOWED_ANALYSIS_MODULES)))
        .replace("__MAX_OUTPUT_CHARS__", str(int(max_output_chars)))
    )
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-u", "-c", sandbox_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        raise RuntimeError("Sandbox subprocess pipes are unavailable.")

    def _write_response(payload: dict[str, Any]) -> None:
        process.stdin.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")
        process.stdin.flush()

    try:
        init_payload = {"code": code, "locals": _json_safe(locals_env or {})}
        process.stdin.write(json.dumps(init_payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

        result_payload: dict[str, Any] | None = None
        deadline = time.monotonic() + max(1, int(timeout_sec))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                process.wait(timeout=1)
                return SandboxExecutionResult(
                    stdout="",
                    stderr=f"Sandbox execution timeout after {max(1, int(timeout_sec))} seconds.",
                    returncode=124,
                    timed_out=True,
                    mode="subprocess",
                )

            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                process.kill()
                process.wait(timeout=1)
                return SandboxExecutionResult(
                    stdout="",
                    stderr=f"Sandbox execution timeout after {max(1, int(timeout_sec))} seconds.",
                    returncode=124,
                    timed_out=True,
                    mode="subprocess",
                )

            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue

            try:
                event_payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event_payload, dict):
                continue
            event_type = str(event_payload.get("event") or "")

            if event_type == "tool_request":
                tool_name = str(event_payload.get("tool_name") or "")
                args_payload = event_payload.get("args")
                tool_args = args_payload if isinstance(args_payload, dict) else {}
                if not callable(tool_handler):
                    _write_response({"ok": False, "error": "Tool handler unavailable."})
                    continue
                try:
                    result = tool_handler(tool_name, dict(tool_args))
                    _write_response({"ok": True, "result": result})
                except Exception as exc:  # noqa: BLE001
                    _write_response({"ok": False, "error": str(exc)})
                continue

            if event_type == "llm_query_request":
                prompt = str(event_payload.get("prompt") or "")
                if not callable(llm_query_handler):
                    _write_response({"ok": False, "error": "llm_query handler unavailable."})
                    continue
                try:
                    answer = llm_query_handler(prompt)
                    _write_response({"ok": True, "result": str(answer)})
                except Exception as exc:  # noqa: BLE001
                    _write_response({"ok": False, "error": str(exc)})
                continue

            if event_type == "llm_query_batched_request":
                prompts_raw = event_payload.get("prompts")
                prompts = [str(item) for item in prompts_raw] if isinstance(prompts_raw, list) else []
                handler = llm_query_batched_handler
                if not callable(handler) and callable(llm_query_handler):
                    handler = lambda batch: [llm_query_handler(item) for item in batch]
                if not callable(handler):
                    _write_response({"ok": False, "error": "llm_query_batched handler unavailable."})
                    continue
                try:
                    answers = handler(prompts)
                    _write_response({"ok": True, "result": [str(item) for item in (answers or [])]})
                except Exception as exc:  # noqa: BLE001
                    _write_response({"ok": False, "error": str(exc)})
                continue

            if event_type == "result":
                result_payload = dict(event_payload)
                break

        if result_payload is None:
            returncode = int(process.poll() or 1)
            stderr_text = _truncate_output(
                str(process.stderr.read() or "").strip(),
                max_chars=max_output_chars,
            )
            return SandboxExecutionResult(
                stdout="",
                stderr=stderr_text or "Sandbox subprocess exited without a result payload.",
                returncode=returncode,
                timed_out=False,
                mode="subprocess",
            )

        process.wait(timeout=1)
        pipe_stderr = str(process.stderr.read() or "").strip()
        result_stdout = _truncate_output(
            str(result_payload.get("stdout") or "").strip(),
            max_chars=max_output_chars,
        )
        result_stderr = _truncate_output(
            str(result_payload.get("stderr") or "").strip(),
            max_chars=max_output_chars,
        )
        if pipe_stderr:
            result_stderr = (
                f"{result_stderr}\n{pipe_stderr}".strip() if result_stderr else pipe_stderr
            )
            result_stderr = _truncate_output(result_stderr, max_chars=max_output_chars)
        submitted_payload = result_payload.get("submitted_output")
        submitted_output = dict(submitted_payload) if isinstance(submitted_payload, dict) else None
        locals_snapshot_payload = result_payload.get("locals_snapshot")
        locals_snapshot = (
            dict(locals_snapshot_payload)
            if isinstance(locals_snapshot_payload, dict)
            else None
        )
        return SandboxExecutionResult(
            stdout=result_stdout,
            stderr=result_stderr,
            returncode=int(result_payload.get("returncode") or 0),
            timed_out=False,
            mode="subprocess",
            submitted_output=submitted_output,
            locals_snapshot=locals_snapshot,
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()


def execute_in_sandbox(
    code: str,
    *,
    timeout_sec: int = 30,
    max_output_chars: int = 8192,
    globals_env: dict[str, Any] | None = None,
    locals_env: dict[str, Any] | None = None,
    submit_signal_type: type[BaseException] | None = None,
    tool_handler: Any = None,
    llm_query_handler: Any = None,
    llm_query_batched_handler: Any = None,
) -> SandboxExecutionResult:
    if not isinstance(code, str):
        raise TypeError("code must be a string.")

    should_use_subprocess = (
        callable(tool_handler)
        or callable(llm_query_handler)
        or callable(llm_query_batched_handler)
        or (globals_env is None and submit_signal_type is None)
    )
    if should_use_subprocess:
        try:
            return _run_subprocess_sandbox(
                code=code,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
                locals_env=locals_env,
                tool_handler=tool_handler,
                llm_query_handler=llm_query_handler,
                llm_query_batched_handler=llm_query_batched_handler,
            )
        except Exception:  # noqa: BLE001
            # Fallback path for environments where subprocess setup is unavailable.
            pass

    return _run_in_process_sandbox(
        code=code,
        globals_env=globals_env,
        locals_env=locals_env,
        submit_signal_type=submit_signal_type,
        max_output_chars=max_output_chars,
    )


def _blocked_import(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    if not args:
        raise ImportError("Import statements are blocked in REPL runtime.")
    module_name = str(args[0])
    globals_payload = args[1] if len(args) > 1 else kwargs.get("globals")
    locals_payload = args[2] if len(args) > 2 else kwargs.get("locals")
    fromlist_payload = args[3] if len(args) > 3 else kwargs.get("fromlist", ())
    level_payload = args[4] if len(args) > 4 else kwargs.get("level", 0)
    return _guarded_import(
        module_name,
        globals_payload if isinstance(globals_payload, dict) else None,
        locals_payload if isinstance(locals_payload, dict) else None,
        fromlist_payload if isinstance(fromlist_payload, (list, tuple)) else (),
        int(level_payload) if isinstance(level_payload, int) else 0,
    )


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
        sandbox_timeout_sec: int = 30,
        sandbox_max_output_chars: int = 8192,
    ) -> None:
        self._tool_registry = tool_registry
        self._model_client = model_client
        self._model_name = model_name
        self._temperature = temperature
        self._max_llm_subcalls = max(0, int(max_llm_subcalls))
        self._sandbox_timeout_sec = max(1, int(sandbox_timeout_sec))
        self._sandbox_max_output_chars = max(1, int(sandbox_max_output_chars))
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

        sandbox_result = execute_in_sandbox(
            code,
            timeout_sec=self._sandbox_timeout_sec,
            max_output_chars=self._sandbox_max_output_chars,
            globals_env=globals_env,
            locals_env=self._locals,
            submit_signal_type=_SubmitSignal,
            tool_handler=lambda tool_name, args: _call_tool(tool_name, **dict(args or {})),
            llm_query_handler=_llm_query,
            llm_query_batched_handler=_llm_query_batched,
        )
        if (
            sandbox_result.mode == "subprocess"
            and isinstance(sandbox_result.locals_snapshot, dict)
        ):
            self._locals = {str(key): value for key, value in sandbox_result.locals_snapshot.items()}
        sandbox_output_text = (
            f"{str(sandbox_result.stdout or '')}\n{str(sandbox_result.stderr or '')}"
        )
        if "[SandboxViolation]" in sandbox_output_text:
            raise SandboxViolationError("This callable is blocked in REPL runtime.")
        submitted_output = (
            dict(sandbox_result.submitted_output)
            if isinstance(sandbox_result.submitted_output, dict)
            else None
        )
        stdout = str(sandbox_result.stdout or "").strip()
        stderr = str(sandbox_result.stderr or "").strip()
        if stderr:
            stdout = f"{stdout}\n{stderr}".strip() if stdout else stderr

        return ReplExecutionResult(
            stdout=stdout,
            submitted_output=submitted_output,
            usage=usage,
            tool_trace=tool_trace,
            subquery_trace=subquery_trace,
        )
