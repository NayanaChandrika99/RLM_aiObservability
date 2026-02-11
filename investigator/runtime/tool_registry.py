# ABOUTME: Provides allowlisted, deterministic Inspection API tool invocation for recursive runtime steps.
# ABOUTME: Normalizes call arguments/results and records stable hashes for runtime audit metadata.

from __future__ import annotations

import inspect
import hashlib
import json
from typing import Any

from investigator.inspection_api.protocol import InspectionAPI
from investigator.runtime.sandbox import SandboxViolationError


DEFAULT_ALLOWED_TOOLS = {
    "list_traces",
    "get_spans",
    "list_spans",
    "get_span",
    "get_children",
    "get_messages",
    "get_tool_io",
    "get_retrieval_chunks",
    "list_controls",
    "get_control",
    "required_evidence",
    "list_config_snapshots",
    "get_config_snapshot",
    "get_config_diff",
    "search_trace",
    "search",
}


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sort_if_span_rows(value: list[Any]) -> list[Any]:
    if not value:
        return value
    if all(isinstance(item, dict) for item in value):
        as_dicts = [item for item in value if isinstance(item, dict)]
        if all(("start_time" in item or "timestamp" in item) for item in as_dicts):
            return sorted(
                as_dicts,
                key=lambda item: (
                    str(item.get("start_time") or item.get("timestamp") or ""),
                    str(item.get("span_id") or item.get("trace_id") or ""),
                ),
            )
    return value


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value.keys(), key=str)}
    if isinstance(value, list):
        normalized_items = [_normalize(item) for item in value]
        return _sort_if_span_rows(normalized_items)
    return value


class ToolRegistry:
    def __init__(
        self,
        *,
        inspection_api: InspectionAPI,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self._inspection_api = inspection_api
        self._allowed_tools = set(allowed_tools or DEFAULT_ALLOWED_TOOLS)

    @property
    def allowed_tools(self) -> set[str]:
        return set(self._allowed_tools)

    def describe_tools(self) -> dict[str, dict[str, Any]]:
        descriptions: dict[str, dict[str, Any]] = {}
        for tool_name in sorted(self._allowed_tools):
            if not hasattr(self._inspection_api, tool_name):
                continue
            method = getattr(self._inspection_api, tool_name)
            required_args: list[str] = []
            optional_args: list[str] = []
            accepts_var_kwargs = False
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                for name, parameter in signature.parameters.items():
                    if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                        accepts_var_kwargs = True
                        continue
                    if parameter.kind not in {
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    }:
                        continue
                    if parameter.default is inspect._empty:
                        required_args.append(str(name))
                    else:
                        optional_args.append(str(name))
            descriptions[tool_name] = {
                "required_args": sorted(required_args),
                "optional_args": sorted(optional_args),
                "accepts_var_kwargs": bool(accepts_var_kwargs),
            }
        return descriptions

    @staticmethod
    def _sanitize_call_args(method: Any, args: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return dict(args)

        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return dict(args)

        allowed_param_names = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        return {name: args[name] for name in args if name in allowed_param_names}

    def call(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._allowed_tools:
            raise SandboxViolationError(f"Tool not allowlisted: {tool_name}.")
        if not hasattr(self._inspection_api, tool_name):
            raise SandboxViolationError(f"Inspection API does not expose tool: {tool_name}.")
        if not isinstance(args, dict):
            raise SandboxViolationError("Tool args must be an object.")
        normalized_args = _normalize(args)
        method = getattr(self._inspection_api, tool_name)
        sanitized_args = self._sanitize_call_args(method, normalized_args)
        try:
            result = method(**sanitized_args)
        except TypeError as exc:
            raise SandboxViolationError(
                f"Tool call argument mismatch for {tool_name}: {exc}"
            ) from exc
        normalized_result = _normalize(result)
        return {
            "tool_name": tool_name,
            "normalized_args": sanitized_args,
            "args_hash": _stable_hash(sanitized_args),
            "result": normalized_result,
            "response_hash": _stable_hash(normalized_result),
        }
