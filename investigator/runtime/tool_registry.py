# ABOUTME: Provides allowlisted, deterministic Inspection API tool invocation for recursive runtime steps.
# ABOUTME: Normalizes call arguments/results and records stable hashes for runtime audit metadata.

from __future__ import annotations

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

    def call(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._allowed_tools:
            raise SandboxViolationError(f"Tool not allowlisted: {tool_name}.")
        if not hasattr(self._inspection_api, tool_name):
            raise SandboxViolationError(f"Inspection API does not expose tool: {tool_name}.")
        if not isinstance(args, dict):
            raise SandboxViolationError("Tool args must be an object.")
        normalized_args = _normalize(args)
        method = getattr(self._inspection_api, tool_name)
        result = method(**normalized_args)
        normalized_result = _normalize(result)
        return {
            "tool_name": tool_name,
            "normalized_args": normalized_args,
            "args_hash": _stable_hash(normalized_args),
            "result": normalized_result,
            "response_hash": _stable_hash(normalized_result),
        }
