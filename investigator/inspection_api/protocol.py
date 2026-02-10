# ABOUTME: Defines the read-only Inspection API protocol used by all RLM engines.
# ABOUTME: Mirrors API.md signatures so engine code compiles against a stable interface.

from __future__ import annotations

from typing import Any, Protocol


class InspectionAPI(Protocol):
    def get_spans(self, trace_id: str, type: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    def list_traces(
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_span(self, span_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def get_children(self, span_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_messages(self, span_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_tool_io(self, span_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def list_controls(
        self,
        controls_version: str,
        app_type: str | None = None,
        tools_used: list[str] | None = None,
        data_domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_control(self, control_id: str, controls_version: str) -> dict[str, Any]:
        raise NotImplementedError

    def required_evidence(self, control_id: str, controls_version: str) -> list[str]:
        raise NotImplementedError

    def list_config_snapshots(
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_config_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def get_config_diff(self, base_snapshot_id: str, target_snapshot_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def search_trace(
        self,
        trace_id: str,
        pattern: str,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def search(self, text_or_chunks: str | list[str], pattern: str) -> list[str]:
        raise NotImplementedError
