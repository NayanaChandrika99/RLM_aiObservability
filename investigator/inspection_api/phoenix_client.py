# ABOUTME: Implements a Phoenix-backed read-only Inspection API for evaluator engines.
# ABOUTME: Provides deterministic span access plus controls/config context loaders for Phase 2.

from __future__ import annotations

from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd
from phoenix.session.client import Client

from investigator.inspection_api.protocol import InspectionAPI


RFC3339_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _parse_rfc3339(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _to_rfc3339(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return value.tz_convert("UTC").strftime(RFC3339_FORMAT)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime(RFC3339_FORMAT)
    return str(value)


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


class PhoenixInspectionAPI(InspectionAPI):
    def __init__(
        self,
        *,
        endpoint: str | None = None,
        project_name: str | None = None,
        client: Client | None = None,
        controls_dir: str | Path = "controls/library",
        snapshots_dir: str | Path = "configs/snapshots",
        spans_dataframe_provider: Callable[..., pd.DataFrame | None] | None = None,
    ) -> None:
        self._project_name = project_name
        self._controls_dir = Path(controls_dir)
        self._snapshots_dir = Path(snapshots_dir)
        self._spans_dataframe_provider = spans_dataframe_provider
        self._client = client or (
            None
            if spans_dataframe_provider is not None
            else Client(endpoint=endpoint, warn_if_server_not_running=False)
        )

    def _get_spans_dataframe(
        self,
        *,
        project_name: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ) -> pd.DataFrame:
        resolved_project = project_name or self._project_name
        if self._spans_dataframe_provider is not None:
            dataframe = self._spans_dataframe_provider(
                project_name=resolved_project,
                start_time=start_time,
                end_time=end_time,
                filter_expr=filter_expr,
            )
            return pd.DataFrame() if dataframe is None else dataframe.copy()
        if self._client is None:
            return pd.DataFrame()
        dataframe = self._client.get_spans_dataframe(
            filter_condition=filter_expr,
            start_time=_parse_rfc3339(start_time),
            end_time=_parse_rfc3339(end_time),
            limit=100000,
            project_name=resolved_project,
        )
        return pd.DataFrame() if dataframe is None else dataframe.copy()

    def _sorted_spans(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe
        sort_columns: list[str] = []
        if "start_time" in dataframe.columns:
            sort_columns.append("start_time")
        if "context.span_id" in dataframe.columns:
            sort_columns.append("context.span_id")
        if not sort_columns:
            return dataframe.reset_index(drop=True)
        return dataframe.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    def _extract_attributes(self, row: pd.Series) -> dict[str, Any]:
        attributes: dict[str, Any] = {}
        for column in row.index:
            if not column.startswith("attributes."):
                continue
            value = _nan_to_none(row[column])
            if value is None:
                continue
            attributes[column.removeprefix("attributes.")] = value
        return attributes

    def _latency_ms(self, row: pd.Series) -> float | None:
        if "latency_ms" in row.index and _nan_to_none(row["latency_ms"]) is not None:
            return float(row["latency_ms"])
        start = row.get("start_time")
        end = row.get("end_time")
        if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
            return float((end - start).total_seconds() * 1000.0)
        return None

    def _span_summary(self, row: pd.Series) -> dict[str, Any]:
        span_id = str(row.get("context.span_id", ""))
        trace_id = str(row.get("context.trace_id", ""))
        parent_id = _nan_to_none(row.get("parent_id"))
        status_code = _nan_to_none(row.get("status_code")) or "UNSET"
        status_message = _nan_to_none(row.get("status_message")) or ""
        span_kind = _nan_to_none(row.get("span_kind")) or "UNKNOWN"
        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_id": parent_id,
            "name": str(_nan_to_none(row.get("name")) or ""),
            "span_kind": str(span_kind),
            "status_code": str(status_code),
            "status_message": str(status_message),
            "start_time": _to_rfc3339(row.get("start_time")),
            "end_time": _to_rfc3339(row.get("end_time")),
            "latency_ms": self._latency_ms(row),
        }

    def _find_span_row(self, span_id: str) -> pd.Series:
        dataframe = self._get_spans_dataframe()
        if "context.span_id" not in dataframe.columns:
            raise KeyError(span_id)
        match = dataframe[dataframe["context.span_id"] == span_id]
        if match.empty:
            raise KeyError(span_id)
        return self._sorted_spans(match).iloc[0]

    def get_spans(self, trace_id: str, type: str | None = None) -> list[dict[str, Any]]:
        spans = self.list_spans(trace_id)
        if not type:
            return spans
        return [span for span in spans if span["span_kind"] == type]

    def list_traces(
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        dataframe = self._get_spans_dataframe(
            project_name=project_name,
            start_time=start_time,
            end_time=end_time,
            filter_expr=filter_expr,
        )
        if dataframe.empty or "context.trace_id" not in dataframe.columns:
            return []
        traces: list[dict[str, Any]] = []
        for trace_id, group in dataframe.groupby("context.trace_id"):
            ordered = self._sorted_spans(group)
            start = ordered["start_time"].min() if "start_time" in ordered.columns else None
            end = ordered["end_time"].max() if "end_time" in ordered.columns else None
            latency_ms = None
            if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                latency_ms = float((end - start).total_seconds() * 1000.0)
            traces.append(
                {
                    "project_name": project_name,
                    "trace_id": str(trace_id),
                    "span_count": int(len(group)),
                    "start_time": _to_rfc3339(start),
                    "end_time": _to_rfc3339(end),
                    "latency_ms": latency_ms,
                }
            )
        return sorted(traces, key=lambda row: ((row["start_time"] or ""), row["trace_id"]))

    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        dataframe = self._get_spans_dataframe(filter_expr=f"context.trace_id == '{trace_id}'")
        if dataframe.empty or "context.trace_id" not in dataframe.columns:
            return []
        trace_dataframe = dataframe[dataframe["context.trace_id"] == trace_id]
        return [self._span_summary(row) for _, row in self._sorted_spans(trace_dataframe).iterrows()]

    def get_span(self, span_id: str) -> dict[str, Any]:
        row = self._find_span_row(span_id)
        events = _safe_json_loads(_nan_to_none(row.get("events")))
        if not isinstance(events, list):
            events = []
        return {
            "summary": self._span_summary(row),
            "attributes": self._extract_attributes(row),
            "events": events,
        }

    def get_children(self, span_id: str) -> list[dict[str, Any]]:
        dataframe = self._get_spans_dataframe()
        if dataframe.empty or "parent_id" not in dataframe.columns:
            return []
        children = dataframe[dataframe["parent_id"] == span_id]
        return [self._span_summary(row) for _, row in self._sorted_spans(children).iterrows()]

    def _message_record(
        self,
        *,
        trace_id: str,
        span_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }

    def _extract_messages_from_payload(
        self,
        payload: Any,
        *,
        trace_id: str,
        span_id: str,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or item.get("type") or "assistant")
                content = str(item.get("content") or "")
                if content:
                    messages.append(
                        self._message_record(
                            trace_id=trace_id,
                            span_id=span_id,
                            role=role,
                            content=content,
                            metadata={k: v for k, v in item.items() if k not in {"role", "type", "content"}},
                        )
                    )
        elif isinstance(payload, dict):
            nested = payload.get("messages")
            if isinstance(nested, list):
                messages.extend(
                    self._extract_messages_from_payload(
                        nested,
                        trace_id=trace_id,
                        span_id=span_id,
                    )
                )
        return messages

    def get_messages(self, span_id: str) -> list[dict[str, Any]]:
        detail = self.get_span(span_id)
        summary = detail["summary"]
        attrs = detail["attributes"]
        trace_id = summary["trace_id"]
        messages: list[dict[str, Any]] = []
        for key in ("llm.input_messages", "llm.output_messages"):
            payload = _safe_json_loads(attrs.get(key))
            messages.extend(
                self._extract_messages_from_payload(payload, trace_id=trace_id, span_id=span_id)
            )
        if not messages:
            for key, role in (("input.value", "user"), ("output.value", "assistant")):
                value = attrs.get(key)
                payload = _safe_json_loads(value)
                extracted = self._extract_messages_from_payload(
                    payload,
                    trace_id=trace_id,
                    span_id=span_id,
                )
                if extracted:
                    messages.extend(extracted)
                elif isinstance(value, str) and value.strip():
                    messages.append(
                        self._message_record(
                            trace_id=trace_id,
                            span_id=span_id,
                            role=role,
                            content=value,
                        )
                    )
        return messages

    def get_tool_io(self, span_id: str) -> dict[str, Any] | None:
        detail = self.get_span(span_id)
        summary = detail["summary"]
        attrs = detail["attributes"]
        span_kind = summary["span_kind"]
        tool_name = attrs.get("tool.name")
        if tool_name is None and span_kind == "TOOL":
            tool_name = summary["name"]
        if tool_name is None:
            return None
        return {
            "trace_id": summary["trace_id"],
            "span_id": summary["span_id"],
            "artifact_id": f"tool:{summary['span_id']}",
            "tool_name": str(tool_name),
            "input": _safe_json_loads(attrs.get("input.value")) or attrs.get("input.value"),
            "output": _safe_json_loads(attrs.get("output.value")) or attrs.get("output.value"),
            "status_code": summary["status_code"],
        }

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        detail = self.get_span(span_id)
        summary = detail["summary"]
        attrs = detail["attributes"]
        documents: list[dict[str, Any]] = []
        for key in ("retrieval.documents", "retriever.documents", "retrieval.chunks"):
            payload = _safe_json_loads(attrs.get(key))
            if isinstance(payload, list):
                documents = [doc for doc in payload if isinstance(doc, dict)]
                if documents:
                    break
        chunks: list[dict[str, Any]] = []
        if not documents and summary["span_kind"] == "RETRIEVER":
            text = attrs.get("output.value") or attrs.get("input.value")
            if isinstance(text, str) and text.strip():
                documents = [{"document_id": "unknown", "content": text}]
        for index, document in enumerate(documents):
            document_id = str(document.get("document_id") or document.get("id") or f"doc-{index}")
            chunk_id = document.get("chunk_id")
            artifact_id = f"retrieval:{span_id}:{index}:{document_id}"
            if chunk_id is not None:
                artifact_id += f":chunk={chunk_id}"
            chunks.append(
                {
                    "trace_id": summary["trace_id"],
                    "span_id": span_id,
                    "artifact_id": artifact_id,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "content": str(document.get("content") or document.get("text") or ""),
                    "score": document.get("score"),
                    "metadata": {
                        key: value
                        for key, value in document.items()
                        if key not in {"document_id", "id", "chunk_id", "content", "text", "score"}
                    },
                }
            )
        return chunks

    def _load_controls(self) -> list[dict[str, Any]]:
        if not self._controls_dir.exists():
            return []
        controls: list[dict[str, Any]] = []
        for path in sorted(self._controls_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            raw = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                parsed = json.loads(raw)
            else:
                try:
                    import yaml
                except ImportError:
                    continue
                parsed = yaml.safe_load(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("controls"), list):
                controls.extend([item for item in parsed["controls"] if isinstance(item, dict)])
            elif isinstance(parsed, dict) and "control_id" in parsed:
                controls.append(parsed)
            elif isinstance(parsed, list):
                controls.extend([item for item in parsed if isinstance(item, dict)])
        return controls

    def list_controls(
        self,
        controls_version: str,
        app_type: str | None = None,
        tools_used: list[str] | None = None,
        data_domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        controls = []
        for control in self._load_controls():
            version = control.get("controls_version") or control.get("version")
            if version != controls_version:
                continue
            applies_when = control.get("applies_when") or {}
            if app_type:
                app_types = applies_when.get("app_types") or []
                if app_types and app_type not in app_types:
                    continue
            if tools_used:
                control_tools = set(applies_when.get("tools") or [])
                if control_tools and not control_tools.intersection(tools_used):
                    continue
            if data_domains:
                control_domains = set(applies_when.get("data_domains") or [])
                if control_domains and not control_domains.intersection(data_domains):
                    continue
            normalized = dict(control)
            normalized["controls_version"] = str(version)
            controls.append(normalized)
        return sorted(
            controls,
            key=lambda row: (
                -SEVERITY_ORDER.get(str(row.get("severity", "low")).lower(), 0),
                str(row.get("control_id", "")),
            ),
        )

    def get_control(self, control_id: str, controls_version: str) -> dict[str, Any]:
        for control in self.list_controls(controls_version=controls_version):
            if control.get("control_id") == control_id:
                return control
        raise KeyError(control_id)

    def required_evidence(self, control_id: str, controls_version: str) -> list[str]:
        control = self.get_control(control_id=control_id, controls_version=controls_version)
        required = control.get("required_evidence") or []
        return [str(item) for item in required]

    def _snapshot_paths(self) -> list[Path]:
        if not self._snapshots_dir.exists():
            return []
        return sorted([path for path in self._snapshots_dir.iterdir() if path.is_dir()], key=lambda p: p.name)

    def _snapshot_object(self, snapshot_path: Path, project_name: str) -> dict[str, Any]:
        metadata_path = snapshot_path / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        created_at = metadata.get("created_at") or datetime.fromtimestamp(
            snapshot_path.stat().st_mtime, tz=timezone.utc
        ).strftime(RFC3339_FORMAT)
        paths = sorted(
            str(path.relative_to(snapshot_path))
            for path in snapshot_path.rglob("*")
            if path.is_file() and path.name != "metadata.json"
        )
        return {
            "snapshot_id": snapshot_path.name,
            "project_name": project_name,
            "tag": metadata.get("tag") or snapshot_path.name,
            "created_at": created_at,
            "git_commit": metadata.get("git_commit"),
            "paths": paths,
            "metadata": metadata,
        }

    def list_config_snapshots(
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        start = _parse_rfc3339(start_time)
        end = _parse_rfc3339(end_time)
        snapshots: list[dict[str, Any]] = []
        for snapshot_path in self._snapshot_paths():
            snapshot = self._snapshot_object(snapshot_path, project_name)
            if tag and snapshot["tag"] != tag:
                continue
            created_at = _parse_rfc3339(snapshot["created_at"])
            if start and created_at and created_at < start:
                continue
            if end and created_at and created_at > end:
                continue
            snapshots.append(snapshot)
        return sorted(snapshots, key=lambda item: (item["created_at"], item["snapshot_id"]))

    def get_config_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        snapshot_path = self._snapshots_dir / snapshot_id
        if not snapshot_path.exists():
            raise KeyError(snapshot_id)
        return self._snapshot_object(snapshot_path, self._project_name or "default")

    def get_config_diff(self, base_snapshot_id: str, target_snapshot_id: str) -> dict[str, Any]:
        base_path = self._snapshots_dir / base_snapshot_id
        target_path = self._snapshots_dir / target_snapshot_id
        if not base_path.exists():
            raise KeyError(base_snapshot_id)
        if not target_path.exists():
            raise KeyError(target_snapshot_id)
        base_files = {
            str(path.relative_to(base_path)): path
            for path in base_path.rglob("*")
            if path.is_file() and path.name != "metadata.json"
        }
        target_files = {
            str(path.relative_to(target_path)): path
            for path in target_path.rglob("*")
            if path.is_file() and path.name != "metadata.json"
        }
        all_paths = sorted(set(base_files).union(target_files))
        changed_paths: list[str] = []
        diff_lines: list[str] = []
        for relative_path in all_paths:
            base_text = base_files.get(relative_path).read_text(encoding="utf-8") if relative_path in base_files else ""
            target_text = target_files.get(relative_path).read_text(encoding="utf-8") if relative_path in target_files else ""
            if base_text == target_text:
                continue
            changed_paths.append(relative_path)
            diff_lines.extend(
                difflib.unified_diff(
                    base_text.splitlines(),
                    target_text.splitlines(),
                    fromfile=f"{base_snapshot_id}/{relative_path}",
                    tofile=f"{target_snapshot_id}/{relative_path}",
                    lineterm="",
                )
            )
        diff_text = "\n".join(diff_lines)
        diff_hash = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
        artifact_id = f"configdiff:{diff_hash}"
        return {
            "project_name": self._project_name or "default",
            "base_snapshot_id": base_snapshot_id,
            "target_snapshot_id": target_snapshot_id,
            "artifact_id": artifact_id,
            "diff_ref": artifact_id,
            "git_commit_base": (self._snapshot_object(base_path, self._project_name or "default").get("git_commit")),
            "git_commit_target": (self._snapshot_object(target_path, self._project_name or "default").get("git_commit")),
            "paths": changed_paths,
            "summary": f"{len(changed_paths)} changed file(s)",
        }

    def search_trace(
        self,
        trace_id: str,
        pattern: str,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        dataframe = self._get_spans_dataframe(filter_expr=f"context.trace_id == '{trace_id}'")
        if dataframe.empty:
            return []
        trace_dataframe = dataframe[dataframe["context.trace_id"] == trace_id]
        if trace_dataframe.empty:
            return []
        compiled = re.compile(pattern, re.IGNORECASE)
        active_fields = fields or [column for column in trace_dataframe.columns if trace_dataframe[column].dtype == object]
        hits: list[dict[str, Any]] = []
        for _, row in self._sorted_spans(trace_dataframe).iterrows():
            span_id = str(row.get("context.span_id", ""))
            for field in active_fields:
                if field not in row.index:
                    continue
                value = _nan_to_none(row[field])
                if value is None:
                    continue
                text = str(value)
                match = compiled.search(text)
                if not match:
                    continue
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 80)
                hits.append(
                    {
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "field": field,
                        "value_snippet": text[start:end],
                    }
                )
        return hits

    def search(self, text_or_chunks: str | list[str], pattern: str) -> list[str]:
        chunks = [text_or_chunks] if isinstance(text_or_chunks, str) else text_or_chunks
        compiled = re.compile(pattern, re.IGNORECASE)
        matches: set[str] = set()
        for chunk in chunks:
            if not isinstance(chunk, str):
                continue
            for match in compiled.finditer(chunk):
                matches.add(match.group(0))
        return sorted(matches)
