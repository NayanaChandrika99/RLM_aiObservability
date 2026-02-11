# ABOUTME: Implements an offline Parquet-backed Inspection API for deterministic evaluator proof runs.
# ABOUTME: Reuses Phoenix inspection semantics while adding manifest run_id-to-trace_id linking helpers.

from __future__ import annotations

from datetime import datetime
import ast
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from investigator.inspection_api.phoenix_client import PhoenixInspectionAPI


_FILTER_EQ_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.]+)\s*==\s*['\"]([^'\"]+)['\"]\s*$")


class ParquetInspectionAPI(PhoenixInspectionAPI):
    def __init__(
        self,
        *,
        parquet_path: str | Path,
        project_name: str | None = None,
        controls_dir: str | Path = "controls/library",
        snapshots_dir: str | Path = "configs/snapshots",
    ) -> None:
        self._parquet_path = Path(parquet_path)
        if not self._parquet_path.exists():
            raise FileNotFoundError(self._parquet_path)
        self._spans_dataframe = pd.read_parquet(self._parquet_path)
        super().__init__(
            project_name=project_name,
            controls_dir=controls_dir,
            snapshots_dir=snapshots_dir,
            spans_dataframe_provider=self._spans_dataframe_provider,
        )

    @staticmethod
    def _parse_attr_value(value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(stripped)
                except Exception:
                    return None
                if isinstance(parsed, dict):
                    return parsed
                return None
        return None

    @staticmethod
    def _as_timestamp(value: str | None) -> pd.Timestamp | None:
        if not value:
            return None
        try:
            timestamp = pd.Timestamp(value)
        except Exception:
            return None
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    @classmethod
    def _apply_filter_expr(cls, dataframe: pd.DataFrame, filter_expr: str | None) -> pd.DataFrame:
        if not filter_expr:
            return dataframe
        match = _FILTER_EQ_PATTERN.match(filter_expr)
        if not match:
            return dataframe
        column, raw_value = match.groups()
        if column not in dataframe.columns:
            return dataframe.iloc[0:0]
        return dataframe[dataframe[column].astype(str) == raw_value]

    @classmethod
    def _apply_time_window(
        cls,
        dataframe: pd.DataFrame,
        *,
        start_time: str | None,
        end_time: str | None,
    ) -> pd.DataFrame:
        if dataframe.empty or "start_time" not in dataframe.columns:
            return dataframe
        result = dataframe
        start = cls._as_timestamp(start_time)
        end = cls._as_timestamp(end_time)
        if start is not None:
            result = result[pd.to_datetime(result["start_time"], utc=True) >= start]
        if end is not None:
            result = result[pd.to_datetime(result["start_time"], utc=True) <= end]
        return result

    def _spans_dataframe_provider(
        self,
        *,
        project_name: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ) -> pd.DataFrame:
        del project_name
        dataframe = self._spans_dataframe.copy()
        dataframe = self._apply_filter_expr(dataframe, filter_expr)
        dataframe = self._apply_time_window(
            dataframe,
            start_time=start_time,
            end_time=end_time,
        )
        return dataframe

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        chunks = super().get_retrieval_chunks(span_id)
        if chunks:
            return chunks

        detail = self.get_span(span_id)
        summary = detail["summary"]
        attrs = detail["attributes"]
        phase1 = self._parse_attr_value(attrs.get("phase1"))
        retrieval = (phase1 or {}).get("retrieval") if isinstance(phase1, dict) else None
        documents = (retrieval or {}).get("documents") if isinstance(retrieval, dict) else None
        if isinstance(documents, list):
            normalized_documents = documents
        elif documents is None:
            return []
        else:
            try:
                normalized_documents = list(documents)
            except TypeError:
                return []

        fallback_chunks: list[dict[str, Any]] = []
        for index, document in enumerate(normalized_documents):
            if not isinstance(document, dict):
                continue
            document_id = str(document.get("document_id") or document.get("id") or f"doc-{index}")
            chunk_id = document.get("chunk_id")
            artifact_id = f"retrieval:{span_id}:{index}:{document_id}"
            if chunk_id is not None:
                artifact_id += f":chunk={chunk_id}"
            fallback_chunks.append(
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
        return fallback_chunks

    def _run_id_trace_mapping(self) -> dict[str, str]:
        dataframe = self._spans_dataframe.copy()
        if dataframe.empty or "attributes.phase1" not in dataframe.columns:
            return {}

        mapping: dict[str, tuple[bool, datetime, str]] = {}
        for _, row in dataframe.iterrows():
            trace_id = str(row.get("context.trace_id") or "")
            if not trace_id:
                continue
            phase1 = self._parse_attr_value(row.get("attributes.phase1"))
            if not isinstance(phase1, dict):
                continue
            run_id = str(phase1.get("run_id") or "")
            if not run_id:
                continue
            parent_id = row.get("parent_id")
            step = phase1.get("step")
            is_root = (parent_id is None or (isinstance(parent_id, float) and pd.isna(parent_id))) and step is None
            start_value = row.get("start_time")
            try:
                if isinstance(start_value, pd.Timestamp):
                    start_time = start_value.to_pydatetime()
                else:
                    start_time = pd.Timestamp(start_value).to_pydatetime()
            except Exception:
                start_time = datetime.min
            existing = mapping.get(run_id)
            if existing is None:
                mapping[run_id] = (is_root, start_time, trace_id)
                continue
            existing_is_root, existing_start, _ = existing
            should_replace = False
            if is_root and not existing_is_root:
                should_replace = True
            elif is_root == existing_is_root and start_time < existing_start:
                should_replace = True
            if should_replace:
                mapping[run_id] = (is_root, start_time, trace_id)

        return {run_id: trace_id for run_id, (_, _, trace_id) in mapping.items()}

    def attach_manifest_trace_ids(self, *, manifest_path: str | Path) -> dict[str, Any]:
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = payload.get("cases")
        if not isinstance(cases, list):
            return payload
        mapping = self._run_id_trace_mapping()
        for case in cases:
            if not isinstance(case, dict):
                continue
            run_id = str(case.get("run_id") or "")
            if not run_id:
                continue
            trace_id = mapping.get(run_id)
            if trace_id:
                case["trace_id"] = trace_id
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload
