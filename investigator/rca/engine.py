# ABOUTME: Implements deterministic Trace RCA narrowing and label selection over inspection APIs.
# ABOUTME: Produces evidence-linked RCA outputs with stable ordering and safe fallback behavior.

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from investigator.inspection_api import PhoenixInspectionAPI
from investigator.runtime.contracts import EvidenceRef, InputRef, RCAReport, TimeWindow, hash_excerpt


@dataclass
class TraceRCARequest:
    trace_id: str
    project_name: str


class TraceRCAEngine:
    engine_type = "rca"
    output_contract_name = "RCAReport"
    engine_version = "0.2.0"
    model_name = "gpt-5-mini"
    prompt_template_hash = "trace-rca-hot-span-v1"
    temperature = 0.0

    def __init__(
        self,
        inspection_api: Any | None = None,
        *,
        max_hot_spans: int = 5,
        max_branch_depth: int = 2,
        max_branch_nodes: int = 30,
    ) -> None:
        self._inspection_api = inspection_api
        self._max_hot_spans = max_hot_spans
        self._max_branch_depth = max_branch_depth
        self._max_branch_nodes = max_branch_nodes

    def build_input_ref(self, request: TraceRCARequest) -> InputRef:
        return InputRef(
            project_name=request.project_name,
            trace_ids=[request.trace_id],
            time_window=TimeWindow(),
            filter_expr=None,
            controls_version=None,
        )

    def _resolve_inspection_api(self, request: TraceRCARequest) -> Any:
        if self._inspection_api is not None:
            return self._inspection_api
        return PhoenixInspectionAPI(project_name=request.project_name)

    @staticmethod
    def _has_exception_event(events: list[dict[str, Any]]) -> bool:
        for event in events:
            name = str(event.get("name") or "").lower()
            if "exception" in name or "error" in name:
                return True
        return False

    @staticmethod
    def _status_text(summary: dict[str, Any], detail: dict[str, Any]) -> str:
        status_message = str(summary.get("status_message") or "")
        event_text = " ".join(
            [
                json.dumps(event, sort_keys=True)
                for event in (detail.get("events") or [])
                if isinstance(event, dict)
            ]
        )
        return f"{summary.get('name', '')} {status_message} {event_text}".lower()

    @staticmethod
    def _sort_hot_spans(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def sort_key(candidate: dict[str, Any]) -> tuple[int, int, float, str]:
            summary = candidate["summary"]
            error_rank = 0 if str(summary.get("status_code")) == "ERROR" else 1
            exception_rank = 0 if candidate["has_exception"] else 1
            latency = float(summary.get("latency_ms") or 0.0)
            span_id = str(summary.get("span_id") or "")
            return (error_rank, exception_rank, -latency, span_id)

        return sorted(candidates, key=sort_key)

    @staticmethod
    def _sort_children(children: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            children,
            key=lambda child: (
                str(child.get("start_time") or ""),
                str(child.get("span_id") or ""),
            ),
        )

    @staticmethod
    def _resolve_branch_root_span_id(
        span_id: str,
        spans_by_id: dict[str, dict[str, Any]],
    ) -> str:
        current = span_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            parent_id = str((spans_by_id.get(current) or {}).get("parent_id") or "")
            if not parent_id:
                return current
            current = parent_id
        return span_id

    def _collect_branch_span_ids(
        self,
        inspection_api: Any,
        root_span_id: str,
        gaps: list[str],
    ) -> list[str]:
        visited: set[str] = {root_span_id}
        ordered: list[str] = [root_span_id]
        frontier: list[tuple[str, int]] = [(root_span_id, 0)]
        while frontier and len(visited) < self._max_branch_nodes:
            current_span_id, depth = frontier.pop(0)
            if depth >= self._max_branch_depth:
                continue
            try:
                children = inspection_api.get_children(current_span_id) or []
            except Exception as exc:
                gaps.append(f"Failed to load children for {current_span_id}: {exc}")
                continue
            for child in self._sort_children(children):
                child_span_id = str(child.get("span_id") or "")
                if not child_span_id or child_span_id in visited:
                    continue
                visited.add(child_span_id)
                ordered.append(child_span_id)
                frontier.append((child_span_id, depth + 1))
                if len(visited) >= self._max_branch_nodes:
                    gaps.append(
                        f"Branch inspection truncated at max_branch_nodes={self._max_branch_nodes}."
                    )
                    break
        return ordered

    def _build_candidate(
        self,
        inspection_api: Any,
        request: TraceRCARequest,
        span_summary: dict[str, Any],
        gaps: list[str],
    ) -> dict[str, Any]:
        span_id = str(span_summary.get("span_id") or "")
        try:
            detail = inspection_api.get_span(span_id)
        except Exception as exc:
            detail = {"summary": span_summary, "attributes": {}, "events": []}
            gaps.append(f"Failed to load span details for {span_id}: {exc}")
        events = detail.get("events") or []
        has_exception = self._has_exception_event(events if isinstance(events, list) else [])
        tool_io = None
        retrieval_chunks: list[dict[str, Any]] = []
        try:
            tool_io = inspection_api.get_tool_io(span_id)
        except Exception as exc:
            gaps.append(f"Failed to load tool IO for {span_id}: {exc}")
        try:
            retrieval_chunks = inspection_api.get_retrieval_chunks(span_id) or []
        except Exception as exc:
            gaps.append(f"Failed to load retrieval chunks for {span_id}: {exc}")
        return {
            "summary": span_summary,
            "detail": detail,
            "has_exception": has_exception,
            "tool_io": tool_io,
            "retrieval_chunks": retrieval_chunks,
            "status_text": self._status_text(span_summary, detail),
        }

    @staticmethod
    def _detect_label(candidates: list[dict[str, Any]]) -> str:
        upstream_pattern = re.compile(
            r"(timeout|timed out|rate.?limit|429|502|503|504|upstream|service unavailable|gateway)",
            re.IGNORECASE,
        )
        schema_pattern = re.compile(
            r"(schema|parse|validation|invalid json|malformed|format mismatch)",
            re.IGNORECASE,
        )
        instruction_pattern = re.compile(
            r"(instruction|prompt|format drift|did not follow|policy)", re.IGNORECASE
        )

        has_tool_failure = False
        has_upstream_failure = False
        has_schema_failure = False
        has_retrieval_failure = False
        has_instruction_failure = False

        for candidate in candidates:
            summary = candidate["summary"]
            status_text = candidate["status_text"]
            span_kind = str(summary.get("span_kind") or "UNKNOWN")
            status_code = str(summary.get("status_code") or "UNSET")
            tool_io = candidate.get("tool_io")
            retrieval_chunks = candidate.get("retrieval_chunks") or []

            if upstream_pattern.search(status_text):
                has_upstream_failure = True
            if schema_pattern.search(status_text):
                has_schema_failure = True
            if instruction_pattern.search(status_text):
                has_instruction_failure = True

            tool_status_error = bool(tool_io) and str(tool_io.get("status_code")) == "ERROR"
            if span_kind == "TOOL" and (status_code == "ERROR" or tool_status_error):
                has_tool_failure = True

            if span_kind == "RETRIEVER":
                if status_code == "ERROR" or len(retrieval_chunks) == 0:
                    has_retrieval_failure = True

        if has_upstream_failure:
            return "upstream_dependency_failure"
        if has_tool_failure:
            return "tool_failure"
        if has_schema_failure:
            return "data_schema_mismatch"
        if has_retrieval_failure:
            return "retrieval_failure"
        if has_instruction_failure:
            return "instruction_failure"
        return "instruction_failure"

    @staticmethod
    def _confidence_from_evidence(label: str, evidence_refs: list[EvidenceRef]) -> float:
        base = {
            "upstream_dependency_failure": 0.68,
            "tool_failure": 0.66,
            "data_schema_mismatch": 0.64,
            "retrieval_failure": 0.6,
            "instruction_failure": 0.45,
        }.get(label, 0.45)
        independent = {(evidence.kind, evidence.ref) for evidence in evidence_refs}
        if len(independent) >= 2:
            return min(0.82, base + 0.1)
        return base

    @staticmethod
    def _remediation_for_label(label: str) -> list[str]:
        remediation = {
            "tool_failure": [
                "Inspect failing tool call input/output and add retry or guardrails around tool invocation."
            ],
            "retrieval_failure": [
                "Review retriever query construction and chunk selection criteria for this trace."
            ],
            "instruction_failure": [
                "Tighten prompt instructions and output constraints to reduce format or policy drift."
            ],
            "upstream_dependency_failure": [
                "Add resilience for upstream API failure paths (timeouts, backoff, fallback handling)."
            ],
            "data_schema_mismatch": [
                "Align tool output schema with parser expectations and add strict validation tests."
            ],
        }
        return remediation.get(label, ["Review hot spans and run targeted debugging for this trace."])

    @staticmethod
    def _fallback_report(request: TraceRCARequest, gap: str) -> RCAReport:
        summary = "RCA fallback used because inspection API data was unavailable."
        evidence = EvidenceRef(
            trace_id=request.trace_id,
            span_id="root-span",
            kind="SPAN",
            ref="root-span",
            excerpt_hash=hash_excerpt(summary),
            ts=None,
        )
        return RCAReport(
            trace_id=request.trace_id,
            primary_label="instruction_failure",
            summary=summary,
            confidence=0.35,
            evidence_refs=[evidence],
            remediation=["Check Phoenix availability and retry RCA run."],
            gaps=[gap],
        )

    def run(self, request: TraceRCARequest) -> RCAReport:
        inspection_api = self._resolve_inspection_api(request)
        try:
            spans = inspection_api.list_spans(request.trace_id)
        except Exception as exc:
            return self._fallback_report(
                request,
                gap=f"Failed to list spans from inspection API: {exc}",
            )
        if not spans:
            return self._fallback_report(
                request,
                gap="No spans were available for trace during RCA evaluation.",
            )

        all_spans_by_id = {
            str(span.get("span_id") or ""): span
            for span in spans
            if str(span.get("span_id") or "")
        }
        candidates: list[dict[str, Any]] = []
        gaps: list[str] = []
        for span in spans:
            candidates.append(self._build_candidate(inspection_api, request, span, gaps))

        hot_roots = self._sort_hot_spans(candidates)[: self._max_hot_spans]
        recursive_candidates: list[dict[str, Any]] = []
        seen_candidate_ids: set[str] = set()
        for root_candidate in hot_roots:
            root_span_id = str(root_candidate["summary"].get("span_id") or "")
            branch_root_span_id = self._resolve_branch_root_span_id(root_span_id, all_spans_by_id)
            branch_ids = self._collect_branch_span_ids(inspection_api, branch_root_span_id, gaps)
            for branch_span_id in branch_ids:
                if branch_span_id in seen_candidate_ids:
                    continue
                branch_summary = all_spans_by_id.get(branch_span_id)
                if branch_summary is None:
                    try:
                        branch_detail = inspection_api.get_span(branch_span_id)
                        branch_summary = branch_detail.get("summary") or {"span_id": branch_span_id}
                    except Exception as exc:
                        gaps.append(f"Failed to resolve branch span {branch_span_id}: {exc}")
                        continue
                recursive_candidates.append(
                    self._build_candidate(inspection_api, request, branch_summary, gaps)
                )
                seen_candidate_ids.add(branch_span_id)

        hot_candidates = self._sort_hot_spans(recursive_candidates or hot_roots)
        label = self._detect_label(hot_candidates)

        evidence_refs: list[EvidenceRef] = []
        for candidate in hot_candidates:
            summary = candidate["summary"]
            span_id = str(summary.get("span_id") or "")
            trace_id = str(summary.get("trace_id") or request.trace_id)
            status_text = candidate["status_text"]
            evidence_refs.append(
                EvidenceRef(
                    trace_id=trace_id,
                    span_id=span_id,
                    kind="SPAN",
                    ref=span_id,
                    excerpt_hash=hash_excerpt(status_text or span_id),
                    ts=summary.get("start_time"),
                )
            )
            tool_io = candidate.get("tool_io")
            if isinstance(tool_io, dict) and tool_io.get("artifact_id"):
                tool_excerpt = json.dumps(
                    {
                        "tool_name": tool_io.get("tool_name"),
                        "status_code": tool_io.get("status_code"),
                    },
                    sort_keys=True,
                )
                evidence_refs.append(
                    EvidenceRef(
                        trace_id=trace_id,
                        span_id=span_id,
                        kind="TOOL_IO",
                        ref=str(tool_io["artifact_id"]),
                        excerpt_hash=hash_excerpt(tool_excerpt),
                        ts=summary.get("start_time"),
                    )
                )
            retrieval_chunks = candidate.get("retrieval_chunks") or []
            if retrieval_chunks:
                first_chunk = retrieval_chunks[0]
                content = str(first_chunk.get("content") or "")
                evidence_refs.append(
                    EvidenceRef(
                        trace_id=trace_id,
                        span_id=span_id,
                        kind="RETRIEVAL_CHUNK",
                        ref=str(first_chunk.get("artifact_id") or f"retrieval:{span_id}:0:unknown"),
                        excerpt_hash=hash_excerpt(content or "retrieval-chunk"),
                        ts=summary.get("start_time"),
                    )
                )

        deduped: list[EvidenceRef] = []
        seen: set[tuple[str, str]] = set()
        for evidence in evidence_refs:
            key = (evidence.kind, evidence.ref)
            if key in seen:
                continue
            deduped.append(evidence)
            seen.add(key)
        if not deduped:
            deduped.append(
                EvidenceRef(
                    trace_id=request.trace_id,
                    span_id="root-span",
                    kind="SPAN",
                    ref="root-span",
                    excerpt_hash=hash_excerpt("no-evidence"),
                    ts=None,
                )
            )
            gaps.append("No evidence refs were produced from hot spans; fallback evidence was injected.")

        confidence = self._confidence_from_evidence(label, deduped)
        hot_span_ids = [str(candidate["summary"].get("span_id") or "") for candidate in hot_candidates]
        summary = (
            f"Deterministic narrowing and recursive branch inspection selected spans {hot_span_ids}; "
            f"RCA labeled trace as {label}."
        )
        return RCAReport(
            trace_id=request.trace_id,
            primary_label=label,
            summary=summary,
            confidence=confidence,
            evidence_refs=deduped,
            remediation=self._remediation_for_label(label),
            gaps=gaps,
        )
