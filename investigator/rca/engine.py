# ABOUTME: Implements deterministic Trace RCA narrowing and label selection over inspection APIs.
# ABOUTME: Produces evidence-linked RCA outputs with stable ordering and safe fallback behavior.

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from investigator.inspection_api import PhoenixInspectionAPI
from investigator.runtime.budget_pool import RuntimeBudgetPool
from investigator.runtime.llm_client import (
    ModelOutputInvalidError,
    OpenAIModelClient,
    RuntimeModelClient,
    StructuredGenerationRequest,
    StructuredGenerationUsage,
)
from investigator.runtime.llm_loop import run_structured_generation_loop
from investigator.runtime.prompt_registry import PromptDefinition, get_prompt_definition
from investigator.runtime.repl_loop import ReplLoop
from investigator.runtime.recursive_loop import RecursiveLoop
from investigator.runtime.recursive_planner import StructuredActionPlanner
from investigator.runtime.sandbox import SandboxGuard
from investigator.runtime.tool_registry import ToolRegistry
from investigator.runtime.contracts import (
    EvidenceRef,
    InputRef,
    RCAReport,
    RuntimeBudget,
    TimeWindow,
    hash_excerpt,
    utc_now_rfc3339,
)


@dataclass
class TraceRCARequest:
    trace_id: str
    project_name: str


_TRACE_RCA_PROMPT_ID = "rca_trace_judgment_v1"
_TRACE_RCA_PROMPT_DEFINITION = get_prompt_definition(_TRACE_RCA_PROMPT_ID)
_RECURSIVE_RCA_PROMPT_ID = "recursive_runtime_action_v1"
_REPL_RCA_PROMPT_ID = "repl_runtime_step_v1"
_ALLOWED_RCA_LABELS = {
    "retrieval_failure",
    "tool_failure",
    "instruction_failure",
    "upstream_dependency_failure",
    "data_schema_mismatch",
}
_RCA_REPL_ALLOWED_TOOLS = {
    "list_spans",
    "get_spans",
    "get_span",
    "get_children",
    "get_messages",
    "get_tool_io",
    "get_retrieval_chunks",
}
_TRACE_RCA_TIPS_PROFILES: dict[str, str] = {
    "none": "",
    "trace_rca_v1": (
        "Prefer branch-root evidence first. Focus on one likely failure family at a time. "
        "For non-trivial RCA traces, execute one llm_query before final SUBMIT; if no subquery has "
        "occurred yet, call llm_query in the same final snippet before SUBMIT. "
        "Before submitting, validate the chosen label against at least one contradictory clue "
        "from messages, tool_io, or retrieval chunks; if contradiction is unresolved, report it in gaps."
    ),
}


def list_trace_rca_tips_profiles() -> list[str]:
    return sorted(_TRACE_RCA_TIPS_PROFILES.keys())


def resolve_trace_rca_tips_profile(tips_profile: str) -> str:
    normalized_profile = str(tips_profile or "none").strip().lower() or "none"
    if normalized_profile not in _TRACE_RCA_TIPS_PROFILES:
        available_profiles = ", ".join(list_trace_rca_tips_profiles())
        raise ValueError(
            f"Unknown trace RCA tips profile: {tips_profile!r}. Available profiles: {available_profiles}"
        )
    return str(_TRACE_RCA_TIPS_PROFILES[normalized_profile]).strip()


class TraceRCAEngine:
    engine_type = "rca"
    output_contract_name = "RCAReport"
    engine_version = "0.2.0"
    model_provider = "openai"
    model_name = "gpt-5-mini"
    prompt_template_hash = _TRACE_RCA_PROMPT_DEFINITION.prompt_template_hash
    temperature = 0.0

    def __init__(
        self,
        inspection_api: Any | None = None,
        model_client: RuntimeModelClient | None = None,
        *,
        max_hot_spans: int = 5,
        max_branch_depth: int = 2,
        max_branch_nodes: int = 30,
        use_llm_judgment: bool = False,
        use_recursive_runtime: bool = False,
        use_repl_runtime: bool = False,
        recursive_budget: RuntimeBudget | None = None,
        fallback_on_llm_error: bool = False,
        repl_env_tips: str | None = None,
    ) -> None:
        self._inspection_api = inspection_api
        self._model_client = model_client
        self._max_hot_spans = max_hot_spans
        self._max_branch_depth = max_branch_depth
        self._max_branch_nodes = max_branch_nodes
        auto_repl_runtime = (
            model_client is not None
            and not use_llm_judgment
            and not use_recursive_runtime
            and not use_repl_runtime
        )
        self._use_llm_judgment = use_llm_judgment
        self._use_recursive_runtime = use_recursive_runtime
        self._use_repl_runtime = use_repl_runtime or auto_repl_runtime
        self._recursive_budget = recursive_budget or RuntimeBudget(
            max_iterations=12,
            max_depth=max_branch_depth,
            max_tool_calls=120,
            max_subcalls=40,
            max_tokens_total=200000,
        )
        self._fallback_on_llm_error = fallback_on_llm_error or auto_repl_runtime
        self._repl_env_tips = str(repl_env_tips or "").strip()
        self._prompt_definition: PromptDefinition = _TRACE_RCA_PROMPT_DEFINITION
        if self._use_llm_judgment and self._use_repl_runtime:
            self._prompt_definition = get_prompt_definition(_REPL_RCA_PROMPT_ID)
        elif self._use_llm_judgment and self._use_recursive_runtime:
            self._prompt_definition = get_prompt_definition(_RECURSIVE_RCA_PROMPT_ID)
        self.prompt_template_hash = self._prompt_definition.prompt_template_hash
        self._runtime_signals: dict[str, object] = self._base_runtime_signals()
        model_provider = str(getattr(self._model_client, "model_provider", "") or "").strip()
        if model_provider:
            self.model_provider = model_provider

    def get_runtime_signals(self) -> dict[str, object]:
        return dict(self._runtime_signals)

    def _base_runtime_signals(self) -> dict[str, object]:
        return {
            "iterations": 1,
            "depth_reached": 0,
            "tool_calls": 0,
            "llm_subcalls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "model_provider": self.model_provider,
            "rca_judgment_mode": "deterministic",
            "runtime_state": "completed",
            "budget_reason": "",
            "state_trajectory": [],
            "subcall_metadata": [],
            "repl_trajectory": [],
            "subcalls": 0,
        }

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
        has_non_schema_tool_failure = False
        has_schema_tool_failure = False
        has_upstream_failure = False
        has_schema_failure = False
        has_retrieval_failure = False
        has_instruction_failure = False

        for candidate in candidates:
            summary = candidate["summary"]
            status_text = candidate["status_text"]
            span_name = str(summary.get("name") or "").lower()
            span_kind = str(summary.get("span_kind") or "UNKNOWN")
            status_code = str(summary.get("status_code") or "UNSET")
            tool_io = candidate.get("tool_io")
            retrieval_chunks = candidate.get("retrieval_chunks") or []

            if upstream_pattern.search(status_text):
                has_upstream_failure = True
            has_schema_signal = bool(schema_pattern.search(status_text))
            if has_schema_signal:
                has_schema_failure = True
            if instruction_pattern.search(status_text):
                has_instruction_failure = True

            tool_status_error = bool(tool_io) and str(tool_io.get("status_code")) == "ERROR"
            is_tool_span = span_kind == "TOOL" or (
                span_kind == "UNKNOWN" and span_name.startswith("tool.")
            )
            if is_tool_span and (status_code == "ERROR" or tool_status_error):
                has_tool_failure = True
                if has_schema_signal:
                    has_schema_tool_failure = True
                else:
                    has_non_schema_tool_failure = True

            is_retriever_span = span_kind == "RETRIEVER" or (
                span_kind == "UNKNOWN" and span_name.startswith("retriever.")
            )
            if is_retriever_span:
                if status_code == "ERROR" or len(retrieval_chunks) == 0:
                    has_retrieval_failure = True

        if has_non_schema_tool_failure:
            return "tool_failure"
        if has_schema_tool_failure:
            return "data_schema_mismatch"
        if has_tool_failure:
            return "tool_failure"
        if has_upstream_failure:
            return "upstream_dependency_failure"
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

    def _resolve_model_client(self) -> RuntimeModelClient:
        if self._model_client is None:
            self._model_client = OpenAIModelClient()
        return self._model_client

    @staticmethod
    def _evidence_dict(evidence: EvidenceRef) -> dict[str, Any]:
        return {
            "trace_id": evidence.trace_id,
            "span_id": evidence.span_id,
            "kind": evidence.kind,
            "ref": evidence.ref,
            "excerpt_hash": evidence.excerpt_hash,
            "ts": evidence.ts,
        }

    @staticmethod
    def _candidate_snapshot(candidate: dict[str, Any]) -> dict[str, Any]:
        summary = candidate.get("summary") or {}
        tool_io = candidate.get("tool_io")
        retrieval_chunks = candidate.get("retrieval_chunks") or []
        return {
            "span_id": str(summary.get("span_id") or ""),
            "name": str(summary.get("name") or ""),
            "span_kind": str(summary.get("span_kind") or ""),
            "status_code": str(summary.get("status_code") or ""),
            "status_message": str(summary.get("status_message") or ""),
            "latency_ms": float(summary.get("latency_ms") or 0.0),
            "has_exception": bool(candidate.get("has_exception")),
            "tool_error": (
                isinstance(tool_io, dict)
                and str(tool_io.get("status_code") or "").upper() == "ERROR"
            ),
            "retrieval_chunk_count": len(retrieval_chunks),
        }

    @staticmethod
    def _delegation_policy() -> dict[str, Any]:
        return {
            "prefer_planner_driven_subcalls": True,
            "goal": "Run focused per-label hypothesis investigations before selecting primary_label.",
            "example_actions": [
                {
                    "type": "delegate_subcall",
                    "objective": "evaluate hypothesis label=tool_failure",
                    "use_planner": True,
                    "context": {
                        "candidate_label": "tool_failure",
                        "focus": "collect tool IO and child-span evidence supporting or refuting tool failure",
                    },
                },
                {
                    "type": "delegate_subcall",
                    "objective": "evaluate hypothesis label=retrieval_failure",
                    "use_planner": True,
                    "context": {
                        "candidate_label": "retrieval_failure",
                        "focus": "collect retrieval chunk quality and retriever error evidence",
                    },
                },
            ],
        }

    @staticmethod
    def _normalize_repl_hypotheses(payload: dict[str, Any]) -> list[dict[str, Any]]:
        hypotheses_payload = payload.get("hypotheses")
        if not isinstance(hypotheses_payload, list):
            return []
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in hypotheses_payload:
            if not isinstance(item, dict):
                continue
            label = str(item.get("hypothesis_label") or item.get("label") or "").strip()
            statement = str(
                item.get("hypothesis_statement") or item.get("statement") or item.get("summary") or ""
            ).strip()
            if label not in _ALLOWED_RCA_LABELS or not statement:
                continue
            relevant_span_ids_raw = item.get("relevant_span_ids")
            relevant_span_ids: list[str] = []
            if isinstance(relevant_span_ids_raw, list):
                relevant_span_ids = [
                    str(span_id).strip() for span_id in relevant_span_ids_raw if str(span_id).strip()
                ]
            investigation_tools_raw = item.get("investigation_tools")
            investigation_tools: list[str] = []
            if isinstance(investigation_tools_raw, list):
                investigation_tools = [
                    str(tool_name).strip()
                    for tool_name in investigation_tools_raw
                    if str(tool_name).strip()
                ]
            key = (label, statement)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "hypothesis_label": label,
                    "hypothesis_statement": statement,
                    "relevant_span_ids": relevant_span_ids,
                    "investigation_tools": investigation_tools,
                }
            )
        return normalized

    @staticmethod
    def _coerce_payload_evidence_refs(
        evidence_payload: Any,
        *,
        trace_id: str,
        default_span_id: str,
    ) -> list[EvidenceRef]:
        if not isinstance(evidence_payload, list):
            return []
        supported_kinds = {"SPAN", "TOOL_IO", "RETRIEVAL_CHUNK", "MESSAGE", "CONFIG_DIFF"}
        evidence_refs: list[EvidenceRef] = []
        for item in evidence_payload:
            if not isinstance(item, dict):
                continue
            evidence_trace_id = str(item.get("trace_id") or trace_id).strip() or trace_id
            span_id = str(item.get("span_id") or default_span_id).strip() or default_span_id
            ref = str(item.get("ref") or span_id).strip() or span_id
            kind_candidate = str(item.get("kind") or "SPAN").strip().upper()
            kind = kind_candidate if kind_candidate in supported_kinds else "SPAN"
            excerpt_hash = str(item.get("excerpt_hash") or hash_excerpt(ref)).strip() or hash_excerpt(ref)
            ts_raw = item.get("ts")
            ts = str(ts_raw).strip() if isinstance(ts_raw, str) and str(ts_raw).strip() else None
            evidence_refs.append(
                EvidenceRef(
                    trace_id=evidence_trace_id,
                    span_id=span_id,
                    kind=kind,  # type: ignore[arg-type]
                    ref=ref,
                    excerpt_hash=excerpt_hash,
                    ts=ts,
                )
            )
        return evidence_refs

    @staticmethod
    def _hypothesis_sort_key(result: dict[str, Any]) -> tuple[int, float, int, int]:
        status_rank = 1 if str(result.get("status") or "") == "succeeded" else 0
        confidence = float(result.get("confidence") or 0.0)
        evidence_refs = result.get("evidence_refs") or []
        independent_refs = 0
        if isinstance(evidence_refs, list):
            independent_refs = len(
                {
                    (str(getattr(item, "kind", "")), str(getattr(item, "ref", "")))
                    for item in evidence_refs
                    if isinstance(item, EvidenceRef)
                }
            )
        supporting_facts = result.get("supporting_facts") or []
        supporting_count = len(supporting_facts) if isinstance(supporting_facts, list) else 0
        return status_rank, confidence, independent_refs, supporting_count

    def _run_repl_hypothesis_subcalls(
        self,
        *,
        request: TraceRCARequest,
        inspection_api: Any,
        hypotheses: list[dict[str, Any]],
        deterministic_label: str,
        evidence_refs: list[EvidenceRef],
    ) -> tuple[
        list[dict[str, Any]],
        list[EvidenceRef],
        list[str],
        list[dict[str, Any]],
        dict[str, float | int],
    ]:
        if not hypotheses:
            return [], [], [], [], {
                "iterations": 0,
                "depth_reached": 0,
                "tool_calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
                "llm_subcalls": 0,
            }

        model_client = self._resolve_model_client()
        planner = StructuredActionPlanner(
            client=model_client,
            model_name=self.model_name,
            temperature=self.temperature,
            prompt_id=_RECURSIVE_RCA_PROMPT_ID,
        )
        tool_registry = ToolRegistry(inspection_api=inspection_api)
        sandbox_guard = SandboxGuard(allowed_tools=tool_registry.allowed_tools)
        recursive_loop = RecursiveLoop(tool_registry=tool_registry, sandbox_guard=sandbox_guard)
        budget_pool = RuntimeBudgetPool(budget=self._recursive_budget)

        hypothesis_results: list[dict[str, Any]] = []
        hypothesis_evidence_refs: list[EvidenceRef] = []
        hypothesis_gaps: list[str] = []
        subcall_metadata: list[dict[str, Any]] = []
        usage_totals: dict[str, float | int] = {
            "iterations": 0,
            "depth_reached": 0,
            "tool_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "llm_subcalls": 0,
        }
        parent_call_id = f"trace_rca:{request.trace_id}"
        total_hypotheses = len(hypotheses)

        for index, hypothesis in enumerate(hypotheses):
            hypothesis_label = str(hypothesis.get("hypothesis_label") or "").strip()
            hypothesis_statement = str(hypothesis.get("hypothesis_statement") or "").strip()
            relevant_span_ids = [
                str(item).strip()
                for item in (hypothesis.get("relevant_span_ids") or [])
                if str(item).strip()
            ]
            investigation_tools = [
                str(item).strip()
                for item in (hypothesis.get("investigation_tools") or [])
                if str(item).strip()
            ]
            if not hypothesis_label or not hypothesis_statement:
                continue

            planner_seed = {
                "engine_type": self.engine_type,
                "trace_id": request.trace_id,
                "allowed_labels": sorted(_ALLOWED_RCA_LABELS),
                "deterministic_label_hint": deterministic_label,
                "hypothesis": {
                    "label": hypothesis_label,
                    "statement": hypothesis_statement,
                    "relevant_span_ids": relevant_span_ids,
                    "investigation_tools": investigation_tools,
                },
                "candidate_label": hypothesis_label,
                "focus": hypothesis_statement,
                "relevant_span_ids": relevant_span_ids,
                "investigation_tools": investigation_tools,
                "evidence_refs": [self._evidence_dict(ref) for ref in evidence_refs[:20]],
                "delegation_policy": self._delegation_policy(),
            }

            def _planner(context: dict[str, Any], *, planner_seed: dict[str, Any] = planner_seed) -> dict[str, Any]:
                merged_context = dict(context)
                merged_context.update(planner_seed)
                return planner(merged_context)

            remaining_siblings = max(1, total_hypotheses - index)
            sub_budget = budget_pool.allocate_run_budget(sibling_count=remaining_siblings)
            subcall_id = f"hypothesis_{index + 1:03d}"
            started_at = utc_now_rfc3339()
            loop_result = recursive_loop.run(
                actions=[],
                planner=_planner,
                budget=sub_budget,
                budget_pool=budget_pool,
                delegation_context={
                    "trace_id": request.trace_id,
                    "candidate_label": hypothesis_label,
                    "focus": hypothesis_statement,
                },
                depth=1,
                objective=hypothesis_statement,
                parent_call_id=parent_call_id,
                input_ref_hash=hash_excerpt(
                    f"{request.trace_id}:{hypothesis_label}:{hypothesis_statement}:{index}"
                ),
            )
            completed_at = utc_now_rfc3339()

            usage_totals["iterations"] = int(usage_totals["iterations"]) + int(loop_result.usage.iterations)
            usage_totals["depth_reached"] = max(
                int(usage_totals["depth_reached"]),
                int(loop_result.usage.depth_reached),
            )
            usage_totals["tool_calls"] = int(usage_totals["tool_calls"]) + int(loop_result.usage.tool_calls)
            usage_totals["tokens_in"] = int(usage_totals["tokens_in"]) + int(loop_result.usage.tokens_in)
            usage_totals["tokens_out"] = int(usage_totals["tokens_out"]) + int(loop_result.usage.tokens_out)
            usage_totals["cost_usd"] = float(usage_totals["cost_usd"]) + float(loop_result.usage.cost_usd)
            usage_totals["llm_subcalls"] = int(usage_totals["llm_subcalls"]) + 1

            status = "succeeded"
            if loop_result.status == "failed":
                status = "failed"
            elif loop_result.status == "terminated_budget":
                status = "terminated_budget"

            subcall_metadata.append(
                {
                    "parent_call_id": parent_call_id,
                    "call_id": subcall_id,
                    "depth": 1,
                    "objective": hypothesis_statement,
                    "input_ref_hash": hash_excerpt(
                        f"{request.trace_id}:{hypothesis_label}:{hypothesis_statement}:{index}"
                    ),
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "status": status,
                    "hypothesis_label": hypothesis_label,
                }
            )
            subcall_metadata.extend(loop_result.subcall_metadata)

            result_payload = loop_result.output if isinstance(loop_result.output, dict) else {}
            result_label = str(
                result_payload.get("label") or result_payload.get("primary_label") or hypothesis_label
            ).strip()
            if result_label not in _ALLOWED_RCA_LABELS:
                result_label = hypothesis_label if hypothesis_label in _ALLOWED_RCA_LABELS else deterministic_label

            confidence_raw = result_payload.get("confidence")
            if isinstance(confidence_raw, (int, float)):
                confidence = max(0.0, min(1.0, float(confidence_raw)))
            else:
                confidence = self._confidence_from_evidence(result_label, evidence_refs)

            supporting_facts_raw = result_payload.get("supporting_facts")
            supporting_facts: list[str] = []
            if isinstance(supporting_facts_raw, list):
                supporting_facts = [
                    str(item).strip() for item in supporting_facts_raw if str(item).strip()
                ]
            elif isinstance(supporting_facts_raw, str) and supporting_facts_raw.strip():
                supporting_facts = [supporting_facts_raw.strip()]
            else:
                summary_text = str(result_payload.get("summary") or "").strip()
                if summary_text:
                    supporting_facts = [summary_text]

            default_span_id = relevant_span_ids[0] if relevant_span_ids else "root-span"
            result_evidence_refs = self._coerce_payload_evidence_refs(
                result_payload.get("evidence_refs"),
                trace_id=request.trace_id,
                default_span_id=default_span_id,
            )
            if not result_evidence_refs and evidence_refs:
                result_evidence_refs = [evidence_refs[0]]
            hypothesis_evidence_refs.extend(result_evidence_refs)

            result_gaps_raw = result_payload.get("gaps")
            result_gaps: list[str] = []
            if isinstance(result_gaps_raw, list):
                result_gaps = [str(item).strip() for item in result_gaps_raw if str(item).strip()]
            if loop_result.status == "terminated_budget":
                result_gaps.append(
                    loop_result.budget_reason or "Hypothesis subcall reached budget limits."
                )
            if loop_result.status == "failed":
                result_gaps.append(loop_result.error_message or "Hypothesis subcall failed.")
            hypothesis_gaps.extend([f"{hypothesis_label}: {gap}" for gap in result_gaps if gap])

            hypothesis_results.append(
                {
                    "label": result_label,
                    "confidence": confidence,
                    "supporting_facts": supporting_facts,
                    "evidence_refs": result_evidence_refs,
                    "gaps": result_gaps,
                    "status": status,
                    "statement": hypothesis_statement,
                }
            )

        deduped_hypothesis_evidence: list[EvidenceRef] = []
        seen_hypothesis_evidence: set[tuple[str, str]] = set()
        for evidence in hypothesis_evidence_refs:
            key = (evidence.kind, evidence.ref)
            if key in seen_hypothesis_evidence:
                continue
            deduped_hypothesis_evidence.append(evidence)
            seen_hypothesis_evidence.add(key)
        return (
            hypothesis_results,
            deduped_hypothesis_evidence,
            hypothesis_gaps,
            subcall_metadata,
            usage_totals,
        )

    @staticmethod
    def _tighten_repl_trace_budget(loop_budget: RuntimeBudget) -> RuntimeBudget:
        max_tokens_total = loop_budget.max_tokens_total
        if max_tokens_total is not None:
            max_tokens_total = max(4000, min(int(max_tokens_total), 18000))
        max_cost_usd = loop_budget.max_cost_usd
        if max_cost_usd is not None:
            max_cost_usd = min(float(max_cost_usd), 0.12)
        max_iterations = max(2, min(int(loop_budget.max_iterations), 5))
        return RuntimeBudget(
            max_iterations=max_iterations,
            max_depth=max(0, int(loop_budget.max_depth)),
            max_tool_calls=max(1, min(int(loop_budget.max_tool_calls), 6)),
            max_subcalls=max(1, min(int(loop_budget.max_subcalls), 5)),
            max_tokens_total=max_tokens_total,
            max_cost_usd=max_cost_usd,
            sampling_seed=loop_budget.sampling_seed,
            max_wall_time_sec=max(20, min(int(loop_budget.max_wall_time_sec), 45)),
        )

    def _repl_llm_rca_judgment(
        self,
        *,
        request: TraceRCARequest,
        inspection_api: Any,
        hot_candidates: list[dict[str, Any]],
        deterministic_label: str,
        evidence_refs: list[EvidenceRef],
    ) -> tuple[str, str, float, list[str], list[str], list[EvidenceRef]]:
        model_client = self._resolve_model_client()
        model_provider = str(getattr(model_client, "model_provider", self.model_provider) or "openai")
        self.prompt_template_hash = get_prompt_definition(_REPL_RCA_PROMPT_ID).prompt_template_hash
        self._runtime_signals["model_provider"] = model_provider
        self._runtime_signals["rca_judgment_mode"] = "repl_llm"

        repl_loop = ReplLoop(
            tool_registry=ToolRegistry(
                inspection_api=inspection_api,
                allowed_tools=_RCA_REPL_ALLOWED_TOOLS,
            ),
            model_client=model_client,
            model_name=self.model_name,
            temperature=self.temperature,
        )
        tuned_loop_budget = self._tighten_repl_trace_budget(self._recursive_budget)
        candidate_summaries = [
            self._candidate_snapshot(candidate) for candidate in hot_candidates[: self._max_hot_spans]
        ]
        pre_filter_context = {
            "hot_spans": candidate_summaries,
            "branch_span_ids": [
                str(item.get("span_id") or "")
                for item in candidate_summaries
                if str(item.get("span_id") or "")
            ],
            "preliminary_label": deterministic_label,
        }
        loop_result = repl_loop.run(
            objective=f"trace_rca repl investigation for trace {request.trace_id}",
            input_vars={
                "trace_id": request.trace_id,
                "allowed_labels": sorted(_ALLOWED_RCA_LABELS),
                "deterministic_label_hint": deterministic_label,
                "candidate_summaries": candidate_summaries,
                "evidence_seed": [self._evidence_dict(ref) for ref in evidence_refs[:20]],
            },
            pre_filter_context=pre_filter_context,
            env_tips=self._repl_env_tips or None,
            budget=tuned_loop_budget,
            require_subquery_for_non_trivial=True,
        )

        self._runtime_signals["iterations"] = max(1, int(loop_result.usage.iterations))
        self._runtime_signals["depth_reached"] = int(loop_result.usage.depth_reached)
        self._runtime_signals["tool_calls"] = int(loop_result.usage.tool_calls)
        self._runtime_signals["llm_subcalls"] = int(loop_result.usage.llm_subcalls)
        self._runtime_signals["tokens_in"] = int(loop_result.usage.tokens_in)
        self._runtime_signals["tokens_out"] = int(loop_result.usage.tokens_out)
        self._runtime_signals["cost_usd"] = float(loop_result.usage.cost_usd)
        self._runtime_signals["runtime_state"] = str(loop_result.status)
        self._runtime_signals["budget_reason"] = str(loop_result.budget_reason or "")
        self._runtime_signals["state_trajectory"] = list(loop_result.state_trajectory)
        self._runtime_signals["subcall_metadata"] = []
        self._runtime_signals["repl_trajectory"] = list(loop_result.repl_trajectory)
        self._runtime_signals["subcalls"] = int(loop_result.usage.llm_subcalls)

        if loop_result.error_code == "SANDBOX_VIOLATION":
            self._runtime_signals["sandbox_violations"] = [str(loop_result.error_message or "")]

        if loop_result.status == "failed":
            if loop_result.error_code == "MODEL_OUTPUT_INVALID":
                raise ModelOutputInvalidError(
                    loop_result.error_message or "REPL RCA runtime output was invalid."
                )
            raise RuntimeError(loop_result.error_message or "REPL RCA runtime failed.")
        if loop_result.status == "terminated_budget":
            raise RuntimeError(
                loop_result.budget_reason
                or "REPL RCA runtime reached budget limits before finalize."
            )

        payload = loop_result.output if isinstance(loop_result.output, dict) else {}
        label_candidate = str(payload.get("primary_label") or "").strip()
        label = label_candidate if label_candidate in _ALLOWED_RCA_LABELS else deterministic_label

        summary = str(payload.get("summary") or "").strip()
        if not summary:
            if loop_result.status == "terminated_budget":
                summary = (
                    "REPL RCA runtime reached budget limits before finalize; "
                    f"deterministic label {label} was retained."
                )
            else:
                summary = f"REPL RCA judgment selected label {label}."

        confidence_raw = payload.get("confidence")
        if isinstance(confidence_raw, (int, float)):
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        else:
            confidence = self._confidence_from_evidence(label, evidence_refs)

        remediation_payload = payload.get("remediation")
        remediation: list[str] = []
        if isinstance(remediation_payload, list):
            remediation = [str(item).strip() for item in remediation_payload if str(item).strip()]
        elif isinstance(remediation_payload, str) and remediation_payload.strip():
            remediation = [remediation_payload.strip()]
        if not remediation:
            remediation = self._remediation_for_label(label)

        llm_gaps: list[str] = []
        gaps_payload = payload.get("gaps")
        if isinstance(gaps_payload, list):
            llm_gaps.extend([str(item).strip() for item in gaps_payload if str(item).strip()])
        if loop_result.status == "terminated_budget":
            llm_gaps.append(
                loop_result.budget_reason
                or "REPL runtime reached a budget limit before finalize."
            )
        llm_evidence_refs: list[EvidenceRef] = []
        hypotheses = self._normalize_repl_hypotheses(payload)
        if hypotheses:
            (
                hypothesis_results,
                hypothesis_evidence_refs,
                hypothesis_gaps,
                hypothesis_subcall_metadata,
                hypothesis_usage,
            ) = self._run_repl_hypothesis_subcalls(
                request=request,
                inspection_api=inspection_api,
                hypotheses=hypotheses,
                deterministic_label=deterministic_label,
                evidence_refs=evidence_refs,
            )
            llm_evidence_refs.extend(hypothesis_evidence_refs)
            llm_gaps.extend(hypothesis_gaps)
            self._runtime_signals["iterations"] = int(self._runtime_signals["iterations"]) + int(
                hypothesis_usage["iterations"]
            )
            self._runtime_signals["depth_reached"] = max(
                int(self._runtime_signals["depth_reached"]),
                int(hypothesis_usage["depth_reached"]),
            )
            self._runtime_signals["tool_calls"] = int(self._runtime_signals["tool_calls"]) + int(
                hypothesis_usage["tool_calls"]
            )
            self._runtime_signals["llm_subcalls"] = int(self._runtime_signals["llm_subcalls"]) + int(
                hypothesis_usage["llm_subcalls"]
            )
            self._runtime_signals["tokens_in"] = int(self._runtime_signals["tokens_in"]) + int(
                hypothesis_usage["tokens_in"]
            )
            self._runtime_signals["tokens_out"] = int(self._runtime_signals["tokens_out"]) + int(
                hypothesis_usage["tokens_out"]
            )
            self._runtime_signals["cost_usd"] = float(self._runtime_signals["cost_usd"]) + float(
                hypothesis_usage["cost_usd"]
            )
            self._runtime_signals["subcall_metadata"] = hypothesis_subcall_metadata
            self._runtime_signals["subcalls"] = len(hypothesis_subcall_metadata)
            if hypothesis_results:
                best_result = max(hypothesis_results, key=self._hypothesis_sort_key)
                label = str(best_result.get("label") or label).strip()
                if label not in _ALLOWED_RCA_LABELS:
                    label = deterministic_label
                confidence = max(
                    0.0,
                    min(
                        1.0,
                        float(best_result.get("confidence") or self._confidence_from_evidence(label, evidence_refs)),
                    ),
                )
                supporting_facts = best_result.get("supporting_facts") or []
                supporting_text = "; ".join(
                    [str(item).strip() for item in supporting_facts if str(item).strip()]
                )
                considered = ", ".join(
                    [
                        f"{str(item.get('label') or '')}:{float(item.get('confidence') or 0.0):.2f}"
                        for item in hypothesis_results
                    ]
                )
                summary = (
                    f"Hypothesis subcalls selected label {label}. "
                    f"Supporting facts: {supporting_text or 'none provided'}. "
                    f"Considered: {considered}."
                )
                remediation = self._remediation_for_label(label)
        return label, summary, confidence, remediation, llm_gaps, llm_evidence_refs

    def _recursive_llm_rca_judgment(
        self,
        *,
        request: TraceRCARequest,
        inspection_api: Any,
        hot_candidates: list[dict[str, Any]],
        deterministic_label: str,
        evidence_refs: list[EvidenceRef],
    ) -> tuple[str, str, float, list[str], list[str]]:
        model_client = self._resolve_model_client()
        model_provider = str(getattr(model_client, "model_provider", self.model_provider) or "openai")
        planner = StructuredActionPlanner(
            client=model_client,
            model_name=self.model_name,
            temperature=self.temperature,
            prompt_id=_RECURSIVE_RCA_PROMPT_ID,
        )
        self.prompt_template_hash = planner.prompt_template_hash
        self._runtime_signals["model_provider"] = model_provider
        self._runtime_signals["rca_judgment_mode"] = "recursive_llm"

        tool_registry = ToolRegistry(inspection_api=inspection_api)
        sandbox_guard = SandboxGuard(allowed_tools=tool_registry.allowed_tools)
        recursive_loop = RecursiveLoop(tool_registry=tool_registry, sandbox_guard=sandbox_guard)
        planner_seed = {
            "engine_type": self.engine_type,
            "trace_id": request.trace_id,
            "deterministic_label_hint": deterministic_label,
            "allowed_labels": sorted(_ALLOWED_RCA_LABELS),
            "candidate_summaries": [
                self._candidate_snapshot(candidate)
                for candidate in hot_candidates[: self._max_hot_spans]
            ],
            "evidence_refs": [self._evidence_dict(ref) for ref in evidence_refs[:20]],
            "delegation_policy": self._delegation_policy(),
        }

        def _planner(context: dict[str, Any]) -> dict[str, Any]:
            merged_context = dict(context)
            merged_context.update(planner_seed)
            return planner(merged_context)

        loop_result = recursive_loop.run(
            actions=[],
            planner=_planner,
            budget=self._recursive_budget,
            objective=f"trace_rca recursive investigation for trace {request.trace_id}",
            parent_call_id=f"trace_rca:{request.trace_id}",
            input_ref_hash=hash_excerpt(request.trace_id),
        )

        self._runtime_signals["iterations"] = max(1, int(loop_result.usage.iterations))
        self._runtime_signals["depth_reached"] = int(loop_result.usage.depth_reached)
        self._runtime_signals["tool_calls"] = int(loop_result.usage.tool_calls)
        self._runtime_signals["tokens_in"] = int(loop_result.usage.tokens_in)
        self._runtime_signals["tokens_out"] = int(loop_result.usage.tokens_out)
        self._runtime_signals["cost_usd"] = float(loop_result.usage.cost_usd)
        self._runtime_signals["runtime_state"] = str(loop_result.status)
        self._runtime_signals["budget_reason"] = str(loop_result.budget_reason or "")
        self._runtime_signals["state_trajectory"] = list(loop_result.state_trajectory)
        self._runtime_signals["subcall_metadata"] = list(loop_result.subcall_metadata)
        self._runtime_signals["subcalls"] = len(loop_result.subcall_metadata)

        if loop_result.error_code == "SANDBOX_VIOLATION":
            self._runtime_signals["sandbox_violations"] = [str(loop_result.error_message or "")]

        if loop_result.status == "failed":
            if loop_result.error_code == "MODEL_OUTPUT_INVALID":
                raise ModelOutputInvalidError(
                    loop_result.error_message or "Recursive planner output was invalid."
                )
            raise RuntimeError(loop_result.error_message or "Recursive planner run failed.")

        payload = loop_result.output if isinstance(loop_result.output, dict) else {}
        label_candidate = str(payload.get("primary_label") or "").strip()
        label = label_candidate if label_candidate in _ALLOWED_RCA_LABELS else deterministic_label

        summary = str(payload.get("summary") or "").strip()
        if not summary:
            if loop_result.status == "terminated_budget":
                summary = (
                    "Recursive RCA runtime reached budget limits before finalize; "
                    f"deterministic label {label} was retained."
                )
            else:
                summary = f"Recursive RCA judgment selected label {label}."

        confidence_raw = payload.get("confidence")
        if isinstance(confidence_raw, (int, float)):
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        else:
            confidence = self._confidence_from_evidence(label, evidence_refs)

        remediation_payload = payload.get("remediation")
        remediation: list[str] = []
        if isinstance(remediation_payload, list):
            remediation = [str(item).strip() for item in remediation_payload if str(item).strip()]
        elif isinstance(remediation_payload, str) and remediation_payload.strip():
            remediation = [remediation_payload.strip()]
        if not remediation:
            remediation = self._remediation_for_label(label)

        llm_gaps: list[str] = []
        gaps_payload = payload.get("gaps")
        if isinstance(gaps_payload, list):
            llm_gaps.extend([str(item).strip() for item in gaps_payload if str(item).strip()])
        if loop_result.status == "terminated_budget":
            llm_gaps.append(
                loop_result.budget_reason
                or "Recursive runtime reached a budget limit before finalize."
            )
        return label, summary, confidence, remediation, llm_gaps

    def _llm_rca_judgment(
        self,
        *,
        request: TraceRCARequest,
        hot_candidates: list[dict[str, Any]],
        deterministic_label: str,
        evidence_refs: list[EvidenceRef],
    ) -> tuple[str, str, float, list[str], list[str]]:
        prompt_payload = {
            "trace_id": request.trace_id,
            "deterministic_label_hint": deterministic_label,
            "candidate_summaries": [
                self._candidate_snapshot(candidate)
                for candidate in hot_candidates[: self._max_hot_spans]
            ],
            "evidence_refs": [self._evidence_dict(ref) for ref in evidence_refs[:20]],
        }
        model_client = self._resolve_model_client()
        model_provider = str(getattr(model_client, "model_provider", self.model_provider) or "openai")
        request_payload = StructuredGenerationRequest(
            model_provider=model_provider,
            model_name=self.model_name,
            temperature=self.temperature,
            system_prompt=self._prompt_definition.prompt_text,
            user_prompt=json.dumps(prompt_payload, sort_keys=True),
            response_schema_name=self._prompt_definition.prompt_id,
            response_schema=self._prompt_definition.response_schema,
        )
        try:
            loop_result = run_structured_generation_loop(
                client=model_client,
                request=request_payload,
            )
        except ModelOutputInvalidError as exc:
            usage = getattr(exc, "usage", None)
            if isinstance(usage, StructuredGenerationUsage):
                self._runtime_signals["tokens_in"] = int(usage.tokens_in)
                self._runtime_signals["tokens_out"] = int(usage.tokens_out)
                self._runtime_signals["cost_usd"] = float(usage.cost_usd)
            self._runtime_signals["iterations"] = int(getattr(exc, "attempt_count", 1))
            self._runtime_signals["model_provider"] = model_provider
            raise
        self._runtime_signals["iterations"] = max(1, int(loop_result.attempt_count))
        self._runtime_signals["tokens_in"] = int(loop_result.usage.tokens_in)
        self._runtime_signals["tokens_out"] = int(loop_result.usage.tokens_out)
        self._runtime_signals["cost_usd"] = float(loop_result.usage.cost_usd)
        self._runtime_signals["model_provider"] = model_provider
        self._runtime_signals["rca_judgment_mode"] = "llm"

        payload = loop_result.output
        label_candidate = str(payload.get("primary_label") or "").strip()
        label = label_candidate if label_candidate in _ALLOWED_RCA_LABELS else deterministic_label
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            summary = f"Structured RCA judgment selected label {label}."
        confidence_raw = payload.get("confidence")
        if isinstance(confidence_raw, (int, float)):
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        else:
            confidence = self._confidence_from_evidence(label, evidence_refs)
        remediation_payload = payload.get("remediation")
        remediation: list[str] = []
        if isinstance(remediation_payload, list):
            remediation = [str(item).strip() for item in remediation_payload if str(item).strip()]
        elif isinstance(remediation_payload, str) and remediation_payload.strip():
            remediation = [remediation_payload.strip()]
        if not remediation:
            remediation = self._remediation_for_label(label)
        gaps_payload = payload.get("gaps")
        llm_gaps: list[str] = []
        if isinstance(gaps_payload, list):
            llm_gaps = [str(item).strip() for item in gaps_payload if str(item).strip()]
        return label, summary, confidence, remediation, llm_gaps

    def run(self, request: TraceRCARequest) -> RCAReport:
        self._runtime_signals = self._base_runtime_signals()
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

        self._runtime_signals["depth_reached"] = min(
            self._max_branch_depth,
            1 if recursive_candidates else 0,
        )
        confidence = self._confidence_from_evidence(label, deduped)
        hot_span_ids = [str(candidate["summary"].get("span_id") or "") for candidate in hot_candidates]
        summary = (
            f"Deterministic narrowing and recursive branch inspection selected spans {hot_span_ids}; "
            f"RCA labeled trace as {label}."
        )
        remediation = self._remediation_for_label(label)
        runtime_mode: str | None = None
        if self._use_repl_runtime:
            runtime_mode = "repl"
        elif self._use_recursive_runtime and self._use_llm_judgment:
            runtime_mode = "recursive"
        elif self._use_llm_judgment:
            runtime_mode = "llm"
        llm_evidence_refs: list[EvidenceRef] = []
        if runtime_mode is not None:
            try:
                if runtime_mode == "repl":
                    (
                        label,
                        summary,
                        confidence,
                        remediation,
                        llm_gaps,
                        llm_evidence_refs,
                    ) = self._repl_llm_rca_judgment(
                        request=request,
                        inspection_api=inspection_api,
                        hot_candidates=hot_candidates,
                        deterministic_label=label,
                        evidence_refs=deduped,
                    )
                elif runtime_mode == "recursive":
                    label, summary, confidence, remediation, llm_gaps = self._recursive_llm_rca_judgment(
                        request=request,
                        inspection_api=inspection_api,
                        hot_candidates=hot_candidates,
                        deterministic_label=label,
                        evidence_refs=deduped,
                    )
                else:
                    label, summary, confidence, remediation, llm_gaps = self._llm_rca_judgment(
                        request=request,
                        hot_candidates=hot_candidates,
                        deterministic_label=label,
                        evidence_refs=deduped,
                    )
                if llm_gaps:
                    gaps.extend(llm_gaps)
                for llm_evidence in llm_evidence_refs:
                    key = (llm_evidence.kind, llm_evidence.ref)
                    if key in seen:
                        continue
                    deduped.append(llm_evidence)
                    seen.add(key)
            except Exception as exc:
                if not self._fallback_on_llm_error:
                    raise
                self._runtime_signals["rca_judgment_mode"] = "deterministic_fallback"
                mode_name = {
                    "repl": "REPL RCA judgment",
                    "recursive": "Recursive RCA judgment",
                    "llm": "LLM RCA judgment",
                }.get(runtime_mode, "RCA judgment")
                gaps.append(f"{mode_name} failed and deterministic fallback was used: {exc}")
        return RCAReport(
            trace_id=request.trace_id,
            primary_label=label,
            summary=summary,
            confidence=confidence,
            evidence_refs=deduped,
            remediation=remediation,
            gaps=gaps,
        )
