# ABOUTME: Implements Policy-to-Trace Compliance control scoping, evidence synthesis, and optional LLM judgment.
# ABOUTME: Produces controls-version traceable findings with evidence refs over the Inspection API.

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
from investigator.runtime.recursive_loop import RecursiveLoop, RecursiveLoopResult
from investigator.runtime.recursive_planner import StructuredActionPlanner
from investigator.runtime.sandbox import SandboxGuard
from investigator.runtime.tool_registry import ToolRegistry
from investigator.runtime.contracts import (
    ComplianceFinding,
    ComplianceReport,
    EvidenceRef,
    InputRef,
    RuntimeBudget,
    TimeWindow,
    hash_excerpt,
)


@dataclass
class PolicyComplianceRequest:
    trace_id: str
    project_name: str
    controls_version: str
    control_scope_override: list[str] | None = None


_POLICY_COMPLIANCE_PROMPT_ID = "policy_compliance_judgment_v1"
_POLICY_COMPLIANCE_PROMPT = get_prompt_definition(_POLICY_COMPLIANCE_PROMPT_ID)
_RECURSIVE_COMPLIANCE_PROMPT_ID = "recursive_runtime_action_v1"
_REPL_COMPLIANCE_PROMPT_ID = "repl_runtime_step_v1"
_ALLOWED_CONTROL_VERDICTS = {"pass", "fail", "not_applicable", "insufficient_evidence"}


class PolicyComplianceEngine:
    engine_type = "policy_compliance"
    output_contract_name = "ComplianceReport"
    engine_version = "0.4.0"
    model_provider = "openai"
    model_name = "gpt-5-mini"
    prompt_template_hash = _POLICY_COMPLIANCE_PROMPT.prompt_template_hash
    temperature = 0.0

    _SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    def __init__(
        self,
        inspection_api: Any | None = None,
        model_client: RuntimeModelClient | None = None,
        *,
        max_controls: int = 50,
        max_prompt_records: int = 20,
        use_llm_judgment: bool = False,
        use_recursive_runtime: bool = False,
        use_repl_runtime: bool = False,
        recursive_budget: RuntimeBudget | None = None,
        fallback_on_llm_error: bool = False,
    ) -> None:
        self._inspection_api = inspection_api
        self._model_client = model_client
        self._max_controls = max_controls
        self._max_prompt_records = max_prompt_records
        self._use_llm_judgment = use_llm_judgment
        self._use_recursive_runtime = use_recursive_runtime
        self._use_repl_runtime = use_repl_runtime
        self._recursive_budget = recursive_budget or RuntimeBudget(
            max_iterations=20,
            max_depth=2,
            max_tool_calls=160,
            max_subcalls=60,
            max_tokens_total=260000,
        )
        self._fallback_on_llm_error = fallback_on_llm_error
        self._prompt_definition: PromptDefinition = _POLICY_COMPLIANCE_PROMPT
        if self._use_llm_judgment and self._use_repl_runtime:
            self._prompt_definition = get_prompt_definition(_REPL_COMPLIANCE_PROMPT_ID)
        elif self._use_llm_judgment and self._use_recursive_runtime:
            self._prompt_definition = get_prompt_definition(_RECURSIVE_COMPLIANCE_PROMPT_ID)
        self.prompt_template_hash = self._prompt_definition.prompt_template_hash
        model_provider = str(getattr(self._model_client, "model_provider", "") or "").strip()
        if model_provider:
            self.model_provider = model_provider
        self._runtime_signals: dict[str, object] = {}
        self._reset_runtime_signals()

    def _reset_runtime_signals(self) -> None:
        self._runtime_signals = {
            "iterations": 1,
            "depth_reached": 0,
            "tool_calls": 0,
            "llm_subcalls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "model_provider": self.model_provider,
            "compliance_judgment_mode": "deterministic",
            "runtime_state": "completed",
            "budget_reason": "",
            "state_trajectory": [],
            "subcall_metadata": [],
            "repl_trajectory": [],
            "subcalls": 0,
        }

    def get_runtime_signals(self) -> dict[str, object]:
        return dict(self._runtime_signals)

    def build_input_ref(self, request: PolicyComplianceRequest) -> InputRef:
        return InputRef(
            project_name=request.project_name,
            trace_ids=[request.trace_id],
            time_window=TimeWindow(),
            filter_expr=None,
            controls_version=request.controls_version,
        )

    def _resolve_inspection_api(self, request: PolicyComplianceRequest) -> Any:
        if self._inspection_api is not None:
            return self._inspection_api
        return PhoenixInspectionAPI(project_name=request.project_name)

    @classmethod
    def _sort_controls(cls, controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            controls,
            key=lambda control: (
                -cls._SEVERITY_ORDER.get(str(control.get("severity", "low")).lower(), 0),
                str(control.get("control_id") or ""),
            ),
        )

    @staticmethod
    def _infer_app_type(spans: list[dict[str, Any]]) -> str | None:
        kinds = {str(span.get("span_kind") or "") for span in spans}
        if "AGENT" in kinds or "CHAIN" in kinds:
            return "agentic"
        return None

    @staticmethod
    def _infer_tools_used(spans: list[dict[str, Any]]) -> list[str]:
        scoped_kinds = {"TOOL", "RETRIEVER", "LLM", "UNKNOWN"}
        tools = {
            str(span.get("name") or "")
            for span in spans
            if str(span.get("span_kind") or "") in scoped_kinds and str(span.get("name") or "")
        }
        return sorted(tools)

    @staticmethod
    def _root_evidence_ref(trace_id: str, spans: list[dict[str, Any]]) -> EvidenceRef:
        if spans:
            selected = sorted(
                spans,
                key=lambda span: (
                    str(span.get("start_time") or ""),
                    str(span.get("span_id") or ""),
                ),
            )[0]
            span_id = str(selected.get("span_id") or "root-span")
            evidence_text = f"{selected.get('name') or ''}:{selected.get('status_code') or ''}"
            return EvidenceRef(
                trace_id=trace_id,
                span_id=span_id,
                kind="SPAN",
                ref=span_id,
                excerpt_hash=hash_excerpt(evidence_text or span_id),
                ts=selected.get("start_time"),
            )
        return EvidenceRef(
            trace_id=trace_id,
            span_id="root-span",
            kind="SPAN",
            ref="root-span",
            excerpt_hash=hash_excerpt("root-span"),
            ts=None,
        )

    @staticmethod
    def _dedupe_evidence_refs(evidence_refs: list[EvidenceRef]) -> list[EvidenceRef]:
        deduped: list[EvidenceRef] = []
        seen: set[tuple[str, str]] = set()
        for evidence in evidence_refs:
            key = (evidence.kind, evidence.ref)
            if key in seen:
                continue
            deduped.append(evidence)
            seen.add(key)
        return deduped

    def _build_evidence_catalog(
        self,
        inspection_api: Any,
        trace_id: str,
        spans: list[dict[str, Any]],
        gaps: list[str],
    ) -> dict[str, list[EvidenceRef]]:
        catalog: dict[str, list[EvidenceRef]] = {
            "required_error_span": [],
            "required_tool_io": [],
            "required_retrieval_chunks": [],
            "required_messages": [],
            "required_span_attributes": [],
        }
        for span in spans:
            status_code = str(span.get("status_code") or "")
            if status_code == "ERROR":
                span_id = str(span.get("span_id") or "root-span")
                status_text = str(span.get("status_message") or "")
                catalog["required_error_span"].append(
                    EvidenceRef(
                        trace_id=trace_id,
                        span_id=span_id,
                        kind="SPAN",
                        ref=span_id,
                        excerpt_hash=hash_excerpt(status_text or span_id),
                        ts=span.get("start_time"),
                    )
                )
        for span in spans:
            span_id = str(span.get("span_id") or "")
            if not span_id:
                continue
            if hasattr(inspection_api, "get_span"):
                try:
                    detail = inspection_api.get_span(span_id)
                    attributes = detail.get("attributes") if isinstance(detail, dict) else {}
                    attributes_text = self._to_text(attributes)
                    if attributes_text and attributes_text != "{}":
                        catalog["required_span_attributes"].append(
                            EvidenceRef(
                                trace_id=trace_id,
                                span_id=span_id,
                                kind="SPAN",
                                ref=f"{span_id}:attributes",
                                excerpt_hash=hash_excerpt(attributes_text),
                                ts=span.get("start_time"),
                            )
                        )
                except Exception as exc:
                    gaps.append(f"Failed to load span attributes for span {span_id}: {exc}")
            try:
                tool_io = inspection_api.get_tool_io(span_id)
                if tool_io:
                    catalog["required_tool_io"].append(
                        EvidenceRef(
                            trace_id=trace_id,
                            span_id=span_id,
                            kind="TOOL_IO",
                            ref=str(tool_io.get("artifact_id") or f"tool:{span_id}"),
                            excerpt_hash=hash_excerpt(
                                str(tool_io.get("status_code") or "")
                                + str(tool_io.get("tool_name") or "")
                            ),
                            ts=span.get("start_time"),
                        )
                    )
            except Exception as exc:
                gaps.append(f"Failed to load tool IO for span {span_id}: {exc}")
            try:
                retrieval_chunks = inspection_api.get_retrieval_chunks(span_id) or []
                for chunk in retrieval_chunks:
                    catalog["required_retrieval_chunks"].append(
                        EvidenceRef(
                            trace_id=trace_id,
                            span_id=span_id,
                            kind="RETRIEVAL_CHUNK",
                            ref=str(chunk.get("artifact_id") or f"retrieval:{span_id}:0:unknown"),
                            excerpt_hash=hash_excerpt(str(chunk.get("content") or "")),
                            ts=span.get("start_time"),
                        )
                    )
            except Exception as exc:
                gaps.append(f"Failed to load retrieval chunks for span {span_id}: {exc}")
            try:
                messages = inspection_api.get_messages(span_id) or []
                for index, message in enumerate(messages):
                    catalog["required_messages"].append(
                        EvidenceRef(
                            trace_id=trace_id,
                            span_id=span_id,
                            kind="MESSAGE",
                            ref=f"message:{span_id}:{index}",
                            excerpt_hash=hash_excerpt(str(message.get("content") or "")),
                            ts=span.get("start_time"),
                        )
                    )
            except Exception as exc:
                gaps.append(f"Failed to load messages for span {span_id}: {exc}")
        for key, value in catalog.items():
            catalog[key] = self._dedupe_evidence_refs(value)
        return catalog

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except Exception:
            return str(value)

    def _build_text_evidence_records(
        self,
        inspection_api: Any,
        trace_id: str,
        spans: list[dict[str, Any]],
        gaps: list[str],
    ) -> list[tuple[str, EvidenceRef]]:
        records: list[tuple[str, EvidenceRef]] = []
        for span in spans:
            span_id = str(span.get("span_id") or "")
            if not span_id:
                continue
            status_message = str(span.get("status_message") or "")
            if status_message:
                records.append(
                    (
                        status_message,
                        EvidenceRef(
                            trace_id=trace_id,
                            span_id=span_id,
                            kind="SPAN",
                            ref=span_id,
                            excerpt_hash=hash_excerpt(status_message),
                            ts=span.get("start_time"),
                        ),
                    )
                )
            if hasattr(inspection_api, "get_span"):
                try:
                    detail = inspection_api.get_span(span_id)
                    attributes = detail.get("attributes") if isinstance(detail, dict) else {}
                    attributes_text = self._to_text(attributes)
                    if attributes_text and attributes_text != "{}":
                        records.append(
                            (
                                attributes_text,
                                EvidenceRef(
                                    trace_id=trace_id,
                                    span_id=span_id,
                                    kind="SPAN",
                                    ref=f"{span_id}:attributes",
                                    excerpt_hash=hash_excerpt(attributes_text),
                                    ts=span.get("start_time"),
                                ),
                            )
                        )
                except Exception as exc:
                    gaps.append(f"Failed to load span attributes for span {span_id}: {exc}")
            try:
                messages = inspection_api.get_messages(span_id) or []
                for index, message in enumerate(messages):
                    content = str(message.get("content") or "")
                    if not content:
                        continue
                    records.append(
                        (
                            content,
                            EvidenceRef(
                                trace_id=trace_id,
                                span_id=span_id,
                                kind="MESSAGE",
                                ref=f"message:{span_id}:{index}",
                                excerpt_hash=hash_excerpt(content),
                                ts=span.get("start_time"),
                            ),
                        )
                    )
            except Exception as exc:
                gaps.append(f"Failed to load messages for span {span_id}: {exc}")
            try:
                tool_io = inspection_api.get_tool_io(span_id)
                if tool_io:
                    tool_text = " ".join(
                        [
                            self._to_text(tool_io.get("tool_name")),
                            self._to_text(tool_io.get("input")),
                            self._to_text(tool_io.get("output")),
                            self._to_text(tool_io.get("status_code")),
                        ]
                    ).strip()
                    if tool_text:
                        records.append(
                            (
                                tool_text,
                                EvidenceRef(
                                    trace_id=trace_id,
                                    span_id=span_id,
                                    kind="TOOL_IO",
                                    ref=str(tool_io.get("artifact_id") or f"tool:{span_id}"),
                                    excerpt_hash=hash_excerpt(tool_text),
                                    ts=span.get("start_time"),
                                ),
                            )
                        )
            except Exception as exc:
                gaps.append(f"Failed to load tool IO for span {span_id}: {exc}")
            try:
                retrieval_chunks = inspection_api.get_retrieval_chunks(span_id) or []
                for chunk in retrieval_chunks:
                    content = str(chunk.get("content") or "")
                    if not content:
                        continue
                    records.append(
                        (
                            content,
                            EvidenceRef(
                                trace_id=trace_id,
                                span_id=span_id,
                                kind="RETRIEVAL_CHUNK",
                                ref=str(chunk.get("artifact_id") or f"retrieval:{span_id}:0:unknown"),
                                excerpt_hash=hash_excerpt(content),
                                ts=span.get("start_time"),
                            ),
                        )
                    )
            except Exception as exc:
                gaps.append(f"Failed to load retrieval chunks for span {span_id}: {exc}")
        deduped: list[tuple[str, EvidenceRef]] = []
        seen: set[tuple[str, str]] = set()
        for text, evidence in records:
            key = (evidence.kind, evidence.ref)
            if key in seen:
                continue
            deduped.append((text, evidence))
            seen.add(key)
        return deduped

    def _evaluate_violation_rules(
        self,
        *,
        control: dict[str, Any],
        text_records: list[tuple[str, EvidenceRef]],
        error_span_evidence: list[EvidenceRef],
        selected_evidence: list[EvidenceRef],
        gaps: list[str],
    ) -> tuple[bool, list[EvidenceRef]]:
        violation_evidence: list[EvidenceRef] = []
        max_error_spans = control.get("max_error_spans")
        if isinstance(max_error_spans, int) and max_error_spans >= 0:
            if len(error_span_evidence) > max_error_spans:
                violation_evidence.extend(error_span_evidence)
        raw_patterns = control.get("violation_patterns") or []
        patterns = [str(item) for item in raw_patterns if isinstance(item, str) and item]
        for pattern in patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                gaps.append(
                    f"Invalid violation pattern for {control.get('control_id') or 'control.unknown'}: {exc}"
                )
                continue
            for text, evidence in text_records:
                if compiled.search(text):
                    violation_evidence.append(evidence)
        merged = self._dedupe_evidence_refs(selected_evidence + violation_evidence)
        return (len(violation_evidence) > 0, merged)

    def _merge_override_controls(
        self,
        inspection_api: Any,
        controls: list[dict[str, Any]],
        request: PolicyComplianceRequest,
        gaps: list[str],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for control in controls:
            control_id = str(control.get("control_id") or "")
            if not control_id:
                continue
            merged[control_id] = control
        for control_id in request.control_scope_override or []:
            if control_id in merged:
                continue
            try:
                override = inspection_api.get_control(control_id, request.controls_version)
                merged[control_id] = override
            except Exception as exc:
                gaps.append(
                    f"Failed to load override control {control_id} for version "
                    f"{request.controls_version}: {exc}"
                )
        return list(merged.values())

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
    def _control_snapshot(control: dict[str, Any]) -> dict[str, Any]:
        return {
            "control_id": str(control.get("control_id") or "control.unknown"),
            "severity": str(control.get("severity") or "medium").lower(),
            "required_evidence": [
                str(item)
                for item in (control.get("required_evidence") or [])
                if isinstance(item, str) and str(item)
            ],
            "violation_patterns": [
                str(item)
                for item in (control.get("violation_patterns") or [])
                if isinstance(item, str) and str(item)
            ],
            "remediation_template": str(control.get("remediation_template") or "").strip(),
        }

    @staticmethod
    def _tighten_repl_control_budget(loop_budget: RuntimeBudget) -> RuntimeBudget:
        max_tokens_total = loop_budget.max_tokens_total
        if max_tokens_total is not None:
            max_tokens_total = max(4000, min(int(max_tokens_total), 20000))
        max_cost_usd = loop_budget.max_cost_usd
        if max_cost_usd is not None:
            max_cost_usd = min(float(max_cost_usd), 0.12)
        return RuntimeBudget(
            max_iterations=1,
            max_depth=max(0, int(loop_budget.max_depth)),
            max_tool_calls=max(1, min(int(loop_budget.max_tool_calls), 8)),
            max_subcalls=max(1, min(int(loop_budget.max_subcalls), 2)),
            max_tokens_total=max_tokens_total,
            max_cost_usd=max_cost_usd,
            sampling_seed=loop_budget.sampling_seed,
            max_wall_time_sec=max(20, min(int(loop_budget.max_wall_time_sec), 45)),
        )

    @staticmethod
    def _delegation_policy(
        *,
        control_id: str,
        required_evidence: list[str],
    ) -> dict[str, Any]:
        return {
            "prefer_planner_driven_subcalls": True,
            "goal": "Use focused child investigations for missing evidence and violation checks per control.",
            "example_actions": [
                {
                    "type": "delegate_subcall",
                    "objective": f"collect missing evidence for control={control_id}",
                    "use_planner": True,
                    "context": {
                        "control_id": control_id,
                        "required_evidence": required_evidence,
                        "focus": "resolve missing evidence coverage before final verdict",
                    },
                },
                {
                    "type": "delegate_subcall",
                    "objective": f"validate violation evidence for control={control_id}",
                    "use_planner": True,
                    "context": {
                        "control_id": control_id,
                        "focus": "gather strongest pass/fail evidence and note unresolved gaps",
                    },
                },
            ],
        }

    def _resolve_model_client(self) -> RuntimeModelClient:
        if self._model_client is None:
            self._model_client = OpenAIModelClient()
        return self._model_client

    def _resolve_model_provider(self) -> str:
        model_provider = str(getattr(self._model_client, "model_provider", "") or "").strip()
        if model_provider:
            return model_provider
        return str(self.model_provider or "openai")

    @classmethod
    def _deterministic_outcome(
        cls,
        *,
        control: dict[str, Any],
        has_violation: bool,
        required_evidence: list[str],
    ) -> tuple[str, float, str]:
        remediation_template = str(control.get("remediation_template") or "").strip()
        if has_violation:
            pass_fail = "fail"
            confidence = 0.78
            remediation = remediation_template or (
                "Address matched control violations and rerun compliance evaluation."
            )
            return pass_fail, confidence, remediation
        pass_fail = "pass"
        confidence = 0.72 if required_evidence else 0.62
        remediation = remediation_template or "No remediation needed."
        return pass_fail, confidence, remediation

    @staticmethod
    def _verdict_priority(pass_fail: str) -> int:
        priorities = {
            "pass": 0,
            "not_applicable": 1,
            "insufficient_evidence": 2,
            "fail": 3,
        }
        return int(priorities.get(pass_fail, -1))

    @classmethod
    def _select_more_conservative_verdict(
        cls,
        *,
        deterministic_pass_fail: str,
        candidate_pass_fail: str,
    ) -> str:
        if candidate_pass_fail not in _ALLOWED_CONTROL_VERDICTS:
            return deterministic_pass_fail
        if cls._verdict_priority(candidate_pass_fail) >= cls._verdict_priority(
            deterministic_pass_fail
        ):
            return candidate_pass_fail
        return deterministic_pass_fail

    def _llm_control_judgment(
        self,
        *,
        request: PolicyComplianceRequest,
        control: dict[str, Any],
        selected_evidence: list[EvidenceRef],
        text_records: list[tuple[str, EvidenceRef]],
        deterministic_pass_fail: str,
    ) -> tuple[str, float, str, list[str], int, StructuredGenerationUsage, str]:
        model_client = self._resolve_model_client()
        model_provider = str(getattr(model_client, "model_provider", self.model_provider) or "openai")
        prompt_payload = {
            "trace_id": request.trace_id,
            "controls_version": request.controls_version,
            "control": self._control_snapshot(control),
            "deterministic_hint": {"pass_fail": deterministic_pass_fail},
            "selected_evidence": [self._evidence_dict(ref) for ref in selected_evidence[:20]],
            "trace_signals": [
                {
                    "text": str(text)[:400],
                    "evidence_ref": self._evidence_dict(evidence),
                }
                for text, evidence in text_records[: self._max_prompt_records]
            ],
        }
        request_payload = StructuredGenerationRequest(
            model_provider=model_provider,
            model_name=self.model_name,
            temperature=self.temperature,
            system_prompt=self._prompt_definition.prompt_text,
            user_prompt=json.dumps(prompt_payload, sort_keys=True),
            response_schema_name=self._prompt_definition.prompt_id,
            response_schema=self._prompt_definition.response_schema,
        )
        loop_result = run_structured_generation_loop(
            client=model_client,
            request=request_payload,
        )
        payload = loop_result.output
        pass_fail = str(payload.get("pass_fail") or "").strip()
        if pass_fail not in _ALLOWED_CONTROL_VERDICTS:
            raise ModelOutputInvalidError(
                f"Unsupported compliance pass_fail value for control {control.get('control_id')}: {pass_fail}"
            )
        confidence_raw = payload.get("confidence")
        if isinstance(confidence_raw, (int, float)):
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        else:
            confidence = 0.5
        remediation = str(payload.get("remediation") or "").strip()
        if not remediation:
            remediation = str(control.get("remediation_template") or "").strip()
        if not remediation:
            remediation = "Review control evidence and rerun compliance evaluation."
        gaps_payload = payload.get("gaps")
        llm_gaps: list[str] = []
        if isinstance(gaps_payload, list):
            llm_gaps = [str(item).strip() for item in gaps_payload if str(item).strip()]
        return (
            pass_fail,
            confidence,
            remediation,
            llm_gaps,
            int(loop_result.attempt_count),
            loop_result.usage,
            model_provider,
        )

    def _update_runtime_signals(
        self,
        *,
        usage_total: StructuredGenerationUsage,
        runtime_tracker: dict[str, Any],
    ) -> None:
        llm_used = bool(runtime_tracker.get("llm_used"))
        fallback_used = bool(runtime_tracker.get("fallback_used"))
        recursive_used = bool(runtime_tracker.get("recursive_used"))
        repl_used = bool(runtime_tracker.get("repl_used"))
        if fallback_used:
            mode = "deterministic_fallback"
        elif repl_used:
            mode = "repl_llm"
        elif recursive_used:
            mode = "recursive_llm"
        elif llm_used:
            mode = "llm"
        else:
            mode = "deterministic"
        model_provider = str(runtime_tracker.get("model_provider") or self.model_provider)
        self._runtime_signals["iterations"] = max(1, int(runtime_tracker.get("attempts") or 1))
        self._runtime_signals["depth_reached"] = int(runtime_tracker.get("depth_reached") or 0)
        if recursive_used:
            self._runtime_signals["tool_calls"] = int(runtime_tracker.get("tool_calls") or 0)
        else:
            self._runtime_signals["tool_calls"] = int(runtime_tracker.get("llm_calls") or 0)
        self._runtime_signals["llm_subcalls"] = int(runtime_tracker.get("llm_subcalls") or 0)
        self._runtime_signals["tokens_in"] = int(usage_total.tokens_in)
        self._runtime_signals["tokens_out"] = int(usage_total.tokens_out)
        self._runtime_signals["cost_usd"] = float(usage_total.cost_usd)
        self._runtime_signals["model_provider"] = model_provider
        self._runtime_signals["compliance_judgment_mode"] = mode
        self._runtime_signals["runtime_state"] = str(runtime_tracker.get("runtime_state") or "completed")
        self._runtime_signals["budget_reason"] = str(runtime_tracker.get("budget_reason") or "")
        raw_trajectory = runtime_tracker.get("state_trajectory") or []
        self._runtime_signals["state_trajectory"] = [
            str(item) for item in raw_trajectory if str(item).strip()
        ]
        raw_subcall_metadata = runtime_tracker.get("subcall_metadata") or []
        subcall_metadata = [item for item in raw_subcall_metadata if isinstance(item, dict)]
        self._runtime_signals["subcall_metadata"] = subcall_metadata
        self._runtime_signals["subcalls"] = len(subcall_metadata)
        raw_repl_trajectory = runtime_tracker.get("repl_trajectory") or []
        self._runtime_signals["repl_trajectory"] = [
            item for item in raw_repl_trajectory if isinstance(item, dict)
        ]
        sandbox_violations = runtime_tracker.get("sandbox_violations") or []
        if sandbox_violations:
            self._runtime_signals["sandbox_violations"] = [
                str(item) for item in sandbox_violations if str(item).strip()
            ]

    @staticmethod
    def _evidence_refs_from_payload(
        payload_refs: Any,
        *,
        trace_id: str,
        default_span_id: str,
    ) -> list[EvidenceRef]:
        allowed_kinds = {"SPAN", "TOOL_IO", "RETRIEVAL_CHUNK", "MESSAGE", "CONFIG_DIFF"}
        parsed: list[EvidenceRef] = []
        if not isinstance(payload_refs, list):
            return parsed
        for item in payload_refs:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "SPAN").upper()
            if kind not in allowed_kinds:
                continue
            ref = str(item.get("ref") or "")
            span_id = str(item.get("span_id") or default_span_id)
            if not ref or not span_id:
                continue
            parsed.append(
                EvidenceRef(
                    trace_id=str(item.get("trace_id") or trace_id),
                    span_id=span_id,
                    kind=kind,  # type: ignore[arg-type]
                    ref=ref,
                    excerpt_hash=str(item.get("excerpt_hash") or hash_excerpt(ref)),
                    ts=(str(item.get("ts")) if item.get("ts") is not None else None),
                )
            )
        return parsed

    def _evaluate_control_repl(
        self,
        inspection_api: Any,
        control: dict[str, Any],
        request: PolicyComplianceRequest,
        default_evidence: EvidenceRef,
        gaps: list[str],
        usage_total: StructuredGenerationUsage,
        runtime_tracker: dict[str, Any],
        loop_budget: RuntimeBudget,
        budget_pool: RuntimeBudgetPool,
    ) -> ComplianceFinding:
        control_id = str(control.get("control_id") or "control.unknown")
        severity = str(control.get("severity") or "medium").lower()
        if severity not in self._SEVERITY_ORDER:
            severity = "medium"

        required_evidence_raw = [
            str(item)
            for item in (control.get("required_evidence") or [])
            if isinstance(item, str) and str(item)
        ]
        required_evidence = list(dict.fromkeys(required_evidence_raw))
        if not required_evidence:
            try:
                required_evidence = list(
                    dict.fromkeys(
                        [
                            str(item)
                            for item in (
                                inspection_api.required_evidence(control_id, request.controls_version)
                                or []
                            )
                            if str(item)
                        ]
                    )
                )
            except Exception as exc:
                gaps.append(
                    f"Failed to load required evidence for {control_id} "
                    f"({request.controls_version}): {exc}"
                )

        try:
            spans = inspection_api.list_spans(request.trace_id) or []
        except Exception as exc:
            spans = []
            gaps.append(
                f"Failed to list spans for REPL compliance deterministic floor on {control_id}: {exc}"
            )
        evidence_catalog = self._build_evidence_catalog(
            inspection_api,
            request.trace_id,
            spans,
            gaps,
        )
        text_records = self._build_text_evidence_records(
            inspection_api,
            request.trace_id,
            spans,
            gaps,
        )
        deterministic_selected_evidence: list[EvidenceRef] = []
        deterministic_missing_evidence: list[str] = []
        for requirement in required_evidence:
            requirement_evidence = evidence_catalog.get(requirement) or []
            if not requirement_evidence:
                deterministic_missing_evidence.append(requirement)
                continue
            deterministic_selected_evidence.extend(requirement_evidence)
        deterministic_selected_evidence = self._dedupe_evidence_refs(deterministic_selected_evidence)
        if not deterministic_selected_evidence:
            deterministic_selected_evidence = [default_evidence]

        if deterministic_missing_evidence:
            deterministic_pass_fail = "insufficient_evidence"
            coverage_ratio = 0.0
            if required_evidence:
                coverage_ratio = (
                    len(required_evidence) - len(deterministic_missing_evidence)
                ) / len(required_evidence)
            deterministic_confidence = 0.25 + (0.25 * max(0.0, coverage_ratio))
            missing_text = ", ".join(deterministic_missing_evidence)
            deterministic_remediation = str(control.get("remediation_template") or "").strip() or (
                f"Collect missing evidence ({missing_text}) via Inspection API and rerun evaluator."
            )
        else:
            deterministic_violation, deterministic_selected_evidence = self._evaluate_violation_rules(
                control=control,
                text_records=text_records,
                error_span_evidence=evidence_catalog.get("required_error_span") or [],
                selected_evidence=deterministic_selected_evidence,
                gaps=gaps,
            )
            (
                deterministic_pass_fail,
                deterministic_confidence,
                deterministic_remediation,
            ) = self._deterministic_outcome(
                control=control,
                has_violation=deterministic_violation,
                required_evidence=required_evidence,
            )

        model_client = self._resolve_model_client()
        model_provider = str(getattr(model_client, "model_provider", self.model_provider) or "openai")
        self.prompt_template_hash = get_prompt_definition(_REPL_COMPLIANCE_PROMPT_ID).prompt_template_hash
        runtime_tracker["repl_used"] = True
        runtime_tracker["llm_used"] = True
        runtime_tracker["model_provider"] = model_provider

        repl_loop = ReplLoop(
            tool_registry=ToolRegistry(inspection_api=inspection_api),
            model_client=model_client,
            model_name=self.model_name,
            temperature=self.temperature,
        )
        tuned_loop_budget = self._tighten_repl_control_budget(loop_budget)
        loop_result = repl_loop.run(
            objective=(
                "policy_compliance repl investigation for "
                f"trace={request.trace_id} control={control_id}"
            ),
            input_vars={
                "trace_id": request.trace_id,
                "controls_version": request.controls_version,
                "control": self._control_snapshot(control),
                "required_evidence": required_evidence,
                "default_evidence": self._evidence_dict(default_evidence),
            },
            budget=tuned_loop_budget,
            require_subquery_for_non_trivial=True,
        )
        usage_total.add(
            StructuredGenerationUsage(
                tokens_in=int(loop_result.usage.tokens_in),
                tokens_out=int(loop_result.usage.tokens_out),
                cost_usd=float(loop_result.usage.cost_usd),
            )
        )
        runtime_tracker["attempts"] = int(runtime_tracker.get("attempts") or 0) + max(
            1, int(loop_result.usage.iterations)
        )
        runtime_tracker["tool_calls"] = int(runtime_tracker.get("tool_calls") or 0) + int(
            loop_result.usage.tool_calls
        )
        runtime_tracker["llm_subcalls"] = int(runtime_tracker.get("llm_subcalls") or 0) + int(
            loop_result.usage.llm_subcalls
        )
        runtime_tracker["depth_reached"] = max(
            int(runtime_tracker.get("depth_reached") or 0),
            int(loop_result.usage.depth_reached),
        )
        runtime_tracker.setdefault("state_trajectory", []).extend(loop_result.state_trajectory)
        runtime_tracker.setdefault("repl_trajectory", []).extend(loop_result.repl_trajectory)
        budget_pool.consume(
            iterations=int(loop_result.usage.iterations),
            depth=int(loop_result.usage.depth_reached),
            tool_calls=int(loop_result.usage.tool_calls),
            tokens_in=int(loop_result.usage.tokens_in),
            tokens_out=int(loop_result.usage.tokens_out),
            cost_usd=float(loop_result.usage.cost_usd),
            subcalls=int(loop_result.usage.llm_subcalls),
        )

        if loop_result.status == "terminated_budget":
            runtime_tracker["runtime_state"] = "terminated_budget"
            if not runtime_tracker.get("budget_reason"):
                runtime_tracker["budget_reason"] = str(loop_result.budget_reason or "")
        if loop_result.status == "failed":
            runtime_tracker["runtime_state"] = "failed"
            if loop_result.error_code == "SANDBOX_VIOLATION":
                runtime_tracker.setdefault("sandbox_violations", []).append(
                    str(loop_result.error_message or "")
                )
            if loop_result.error_code == "MODEL_OUTPUT_INVALID":
                if not self._fallback_on_llm_error:
                    raise ModelOutputInvalidError(
                        loop_result.error_message or "REPL compliance output was invalid."
                    )
                runtime_tracker["fallback_used"] = True
                gaps.append(
                    f"LLM compliance judgment failed and deterministic fallback was used for {control_id}: "
                    f"{loop_result.error_message or loop_result.error_code or 'unknown error'}"
                )
            else:
                raise RuntimeError(loop_result.error_message or "REPL compliance runtime failed.")

        payload = loop_result.output if isinstance(loop_result.output, dict) else {}
        selected_evidence = self._evidence_refs_from_payload(
            payload.get("evidence_refs"),
            trace_id=request.trace_id,
            default_span_id=default_evidence.span_id,
        )
        if not selected_evidence:
            selected_evidence = list(deterministic_selected_evidence)

        covered_requirements = [
            str(item)
            for item in (payload.get("covered_requirements") or [])
            if isinstance(item, str) and str(item)
        ]
        payload_missing_evidence = [
            str(item)
            for item in (payload.get("missing_evidence") or [])
            if isinstance(item, str) and str(item)
        ]
        if required_evidence:
            for requirement in required_evidence:
                if requirement not in covered_requirements and requirement not in payload_missing_evidence:
                    payload_missing_evidence.append(requirement)
        missing_evidence = list(
            dict.fromkeys(deterministic_missing_evidence + payload_missing_evidence)
        )

        pass_fail_candidate = str(payload.get("pass_fail") or "").strip()
        confidence_raw = payload.get("confidence")
        confidence = deterministic_confidence
        if isinstance(confidence_raw, (int, float)):
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        remediation = str(payload.get("remediation") or "").strip()
        if not remediation:
            remediation = deterministic_remediation

        if missing_evidence:
            pass_fail_candidate = "insufficient_evidence"
            confidence = min(confidence, 0.5)

        pass_fail = self._select_more_conservative_verdict(
            deterministic_pass_fail=deterministic_pass_fail,
            candidate_pass_fail=pass_fail_candidate,
        )
        if pass_fail == deterministic_pass_fail and pass_fail != pass_fail_candidate:
            confidence = deterministic_confidence

        if not remediation:
            remediation = "Review control evidence and rerun compliance evaluation."

        llm_gaps: list[str] = []
        gaps_payload = payload.get("gaps")
        if isinstance(gaps_payload, list):
            llm_gaps.extend([str(item).strip() for item in gaps_payload if str(item).strip()])
        if loop_result.status == "terminated_budget":
            llm_gaps.append(
                loop_result.budget_reason
                or "REPL runtime reached a budget limit before control finalize."
            )
        if llm_gaps:
            gaps.extend(llm_gaps)

        return ComplianceFinding(
            controls_version=request.controls_version,
            control_id=control_id,
            pass_fail=pass_fail,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            confidence=confidence,
            evidence_refs=self._dedupe_evidence_refs(selected_evidence),
            missing_evidence=missing_evidence,
            remediation=remediation,
        )

    def _evaluate_control_recursive(
        self,
        inspection_api: Any,
        control: dict[str, Any],
        request: PolicyComplianceRequest,
        default_evidence: EvidenceRef,
        gaps: list[str],
        usage_total: StructuredGenerationUsage,
        runtime_tracker: dict[str, Any],
        loop_budget: RuntimeBudget,
        budget_pool: RuntimeBudgetPool,
    ) -> ComplianceFinding:
        control_id = str(control.get("control_id") or "control.unknown")
        severity = str(control.get("severity") or "medium").lower()
        if severity not in self._SEVERITY_ORDER:
            severity = "medium"

        required_evidence_raw = [
            str(item)
            for item in (control.get("required_evidence") or [])
            if isinstance(item, str) and str(item)
        ]
        required_evidence = list(dict.fromkeys(required_evidence_raw))
        if not required_evidence:
            try:
                required_evidence = list(
                    dict.fromkeys(
                        [
                            str(item)
                            for item in (
                                inspection_api.required_evidence(control_id, request.controls_version)
                                or []
                            )
                            if str(item)
                        ]
                    )
                )
            except Exception as exc:
                gaps.append(
                    f"Failed to load required evidence for {control_id} "
                    f"({request.controls_version}): {exc}"
                )

        model_client = self._resolve_model_client()
        model_provider = str(getattr(model_client, "model_provider", self.model_provider) or "openai")
        planner = StructuredActionPlanner(
            client=model_client,
            model_name=self.model_name,
            temperature=self.temperature,
            prompt_id=_RECURSIVE_COMPLIANCE_PROMPT_ID,
        )
        self.prompt_template_hash = planner.prompt_template_hash
        runtime_tracker["recursive_used"] = True
        runtime_tracker["llm_used"] = True
        runtime_tracker["model_provider"] = model_provider

        tool_registry = ToolRegistry(inspection_api=inspection_api)
        sandbox_guard = SandboxGuard(allowed_tools=tool_registry.allowed_tools)
        recursive_loop = RecursiveLoop(tool_registry=tool_registry, sandbox_guard=sandbox_guard)

        planner_seed = {
            "engine_type": self.engine_type,
            "trace_id": request.trace_id,
            "controls_version": request.controls_version,
            "control": self._control_snapshot(control),
            "required_evidence": required_evidence,
            "delegation_policy": self._delegation_policy(
                control_id=control_id,
                required_evidence=required_evidence,
            ),
        }

        def _planner(context: dict[str, Any]) -> dict[str, Any]:
            merged_context = dict(context)
            merged_context.update(planner_seed)
            return planner(merged_context)

        loop_result: RecursiveLoopResult = recursive_loop.run(
            actions=[],
            planner=_planner,
            budget=loop_budget,
            budget_pool=budget_pool,
            objective=(
                "policy_compliance recursive investigation for "
                f"trace={request.trace_id} control={control_id}"
            ),
            parent_call_id=f"policy_compliance:{control_id}",
            input_ref_hash=hash_excerpt(f"{request.trace_id}:{control_id}"),
        )
        usage_total.add(
            StructuredGenerationUsage(
                tokens_in=int(loop_result.usage.tokens_in),
                tokens_out=int(loop_result.usage.tokens_out),
                cost_usd=float(loop_result.usage.cost_usd),
            )
        )
        runtime_tracker["attempts"] = int(runtime_tracker.get("attempts") or 0) + max(
            1, int(loop_result.usage.iterations)
        )
        runtime_tracker["tool_calls"] = int(runtime_tracker.get("tool_calls") or 0) + int(
            loop_result.usage.tool_calls
        )
        runtime_tracker["depth_reached"] = max(
            int(runtime_tracker.get("depth_reached") or 0),
            int(loop_result.usage.depth_reached),
        )
        runtime_tracker.setdefault("state_trajectory", []).extend(loop_result.state_trajectory)
        runtime_tracker.setdefault("subcall_metadata", []).extend(loop_result.subcall_metadata)

        if loop_result.status == "terminated_budget":
            runtime_tracker["runtime_state"] = "terminated_budget"
            if not runtime_tracker.get("budget_reason"):
                runtime_tracker["budget_reason"] = str(loop_result.budget_reason or "")
        if loop_result.status == "failed":
            runtime_tracker["runtime_state"] = "failed"
            if loop_result.error_code == "SANDBOX_VIOLATION":
                runtime_tracker.setdefault("sandbox_violations", []).append(
                    str(loop_result.error_message or "")
                )
            if loop_result.error_code == "MODEL_OUTPUT_INVALID":
                if not self._fallback_on_llm_error:
                    raise ModelOutputInvalidError(
                        loop_result.error_message or "Recursive compliance planner output was invalid."
                    )
                runtime_tracker["fallback_used"] = True
                gaps.append(
                    f"LLM compliance judgment failed and deterministic fallback was used for {control_id}: "
                    f"{loop_result.error_message or loop_result.error_code or 'unknown error'}"
                )
            else:
                raise RuntimeError(loop_result.error_message or "Recursive compliance planner run failed.")

        payload = loop_result.output if isinstance(loop_result.output, dict) else {}
        selected_evidence = self._evidence_refs_from_payload(
            payload.get("evidence_refs"),
            trace_id=request.trace_id,
            default_span_id=default_evidence.span_id,
        )
        if not selected_evidence:
            selected_evidence = [default_evidence]

        covered_requirements = [
            str(item)
            for item in (payload.get("covered_requirements") or [])
            if isinstance(item, str) and str(item)
        ]
        missing_evidence = [
            str(item)
            for item in (payload.get("missing_evidence") or [])
            if isinstance(item, str) and str(item)
        ]
        if required_evidence:
            for requirement in required_evidence:
                if requirement not in covered_requirements and requirement not in missing_evidence:
                    missing_evidence.append(requirement)
        missing_evidence = list(dict.fromkeys(missing_evidence))

        deterministic_pass_fail, deterministic_confidence, deterministic_remediation = (
            self._deterministic_outcome(
                control=control,
                has_violation=False,
                required_evidence=required_evidence,
            )
        )
        pass_fail = str(payload.get("pass_fail") or "").strip()
        confidence_raw = payload.get("confidence")
        confidence = deterministic_confidence
        if isinstance(confidence_raw, (int, float)):
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        remediation = str(payload.get("remediation") or "").strip()
        if not remediation:
            remediation = deterministic_remediation

        if missing_evidence:
            pass_fail = "insufficient_evidence"
            confidence = min(confidence, 0.5)
        elif pass_fail not in _ALLOWED_CONTROL_VERDICTS:
            pass_fail = deterministic_pass_fail

        if not remediation:
            remediation = "Review control evidence and rerun compliance evaluation."

        llm_gaps: list[str] = []
        gaps_payload = payload.get("gaps")
        if isinstance(gaps_payload, list):
            llm_gaps.extend([str(item).strip() for item in gaps_payload if str(item).strip()])
        if loop_result.status == "terminated_budget":
            llm_gaps.append(
                loop_result.budget_reason
                or "Recursive runtime reached a budget limit before control finalize."
            )
        if llm_gaps:
            gaps.extend(llm_gaps)

        return ComplianceFinding(
            controls_version=request.controls_version,
            control_id=control_id,
            pass_fail=pass_fail,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            confidence=confidence,
            evidence_refs=self._dedupe_evidence_refs(selected_evidence),
            missing_evidence=missing_evidence,
            remediation=remediation,
        )

    def _evaluate_control(
        self,
        inspection_api: Any,
        control: dict[str, Any],
        request: PolicyComplianceRequest,
        evidence_catalog: dict[str, list[EvidenceRef]],
        text_records: list[tuple[str, EvidenceRef]],
        default_evidence: EvidenceRef,
        gaps: list[str],
        usage_total: StructuredGenerationUsage,
        runtime_tracker: dict[str, Any],
    ) -> ComplianceFinding:
        control_id = str(control.get("control_id") or "control.unknown")
        severity = str(control.get("severity") or "medium").lower()
        if severity not in self._SEVERITY_ORDER:
            severity = "medium"
        required_evidence_raw = [
            str(item)
            for item in (control.get("required_evidence") or [])
            if isinstance(item, str) and item
        ]
        required_evidence = list(dict.fromkeys(required_evidence_raw))
        if not required_evidence:
            try:
                required_evidence = list(
                    dict.fromkeys(
                        [
                            str(item)
                            for item in (
                                inspection_api.required_evidence(control_id, request.controls_version)
                                or []
                            )
                            if str(item)
                        ]
                    )
                )
            except Exception as exc:
                gaps.append(
                    f"Failed to load required evidence for {control_id} "
                    f"({request.controls_version}): {exc}"
                )
        selected_evidence: list[EvidenceRef] = []
        missing_evidence: list[str] = []
        for requirement in required_evidence:
            requirement_evidence = evidence_catalog.get(requirement) or []
            if not requirement_evidence:
                missing_evidence.append(requirement)
                continue
            selected_evidence.extend(requirement_evidence)
        selected_evidence = self._dedupe_evidence_refs(selected_evidence)
        if not selected_evidence:
            selected_evidence = [default_evidence]

        if missing_evidence:
            pass_fail = "insufficient_evidence"
            coverage_ratio = 0.0
            if required_evidence:
                coverage_ratio = (len(required_evidence) - len(missing_evidence)) / len(required_evidence)
            confidence = 0.25 + (0.25 * max(0.0, coverage_ratio))
            remediation_template = str(control.get("remediation_template") or "").strip()
            missing_text = ", ".join(missing_evidence)
            remediation = remediation_template or (
                f"Collect missing evidence ({missing_text}) via Inspection API and rerun evaluator."
            )
            return ComplianceFinding(
                controls_version=request.controls_version,
                control_id=control_id,
                pass_fail=pass_fail,
                severity=severity,
                confidence=confidence,
                evidence_refs=selected_evidence,
                missing_evidence=missing_evidence,
                remediation=remediation,
            )

        deterministic_violation, selected_evidence = self._evaluate_violation_rules(
            control=control,
            text_records=text_records,
            error_span_evidence=evidence_catalog.get("required_error_span") or [],
            selected_evidence=selected_evidence,
            gaps=gaps,
        )
        deterministic_pass_fail, deterministic_confidence, deterministic_remediation = (
            self._deterministic_outcome(
                control=control,
                has_violation=deterministic_violation,
                required_evidence=required_evidence,
            )
        )

        pass_fail = deterministic_pass_fail
        confidence = deterministic_confidence
        remediation = deterministic_remediation

        if self._use_llm_judgment:
            runtime_tracker["llm_calls"] = int(runtime_tracker.get("llm_calls") or 0) + 1
            try:
                (
                    pass_fail,
                    confidence,
                    remediation,
                    llm_gaps,
                    attempt_count,
                    usage,
                    model_provider,
                ) = self._llm_control_judgment(
                    request=request,
                    control=control,
                    selected_evidence=selected_evidence,
                    text_records=text_records,
                    deterministic_pass_fail=deterministic_pass_fail,
                )
                runtime_tracker["attempts"] = int(runtime_tracker.get("attempts") or 0) + max(
                    1, int(attempt_count)
                )
                usage_total.add(usage)
                runtime_tracker["llm_used"] = True
                runtime_tracker["model_provider"] = model_provider
                if llm_gaps:
                    gaps.extend(llm_gaps)
            except ModelOutputInvalidError as exc:
                runtime_tracker["attempts"] = int(runtime_tracker.get("attempts") or 0) + int(
                    getattr(exc, "attempt_count", 1) or 1
                )
                usage = getattr(exc, "usage", None)
                if isinstance(usage, StructuredGenerationUsage):
                    usage_total.add(usage)
                runtime_tracker["model_provider"] = self._resolve_model_provider()
                if not self._fallback_on_llm_error:
                    raise
                runtime_tracker["fallback_used"] = True
                gaps.append(
                    f"LLM compliance judgment failed and deterministic fallback was used for {control_id}: {exc}"
                )
                pass_fail = deterministic_pass_fail
                confidence = deterministic_confidence
                remediation = deterministic_remediation

        return ComplianceFinding(
            controls_version=request.controls_version,
            control_id=control_id,
            pass_fail=pass_fail,
            severity=severity,
            confidence=confidence,
            evidence_refs=selected_evidence,
            missing_evidence=missing_evidence,
            remediation=remediation,
        )

    def run(self, request: PolicyComplianceRequest) -> ComplianceReport:
        self._reset_runtime_signals()
        usage_total = StructuredGenerationUsage()
        runtime_tracker: dict[str, Any] = {
            "attempts": 0,
            "llm_calls": 0,
            "llm_used": False,
            "repl_used": False,
            "fallback_used": False,
            "recursive_used": False,
            "model_provider": self.model_provider,
            "tool_calls": 0,
            "llm_subcalls": 0,
            "depth_reached": 0,
            "runtime_state": "completed",
            "budget_reason": "",
            "state_trajectory": [],
            "subcall_metadata": [],
            "repl_trajectory": [],
            "sandbox_violations": [],
        }

        inspection_api = self._resolve_inspection_api(request)
        gaps: list[str] = []
        findings: list[ComplianceFinding] = []

        try:
            try:
                spans = inspection_api.list_spans(request.trace_id) or []
            except Exception as exc:
                spans = []
                gaps.append(f"Failed to list spans for compliance evaluation: {exc}")

            app_type = self._infer_app_type(spans)
            tools_used = self._infer_tools_used(spans)
            try:
                scoped_controls = inspection_api.list_controls(
                    controls_version=request.controls_version,
                    app_type=app_type,
                    tools_used=tools_used or None,
                    data_domains=None,
                ) or []
            except Exception as exc:
                scoped_controls = []
                gaps.append(
                    f"Failed to load controls for version {request.controls_version}: {exc}"
                )

            merged_controls = self._merge_override_controls(
                inspection_api,
                scoped_controls,
                request,
                gaps,
            )
            ordered_controls = self._sort_controls(merged_controls)[: self._max_controls]
            default_evidence = self._root_evidence_ref(request.trace_id, spans)
            repl_mode = self._use_llm_judgment and self._use_repl_runtime
            recursive_mode = self._use_llm_judgment and self._use_recursive_runtime and not repl_mode
            evidence_catalog: dict[str, list[EvidenceRef]] = {}
            text_records: list[tuple[str, EvidenceRef]] = []
            budget_pool: RuntimeBudgetPool | None = None
            if recursive_mode or repl_mode:
                budget_pool = RuntimeBudgetPool(budget=self._recursive_budget)
            if not recursive_mode and not repl_mode:
                evidence_catalog = self._build_evidence_catalog(
                    inspection_api,
                    request.trace_id,
                    spans,
                    gaps,
                )
                text_records = self._build_text_evidence_records(
                    inspection_api,
                    request.trace_id,
                    spans,
                    gaps,
                )

            for index, control in enumerate(ordered_controls):
                if repl_mode:
                    if budget_pool is None:
                        raise RuntimeError("REPL compliance budget pool was not initialized.")
                    remaining_controls = max(1, len(ordered_controls) - index)
                    finding = self._evaluate_control_repl(
                        inspection_api=inspection_api,
                        control=control,
                        request=request,
                        default_evidence=default_evidence,
                        gaps=gaps,
                        usage_total=usage_total,
                        runtime_tracker=runtime_tracker,
                        loop_budget=budget_pool.allocate_run_budget(
                            sibling_count=remaining_controls,
                        ),
                        budget_pool=budget_pool,
                    )
                elif recursive_mode:
                    if budget_pool is None:
                        raise RuntimeError("Recursive compliance budget pool was not initialized.")
                    remaining_controls = max(1, len(ordered_controls) - index)
                    finding = self._evaluate_control_recursive(
                        inspection_api=inspection_api,
                        control=control,
                        request=request,
                        default_evidence=default_evidence,
                        gaps=gaps,
                        usage_total=usage_total,
                        runtime_tracker=runtime_tracker,
                        loop_budget=budget_pool.allocate_run_budget(
                            sibling_count=remaining_controls,
                        ),
                        budget_pool=budget_pool,
                    )
                else:
                    finding = self._evaluate_control(
                        inspection_api=inspection_api,
                        control=control,
                        request=request,
                        evidence_catalog=evidence_catalog,
                        text_records=text_records,
                        default_evidence=default_evidence,
                        gaps=gaps,
                        usage_total=usage_total,
                        runtime_tracker=runtime_tracker,
                    )
                findings.append(finding)

            if not findings:
                findings.append(
                    ComplianceFinding(
                        controls_version=request.controls_version,
                        control_id="control.no_applicable_controls",
                        pass_fail="not_applicable",
                        severity="low",
                        confidence=0.4,
                        evidence_refs=[default_evidence],
                        missing_evidence=[],
                        remediation="No scoped controls matched this trace and controls version.",
                    )
                )
                gaps.append(
                    f"No controls were scoped for controls_version={request.controls_version}."
                )

            if any(finding.pass_fail == "fail" for finding in findings):
                overall_verdict = "non_compliant"
            elif any(finding.pass_fail == "insufficient_evidence" for finding in findings):
                overall_verdict = "needs_review"
            elif all(finding.pass_fail == "not_applicable" for finding in findings):
                overall_verdict = "needs_review"
            else:
                overall_verdict = "compliant"
            overall_confidence = sum(finding.confidence for finding in findings) / len(findings)

            return ComplianceReport(
                trace_id=request.trace_id,
                controls_version=request.controls_version,
                controls_evaluated=findings,
                overall_verdict=overall_verdict,
                overall_confidence=overall_confidence,
                gaps=gaps,
            )
        finally:
            self._update_runtime_signals(
                usage_total=usage_total,
                runtime_tracker=runtime_tracker,
            )
