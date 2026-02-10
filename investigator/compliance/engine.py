# ABOUTME: Implements deterministic Policy-to-Trace Compliance control scoping and finding synthesis.
# ABOUTME: Produces controls-version traceable findings with evidence refs over the Inspection API.

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from investigator.inspection_api import PhoenixInspectionAPI
from investigator.runtime.contracts import (
    ComplianceFinding,
    ComplianceReport,
    EvidenceRef,
    InputRef,
    TimeWindow,
    hash_excerpt,
)


@dataclass
class PolicyComplianceRequest:
    trace_id: str
    project_name: str
    controls_version: str
    control_scope_override: list[str] | None = None


class PolicyComplianceEngine:
    engine_type = "policy_compliance"
    output_contract_name = "ComplianceReport"
    engine_version = "0.3.0"
    model_name = "gpt-5-mini"
    prompt_template_hash = "policy-compliance-verdict-v1"
    temperature = 0.0

    _SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    def __init__(
        self,
        inspection_api: Any | None = None,
        *,
        max_controls: int = 50,
    ) -> None:
        self._inspection_api = inspection_api
        self._max_controls = max_controls

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
        tools = {
            str(span.get("name") or "")
            for span in spans
            if str(span.get("span_kind") or "") == "TOOL" and str(span.get("name") or "")
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

    def _evaluate_control(
        self,
        inspection_api: Any,
        control: dict[str, Any],
        request: PolicyComplianceRequest,
        evidence_catalog: dict[str, list[EvidenceRef]],
        text_records: list[tuple[str, EvidenceRef]],
        default_evidence: EvidenceRef,
        gaps: list[str],
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
        else:
            has_violation, selected_evidence = self._evaluate_violation_rules(
                control=control,
                text_records=text_records,
                error_span_evidence=evidence_catalog.get("required_error_span") or [],
                selected_evidence=selected_evidence,
                gaps=gaps,
            )
            if has_violation:
                pass_fail = "fail"
                confidence = 0.78
                remediation = str(control.get("remediation_template") or "").strip() or (
                    "Address matched control violations and rerun compliance evaluation."
                )
            else:
                pass_fail = "pass"
                confidence = 0.72 if required_evidence else 0.62
                remediation = (
                    str(control.get("remediation_template") or "").strip()
                    or "No remediation needed."
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

    def run(self, request: PolicyComplianceRequest) -> ComplianceReport:
        inspection_api = self._resolve_inspection_api(request)
        gaps: list[str] = []
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

        findings: list[ComplianceFinding] = []
        for control in ordered_controls:
            findings.append(
                self._evaluate_control(
                    inspection_api=inspection_api,
                    control=control,
                    request=request,
                    evidence_catalog=evidence_catalog,
                    text_records=text_records,
                    default_evidence=default_evidence,
                    gaps=gaps,
                )
            )

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
