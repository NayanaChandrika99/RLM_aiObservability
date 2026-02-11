# ABOUTME: Validates engine outputs against runtime schema and evidence pointer contracts.
# ABOUTME: Produces deterministic validation errors used by runtime failure taxonomy handling.

from __future__ import annotations

from typing import Any


ALLOWED_EVIDENCE_KINDS = {"SPAN", "TOOL_IO", "RETRIEVAL_CHUNK", "MESSAGE", "CONFIG_DIFF"}
ALLOWED_COMPLIANCE_VERDICTS = {"pass", "fail", "not_applicable", "insufficient_evidence"}


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_evidence_ref(ref: Any, *, path: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(ref, dict):
        return [f"{path} must be an object."]
    if not _is_non_empty_string(ref.get("trace_id")):
        errors.append(f"{path}.trace_id must be a non-empty string.")
    if not _is_non_empty_string(ref.get("span_id")):
        errors.append(f"{path}.span_id must be a non-empty string.")
    if not _is_non_empty_string(ref.get("ref")):
        errors.append(f"{path}.ref must be a non-empty string.")
    if not _is_non_empty_string(ref.get("excerpt_hash")):
        errors.append(f"{path}.excerpt_hash must be a non-empty string.")
    kind = ref.get("kind")
    if kind not in ALLOWED_EVIDENCE_KINDS:
        errors.append(
            f"{path}.kind must be one of {sorted(ALLOWED_EVIDENCE_KINDS)}."
        )
    return errors


def validate_output_schema(*, contract_name: str, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["output payload must be an object."]

    if contract_name == "RCAReport":
        for key in ("trace_id", "primary_label", "summary"):
            if not _is_non_empty_string(payload.get(key)):
                errors.append(f"{key} must be a non-empty string.")
        if not isinstance(payload.get("confidence"), (int, float)):
            errors.append("confidence must be numeric.")
        if not isinstance(payload.get("evidence_refs"), list):
            errors.append("evidence_refs must be a list.")

    elif contract_name == "ComplianceReport":
        for key in ("trace_id", "controls_version", "overall_verdict"):
            if not _is_non_empty_string(payload.get(key)):
                errors.append(f"{key} must be a non-empty string.")
        if not isinstance(payload.get("overall_confidence"), (int, float)):
            errors.append("overall_confidence must be numeric.")
        controls_evaluated = payload.get("controls_evaluated")
        if not isinstance(controls_evaluated, list):
            errors.append("controls_evaluated must be a list.")
        else:
            for index, finding in enumerate(controls_evaluated):
                path = f"controls_evaluated[{index}]"
                if not isinstance(finding, dict):
                    errors.append(f"{path} must be an object.")
                    continue
                if not _is_non_empty_string(finding.get("control_id")):
                    errors.append(f"{path}.control_id must be a non-empty string.")
                verdict = finding.get("pass_fail")
                if verdict not in ALLOWED_COMPLIANCE_VERDICTS:
                    errors.append(
                        f"{path}.pass_fail must be one of {sorted(ALLOWED_COMPLIANCE_VERDICTS)}."
                    )
                if not isinstance(finding.get("evidence_refs"), list):
                    errors.append(f"{path}.evidence_refs must be a list.")
                if verdict == "insufficient_evidence":
                    missing = finding.get("missing_evidence")
                    if not isinstance(missing, list) or not missing:
                        errors.append(
                            f"{path}.missing_evidence must be a non-empty list for insufficient_evidence."
                        )

    elif contract_name == "IncidentDossier":
        for key in ("incident_summary",):
            if not _is_non_empty_string(payload.get(key)):
                errors.append(f"{key} must be a non-empty string.")
        for key in ("impacted_components", "timeline", "representative_traces", "hypotheses", "recommended_actions"):
            if not isinstance(payload.get(key), list):
                errors.append(f"{key} must be a list.")
        if not isinstance(payload.get("suspected_change"), dict):
            errors.append("suspected_change must be an object.")
        if not isinstance(payload.get("confidence"), (int, float)):
            errors.append("confidence must be numeric.")
    else:
        errors.append(f"Unknown output contract: {contract_name}")
    return errors


def validate_output_evidence(*, contract_name: str, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract_name == "RCAReport":
        evidence_refs = payload.get("evidence_refs") if isinstance(payload, dict) else None
        if not isinstance(evidence_refs, list) or not evidence_refs:
            errors.append("RCAReport must contain at least one evidence_ref.")
            return errors
        for index, ref in enumerate(evidence_refs):
            errors.extend(_validate_evidence_ref(ref, path=f"evidence_refs[{index}]"))
        return errors

    if contract_name == "ComplianceReport":
        controls_evaluated = payload.get("controls_evaluated") if isinstance(payload, dict) else None
        if not isinstance(controls_evaluated, list):
            return errors
        for index, finding in enumerate(controls_evaluated):
            if not isinstance(finding, dict):
                continue
            evidence_refs = finding.get("evidence_refs")
            verdict = finding.get("pass_fail")
            if verdict == "fail" and (not isinstance(evidence_refs, list) or not evidence_refs):
                errors.append(
                    f"controls_evaluated[{index}] fail verdict requires at least one evidence_ref."
                )
                continue
            if isinstance(evidence_refs, list):
                for ref_index, ref in enumerate(evidence_refs):
                    errors.extend(
                        _validate_evidence_ref(
                            ref,
                            path=f"controls_evaluated[{index}].evidence_refs[{ref_index}]",
                        )
                    )
        return errors

    if contract_name == "IncidentDossier":
        timeline = payload.get("timeline") if isinstance(payload, dict) else None
        if isinstance(timeline, list):
            for index, event in enumerate(timeline):
                if not isinstance(event, dict):
                    continue
                refs = event.get("evidence_refs")
                if not isinstance(refs, list) or not refs:
                    errors.append(f"timeline[{index}] must include at least one evidence_ref.")
                    continue
                for ref_index, ref in enumerate(refs):
                    errors.extend(
                        _validate_evidence_ref(
                            ref,
                            path=f"timeline[{index}].evidence_refs[{ref_index}]",
                        )
                    )
        hypotheses = payload.get("hypotheses") if isinstance(payload, dict) else None
        if isinstance(hypotheses, list):
            for index, hypothesis in enumerate(hypotheses):
                if not isinstance(hypothesis, dict):
                    continue
                refs = hypothesis.get("evidence_refs")
                if not isinstance(refs, list) or not refs:
                    errors.append(f"hypotheses[{index}] must include at least one evidence_ref.")
                    continue
                for ref_index, ref in enumerate(refs):
                    errors.extend(
                        _validate_evidence_ref(
                            ref,
                            path=f"hypotheses[{index}].evidence_refs[{ref_index}]",
                        )
                    )
        suspected_change = payload.get("suspected_change") if isinstance(payload, dict) else None
        if isinstance(suspected_change, dict):
            diff_ref = suspected_change.get("diff_ref")
            change_type = suspected_change.get("change_type")
            if change_type != "unknown" and not _is_non_empty_string(diff_ref):
                errors.append("suspected_change with asserted change_type requires non-empty diff_ref.")
            refs = suspected_change.get("evidence_refs")
            if isinstance(refs, list):
                for ref_index, ref in enumerate(refs):
                    errors.extend(
                        _validate_evidence_ref(
                            ref,
                            path=f"suspected_change.evidence_refs[{ref_index}]",
                        )
                    )
            if _is_non_empty_string(diff_ref):
                has_diff_evidence = False
                if isinstance(refs, list):
                    has_diff_evidence = any(
                        isinstance(ref, dict)
                        and ref.get("kind") == "CONFIG_DIFF"
                        and ref.get("ref") == diff_ref
                        for ref in refs
                    )
                if not has_diff_evidence:
                    errors.append("suspected_change.diff_ref must be represented by CONFIG_DIFF evidence_ref.")
        return errors

    return errors
