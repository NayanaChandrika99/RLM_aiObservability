# ABOUTME: Enforces semantic faithfulness for TRAIL predictions against source trace content.
# ABOUTME: Applies strict drop-or-repair rules for invalid locations, ungrounded evidence, and malformed fields.

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

VALID_IMPACTS = {"LOW", "MEDIUM", "HIGH"}


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _walk_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for span in spans:
        flat.append(span)
        children = span.get("child_spans") or []
        if isinstance(children, list):
            flat.extend(_walk_spans([child for child in children if isinstance(child, dict)]))
    return flat


def _span_text(span: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("span_name", "status_code", "status_message"):
        value = span.get(key)
        if isinstance(value, str):
            parts.append(value)

    span_attributes = span.get("span_attributes")
    if span_attributes is not None:
        parts.append(json.dumps(span_attributes, ensure_ascii=False))

    logs = span.get("logs")
    if isinstance(logs, list):
        for log in logs:
            if not isinstance(log, dict):
                continue
            parts.append(json.dumps(log.get("body", ""), ensure_ascii=False))
            parts.append(json.dumps(log.get("log_attributes", ""), ensure_ascii=False))
    return " ".join(parts)


def _build_span_index(trace_payload: dict[str, Any]) -> tuple[dict[str, str], str]:
    spans = trace_payload.get("spans", [])
    flat_spans = _walk_spans(spans if isinstance(spans, list) else [])

    index: dict[str, str] = {}
    default_location = ""
    for span in flat_spans:
        span_id = span.get("span_id")
        if not isinstance(span_id, str):
            continue
        if not default_location:
            default_location = span_id
        index[span_id] = _span_text(span)
    return index, default_location


def _is_grounded(evidence: str, span_index: dict[str, str]) -> bool:
    normalized_evidence = _normalize_text(evidence)
    if not normalized_evidence:
        return False
    for text in span_index.values():
        if normalized_evidence in _normalize_text(text):
            return True
    return False


def _evidence_from_text(text: str) -> str:
    return " ".join(text.split())[:280]


def _repair_evidence(location: str, span_index: dict[str, str]) -> str:
    local_text = span_index.get(location, "")
    if local_text.strip():
        return _evidence_from_text(local_text)
    for text in span_index.values():
        if text.strip():
            return _evidence_from_text(text)
    return ""


def enforce_semantic_faithfulness(
    trace_payload: dict[str, Any],
    prediction: dict[str, Any],
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = prediction.get("errors", [])
    total_errors = len(errors) if isinstance(errors, list) else 0

    report = {
        "trace_id": str(trace_payload.get("trace_id", "")),
        "mode": mode,
        "total_errors": total_errors,
        "kept_errors": total_errors,
        "dropped_errors": 0,
        "repaired_errors": 0,
        "grounded_evidence_rate": 1.0 if total_errors == 0 else 0.0,
        "drop_reasons": {
            "invalid_error_shape": 0,
            "missing_category": 0,
            "unrepairable_location": 0,
            "unrepairable_evidence": 0,
        },
        "repair_actions": {
            "location_repaired": 0,
            "evidence_repaired": 0,
            "impact_repaired": 0,
            "description_repaired": 0,
        },
    }

    if mode != "strict":
        return prediction, report

    span_index, default_location = _build_span_index(trace_payload)
    sanitized = deepcopy(prediction)
    raw_errors = prediction.get("errors", [])
    if not isinstance(raw_errors, list):
        raw_errors = []

    kept_errors: list[dict[str, Any]] = []

    for item in raw_errors:
        if not isinstance(item, dict):
            report["drop_reasons"]["invalid_error_shape"] += 1
            continue

        category = item.get("category")
        if not isinstance(category, str) or not category.strip():
            report["drop_reasons"]["missing_category"] += 1
            continue

        location = item.get("location")
        if not isinstance(location, str) or location not in span_index:
            if default_location:
                location = default_location
                report["repair_actions"]["location_repaired"] += 1
            else:
                report["drop_reasons"]["unrepairable_location"] += 1
                continue

        evidence = item.get("evidence")
        evidence_text = evidence if isinstance(evidence, str) else ""
        if not _is_grounded(evidence_text, span_index):
            repaired_evidence = _repair_evidence(location, span_index)
            if not repaired_evidence:
                report["drop_reasons"]["unrepairable_evidence"] += 1
                continue
            evidence_text = repaired_evidence
            report["repair_actions"]["evidence_repaired"] += 1

        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            description = f"Trace evidence indicates {category.lower()}."
            report["repair_actions"]["description_repaired"] += 1

        impact = item.get("impact")
        impact_value = impact.upper() if isinstance(impact, str) else "MEDIUM"
        if impact_value not in VALID_IMPACTS:
            impact_value = "MEDIUM"
            report["repair_actions"]["impact_repaired"] += 1

        kept_errors.append(
            {
                "category": category,
                "location": location,
                "evidence": evidence_text,
                "description": description,
                "impact": impact_value,
            }
        )

    sanitized["errors"] = kept_errors

    report["kept_errors"] = len(kept_errors)
    report["dropped_errors"] = total_errors - len(kept_errors)
    report["repaired_errors"] = sum(report["repair_actions"].values())
    report["grounded_evidence_rate"] = (
        1.0 if total_errors == 0 else len(kept_errors) / total_errors
    )
    return sanitized, report
