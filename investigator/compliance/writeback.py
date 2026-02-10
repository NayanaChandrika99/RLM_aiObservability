# ABOUTME: Writes compliance outputs to Phoenix annotations with control-linked evidence rows.
# ABOUTME: Preserves run_id and controls_version metadata in explanation payloads for auditability.

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

import pandas as pd
from phoenix.client import Client as PhoenixClient
from phoenix.trace import SpanEvaluations
from phoenix.session.client import Client as SessionClient

from investigator.runtime.contracts import ComplianceFinding, ComplianceReport, EvidenceRef


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _root_span_id(report: ComplianceReport) -> str:
    refs: list[EvidenceRef] = []
    for finding in report.controls_evaluated:
        refs.extend(finding.evidence_refs)
    if not refs:
        return "root-span"
    selected = sorted(refs, key=lambda ref: (ref.ts or "", ref.span_id, ref.kind, ref.ref))[0]
    return selected.span_id or "root-span"


def _control_label(finding: ComplianceFinding) -> str:
    if finding.pass_fail == "fail":
        return "control_violation"
    if finding.pass_fail == "insufficient_evidence":
        return "control_gap"
    return "control_evidence"


def _sort_findings(findings: list[ComplianceFinding]) -> list[ComplianceFinding]:
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    return sorted(
        findings,
        key=lambda finding: (
            -severity_order.get(finding.severity, 0),
            finding.control_id,
        ),
    )


def _build_span_annotations(report: ComplianceReport, run_id: str) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    root_annotation = {
        "span_id": _root_span_id(report),
        "name": "compliance.overall",
        "annotator_kind": "LLM",
        "result": {
            "label": report.overall_verdict,
            "score": report.overall_confidence,
            "explanation": _dumps(
                {
                    "annotator_kind": "LLM",
                    "controls_version": report.controls_version,
                    "report": report.to_dict(),
                    "run_id": run_id,
                }
            ),
        },
    }
    annotations.append(root_annotation)

    for finding in _sort_findings(report.controls_evaluated):
        control_name = f"compliance.control.{finding.control_id}.evidence"
        evidence_refs = sorted(
            finding.evidence_refs,
            key=lambda evidence: (evidence.span_id, evidence.kind, evidence.ref),
        )
        if not evidence_refs:
            evidence_refs = [
                EvidenceRef(
                    trace_id=report.trace_id,
                    span_id=_root_span_id(report),
                    kind="SPAN",
                    ref=_root_span_id(report),
                    excerpt_hash="",
                    ts=None,
                )
            ]
        for evidence in evidence_refs:
            annotations.append(
                {
                    "span_id": evidence.span_id,
                    "name": control_name,
                    "annotator_kind": "CODE",
                    "result": {
                        "label": _control_label(finding),
                        "score": finding.confidence,
                        "explanation": _dumps(
                            {
                                "annotator_kind": "CODE",
                                "control_id": finding.control_id,
                                "controls_version": finding.controls_version,
                                "evidence_ref": asdict(evidence),
                                "missing_evidence": finding.missing_evidence,
                                "pass_fail": finding.pass_fail,
                                "run_id": run_id,
                            }
                        ),
                    },
                }
            )
    return annotations


def _write_via_span_annotations(active_client: Any, annotations: list[dict[str, Any]]) -> list[str]:
    last_error: Exception | None = None
    if hasattr(active_client, "spans") and hasattr(active_client.spans, "log_span_annotations"):
        try:
            inserted = active_client.spans.log_span_annotations(span_annotations=annotations)
            if isinstance(inserted, list):
                return [str(item.get("id")) for item in inserted if isinstance(item, dict) and item.get("id")]
            return []
        except Exception as exc:  # pragma: no cover - exercised in live integration fallback paths
            last_error = exc
    if hasattr(active_client, "annotations") and hasattr(active_client.annotations, "log_span_annotations"):
        try:
            inserted = active_client.annotations.log_span_annotations(span_annotations=annotations)
            if isinstance(inserted, list):
                return [str(item.get("id")) for item in inserted if isinstance(item, dict) and item.get("id")]
            return []
        except Exception as exc:  # pragma: no cover - exercised in live integration fallback paths
            last_error = exc
    if last_error is not None:
        raise last_error
    raise AttributeError("Client does not support span annotation logging APIs.")


def _write_via_evaluations(active_client: Any, annotations: list[dict[str, Any]]) -> None:
    evals: list[SpanEvaluations] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        name = str(annotation["name"])
        result = annotation.get("result") or {}
        grouped.setdefault(name, []).append(
            {
                "span_id": annotation["span_id"],
                "label": result.get("label"),
                "score": result.get("score"),
                "explanation": result.get("explanation"),
            }
        )
    for name, rows in grouped.items():
        evals.append(SpanEvaluations(eval_name=name, dataframe=pd.DataFrame(rows)))
    active_client.log_evaluations(*evals)


def _endpoint_from_client(active_client: Any) -> str | None:
    wrapped_client = getattr(active_client, "_client", None)
    base_url = getattr(wrapped_client, "base_url", None)
    if base_url is None:
        return None
    return str(base_url).rstrip("/")


def write_compliance_to_phoenix(
    *,
    report: ComplianceReport,
    run_id: str,
    client: Any | None = None,
) -> dict[str, Any]:
    active_client = client or PhoenixClient()
    annotations = _build_span_annotations(report, run_id)
    phoenix_annotation_ids: list[str] = []

    try:
        phoenix_annotation_ids = _write_via_span_annotations(active_client, annotations)
    except Exception:
        fallback_client = SessionClient(
            endpoint=_endpoint_from_client(active_client),
            warn_if_server_not_running=False,
        )
        _write_via_evaluations(fallback_client, annotations)

    annotation_names: list[str] = []
    annotator_kinds: list[str] = []
    for annotation in annotations:
        name = str(annotation["name"])
        kind = str(annotation["annotator_kind"])
        if name not in annotation_names:
            annotation_names.append(name)
        if kind not in annotator_kinds:
            annotator_kinds.append(kind)

    return {
        "annotation_names": annotation_names,
        "annotator_kinds": annotator_kinds,
        "phoenix_annotation_ids": phoenix_annotation_ids,
        "writeback_status": "succeeded",
    }
