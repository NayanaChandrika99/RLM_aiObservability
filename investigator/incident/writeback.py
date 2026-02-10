# ABOUTME: Writes incident dossier outputs to Phoenix annotations with trace and timeline evidence rows.
# ABOUTME: Preserves run_id and evidence-pointer metadata in explanation payloads for auditability.

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

import pandas as pd
from phoenix.client import Client as PhoenixClient
from phoenix.trace import SpanEvaluations
from phoenix.session.client import Client as SessionClient

from investigator.runtime.contracts import EvidenceRef, IncidentDossier, IncidentTimelineEvent


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _normalize_annotator_kind(kind: str) -> str:
    normalized = str(kind or "").strip().upper()
    if normalized in {"LLM", "HUMAN", "CODE"}:
        return normalized
    return "CODE"


def _build_root_annotations(
    report: IncidentDossier,
    run_id: str,
    *,
    primary_annotator_kind: str,
) -> list[dict[str, Any]]:
    root_kind = _normalize_annotator_kind(primary_annotator_kind)
    rows: list[dict[str, Any]] = []
    seen_spans: set[str] = set()
    for representative in report.representative_traces:
        if not representative.evidence_refs:
            continue
        span_id = representative.evidence_refs[0].span_id
        if not span_id or span_id in seen_spans:
            continue
        seen_spans.add(span_id)
        rows.append(
            {
                "span_id": span_id,
                "name": "incident.dossier",
                "annotator_kind": root_kind,
                "result": {
                    "label": "incident_dossier",
                    "score": report.confidence,
                    "explanation": _dumps(
                        {
                            "annotator_kind": root_kind,
                            "report": report.to_dict(),
                            "run_id": run_id,
                        }
                    ),
                },
            }
        )
    if rows:
        return rows
    return [
        {
            "span_id": "root-span",
            "name": "incident.dossier",
            "annotator_kind": root_kind,
            "result": {
                "label": "incident_dossier",
                "score": report.confidence,
                "explanation": _dumps(
                    {
                        "annotator_kind": root_kind,
                        "report": report.to_dict(),
                        "run_id": run_id,
                    }
                ),
            },
        }
    ]


def _timeline_label(event: IncidentTimelineEvent, evidence: EvidenceRef) -> str:
    if evidence.kind == "CONFIG_DIFF" or "diff" in event.event.lower():
        return "change_evidence"
    if "hypothesis" in event.event.lower():
        return "hypothesis_evidence"
    return "timeline_event"


def _timeline_score(evidence: EvidenceRef) -> float:
    if evidence.kind == "CONFIG_DIFF":
        return 0.9
    if evidence.kind in {"TOOL_IO", "RETRIEVAL_CHUNK"}:
        return 0.85
    return 0.8


def _build_timeline_annotations(report: IncidentDossier, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sorted_timeline = sorted(report.timeline, key=lambda item: item.timestamp)
    for event in sorted_timeline:
        evidence_refs = sorted(
            event.evidence_refs,
            key=lambda evidence: (evidence.span_id, evidence.kind, evidence.ref),
        )
        for evidence in evidence_refs:
            rows.append(
                {
                    "span_id": evidence.span_id,
                    "name": "incident.timeline.evidence",
                    "annotator_kind": "CODE",
                    "result": {
                        "label": _timeline_label(event, evidence),
                        "score": _timeline_score(evidence),
                        "explanation": _dumps(
                            {
                                "annotator_kind": "CODE",
                                "event": event.event,
                                "timestamp": event.timestamp,
                                "evidence_ref": asdict(evidence),
                                "run_id": run_id,
                            }
                        ),
                    },
                }
            )
    return rows


def _build_span_annotations(
    report: IncidentDossier,
    run_id: str,
    *,
    primary_annotator_kind: str,
) -> list[dict[str, Any]]:
    return _build_root_annotations(
        report,
        run_id,
        primary_annotator_kind=primary_annotator_kind,
    ) + _build_timeline_annotations(report, run_id)


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


def write_incident_to_phoenix(
    *,
    report: IncidentDossier,
    run_id: str,
    client: Any | None = None,
    primary_annotator_kind: str = "LLM",
) -> dict[str, Any]:
    active_client = client or PhoenixClient()
    annotations = _build_span_annotations(
        report,
        run_id,
        primary_annotator_kind=primary_annotator_kind,
    )
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
