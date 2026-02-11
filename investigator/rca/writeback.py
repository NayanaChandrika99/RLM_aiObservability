# ABOUTME: Writes RCA outputs to Phoenix using eval-compatible payloads and stable annotation names.
# ABOUTME: Encodes annotator metadata and evidence pointers in explanation JSON for auditability.

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

import pandas as pd
from phoenix.trace import SpanEvaluations, TraceEvaluations
from phoenix.session.client import Client

from investigator.runtime.contracts import EvidenceRef, RCAReport


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _evidence_label(evidence: EvidenceRef) -> str:
    if evidence.kind == "TOOL_IO":
        return "tool_error"
    if evidence.kind == "RETRIEVAL_CHUNK":
        return "retrieval_signal"
    if evidence.kind == "CONFIG_DIFF":
        return "config_signal"
    return "hot_span"


def _evidence_weight(evidence: EvidenceRef) -> float:
    if evidence.kind == "TOOL_IO":
        return 0.95
    if evidence.kind == "RETRIEVAL_CHUNK":
        return 0.9
    if evidence.kind == "CONFIG_DIFF":
        return 0.85
    return 0.8


def _build_primary_evaluation(
    report: RCAReport,
    run_id: str,
    *,
    annotator_kind: str,
) -> TraceEvaluations:
    explanation = _dumps(
        {
            "annotator_kind": annotator_kind,
            "run_id": run_id,
            "report": report.to_dict(),
        }
    )
    dataframe = pd.DataFrame(
        [
            {
                "trace_id": report.trace_id,
                "label": report.primary_label,
                "score": report.confidence,
                "explanation": explanation,
            }
        ]
    )
    return TraceEvaluations(eval_name="rca.primary", dataframe=dataframe)


def _build_evidence_evaluation(report: RCAReport, run_id: str) -> SpanEvaluations:
    sorted_evidence = sorted(report.evidence_refs, key=lambda item: (item.span_id, item.kind, item.ref))
    rows = []
    for evidence in sorted_evidence:
        rows.append(
            {
                "span_id": evidence.span_id,
                "label": _evidence_label(evidence),
                "score": _evidence_weight(evidence),
                "explanation": _dumps(
                    {
                        "annotator_kind": "CODE",
                        "evidence_ref": asdict(evidence),
                        "run_id": run_id,
                        "why": "Selected as RCA evidence pointer.",
                    }
                ),
            }
        )
    return SpanEvaluations(eval_name="rca.evidence", dataframe=pd.DataFrame(rows))


def write_rca_to_phoenix(
    *,
    report: RCAReport,
    run_id: str,
    client: Any | None = None,
    primary_annotator_kind: str = "LLM",
) -> dict[str, Any]:
    active_client = client or Client(warn_if_server_not_running=False)
    primary_eval = _build_primary_evaluation(
        report,
        run_id,
        annotator_kind=primary_annotator_kind,
    )
    evidence_eval = _build_evidence_evaluation(report, run_id)
    active_client.log_evaluations(primary_eval, evidence_eval)
    return {
        "annotation_names": ["rca.primary", "rca.evidence"],
        "annotator_kinds": [primary_annotator_kind, "CODE"],
        "phoenix_annotation_ids": [],
        "writeback_status": "succeeded",
    }
