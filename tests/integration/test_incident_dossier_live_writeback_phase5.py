# ABOUTME: Validates incident write-back against a live local Phoenix server using real span annotations.
# ABOUTME: Ensures incident root and timeline annotation contracts round-trip with evidence metadata.

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import time
from typing import Any
from uuid import uuid4

import pytest
import requests
from phoenix.client import Client

from investigator.incident.writeback import write_incident_to_phoenix
from investigator.runtime.contracts import (
    EvidenceRef,
    IncidentDossier,
    IncidentHypothesis,
    IncidentTimelineEvent,
    RecommendedAction,
    RepresentativeTrace,
    SuspectedChange,
    hash_excerpt,
)


def _maybe_skip_without_live_server() -> tuple[str, str]:
    if os.getenv("PHASE5C_LIVE", "0") != "1":
        pytest.skip("Set PHASE5C_LIVE=1 to run live Phoenix incident write-back integration tests.")
    base_url = os.getenv("PHOENIX_BASE_URL", "http://127.0.0.1:6006").rstrip("/")
    try:
        requests.get(base_url, timeout=2)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Live Phoenix server is not reachable at {base_url}: {exc}")
    project_identifier = os.getenv(
        "PHASE5C_LIVE_PROJECT",
        f"phase5c-live-{uuid4().hex[:8]}",
    )
    return base_url, project_identifier


def _field(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _wait_for_annotations(
    *,
    client: Client,
    span_ids: list[str],
    project_identifier: str,
    include_annotation_names: list[str],
    timeout_seconds: float = 8.0,
) -> list[Any]:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            annotations = client.spans.get_span_annotations(
                span_ids=span_ids,
                project_identifier=project_identifier,
                include_annotation_names=include_annotation_names,
                limit=200,
            )
            if annotations:
                return annotations
        except Exception as exc:
            last_error = exc
        time.sleep(0.4)
    if last_error is not None:
        raise last_error
    return []


def test_incident_writeback_roundtrip_with_live_phoenix() -> None:
    base_url, project_identifier = _maybe_skip_without_live_server()
    client = Client(base_url=base_url)

    now = datetime.now(tz=timezone.utc)
    trace_id = uuid4().hex
    root_span_id = uuid4().hex[:16]
    tool_span_id = uuid4().hex[:16]
    spans = [
        {
            "name": "phase5c.live.agent",
            "context": {"trace_id": trace_id, "span_id": root_span_id},
            "span_kind": "AGENT",
            "start_time": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": (now + timedelta(milliseconds=150)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status_code": "ERROR",
            "status_message": "timeout",
            "attributes": {"phase": "5c-live"},
        },
        {
            "name": "phase5c.live.tool",
            "context": {"trace_id": trace_id, "span_id": tool_span_id},
            "parent_id": root_span_id,
            "span_kind": "TOOL",
            "start_time": (now + timedelta(milliseconds=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": (now + timedelta(milliseconds=140)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status_code": "ERROR",
            "status_message": "upstream timeout",
            "attributes": {"phase": "5c-live"},
        },
    ]
    client.spans.log_spans(project_identifier=project_identifier, spans=spans)

    lead_evidence = EvidenceRef(
        trace_id=trace_id,
        span_id=root_span_id,
        kind="SPAN",
        ref=root_span_id,
        excerpt_hash=hash_excerpt("phase5c-live-root"),
        ts=spans[0]["start_time"],
    )
    diff_evidence = EvidenceRef(
        trace_id=trace_id,
        span_id=root_span_id,
        kind="CONFIG_DIFF",
        ref="configdiff:phase5c-live",
        excerpt_hash=hash_excerpt("phase5c-live-diff"),
        ts=spans[1]["end_time"],
    )
    report = IncidentDossier(
        incident_summary="live phase5c dossier",
        impacted_components=["phase5c.live.agent"],
        timeline=[
            IncidentTimelineEvent(
                timestamp=spans[0]["start_time"],
                event="Incident window opened.",
                evidence_refs=[lead_evidence],
            ),
            IncidentTimelineEvent(
                timestamp=spans[1]["end_time"],
                event="Config diff correlated with incident.",
                evidence_refs=[lead_evidence, diff_evidence],
            ),
        ],
        representative_traces=[
            RepresentativeTrace(
                trace_id=trace_id,
                why_selected="error_bucket(score=2)",
                evidence_refs=[lead_evidence],
            )
        ],
        suspected_change=SuspectedChange(
            change_type="config",
            change_ref="live-commit",
            diff_ref="configdiff:phase5c-live",
            summary="Live config change correlated.",
            evidence_refs=[lead_evidence, diff_evidence],
        ),
        hypotheses=[
            IncidentHypothesis(
                rank=1,
                statement="Likely timeout due to config change.",
                evidence_refs=[lead_evidence, diff_evidence],
                confidence=0.77,
            )
        ],
        recommended_actions=[
            RecommendedAction(
                priority="P1",
                type="follow_up_fix",
                action="Revert config and monitor latency.",
            )
        ],
        confidence=0.77,
    )

    result = write_incident_to_phoenix(
        report=report,
        run_id=f"run-{uuid4().hex[:10]}",
        client=client,
    )
    assert "incident.dossier" in result["annotation_names"]
    assert "incident.timeline.evidence" in result["annotation_names"]

    annotations = _wait_for_annotations(
        client=client,
        span_ids=[root_span_id, tool_span_id],
        project_identifier=project_identifier,
        include_annotation_names=result["annotation_names"],
    )
    assert annotations
    names = {str(_field(annotation, "name", "")) for annotation in annotations}
    assert "incident.dossier" in names
    assert "incident.timeline.evidence" in names

    root = next(annotation for annotation in annotations if _field(annotation, "name") == "incident.dossier")
    root_result = _field(root, "result", {}) or {}
    explanation = _field(root_result, "explanation", "")
    payload = json.loads(explanation)
    assert payload["report"]["incident_summary"] == "live phase5c dossier"
