# ABOUTME: Validates compliance write-back against a live local Phoenix server using real span annotations.
# ABOUTME: Ensures production annotation names and explanation payload contracts round-trip correctly.

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

from investigator.compliance.writeback import write_compliance_to_phoenix
from investigator.runtime.contracts import ComplianceFinding, ComplianceReport, EvidenceRef, hash_excerpt


def _maybe_skip_without_live_server() -> tuple[str, str]:
    if os.getenv("PHASE4C_LIVE", "0") != "1":
        pytest.skip("Set PHASE4C_LIVE=1 to run live Phoenix integration tests.")
    base_url = os.getenv("PHOENIX_BASE_URL", "http://127.0.0.1:6006").rstrip("/")
    try:
        requests.get(base_url, timeout=2)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Live Phoenix server is not reachable at {base_url}: {exc}")
    project_identifier = os.getenv(
        "PHASE4C_LIVE_PROJECT",
        f"phase4c-live-{uuid4().hex[:8]}",
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
                limit=100,
            )
            if annotations:
                return annotations
        except Exception as exc:
            last_error = exc
        time.sleep(0.4)
    if last_error is not None:
        raise last_error
    return []


def test_compliance_writeback_roundtrip_with_live_phoenix() -> None:
    base_url, project_identifier = _maybe_skip_without_live_server()
    client = Client(base_url=base_url)

    now = datetime.now(tz=timezone.utc)
    trace_id = uuid4().hex
    span_id = uuid4().hex[:16]
    span = {
        "name": "phase4c.live.agent",
        "context": {"trace_id": trace_id, "span_id": span_id},
        "span_kind": "AGENT",
        "start_time": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": (now + timedelta(milliseconds=100)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status_code": "OK",
        "status_message": "",
        "attributes": {"phase": "4c-live"},
    }
    client.spans.log_spans(project_identifier=project_identifier, spans=[span])

    report = ComplianceReport(
        trace_id=trace_id,
        controls_version="controls-v1",
        controls_evaluated=[
            ComplianceFinding(
                controls_version="controls-v1",
                control_id="control.live",
                pass_fail="pass",
                severity="high",
                confidence=0.88,
                evidence_refs=[
                    EvidenceRef(
                        trace_id=trace_id,
                        span_id=span_id,
                        kind="SPAN",
                        ref=span_id,
                        excerpt_hash=hash_excerpt("phase4c-live"),
                        ts=span["start_time"],
                    )
                ],
                missing_evidence=[],
                remediation="No remediation needed.",
            )
        ],
        overall_verdict="compliant",
        overall_confidence=0.88,
    )
    result = write_compliance_to_phoenix(report=report, run_id=f"run-{uuid4().hex[:10]}", client=client)
    assert "compliance.overall" in result["annotation_names"]

    annotations = _wait_for_annotations(
        client=client,
        span_ids=[span_id],
        project_identifier=project_identifier,
        include_annotation_names=result["annotation_names"],
    )
    assert annotations
    names = {str(_field(annotation, "name", "")) for annotation in annotations}
    assert "compliance.overall" in names
    assert "compliance.control.control.live.evidence" in names

    root = next(annotation for annotation in annotations if _field(annotation, "name") == "compliance.overall")
    root_result = _field(root, "result", {}) or {}
    explanation = _field(root_result, "explanation", "")
    payload = json.loads(explanation)
    assert payload["controls_version"] == "controls-v1"
    assert payload["report"]["trace_id"] == trace_id
