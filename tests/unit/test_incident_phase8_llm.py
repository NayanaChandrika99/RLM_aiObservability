# ABOUTME: Validates Phase 8E incident engine migration to per-trace shared-LLM synthesis with runtime usage accounting.
# ABOUTME: Ensures deterministic fallback and writeback provenance reflect true synthesis source.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.incident.engine import IncidentDossierEngine, IncidentDossierRequest
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
from investigator.runtime.llm_client import StructuredGenerationResult, StructuredGenerationUsage
from investigator.runtime.runner import run_engine


class _FakeIncidentInspectionAPI:
    def list_traces(  # noqa: ANN201
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ):
        del project_name, start_time, end_time, filter_expr
        return [
            {
                "project_name": "phase8",
                "trace_id": "trace-inc-a",
                "span_count": 2,
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:02Z",
                "latency_ms": 180.0,
            },
            {
                "project_name": "phase8",
                "trace_id": "trace-inc-b",
                "span_count": 1,
                "start_time": "2026-02-10T00:01:00Z",
                "end_time": "2026-02-10T00:01:02Z",
                "latency_ms": 150.0,
            },
        ]

    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        if trace_id == "trace-inc-a":
            return [
                {
                    "trace_id": trace_id,
                    "span_id": "root-a",
                    "parent_id": None,
                    "name": "svc.a.agent",
                    "span_kind": "AGENT",
                    "status_code": "ERROR",
                    "status_message": "timeout",
                    "start_time": "2026-02-10T00:00:00Z",
                    "end_time": "2026-02-10T00:00:02Z",
                    "latency_ms": 180.0,
                },
                {
                    "trace_id": trace_id,
                    "span_id": "tool-a",
                    "parent_id": "root-a",
                    "name": "db_query",
                    "span_kind": "TOOL",
                    "status_code": "ERROR",
                    "status_message": "timeout",
                    "start_time": "2026-02-10T00:00:01Z",
                    "end_time": "2026-02-10T00:00:02Z",
                    "latency_ms": 120.0,
                },
            ]
        return [
            {
                "trace_id": trace_id,
                "span_id": "root-b",
                "parent_id": None,
                "name": "svc.b.agent",
                "span_kind": "AGENT",
                "status_code": "OK",
                "status_message": "",
                "start_time": "2026-02-10T00:01:00Z",
                "end_time": "2026-02-10T00:01:02Z",
                "latency_ms": 150.0,
            }
        ]

    def list_config_snapshots(  # noqa: ANN201
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        tag: str | None = None,
    ):
        del project_name, start_time, end_time, tag
        return [
            {
                "snapshot_id": "snap-a",
                "project_name": "phase8",
                "tag": "baseline",
                "created_at": "2026-02-10T00:00:00Z",
                "git_commit": "abc111",
                "paths": ["agent.yaml"],
                "metadata": {},
            },
            {
                "snapshot_id": "snap-b",
                "project_name": "phase8",
                "tag": "candidate",
                "created_at": "2026-02-10T00:20:00Z",
                "git_commit": "abc222",
                "paths": ["prompts/incident.txt"],
                "metadata": {},
            },
        ]

    def get_config_diff(self, base_snapshot_id: str, target_snapshot_id: str):  # noqa: ANN201
        del base_snapshot_id, target_snapshot_id
        return {
            "project_name": "phase8",
            "base_snapshot_id": "snap-a",
            "target_snapshot_id": "snap-b",
            "artifact_id": "configdiff:phase8-1",
            "diff_ref": "configdiff:phase8-1",
            "git_commit_base": "abc111",
            "git_commit_target": "abc222",
            "paths": ["prompts/incident.txt"],
            "summary": "1 changed file(s)",
        }


class _FakeModelClient:
    model_provider = "openai"

    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    def generate_structured(self, request):  # noqa: ANN001, ANN201
        del request
        if not self._outputs:
            raise AssertionError("No fake outputs configured.")
        self.calls += 1
        payload = self._outputs.pop(0)
        return StructuredGenerationResult(
            output=payload,
            raw_text=json.dumps(payload, sort_keys=True),
            usage=StructuredGenerationUsage(tokens_in=40, tokens_out=15, cost_usd=0.03),
        )


class _FakeSpansResource:
    def __init__(self) -> None:
        self.logged_annotations = []

    def log_span_annotations(self, *, span_annotations, sync: bool = False) -> None:  # noqa: ANN001
        del sync
        self.logged_annotations.extend(span_annotations)


class _FakePhoenixClient:
    def __init__(self) -> None:
        self.spans = _FakeSpansResource()


def _incident_report_fixture() -> IncidentDossier:
    lead = EvidenceRef(
        trace_id="trace-inc-a",
        span_id="root-a",
        kind="SPAN",
        ref="root-a",
        excerpt_hash=hash_excerpt("lead"),
        ts="2026-02-10T00:00:00Z",
    )
    diff = EvidenceRef(
        trace_id="trace-inc-a",
        span_id="root-a",
        kind="CONFIG_DIFF",
        ref="configdiff:phase8-1",
        excerpt_hash=hash_excerpt("diff"),
        ts="2026-02-10T00:20:00Z",
    )
    return IncidentDossier(
        incident_summary="phase8 incident summary",
        impacted_components=["svc.a.agent"],
        timeline=[
            IncidentTimelineEvent(
                timestamp="2026-02-10T00:00:00Z",
                event="window opened",
                evidence_refs=[lead],
            )
        ],
        representative_traces=[
            RepresentativeTrace(
                trace_id="trace-inc-a",
                why_selected="error_bucket(score=1.00, error_spans=1, latency_ms=180.00)",
                evidence_refs=[lead],
            )
        ],
        suspected_change=SuspectedChange(
            change_type="prompt",
            change_ref="abc222",
            diff_ref="configdiff:phase8-1",
            summary="prompt change",
            evidence_refs=[lead, diff],
        ),
        hypotheses=[
            IncidentHypothesis(
                rank=1,
                statement="Prompt drift likely contributed to incident.",
                evidence_refs=[lead, diff],
                confidence=0.74,
            )
        ],
        recommended_actions=[
            RecommendedAction(
                priority="P1",
                type="follow_up_fix",
                action="Rollback prompt and compare trace behavior.",
            )
        ],
        confidence=0.74,
    )


def test_incident_engine_llm_per_trace_synthesis_emits_runtime_usage(tmp_path: Path) -> None:
    model_client = _FakeModelClient(
        outputs=[
            {
                "incident_summary": "Trace A indicates timeout concentration.",
                "hypotheses": [
                    {
                        "statement": "Timeouts originate from db_query path.",
                        "confidence": 0.8,
                    }
                ],
                "recommended_actions": [
                    {
                        "priority": "P1",
                        "type": "follow_up_fix",
                        "action": "Add timeout backoff.",
                    }
                ],
                "gaps": [],
            },
            {
                "incident_summary": "Trace B suggests latency spillover.",
                "hypotheses": [
                    {
                        "statement": "Latency increase follows prompt change.",
                        "confidence": 0.7,
                    }
                ],
                "recommended_actions": [
                    {
                        "priority": "P2",
                        "type": "mitigation",
                        "action": "Throttle traffic during rollout.",
                    }
                ],
                "gaps": [],
            },
        ]
    )
    engine = IncidentDossierEngine(
        inspection_api=_FakeIncidentInspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
        max_representatives=2,
        error_quota=1,
        latency_quota=1,
        cluster_quota=0,
    )

    report, run_record = run_engine(
        engine=engine,
        request=IncidentDossierRequest(
            project_name="phase8",
            time_window_start="2026-02-10T00:00:00Z",
            time_window_end="2026-02-10T01:00:00Z",
        ),
        run_id="run-phase8-incident-llm",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert len(report.representative_traces) == 2
    assert len(report.hypotheses) >= 2
    assert report.recommended_actions
    assert model_client.calls == 2
    assert run_record.runtime_ref.usage.tokens_in == 80
    assert run_record.runtime_ref.usage.tokens_out == 30
    assert run_record.runtime_ref.usage.cost_usd == 0.06
    assert run_record.runtime_ref.model_provider == "openai"


def test_incident_engine_llm_invalid_output_can_fallback_to_deterministic() -> None:
    model_client = _FakeModelClient(
        outputs=[
            {"incident_summary": "missing required fields"},
            {"still": "invalid"},
        ]
    )
    engine = IncidentDossierEngine(
        inspection_api=_FakeIncidentInspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
        fallback_on_llm_error=True,
        max_representatives=1,
        error_quota=1,
        latency_quota=0,
        cluster_quota=0,
    )

    report = engine.run(
        IncidentDossierRequest(
            project_name="phase8",
            time_window_start="2026-02-10T00:00:00Z",
            time_window_end="2026-02-10T01:00:00Z",
        )
    )
    signals = engine.get_runtime_signals()

    assert report.hypotheses
    assert model_client.calls == 2
    assert signals["incident_judgment_mode"] == "deterministic_fallback"


def test_write_incident_to_phoenix_supports_primary_annotator_provenance() -> None:
    client = _FakePhoenixClient()
    report = _incident_report_fixture()

    result = write_incident_to_phoenix(
        report=report,
        run_id="run-phase8-incident-writeback",
        client=client,
        primary_annotator_kind="CODE",
    )

    assert result["annotator_kinds"] == ["CODE"]
    root = next(item for item in client.spans.logged_annotations if item["name"] == "incident.dossier")
    payload = json.loads(root["result"]["explanation"])
    assert payload["annotator_kind"] == "CODE"
