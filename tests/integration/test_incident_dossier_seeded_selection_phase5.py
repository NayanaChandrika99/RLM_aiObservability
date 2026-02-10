# ABOUTME: Validates deterministic incident representative selection over seeded Phase 1 style failure traces.
# ABOUTME: Ensures repeated runs over the same seeded manifest yield stable representative trace ordering.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.demo_agent.phase1_seeded_failures import build_seed_manifest
from investigator.incident.engine import IncidentDossierEngine, IncidentDossierRequest


@dataclass
class _TraceRow:
    trace_id: str
    latency_ms: float
    start_time: str
    end_time: str


@dataclass
class _SnapshotRow:
    snapshot_id: str
    created_at: str
    tag: str
    git_commit: str | None
    paths: list[str]


class _SeededIncidentInspectionAPI:
    def __init__(
        self,
        manifest: dict[str, Any],
        *,
        snapshots: list[_SnapshotRow] | None = None,
        diff_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self._trace_rows: list[_TraceRow] = []
        self._spans_by_trace: dict[str, list[dict[str, Any]]] = {}
        self._snapshots = snapshots or []
        self._diff_lookup = diff_lookup or {}
        self.last_diff_args: tuple[str, str] | None = None
        for index, case in enumerate(manifest["cases"]):
            trace_id = f"trace-seeded-{index:03d}"
            label = str(case["expected_label"])
            latency_ms = 300.0 + (index * 7.0)
            if label in {"upstream_dependency_failure", "tool_failure"}:
                latency_ms += 600.0
            start = f"2026-02-10T00:{index % 60:02d}:00Z"
            end = f"2026-02-10T00:{index % 60:02d}:01Z"
            self._trace_rows.append(
                _TraceRow(
                    trace_id=trace_id,
                    latency_ms=latency_ms,
                    start_time=start,
                    end_time=end,
                )
            )
            root_status = "ERROR" if label != "retrieval_failure" else "OK"
            tool_status = "ERROR" if label in {"tool_failure", "upstream_dependency_failure"} else "OK"
            error_message = label.replace("_", " ")
            self._spans_by_trace[trace_id] = [
                {
                    "trace_id": trace_id,
                    "span_id": f"{trace_id}-root",
                    "parent_id": None,
                    "name": f"svc.{label}.agent",
                    "span_kind": "AGENT",
                    "status_code": root_status,
                    "status_message": error_message if root_status == "ERROR" else "",
                    "start_time": start,
                    "end_time": end,
                    "latency_ms": latency_ms,
                },
                {
                    "trace_id": trace_id,
                    "span_id": f"{trace_id}-tool",
                    "parent_id": f"{trace_id}-root",
                    "name": "seeded_tool",
                    "span_kind": "TOOL",
                    "status_code": tool_status,
                    "status_message": error_message if tool_status == "ERROR" else "",
                    "start_time": start,
                    "end_time": end,
                    "latency_ms": 30.0,
                },
            ]

    def list_traces(  # noqa: ANN201
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ):
        return [
            {
                "project_name": project_name,
                "trace_id": row.trace_id,
                "span_count": len(self._spans_by_trace[row.trace_id]),
                "start_time": row.start_time,
                "end_time": row.end_time,
                "latency_ms": row.latency_ms,
            }
            for row in self._trace_rows
        ]

    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return list(self._spans_by_trace.get(trace_id, []))

    def list_config_snapshots(
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        del project_name, start_time, end_time
        rows = list(self._snapshots)
        if tag is not None:
            rows = [row for row in rows if row.tag == tag]
        return [
            {
                "snapshot_id": row.snapshot_id,
                "project_name": "seeded-phase5",
                "tag": row.tag,
                "created_at": row.created_at,
                "git_commit": row.git_commit,
                "paths": row.paths,
                "metadata": {},
            }
            for row in rows
        ]

    def get_config_diff(self, base_snapshot_id: str, target_snapshot_id: str) -> dict[str, Any]:
        self.last_diff_args = (base_snapshot_id, target_snapshot_id)
        key = (base_snapshot_id, target_snapshot_id)
        if key not in self._diff_lookup:
            raise KeyError(key)
        return self._diff_lookup[key]


def test_incident_selector_is_stable_on_seeded_manifest() -> None:
    manifest = build_seed_manifest(seed=42, num_traces=30, dataset_id="seeded_failures_v1")
    api = _SeededIncidentInspectionAPI(manifest)
    engine = IncidentDossierEngine(
        inspection_api=api,
        max_representatives=8,
        error_quota=4,
        latency_quota=3,
        cluster_quota=1,
    )
    request = IncidentDossierRequest(
        project_name="seeded-phase5",
        time_window_start="2026-02-10T00:00:00Z",
        time_window_end="2026-02-10T01:00:00Z",
    )

    first = engine.run(request)
    second = engine.run(request)
    first_ids = [item.trace_id for item in first.representative_traces]
    second_ids = [item.trace_id for item in second.representative_traces]

    assert first_ids == second_ids
    assert len(first_ids) <= 8
    assert any(item.why_selected.startswith("error_bucket") for item in first.representative_traces)
    assert any(item.why_selected.startswith("latency_bucket") for item in first.representative_traces)


def test_incident_selector_adds_deterministic_config_diff_evidence() -> None:
    manifest = build_seed_manifest(seed=7, num_traces=24, dataset_id="seeded_failures_v1")
    snapshots = [
        _SnapshotRow("snap-001", "2026-02-10T00:00:00Z", "baseline", "def111", ["app.env"]),
        _SnapshotRow("snap-002", "2026-02-10T00:30:00Z", "candidate", "def222", ["app.env"]),
        _SnapshotRow("snap-003", "2026-02-10T00:45:00Z", "candidate", "def333", ["configs/tooling.json"]),
    ]
    diff_lookup = {
        ("snap-002", "snap-003"): {
            "project_name": "seeded-phase5",
            "base_snapshot_id": "snap-002",
            "target_snapshot_id": "snap-003",
            "artifact_id": "configdiff:seeded-456",
            "diff_ref": "configdiff:seeded-456",
            "git_commit_base": "def222",
            "git_commit_target": "def333",
            "paths": ["configs/tooling.json"],
            "summary": "1 changed file(s)",
        }
    }
    api = _SeededIncidentInspectionAPI(
        manifest,
        snapshots=snapshots,
        diff_lookup=diff_lookup,
    )
    engine = IncidentDossierEngine(
        inspection_api=api,
        max_representatives=6,
        error_quota=3,
        latency_quota=2,
        cluster_quota=1,
    )
    request = IncidentDossierRequest(
        project_name="seeded-phase5",
        time_window_start="2026-02-10T00:00:00Z",
        time_window_end="2026-02-10T01:00:00Z",
    )

    dossier = engine.run(request)

    assert api.last_diff_args == ("snap-002", "snap-003")
    assert dossier.suspected_change.change_type == "config"
    assert dossier.suspected_change.diff_ref == "configdiff:seeded-456"
    assert any(
        evidence.kind == "CONFIG_DIFF" and evidence.ref == "configdiff:seeded-456"
        for evidence in dossier.suspected_change.evidence_refs
    )
