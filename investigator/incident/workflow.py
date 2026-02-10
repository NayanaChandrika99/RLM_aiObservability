# ABOUTME: Runs incident-dossier evaluation end-to-end and records Phoenix write-back metadata in RunRecord.
# ABOUTME: Persists updated run_record.json so each incident run has auditable write-back status.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.incident.engine import IncidentDossierEngine, IncidentDossierRequest
from investigator.incident.writeback import write_incident_to_phoenix
from investigator.runtime.contracts import DatasetRef, RunError, RunRecord, RuntimeBudget
from investigator.runtime.runner import run_engine


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_record_path(*, artifacts_root: str | Path, run_id: str) -> Path:
    return Path(artifacts_root) / run_id / "run_record.json"


def run_incident_dossier_workflow(
    *,
    request: IncidentDossierRequest,
    engine: IncidentDossierEngine | None = None,
    run_id: str | None = None,
    budget: RuntimeBudget | None = None,
    dataset_ref: DatasetRef | None = None,
    artifacts_root: str | Path = "artifacts/investigator_runs",
    writeback_client: Any | None = None,
) -> tuple[Any, RunRecord]:
    active_engine = engine or IncidentDossierEngine()
    report, run_record = run_engine(
        engine=active_engine,
        request=request,
        run_id=run_id,
        budget=budget,
        dataset_ref=dataset_ref,
        artifacts_root=artifacts_root,
    )
    try:
        writeback_result = write_incident_to_phoenix(
            report=report,
            run_id=run_record.run_id,
            client=writeback_client,
        )
        run_record.writeback_ref.writeback_status = str(
            writeback_result.get("writeback_status") or "succeeded"
        )
        run_record.writeback_ref.annotation_names = [
            str(name) for name in (writeback_result.get("annotation_names") or [])
        ]
        run_record.writeback_ref.annotator_kinds = [
            str(kind) for kind in (writeback_result.get("annotator_kinds") or [])
        ]
        run_record.writeback_ref.phoenix_annotation_ids = [
            str(identifier)
            for identifier in (writeback_result.get("phoenix_annotation_ids") or [])
        ]
    except Exception as exc:
        run_record.status = "partial"
        run_record.writeback_ref.writeback_status = "failed"
        run_record.error = RunError(
            code="INCIDENT_WRITEBACK_FAILED",
            message=str(exc),
            stage="incident_writeback",
            retryable=True,
        )
    _write_json(_run_record_path(artifacts_root=artifacts_root, run_id=run_record.run_id), run_record.to_dict())
    return report, run_record
