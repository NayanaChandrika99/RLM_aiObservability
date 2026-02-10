# ABOUTME: Runs Trace RCA end-to-end and updates RunRecord with Phoenix write-back metadata.
# ABOUTME: Persists run_record.json after write-back so every RCA run has auditable artifacts.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.rca.writeback import write_rca_to_phoenix
from investigator.runtime.contracts import DatasetRef, RunError, RunRecord, RuntimeBudget
from investigator.runtime.runner import run_engine


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_record_path(*, artifacts_root: str | Path, run_id: str) -> Path:
    return Path(artifacts_root) / run_id / "run_record.json"


def _primary_annotator_kind(*, engine: TraceRCAEngine, run_record: RunRecord) -> str:
    if hasattr(engine, "get_runtime_signals"):
        candidate = engine.get_runtime_signals()
        if isinstance(candidate, dict):
            mode = str(candidate.get("rca_judgment_mode") or "").strip().lower()
            if mode in {"deterministic", "deterministic_fallback"}:
                return "CODE"
            if mode == "llm":
                return "LLM"
    usage = run_record.runtime_ref.usage
    if int(usage.tokens_in) <= 0 and int(usage.tokens_out) <= 0:
        return "CODE"
    return "LLM"


def run_trace_rca_workflow(
    *,
    request: TraceRCARequest,
    engine: TraceRCAEngine | None = None,
    run_id: str | None = None,
    budget: RuntimeBudget | None = None,
    dataset_ref: DatasetRef | None = None,
    artifacts_root: str | Path = "artifacts/investigator_runs",
    writeback_client: Any | None = None,
) -> tuple[Any, RunRecord]:
    active_engine = engine or TraceRCAEngine()
    report, run_record = run_engine(
        engine=active_engine,
        request=request,
        run_id=run_id,
        budget=budget,
        dataset_ref=dataset_ref,
        artifacts_root=artifacts_root,
    )
    try:
        primary_annotator_kind = _primary_annotator_kind(engine=active_engine, run_record=run_record)
        writeback_result = write_rca_to_phoenix(
            report=report,
            run_id=run_record.run_id,
            client=writeback_client,
            primary_annotator_kind=primary_annotator_kind,
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
            code="RCA_WRITEBACK_FAILED",
            message=str(exc),
            stage="rca_writeback",
            retryable=True,
        )
    _write_json(_run_record_path(artifacts_root=artifacts_root, run_id=run_record.run_id), run_record.to_dict())
    return report, run_record
