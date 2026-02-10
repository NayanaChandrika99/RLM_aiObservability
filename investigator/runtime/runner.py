# ABOUTME: Provides a shared runtime entrypoint that executes one engine invocation.
# ABOUTME: Produces a RunRecord artifact object with contract-aligned metadata and status.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Protocol, TypeVar
from uuid import uuid4

from investigator.runtime.contracts import (
    DatasetRef,
    InputRef,
    OutputRef,
    RunError,
    RunRecord,
    RuntimeBudget,
    RuntimeRef,
    RuntimeUsage,
    WritebackRef,
    utc_now_rfc3339,
)
from investigator.runtime.llm_client import ModelOutputInvalidError
from investigator.runtime.validation import validate_output_evidence, validate_output_schema


RequestT = TypeVar("RequestT")
OutputT = TypeVar("OutputT")


class Engine(Protocol[RequestT, OutputT]):
    engine_type: str
    output_contract_name: str
    engine_version: str
    model_name: str
    prompt_template_hash: str
    temperature: float

    def build_input_ref(self, request: RequestT) -> InputRef:
        raise NotImplementedError

    def run(self, request: RequestT) -> OutputT:
        raise NotImplementedError


@dataclass
class RuntimeResult:
    output: object
    run_record: RunRecord


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_runtime_signals(
    engine: Engine[RequestT, OutputT],
) -> tuple[
    RuntimeUsage,
    int,
    list[str],
    str | None,
    str | None,
    str | None,
    list[str],
    list[dict[str, Any]],
]:
    runtime_signals: dict[str, Any] = {}
    if hasattr(engine, "get_runtime_signals"):
        candidate = engine.get_runtime_signals()
        if isinstance(candidate, dict):
            runtime_signals = candidate
    usage = RuntimeUsage(
        iterations=_safe_int(runtime_signals.get("iterations"), default=1),
        depth_reached=_safe_int(runtime_signals.get("depth_reached"), default=0),
        tool_calls=_safe_int(runtime_signals.get("tool_calls"), default=0),
        tokens_in=_safe_int(runtime_signals.get("tokens_in"), default=0),
        tokens_out=_safe_int(runtime_signals.get("tokens_out"), default=0),
        cost_usd=_safe_float(runtime_signals.get("cost_usd"), default=0.0),
    )
    subcalls = _safe_int(runtime_signals.get("subcalls"), default=0)
    raw_violations = runtime_signals.get("sandbox_violations") or []
    sandbox_violations = [str(item) for item in raw_violations if str(item).strip()]
    provider_candidate = str(runtime_signals.get("model_provider") or "").strip()
    model_provider = provider_candidate if provider_candidate else None
    runtime_state_raw = str(runtime_signals.get("runtime_state") or "").strip().lower()
    runtime_state = runtime_state_raw if runtime_state_raw else None
    budget_reason_raw = str(runtime_signals.get("budget_reason") or "").strip()
    budget_reason = budget_reason_raw if budget_reason_raw else None
    raw_trajectory = runtime_signals.get("state_trajectory") or []
    state_trajectory = [str(item) for item in raw_trajectory if str(item).strip()]
    raw_subcall_metadata = runtime_signals.get("subcall_metadata") or []
    subcall_metadata = [item for item in raw_subcall_metadata if isinstance(item, dict)]
    return (
        usage,
        subcalls,
        sandbox_violations,
        model_provider,
        runtime_state,
        budget_reason,
        state_trajectory,
        subcall_metadata,
    )


def _recursion_limit_messages(
    *,
    budget: RuntimeBudget,
    usage: RuntimeUsage,
    subcalls: int,
) -> list[str]:
    messages: list[str] = []
    if usage.iterations > budget.max_iterations:
        messages.append(f"iterations {usage.iterations} exceeded max_iterations {budget.max_iterations}")
    if usage.depth_reached > budget.max_depth:
        messages.append(f"depth {usage.depth_reached} exceeded max_depth {budget.max_depth}")
    if usage.tool_calls > budget.max_tool_calls:
        messages.append(f"tool_calls {usage.tool_calls} exceeded max_tool_calls {budget.max_tool_calls}")
    if subcalls > budget.max_subcalls:
        messages.append(f"subcalls {subcalls} exceeded max_subcalls {budget.max_subcalls}")
    if budget.max_tokens_total is not None:
        token_total = usage.tokens_in + usage.tokens_out
        if token_total > budget.max_tokens_total:
            messages.append(f"tokens_total {token_total} exceeded max_tokens_total {budget.max_tokens_total}")
    if budget.max_cost_usd is not None and usage.cost_usd > budget.max_cost_usd:
        messages.append(f"cost_usd {usage.cost_usd:.6f} exceeded max_cost_usd {budget.max_cost_usd}")
    return messages


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _raise_runtime_failure(
    *,
    run_record: RunRecord,
    run_record_path: Path,
) -> None:
    _write_json(run_record_path, run_record.to_dict())
    raise RuntimeError(run_record.to_dict())


def run_engine(
    engine: Engine[RequestT, OutputT],
    request: RequestT,
    *,
    run_id: str | None = None,
    budget: RuntimeBudget | None = None,
    dataset_ref: DatasetRef | None = None,
    artifacts_root: str | Path = "artifacts/investigator_runs",
) -> tuple[OutputT, RunRecord]:
    active_run_id = run_id or str(uuid4())
    active_budget = budget or RuntimeBudget()
    active_dataset_ref = dataset_ref or DatasetRef()
    started_at = utc_now_rfc3339()
    run_dir = Path(artifacts_root) / active_run_id
    run_record_path = run_dir / "run_record.json"
    output_path = run_dir / "output.json"
    start_monotonic = time.monotonic()

    try:
        output = engine.run(request)
        elapsed_seconds = time.monotonic() - start_monotonic
        (
            usage,
            subcalls,
            sandbox_violations,
            signal_model_provider,
            runtime_state,
            budget_reason,
            state_trajectory,
            subcall_metadata,
        ) = _collect_runtime_signals(engine)
        model_provider = signal_model_provider or str(getattr(engine, "model_provider", "openai"))
        output_payload = output.to_dict() if hasattr(output, "to_dict") else {"output": str(output)}
        completed_at = utc_now_rfc3339()
        run_record = RunRecord(
            run_id=active_run_id,
            run_type=engine.engine_type,
            status="succeeded",
            started_at=started_at,
            completed_at=completed_at,
            dataset_ref=active_dataset_ref,
            input_ref=engine.build_input_ref(request),
            runtime_ref=RuntimeRef(
                engine_version=engine.engine_version,
                model_provider=model_provider,
                model_name=engine.model_name,
                temperature=engine.temperature,
                prompt_template_hash=engine.prompt_template_hash,
                budget=active_budget,
                usage=usage,
                state_trajectory=state_trajectory,
                subcall_metadata=subcall_metadata,
            ),
            output_ref=OutputRef(
                artifact_type=engine.output_contract_name,
                artifact_path=str(output_path),
                schema_version=getattr(output, "schema_version", None),
            ),
            writeback_ref=WritebackRef(),
            error=None,
        )
        schema_errors = validate_output_schema(
            contract_name=engine.output_contract_name,
            payload=output_payload,
        )
        if schema_errors:
            run_record.status = "failed"
            run_record.output_ref.artifact_path = None
            run_record.output_ref.schema_version = None
            run_record.writeback_ref.writeback_status = "failed"
            run_record.error = RunError(
                code="SCHEMA_VALIDATION_FAILED",
                message="; ".join(schema_errors),
                stage="runtime_validation",
                retryable=False,
            )
            _raise_runtime_failure(run_record=run_record, run_record_path=run_record_path)

        evidence_errors = validate_output_evidence(
            contract_name=engine.output_contract_name,
            payload=output_payload,
        )
        if evidence_errors:
            run_record.status = "failed"
            run_record.output_ref.artifact_path = None
            run_record.output_ref.schema_version = None
            run_record.writeback_ref.writeback_status = "failed"
            run_record.error = RunError(
                code="EVIDENCE_VALIDATION_FAILED",
                message="; ".join(evidence_errors),
                stage="runtime_validation",
                retryable=False,
            )
            _raise_runtime_failure(run_record=run_record, run_record_path=run_record_path)

        if sandbox_violations:
            run_record.status = "failed"
            run_record.output_ref.artifact_path = None
            run_record.output_ref.schema_version = None
            run_record.writeback_ref.writeback_status = "failed"
            run_record.error = RunError(
                code="SANDBOX_VIOLATION",
                message="; ".join(sandbox_violations),
                stage="runtime_sandbox",
                retryable=False,
            )
            _raise_runtime_failure(run_record=run_record, run_record_path=run_record_path)

        _write_json(output_path, output_payload)

        if elapsed_seconds > float(active_budget.max_wall_time_sec):
            run_record.status = "partial"
            run_record.error = RunError(
                code="WALL_TIME_LIMIT_REACHED",
                message=(
                    f"Elapsed {elapsed_seconds:.3f}s exceeded max_wall_time_sec "
                    f"{active_budget.max_wall_time_sec}."
                ),
                stage="runtime_budget",
                retryable=True,
            )

        recursion_messages = _recursion_limit_messages(
            budget=active_budget,
            usage=usage,
            subcalls=subcalls,
        )
        budget_messages = list(recursion_messages)
        if runtime_state == "terminated_budget":
            if budget_reason:
                budget_messages.insert(0, budget_reason)
            elif not budget_messages:
                budget_messages.append("Runtime entered terminated_budget state.")
        if budget_messages:
            run_record.status = "partial"
            run_record.error = RunError(
                code="RECURSION_LIMIT_REACHED",
                message="; ".join(budget_messages),
                stage="runtime_budget",
                retryable=True,
            )

        _write_json(run_record_path, run_record.to_dict())
        return output, run_record
    except Exception as exc:  # pragma: no cover
        if isinstance(exc, RuntimeError) and exc.args and isinstance(exc.args[0], dict):
            raise
        (
            usage,
            _,
            _,
            signal_model_provider,
            _,
            _,
            state_trajectory,
            subcall_metadata,
        ) = _collect_runtime_signals(engine)
        if usage.iterations <= 0:
            usage.iterations = 1
        model_provider = signal_model_provider or str(getattr(engine, "model_provider", "openai"))
        error_code = "UNEXPECTED_RUNTIME_ERROR"
        error_stage = "run_engine"
        retryable = False
        if isinstance(exc, ModelOutputInvalidError):
            error_code = "MODEL_OUTPUT_INVALID"
            error_stage = "runtime_model"
        completed_at = utc_now_rfc3339()
        run_record = RunRecord(
            run_id=active_run_id,
            run_type=engine.engine_type,
            status="failed",
            started_at=started_at,
            completed_at=completed_at,
            dataset_ref=active_dataset_ref,
            input_ref=engine.build_input_ref(request),
            runtime_ref=RuntimeRef(
                engine_version=engine.engine_version,
                model_provider=model_provider,
                model_name=engine.model_name,
                temperature=engine.temperature,
                prompt_template_hash=engine.prompt_template_hash,
                budget=active_budget,
                usage=usage,
                state_trajectory=state_trajectory,
                subcall_metadata=subcall_metadata,
            ),
            output_ref=OutputRef(
                artifact_type=engine.output_contract_name,
                artifact_path=None,
                schema_version=None,
            ),
            writeback_ref=WritebackRef(writeback_status="failed"),
            error=RunError(
                code=error_code,
                message=str(exc),
                stage=error_stage,
                retryable=retryable,
            ),
        )
        _write_json(run_record_path, run_record.to_dict())
        raise RuntimeError(run_record.to_dict()) from exc
