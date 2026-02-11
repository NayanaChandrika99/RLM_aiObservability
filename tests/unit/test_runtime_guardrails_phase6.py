# ABOUTME: Validates runtime Phase 6A guardrails for budget termination and sandbox violation handling.
# ABOUTME: Ensures run artifacts persist with explicit error taxonomy when guardrails trigger.

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from investigator.runtime.contracts import InputRef, RuntimeBudget
from investigator.runtime.runner import run_engine


class _Output:
    schema_version = "1.0.0"

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": "trace-phase6",
            "primary_label": "instruction_failure",
            "summary": "phase6 guardrail output",
            "confidence": 0.5,
            "evidence_refs": [
                {
                    "trace_id": "trace-phase6",
                    "span_id": "root-span",
                    "kind": "SPAN",
                    "ref": "root-span",
                    "excerpt_hash": "phase6hash",
                    "ts": "2026-02-10T00:00:00Z",
                }
            ],
            "schema_version": self.schema_version,
        }


class _EngineWithSignals:
    engine_type = "rca"
    output_contract_name = "RCAReport"
    engine_version = "phase6-test"
    model_name = "gpt-5-mini"
    prompt_template_hash = "phase6-test"
    temperature = 0.0

    def __init__(self, *, signals: dict[str, object] | None = None, sleep_ms: int = 0) -> None:
        self._signals = signals or {}
        self._sleep_ms = sleep_ms

    def build_input_ref(self, request: str) -> InputRef:  # noqa: ARG002
        return InputRef(project_name="phase6")

    def run(self, request: str) -> _Output:  # noqa: ARG002
        if self._sleep_ms > 0:
            time.sleep(self._sleep_ms / 1000.0)
        return _Output()

    def get_runtime_signals(self) -> dict[str, object]:
        return dict(self._signals)


def test_runtime_marks_partial_when_iteration_budget_exceeded(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    engine = _EngineWithSignals(signals={"iterations": 7, "depth_reached": 1, "tool_calls": 2})
    budget = RuntimeBudget(max_iterations=2)

    _, run_record = run_engine(
        engine=engine,
        request="demo",
        run_id="run-phase6-iter",
        budget=budget,
        artifacts_root=artifacts_root,
    )

    assert run_record.status == "partial"
    assert run_record.error is not None
    assert run_record.error.code == "RECURSION_LIMIT_REACHED"
    assert run_record.runtime_ref.usage.iterations == 7
    persisted = json.loads((artifacts_root / "run-phase6-iter" / "run_record.json").read_text())
    assert persisted["status"] == "partial"
    assert persisted["error"]["code"] == "RECURSION_LIMIT_REACHED"


def test_runtime_marks_partial_when_wall_time_exceeded(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    engine = _EngineWithSignals(sleep_ms=30)
    budget = RuntimeBudget(max_wall_time_sec=0)

    _, run_record = run_engine(
        engine=engine,
        request="demo",
        run_id="run-phase6-wall",
        budget=budget,
        artifacts_root=artifacts_root,
    )

    assert run_record.status == "partial"
    assert run_record.error is not None
    assert run_record.error.code == "WALL_TIME_LIMIT_REACHED"


def test_runtime_fails_when_sandbox_violation_signaled(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    engine = _EngineWithSignals(
        signals={"sandbox_violations": ["Blocked subprocess execution attempt."]},
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_engine(
            engine=engine,
            request="demo",
            run_id="run-phase6-sandbox",
            artifacts_root=artifacts_root,
        )

    payload = exc_info.value.args[0]
    assert isinstance(payload, dict)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "SANDBOX_VIOLATION"
    persisted = json.loads((artifacts_root / "run-phase6-sandbox" / "run_record.json").read_text())
    assert persisted["status"] == "failed"
    assert persisted["error"]["code"] == "SANDBOX_VIOLATION"
