# ABOUTME: Validates Phase 6B runtime validation failures for schema and evidence contracts.
# ABOUTME: Ensures failed validations persist RunRecord with contract error codes.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from investigator.runtime.contracts import InputRef
from investigator.runtime.runner import run_engine


class _OutputWithPayload:
    schema_version = "1.0.0"

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)


class _EngineWithPayload:
    engine_type = "rca"
    output_contract_name = "RCAReport"
    engine_version = "phase6-validation"
    model_name = "gpt-5-mini"
    prompt_template_hash = "phase6-validation"
    temperature = 0.0

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def build_input_ref(self, request: str) -> InputRef:  # noqa: ARG002
        return InputRef(project_name="phase6")

    def run(self, request: str) -> _OutputWithPayload:  # noqa: ARG002
        return _OutputWithPayload(self._payload)


def test_runtime_fails_on_schema_validation_error(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    engine = _EngineWithPayload(
        payload={
            "trace_id": "trace-invalid",
            "summary": "missing required primary_label and invalid confidence",
            "confidence": "not-a-number",
            "evidence_refs": [],
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_engine(
            engine=engine,
            request="demo",
            run_id="run-phase6-schema",
            artifacts_root=artifacts_root,
        )

    payload = exc_info.value.args[0]
    assert isinstance(payload, dict)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "SCHEMA_VALIDATION_FAILED"
    persisted = json.loads((artifacts_root / "run-phase6-schema" / "run_record.json").read_text())
    assert persisted["status"] == "failed"
    assert persisted["error"]["code"] == "SCHEMA_VALIDATION_FAILED"


def test_runtime_fails_on_evidence_validation_error(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    engine = _EngineWithPayload(
        payload={
            "trace_id": "trace-invalid-evidence",
            "primary_label": "tool_failure",
            "summary": "has malformed evidence",
            "confidence": 0.62,
            "evidence_refs": [
                {
                    "trace_id": "trace-invalid-evidence",
                    "span_id": "",
                    "kind": "SPAN",
                    "ref": "root-span",
                    "excerpt_hash": "abc",
                    "ts": None,
                }
            ],
            "schema_version": "1.0.0",
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_engine(
            engine=engine,
            request="demo",
            run_id="run-phase6-evidence",
            artifacts_root=artifacts_root,
        )

    payload = exc_info.value.args[0]
    assert isinstance(payload, dict)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "EVIDENCE_VALIDATION_FAILED"
    persisted = json.loads((artifacts_root / "run-phase6-evidence" / "run_record.json").read_text())
    assert persisted["status"] == "failed"
    assert persisted["error"]["code"] == "EVIDENCE_VALIDATION_FAILED"
