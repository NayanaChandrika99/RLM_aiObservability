# ABOUTME: Validates Phase 8 runtime run-record accounting for model provider and cost usage fields.
# ABOUTME: Ensures runtime signals emitted by engines persist into run_record.json artifacts.

from __future__ import annotations

import json
from pathlib import Path

from investigator.runtime.contracts import InputRef
from investigator.runtime.runner import run_engine


class _Output:
    schema_version = "1.0.0"

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": "trace-phase8",
            "primary_label": "tool_failure",
            "summary": "phase8 output",
            "confidence": 0.63,
            "evidence_refs": [
                {
                    "trace_id": "trace-phase8",
                    "span_id": "root-span",
                    "kind": "SPAN",
                    "ref": "root-span",
                    "excerpt_hash": "phase8hash",
                    "ts": "2026-02-10T00:00:00Z",
                }
            ],
            "schema_version": self.schema_version,
        }


class _EngineWithModelUsage:
    engine_type = "rca"
    output_contract_name = "RCAReport"
    engine_version = "phase8-runner"
    model_provider = "openai"
    model_name = "gpt-5-mini"
    prompt_template_hash = "phase8-hash"
    temperature = 0.0

    def build_input_ref(self, request: str) -> InputRef:  # noqa: ARG002
        return InputRef(project_name="phase8")

    def run(self, request: str) -> _Output:  # noqa: ARG002
        return _Output()

    def get_runtime_signals(self) -> dict[str, object]:
        return {
            "tokens_in": 144,
            "tokens_out": 39,
            "cost_usd": 0.1234,
            "iterations": 1,
            "depth_reached": 0,
            "tool_calls": 0,
        }


def test_run_engine_persists_model_provider_and_cost_usage(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    _, run_record = run_engine(
        engine=_EngineWithModelUsage(),
        request="demo",
        run_id="run-phase8-usage",
        artifacts_root=artifacts_root,
    )

    assert run_record.runtime_ref.model_provider == "openai"
    assert run_record.runtime_ref.usage.tokens_in == 144
    assert run_record.runtime_ref.usage.tokens_out == 39
    assert run_record.runtime_ref.usage.cost_usd == 0.1234

    payload = json.loads((artifacts_root / "run-phase8-usage" / "run_record.json").read_text())
    assert payload["runtime_ref"]["usage"]["cost_usd"] == 0.1234
