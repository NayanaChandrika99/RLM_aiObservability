# ABOUTME: Validates Phase 8D compliance engine migration to shared LLM per-control judgment flow.
# ABOUTME: Ensures missing-evidence precedence and deterministic fallback semantics are enforced.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.compliance.engine import PolicyComplianceEngine, PolicyComplianceRequest
from investigator.runtime.llm_client import StructuredGenerationResult, StructuredGenerationUsage
from investigator.runtime.runner import run_engine


class _FakeComplianceInspectionAPI:
    def __init__(
        self,
        *,
        controls: list[dict[str, Any]],
        messages_by_span: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._controls = controls
        self._messages_by_span = messages_by_span or {}

    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [
            {
                "trace_id": trace_id,
                "span_id": "root",
                "parent_id": None,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "ERROR",
                "status_message": "forced tool timeout",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 10.0,
            }
        ]

    def list_controls(
        self,
        controls_version: str,
        app_type: str | None = None,
        tools_used: list[str] | None = None,
        data_domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        del app_type, tools_used, data_domains
        return [
            control
            for control in self._controls
            if str(control.get("controls_version") or "") == controls_version
        ]

    def get_control(self, control_id: str, controls_version: str) -> dict[str, Any]:
        for control in self._controls:
            if (
                str(control.get("control_id") or "") == control_id
                and str(control.get("controls_version") or "") == controls_version
            ):
                return control
        raise KeyError(control_id)

    def required_evidence(self, control_id: str, controls_version: str) -> list[str]:
        control = self.get_control(control_id, controls_version)
        return [str(item) for item in (control.get("required_evidence") or []) if str(item)]

    def get_tool_io(self, span_id: str) -> dict[str, Any] | None:
        del span_id
        return None

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        del span_id
        return []

    def get_messages(self, span_id: str) -> list[dict[str, Any]]:
        return self._messages_by_span.get(span_id, [])

    def get_span(self, span_id: str) -> dict[str, Any]:
        return {
            "summary": {
                "trace_id": "trace-cmp-llm",
                "span_id": span_id,
                "parent_id": None,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "ERROR",
                "status_message": "forced tool timeout",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 10.0,
            },
            "attributes": {
                "phase1": json.dumps({"step": "tool.call", "output": {"format": "unexpected"}}),
                "http.status_code": 503,
            },
            "events": [],
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
            usage=StructuredGenerationUsage(tokens_in=50, tokens_out=12, cost_usd=0.02),
        )


def test_policy_compliance_engine_llm_path_updates_control_verdict_and_usage(tmp_path: Path) -> None:
    controls = [
        {
            "control_id": "control.execution.hard_failures",
            "controls_version": "controls-v1",
            "severity": "high",
            "required_evidence": ["required_error_span"],
            "remediation_template": "Address hard failures.",
        }
    ]
    model_client = _FakeModelClient(
        outputs=[
            {
                "pass_fail": "fail",
                "confidence": 0.88,
                "remediation": "Address hard failures and rerun.",
                "rationale": "Timeout evidence indicates policy violation.",
                "gaps": [],
            }
        ]
    )
    engine = PolicyComplianceEngine(
        inspection_api=_FakeComplianceInspectionAPI(controls=controls),
        model_client=model_client,
        use_llm_judgment=True,
    )

    report, run_record = run_engine(
        engine=engine,
        request=PolicyComplianceRequest(
            trace_id="trace-cmp-llm",
            project_name="phase8",
            controls_version="controls-v1",
        ),
        run_id="run-phase8-compliance-llm",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    finding = report.controls_evaluated[0]
    assert finding.pass_fail == "fail"
    assert finding.confidence == 0.88
    assert model_client.calls == 1
    assert run_record.runtime_ref.usage.tokens_in == 50
    assert run_record.runtime_ref.usage.tokens_out == 12
    assert run_record.runtime_ref.usage.cost_usd == 0.02
    assert run_record.runtime_ref.model_provider == "openai"


def test_policy_compliance_engine_missing_evidence_precedence_skips_llm() -> None:
    controls = [
        {
            "control_id": "control.instruction.format.review",
            "controls_version": "controls-v1",
            "severity": "medium",
            "required_evidence": ["required_messages"],
            "remediation_template": "Collect required message evidence.",
        }
    ]
    model_client = _FakeModelClient(
        outputs=[
            {
                "pass_fail": "pass",
                "confidence": 0.9,
                "remediation": "No action needed.",
                "rationale": "Evidence appears complete.",
                "gaps": [],
            }
        ]
    )
    engine = PolicyComplianceEngine(
        inspection_api=_FakeComplianceInspectionAPI(controls=controls, messages_by_span={}),
        model_client=model_client,
        use_llm_judgment=True,
    )
    report = engine.run(
        PolicyComplianceRequest(
            trace_id="trace-cmp-llm",
            project_name="phase8",
            controls_version="controls-v1",
        )
    )

    finding = report.controls_evaluated[0]
    assert finding.pass_fail == "insufficient_evidence"
    assert finding.missing_evidence == ["required_messages"]
    assert model_client.calls == 0


def test_policy_compliance_engine_falls_back_to_deterministic_on_llm_error() -> None:
    controls = [
        {
            "control_id": "control.execution.hard_failures",
            "controls_version": "controls-v1",
            "severity": "high",
            "required_evidence": [],
            "violation_patterns": ["forced tool timeout"],
            "remediation_template": "Address hard failures.",
        }
    ]
    model_client = _FakeModelClient(
        outputs=[
            {"confidence": 0.5},
            {"still": "invalid"},
        ]
    )
    engine = PolicyComplianceEngine(
        inspection_api=_FakeComplianceInspectionAPI(controls=controls),
        model_client=model_client,
        use_llm_judgment=True,
        fallback_on_llm_error=True,
    )

    report = engine.run(
        PolicyComplianceRequest(
            trace_id="trace-cmp-llm",
            project_name="phase8",
            controls_version="controls-v1",
        )
    )

    finding = report.controls_evaluated[0]
    signals = engine.get_runtime_signals()
    assert model_client.calls == 2
    assert finding.pass_fail == "fail"
    assert signals["compliance_judgment_mode"] == "deterministic_fallback"
