# ABOUTME: Validates Phase 9C compliance recursive runtime wiring and per-control runtime metadata propagation.
# ABOUTME: Ensures recursive per-control execution can emit subcalls and map budget termination to partial runs.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.compliance.engine import PolicyComplianceEngine, PolicyComplianceRequest
from investigator.runtime.contracts import RuntimeBudget
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
                "status_message": "forced timeout",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 12.0,
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

    def get_messages(self, span_id: str) -> list[dict[str, Any]]:
        return self._messages_by_span.get(span_id, [])

    def get_tool_io(self, span_id: str) -> dict[str, Any] | None:
        return {
            "trace_id": "trace-cmp-recursive",
            "span_id": span_id,
            "artifact_id": f"tool:{span_id}",
            "tool_name": "search",
            "status_code": "ERROR",
        }

    def get_retrieval_chunks(self, span_id: str) -> list[dict[str, Any]]:
        del span_id
        return []

    def get_span(self, span_id: str) -> dict[str, Any]:
        return {
            "summary": {
                "trace_id": "trace-cmp-recursive",
                "span_id": span_id,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "ERROR",
                "status_message": "forced timeout",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 12.0,
            },
            "attributes": {"control_hint": "needs strict handling"},
            "events": [],
        }


class _FakeModelClient:
    model_provider = "openai"

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
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
            usage=StructuredGenerationUsage(tokens_in=70, tokens_out=15, cost_usd=0.03),
        )


def test_policy_compliance_recursive_runtime_emits_trajectory_and_subcalls(tmp_path: Path) -> None:
    controls = [
        {
            "control_id": "control.execution.hard_failures",
            "controls_version": "controls-v1",
            "severity": "high",
            "required_evidence": [],
            "remediation_template": "Address hard failures.",
        }
    ]
    model_client = _FakeModelClient(
        outputs=[
            {
                "action": {
                    "type": "delegate_subcall",
                    "objective": "gather control evidence",
                    "actions": [
                        {"type": "synthesize", "output": {"evidence_refs": [], "gaps": []}},
                        {"type": "finalize", "output": {"summary": "child done"}},
                    ],
                }
            },
            {
                "action": {
                    "type": "synthesize",
                    "output": {
                        "pass_fail": "fail",
                        "confidence": 0.83,
                        "remediation": "Address hard failures and rerun.",
                        "missing_evidence": [],
                        "gaps": [],
                    },
                }
            },
            {"action": {"type": "finalize", "output": {"summary": "control evaluated"}}},
        ]
    )
    engine = PolicyComplianceEngine(
        inspection_api=_FakeComplianceInspectionAPI(controls=controls),
        model_client=model_client,
        use_llm_judgment=True,
        use_recursive_runtime=True,
    )

    report, run_record = run_engine(
        engine=engine,
        request=PolicyComplianceRequest(
            trace_id="trace-cmp-recursive",
            project_name="phase9",
            controls_version="controls-v1",
        ),
        run_id="run-phase9-compliance-recursive",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert report.controls_evaluated[0].pass_fail == "fail"
    assert model_client.calls == 3
    assert run_record.runtime_ref.state_trajectory
    assert "delegating" in run_record.runtime_ref.state_trajectory
    assert run_record.runtime_ref.subcall_metadata
    assert run_record.runtime_ref.usage.tokens_in > 0
    assert run_record.runtime_ref.usage.cost_usd > 0.0


def test_policy_compliance_recursive_runtime_budget_termination_maps_partial(tmp_path: Path) -> None:
    controls = [
        {
            "control_id": "control.execution.hard_failures",
            "controls_version": "controls-v1",
            "severity": "high",
            "required_evidence": [],
            "remediation_template": "Address hard failures.",
        }
    ]
    model_client = _FakeModelClient(
        outputs=[
            {"action": {"type": "tool_call", "tool_name": "list_spans", "args": {"trace_id": "trace-cmp-recursive"}}},
            {"action": {"type": "finalize", "output": {"summary": "late finalize"}}},
        ]
    )
    engine = PolicyComplianceEngine(
        inspection_api=_FakeComplianceInspectionAPI(controls=controls),
        model_client=model_client,
        use_llm_judgment=True,
        use_recursive_runtime=True,
        recursive_budget=RuntimeBudget(max_iterations=1),
        fallback_on_llm_error=True,
    )

    _, run_record = run_engine(
        engine=engine,
        request=PolicyComplianceRequest(
            trace_id="trace-cmp-recursive",
            project_name="phase9",
            controls_version="controls-v1",
        ),
        run_id="run-phase9-compliance-recursive-budget",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert run_record.status == "partial"
    assert run_record.error is not None
    assert run_record.error.code == "RECURSION_LIMIT_REACHED"
    assert "max_iterations" in run_record.error.message
