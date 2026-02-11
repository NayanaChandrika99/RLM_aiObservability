# ABOUTME: Validates Phase 10 policy compliance engine wiring to shared REPL runtime loops.
# ABOUTME: Ensures per-control non-trivial REPL runs emit llm_subcalls and trajectory metadata.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.compliance.engine import PolicyComplianceEngine, PolicyComplianceRequest
from investigator.runtime.llm_client import StructuredGenerationResult, StructuredGenerationUsage
from investigator.runtime.contracts import RuntimeBudget
from investigator.runtime.runner import run_engine


class _InspectionAPI:
    def __init__(self, controls: list[dict[str, Any]]) -> None:
        self._controls = controls

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
        return [{"role": "assistant", "content": f"message for {span_id}"}]

    def get_tool_io(self, span_id: str) -> dict[str, Any] | None:
        return {
            "trace_id": "trace-cmp-repl",
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
                "trace_id": "trace-cmp-repl",
                "span_id": span_id,
                "name": "agent.run",
                "span_kind": "AGENT",
                "status_code": "ERROR",
                "status_message": "forced timeout",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:01Z",
                "latency_ms": 12.0,
            },
            "attributes": {"control_hint": "strict"},
            "events": [],
        }


class _FakeModelClient:
    model_provider = "openai"

    def __init__(self, *, step_outputs: list[dict[str, Any]], subquery_outputs: list[str]) -> None:
        self._step_outputs = list(step_outputs)
        self._subquery_outputs = list(subquery_outputs)
        self.calls = 0

    def generate_structured(self, request):  # noqa: ANN001, ANN201
        self.calls += 1
        schema_name = str(getattr(request, "response_schema_name", ""))
        if schema_name == "repl_runtime_step_v1":
            payload = self._step_outputs.pop(0)
            usage = StructuredGenerationUsage(tokens_in=110, tokens_out=21, cost_usd=0.04)
            return StructuredGenerationResult(
                output=payload,
                raw_text=json.dumps(payload, sort_keys=True),
                usage=usage,
            )
        if schema_name == "repl_runtime_subquery_v1":
            answer = self._subquery_outputs.pop(0)
            payload = {"answer": answer}
            usage = StructuredGenerationUsage(tokens_in=50, tokens_out=13, cost_usd=0.015)
            return StructuredGenerationResult(
                output=payload,
                raw_text=json.dumps(payload, sort_keys=True),
                usage=usage,
            )
        raise AssertionError(f"Unexpected schema name: {schema_name}")


def test_policy_compliance_repl_runtime_emits_llm_subcalls_and_repl_trajectory(tmp_path: Path) -> None:
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
        step_outputs=[
            {
                "reasoning": "Evaluate control with one semantic subquery then finalize.",
                "code": (
                    "required_payload = call_tool("
                    "'required_evidence', control_id=control['control_id'], controls_version=controls_version)\n"
                    "del required_payload\n"
                    "verdict_note = llm_query('Given the gathered evidence, return pass/fail for this control.')\n"
                    "SUBMIT("
                    "pass_fail='fail',"
                    "confidence=0.83,"
                    "remediation='Address hard failures and rerun.',"
                    "rationale=verdict_note,"
                    "covered_requirements=[],"
                    "missing_evidence=[],"
                    "evidence_refs=[default_evidence],"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=["fail"],
    )
    engine = PolicyComplianceEngine(
        inspection_api=_InspectionAPI(controls=controls),
        model_client=model_client,
        use_llm_judgment=True,
        use_repl_runtime=True,
    )

    report, run_record = run_engine(
        engine=engine,
        request=PolicyComplianceRequest(
            trace_id="trace-cmp-repl",
            project_name="phase10",
            controls_version="controls-v1",
        ),
        run_id="run-phase10-compliance-repl",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert report.controls_evaluated[0].pass_fail == "fail"
    assert run_record.runtime_ref.usage.llm_subcalls > 0
    assert run_record.runtime_ref.repl_trajectory
    assert run_record.runtime_ref.usage.tokens_in > 0
    assert run_record.runtime_ref.usage.cost_usd > 0.0
    first_step = run_record.runtime_ref.repl_trajectory[0]
    assert first_step["subquery_trace"]
    assert "Given the gathered evidence" in first_step["subquery_trace"][0]["prompt"]


def test_policy_compliance_repl_applies_deterministic_fail_floor(tmp_path: Path) -> None:
    class _ViolationInspectionAPI(_InspectionAPI):
        def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
            return [
                {
                    "trace_id": trace_id,
                    "span_id": "root",
                    "parent_id": None,
                    "name": "tool.parse",
                    "span_kind": "TOOL",
                    "status_code": "ERROR",
                    "status_message": "forced tool timeout",
                    "start_time": "2026-02-10T00:00:00Z",
                    "end_time": "2026-02-10T00:00:01Z",
                    "latency_ms": 22.0,
                }
            ]

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
        step_outputs=[
            {
                "reasoning": "Submit a permissive verdict after one subquery.",
                "code": (
                    "note = llm_query('return PASS or FAIL')\n"
                    "SUBMIT("
                    "pass_fail='pass',"
                    "confidence=0.91,"
                    "remediation='No action.',"
                    "rationale=note,"
                    "covered_requirements=[],"
                    "missing_evidence=[],"
                    "evidence_refs=[default_evidence],"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=["PASS"],
    )
    engine = PolicyComplianceEngine(
        inspection_api=_ViolationInspectionAPI(controls=controls),
        model_client=model_client,
        use_llm_judgment=True,
        use_repl_runtime=True,
    )

    report, _run_record = run_engine(
        engine=engine,
        request=PolicyComplianceRequest(
            trace_id="trace-cmp-repl-fail-floor",
            project_name="phase10",
            controls_version="controls-v1",
        ),
        run_id="run-phase10-compliance-repl-fail-floor",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert report.controls_evaluated[0].pass_fail == "fail"


def test_policy_compliance_repl_tightens_per_control_budget() -> None:
    source_budget = RuntimeBudget(
        max_iterations=9,
        max_depth=3,
        max_tool_calls=40,
        max_subcalls=12,
        max_tokens_total=70000,
        max_cost_usd=0.4,
        max_wall_time_sec=180,
    )
    tightened = PolicyComplianceEngine._tighten_repl_control_budget(source_budget)

    assert tightened.max_iterations == 1
    assert tightened.max_depth == 3
    assert tightened.max_tool_calls == 8
    assert tightened.max_subcalls == 2
    assert tightened.max_tokens_total == 20000
    assert tightened.max_cost_usd == 0.12
    assert tightened.max_wall_time_sec == 45
