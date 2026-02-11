# ABOUTME: Validates the investigator runtime scaffold and three RLM stub engines compile and run.
# ABOUTME: Guards contract-aligned defaults and minimal output shapes before deeper implementation.

from __future__ import annotations

from investigator.compliance.engine import PolicyComplianceEngine, PolicyComplianceRequest
from investigator.incident.engine import IncidentDossierEngine, IncidentDossierRequest
from investigator.rca.engine import TraceRCAEngine, TraceRCARequest
from investigator.runtime.contracts import RuntimeBudget
from investigator.runtime.runner import run_engine


def test_runtime_budget_defaults_match_contract() -> None:
    budget = RuntimeBudget()
    assert budget.max_depth == 2
    assert budget.max_iterations == 40
    assert budget.max_tool_calls == 200
    assert budget.max_subcalls == 80
    assert budget.max_tokens_total == 300000
    assert budget.max_wall_time_sec == 180


def test_trace_rca_stub_engine_runs() -> None:
    engine = TraceRCAEngine()
    request = TraceRCARequest(trace_id="trace-1", project_name="phoenix-rlm")
    report, run_record = run_engine(engine=engine, request=request)
    payload = report.to_dict()

    assert payload["trace_id"] == "trace-1"
    assert payload["primary_label"] == "instruction_failure"
    assert payload["evidence_refs"]
    assert run_record.run_type == "rca"
    assert run_record.status == "succeeded"


def test_policy_compliance_stub_engine_runs() -> None:
    engine = PolicyComplianceEngine()
    request = PolicyComplianceRequest(
        trace_id="trace-2",
        project_name="phoenix-rlm",
        controls_version="controls-v1",
    )
    report, run_record = run_engine(engine=engine, request=request)
    payload = report.to_dict()

    assert payload["trace_id"] == "trace-2"
    assert payload["controls_version"] == "controls-v1"
    assert payload["controls_evaluated"]
    assert run_record.run_type == "policy_compliance"
    assert run_record.status == "succeeded"


def test_incident_dossier_stub_engine_runs() -> None:
    engine = IncidentDossierEngine()
    request = IncidentDossierRequest(
        project_name="phoenix-rlm",
        time_window_start="2026-02-10T00:00:00Z",
        time_window_end="2026-02-10T01:00:00Z",
        filter_expr="status_code == 'ERROR'",
        trace_ids_override=["trace-3"],
    )
    report, run_record = run_engine(engine=engine, request=request)
    payload = report.to_dict()

    assert payload["incident_summary"]
    assert payload["representative_traces"]
    assert payload["timeline"]
    assert run_record.run_type == "incident_dossier"
    assert run_record.status == "succeeded"
