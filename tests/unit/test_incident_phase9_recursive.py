# ABOUTME: Validates Phase 9D incident recursive runtime wiring and hierarchical runtime metadata propagation.
# ABOUTME: Ensures recursive incident synthesis emits trajectory/subcalls and maps budget termination to partial runs.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.incident.engine import IncidentDossierEngine, IncidentDossierRequest
from investigator.runtime.contracts import RuntimeBudget
from investigator.runtime.llm_client import StructuredGenerationResult, StructuredGenerationUsage
from investigator.runtime.runner import run_engine


class _FakeIncidentInspectionAPI:
    def list_traces(  # noqa: ANN201
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        filter_expr: str | None = None,
    ):
        del project_name, start_time, end_time, filter_expr
        return [
            {
                "trace_id": "trace-inc-r1",
                "start_time": "2026-02-10T00:00:00Z",
                "end_time": "2026-02-10T00:00:02Z",
                "latency_ms": 210.0,
            },
            {
                "trace_id": "trace-inc-r2",
                "start_time": "2026-02-10T00:03:00Z",
                "end_time": "2026-02-10T00:03:01Z",
                "latency_ms": 170.0,
            },
        ]

    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        if trace_id == "trace-inc-r1":
            return [
                {
                    "trace_id": trace_id,
                    "span_id": "root-r1",
                    "parent_id": None,
                    "name": "svc.a.agent",
                    "span_kind": "AGENT",
                    "status_code": "ERROR",
                    "status_message": "timeout",
                    "start_time": "2026-02-10T00:00:00Z",
                    "end_time": "2026-02-10T00:00:02Z",
                    "latency_ms": 210.0,
                }
            ]
        return [
            {
                "trace_id": trace_id,
                "span_id": "root-r2",
                "parent_id": None,
                "name": "svc.b.agent",
                "span_kind": "AGENT",
                "status_code": "OK",
                "status_message": "",
                "start_time": "2026-02-10T00:03:00Z",
                "end_time": "2026-02-10T00:03:01Z",
                "latency_ms": 170.0,
            }
        ]

    def get_children(self, span_id: str) -> list[dict[str, Any]]:
        return [{"span_id": f"{span_id}:child"}]

    def list_config_snapshots(  # noqa: ANN201
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        tag: str | None = None,
    ):
        del project_name, start_time, end_time, tag
        return [
            {"snapshot_id": "snap-a", "created_at": "2026-02-10T00:00:00Z"},
            {"snapshot_id": "snap-b", "created_at": "2026-02-10T00:20:00Z"},
        ]

    def get_config_diff(self, base_snapshot_id: str, target_snapshot_id: str):  # noqa: ANN201
        del base_snapshot_id, target_snapshot_id
        return {
            "diff_ref": "configdiff:phase9",
            "artifact_id": "configdiff:phase9",
            "git_commit_target": "abc999",
            "paths": ["prompts/incident.txt"],
            "summary": "1 changed file(s)",
        }


class _FakeModelClient:
    model_provider = "openai"

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self._outputs = list(outputs)
        self.calls = 0
        self.requests: list[Any] = []

    def generate_structured(self, request):  # noqa: ANN001, ANN201
        self.requests.append(request)
        if not self._outputs:
            raise AssertionError("No fake outputs configured.")
        self.calls += 1
        payload = self._outputs.pop(0)
        return StructuredGenerationResult(
            output=payload,
            raw_text=json.dumps(payload, sort_keys=True),
            usage=StructuredGenerationUsage(tokens_in=55, tokens_out=18, cost_usd=0.025),
        )


def _planner_context_from_request(request: Any) -> dict[str, Any]:
    user_prompt = str(getattr(request, "user_prompt", ""))
    prefix = "Runtime planner context JSON:\n"
    suffix = "\n\nReturn only the next typed action object wrapped in the required schema."
    if not user_prompt.startswith(prefix) or suffix not in user_prompt:
        raise AssertionError("Planner request user_prompt format is unexpected.")
    payload = user_prompt[len(prefix) : user_prompt.index(suffix)]
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise AssertionError("Planner context payload must be an object.")
    return parsed


def test_incident_recursive_runtime_emits_trajectory_subcalls_and_usage(tmp_path: Path) -> None:
    model_client = _FakeModelClient(
        outputs=[
            {
                "action": {
                    "type": "delegate_subcall",
                    "objective": "trace drilldown trace-inc-r1",
                    "use_planner": True,
                    "context": {"trace_id": "trace-inc-r1"},
                }
            },
            {"action": {"type": "synthesize", "output": {"evidence_refs": [], "gaps": []}}},
            {"action": {"type": "finalize", "output": {"summary": "trace drilldown done"}}},
            {
                "action": {
                    "type": "synthesize",
                    "output": {
                        "hypotheses": [
                            {
                                "statement": "Timeout signatures cluster on trace-inc-r1",
                                "confidence": 0.82,
                            }
                        ],
                        "recommended_actions": [
                            {
                                "priority": "P1",
                                "type": "follow_up_fix",
                                "action": "Add upstream timeout backoff.",
                            }
                        ],
                        "gaps": [],
                    },
                }
            },
            {"action": {"type": "finalize", "output": {"incident_summary": "trace-inc-r1 drilldown complete"}}},
            {
                "action": {
                    "type": "synthesize",
                    "output": {
                        "hypotheses": [
                            {
                                "statement": "Latency spillover appears on trace-inc-r2",
                                "confidence": 0.66,
                            }
                        ],
                        "recommended_actions": [
                            {
                                "priority": "P2",
                                "type": "mitigation",
                                "action": "Throttle traffic during mitigation window.",
                            }
                        ],
                        "gaps": [],
                    },
                }
            },
            {"action": {"type": "finalize", "output": {"incident_summary": "trace-inc-r2 drilldown complete"}}},
            {
                "action": {
                    "type": "finalize",
                    "output": {
                        "incident_summary": "Cross-trace recursive synthesis complete.",
                        "confidence": 0.79,
                    },
                }
            },
        ]
    )
    engine = IncidentDossierEngine(
        inspection_api=_FakeIncidentInspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
        use_recursive_runtime=True,
        max_representatives=2,
        error_quota=1,
        latency_quota=1,
        cluster_quota=0,
    )

    report, run_record = run_engine(
        engine=engine,
        request=IncidentDossierRequest(
            project_name="phase9",
            time_window_start="2026-02-10T00:00:00Z",
            time_window_end="2026-02-10T01:00:00Z",
        ),
        run_id="run-phase9-incident-recursive",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert report.hypotheses
    assert report.recommended_actions
    assert "recursive synthesis" in report.incident_summary.lower()
    assert model_client.calls == 8
    assert run_record.runtime_ref.state_trajectory
    assert "delegating" in run_record.runtime_ref.state_trajectory
    assert run_record.runtime_ref.subcall_metadata
    assert bool(run_record.runtime_ref.subcall_metadata[0].get("planner_driven")) is True
    planner_contexts = [_planner_context_from_request(request) for request in model_client.requests]
    per_trace_context = next(
        (
            context
            for context in planner_contexts
            if str(context.get("incident_scope") or "") == "per_trace_drilldown"
            and isinstance(context.get("delegation_policy"), dict)
        ),
        None,
    )
    assert isinstance(per_trace_context, dict)
    delegation_policy = next(
        (
            context.get("delegation_policy")
            for context in [per_trace_context]
            if isinstance(context, dict) and isinstance(context.get("delegation_policy"), dict)
        ),
        None,
    )
    assert isinstance(delegation_policy, dict)
    assert bool(delegation_policy.get("prefer_planner_driven_subcalls")) is True
    example_actions = delegation_policy.get("example_actions")
    assert isinstance(example_actions, list) and example_actions
    assert bool(example_actions[0].get("use_planner")) is True
    assert run_record.runtime_ref.usage.tokens_in > 0
    assert run_record.runtime_ref.usage.cost_usd > 0.0


def test_incident_recursive_runtime_budget_termination_maps_partial(tmp_path: Path) -> None:
    model_client = _FakeModelClient(
        outputs=[
            {"action": {"type": "tool_call", "tool_name": "list_spans", "args": {"trace_id": "trace-inc-r1"}}},
            {"action": {"type": "finalize", "output": {"incident_summary": "late finalize"}}},
        ]
    )
    engine = IncidentDossierEngine(
        inspection_api=_FakeIncidentInspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
        use_recursive_runtime=True,
        recursive_budget=RuntimeBudget(max_iterations=1),
        fallback_on_llm_error=True,
        max_representatives=1,
        error_quota=1,
        latency_quota=0,
        cluster_quota=0,
    )

    _, run_record = run_engine(
        engine=engine,
        request=IncidentDossierRequest(
            project_name="phase9",
            time_window_start="2026-02-10T00:00:00Z",
            time_window_end="2026-02-10T01:00:00Z",
        ),
        run_id="run-phase9-incident-recursive-budget",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert run_record.status == "partial"
    assert run_record.error is not None
    assert run_record.error.code == "RECURSION_LIMIT_REACHED"
    assert "max_iterations" in run_record.error.message


def test_incident_recursive_runtime_pools_budget_across_trace_and_cross_scopes(
    tmp_path: Path,
) -> None:
    model_client = _FakeModelClient(
        outputs=[
            {
                "action": {
                    "type": "synthesize",
                    "output": {
                        "incident_summary": "trace-inc-r1 drilldown summary",
                        "hypotheses": [
                            {"statement": "r1 timeout cluster", "confidence": 0.8},
                        ],
                        "recommended_actions": [
                            {
                                "priority": "P1",
                                "type": "follow_up_fix",
                                "action": "Tune timeout for svc.a.",
                            }
                        ],
                        "gaps": [],
                    },
                }
            },
            {"action": {"type": "finalize", "output": {"incident_summary": "trace-inc-r1 complete"}}},
            {
                "action": {
                    "type": "synthesize",
                    "output": {
                        "incident_summary": "trace-inc-r2 drilldown summary",
                        "hypotheses": [
                            {"statement": "r2 latency spillover", "confidence": 0.66},
                        ],
                        "recommended_actions": [
                            {
                                "priority": "P2",
                                "type": "mitigation",
                                "action": "Rate limit background jobs.",
                            }
                        ],
                        "gaps": [],
                    },
                }
            },
            {"action": {"type": "finalize", "output": {"incident_summary": "trace-inc-r2 complete"}}},
            {
                "action": {
                    "type": "synthesize",
                    "output": {
                        "incident_summary": "cross-trace synthesis draft",
                        "hypotheses": [],
                        "recommended_actions": [],
                        "gaps": [],
                    },
                }
            },
            {
                "action": {
                    "type": "synthesize",
                    "output": {
                        "incident_summary": "cross-trace synthesis draft 2",
                        "hypotheses": [],
                        "recommended_actions": [],
                        "gaps": [],
                    },
                }
            },
            {
                "action": {
                    "type": "finalize",
                    "output": {"incident_summary": "cross-trace synthesis complete"},
                }
            },
        ]
    )
    engine = IncidentDossierEngine(
        inspection_api=_FakeIncidentInspectionAPI(),
        model_client=model_client,
        use_llm_judgment=True,
        use_recursive_runtime=True,
        recursive_budget=RuntimeBudget(max_iterations=6),
        fallback_on_llm_error=True,
        max_representatives=2,
        error_quota=1,
        latency_quota=1,
        cluster_quota=0,
    )

    _, run_record = run_engine(
        engine=engine,
        request=IncidentDossierRequest(
            project_name="phase9",
            time_window_start="2026-02-10T00:00:00Z",
            time_window_end="2026-02-10T01:00:00Z",
        ),
        run_id="run-phase91-incident-budget-pool",
        artifacts_root=tmp_path / "artifacts" / "investigator_runs",
    )

    assert model_client.calls == 6
    assert run_record.status == "partial"
    assert run_record.error is not None
    assert run_record.error.code == "RECURSION_LIMIT_REACHED"
    assert "max_iterations" in run_record.error.message
