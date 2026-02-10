# ABOUTME: Validates Phase 8B recursive runtime loop behavior under budget and sandbox constraints.
# ABOUTME: Ensures terminated-budget states propagate to partial run records in shared runtime runner.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.runtime.contracts import InputRef, RuntimeBudget
from investigator.runtime.recursive_loop import RecursiveLoop
from investigator.runtime.runner import run_engine
from investigator.runtime.sandbox import SandboxGuard
from investigator.runtime.tool_registry import ToolRegistry


class _InspectionAPI:
    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [{"trace_id": trace_id, "span_id": "root"}]


def test_recursive_loop_terminates_budget_when_subcall_depth_exceeded() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    result = loop.run(
        actions=[
            {
                "type": "delegate_subcall",
                "objective": "inspect child",
                "actions": [
                    {
                        "type": "finalize",
                        "output": {"summary": "child summary", "evidence_refs": [], "gaps": []},
                    }
                ],
            }
        ],
        budget=RuntimeBudget(max_depth=0),
    )

    assert result.status == "terminated_budget"
    assert "max_depth" in (result.budget_reason or "")
    assert result.state_trajectory[-1] == "terminated_budget"


def test_recursive_loop_fails_fast_on_sandbox_violation() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools={"list_spans"})
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    result = loop.run(
        actions=[
            {
                "type": "tool_call",
                "tool_name": "forbidden_tool",
                "args": {"trace_id": "trace-1"},
            }
        ],
        budget=RuntimeBudget(),
    )

    assert result.status == "failed"
    assert result.error_code == "SANDBOX_VIOLATION"


class _Output:
    schema_version = "1.0.0"

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": "trace-budget",
            "primary_label": "instruction_failure",
            "summary": "budget terminated during recursion",
            "confidence": 0.4,
            "evidence_refs": [
                {
                    "trace_id": "trace-budget",
                    "span_id": "root",
                    "kind": "SPAN",
                    "ref": "root",
                    "excerpt_hash": "budget",
                    "ts": None,
                }
            ],
            "schema_version": self.schema_version,
        }


class _TerminatedBudgetEngine:
    engine_type = "rca"
    output_contract_name = "RCAReport"
    engine_version = "phase8b-recursive"
    model_provider = "openai"
    model_name = "gpt-5-mini"
    prompt_template_hash = "phase8b-recursive"
    temperature = 0.0

    def build_input_ref(self, request: str) -> InputRef:  # noqa: ARG002
        return InputRef(project_name="phase8b")

    def run(self, request: str) -> _Output:  # noqa: ARG002
        return _Output()

    def get_runtime_signals(self) -> dict[str, object]:
        return {
            "iterations": 2,
            "depth_reached": 2,
            "tool_calls": 1,
            "subcalls": 1,
            "runtime_state": "terminated_budget",
            "budget_reason": "max_depth reached in delegate_subcall",
            "state_trajectory": ["initialized", "running", "terminated_budget"],
        }


def test_runner_maps_terminated_budget_runtime_state_to_partial(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"
    _, run_record = run_engine(
        engine=_TerminatedBudgetEngine(),
        request="demo",
        run_id="run-phase8b-terminated-budget",
        artifacts_root=artifacts_root,
    )

    assert run_record.status == "partial"
    assert run_record.error is not None
    assert run_record.error.code == "RECURSION_LIMIT_REACHED"
    assert "max_depth" in run_record.error.message

    persisted = json.loads((artifacts_root / "run-phase8b-terminated-budget" / "run_record.json").read_text())
    assert persisted["status"] == "partial"
    assert persisted["error"]["code"] == "RECURSION_LIMIT_REACHED"
