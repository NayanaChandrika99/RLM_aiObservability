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


class _InspectionAPIWithConfig:
    def get_config_diff(self, base_snapshot_id: str, target_snapshot_id: str) -> dict[str, Any]:
        return {
            "base_snapshot_id": base_snapshot_id,
            "target_snapshot_id": target_snapshot_id,
            "artifact_id": "configdiff:test",
        }


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


def test_tool_registry_drops_unknown_kwargs_before_tool_invocation() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPIWithConfig())
    payload = registry.call(
        "get_config_diff",
        {
            "base_snapshot_id": "snap-a",
            "target_snapshot_id": "snap-b",
            "project_name": "phase9-live",
        },
    )

    assert payload["normalized_args"] == {
        "base_snapshot_id": "snap-a",
        "target_snapshot_id": "snap-b",
    }
    assert payload["result"]["artifact_id"] == "configdiff:test"


def test_recursive_loop_records_tool_argument_mismatch_and_continues() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPIWithConfig())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    result = loop.run(
        actions=[
            {
                "type": "tool_call",
                "tool_name": "get_config_diff",
                "args": {"project_name": "phase9-live"},
            },
            {"type": "finalize", "output": {"summary": "continued after tool error"}},
        ],
        budget=RuntimeBudget(),
    )

    assert result.status == "completed"
    assert isinstance(result.output, dict)
    tool_calls = result.output.get("tool_calls")
    assert isinstance(tool_calls, list) and tool_calls
    first_call = tool_calls[0]
    assert isinstance(first_call, dict)
    call_result = first_call.get("result")
    assert isinstance(call_result, dict)
    assert call_result.get("error_code") == "SANDBOX_VIOLATION"


def test_recursive_loop_fails_on_fatal_tool_argument_mismatch() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPIWithConfig())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    result = loop.run(
        actions=[
            {
                "type": "tool_call",
                "tool_name": "get_config_diff",
                "args": {"project_name": "phase9-live"},
                "fatal": True,
            }
        ],
        budget=RuntimeBudget(),
    )

    assert result.status == "failed"
    assert result.error_code == "SANDBOX_VIOLATION"


def test_recursive_loop_collects_structured_hypothesis_results_from_subcalls() -> None:
    registry = ToolRegistry(inspection_api=_InspectionAPI())
    guard = SandboxGuard(allowed_tools=registry.allowed_tools)
    loop = RecursiveLoop(tool_registry=registry, sandbox_guard=guard)

    result = loop.run(
        actions=[
            {
                "type": "delegate_subcall",
                "objective": "evaluate tool hypothesis",
                "actions": [
                    {
                        "type": "finalize",
                        "output": {
                            "label": "tool_failure",
                            "confidence": 0.81,
                            "supporting_facts": ["tool timeout error seen in root span"],
                            "evidence_refs": [
                                {
                                    "trace_id": "trace-1",
                                    "span_id": "root",
                                    "kind": "TOOL_IO",
                                    "ref": "tool:root",
                                    "excerpt_hash": "tool-hash",
                                    "ts": "2026-02-10T00:00:00Z",
                                }
                            ],
                            "gaps": [],
                        },
                    }
                ],
            },
            {
                "type": "delegate_subcall",
                "objective": "evaluate retrieval hypothesis",
                "actions": [
                    {
                        "type": "finalize",
                        "output": {
                            "label": "retrieval_failure",
                            "confidence": 0.33,
                            "supporting_facts": ["retrieval evidence weaker than tool failure"],
                            "evidence_refs": [
                                {
                                    "trace_id": "trace-1",
                                    "span_id": "root",
                                    "kind": "RETRIEVAL_CHUNK",
                                    "ref": "retrieval:root:0:doc",
                                    "excerpt_hash": "retrieval-hash",
                                    "ts": "2026-02-10T00:00:00Z",
                                }
                            ],
                            "gaps": [],
                        },
                    }
                ],
            },
            {"type": "finalize", "output": {"summary": "collected hypothesis outputs"}},
        ],
        budget=RuntimeBudget(max_depth=2, max_iterations=30),
    )

    assert result.status == "completed"
    assert isinstance(result.output, dict)
    hypothesis_results = result.output.get("hypothesis_results")
    assert isinstance(hypothesis_results, list)
    assert len(hypothesis_results) == 2
    assert hypothesis_results[0]["label"] == "tool_failure"
    assert hypothesis_results[1]["label"] == "retrieval_failure"


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
