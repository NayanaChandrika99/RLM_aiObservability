# ABOUTME: Exercises deterministic pair-precision recovery rules for weak full-GAIA TRAIL categories.
# ABOUTME: Validates location remapping and conservative category recovery behavior in trace post-processing.

from __future__ import annotations

from arcgentica.trail_agent import _boost_joint_recall, _recover_targeted_tp


def test_recover_targeted_tp_does_not_remap_goal_deviation_location() -> None:
    trace_payload = {
        "trace_id": "t1",
        "spans": [
            {"span_id": "goal_old", "span_name": "LiteLLMModel.__call__", "status_code": "Ok"},
            {"span_id": "goal_new", "span_name": "LiteLLMModel.__call__", "status_code": "Ok"},
        ],
    }
    findings = [
        {
            "category": "Goal Deviation",
            "location": "goal_old",
            "evidence": "goal drift",
            "description": "drift",
            "impact": "HIGH",
        },
        {
            "category": "Instruction Non-compliance",
            "location": "goal_new",
            "evidence": "instruction miss",
            "description": "format requirement violated",
            "impact": "MEDIUM",
        },
    ]

    recovered = _recover_targeted_tp(trace_payload, findings)

    goal_entries = [entry for entry in recovered if entry["category"] == "Goal Deviation"]
    assert len(goal_entries) == 1
    assert goal_entries[0]["location"] == "goal_old"


def test_recover_targeted_tp_adds_task_orchestration_from_service_and_retrieval_signal() -> None:
    trace_payload = {
        "trace_id": "t2",
        "spans": [
            {"span_id": "loc_r", "span_name": "LiteLLMModel.__call__", "status_code": "Ok"},
            {"span_id": "loc_s", "span_name": "VisitTool", "status_code": "Error"},
        ],
    }
    findings = [
        {
            "category": "Poor Information Retrieval",
            "location": "loc_r",
            "evidence": "retrieval failed",
            "description": "could not fetch required source",
            "impact": "HIGH",
        },
        {
            "category": "Service Errors",
            "location": "loc_s",
            "evidence": "500 upstream",
            "description": "service unavailable",
            "impact": "MEDIUM",
        },
    ]

    recovered = _recover_targeted_tp(trace_payload, findings)

    assert any(
        entry["category"] == "Task Orchestration" and entry["location"] == "loc_r"
        for entry in recovered
    )


def test_recover_targeted_tp_adds_resource_abuse_for_page_navigation_tool_definition_failure() -> None:
    trace_payload = {
        "trace_id": "t3",
        "spans": [
            {"span_id": "loc_page", "span_name": "PageDownTool", "status_code": "Error"}
        ],
    }
    findings = [
        {
            "category": "Tool Definition Issues",
            "location": "loc_page",
            "evidence": "unexpected keyword argument",
            "description": "tool signature mismatch",
            "impact": "MEDIUM",
        }
    ]

    recovered = _recover_targeted_tp(trace_payload, findings)

    assert any(
        entry["category"] == "Resource Abuse" and entry["location"] == "loc_page"
        for entry in recovered
    )


def test_recover_targeted_tp_replaces_freeform_ipi_with_conservative_signal() -> None:
    trace_payload = {
        "trace_id": "t4",
        "spans": [
            {"span_id": "ipi_old", "span_name": "LiteLLMModel.__call__", "status_code": "Ok"},
            {"span_id": "loc_rnf", "span_name": "TextInspectorTool", "status_code": "Error"},
            {"span_id": "loc_re", "span_name": "CodeAgent.run", "status_code": "Error"},
        ],
    }
    findings = [
        {
            "category": "Incorrect Problem Identification",
            "location": "ipi_old",
            "evidence": "generic claim",
            "description": "free-form classification",
            "impact": "MEDIUM",
        },
        {
            "category": "Resource Not Found",
            "location": "loc_rnf",
            "evidence": "file not found",
            "description": "missing input artifact",
            "impact": "MEDIUM",
        },
        {
            "category": "Resource Exhaustion",
            "location": "loc_re",
            "evidence": "resource exhausted",
            "description": "memory pressure",
            "impact": "HIGH",
        },
    ]

    recovered = _recover_targeted_tp(trace_payload, findings)

    ipi_entries = [entry for entry in recovered if entry["category"] == "Incorrect Problem Identification"]
    assert len(ipi_entries) == 1
    assert ipi_entries[0]["location"] == "loc_rnf"


def test_boost_joint_recall_expands_colocated_chain_without_duplicates() -> None:
    findings = [
        {
            "category": "Tool-related",
            "location": "loc_a",
            "evidence": "tool execution failed at this step",
            "description": "tool path had an issue",
            "impact": "MEDIUM",
        }
    ]

    boosted = _boost_joint_recall(findings)
    categories_at_location = {
        entry["category"]
        for entry in boosted
        if entry["location"] == "loc_a"
    }

    assert "Tool-related" in categories_at_location
    assert "Instruction Non-compliance" in categories_at_location
    assert "Formatting Errors" in categories_at_location
    assert "Tool Selection Errors" in categories_at_location
    assert "Goal Deviation" in categories_at_location

    boosted_again = _boost_joint_recall(boosted)
    assert len(boosted_again) == len(boosted)


def test_boost_joint_recall_adds_instruction_non_compliance_for_formatting_anchor() -> None:
    findings = [
        {
            "category": "Formatting Errors",
            "location": "loc_fmt",
            "evidence": "final answer must end with END_PLAN but output omitted it",
            "description": "output format did not satisfy instruction",
            "impact": "MEDIUM",
        }
    ]

    boosted = _boost_joint_recall(findings)
    categories_at_location = {
        entry["category"]
        for entry in boosted
        if entry["location"] == "loc_fmt"
    }

    assert "Formatting Errors" in categories_at_location
    assert "Instruction Non-compliance" in categories_at_location
    assert "Tool Selection Errors" in categories_at_location
    assert "Goal Deviation" in categories_at_location
