# ABOUTME: Validates offline TRAIL output reprocessing so joint-recall rules can be applied without new model calls.
# ABOUTME: Ensures boosted outputs stay schema-valid, semantically grounded, and scorer-compatible.

from __future__ import annotations

import json
from pathlib import Path

from arcgentica.trail_agent import (
    apply_joint_recall_boost_to_prediction,
    apply_trajectory_action_correlation_to_prediction,
)
from arcgentica.trail_reprocess_outputs import reprocess_outputs


def test_apply_joint_recall_boost_to_prediction_preserves_scores_and_boosts_errors() -> None:
    prediction = {
        "trace_id": "trace_1",
        "errors": [
            {
                "category": "Formatting Errors",
                "location": "span_a",
                "evidence": "Output violated END_PLAN format requirement",
                "description": "Formatting violation",
                "impact": "MEDIUM",
            }
        ],
        "scores": [{"overall": 3.0}],
    }

    boosted = apply_joint_recall_boost_to_prediction(prediction)
    categories = {
        item["category"]
        for item in boosted["errors"]
        if item["location"] == "span_a"
    }

    assert boosted["scores"] == prediction["scores"]
    assert "Formatting Errors" in categories
    assert "Instruction Non-compliance" in categories
    assert "Tool Selection Errors" in categories
    assert "Goal Deviation" in categories


def test_apply_trajectory_action_correlation_to_prediction_relocates_tool_error() -> None:
    trace_payload = {
        "trace_id": "trace_tac",
        "spans": [
            {
                "span_id": "step_span",
                "span_name": "Step 2",
                "status_code": "Ok",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "attempt page navigation"}],
                "child_spans": [
                    {
                        "span_id": "tool_span",
                        "span_name": "PageDownTool",
                        "status_code": "Error",
                        "status_message": "Error when executing tool",
                        "span_attributes": {"tool_name": "PageDownTool"},
                        "logs": [{"body": "unexpected keyword argument 'step_count'"}],
                        "child_spans": [],
                    }
                ],
            }
        ],
    }
    prediction = {
        "trace_id": "trace_tac",
        "errors": [
            {
                "category": "Tool Definition Issues",
                "location": "step_span",
                "evidence": "PageDownTool raised unexpected keyword argument step_count",
                "description": "tool schema mismatch",
                "impact": "MEDIUM",
            }
        ],
        "scores": [{"overall": 3.0}],
    }

    correlated = apply_trajectory_action_correlation_to_prediction(trace_payload, prediction)

    assert correlated["scores"] == prediction["scores"]
    assert correlated["errors"][0]["location"] == "tool_span"


def test_reprocess_outputs_writes_boosted_results_and_scores(tmp_path: Path) -> None:
    trace_dir = tmp_path / "data" / "GAIA"
    trace_dir.mkdir(parents=True)
    trace_payload = {
        "trace_id": "trace_a",
        "spans": [
            {
                "span_id": "span_a",
                "span_name": "LiteLLMModel.__call__",
                "status_code": "Error",
                "status_message": "Output violated END_PLAN format requirement",
                "span_attributes": {},
                "logs": [{"body": "Output violated END_PLAN format requirement"}],
                "child_spans": [],
            }
        ],
    }
    (trace_dir / "trace_a.json").write_text(json.dumps(trace_payload), encoding="utf-8")

    input_dir = tmp_path / "input_outputs"
    input_dir.mkdir()
    prediction = {
        "trace_id": "trace_a",
        "errors": [
            {
                "category": "Formatting Errors",
                "location": "span_a",
                "evidence": "Output violated END_PLAN format requirement",
                "description": "Formatting violation",
                "impact": "MEDIUM",
            }
        ],
        "scores": [{"overall": 3.0}],
    }
    (input_dir / "trace_a.json").write_text(json.dumps(prediction), encoding="utf-8")

    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    gold = {
        "trace_id": "trace_a",
        "errors": [
            {
                "category": "Formatting Errors",
                "location": "span_a",
                "evidence": "x",
                "description": "x",
                "impact": "MEDIUM",
            },
            {
                "category": "Instruction Non-compliance",
                "location": "span_a",
                "evidence": "x",
                "description": "x",
                "impact": "MEDIUM",
            },
        ],
        "scores": [{"overall": 3.0}],
    }
    (gold_dir / "trace_a.json").write_text(json.dumps(gold), encoding="utf-8")

    output_dir = tmp_path / "reprocessed_outputs"
    summary = reprocess_outputs(
        input_dir=input_dir,
        output_dir=output_dir,
        trail_data_dir=tmp_path / "data",
        split="GAIA",
        semantic_checks="strict",
        joint_recall_boost=True,
        gold_dir=gold_dir,
    )

    output_payload = json.loads((output_dir / "trace_a.json").read_text(encoding="utf-8"))
    output_categories = {error["category"] for error in output_payload["errors"]}

    assert summary["files_processed"] == 1
    assert summary["trajectory_action_correlation"] is True
    assert summary["semantic"]["totals"]["dropped_errors"] == 0
    assert summary["metrics"] is not None
    assert summary["metrics"]["joint_accuracy"] == 1.0
    assert "Formatting Errors" in output_categories
    assert "Instruction Non-compliance" in output_categories
