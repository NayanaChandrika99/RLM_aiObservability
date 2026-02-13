# ABOUTME: Validates TRAIL experiment infrastructure including dev subset, prompt v2, and experiment runner.
# ABOUTME: Tests are deterministic and do not call external LLM APIs.

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_dev_subset_manifest_is_valid_json() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "arcgentica" / "dev_subset_manifest.json"
    assert manifest_path.exists(), f"Manifest not found at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "subset_id" in manifest
    assert manifest["subset_id"] == "dev18"
    assert "trace_ids" in manifest
    assert isinstance(manifest["trace_ids"], list)
    assert len(manifest["trace_ids"]) == 18
    # All IDs should be 32-char hex strings
    for tid in manifest["trace_ids"]:
        assert isinstance(tid, str)
        assert len(tid) == 32, f"trace_id {tid} is not 32 chars"
    # Should be sorted for determinism
    assert manifest["trace_ids"] == sorted(manifest["trace_ids"])


# ---------------------------------------------------------------------------
# Prompt V2 + smart truncation tests
# ---------------------------------------------------------------------------

from arcgentica.trail_prompt_v2 import (
    TRAIL_SINGLE_PASS_PROMPT_V2,
    build_single_pass_message,
    smart_truncate_trace,
)
from arcgentica.trail_common import TRAIL_LEAF_CATEGORIES


def test_prompt_v2_contains_all_leaf_categories() -> None:
    for category in TRAIL_LEAF_CATEGORIES:
        assert category in TRAIL_SINGLE_PASS_PROMPT_V2, f"Missing category: {category}"


def test_prompt_v2_contains_location_rules() -> None:
    prompt = TRAIL_SINGLE_PASS_PROMPT_V2
    assert "Resource Abuse" in prompt
    assert "first" in prompt.lower()
    assert "last" in prompt.lower()
    assert "span_id" in prompt


def test_prompt_v2_contains_detection_hints() -> None:
    prompt = TRAIL_SINGLE_PASS_PROMPT_V2
    # Check that detection guidance exists for categories
    assert "DETECT:" in prompt or "detect:" in prompt.lower()


def test_prompt_v2_contains_score_guidance() -> None:
    prompt = TRAIL_SINGLE_PASS_PROMPT_V2
    assert "reliability_score" in prompt
    assert "security_score" in prompt
    assert "instruction_adherence_score" in prompt
    assert "plan_opt_score" in prompt


def test_smart_truncate_preserves_small_trace() -> None:
    trace = {
        "trace_id": "abc123",
        "spans": [
            {
                "span_id": "span_1",
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "hello world"}],
                "child_spans": [],
            }
        ],
    }
    result = smart_truncate_trace(trace, max_chars=500_000)
    assert result["trace_id"] == "abc123"
    assert len(result["spans"]) == 1


def test_smart_truncate_truncates_large_trace() -> None:
    spans = []
    for i in range(50):
        spans.append({
            "span_id": f"span_{i:04d}",
            "span_name": f"op_{i}",
            "status_code": "Error" if i == 5 else "Unset",
            "status_message": "some failure" if i == 5 else "",
            "span_attributes": {},
            "logs": [{"body": "x" * 2000}],
            "child_spans": [],
        })
    trace = {"trace_id": "big_trace", "spans": spans}
    result = smart_truncate_trace(trace, max_chars=10_000)
    result_text = json.dumps(result)
    # Should be truncated
    assert len(result_text) <= 20_000  # some overhead
    # Error span should be preserved
    error_spans = [s for s in result["spans"] if s.get("status_code") == "Error"]
    assert len(error_spans) >= 1
    # Span index should be appended
    assert "span_index" in result


def test_build_single_pass_message_returns_string() -> None:
    trace = {
        "trace_id": "test_trace",
        "spans": [
            {
                "span_id": "sp1",
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [],
                "child_spans": [],
            }
        ],
    }
    msg = build_single_pass_message(trace)
    assert isinstance(msg, str)
    assert "sp1" in msg
    assert "errors" in msg.lower()


# ---------------------------------------------------------------------------
# Single-pass mode tests
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock
from arcgentica.trail_agent import analyze_trace


def _make_trace(trace_id: str = "t1", span_id: str = "s1", log_text: str = "ok") -> dict:
    return {
        "trace_id": trace_id,
        "spans": [
            {
                "span_id": span_id,
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": log_text}],
                "child_spans": [],
            }
        ],
    }


def test_single_pass_mode_calls_litellm() -> None:
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message={"content": json.dumps({
            "errors": [
                {
                    "category": "Formatting Errors",
                    "location": "s1",
                    "evidence": "missing format",
                    "description": "Output format wrong",
                    "impact": "LOW",
                }
            ],
            "scores": [
                {
                    "reliability_score": 4,
                    "reliability_reasoning": "Mostly reliable",
                    "security_score": 5,
                    "security_reasoning": "No issues",
                    "instruction_adherence_score": 3,
                    "instruction_adherence_reasoning": "Missed format",
                    "plan_opt_score": 4,
                    "plan_opt_reasoning": "Decent plan",
                    "overall": 4.0,
                }
            ],
        })})
    ]

    with patch("arcgentica.trail_agent.completion", return_value=mock_response) as mock_comp:
        result = analyze_trace(
            _make_trace(),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    mock_comp.assert_called_once()
    assert result["trace_id"] == "t1"
    assert len(result["errors"]) == 1
    assert result["errors"][0]["category"] == "Formatting Errors"
    assert result["errors"][0]["location"] == "s1"
    assert len(result["scores"]) == 1
    assert result["scores"][0]["reliability_score"] == 4


def test_single_pass_mode_falls_back_on_error() -> None:
    with patch("arcgentica.trail_agent.completion", side_effect=Exception("API down")):
        result = analyze_trace(
            _make_trace(log_text="timed out while waiting"),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    assert result["trace_id"] == "t1"
    categories = [e["category"] for e in result["errors"]]
    assert "Timeout Issues" in categories


def test_single_pass_validates_locations() -> None:
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message={"content": json.dumps({
            "errors": [
                {
                    "category": "Formatting Errors",
                    "location": "s1",
                    "evidence": "valid location",
                    "description": "ok",
                    "impact": "LOW",
                },
                {
                    "category": "Goal Deviation",
                    "location": "INVALID_SPAN",
                    "evidence": "bad location",
                    "description": "wrong",
                    "impact": "HIGH",
                },
            ],
            "scores": [{"reliability_score": 3, "reliability_reasoning": "ok",
                        "security_score": 5, "security_reasoning": "ok",
                        "instruction_adherence_score": 3, "instruction_adherence_reasoning": "ok",
                        "plan_opt_score": 3, "plan_opt_reasoning": "ok", "overall": 3.5}],
        })})
    ]

    with patch("arcgentica.trail_agent.completion", return_value=mock_response):
        result = analyze_trace(
            _make_trace(),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    locations = [e["location"] for e in result["errors"]]
    assert all(loc == "s1" for loc in locations), f"Expected all locations to be 's1', got {locations}"
