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


# ---------------------------------------------------------------------------
# Experiment runner tests
# ---------------------------------------------------------------------------

from arcgentica.trail_experiment import (
    load_subset_trace_ids,
    ExperimentConfig,
    run_experiment,
)


def test_load_subset_trace_ids_dev18() -> None:
    ids = load_subset_trace_ids("dev18")
    assert isinstance(ids, list)
    assert len(ids) == 18
    assert all(isinstance(tid, str) for tid in ids)


def test_load_subset_trace_ids_full_returns_none() -> None:
    ids = load_subset_trace_ids("full")
    assert ids is None


def test_experiment_config_defaults() -> None:
    cfg = ExperimentConfig(
        experiment_id="test_001",
        trail_data_dir=Path("/tmp/data"),
        gold_dir=Path("/tmp/gold"),
        output_dir=Path("/tmp/out"),
    )
    assert cfg.model == "openai/gpt-5-mini"
    assert cfg.approach == "single_pass"
    assert cfg.subset == "dev18"
    assert cfg.prompt_version == "v2"
    assert cfg.split == "GAIA"
    assert cfg.max_workers == 5


def test_run_experiment_writes_outputs_and_metrics(tmp_path: Path) -> None:
    """Run experiment with mocked LLM on 2 synthetic traces."""
    data_dir = tmp_path / "data" / "GAIA"
    data_dir.mkdir(parents=True)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)

    trace1 = {
        "trace_id": "aaaa1111bbbb2222cccc3333dddd4444",
        "spans": [{"span_id": "sp1", "span_name": "main", "status_code": "Unset",
                    "status_message": "", "span_attributes": {}, "logs": [],
                    "child_spans": []}],
    }
    trace2 = {
        "trace_id": "eeee5555ffff6666aaaa7777bbbb8888",
        "spans": [{"span_id": "sp2", "span_name": "main", "status_code": "Error",
                    "status_message": "timed out", "span_attributes": {}, "logs": [],
                    "child_spans": []}],
    }
    (data_dir / "aaaa1111bbbb2222cccc3333dddd4444.json").write_text(json.dumps(trace1))
    (data_dir / "eeee5555ffff6666aaaa7777bbbb8888.json").write_text(json.dumps(trace2))

    gold1 = {"errors": [], "scores": [{"reliability_score": 5, "reliability_reasoning": "ok",
             "security_score": 5, "security_reasoning": "ok",
             "instruction_adherence_score": 5, "instruction_adherence_reasoning": "ok",
             "plan_opt_score": 5, "plan_opt_reasoning": "ok", "overall": 5.0}]}
    gold2 = {"errors": [{"category": "Timeout Issues", "location": "sp2",
             "evidence": "timed out", "description": "timeout", "impact": "HIGH"}],
             "scores": [{"reliability_score": 2, "reliability_reasoning": "bad",
             "security_score": 5, "security_reasoning": "ok",
             "instruction_adherence_score": 3, "instruction_adherence_reasoning": "ok",
             "plan_opt_score": 2, "plan_opt_reasoning": "bad", "overall": 3.0}]}
    (gold_dir / "aaaa1111bbbb2222cccc3333dddd4444.json").write_text(json.dumps(gold1))
    (gold_dir / "eeee5555ffff6666aaaa7777bbbb8888.json").write_text(json.dumps(gold2))

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message={"content": json.dumps({
        "errors": [], "scores": [{"reliability_score": 5, "reliability_reasoning": "ok",
        "security_score": 5, "security_reasoning": "ok",
        "instruction_adherence_score": 5, "instruction_adherence_reasoning": "ok",
        "plan_opt_score": 5, "plan_opt_reasoning": "ok", "overall": 5.0}]
    })})]

    cfg = ExperimentConfig(
        experiment_id="test_run",
        trail_data_dir=tmp_path / "data",
        gold_dir=gold_dir,
        output_dir=tmp_path / "out",
        model="openai/gpt-5-mini",
        subset="full",
        split="GAIA",
    )

    with patch("arcgentica.trail_agent.completion", return_value=mock_response):
        result = run_experiment(cfg)

    assert result["experiment_id"] == "test_run"
    assert "metrics" in result
    assert "weighted_f1" in result["metrics"]
    assert result["traces_processed"] == 2

    output_dir = tmp_path / "out" / "test_run"
    assert output_dir.exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics.json").exists()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert "weighted_f1" in metrics

    log_path = tmp_path / "out" / "experiment_log.json"
    assert log_path.exists()


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
