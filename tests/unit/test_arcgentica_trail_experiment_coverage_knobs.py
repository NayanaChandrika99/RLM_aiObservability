# ABOUTME: Verifies TRAIL experiment CLI and runtime pass-through for coverage controls.
# ABOUTME: Ensures max chunk selection can be configured and forwarded to trace analysis.

from __future__ import annotations

import json
from pathlib import Path

from arcgentica import trail_experiment


def test_parse_args_accepts_coverage_knobs() -> None:
    args = trail_experiment.parse_args(
        [
            "--experiment-id",
            "exp_test",
            "--trail-data-dir",
            "data",
            "--gold-dir",
            "gold",
            "--max-chunks",
            "9",
            "--max-num-agents",
            "14",
            "--agent-call-timeout-seconds",
            "120",
            "--agent-timeout-retries",
            "0",
            "--joint-recall-boost",
        ]
    )

    assert args.max_chunks == 9
    assert args.max_num_agents == 14
    assert args.agent_call_timeout_seconds == 120
    assert args.agent_timeout_retries == 0
    assert args.joint_recall_boost is True


def test_process_trace_file_passes_coverage_knobs(monkeypatch: object, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_analyze_trace(trace_payload: dict[str, object], model: str, **kwargs: object) -> dict[str, object]:
        captured["trace_payload"] = trace_payload
        captured["model"] = model
        captured.update(kwargs)
        return {"trace_id": str(trace_payload.get("trace_id", "")), "errors": [], "scores": [{}]}

    def fake_enforce_semantic_faithfulness(
        trace_payload: dict[str, object],
        prediction: dict[str, object],
        mode: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del trace_payload
        del mode
        return prediction, {
            "total_errors": 0,
            "kept_errors": 0,
            "dropped_errors": 0,
            "repair_actions": {},
            "drop_reasons": {},
        }

    monkeypatch.setattr(trail_experiment, "analyze_trace", fake_analyze_trace)
    monkeypatch.setattr(trail_experiment, "enforce_semantic_faithfulness", fake_enforce_semantic_faithfulness)

    trace_file = tmp_path / "trace_a.json"
    trace_file.write_text(json.dumps({"trace_id": "trace_a", "spans": []}), encoding="utf-8")

    config = trail_experiment.ExperimentConfig(
        experiment_id="exp_test",
        trail_data_dir=tmp_path,
        gold_dir=tmp_path,
        output_dir=tmp_path,
        model="openai/gpt-5-mini",
        root_model="openai/gpt-5.2",
        chunk_model="openai/gpt-5-mini",
        approach="on",
        subset="full",
        prompt_version="v2",
        split="GAIA",
        max_workers=1,
        max_chunks=9,
        max_num_agents=14,
        agent_call_timeout_seconds=120,
        agent_timeout_retries=0,
        joint_recall_boost=True,
        semantic_checks="strict",
        resume=False,
        notes="",
    )

    result = trail_experiment._process_trace_file(trace_file, config)

    assert result["status"] == "ok"
    assert captured["max_chunks"] == 9
    assert captured["max_num_agents"] == 14
    assert captured["agent_call_timeout_seconds"] == 120
    assert captured["agent_timeout_retries"] == 0
    assert captured["joint_recall_boost"] is True
