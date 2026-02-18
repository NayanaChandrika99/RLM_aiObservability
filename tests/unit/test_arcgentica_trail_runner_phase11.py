# ABOUTME: Validates Phase 11 GAIA runner file generation and scorer-compatible output structure.
# ABOUTME: Ensures deterministic path conventions and basic category detection for TRAIL traces.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arcgentica.trail_agent import analyze_trace
from arcgentica.trail_common import TrailRunConfig, run_output_dir
from arcgentica.trail_main import generate_split_outputs


def _write_trace(path: Path, trace_id: str, span_id: str, log_text: str) -> None:
    payload = {
        "trace_id": trace_id,
        "spans": [
            {
                "span_id": span_id,
                "status_code": "Unset",
                "span_attributes": {},
                "logs": [{"body": {"message": log_text}}],
                "child_spans": [],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_output_dir_uses_official_pattern(tmp_path: Path) -> None:
    config = TrailRunConfig(
        trail_data_dir=tmp_path / "data",
        split="GAIA",
        model="openai/gpt-5.2",
        output_dir=tmp_path / "results",
        semantic_checks="off",
    )
    expected = tmp_path / "results" / "outputs_openai-gpt-5.2-GAIA"
    assert run_output_dir(config) == expected


def test_generate_split_outputs_writes_one_file_per_trace(tmp_path: Path) -> None:
    split_dir = tmp_path / "data" / "GAIA"
    split_dir.mkdir(parents=True)
    _write_trace(split_dir / "trace_a.json", "trace_a", "span_a", "all good")
    _write_trace(split_dir / "trace_b.json", "trace_b", "span_b", "timed out while waiting")

    config = TrailRunConfig(
        trail_data_dir=tmp_path / "data",
        split="GAIA",
        model="openai/gpt-5.2",
        output_dir=tmp_path / "results",
        semantic_checks="off",
    )

    out_dir = generate_split_outputs(config)
    written = sorted(path.name for path in out_dir.glob("*.json"))
    assert written == ["trace_a.json", "trace_b.json"]


def test_generated_file_has_required_schema_keys(tmp_path: Path) -> None:
    split_dir = tmp_path / "data" / "GAIA"
    split_dir.mkdir(parents=True)
    _write_trace(split_dir / "trace_x.json", "trace_x", "span_x", "ok")

    config = TrailRunConfig(
        trail_data_dir=tmp_path / "data",
        split="GAIA",
        model="openai/gpt-5.2",
        output_dir=tmp_path / "results",
        semantic_checks="off",
    )

    out_dir = generate_split_outputs(config)
    payload = json.loads((out_dir / "trace_x.json").read_text(encoding="utf-8"))

    assert "errors" in payload
    assert "scores" in payload
    assert isinstance(payload["errors"], list)
    assert isinstance(payload["scores"], list)
    assert len(payload["scores"]) == 1
    assert set(payload["scores"][0].keys()) == {
        "reliability_score",
        "reliability_reasoning",
        "security_score",
        "security_reasoning",
        "instruction_adherence_score",
        "instruction_adherence_reasoning",
        "plan_opt_score",
        "plan_opt_reasoning",
        "overall",
    }


def test_analyze_trace_detects_timeout_issue() -> None:
    trace_payload = {
        "trace_id": "trace_1",
        "spans": [
            {
                "span_id": "span_1",
                "status_code": "Unset",
                "span_attributes": {"note": "request timed out during call"},
                "logs": [],
                "child_spans": [],
            }
        ],
    }

    result = analyze_trace(trace_payload, model="openai/gpt-5.2")
    categories = {entry["category"] for entry in result["errors"]}
    assert "Timeout Issues" in categories


def test_generate_split_outputs_strict_repairs_ungrounded_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_dir = tmp_path / "data" / "GAIA"
    split_dir.mkdir(parents=True)
    _write_trace(split_dir / "trace_strict.json", "trace_strict", "span_strict", "timed out hard")

    def fake_analyze_trace(trace_payload: dict, model: str, **kwargs: object) -> dict:
        del trace_payload
        del model
        del kwargs
        return {
            "trace_id": "trace_strict",
            "errors": [
                {
                    "category": "Timeout Issues",
                    "location": "not_in_trace",
                    "evidence": "ungrounded evidence",
                    "description": "desc",
                    "impact": "HIGH",
                }
            ],
            "scores": [{"overall": 3.0}],
        }

    monkeypatch.setattr("arcgentica.trail_main.analyze_trace", fake_analyze_trace)

    config = TrailRunConfig(
        trail_data_dir=tmp_path / "data",
        split="GAIA",
        model="openai/gpt-5.2",
        output_dir=tmp_path / "results",
        semantic_checks="strict",
    )

    out_dir = generate_split_outputs(config)
    payload = json.loads((out_dir / "trace_strict.json").read_text(encoding="utf-8"))
    assert len(payload["errors"]) == 1
    assert payload["errors"][0]["location"] == "span_strict"
    assert "timed out" in payload["errors"][0]["evidence"].lower()


def test_generate_split_outputs_writes_semantic_report_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_dir = tmp_path / "data" / "GAIA"
    split_dir.mkdir(parents=True)
    _write_trace(split_dir / "trace_sem.json", "trace_sem", "span_sem", "timed out waiting")

    def fake_analyze_trace(trace_payload: dict, model: str, **kwargs: object) -> dict:
        del trace_payload
        del model
        del kwargs
        return {
            "trace_id": "trace_sem",
            "errors": [
                {
                    "category": "Timeout Issues",
                    "location": "unknown_span",
                    "evidence": "ungrounded text",
                    "description": "desc",
                    "impact": "HIGH",
                }
            ],
            "scores": [{"overall": 3.0}],
        }

    monkeypatch.setattr("arcgentica.trail_main.analyze_trace", fake_analyze_trace)

    config = TrailRunConfig(
        trail_data_dir=tmp_path / "data",
        split="GAIA",
        model="openai/gpt-5.2",
        output_dir=tmp_path / "results",
        semantic_checks="strict",
    )
    semantic_report_path = tmp_path / "output" / "trail_semantic_report.json"
    generate_split_outputs(config, semantic_report_path=semantic_report_path)

    report = json.loads(semantic_report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "strict"
    assert report["split"] == "GAIA"
    assert report["totals"]["traces_processed"] == 1
    assert report["totals"]["location_repaired"] == 1
    assert report["totals"]["evidence_repaired"] == 1
    assert report["files"][0]["trace_file"] == "trace_sem.json"
