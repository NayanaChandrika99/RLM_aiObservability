# ABOUTME: Validates comparative evaluation across scaffold presets using synthetic run records.
# ABOUTME: Ensures report structure, delta computations, and format output are deterministic.

from __future__ import annotations

import json
from pathlib import Path

from investigator.rca.evaluate import (
    evaluate_rca_runs_comparative,
    format_comparative_report,
)


def _write_run(
    *,
    runs_root: Path,
    run_id: str,
    trace_id: str,
    label: str,
    confidence: float,
    cost_usd: float,
    tokens_in: int,
    tokens_out: int,
) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "output.json"
    output_path.write_text(
        json.dumps(
            {
                "trace_id": trace_id,
                "primary_label": label,
                "summary": "test",
                "confidence": confidence,
                "evidence_refs": [{"ref": "evidence-0"}],
                "remediation": [],
                "gaps": [],
                "schema_version": "1.0.0",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    run_record = {
        "run_id": run_id,
        "run_type": "rca",
        "status": "succeeded",
        "started_at": "2026-02-13T00:00:00Z",
        "completed_at": "2026-02-13T00:00:05Z",
        "dataset_ref": {},
        "input_ref": {"trace_ids": [trace_id]},
        "runtime_ref": {
            "model_name": "gpt-4o-mini",
            "usage": {
                "iterations": 2,
                "tool_calls": 4,
                "llm_subcalls": 1,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd,
            },
            "budget": {"max_iterations": 8, "max_tool_calls": 40},
        },
        "output_ref": {"artifact_path": str(output_path), "schema_version": "1.0.0"},
        "writeback_ref": {"writeback_status": "succeeded"},
        "error": None,
        "schema_version": "1.0.0",
    }
    (run_dir / "run_record.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_evaluate_rca_runs_comparative_produces_correct_structure(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "test_compare",
                "cases": [
                    {"trace_id": "t1", "expected_label": "tool_failure"},
                    {"trace_id": "t2", "expected_label": "retrieval_failure"},
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    heuristic_dir = tmp_path / "runs" / "heuristic"
    rlm_dir = tmp_path / "runs" / "rlm"

    _write_run(
        runs_root=heuristic_dir, run_id="h1", trace_id="t1",
        label="tool_failure", confidence=0.7, cost_usd=0.0, tokens_in=0, tokens_out=0,
    )
    _write_run(
        runs_root=heuristic_dir, run_id="h2", trace_id="t2",
        label="tool_failure", confidence=0.6, cost_usd=0.0, tokens_in=0, tokens_out=0,
    )
    _write_run(
        runs_root=rlm_dir, run_id="r1", trace_id="t1",
        label="tool_failure", confidence=0.9, cost_usd=0.05, tokens_in=500, tokens_out=200,
    )
    _write_run(
        runs_root=rlm_dir, run_id="r2", trace_id="t2",
        label="retrieval_failure", confidence=0.85, cost_usd=0.04, tokens_in=400, tokens_out=150,
    )

    report = evaluate_rca_runs_comparative(
        manifest_path=manifest_path,
        scaffold_runs={"heuristic": heuristic_dir, "rlm": rlm_dir},
    )

    assert "scaffolds" in report
    assert "per_scaffold_full" in report
    assert "heuristic" in report["scaffolds"]
    assert "rlm" in report["scaffolds"]

    heuristic_summary = report["scaffolds"]["heuristic"]
    rlm_summary = report["scaffolds"]["rlm"]

    assert heuristic_summary["correct"] == 1
    assert heuristic_summary["total"] == 2
    assert abs(heuristic_summary["accuracy"] - 0.5) < 1e-9

    assert rlm_summary["correct"] == 2
    assert rlm_summary["total"] == 2
    assert abs(rlm_summary["accuracy"] - 1.0) < 1e-9


def test_delta_vs_heuristic_computed_correctly(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "test_delta",
                "cases": [
                    {"trace_id": "t1", "expected_label": "tool_failure"},
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    heuristic_dir = tmp_path / "runs" / "heuristic"
    rlm_dir = tmp_path / "runs" / "rlm"

    _write_run(
        runs_root=heuristic_dir, run_id="h1", trace_id="t1",
        label="retrieval_failure", confidence=0.5, cost_usd=0.0, tokens_in=0, tokens_out=0,
    )
    _write_run(
        runs_root=rlm_dir, run_id="r1", trace_id="t1",
        label="tool_failure", confidence=0.9, cost_usd=0.10, tokens_in=500, tokens_out=200,
    )

    report = evaluate_rca_runs_comparative(
        manifest_path=manifest_path,
        scaffold_runs={"heuristic": heuristic_dir, "rlm": rlm_dir},
    )

    heuristic_delta = report["scaffolds"]["heuristic"]["delta_vs_heuristic"]
    assert abs(heuristic_delta["accuracy_gain"]) < 1e-9
    assert abs(heuristic_delta["cost_delta_usd"]) < 1e-9

    rlm_delta = report["scaffolds"]["rlm"]["delta_vs_heuristic"]
    assert abs(rlm_delta["accuracy_gain"] - 1.0) < 1e-9
    assert abs(rlm_delta["cost_delta_usd"] - 0.10) < 1e-9


def test_format_comparative_report_includes_scaffold_names(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "test_format",
                "cases": [
                    {"trace_id": "t1", "expected_label": "tool_failure"},
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    heuristic_dir = tmp_path / "runs" / "heuristic"
    rlm_tips_dir = tmp_path / "runs" / "rlm_tips"

    _write_run(
        runs_root=heuristic_dir, run_id="h1", trace_id="t1",
        label="tool_failure", confidence=0.7, cost_usd=0.0, tokens_in=0, tokens_out=0,
    )
    _write_run(
        runs_root=rlm_tips_dir, run_id="rt1", trace_id="t1",
        label="tool_failure", confidence=0.95, cost_usd=0.08, tokens_in=600, tokens_out=250,
    )

    report = evaluate_rca_runs_comparative(
        manifest_path=manifest_path,
        scaffold_runs={"heuristic": heuristic_dir, "rlm_tips": rlm_tips_dir},
    )

    rendered = format_comparative_report(report)
    assert "heuristic" in rendered
    assert "rlm_tips" in rendered
    assert "RLM-RCA Comparative Report" in rendered
    assert "Per-Label F1 Breakdown:" in rendered


def test_comparative_per_label_f1_present(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "test_f1",
                "cases": [
                    {"trace_id": "t1", "expected_label": "tool_failure"},
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    heuristic_dir = tmp_path / "runs" / "heuristic"
    _write_run(
        runs_root=heuristic_dir, run_id="h1", trace_id="t1",
        label="tool_failure", confidence=0.7, cost_usd=0.0, tokens_in=0, tokens_out=0,
    )

    report = evaluate_rca_runs_comparative(
        manifest_path=manifest_path,
        scaffold_runs={"heuristic": heuristic_dir},
    )

    heuristic_summary = report["scaffolds"]["heuristic"]
    assert "per_label_f1" in heuristic_summary
    assert "tool_failure" in heuristic_summary["per_label_f1"]
    assert abs(heuristic_summary["per_label_f1"]["tool_failure"] - 1.0) < 1e-9
