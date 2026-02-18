# ABOUTME: Validates Phase 10 RCA evaluation metrics over manifest and run-record artifacts.
# ABOUTME: Ensures accuracy, per-label diagnostics, runtime stats, and persisted report output are deterministic.

from __future__ import annotations

import json
from pathlib import Path

from investigator.rca.evaluate import evaluate_rca_runs, format_evaluation_report, main


def _write_run(
    *,
    runs_root: Path,
    run_id: str,
    trace_id: str,
    label: str,
    confidence: float,
    evidence_count: int,
    started_at: str,
    completed_at: str,
    cost_usd: float,
    tokens_in: int,
    tokens_out: int,
    iterations: int,
    max_iterations: int,
    tool_calls: int,
    max_tool_calls: int,
) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "output.json"
    output_path.write_text(
        json.dumps(
            {
                "trace_id": trace_id,
                "primary_label": label,
                "summary": "evaluation-test",
                "confidence": confidence,
                "evidence_refs": [{"ref": f"evidence-{index}"} for index in range(evidence_count)],
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
        "started_at": started_at,
        "completed_at": completed_at,
        "dataset_ref": {},
        "input_ref": {"trace_ids": [trace_id]},
        "runtime_ref": {
            "model_name": "gpt-4o-mini",
            "usage": {
                "iterations": iterations,
                "tool_calls": tool_calls,
                "llm_subcalls": 1,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd,
            },
            "budget": {
                "max_iterations": max_iterations,
                "max_tool_calls": max_tool_calls,
            },
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


def test_evaluate_rca_runs_computes_metrics_and_uses_latest_run(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    runs_dir = tmp_path / "runs"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "seeded_failures_v1",
                "cases": [
                    {"trace_id": "trace-1", "expected_label": "tool_failure"},
                    {"trace_id": "trace-2", "expected_label": "retrieval_failure"},
                    {"trace_id": "trace-3", "expected_label": "instruction_failure"},
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    _write_run(
        runs_root=runs_dir,
        run_id="run-older",
        trace_id="trace-1",
        label="retrieval_failure",
        confidence=0.30,
        evidence_count=1,
        started_at="2026-02-13T00:00:00Z",
        completed_at="2026-02-13T00:00:05Z",
        cost_usd=0.05,
        tokens_in=100,
        tokens_out=50,
        iterations=2,
        max_iterations=8,
        tool_calls=3,
        max_tool_calls=40,
    )
    _write_run(
        runs_root=runs_dir,
        run_id="run-latest",
        trace_id="trace-1",
        label="tool_failure",
        confidence=0.92,
        evidence_count=2,
        started_at="2026-02-13T00:00:10Z",
        completed_at="2026-02-13T00:00:20Z",
        cost_usd=0.20,
        tokens_in=200,
        tokens_out=100,
        iterations=4,
        max_iterations=8,
        tool_calls=10,
        max_tool_calls=40,
    )
    _write_run(
        runs_root=runs_dir,
        run_id="run-trace-2",
        trace_id="trace-2",
        label="tool_failure",
        confidence=0.12,
        evidence_count=1,
        started_at="2026-02-13T00:01:00Z",
        completed_at="2026-02-13T00:01:20Z",
        cost_usd=0.10,
        tokens_in=120,
        tokens_out=80,
        iterations=6,
        max_iterations=12,
        tool_calls=12,
        max_tool_calls=24,
    )

    report = evaluate_rca_runs(manifest_path=manifest_path, runs_dir=runs_dir)

    top1 = report["metrics"]["top1_accuracy"]
    assert top1["correct"] == 1
    assert top1["total"] == 3
    assert abs(top1["score"] - (1.0 / 3.0)) < 1e-9
    assert report["dataset"]["missing_run_count"] == 1
    assert report["model"]["name"] == "gpt-4o-mini"

    per_label = report["metrics"]["per_label"]
    assert per_label["tool_failure"]["support"] == 1
    assert abs(per_label["tool_failure"]["precision"] - 0.5) < 1e-9
    assert abs(per_label["tool_failure"]["recall"] - 1.0) < 1e-9
    assert per_label["retrieval_failure"]["support"] == 1
    assert per_label["retrieval_failure"]["recall"] == 0.0
    assert per_label["instruction_failure"]["support"] == 1

    confidence = report["metrics"]["confidence"]
    assert abs(confidence["avg_correct"] - 0.92) < 1e-9
    assert abs(confidence["avg_incorrect"] - 0.12) < 1e-9

    evidence_quality = report["metrics"]["evidence_quality"]
    assert abs(evidence_quality["avg_correct_evidence_refs"] - 2.0) < 1e-9
    assert abs(evidence_quality["avg_incorrect_evidence_refs"] - 0.5) < 1e-9

    runtime = report["metrics"]["runtime"]
    assert abs(runtime["cost_total_usd"] - 0.3) < 1e-9
    assert abs(runtime["cost_avg_usd"] - 0.15) < 1e-9
    assert abs(runtime["wall_time_total_sec"] - 30.0) < 1e-9
    assert abs(runtime["wall_time_avg_sec"] - 15.0) < 1e-9
    assert runtime["tokens_total"] == 500
    assert abs(runtime["tokens_avg"] - 250.0) < 1e-9
    assert abs(runtime["avg_iteration_utilization_pct"] - 50.0) < 1e-9
    assert abs(runtime["avg_tool_call_utilization_pct"] - 37.5) < 1e-9

    cases = {item["trace_id"]: item for item in report["cases"]}
    assert cases["trace-1"]["run_id"] == "run-latest"
    assert cases["trace-1"]["correct"] is True
    assert cases["trace-3"]["run_status"] == "missing"


def test_trace_rca_evaluate_main_writes_report_and_prints_summary(
    tmp_path: Path,
    capsys,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    runs_dir = tmp_path / "runs"
    output_path = tmp_path / "artifacts" / "evaluation" / "eval_report.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "seeded_failures_v1",
                "cases": [{"trace_id": "trace-1", "expected_label": "tool_failure"}],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_run(
        runs_root=runs_dir,
        run_id="run-one",
        trace_id="trace-1",
        label="tool_failure",
        confidence=0.88,
        evidence_count=1,
        started_at="2026-02-13T00:00:00Z",
        completed_at="2026-02-13T00:00:10Z",
        cost_usd=0.05,
        tokens_in=50,
        tokens_out=25,
        iterations=2,
        max_iterations=4,
        tool_calls=4,
        max_tool_calls=8,
    )

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--runs-dir",
            str(runs_dir),
            "--output",
            str(output_path),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "RLM-RCA Evaluation Report" in stdout
    assert "Top-1 Accuracy: 1/1" in stdout
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["top1_accuracy"]["correct"] == 1
    assert payload["metrics"]["top1_accuracy"]["total"] == 1


def test_backward_compat_run_records_without_scaffold_field_still_evaluate(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    runs_dir = tmp_path / "runs"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "seeded_failures_v1",
                "cases": [{"trace_id": "trace-1", "expected_label": "tool_failure"}],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_run(
        runs_root=runs_dir,
        run_id="run-no-scaffold",
        trace_id="trace-1",
        label="tool_failure",
        confidence=0.88,
        evidence_count=1,
        started_at="2026-02-13T00:00:00Z",
        completed_at="2026-02-13T00:00:10Z",
        cost_usd=0.05,
        tokens_in=50,
        tokens_out=25,
        iterations=2,
        max_iterations=4,
        tool_calls=4,
        max_tool_calls=8,
    )

    report = evaluate_rca_runs(manifest_path=manifest_path, runs_dir=runs_dir)

    top1 = report["metrics"]["top1_accuracy"]
    assert top1["correct"] == 1
    assert top1["total"] == 1
    assert abs(top1["score"] - 1.0) < 1e-9


def test_format_evaluation_report_includes_expected_sections(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    runs_dir = tmp_path / "runs"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "seeded_failures_v1",
                "cases": [{"trace_id": "trace-1", "expected_label": "tool_failure"}],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_run(
        runs_root=runs_dir,
        run_id="run-one",
        trace_id="trace-1",
        label="tool_failure",
        confidence=0.88,
        evidence_count=1,
        started_at="2026-02-13T00:00:00Z",
        completed_at="2026-02-13T00:00:10Z",
        cost_usd=0.05,
        tokens_in=50,
        tokens_out=25,
        iterations=2,
        max_iterations=4,
        tool_calls=4,
        max_tool_calls=8,
    )
    report = evaluate_rca_runs(manifest_path=manifest_path, runs_dir=runs_dir)

    rendered = format_evaluation_report(report)
    assert "Per-Label Results:" in rendered
    assert "Evidence Quality:" in rendered
    assert "Confidence:" in rendered
    assert "Budget Utilization:" in rendered
