# ABOUTME: Validates Phase 11 TRAIL comparison reporting across baseline, candidate, and semantic artifacts.
# ABOUTME: Ensures metric deltas and acceptance gates are deterministic and machine-readable.

from __future__ import annotations

import json
from pathlib import Path

from arcgentica.trail_compare import (
    compare_runs,
    find_metrics_file,
    parse_metrics_file,
    write_report,
)


def _write_metrics(path: Path, weighted_f1: float, location: float, joint: float) -> None:
    path.write_text(
        "\n".join(
            [
                f"Weighted F1: {weighted_f1:.4f}",
                f"Average Location Accuracy: {location:.4f}",
                f"Average Location-Category Joint Accuracy: {joint:.4f}",
            ]
        ),
        encoding="utf-8",
    )


def test_find_metrics_file_for_split(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    gaia_metrics = results_dir / "outputs_openai-gpt-5.2-GAIA-metrics.txt"
    swe_metrics = results_dir / "outputs_openai-gpt-5.2-SWE Bench-metrics.txt"
    _write_metrics(gaia_metrics, weighted_f1=0.1, location=0.2, joint=0.3)
    _write_metrics(swe_metrics, weighted_f1=0.1, location=0.2, joint=0.3)

    selected = find_metrics_file(results_dir, split="GAIA")
    assert selected == gaia_metrics


def test_parse_metrics_file_extracts_required_values(tmp_path: Path) -> None:
    metrics_file = tmp_path / "metrics.txt"
    _write_metrics(metrics_file, weighted_f1=0.1234, location=0.2345, joint=0.3456)

    parsed = parse_metrics_file(metrics_file)
    assert parsed["weighted_f1"] == 0.1234
    assert parsed["location_accuracy"] == 0.2345
    assert parsed["joint_accuracy"] == 0.3456


def test_compare_runs_builds_deltas_and_acceptance(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()

    _write_metrics(
        baseline_dir / "outputs_openai-gpt-5.2-GAIA-metrics.txt",
        weighted_f1=0.11,
        location=0.21,
        joint=0.17,
    )
    _write_metrics(
        candidate_dir / "outputs_openai-gpt-5.2-GAIA-metrics.txt",
        weighted_f1=0.22,
        location=0.31,
        joint=0.23,
    )

    semantic_report_path = tmp_path / "trail_semantic_report.json"
    semantic_report_path.write_text(
        json.dumps(
            {
                "mode": "strict",
                "split": "GAIA",
                "totals": {
                    "traces_processed": 117,
                    "grounded_evidence_rate": 0.99,
                    "dropped_errors": 0,
                    "location_repaired": 3,
                    "evidence_repaired": 8,
                },
            }
        ),
        encoding="utf-8",
    )

    report = compare_runs(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        semantic_report_path=semantic_report_path,
        split="GAIA",
        paper_joint_reference=0.18,
        grounded_threshold=0.95,
    )

    assert report["deltas"]["joint_accuracy"] == 0.06
    assert report["acceptance"]["beats_baseline_joint_accuracy"] is True
    assert report["acceptance"]["beats_paper_joint_reference"] is True
    assert report["acceptance"]["meets_semantic_grounded_threshold"] is True
    assert report["acceptance"]["has_no_semantic_drops"] is True
    assert report["acceptance"]["accepted"] is True


def test_write_report_persists_json(tmp_path: Path) -> None:
    out_path = tmp_path / "report.json"
    payload = {"status": "ok", "value": 1}
    write_report(payload, out_path)
    assert json.loads(out_path.read_text(encoding="utf-8")) == payload
