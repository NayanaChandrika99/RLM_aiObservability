# ABOUTME: Validates the Phase 7 proof-run artifact writer over frozen dataset benchmark inputs.
# ABOUTME: Ensures proof reports are persisted deterministically under artifacts/proof_runs/<proof_run_id>/.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from investigator.proof.run_phase7_proof import run_frozen_dataset_proof


def _write_dataset(parquet_path: Path, manifest_path: Path) -> None:
    rows = [
        {
            "name": "agent.run",
            "span_kind": "AGENT",
            "parent_id": None,
            "start_time": pd.Timestamp("2026-02-10T00:00:00Z"),
            "end_time": pd.Timestamp("2026-02-10T00:00:01Z"),
            "status_code": "UNSET",
            "status_message": "",
            "events": [],
            "context.span_id": "root-1",
            "context.trace_id": "trace-1",
            "attributes.phase1": {"run_id": "seed_run_0000", "step": None, "project": "phase7-proof"},
            "attributes.http": None,
        },
        {
            "name": "tool.call",
            "span_kind": "TOOL",
            "parent_id": "root-1",
            "start_time": pd.Timestamp("2026-02-10T00:00:00Z"),
            "end_time": pd.Timestamp("2026-02-10T00:00:01Z"),
            "status_code": "ERROR",
            "status_message": "forced tool timeout",
            "events": [],
            "context.span_id": "tool-1",
            "context.trace_id": "trace-1",
            "attributes.phase1": {"run_id": "seed_run_0000", "step": "tool.call"},
            "attributes.http": None,
        },
    ]
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    manifest = {
        "dataset_id": "seeded_failures_v1",
        "generator_version": "0.1.0",
        "seed": 42,
        "cases": [
            {"run_id": "seed_run_0000", "trace_id": None, "expected_label": "tool_failure"},
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_controls(controls_dir: Path) -> None:
    controls_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "controls": [
            {
                "control_id": "control.error.free",
                "controls_version": "controls-v1",
                "severity": "high",
                "required_evidence": [],
                "max_error_spans": 0,
                "remediation_template": "Fix error spans.",
                "applies_when": {"app_types": [], "tools": [], "data_domains": []},
            }
        ]
    }
    (controls_dir / "controls_v1.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_snapshots(snapshots_dir: Path) -> None:
    snap_a = snapshots_dir / "snap-a"
    snap_b = snapshots_dir / "snap-b"
    snap_a.mkdir(parents=True, exist_ok=True)
    snap_b.mkdir(parents=True, exist_ok=True)
    (snap_a / "metadata.json").write_text(
        json.dumps({"created_at": "2026-02-10T00:00:00Z", "git_commit": "abc111", "tag": "baseline"}),
        encoding="utf-8",
    )
    (snap_b / "metadata.json").write_text(
        json.dumps({"created_at": "2026-02-10T00:30:00Z", "git_commit": "abc222", "tag": "candidate"}),
        encoding="utf-8",
    )
    (snap_a / "runtime.env").write_text("MODEL=gpt-5-mini\n", encoding="utf-8")
    (snap_b / "runtime.env").write_text("MODEL=gpt-5-mini\nFEATURE_FLAG=on\n", encoding="utf-8")


def test_run_frozen_dataset_proof_writes_artifact(tmp_path: Path) -> None:
    parquet_path = tmp_path / "spans.parquet"
    manifest_path = tmp_path / "manifest.json"
    controls_dir = tmp_path / "controls"
    snapshots_dir = tmp_path / "snapshots"
    proof_root = tmp_path / "artifacts" / "proof_runs"
    run_artifacts_root = tmp_path / "artifacts" / "investigator_runs"

    _write_dataset(parquet_path, manifest_path)
    _write_controls(controls_dir)
    _write_snapshots(snapshots_dir)

    report = run_frozen_dataset_proof(
        proof_run_id="phase7-test",
        spans_parquet_path=parquet_path,
        manifest_path=manifest_path,
        project_name="phase7-proof",
        controls_version="controls-v1",
        controls_dir=controls_dir,
        snapshots_dir=snapshots_dir,
        proof_artifacts_root=proof_root,
        evaluator_artifacts_root=run_artifacts_root,
    )

    output_path = proof_root / "phase7-test" / "proof_report.json"
    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["proof_run_id"] == "phase7-test"
    assert persisted["capabilities"] == report["capabilities"]
    assert persisted["dataset"]["trace_count"] == 1
