# ABOUTME: Validates Phase 7 benchmark runner for baseline-vs-RLM comparisons on frozen datasets.
# ABOUTME: Ensures RCA, compliance, and incident metrics are produced from one deterministic Parquet/manifest input.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from investigator.proof.benchmark import run_dataset_benchmark


def _write_fixture_dataset(parquet_path: Path, manifest_path: Path) -> None:
    rows = [
        {
            "name": "agent.run",
            "span_kind": "AGENT",
            "parent_id": None,
            "start_time": pd.Timestamp("2026-02-10T00:00:00Z"),
            "end_time": pd.Timestamp("2026-02-10T00:00:02Z"),
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
            "end_time": pd.Timestamp("2026-02-10T00:00:02Z"),
            "status_code": "ERROR",
            "status_message": "forced tool timeout",
            "events": [],
            "context.span_id": "tool-1",
            "context.trace_id": "trace-1",
            "attributes.phase1": {"run_id": "seed_run_0000", "step": "tool.call"},
            "attributes.http": None,
        },
        {
            "name": "agent.run",
            "span_kind": "AGENT",
            "parent_id": None,
            "start_time": pd.Timestamp("2026-02-10T00:03:00Z"),
            "end_time": pd.Timestamp("2026-02-10T00:03:04Z"),
            "status_code": "UNSET",
            "status_message": "",
            "events": [],
            "context.span_id": "root-2",
            "context.trace_id": "trace-2",
            "attributes.phase1": {"run_id": "seed_run_0001", "step": None, "project": "phase7-proof"},
            "attributes.http": None,
        },
        {
            "name": "retriever.fetch",
            "span_kind": "RETRIEVER",
            "parent_id": "root-2",
            "start_time": pd.Timestamp("2026-02-10T00:03:01Z"),
            "end_time": pd.Timestamp("2026-02-10T00:03:04Z"),
            "status_code": "OK",
            "status_message": "",
            "events": [],
            "context.span_id": "retr-2",
            "context.trace_id": "trace-2",
            "attributes.phase1": {
                "run_id": "seed_run_0001",
                "step": "retriever.fetch",
                "retrieval": {"documents": []},
            },
            "attributes.http": None,
        },
        {
            "name": "agent.run",
            "span_kind": "AGENT",
            "parent_id": None,
            "start_time": pd.Timestamp("2026-02-10T00:05:00Z"),
            "end_time": pd.Timestamp("2026-02-10T00:05:02Z"),
            "status_code": "UNSET",
            "status_message": "",
            "events": [],
            "context.span_id": "root-3",
            "context.trace_id": "trace-3",
            "attributes.phase1": {"run_id": "seed_run_0002", "step": None, "project": "phase7-proof"},
            "attributes.http": None,
        },
        {
            "name": "tool.parse",
            "span_kind": "TOOL",
            "parent_id": "root-3",
            "start_time": pd.Timestamp("2026-02-10T00:05:00Z"),
            "end_time": pd.Timestamp("2026-02-10T00:05:02Z"),
            "status_code": "ERROR",
            "status_message": "schema mismatch",
            "events": [],
            "context.span_id": "tool-3",
            "context.trace_id": "trace-3",
            "attributes.phase1": {"run_id": "seed_run_0002", "step": "tool.parse"},
            "attributes.http": None,
        },
        {
            "name": "agent.run",
            "span_kind": "AGENT",
            "parent_id": None,
            "start_time": pd.Timestamp("2026-02-10T00:07:00Z"),
            "end_time": pd.Timestamp("2026-02-10T00:07:02Z"),
            "status_code": "UNSET",
            "status_message": "",
            "events": [],
            "context.span_id": "root-4",
            "context.trace_id": "trace-4",
            "attributes.phase1": {"run_id": "seed_run_0003", "step": None, "project": "phase7-proof"},
            "attributes.http": None,
        },
        {
            "name": "llm.generate",
            "span_kind": "UNKNOWN",
            "parent_id": "root-4",
            "start_time": pd.Timestamp("2026-02-10T00:07:00Z"),
            "end_time": pd.Timestamp("2026-02-10T00:07:02Z"),
            "status_code": "ERROR",
            "status_message": "format drift",
            "events": [],
            "context.span_id": "llm-4",
            "context.trace_id": "trace-4",
            "attributes.phase1": {"run_id": "seed_run_0003", "step": "llm.generate"},
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
            {"run_id": "seed_run_0001", "trace_id": None, "expected_label": "retrieval_failure"},
            {"run_id": "seed_run_0002", "trace_id": None, "expected_label": "data_schema_mismatch"},
            {"run_id": "seed_run_0003", "trace_id": None, "expected_label": "instruction_failure"},
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_controls(controls_dir: Path) -> None:
    controls_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "controls": [
            {
                "control_id": "control.execution.hard_failures",
                "controls_version": "controls-v1",
                "severity": "high",
                "required_evidence": [],
                "violation_patterns": [
                    "forced tool timeout",
                    "upstream unavailable",
                    "schema mismatch",
                ],
                "remediation_template": "Address hard-failure signals before approval.",
                "applies_when": {"app_types": [], "tools": [], "data_domains": []},
            },
            {
                "control_id": "control.instruction.format.review",
                "controls_version": "controls-v1",
                "severity": "medium",
                "required_evidence": ["required_messages"],
                "remediation_template": "Collect message evidence for format-drift traces before approval.",
                "applies_when": {"app_types": [], "tools": ["llm.generate"], "data_domains": []},
            }
        ]
    }
    (controls_dir / "controls_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


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


def test_run_dataset_benchmark_returns_three_capability_comparisons(tmp_path: Path) -> None:
    parquet_path = tmp_path / "spans.parquet"
    manifest_path = tmp_path / "manifest.json"
    controls_dir = tmp_path / "controls"
    snapshots_dir = tmp_path / "snapshots"
    artifacts_root = tmp_path / "artifacts" / "investigator_runs"

    _write_fixture_dataset(parquet_path, manifest_path)
    _write_controls(controls_dir)
    _write_snapshots(snapshots_dir)

    report = run_dataset_benchmark(
        spans_parquet_path=parquet_path,
        manifest_path=manifest_path,
        controls_version="controls-v1",
        controls_dir=controls_dir,
        snapshots_dir=snapshots_dir,
        project_name="phase7-proof",
        artifacts_root=artifacts_root,
    )

    assert report["dataset"]["trace_count"] == 4
    assert set(report["capabilities"].keys()) == {"rca", "compliance", "incident"}

    rca = report["capabilities"]["rca"]
    assert rca["baseline"]["accuracy"] <= rca["rlm"]["accuracy"]
    assert rca["delta"]["accuracy"] == rca["rlm"]["accuracy"] - rca["baseline"]["accuracy"]
    rca_diagnostics = rca["diagnostics"]
    assert rca_diagnostics["dataset_hash"] == report["dataset"]["dataset_hash"]
    assert rca_diagnostics["per_label"]
    assert set(rca_diagnostics["per_label"].keys()) == {
        "data_schema_mismatch",
        "instruction_failure",
        "retrieval_failure",
        "tool_failure",
    }
    for label, row in rca_diagnostics["per_label"].items():
        assert row["label"] == label
        assert row["support"] >= 0
        assert 0.0 <= row["baseline_accuracy"] <= 1.0
        assert 0.0 <= row["rlm_accuracy"] <= 1.0


    compliance = report["capabilities"]["compliance"]
    assert compliance["sample_count"] == 4
    assert compliance["rlm"]["accuracy"] > compliance["baseline"]["accuracy"]
    assert compliance["delta"]["accuracy"] >= 0.05

    incident = report["capabilities"]["incident"]
    assert 0.0 <= incident["baseline"]["overlap_at_k"] <= 1.0
    assert 0.0 <= incident["rlm"]["overlap_at_k"] <= 1.0
    assert incident["delta"]["overlap_at_k"] == (
        incident["rlm"]["overlap_at_k"] - incident["baseline"]["overlap_at_k"]
    )
    diagnostics = incident["diagnostics"]
    assert diagnostics["k"] == len(diagnostics["baseline"]["expected_ids"])
    assert diagnostics["baseline"]["expected_ids"]
    assert diagnostics["baseline"]["predicted_ids"]
    assert "intersection" in diagnostics["baseline"]
    assert diagnostics["rlm"]["expected_ids"] == diagnostics["baseline"]["expected_ids"]

    gates = report["gates"]
    assert set(gates["thresholds"].keys()) == {"rca", "compliance", "incident"}
    assert set(gates["results"].keys()) == {"rca", "compliance", "incident"}
    assert isinstance(gates["all_passed"], bool)
    rca_calibration = report["capabilities"]["rca"]["diagnostics"]["threshold_calibration"]
    assert rca_calibration["threshold"] == gates["thresholds"]["rca"]
    assert rca_calibration["threshold_miss"] == (rca["delta"]["accuracy"] < gates["thresholds"]["rca"])
    assert "concentrated_in_label_family" in rca_calibration["miss_concentration"]
    assert "top_label" in rca_calibration["miss_concentration"]

    assert report["run_artifacts"]["rca"]
    assert report["run_artifacts"]["compliance"]
    assert report["run_artifacts"]["incident"]
