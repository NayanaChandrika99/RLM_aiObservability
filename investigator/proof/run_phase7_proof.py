# ABOUTME: Orchestrates Phase 7 proof runs from live trace generation through frozen-dataset benchmarking.
# ABOUTME: Persists reproducible proof artifacts so baseline-vs-RLM claims are auditable and rerunnable.

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv
import requests

from investigator.proof.benchmark import DEFAULT_DELTA_THRESHOLDS, run_dataset_benchmark


DEFAULT_PHOENIX_BASE_URL = "http://127.0.0.1:6006"
DEFAULT_DATASET_ID = "seeded_failures_v1"
DEFAULT_SEEDED_PROJECT = "phase7-seeded-failures"
DEFAULT_PROOF_PROJECT = "phase7-proof"
DEFAULT_CONTROLS_VERSION = "controls-v1"


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _assert_phoenix_reachable(base_url: str) -> None:
    response = requests.get(base_url.rstrip("/"), timeout=4)
    response.raise_for_status()


def _parse_json_from_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise RuntimeError("Expected JSON output but command produced no stdout.")
    lines = text.splitlines()
    for start in range(len(lines)):
        candidate = "\n".join(lines[start:])
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"Failed to parse JSON output from command:\n{text}")


def _run_python_module(module_name: str, *, env: dict[str, str]) -> dict[str, Any]:
    command = [sys.executable, "-m", module_name]
    completed = subprocess.run(
        command,
        env=env,
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    return _parse_json_from_stdout(completed.stdout)


def _ensure_default_controls(controls_dir: Path, controls_version: str) -> Path:
    controls_dir.mkdir(parents=True, exist_ok=True)
    controls_file = controls_dir / "controls_v1.json"
    if controls_file.exists():
        return controls_file
    payload = {
        "controls": [
            {
                "control_id": "control.execution.hard_failures",
                "controls_version": controls_version,
                "severity": "high",
                "required_evidence": [],
                "violation_patterns": [
                    "forced tool timeout",
                    "upstream unavailable",
                    "schema mismatch",
                ],
                "remediation_template": "Address hard-failure execution signals before approval.",
                "applies_when": {"app_types": [], "tools": [], "data_domains": []},
            },
            {
                "control_id": "control.instruction.format.review",
                "controls_version": controls_version,
                "severity": "medium",
                "required_evidence": ["required_messages"],
                "remediation_template": "Collect message evidence for format-drift traces before approval.",
                "applies_when": {
                    "app_types": [],
                    "tools": ["llm.generate"],
                    "data_domains": [],
                },
            },
            {
                "control_id": "control.upstream.http.review",
                "controls_version": controls_version,
                "severity": "high",
                "required_evidence": ["required_error_span", "required_span_attributes"],
                "violation_patterns": [
                    "upstream unavailable",
                    "http\\.status_code[^0-9]*503",
                    "503",
                ],
                "remediation_template": "Confirm upstream dependency health and add resilience for HTTP 503 paths.",
                "applies_when": {"app_types": [], "tools": ["dependency.http"], "data_domains": []},
            },
            {
                "control_id": "control.schema.mismatch.review",
                "controls_version": controls_version,
                "severity": "high",
                "required_evidence": ["required_error_span"],
                "violation_patterns": ["schema mismatch", "tool\\.parse"],
                "remediation_template": "Align parser expectations with tool schema and validate payload boundaries.",
                "applies_when": {"app_types": [], "tools": ["tool.parse"], "data_domains": []},
            },
            {
                "control_id": "control.output.format.review",
                "controls_version": controls_version,
                "severity": "medium",
                "required_evidence": ["required_messages", "required_span_attributes"],
                "violation_patterns": [
                    "format drift",
                    "phase1\\.output\\.format[^A-Za-z0-9]*unexpected",
                ],
                "remediation_template": "Capture format evidence and tighten output constraints before approval.",
                "applies_when": {"app_types": [], "tools": ["llm.generate"], "data_domains": []},
            },
            {
                "control_id": "control.retrieval.quality.review",
                "controls_version": controls_version,
                "severity": "medium",
                "required_evidence": ["required_retrieval_chunks", "required_span_attributes"],
                "violation_patterns": [
                    "phase1\\.retrieval\\.relevance[^0-9]*(0\\.[0-2][0-9]?|0\\.11)",
                ],
                "remediation_template": "Collect retrieval chunk evidence and verify relevance for approval.",
                "applies_when": {"app_types": [], "tools": ["retriever.fetch"], "data_domains": []},
            },
        ]
    }
    _ensure_json(controls_file, payload)
    return controls_file


def _ensure_default_snapshots(snapshots_root: Path) -> tuple[Path, Path]:
    baseline = snapshots_root / "phase7-baseline"
    candidate = snapshots_root / "phase7-candidate"
    baseline.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(parents=True, exist_ok=True)

    baseline_metadata = {
        "created_at": "2026-02-10T00:00:00Z",
        "git_commit": "phase7-baseline",
        "tag": "baseline",
    }
    candidate_metadata = {
        "created_at": "2026-02-10T00:30:00Z",
        "git_commit": "phase7-candidate",
        "tag": "candidate",
    }
    _ensure_json(baseline / "metadata.json", baseline_metadata)
    _ensure_json(candidate / "metadata.json", candidate_metadata)

    baseline_env = baseline / "runtime.env"
    candidate_env = candidate / "runtime.env"
    if not baseline_env.exists():
        baseline_env.write_text("MODEL=gpt-5-mini\n", encoding="utf-8")
    if not candidate_env.exists():
        candidate_env.write_text("MODEL=gpt-5-mini\nFEATURE_FLAG=phase7\n", encoding="utf-8")
    return baseline, candidate


def run_frozen_dataset_proof(
    *,
    proof_run_id: str,
    spans_parquet_path: str | Path,
    manifest_path: str | Path,
    project_name: str,
    controls_version: str,
    controls_dir: str | Path,
    snapshots_dir: str | Path,
    proof_artifacts_root: str | Path = "artifacts/proof_runs",
    evaluator_artifacts_root: str | Path = "artifacts/investigator_runs",
    delta_thresholds: dict[str, float] | None = None,
    enforce_thresholds: bool = False,
) -> dict[str, Any]:
    benchmark = run_dataset_benchmark(
        spans_parquet_path=spans_parquet_path,
        manifest_path=manifest_path,
        controls_version=controls_version,
        controls_dir=controls_dir,
        snapshots_dir=snapshots_dir,
        project_name=project_name,
        artifacts_root=evaluator_artifacts_root,
        delta_thresholds=delta_thresholds,
    )
    report = {
        "proof_run_id": proof_run_id,
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": benchmark["dataset"],
        "capabilities": benchmark["capabilities"],
        "gates": benchmark["gates"],
        "run_artifacts": benchmark["run_artifacts"],
    }
    output_path = Path(proof_artifacts_root) / proof_run_id / "proof_report.json"
    _ensure_json(output_path, report)
    report["proof_report_path"] = str(output_path)
    if enforce_thresholds and not bool(report["gates"]["all_passed"]):
        raise RuntimeError(
            f"Proof thresholds failed for {proof_run_id}. See {output_path} for gate details."
        )
    return report


def run_phase7_proof() -> dict[str, Any]:
    repo_root = Path.cwd()
    load_dotenv(repo_root / ".env", override=False)

    phoenix_base_url = os.getenv("PHOENIX_BASE_URL", DEFAULT_PHOENIX_BASE_URL).rstrip("/")
    _assert_phoenix_reachable(phoenix_base_url)
    env = os.environ.copy()
    env["PHOENIX_BASE_URL"] = phoenix_base_url
    env["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_base_url

    agent_summary = _run_python_module("apps.demo_agent.phase1_tutorial_run", env=env)
    seeded_summary = _run_python_module("apps.demo_agent.phase1_seeded_failures", env=env)

    manifest_path = Path(str(seeded_summary["manifest_path"]))
    spans_parquet_path = Path(str(seeded_summary["parquet_path"]))
    exported_rows = int(seeded_summary.get("rows_exported") or 0)
    if exported_rows <= 0:
        raise RuntimeError(
            "Seeded dataset export produced zero rows; proof run aborted to avoid stale comparisons."
        )

    controls_dir = repo_root / "controls" / "library"
    snapshots_dir = repo_root / "configs" / "snapshots"
    _ensure_default_controls(controls_dir, DEFAULT_CONTROLS_VERSION)
    _ensure_default_snapshots(snapshots_dir)

    enforce_thresholds = os.getenv("PHASE7_ENFORCE_THRESHOLDS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    proof_run_id = f"phase7-proof-{_utc_timestamp()}"
    report = run_frozen_dataset_proof(
        proof_run_id=proof_run_id,
        spans_parquet_path=spans_parquet_path,
        manifest_path=manifest_path,
        project_name=str(seeded_summary.get("project_name") or DEFAULT_SEEDED_PROJECT),
        controls_version=DEFAULT_CONTROLS_VERSION,
        controls_dir=controls_dir,
        snapshots_dir=snapshots_dir,
        proof_artifacts_root=repo_root / "artifacts" / "proof_runs",
        evaluator_artifacts_root=repo_root / "artifacts" / "investigator_runs",
        delta_thresholds=DEFAULT_DELTA_THRESHOLDS,
        enforce_thresholds=False,
    )
    report["agent_trace_summary"] = agent_summary
    report["seeded_export_rows"] = exported_rows

    summary_path = Path(report["proof_report_path"])
    _ensure_json(summary_path, report)
    if enforce_thresholds and not bool(report["gates"]["all_passed"]):
        raise RuntimeError(
            f"Proof thresholds failed for {proof_run_id}. See {summary_path} for gate details."
        )
    return report


def main() -> None:
    report = run_phase7_proof()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
