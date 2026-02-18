# ABOUTME: Validates the Phase 10 RCA CLI for single-trace and manifest batch execution paths.
# ABOUTME: Ensures budget/writeback wiring, summary output, and exit codes are deterministic.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from investigator.rca import cli


class _FakeReport:
    def __init__(self, trace_id: str, primary_label: str) -> None:
        self.trace_id = trace_id
        self.primary_label = primary_label

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "primary_label": self.primary_label,
            "summary": "fake-summary",
            "confidence": 0.7,
            "evidence_refs": [],
            "remediation": [],
            "gaps": [],
            "schema_version": "1.0.0",
        }


class _FakeRunRecord:
    def __init__(self, *, run_id: str, status: str) -> None:
        self.run_id = run_id
        self.status = status
        self.runtime_ref = SimpleNamespace(repl_trajectory=[])

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "runtime_ref": {"repl_trajectory": []},
        }


def test_rca_cli_single_trace_prints_report_and_uses_no_writeback(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "PhoenixInspectionAPI", lambda endpoint: {"endpoint": endpoint})
    captured: dict[str, object] = {}

    def _fake_workflow(**kwargs):  # noqa: ANN003
        captured["engine_model_name"] = kwargs["engine"].model_name
        captured["writeback_client"] = kwargs["writeback_client"]
        return _FakeReport("trace-1", "tool_failure"), _FakeRunRecord(
            run_id="run-single",
            status="succeeded",
        )

    monkeypatch.setattr(cli, "run_trace_rca_workflow", _fake_workflow)

    exit_code = cli.main(["--trace-id", "trace-1", "--no-writeback"])

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["trace_id"] == "trace-1"
    assert payload["primary_label"] == "tool_failure"
    assert captured["engine_model_name"] == "gpt-4o-mini"
    assert captured["writeback_client"] is not None


def test_rca_cli_single_trace_returns_nonzero_on_failed_run(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "PhoenixInspectionAPI", lambda endpoint: {"endpoint": endpoint})

    def _fake_workflow(**kwargs):  # noqa: ANN003
        del kwargs
        raise RuntimeError(
            {
                "run_id": "run-failed",
                "status": "failed",
                "error": {"code": "MODEL_OUTPUT_INVALID"},
            }
        )

    monkeypatch.setattr(cli, "run_trace_rca_workflow", _fake_workflow)

    exit_code = cli.main(["--trace-id", "trace-failed", "--no-writeback"])

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "run-failed" in stderr


def test_rca_cli_single_trace_verbose_prints_repl_code_on_failed_run(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "PhoenixInspectionAPI", lambda endpoint: {"endpoint": endpoint})

    def _fake_workflow(**kwargs):  # noqa: ANN003
        del kwargs
        raise RuntimeError(
            {
                "run_id": "run-failed",
                "status": "failed",
                "error": {"code": "SUBMIT_DEADLINE_REACHED"},
                "runtime_ref": {
                    "repl_trajectory": [
                        {
                            "reasoning": "inspect before submit",
                            "code": "spans = list_spans(trace_id)\nprint(len(spans))",
                            "output": "42",
                        }
                    ]
                },
            }
        )

    monkeypatch.setattr(cli, "run_trace_rca_workflow", _fake_workflow)

    exit_code = cli.main(["--trace-id", "trace-failed", "--no-writeback", "--verbose"])

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "[step 1] code=spans = list_spans(trace_id)" in stderr
    assert "[step 1] output=42" in stderr


def test_rca_cli_batch_prints_summary_for_non_null_trace_ids(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "seeded_failures_v1",
                "cases": [
                    {"trace_id": "trace-1", "expected_label": "tool_failure"},
                    {"trace_id": None, "expected_label": "instruction_failure"},
                    {"trace_id": "trace-2", "expected_label": "retrieval_failure"},
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "PhoenixInspectionAPI", lambda endpoint: {"endpoint": endpoint})
    calls: list[str] = []

    def _fake_workflow(**kwargs):  # noqa: ANN003
        request = kwargs["request"]
        calls.append(request.trace_id)
        if request.trace_id == "trace-1":
            return _FakeReport("trace-1", "tool_failure"), _FakeRunRecord(
                run_id="run-1",
                status="succeeded",
            )
        return _FakeReport("trace-2", "instruction_failure"), _FakeRunRecord(
            run_id="run-2",
            status="succeeded",
        )

    monkeypatch.setattr(cli, "run_trace_rca_workflow", _fake_workflow)

    exit_code = cli.main(["--manifest", str(manifest_path), "--no-writeback"])

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert calls == ["trace-1", "trace-2"]
    assert "run_id" in stdout
    assert "trace-1" in stdout
    assert "trace-2" in stdout
    assert "None" not in stdout


def test_rca_cli_batch_parquet_mode_attaches_manifest_trace_ids(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "seeded_failures_v1",
                "cases": [
                    {
                        "run_id": "seed_run_0000",
                        "trace_id": None,
                        "expected_label": "tool_failure",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    class _FakeParquetAPI:
        def __init__(self) -> None:
            self.attach_calls = 0

        def attach_manifest_trace_ids(self, *, manifest_path: str | Path) -> None:
            self.attach_calls += 1
            payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            payload["cases"][0]["trace_id"] = "trace-from-parquet"
            Path(manifest_path).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    fake_api = _FakeParquetAPI()
    monkeypatch.setattr(cli, "ParquetInspectionAPI", lambda parquet_path: fake_api)
    seen_trace_ids: list[str] = []

    def _fake_workflow(**kwargs):  # noqa: ANN003
        request = kwargs["request"]
        seen_trace_ids.append(request.trace_id)
        return _FakeReport(request.trace_id, "tool_failure"), _FakeRunRecord(
            run_id="run-parquet",
            status="succeeded",
        )

    monkeypatch.setattr(cli, "run_trace_rca_workflow", _fake_workflow)

    exit_code = cli.main(
        [
            "--manifest",
            str(manifest_path),
            "--parquet",
            str(tmp_path / "spans.parquet"),
            "--no-writeback",
        ]
    )

    assert exit_code == 0
    assert fake_api.attach_calls == 1
    assert seen_trace_ids == ["trace-from-parquet"]


def test_rca_cli_build_engine_disables_deterministic_fallback_by_default() -> None:
    engine = cli._build_engine(inspection_api={"endpoint": "http://127.0.0.1:6006"}, model_name="gpt-4o-mini")

    assert engine._use_repl_runtime is True
    assert engine._fallback_on_llm_error is False


def test_rca_cli_build_engine_sets_trace_rca_tips_profile() -> None:
    engine = cli._build_engine(
        inspection_api={"endpoint": "http://127.0.0.1:6006"},
        model_name="gpt-4o-mini",
        tips_profile="trace_rca_v1",
    )

    assert "Prefer branch-root evidence first" in str(getattr(engine, "_repl_env_tips", ""))


def test_rca_cli_scaffold_heuristic_creates_engine_without_llm(monkeypatch) -> None:
    monkeypatch.setattr(cli, "PhoenixInspectionAPI", lambda endpoint: {"endpoint": endpoint})

    engine = cli._build_engine(
        inspection_api={"endpoint": "http://127.0.0.1:6006"},
        model_name="gpt-4o-mini",
        scaffold_name="heuristic",
    )

    assert engine._use_llm_judgment is False
    assert engine._use_repl_runtime is False


def test_rca_cli_scaffold_rlm_tips_creates_engine_with_repl_and_tips(monkeypatch) -> None:
    monkeypatch.setattr(cli, "PhoenixInspectionAPI", lambda endpoint: {"endpoint": endpoint})

    engine = cli._build_engine(
        inspection_api={"endpoint": "http://127.0.0.1:6006"},
        model_name="gpt-4o-mini",
        scaffold_name="rlm_tips",
    )

    assert engine._use_llm_judgment is True
    assert engine._use_repl_runtime is True
    assert "Prefer branch-root evidence first" in str(getattr(engine, "_repl_env_tips", ""))


def test_rca_cli_scaffold_sets_run_record_scaffold(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "PhoenixInspectionAPI", lambda endpoint: {"endpoint": endpoint})
    captured_scaffold: list[str | None] = []

    def _fake_workflow(**kwargs):  # noqa: ANN003
        captured_scaffold.append(kwargs.get("scaffold"))
        return _FakeReport("trace-1", "tool_failure"), _FakeRunRecord(
            run_id="run-scaffold",
            status="succeeded",
        )

    monkeypatch.setattr(cli, "run_trace_rca_workflow", _fake_workflow)

    exit_code = cli.main(["--trace-id", "trace-1", "--no-writeback", "--scaffold", "heuristic"])

    assert exit_code == 0
    assert captured_scaffold == ["heuristic"]


def test_rca_cli_verbose_repl_trajectory_prints_code_and_output(capsys) -> None:
    run_record = _FakeRunRecord(run_id="run-verbose", status="succeeded")
    run_record.runtime_ref = SimpleNamespace(
        repl_trajectory=[
            {
                "reasoning": "Inspect and finalize.",
                "code": "spans = list_spans(trace_id)\nSUBMIT(primary_label='tool_failure')",
                "output": "done",
            }
        ]
    )

    cli._print_repl_trajectory(run_record)

    stderr = capsys.readouterr().err
    assert "REPL trajectory (1 step(s))" in stderr
    assert "[step 1] reasoning=Inspect and finalize." in stderr
    assert "[step 1] code=spans = list_spans(trace_id)" in stderr
    assert "SUBMIT(primary_label='tool_failure')" in stderr
    assert "[step 1] output=done" in stderr
