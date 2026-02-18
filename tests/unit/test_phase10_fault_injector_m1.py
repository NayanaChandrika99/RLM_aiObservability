# ABOUTME: Tests Milestone 1 fault-injector orchestration for seeded trace generation and manifest updates.
# ABOUTME: Verifies profile validation, trace-id resolution wiring, and batch manifest persistence behavior.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pandas as pd

from apps.demo_agent import fault_injector, run_seeded_failures


def test_run_with_fault_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unsupported fault profile"):
        fault_injector.run_with_fault(fault_profile="profile_unknown", run_id="seed_run_0000")


def test_run_with_fault_emits_case_and_returns_trace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_emit(manifest, *, project_name: str, endpoint: str | None = None) -> None:
        captured["manifest"] = manifest
        captured["project_name"] = project_name
        captured["endpoint"] = endpoint

    def fake_lookup(*, endpoint: str, project_name: str, run_ids: list[str], limit: int = 100000):
        captured["lookup"] = {
            "endpoint": endpoint,
            "project_name": project_name,
            "run_ids": list(run_ids),
            "limit": limit,
        }
        return {run_ids[0]: "trace_abc123"}

    monkeypatch.setattr(fault_injector, "emit_seeded_traces", fake_emit)
    monkeypatch.setattr(fault_injector, "_lookup_trace_ids_by_run_id", fake_lookup)

    def fake_live_unavailable(*, fault_profile: str, run_id: str, phoenix_endpoint: str, project_name: str) -> str:
        raise fault_injector.LiveFaultInjectionUnavailableError("llama-index unavailable for fallback branch test")

    monkeypatch.setattr(fault_injector, "_run_live_llamaindex_fault", fake_live_unavailable)

    trace_id = fault_injector.run_with_fault(
        fault_profile="profile_tool_failure",
        run_id="seed_run_0000",
        phoenix_endpoint="http://127.0.0.1:6006",
        project_name="phase1-seeded-failures",
    )

    assert trace_id == "trace_abc123"
    manifest = captured["manifest"]
    assert isinstance(manifest, dict)
    cases = manifest["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 1
    assert cases[0]["run_id"] == "seed_run_0000"
    assert cases[0]["expected_label"] == "tool_failure"
    assert captured["project_name"] == "phase1-seeded-failures"
    assert captured["lookup"] == {
        "endpoint": "http://127.0.0.1:6006",
        "project_name": "phase1-seeded-failures",
        "run_ids": ["seed_run_0000"],
        "limit": 100000,
    }


def test_run_all_seeded_failures_updates_manifest_and_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
                        "fault_profile": "profile_tool_failure",
                        "notes": "case 0",
                    },
                    {
                        "run_id": "seed_run_0001",
                        "trace_id": None,
                        "expected_label": "retrieval_failure",
                        "fault_profile": "profile_retrieval_failure",
                        "notes": "case 1",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    returned = {
        "seed_run_0000": "trace_0",
        "seed_run_0001": "trace_1",
    }

    def fake_run_with_fault(
        *,
        fault_profile: str,
        run_id: str,
        phoenix_endpoint: str,
        project_name: str,
        lookup_limit: int,
        live_only: bool,
    ) -> str:
        assert fault_profile in {"profile_tool_failure", "profile_retrieval_failure"}
        assert project_name == "phase1-seeded-failures"
        assert lookup_limit == 100000
        assert live_only is False
        return returned[run_id]

    export_calls: dict[str, object] = {}

    def fake_export(*, endpoint: str | None, project_name: str, output_path: Path, limit: int = 100000) -> int:
        export_calls["endpoint"] = endpoint
        export_calls["project_name"] = project_name
        export_calls["output_path"] = output_path
        export_calls["limit"] = limit
        return 42

    monkeypatch.setattr(fault_injector, "run_with_fault", fake_run_with_fault)
    monkeypatch.setattr(fault_injector, "export_spans_to_parquet", fake_export)

    mapping = fault_injector.run_all_seeded_failures(
        manifest_path=str(manifest_path),
        phoenix_endpoint="http://127.0.0.1:6006",
        project_name="phase1-seeded-failures",
        export_path=tmp_path / "spans.parquet",
    )

    assert mapping == returned

    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [case["trace_id"] for case in updated_manifest["cases"]] == ["trace_0", "trace_1"]
    assert export_calls["project_name"] == "phase1-seeded-failures"
    assert export_calls["output_path"] == tmp_path / "spans.parquet"


def test_run_all_seeded_failures_passes_live_only_to_run_with_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
                        "fault_profile": "profile_tool_failure",
                        "notes": "case 0",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    call_args: dict[str, object] = {}

    def fake_run_with_fault(
        *,
        fault_profile: str,
        run_id: str,
        phoenix_endpoint: str,
        project_name: str,
        lookup_limit: int,
        live_only: bool,
    ) -> str:
        call_args["fault_profile"] = fault_profile
        call_args["run_id"] = run_id
        call_args["phoenix_endpoint"] = phoenix_endpoint
        call_args["project_name"] = project_name
        call_args["lookup_limit"] = lookup_limit
        call_args["live_only"] = live_only
        return "trace_live_only"

    monkeypatch.setattr(fault_injector, "run_with_fault", fake_run_with_fault)
    monkeypatch.setattr(
        fault_injector,
        "export_spans_to_parquet",
        lambda *, endpoint, project_name, output_path, limit=100000: 1,
    )

    mapping = fault_injector.run_all_seeded_failures(
        manifest_path=str(manifest_path),
        phoenix_endpoint="http://127.0.0.1:6006",
        project_name="phase1-seeded-failures",
        export_path=tmp_path / "spans.parquet",
        live_only=True,
    )

    assert mapping == {"seed_run_0000": "trace_live_only"}
    assert call_args == {
        "fault_profile": "profile_tool_failure",
        "run_id": "seed_run_0000",
        "phoenix_endpoint": "http://127.0.0.1:6006",
        "project_name": "phase1-seeded-failures",
        "lookup_limit": 100000,
        "live_only": True,
    }


def test_run_seeded_failures_cli_passes_live_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_all(
        *,
        manifest_path: str,
        phoenix_endpoint: str,
        project_name: str,
        export_path: Path,
        lookup_limit: int,
        live_only: bool,
    ) -> dict[str, str]:
        captured["manifest_path"] = manifest_path
        captured["phoenix_endpoint"] = phoenix_endpoint
        captured["project_name"] = project_name
        captured["export_path"] = export_path
        captured["lookup_limit"] = lookup_limit
        captured["live_only"] = live_only
        return {"seed_run_0000": "trace_live_123"}

    monkeypatch.setattr(run_seeded_failures, "run_all_seeded_failures", fake_run_all)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_seeded_failures.py",
            "--manifest",
            "datasets/seeded_failures/manifest.json",
            "--phoenix-endpoint",
            "http://127.0.0.1:6006",
            "--project-name",
            "phase1-seeded-failures",
            "--export-path",
            "datasets/seeded_failures/exports/spans.parquet",
            "--lookup-limit",
            "100000",
            "--live-only",
        ],
    )

    run_seeded_failures.main()

    assert captured == {
        "manifest_path": "datasets/seeded_failures/manifest.json",
        "phoenix_endpoint": "http://127.0.0.1:6006",
        "project_name": "phase1-seeded-failures",
        "export_path": Path("datasets/seeded_failures/exports/spans.parquet"),
        "lookup_limit": 100000,
        "live_only": True,
    }


def test_lookup_trace_ids_supports_nested_phase1_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = pd.DataFrame(
        {
            "attributes.phase1": [
                {"run_id": "seed_run_0000", "project": "phase1-seeded-failures"},
                {"run_id": "seed_run_9999", "project": "phase1-seeded-failures"},
            ],
            "context.trace_id": ["trace_a", "trace_b"],
            "start_time": [2, 1],
        }
    )

    class FakeClient:
        def __init__(self, *, endpoint: str) -> None:
            self.endpoint = endpoint

        def get_spans_dataframe(self, *, project_name: str, limit: int):
            assert project_name == "phase1-seeded-failures"
            assert limit == 100000
            return dataframe

    monkeypatch.setattr(fault_injector, "Client", FakeClient)

    mapping = fault_injector._lookup_trace_ids_by_run_id(
        endpoint="http://127.0.0.1:6006",
        project_name="phase1-seeded-failures",
        run_ids=["seed_run_0000"],
    )

    assert mapping == {"seed_run_0000": "trace_a"}


def test_resolve_trace_id_with_retry_eventually_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"value": 0}

    def fake_lookup(*, endpoint: str, project_name: str, run_ids: list[str], limit: int = 100000):
        call_count["value"] += 1
        if call_count["value"] < 3:
            return {}
        return {run_ids[0]: "trace_retry_ok"}

    monkeypatch.setattr(fault_injector, "_lookup_trace_ids_by_run_id", fake_lookup)
    monkeypatch.setattr(fault_injector.time, "sleep", lambda *_args, **_kwargs: None)

    trace_id = fault_injector._resolve_trace_id_with_retry(
        endpoint="http://127.0.0.1:6006",
        project_name="phase1-seeded-failures",
        run_id="seed_run_0000",
        limit=100000,
        attempts=5,
        sleep_sec=0.01,
    )

    assert trace_id == "trace_retry_ok"
    assert call_count["value"] == 3
