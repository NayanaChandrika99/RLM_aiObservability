# ABOUTME: Tests Milestone 1 fault-injector orchestration for seeded trace generation and manifest updates.
# ABOUTME: Verifies profile validation, trace-id resolution wiring, and batch manifest persistence behavior.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.demo_agent import fault_injector


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
    ) -> str:
        assert fault_profile in {"profile_tool_failure", "profile_retrieval_failure"}
        assert project_name == "phase1-seeded-failures"
        assert lookup_limit == 100000
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
