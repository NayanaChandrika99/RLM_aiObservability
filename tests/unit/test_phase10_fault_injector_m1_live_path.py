# ABOUTME: Tests live LlamaIndex fault path orchestration and controlled fallback behavior.
# ABOUTME: Ensures run_with_fault prioritizes live execution and only falls back when needed.

from __future__ import annotations

import pytest

from apps.demo_agent import fault_injector


def test_run_with_fault_prefers_live_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_live_run(*, fault_profile: str, run_id: str, phoenix_endpoint: str, project_name: str) -> str:
        calls.append(f"live:{fault_profile}:{run_id}:{phoenix_endpoint}:{project_name}")
        return "trace_live_123"

    def fake_seeded_fallback(*args, **kwargs):
        raise AssertionError("seeded fallback should not be used when live path succeeds")

    monkeypatch.setattr(fault_injector, "_run_live_llamaindex_fault", fake_live_run)
    monkeypatch.setattr(fault_injector, "_run_seeded_fallback_fault", fake_seeded_fallback)

    trace_id = fault_injector.run_with_fault(
        fault_profile="profile_tool_failure",
        run_id="seed_run_0000",
        phoenix_endpoint="http://127.0.0.1:6006",
        project_name="phase1-seeded-failures",
    )

    assert trace_id == "trace_live_123"
    assert calls == ["live:profile_tool_failure:seed_run_0000:http://127.0.0.1:6006:phase1-seeded-failures"]


def test_run_with_fault_falls_back_when_live_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_live_run(*, fault_profile: str, run_id: str, phoenix_endpoint: str, project_name: str) -> str:
        raise fault_injector.LiveFaultInjectionUnavailableError("llama-index unavailable")

    fallback_calls: list[str] = []

    def fake_seeded_fallback(
        *,
        fault_profile: str,
        run_id: str,
        phoenix_endpoint: str,
        project_name: str,
        lookup_limit: int,
    ) -> str:
        fallback_calls.append(f"fallback:{fault_profile}:{run_id}:{lookup_limit}")
        return "trace_fallback_321"

    monkeypatch.setattr(fault_injector, "_run_live_llamaindex_fault", fake_live_run)
    monkeypatch.setattr(fault_injector, "_run_seeded_fallback_fault", fake_seeded_fallback)

    trace_id = fault_injector.run_with_fault(
        fault_profile="profile_retrieval_failure",
        run_id="seed_run_0001",
        phoenix_endpoint="http://127.0.0.1:6006",
        project_name="phase1-seeded-failures",
    )

    assert trace_id == "trace_fallback_321"
    assert fallback_calls == ["fallback:profile_retrieval_failure:seed_run_0001:100000"]


def test_run_with_fault_raises_when_live_and_fallback_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_live_run(*, fault_profile: str, run_id: str, phoenix_endpoint: str, project_name: str) -> str:
        raise RuntimeError("live path failed unexpectedly")

    def fake_seeded_fallback(*, fault_profile: str, run_id: str, phoenix_endpoint: str, project_name: str, lookup_limit: int) -> str:
        raise RuntimeError("fallback path failed")

    monkeypatch.setattr(fault_injector, "_run_live_llamaindex_fault", fake_live_run)
    monkeypatch.setattr(fault_injector, "_run_seeded_fallback_fault", fake_seeded_fallback)

    with pytest.raises(RuntimeError, match="Both live LlamaIndex and deterministic fallback paths failed"):
        fault_injector.run_with_fault(
            fault_profile="profile_instruction_failure",
            run_id="seed_run_0002",
            phoenix_endpoint="http://127.0.0.1:6006",
            project_name="phase1-seeded-failures",
        )
