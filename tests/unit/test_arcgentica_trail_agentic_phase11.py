# ABOUTME: Verifies Phase 11 TRAIL agentic analysis behavior using recursive Agentica-style delegation.
# ABOUTME: Enforces budget limits, deterministic merge behavior, and heuristic fallback on agent failures.

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import pytest

from arcgentica import trail_agent


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del return_type
        self.calls.append({"task": task, "objects": objects})

        if "# Chunk Catalog (JSON)" in task:
            return '{"chunk_ids":[0,1,2,3]}'

        chunk_id_match = re.search(r"# Chunk ID\n(\d+)", task)
        chunk_id = int(chunk_id_match.group(1)) if chunk_id_match else -1

        span_ids_match = re.search(
            r"# Chunk Span IDs \(JSON\)\n(.*?)\n\n# Chunk Payload",
            task,
            flags=re.DOTALL,
        )
        span_ids: list[str] = []
        if span_ids_match:
            span_ids = json.loads(span_ids_match.group(1))
        location = span_ids[0] if span_ids else ""
        return (
            "{"
            '"errors":['
            "{"
            '"category":"Timeout Issues",'
            f'"location":"{location}",'
            f'"evidence":"agentic evidence from chunk {chunk_id}",'
            '"description":"Detected timeout pattern.",'
            '"impact":"HIGH"'
            "}"
            "]"
            "}"
        )


class _FailingRuntime:
    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del task
        del return_type
        del objects
        raise RuntimeError("simulated runtime failure")


class _SlotLimitedRuntime:
    def __init__(self, max_num_agents: int) -> None:
        self.max_num_agents = max_num_agents
        self.calls: list[str] = []
        self._agents_used = 0

    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del return_type
        del objects
        if self._agents_used >= self.max_num_agents:
            raise ValueError(f"Maximum total number of agents reached for this trace: {self.max_num_agents}")
        self._agents_used += 1
        self.calls.append(task)

        if "# Chunk Catalog (JSON)" in task:
            return '{"chunk_ids":[0,1,2,3]}'
        return (
            "{"
            '"errors":['
            "{"
            '"category":"Timeout Issues",'
            '"location":"span_0",'
            '"evidence":"timed out",'
            '"description":"Detected timeout pattern.",'
            '"impact":"HIGH"'
            "}"
            "]"
            "}"
        )


class _PartialChunkFailureRuntime:
    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del return_type
        del objects
        if "# Chunk Catalog (JSON)" in task:
            return '{"chunk_ids":[0,1]}'

        chunk_id_match = re.search(r"# Chunk ID\n(\d+)", task)
        chunk_id = int(chunk_id_match.group(1)) if chunk_id_match else -1
        if chunk_id == 1:
            raise RuntimeError("delegated chunk failure")
        return (
            "{"
            '"errors":['
            "{"
            '"category":"Timeout Issues",'
            '"location":"span_0",'
            '"evidence":"timed out",'
            '"description":"Detected timeout pattern.",'
            '"impact":"HIGH"'
            "}"
            "]"
            "}"
        )


class _RootTimeoutThenSuccessRuntime:
    def __init__(self) -> None:
        self.root_calls = 0

    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del return_type
        del objects
        if "# Chunk Catalog (JSON)" in task:
            self.root_calls += 1
            if self.root_calls == 1:
                raise TimeoutError()
            return '{"chunk_ids":[0]}'
        return (
            "{"
            '"errors":['
            "{"
            '"category":"Timeout Issues",'
            '"location":"span_0",'
            '"evidence":"timed out",'
            '"description":"Detected timeout pattern.",'
            '"impact":"HIGH"'
            "}"
            "]"
            "}"
        )


class _RootTimeoutThenSuccessSlotLimitedRuntime:
    def __init__(self, max_num_agents: int) -> None:
        self.max_num_agents = max_num_agents
        self.root_calls = 0
        self._agents_used = 0

    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del return_type
        del objects
        if self._agents_used >= self.max_num_agents:
            raise ValueError(f"Maximum total number of agents reached for this trace: {self.max_num_agents}")
        self._agents_used += 1

        if "# Chunk Catalog (JSON)" in task:
            self.root_calls += 1
            if self.root_calls == 1:
                raise TimeoutError()
            return '{"chunk_ids":[0]}'

        return (
            "{"
            '"errors":['
            "{"
            '"category":"Timeout Issues",'
            '"location":"span_0",'
            '"evidence":"timed out",'
            '"description":"Detected timeout pattern.",'
            '"impact":"HIGH"'
            "}"
            "]"
            "}"
        )


class _ChunkTimeoutThenSuccessRuntime:
    def __init__(self) -> None:
        self.chunk_calls = 0

    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del return_type
        del objects
        if "# Chunk Catalog (JSON)" in task:
            return '{"chunk_ids":[0]}'
        self.chunk_calls += 1
        if self.chunk_calls == 1:
            raise TimeoutError()
        return (
            "{"
            '"errors":['
            "{"
            '"category":"Timeout Issues",'
            '"location":"span_0",'
            '"evidence":"timed out",'
            '"description":"Detected timeout pattern.",'
            '"impact":"HIGH"'
            "}"
            "]"
            "}"
        )


class _RootAlwaysTimeoutRuntime:
    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del return_type
        del objects
        if "# Chunk Catalog (JSON)" in task:
            raise TimeoutError()
        return (
            "{"
            '"errors":['
            "{"
            '"category":"Timeout Issues",'
            '"location":"span_0",'
            '"evidence":"timed out",'
            '"description":"Detected timeout pattern.",'
            '"impact":"HIGH"'
            "}"
            "]"
            "}"
        )


class _NoRootPlanningRuntime:
    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del return_type
        del objects
        if "# Chunk Catalog (JSON)" in task:
            raise AssertionError("root planner should not be called when all chunks fit in budget")
        return (
            "{"
            '"errors":['
            "{"
            '"category":"Timeout Issues",'
            '"location":"span_0",'
            '"evidence":"timed out",'
            '"description":"Detected timeout pattern.",'
            '"impact":"HIGH"'
            "}"
            "]"
            "}"
        )


class _SlowRuntime:
    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del task
        del return_type
        del objects
        await asyncio.sleep(0.05)
        return "{}"


class _ChunkAlwaysTimeoutRuntime:
    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> str:
        del return_type
        del objects
        if "# Chunk Catalog (JSON)" in task:
            return '{"chunk_ids":[0]}'
        raise TimeoutError()


def _make_trace(span_count: int = 5) -> dict[str, Any]:
    spans = []
    for index in range(span_count):
        spans.append(
            {
                "span_id": f"span_{index}",
                "span_name": f"tool_span_{index}",
                "status_code": "Error" if index % 2 == 0 else "Unset",
                "status_message": "request timed out while waiting" if index % 2 == 0 else "",
                "span_attributes": {"detail": f"timeout context {index}"},
                "logs": [{"body": {"message": f"log timeout {index}"}}],
                "child_spans": [],
            }
        )
    return {"trace_id": "trace_agentic", "spans": spans}


def test_agentic_mode_uses_chunk_delegation_with_budget(monkeypatch) -> None:
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(
        trail_agent,
        "_make_agentic_runtime",
        lambda *args, **kwargs: fake_runtime,
    )

    result = trail_agent.analyze_trace(
        _make_trace(span_count=8),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        max_num_agents=4,
        max_chunks=4,
        max_spans_per_chunk=2,
    )

    # One planning call + 3 delegated chunk calls due to delegated chunk budget.
    assert len(fake_runtime.calls) == 4
    assert len(result["errors"]) == 3
    assert [entry["location"] for entry in result["errors"]] == ["span_0", "span_2", "span_4"]
    assert all(entry["category"] == "Timeout Issues" for entry in result["errors"])


def test_agentic_mode_falls_back_to_heuristics_on_runtime_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        trail_agent,
        "_make_agentic_runtime",
        lambda *args, **kwargs: _FailingRuntime(),
    )
    trace_payload = {
        "trace_id": "trace_fallback",
        "spans": [
            {
                "span_id": "span_auth",
                "span_name": "api_call",
                "status_code": "Error",
                "status_message": "401 unauthorized",
                "span_attributes": {"detail": "auth failed"},
                "logs": [],
                "child_spans": [],
            },
            {
                "span_id": "span_other_1",
                "span_name": "api_call",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {"detail": "normal"},
                "logs": [],
                "child_spans": [],
            },
            {
                "span_id": "span_other_2",
                "span_name": "api_call",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {"detail": "normal"},
                "logs": [],
                "child_spans": [],
            },
        ],
    }

    result = trail_agent.analyze_trace(
        trace_payload,
        model="openai/gpt-5-mini",
        agentic_mode="on",
        max_num_agents=3,
        max_chunks=4,
        max_spans_per_chunk=1,
    )
    categories = {entry["category"] for entry in result["errors"]}
    assert "Authentication Errors" in categories


def test_agentic_mode_caps_selected_chunks_to_available_agent_slots(monkeypatch) -> None:
    runtimes: list[_SlotLimitedRuntime] = []

    def _make_runtime(*args: Any, **kwargs: Any) -> _SlotLimitedRuntime:
        del args
        runtime = _SlotLimitedRuntime(max_num_agents=int(kwargs["max_num_agents"]))
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(trail_agent, "_make_agentic_runtime", _make_runtime)

    result = trail_agent.analyze_trace(
        _make_trace(span_count=8),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        max_num_agents=2,
        max_chunks=4,
        max_spans_per_chunk=2,
    )

    assert len(runtimes) == 2
    total_calls = sum(len(runtime.calls) for runtime in runtimes)
    assert total_calls == 2
    diagnostics = result.get("analysis_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("delegation_failures") == 0
    assert diagnostics.get("selected_chunk_ids") == [0]


def test_agentic_mode_reports_delegation_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        trail_agent,
        "_make_agentic_runtime",
        lambda *args, **kwargs: _PartialChunkFailureRuntime(),
    )

    result = trail_agent.analyze_trace(
        _make_trace(span_count=4),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        max_num_agents=4,
        max_chunks=2,
        max_spans_per_chunk=2,
    )

    diagnostics = result.get("analysis_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("delegation_failures") == 1
    assert diagnostics.get("delegation_failed_chunk_ids") == [1]


def test_analyze_trace_passes_split_models_to_agentic_runtime(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_analyze_trace_agentic(
        trace_payload: dict[str, Any],
        root_model: str,
        chunk_model: str,
        max_num_agents: int,
        max_chunks: int,
        max_spans_per_chunk: int,
        max_span_text_chars: int,
        agent_call_timeout_seconds: float = 60.0,
        agent_timeout_retries: int = 1,
        joint_recall_boost: bool = False,
    ) -> dict[str, Any]:
        captured["trace_id"] = trace_payload.get("trace_id")
        captured["root_model"] = root_model
        captured["chunk_model"] = chunk_model
        captured["max_num_agents"] = max_num_agents
        captured["max_chunks"] = max_chunks
        captured["max_spans_per_chunk"] = max_spans_per_chunk
        captured["max_span_text_chars"] = max_span_text_chars
        captured["agent_call_timeout_seconds"] = agent_call_timeout_seconds
        captured["agent_timeout_retries"] = agent_timeout_retries
        captured["joint_recall_boost"] = joint_recall_boost
        return {"trace_id": str(trace_payload.get("trace_id", "")), "errors": [], "scores": [{"overall": 5.0}]}

    monkeypatch.setattr(trail_agent, "_analyze_trace_agentic", _fake_analyze_trace_agentic)

    result = trail_agent.analyze_trace(
        _make_trace(span_count=3),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        root_model="openai/gpt-5.2",
        chunk_model="openai/gpt-5-mini",
        max_num_agents=4,
        max_chunks=2,
        max_spans_per_chunk=2,
        max_span_text_chars=500,
        agent_call_timeout_seconds=75,
        agent_timeout_retries=0,
    )

    assert result["trace_id"] == "trace_agentic"
    assert captured["trace_id"] == "trace_agentic"
    assert captured["root_model"] == "openai/gpt-5.2"
    assert captured["chunk_model"] == "openai/gpt-5-mini"
    assert captured["max_num_agents"] == 4
    assert captured["max_chunks"] == 2
    assert captured["max_spans_per_chunk"] == 2
    assert captured["max_span_text_chars"] == 500
    assert captured["agent_call_timeout_seconds"] == 75
    assert captured["agent_timeout_retries"] == 0
    assert captured["joint_recall_boost"] is False


def test_split_models_disable_root_delegation(monkeypatch) -> None:
    runtime_kwargs: list[dict[str, Any]] = []

    def _make_runtime(*args: Any, **kwargs: Any) -> _FakeRuntime:
        del args
        runtime_kwargs.append(dict(kwargs))
        return _FakeRuntime()

    monkeypatch.setattr(trail_agent, "_make_agentic_runtime", _make_runtime)

    result = trail_agent.analyze_trace(
        _make_trace(span_count=4),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        root_model="openai/gpt-5.2",
        chunk_model="openai/gpt-5-mini",
        max_num_agents=4,
        max_chunks=1,
        max_spans_per_chunk=2,
    )

    assert len(runtime_kwargs) == 2
    assert runtime_kwargs[0]["model"] == "openai/gpt-5.2"
    assert runtime_kwargs[0]["max_num_agents"] == 2
    assert runtime_kwargs[0]["allow_delegation"] is False
    assert runtime_kwargs[1]["model"] == "openai/gpt-5-mini"
    assert runtime_kwargs[1]["max_num_agents"] == 6
    assert runtime_kwargs[1]["allow_delegation"] is True
    diagnostics = result.get("analysis_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("analysis_mode") == "agentic_repl"


def test_agentic_mode_retries_root_timeout_once(monkeypatch) -> None:
    runtime = _RootTimeoutThenSuccessRuntime()
    monkeypatch.setattr(
        trail_agent,
        "_make_agentic_runtime",
        lambda *args, **kwargs: runtime,
    )

    result = trail_agent.analyze_trace(
        _make_trace(span_count=6),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        max_num_agents=3,
        max_chunks=4,
        max_spans_per_chunk=2,
    )

    diagnostics = result.get("analysis_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("analysis_mode") == "agentic_repl"
    assert diagnostics.get("delegation_failures") == 0
    assert runtime.root_calls == 2


def test_agentic_mode_root_timeout_retry_does_not_exhaust_runtime_slots(monkeypatch) -> None:
    runtimes: list[_RootTimeoutThenSuccessSlotLimitedRuntime] = []

    def _make_runtime(*args: Any, **kwargs: Any) -> _RootTimeoutThenSuccessSlotLimitedRuntime:
        del args
        runtime = _RootTimeoutThenSuccessSlotLimitedRuntime(max_num_agents=int(kwargs["max_num_agents"]))
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(trail_agent, "_make_agentic_runtime", _make_runtime)

    result = trail_agent.analyze_trace(
        _make_trace(span_count=6),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        max_num_agents=3,
        max_chunks=4,
        max_spans_per_chunk=2,
    )

    diagnostics = result.get("analysis_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("analysis_mode") == "agentic_repl"
    assert diagnostics.get("delegation_failures") == 0
    assert runtimes[0].root_calls == 2


def test_agentic_mode_retries_chunk_timeout_once(monkeypatch) -> None:
    runtime = _ChunkTimeoutThenSuccessRuntime()
    monkeypatch.setattr(
        trail_agent,
        "_make_agentic_runtime",
        lambda *args, **kwargs: runtime,
    )

    result = trail_agent.analyze_trace(
        _make_trace(span_count=4),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        max_num_agents=3,
        max_chunks=1,
        max_spans_per_chunk=2,
    )

    diagnostics = result.get("analysis_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("analysis_mode") == "agentic_repl"
    assert diagnostics.get("delegation_failures") == 0
    assert runtime.chunk_calls == 2
    assert len(result.get("errors", [])) == 1


def test_agentic_mode_continues_when_root_planner_times_out(monkeypatch) -> None:
    monkeypatch.setattr(
        trail_agent,
        "_make_agentic_runtime",
        lambda *args, **kwargs: _RootAlwaysTimeoutRuntime(),
    )

    result = trail_agent.analyze_trace(
        _make_trace(span_count=6),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        max_num_agents=3,
        max_chunks=4,
        max_spans_per_chunk=2,
    )

    diagnostics = result.get("analysis_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("analysis_mode") == "agentic_repl"
    assert diagnostics.get("root_planner_fallback") == "timeout"
    assert diagnostics.get("selected_chunk_ids") == [0, 1]
    assert diagnostics.get("delegation_failures") == 0
    assert len(result.get("errors", [])) == 2


def test_agentic_mode_skips_root_planner_when_all_chunks_fit_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        trail_agent,
        "_make_agentic_runtime",
        lambda *args, **kwargs: _NoRootPlanningRuntime(),
    )

    result = trail_agent.analyze_trace(
        _make_trace(span_count=4),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        max_num_agents=6,
        max_chunks=6,
        max_spans_per_chunk=12,
    )

    diagnostics = result.get("analysis_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("analysis_mode") == "agentic_repl"
    assert diagnostics.get("root_planner_fallback") == "skipped_all_chunks_fit_budget"
    assert diagnostics.get("selected_chunk_ids") == [0]
    assert len(result.get("errors", [])) == 1


def test_call_agent_with_timeout_retry_enforces_hard_timeout() -> None:
    with pytest.raises(TimeoutError):
        asyncio.run(
            trail_agent._call_agent_with_timeout_retry(
                _SlowRuntime(),
                task="slow call",
                return_type=str,
                timeout_retries=0,
                per_call_timeout_seconds=0.01,
            )
        )


def test_agentic_mode_recovers_chunk_timeout_without_delegation_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        trail_agent,
        "_make_agentic_runtime",
        lambda *args, **kwargs: _ChunkAlwaysTimeoutRuntime(),
    )

    result = trail_agent.analyze_trace(
        _make_trace(span_count=4),
        model="openai/gpt-5-mini",
        agentic_mode="on",
        max_num_agents=3,
        max_chunks=1,
        max_spans_per_chunk=2,
    )

    diagnostics = result.get("analysis_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("analysis_mode") == "agentic_repl"
    assert diagnostics.get("delegation_failures") == 0
    assert diagnostics.get("chunk_timeout_recoveries") == 1
    assert diagnostics.get("chunk_timeout_recovery_ids") == [0]
    categories = {entry.get("category") for entry in result.get("errors", [])}
    assert "Timeout Issues" in categories


def test_summarize_trace_truncates_top_signal_text() -> None:
    trace_payload = {"trace_id": "trace_summary", "spans": []}
    span_records = [
        {"span_id": "s1", "text": "x" * 800},
        {"span_id": "s2", "text": "short"},
    ]
    summary = trail_agent._summarize_trace(trace_payload, span_records)

    assert summary["trace_id"] == "trace_summary"
    assert summary["top_span_ids"] == ["s1", "s2"]
    assert len(summary["top_signals"][0]) <= 283
    assert summary["top_signals"][0].endswith("...")


def test_prefer_local_session_manager_env_masks_platform_vars(monkeypatch) -> None:
    monkeypatch.setenv("S_M_BASE_URL", "http://localhost:2345")
    monkeypatch.setenv("AGENTICA_API_KEY", "token123")
    monkeypatch.setenv("AGENTICA_BASE_URL", "https://api.platform.symbolica.ai")

    with trail_agent._prefer_local_session_manager_env():
        assert os.getenv("AGENTICA_API_KEY") is None
        assert os.getenv("AGENTICA_BASE_URL") is None

    assert os.getenv("AGENTICA_API_KEY") == "token123"
    assert os.getenv("AGENTICA_BASE_URL") == "https://api.platform.symbolica.ai"


def test_prefer_local_session_manager_env_keeps_platform_vars_without_local_base(monkeypatch) -> None:
    monkeypatch.delenv("S_M_BASE_URL", raising=False)
    monkeypatch.setenv("AGENTICA_API_KEY", "token123")

    with trail_agent._prefer_local_session_manager_env():
        assert os.getenv("AGENTICA_API_KEY") == "token123"

    assert os.getenv("AGENTICA_API_KEY") == "token123"


def test_parse_chunk_errors_prefers_evidence_matched_span_for_location() -> None:
    raw = json.dumps(
        {
            "errors": [
                {
                    "category": "Timeout Issues",
                    "location": "not_a_real_span",
                    "evidence": "connection timed out while reading service response",
                    "description": "timeout",
                    "impact": "HIGH",
                }
            ]
        }
    )
    parsed = trail_agent._parse_chunk_errors(
        raw,
        chunk_span_ids=["span_a", "span_b"],
        chunk_span_texts={
            "span_a": "normal execution with no issues",
            "span_b": "external call failed: connection timed out while reading service response",
        },
    )
    assert len(parsed) == 1
    assert parsed[0]["location"] == "span_b"


def test_refine_agentic_locations_moves_to_better_evidence_match() -> None:
    trace_payload = {
        "trace_id": "trace_refine",
        "spans": [
            {
                "span_id": "span_a",
                "span_name": "tool_call",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "normal execution"}],
                "child_spans": [],
            },
            {
                "span_id": "span_b",
                "span_name": "tool_call",
                "status_code": "Error",
                "status_message": "request timed out",
                "span_attributes": {},
                "logs": [{"body": "external service request timed out after 30s"}],
                "child_spans": [],
            },
        ],
    }
    findings = [
        {
            "category": "Timeout Issues",
            "location": "span_a",
            "evidence": "external service request timed out after 30s",
            "description": "timeout",
            "impact": "HIGH",
        }
    ]
    refined = trail_agent._refine_agentic_locations(trace_payload, findings)
    assert refined[0]["location"] == "span_b"


def test_refine_agentic_locations_uses_last_match_for_resource_abuse() -> None:
    trace_payload = {
        "trace_id": "trace_abuse",
        "spans": [
            {
                "span_id": "s1",
                "span_name": "tool_call",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "repeated request same parameters"}],
                "child_spans": [],
            },
            {
                "span_id": "s2",
                "span_name": "tool_call",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "repeated request same parameters"}],
                "child_spans": [],
            },
        ],
    }
    findings = [
        {
            "category": "Resource Abuse",
            "location": "s1",
            "evidence": "repeated request same parameters",
            "description": "abuse",
            "impact": "MEDIUM",
        }
    ]
    refined = trail_agent._refine_agentic_locations(trace_payload, findings)
    assert refined[0]["location"] == "s2"


def test_refine_agentic_locations_does_not_move_on_description_only_signal() -> None:
    trace_payload = {
        "trace_id": "trace_desc_signal",
        "spans": [
            {
                "span_id": "span_a",
                "span_name": "tool_call",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "output issue observed"}],
                "child_spans": [],
            },
            {
                "span_id": "span_b",
                "span_name": "tool_call",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "schema mismatch while formatting final response"}],
                "child_spans": [],
            },
        ],
    }
    findings = [
        {
            "category": "Formatting Errors",
            "location": "span_a",
            "evidence": "output issue",
            "description": "schema mismatch while formatting final response",
            "impact": "HIGH",
        }
    ]
    refined = trail_agent._refine_agentic_locations(trace_payload, findings)
    assert refined[0]["location"] == "span_a"


def test_refine_agentic_locations_moves_on_small_score_advantage() -> None:
    trace_payload = {
        "trace_id": "trace_small_advantage",
        "spans": [
            {
                "span_id": "span_a",
                "span_name": "tool_call",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "alpha"}],
                "child_spans": [],
            },
            {
                "span_id": "span_b",
                "span_name": "tool_call",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "alpha alpha alpha alpha beta"}],
                "child_spans": [],
            },
        ],
    }
    findings = [
        {
            "category": "Tool-related",
            "location": "span_a",
            "evidence": "alpha beta gamma",
            "description": "timeout",
            "impact": "HIGH",
        }
    ]
    refined = trail_agent._refine_agentic_locations(trace_payload, findings)
    assert refined[0]["location"] == "span_b"


def test_refine_agentic_locations_prefers_litellm_span_for_semantic_category() -> None:
    trace_payload = {
        "trace_id": "trace_litellm_bias",
        "spans": [
            {
                "span_id": "span_tool",
                "span_name": "PageDownTool",
                "status_code": "Error",
                "status_message": "invalid json format",
                "span_attributes": {},
                "logs": [{"body": "invalid json format in response"}],
                "child_spans": [],
            },
            {
                "span_id": "span_llm",
                "span_name": "LiteLLMModel.__call__",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "invalid json format in response"}],
                "child_spans": [],
            },
        ],
    }
    findings = [
        {
            "category": "Formatting Errors",
            "location": "span_tool",
            "evidence": "invalid json format in response",
            "description": "output formatting issue",
            "impact": "HIGH",
        }
    ]
    refined = trail_agent._refine_agentic_locations(trace_payload, findings)
    assert refined[0]["location"] == "span_llm"


def test_refine_agentic_locations_avoids_step_span_for_resource_not_found() -> None:
    trace_payload = {
        "trace_id": "trace_resource_not_found_visit_bias",
        "spans": [
            {
                "span_id": "span_step",
                "span_name": "Step 3 Error",
                "status_code": "Error",
                "status_message": "404 not found",
                "span_attributes": {},
                "logs": [{"body": "404 not found while opening target page"}],
                "child_spans": [],
            },
            {
                "span_id": "span_visit",
                "span_name": "VisitTool",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "404 not found while opening target page"}],
                "child_spans": [],
            },
        ],
    }
    findings = [
        {
            "category": "Resource Not Found",
            "location": "span_step",
            "evidence": "404 not found while opening target page",
            "description": "resource missing",
            "impact": "HIGH",
        }
    ]
    refined = trail_agent._refine_agentic_locations(trace_payload, findings)
    assert refined[0]["location"] == "span_visit"


def test_refine_agentic_locations_avoids_step_span_for_tool_related_execution_error() -> None:
    trace_payload = {
        "trace_id": "trace_tool_related_page_bias",
        "spans": [
            {
                "span_id": "span_step",
                "span_name": "Step 10 Error",
                "status_code": "Error",
                "status_message": "Error when executing tool page_down",
                "span_attributes": {},
                "logs": [{"body": "Error when executing tool page_down with arguments {'': ''}"}],
                "child_spans": [],
            },
            {
                "span_id": "span_page",
                "span_name": "PageDownTool",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "TypeError: PageDownTool.forward() got an unexpected keyword argument ''"}],
                "child_spans": [],
            },
        ],
    }
    findings = [
        {
            "category": "Tool-related",
            "location": "span_step",
            "evidence": "TypeError: PageDownTool.forward() got an unexpected keyword argument ''",
            "description": "tool invocation failed",
            "impact": "HIGH",
        }
    ]
    refined = trail_agent._refine_agentic_locations(trace_payload, findings)
    assert refined[0]["location"] == "span_page"


def test_refine_agentic_locations_prefers_non_step_when_scores_tie() -> None:
    trace_payload = {
        "trace_id": "trace_step_tie",
        "spans": [
            {
                "span_id": "span_step",
                "span_name": "Step 4",
                "status_code": "Error",
                "status_message": "invalid json format",
                "span_attributes": {},
                "logs": [{"body": "invalid json format in final output"}],
                "child_spans": [],
            },
            {
                "span_id": "span_llm",
                "span_name": "LiteLLMModel.__call__",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "invalid json format in final output"}],
                "child_spans": [],
            },
        ],
    }
    findings = [
        {
            "category": "Formatting Errors",
            "location": "span_step",
            "evidence": "invalid json format in final output",
            "description": "format issue",
            "impact": "HIGH",
        }
    ]

    refined = trail_agent._refine_agentic_locations(trace_payload, findings)
    assert refined[0]["location"] == "span_llm"


def test_refine_agentic_locations_uses_category_hint_pool_for_infra_errors() -> None:
    trace_payload = {
        "trace_id": "trace_category_hint_pool",
        "spans": [
            {
                "span_id": "span_step",
                "span_name": "Step 2",
                "status_code": "Error",
                "status_message": "request failed",
                "span_attributes": {},
                "logs": [{"body": "request failed for unknown reason"}],
                "child_spans": [],
            },
            {
                "span_id": "span_http",
                "span_name": "VisitTool",
                "status_code": "Error",
                "status_message": "429 too many requests",
                "span_attributes": {},
                "logs": [{"body": "429 too many requests from service"}],
                "child_spans": [],
            },
        ],
    }
    findings = [
        {
            "category": "Rate Limiting",
            "location": "span_step",
            "evidence": "429 too many requests from service",
            "description": "rate limited",
            "impact": "HIGH",
        }
    ]

    refined = trail_agent._refine_agentic_locations(trace_payload, findings)
    assert refined[0]["location"] == "span_http"


def test_reduce_tool_related_fp_drift_collapses_repeated_page_nav_signature() -> None:
    findings = [
        {
            "category": "Tool-related",
            "location": "span_step",
            "evidence": "Error when executing tool page_down with arguments {'': {}}: TypeError",
            "description": "AgentExecutionError while invoking page_down.",
            "impact": "HIGH",
        },
        {
            "category": "Tool-related",
            "location": "span_tool",
            "evidence": "TypeError: unexpected keyword argument ''",
            "description": "PageDownTool.forward() got an unexpected keyword argument.",
            "impact": "HIGH",
        }
    ]
    reduced = trail_agent._reduce_tool_related_fp_drift(findings)
    assert len(reduced) == 1
    assert reduced[0]["category"] == "Tool-related"
    assert reduced[0]["location"] == "span_tool"


def test_reduce_tool_related_fp_drift_keeps_distinct_tool_related_signals() -> None:
    findings = [
        {
            "category": "Tool-related",
            "location": "span_tool",
            "evidence": "PageDownTool Error TypeError: unexpected keyword argument ''",
            "description": "page_down invocation failed",
            "impact": "HIGH",
        },
        {
            "category": "Tool-related",
            "location": "span_convert",
            "evidence": "UnsupportedFormatException: Could not convert file to Markdown.",
            "description": "text inspector failed to parse file format",
            "impact": "MEDIUM",
        },
        {
            "category": "Formatting Errors",
            "location": "span_llm",
            "evidence": "invalid json",
            "description": "output malformed",
            "impact": "HIGH",
        }
    ]
    reduced = trail_agent._reduce_tool_related_fp_drift(findings)
    assert len(reduced) == 3
    assert sum(1 for item in reduced if item["category"] == "Tool-related") == 2
    assert any(item["location"] == "span_convert" for item in reduced)


def test_reduce_formatting_fp_drift_caps_weak_truncation_to_one() -> None:
    findings = [
        {
            "category": "Formatting Errors",
            "location": "span_a",
            "evidence": "payload truncat",
            "description": "payload truncated mid-sentence",
            "impact": "MEDIUM",
        },
        {
            "category": "Formatting Errors",
            "location": "span_b",
            "evidence": "Your answer should use th",
            "description": "message cut off mid-sentence",
            "impact": "MEDIUM",
        },
        {
            "category": "Formatting Errors",
            "location": "span_c",
            "evidence": "So your final out",
            "description": "instruction is truncated",
            "impact": "MEDIUM",
        },
    ]
    reduced = trail_agent._reduce_formatting_fp_drift(findings)

    formatting = [item for item in reduced if item["category"] == "Formatting Errors"]
    assert len(formatting) == 1
    assert formatting[0]["location"] == "span_c"


def test_reduce_formatting_fp_drift_keeps_strong_formatting_signal() -> None:
    findings = [
        {
            "category": "Formatting Errors",
            "location": "span_strong",
            "evidence": "Error when executing tool page_down ... TypeError: unexpected keyword argument ''",
            "description": "tool call argument format mismatch",
            "impact": "HIGH",
        },
        {
            "category": "Formatting Errors",
            "location": "span_weak",
            "evidence": "partial payloa",
            "description": "payload truncated mid-word",
            "impact": "LOW",
        },
        {
            "category": "Tool-related",
            "location": "span_tool",
            "evidence": "tool execution failed",
            "description": "tool call failed",
            "impact": "HIGH",
        },
    ]
    reduced = trail_agent._reduce_formatting_fp_drift(findings)

    assert any(item["location"] == "span_strong" for item in reduced)
    assert any(item["location"] == "span_weak" for item in reduced)
    assert any(item["category"] == "Tool-related" for item in reduced)


def test_reduce_formatting_fp_drift_soft_keeps_two_weak_candidates() -> None:
    findings = [
        {
            "category": "Formatting Errors",
            "location": "span_a",
            "evidence": "payload truncat",
            "description": "payload truncated mid-sentence",
            "impact": "MEDIUM",
        },
        {
            "category": "Formatting Errors",
            "location": "span_b",
            "evidence": "Your answer should use th",
            "description": "message cut off mid-sentence",
            "impact": "MEDIUM",
        },
        {
            "category": "Formatting Errors",
            "location": "span_c",
            "evidence": "So your final out",
            "description": "instruction is truncated",
            "impact": "MEDIUM",
        },
    ]
    reduced = trail_agent._reduce_formatting_fp_drift_soft(findings, max_weak_keep=2)

    formatting = [item for item in reduced if item["category"] == "Formatting Errors"]
    assert len(formatting) == 2
    assert any(item["location"] == "span_c" for item in formatting)


def test_apply_post_filters_uses_winning_soft_formatting_path() -> None:
    findings = [
        {
            "category": "Tool-related",
            "location": "span_step",
            "evidence": "Error when executing tool page_down with arguments {'': {}}: TypeError",
            "description": "AgentExecutionError while invoking page_down.",
            "impact": "HIGH",
        },
        {
            "category": "Tool-related",
            "location": "span_tool",
            "evidence": "TypeError: unexpected keyword argument ''",
            "description": "PageDownTool.forward() got an unexpected keyword argument.",
            "impact": "HIGH",
        },
        {
            "category": "Formatting Errors",
            "location": "span_a",
            "evidence": "payload truncat",
            "description": "payload truncated mid-sentence",
            "impact": "MEDIUM",
        },
        {
            "category": "Formatting Errors",
            "location": "span_b",
            "evidence": "Your answer should use th",
            "description": "message cut off mid-sentence",
            "impact": "MEDIUM",
        },
        {
            "category": "Formatting Errors",
            "location": "span_c",
            "evidence": "So your final out",
            "description": "instruction is truncated",
            "impact": "MEDIUM",
        },
    ]
    filtered = trail_agent._apply_post_filters(findings)

    assert len(filtered) == 4
    formatting = [item for item in filtered if item["category"] == "Formatting Errors"]
    assert len(formatting) == 2


def test_resolve_pair_conflicts_prefers_formatting_for_tool_argument_contract_signal() -> None:
    findings = [
        {
            "category": "Tool-related",
            "location": "span_x",
            "evidence": "Error when executing tool page_down with arguments {'': {}}: TypeError",
            "description": "tool invocation failed",
            "impact": "HIGH",
        },
        {
            "category": "Tool Definition Issues",
            "location": "span_x",
            "evidence": "PageDownTool.forward() got an unexpected keyword argument ''",
            "description": "signature mismatch",
            "impact": "HIGH",
        },
        {
            "category": "Formatting Errors",
            "location": "span_x",
            "evidence": "TypeError ... only use this tool with a correct input. It takes inputs: {}",
            "description": "tool contract/input format was violated",
            "impact": "HIGH",
        },
    ]

    resolved = trail_agent._resolve_pair_conflicts(findings)
    assert len(resolved) == 1
    assert resolved[0]["category"] == "Formatting Errors"
    assert resolved[0]["location"] == "span_x"


def test_resolve_pair_conflicts_prefers_tool_definition_for_schema_signal() -> None:
    findings = [
        {
            "category": "Tool-related",
            "location": "span_schema",
            "evidence": "Tool invocation failed due to mismatch",
            "description": "execution failed",
            "impact": "HIGH",
        },
        {
            "category": "Tool Definition Issues",
            "location": "span_schema",
            "evidence": "tool schema has incorrect parameter type and missing required field",
            "description": "tool schema mismatch",
            "impact": "HIGH",
        },
    ]

    resolved = trail_agent._resolve_pair_conflicts(findings)
    assert len(resolved) == 1
    assert resolved[0]["category"] == "Tool Definition Issues"
    assert resolved[0]["location"] == "span_schema"


def test_recover_targeted_tp_adds_instruction_non_compliance_on_end_plan_violation() -> None:
    trace_payload = {
        "trace_id": "trace_tp_instruction_recovery",
        "spans": [
            {
                "span_id": "span_plan",
                "span_name": "LiteLLMModel.__call__",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [
                    {
                        "body": {
                            "message": 'The output plan ends with "6. Verify ..." instead of ending with "<end_plan>".'
                        }
                    }
                ],
                "child_spans": [],
            }
        ],
    }
    findings = [
        {
            "category": "Context Handling Failures",
            "location": "span_plan",
            "evidence": "plan text was incomplete",
            "description": "context issue",
            "impact": "MEDIUM",
        }
    ]

    recovered = trail_agent._recover_targeted_tp(trace_payload, findings)
    assert any(
        item["category"] == "Instruction Non-compliance" and item["location"] == "span_plan"
        for item in recovered
    )


def test_recover_targeted_tp_adds_tool_selection_error_for_unsupported_inspect_file() -> None:
    trace_payload = {
        "trace_id": "trace_tp_tool_selection_recovery",
        "spans": [
            {
                "span_id": "span_tool_select",
                "span_name": "LiteLLMModel.__call__",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [
                    {
                        "body": {
                            "message": (
                                "Error when executing tool inspect_file_as_text with arguments "
                                "{'file_path':'x.jsonld'}: UnsupportedFormatException: could not convert file; "
                                "this tool does not support this file format."
                            )
                        }
                    }
                ],
                "child_spans": [],
            }
        ],
    }
    findings = [
        {
            "category": "Tool Definition Issues",
            "location": "span_tool_select",
            "evidence": "UnsupportedFormatException",
            "description": "tool failed",
            "impact": "HIGH",
        }
    ]

    recovered = trail_agent._recover_targeted_tp(trace_payload, findings)
    assert any(
        item["category"] == "Tool Selection Errors" and item["location"] == "span_tool_select"
        for item in recovered
    )
