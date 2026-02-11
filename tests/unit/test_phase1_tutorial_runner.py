# ABOUTME: Validates Phase 1 tutorial-run helper configuration so execution is reproducible.
# ABOUTME: Covers deterministic project/model defaults without requiring network calls.

import pytest

from apps.demo_agent.phase1_tutorial_run import resolved_model_name, tutorial_project_name


def test_tutorial_project_name_default() -> None:
    assert tutorial_project_name() == "phase1-langgraph-tutorial"


def test_tutorial_project_name_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHASE1_PROJECT_NAME", "custom-project")
    assert tutorial_project_name() == "custom-project"


def test_resolved_model_name_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHASE1_LLM_MODEL", raising=False)
    assert resolved_model_name() == "gpt-5-mini"


def test_resolved_model_name_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHASE1_LLM_MODEL", "gpt-5.2")
    assert resolved_model_name() == "gpt-5.2"
