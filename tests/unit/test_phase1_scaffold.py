# ABOUTME: Verifies the Phase 1 scaffold helpers for tutorial selection and seeded-failure manifest generation.
# ABOUTME: Keeps Phase 1 setup deterministic and contract-aligned before runtime execution.

from pathlib import Path

import pandas as pd
import pytest

from apps.demo_agent.phase1_langgraph_runner import (
    default_collector_endpoint,
    tutorial_notebook_path,
)
from apps.demo_agent.phase1_seeded_failures import (
    build_seed_manifest,
    normalize_dataframe_for_parquet,
)


def test_tutorial_notebook_path_exists() -> None:
    notebook = tutorial_notebook_path()
    assert notebook.name == "langgraph_agent_tracing_tutorial.ipynb"
    assert notebook.exists()


def test_default_collector_endpoint_falls_back_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    assert default_collector_endpoint() == "http://127.0.0.1:6006"


def test_default_collector_endpoint_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:7007")
    assert default_collector_endpoint() == "http://localhost:7007"


def test_build_seed_manifest_is_deterministic() -> None:
    manifest_a = build_seed_manifest(seed=7, num_traces=8, dataset_id="seeded_failures_v1")
    manifest_b = build_seed_manifest(seed=7, num_traces=8, dataset_id="seeded_failures_v1")
    assert manifest_a == manifest_b


def test_build_seed_manifest_uses_external_labels() -> None:
    manifest = build_seed_manifest(seed=11, num_traces=12, dataset_id="seeded_failures_v1")
    assert manifest["dataset_id"] == "seeded_failures_v1"
    assert len(manifest["cases"]) == 12
    assert all("expected_label" in case for case in manifest["cases"])
    assert all(case["trace_id"] is None for case in manifest["cases"])
    assert {case["expected_label"] for case in manifest["cases"]}.issubset(
        {
            "tool_failure",
            "retrieval_failure",
            "instruction_failure",
            "upstream_dependency_failure",
            "data_schema_mismatch",
        }
    )


def test_manifest_output_path_convention() -> None:
    expected = Path("datasets/seeded_failures/manifest.json")
    assert str(expected).endswith("datasets/seeded_failures/manifest.json")


def test_normalize_dataframe_for_parquet_handles_mixed_object_values() -> None:
    dataframe = pd.DataFrame(
        {
            "context.trace_id": ["a", "b"],
            "attributes.metadata": [{"x": 1}, 2],
            "events": [[{"name": "event"}], None],
        }
    )
    normalized = normalize_dataframe_for_parquet(dataframe)
    assert isinstance(normalized.loc[0, "attributes.metadata"], str)
    assert isinstance(normalized.loc[1, "attributes.metadata"], str)
    assert pd.isna(normalized.loc[1, "events"])
