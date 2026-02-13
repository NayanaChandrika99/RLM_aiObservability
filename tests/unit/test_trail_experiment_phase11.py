# ABOUTME: Validates TRAIL experiment infrastructure including dev subset, prompt v2, and experiment runner.
# ABOUTME: Tests are deterministic and do not call external LLM APIs.

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_dev_subset_manifest_is_valid_json() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "arcgentica" / "dev_subset_manifest.json"
    assert manifest_path.exists(), f"Manifest not found at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "subset_id" in manifest
    assert manifest["subset_id"] == "dev18"
    assert "trace_ids" in manifest
    assert isinstance(manifest["trace_ids"], list)
    assert len(manifest["trace_ids"]) == 18
    # All IDs should be 32-char hex strings
    for tid in manifest["trace_ids"]:
        assert isinstance(tid, str)
        assert len(tid) == 32, f"trace_id {tid} is not 32 chars"
    # Should be sorted for determinism
    assert manifest["trace_ids"] == sorted(manifest["trace_ids"])
