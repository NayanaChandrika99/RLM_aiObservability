# ABOUTME: Validates Phase 8 prompt registry loading, hash determinism, and hash-to-file resolution.
# ABOUTME: Ensures prompt template hashes map to immutable prompt+schema assets on disk.

from __future__ import annotations

from investigator.runtime.prompt_registry import get_prompt_definition, get_prompt_definition_by_hash


def test_prompt_registry_hash_is_deterministic_for_same_prompt() -> None:
    first = get_prompt_definition("rca_trace_judgment_v1")
    second = get_prompt_definition("rca_trace_judgment_v1")

    assert first.prompt_template_hash == second.prompt_template_hash
    assert first.prompt_text
    assert first.response_schema


def test_prompt_registry_resolves_hash_back_to_prompt_definition() -> None:
    prompt = get_prompt_definition("rca_trace_judgment_v1")
    resolved = get_prompt_definition_by_hash(prompt.prompt_template_hash)

    assert resolved.prompt_id == "rca_trace_judgment_v1"
    assert resolved.prompt_path == prompt.prompt_path
    assert resolved.schema_path == prompt.schema_path
