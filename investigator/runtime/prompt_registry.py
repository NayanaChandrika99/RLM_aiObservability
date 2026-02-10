# ABOUTME: Loads immutable prompt templates and JSON schemas for shared runtime model calls.
# ABOUTME: Computes deterministic prompt hashes from canonical prompt and schema bytes.

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any


PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"


_PROMPT_PATHS = {
    "rca_trace_judgment_v1": (
        "rca/trace_rca_judgment_v1.md",
        "rca/trace_rca_judgment_v1.schema.json",
    ),
    "policy_compliance_judgment_v1": (
        "compliance/policy_compliance_judgment_v1.md",
        "compliance/policy_compliance_judgment_v1.schema.json",
    ),
    "incident_dossier_judgment_v1": (
        "incident/incident_dossier_judgment_v1.md",
        "incident/incident_dossier_judgment_v1.schema.json",
    ),
    "recursive_runtime_action_v1": (
        "runtime/recursive_runtime_action_v1.md",
        "runtime/recursive_runtime_action_v1.schema.json",
    ),
}


@dataclass(frozen=True)
class PromptDefinition:
    prompt_id: str
    prompt_path: str
    schema_path: str
    prompt_text: str
    response_schema: dict[str, Any]
    prompt_template_hash: str


def _canonical_schema_bytes(schema: dict[str, Any]) -> bytes:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return canonical.encode("utf-8")


def _prompt_hash(prompt_text: str, schema: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(prompt_text.encode("utf-8"))
    digest.update(b"\n---\n")
    digest.update(_canonical_schema_bytes(schema))
    return f"sha256:{digest.hexdigest()}"


def list_prompt_ids() -> list[str]:
    return sorted(_PROMPT_PATHS.keys())


def _load_prompt_assets(prompt_id: str) -> tuple[str, str, str, str, dict[str, Any], str]:
    if prompt_id not in _PROMPT_PATHS:
        raise KeyError(f"Unknown prompt_id: {prompt_id}")
    prompt_rel, schema_rel = _PROMPT_PATHS[prompt_id]
    prompt_file = PROMPTS_ROOT / prompt_rel
    schema_file = PROMPTS_ROOT / schema_rel
    prompt_text = prompt_file.read_text(encoding="utf-8")
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError(f"Schema must be an object for prompt_id={prompt_id}.")
    prompt_hash = _prompt_hash(prompt_text, schema)
    return (
        prompt_id,
        str(prompt_file),
        str(schema_file),
        prompt_text,
        schema,
        prompt_hash,
    )


@lru_cache(maxsize=None)
def get_prompt_definition(prompt_id: str) -> PromptDefinition:
    prompt_id_value, prompt_path, schema_path, prompt_text, schema, prompt_hash = _load_prompt_assets(prompt_id)
    return PromptDefinition(
        prompt_id=prompt_id_value,
        prompt_path=prompt_path,
        schema_path=schema_path,
        prompt_text=prompt_text,
        response_schema=schema,
        prompt_template_hash=prompt_hash,
    )


def get_prompt_definition_by_hash(prompt_template_hash: str) -> PromptDefinition:
    for prompt_id in list_prompt_ids():
        prompt = get_prompt_definition(prompt_id)
        if prompt.prompt_template_hash == prompt_template_hash:
            return prompt
    raise KeyError(f"Unknown prompt_template_hash: {prompt_template_hash}")
