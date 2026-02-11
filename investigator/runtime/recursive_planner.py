# ABOUTME: Produces one typed recursive runtime action per turn from a structured model response.
# ABOUTME: Wraps shared prompt registry and structured-generation retry logic for planner calls.

from __future__ import annotations

import json
from typing import Any

from investigator.runtime.llm_client import (
    RuntimeModelClient,
    StructuredGenerationRequest,
    StructuredGenerationUsage,
)
from investigator.runtime.llm_loop import run_structured_generation_loop
from investigator.runtime.prompt_registry import get_prompt_definition


def _usage_payload(usage: StructuredGenerationUsage) -> dict[str, Any]:
    return {
        "tokens_in": int(usage.tokens_in),
        "tokens_out": int(usage.tokens_out),
        "cost_usd": float(usage.cost_usd),
    }


def _build_user_prompt(context: dict[str, Any]) -> str:
    context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
    return (
        "Runtime planner context JSON:\n"
        f"{context_json}\n\n"
        "Return only the next typed action object wrapped in the required schema."
    )


class StructuredActionPlanner:
    def __init__(
        self,
        *,
        client: RuntimeModelClient,
        model_name: str,
        temperature: float | None,
        prompt_id: str = "recursive_runtime_action_v1",
        max_attempts: int = 2,
        max_output_tokens: int | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")
        self._client = client
        self._model_name = model_name
        self._temperature = temperature
        self._max_attempts = max_attempts
        self._max_output_tokens = max_output_tokens
        self._prompt = get_prompt_definition(prompt_id)
        self._usage_total = StructuredGenerationUsage()

    @property
    def prompt_template_hash(self) -> str:
        return self._prompt.prompt_template_hash

    @property
    def prompt_id(self) -> str:
        return self._prompt.prompt_id

    @property
    def usage_total(self) -> StructuredGenerationUsage:
        return StructuredGenerationUsage(
            tokens_in=int(self._usage_total.tokens_in),
            tokens_out=int(self._usage_total.tokens_out),
            cost_usd=float(self._usage_total.cost_usd),
        )

    def __call__(self, context: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(context, dict):
            raise ValueError("Planner context must be an object.")
        request = StructuredGenerationRequest(
            model_provider=getattr(self._client, "model_provider", "openai"),
            model_name=self._model_name,
            temperature=self._temperature,
            system_prompt=self._prompt.prompt_text,
            user_prompt=_build_user_prompt(context),
            response_schema_name=self._prompt.prompt_id,
            response_schema=self._prompt.response_schema,
            max_output_tokens=self._max_output_tokens,
        )
        result = run_structured_generation_loop(
            client=self._client,
            request=request,
            max_attempts=self._max_attempts,
        )
        self._usage_total.add(result.usage)
        action = result.output.get("action")
        if not isinstance(action, dict):
            raise ValueError("Planner action must be an object.")
        return {
            "action": action,
            "usage": _usage_payload(result.usage),
        }
