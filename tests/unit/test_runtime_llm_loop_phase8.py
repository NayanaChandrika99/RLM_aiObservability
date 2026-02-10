# ABOUTME: Validates Phase 8 shared LLM loop retry behavior for schema-invalid structured outputs.
# ABOUTME: Ensures retry exhaustion fails with MODEL_OUTPUT_INVALID semantics and aggregated usage.

from __future__ import annotations

import pytest

from investigator.runtime.llm_client import (
    ModelOutputInvalidError,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    StructuredGenerationUsage,
)
from investigator.runtime.llm_loop import run_structured_generation_loop


class _FakeModelClient:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    def generate_structured(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        del request
        if not self._outputs:
            raise AssertionError("No fake outputs left.")
        self.calls += 1
        payload = self._outputs.pop(0)
        return StructuredGenerationResult(
            output=payload,
            raw_text=str(payload),
            usage=StructuredGenerationUsage(tokens_in=10, tokens_out=5, cost_usd=0.02),
        )


def _request() -> StructuredGenerationRequest:
    return StructuredGenerationRequest(
        model_provider="openai",
        model_name="gpt-5-mini",
        temperature=0.0,
        system_prompt="Return only JSON.",
        user_prompt="Pick an RCA label.",
        response_schema_name="rca_trace_judgment_v1",
        response_schema={
            "type": "object",
            "required": ["primary_label", "summary", "confidence"],
            "properties": {
                "primary_label": {"type": "string"},
                "summary": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "additionalProperties": True,
        },
    )


def test_llm_loop_retries_once_when_first_output_is_schema_invalid() -> None:
    client = _FakeModelClient(
        outputs=[
            {"summary": "missing required fields"},
            {"primary_label": "tool_failure", "summary": "Tool failed", "confidence": 0.66},
        ]
    )

    result = run_structured_generation_loop(client=client, request=_request())

    assert client.calls == 2
    assert result.output["primary_label"] == "tool_failure"
    assert result.attempt_count == 2
    assert result.usage.tokens_in == 20
    assert result.usage.tokens_out == 10
    assert result.usage.cost_usd == pytest.approx(0.04)


def test_llm_loop_fails_when_schema_invalid_after_retry() -> None:
    client = _FakeModelClient(outputs=[{"summary": "bad"}, {"still": "bad"}])

    with pytest.raises(ModelOutputInvalidError):
        run_structured_generation_loop(client=client, request=_request())

    assert client.calls == 2
