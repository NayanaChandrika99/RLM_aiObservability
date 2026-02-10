# ABOUTME: Validates Phase 8 OpenAI model client structured-output parsing and usage accounting.
# ABOUTME: Ensures invalid model JSON output is surfaced as deterministic runtime errors.

from __future__ import annotations

import pytest

from investigator.runtime.llm_client import (
    ModelOutputInvalidError,
    OpenAIModelClient,
    StructuredGenerationRequest,
)


class _FakeUsage:
    def __init__(self, *, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, *, output_text: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.output_text = output_text
        self.usage = _FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens)


class _FakeResponses:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_create_kwargs: dict[str, object] | None = None

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.last_create_kwargs = kwargs
        return self._response


class _FakeOpenAIClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.responses = _FakeResponses(response)


def _request() -> StructuredGenerationRequest:
    return StructuredGenerationRequest(
        model_provider="openai",
        model_name="gpt-5-mini",
        temperature=0.0,
        system_prompt="Return only JSON.",
        user_prompt="Summarize the trace.",
        response_schema_name="rca_trace_judgment_v1",
        response_schema={
            "type": "object",
            "required": ["primary_label", "summary", "confidence", "remediation", "gaps"],
            "properties": {
                "primary_label": {"type": "string"},
                "summary": {"type": "string"},
                "confidence": {"type": "number"},
                "remediation": {"type": "array", "items": {"type": "string"}},
                "gaps": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    )


def test_openai_model_client_parses_structured_json_and_usage() -> None:
    fake_client = _FakeOpenAIClient(
        _FakeResponse(
            output_text=(
                '{"primary_label":"tool_failure","summary":"Tool call failed","confidence":0.71,'
                '"remediation":["Retry tool call"],"gaps":[]}'
            ),
            input_tokens=101,
            output_tokens=37,
        )
    )
    client = OpenAIModelClient(openai_client=fake_client)

    result = client.generate_structured(_request())

    assert result.output["primary_label"] == "tool_failure"
    assert result.usage.tokens_in == 101
    assert result.usage.tokens_out == 37
    assert result.usage.cost_usd > 0.0
    assert fake_client.responses.last_create_kwargs is not None
    assert fake_client.responses.last_create_kwargs["text"]["format"]["type"] == "json_schema"


def test_openai_model_client_raises_model_output_invalid_for_non_json() -> None:
    fake_client = _FakeOpenAIClient(_FakeResponse(output_text="not-json", input_tokens=20, output_tokens=10))
    client = OpenAIModelClient(openai_client=fake_client)

    with pytest.raises(ModelOutputInvalidError):
        client.generate_structured(_request())


def test_openai_model_client_omits_temperature_when_request_temperature_is_none() -> None:
    fake_client = _FakeOpenAIClient(
        _FakeResponse(
            output_text=(
                '{"primary_label":"tool_failure","summary":"Tool call failed","confidence":0.71,'
                '"remediation":["Retry tool call"],"gaps":[]}'
            ),
            input_tokens=33,
            output_tokens=11,
        )
    )
    request = StructuredGenerationRequest(
        model_provider="openai",
        model_name="gpt-5-mini",
        temperature=None,
        system_prompt="Return only JSON.",
        user_prompt="Summarize the trace.",
        response_schema_name="rca_trace_judgment_v1",
        response_schema={
            "type": "object",
            "required": ["primary_label", "summary", "confidence", "remediation", "gaps"],
            "properties": {
                "primary_label": {"type": "string"},
                "summary": {"type": "string"},
                "confidence": {"type": "number"},
                "remediation": {"type": "array", "items": {"type": "string"}},
                "gaps": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    )
    client = OpenAIModelClient(openai_client=fake_client)

    result = client.generate_structured(request)

    assert result.output["primary_label"] == "tool_failure"
    assert fake_client.responses.last_create_kwargs is not None
    assert "temperature" not in fake_client.responses.last_create_kwargs


def test_openai_model_client_omits_temperature_for_gpt5_models() -> None:
    fake_client = _FakeOpenAIClient(
        _FakeResponse(
            output_text=(
                '{"primary_label":"tool_failure","summary":"Tool call failed","confidence":0.71,'
                '"remediation":["Retry tool call"],"gaps":[]}'
            ),
            input_tokens=33,
            output_tokens=11,
        )
    )
    request = _request()
    client = OpenAIModelClient(openai_client=fake_client)

    result = client.generate_structured(request)

    assert result.output["primary_label"] == "tool_failure"
    assert fake_client.responses.last_create_kwargs is not None
    assert "temperature" not in fake_client.responses.last_create_kwargs
