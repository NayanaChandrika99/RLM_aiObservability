# ABOUTME: Defines shared structured-generation model client interfaces for runtime model calls.
# ABOUTME: Provides an OpenAI implementation that returns parsed JSON plus token and cost usage.

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol


DEFAULT_MODEL_PRICING_USD_PER_MILLION = {
    "gpt-5-mini": {"input": 0.25, "output": 2.0},
    "gpt-5.2": {"input": 2.0, "output": 8.0},
}
DEFAULT_OPENAI_REQUEST_TIMEOUT_SEC = 120.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class StructuredGenerationRequest:
    model_provider: str
    model_name: str
    temperature: float | None
    system_prompt: str
    user_prompt: str
    response_schema_name: str
    response_schema: dict[str, Any]
    max_output_tokens: int | None = None


@dataclass
class StructuredGenerationUsage:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def add(self, other: StructuredGenerationUsage) -> StructuredGenerationUsage:
        self.tokens_in += int(other.tokens_in)
        self.tokens_out += int(other.tokens_out)
        self.cost_usd += float(other.cost_usd)
        return self


@dataclass
class StructuredGenerationResult:
    output: dict[str, Any]
    raw_text: str
    usage: StructuredGenerationUsage


@dataclass
class TextGenerationResult:
    text: str
    usage: StructuredGenerationUsage


class ModelOutputInvalidError(RuntimeError):
    pass


class RuntimeModelClient(Protocol):
    model_provider: str

    def generate_structured(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        raise NotImplementedError


class OpenAIModelClient:
    model_provider = "openai"

    def __init__(
        self,
        *,
        openai_client: Any | None = None,
        pricing_usd_per_million: dict[str, dict[str, float]] | None = None,
        request_timeout_sec: float | None = DEFAULT_OPENAI_REQUEST_TIMEOUT_SEC,
    ) -> None:
        if openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI()
        else:
            self._openai_client = openai_client
        self._pricing_usd_per_million = pricing_usd_per_million or DEFAULT_MODEL_PRICING_USD_PER_MILLION
        self._request_timeout_sec = request_timeout_sec

    @staticmethod
    def _supports_temperature(model_name: str) -> bool:
        normalized = str(model_name or "").strip().lower()
        if normalized.startswith("gpt-5"):
            return False
        return True

    @staticmethod
    def _supports_reasoning_effort(model_name: str) -> bool:
        normalized = str(model_name or "").strip().lower()
        return normalized.startswith("gpt-5")

    def _estimate_cost(self, *, model_name: str, tokens_in: int, tokens_out: int) -> float:
        pricing = self._pricing_usd_per_million.get(model_name)
        if not isinstance(pricing, dict):
            return 0.0
        input_rate = _safe_float(pricing.get("input"), default=0.0)
        output_rate = _safe_float(pricing.get("output"), default=0.0)
        input_cost = (tokens_in / 1_000_000.0) * input_rate
        output_cost = (tokens_out / 1_000_000.0) * output_rate
        return input_cost + output_cost

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        raw_text = str(getattr(response, "output_text", "") or "").strip()
        if raw_text:
            return raw_text
        output_items = getattr(response, "output", None)
        if not isinstance(output_items, list):
            return ""
        collected: list[str] = []
        for item in output_items:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                block_text = getattr(block, "text", None)
                if block_text is None and isinstance(block, dict):
                    block_text = block.get("text")
                if isinstance(block_text, str) and block_text.strip():
                    collected.append(block_text.strip())
        return "\n".join(collected).strip()

    def generate_text(
        self,
        *,
        model_name: str,
        temperature: float | None,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
    ) -> TextGenerationResult:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "instructions": system_prompt,
            "input": user_prompt,
        }
        if self._supports_reasoning_effort(model_name):
            kwargs["reasoning"] = {"effort": "minimal"}
        if temperature is not None and self._supports_temperature(model_name):
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if self._request_timeout_sec is not None:
            kwargs["timeout"] = float(self._request_timeout_sec)
        response = self._openai_client.responses.create(**kwargs)
        text = self._extract_response_text(response)
        if not text:
            raise ModelOutputInvalidError("Model response did not include text output.")
        usage = getattr(response, "usage", None)
        tokens_in = _safe_int(getattr(usage, "input_tokens", 0), default=0)
        tokens_out = _safe_int(getattr(usage, "output_tokens", 0), default=0)
        cost_usd = self._estimate_cost(
            model_name=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        return TextGenerationResult(
            text=text,
            usage=StructuredGenerationUsage(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
            ),
        )

    def generate_structured(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        strict_schema = True
        if str(request.response_schema_name) in {
            "recursive_runtime_action_v1",
            "repl_runtime_step_v1",
        }:
            strict_schema = False
        kwargs: dict[str, Any] = {
            "model": request.model_name,
            "instructions": request.system_prompt,
            "input": request.user_prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.response_schema_name,
                    "schema": request.response_schema,
                    "strict": strict_schema,
                }
            },
        }
        if request.temperature is not None and self._supports_temperature(request.model_name):
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if self._request_timeout_sec is not None:
            kwargs["timeout"] = float(self._request_timeout_sec)
        response = self._openai_client.responses.create(**kwargs)
        raw_text = self._extract_response_text(response)
        if not raw_text.strip():
            raise ModelOutputInvalidError("Model response did not include structured output text.")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ModelOutputInvalidError(f"Model returned non-JSON structured output: {exc}") from exc
        if not isinstance(payload, dict):
            raise ModelOutputInvalidError("Model structured output must be a JSON object.")
        usage = getattr(response, "usage", None)
        tokens_in = _safe_int(getattr(usage, "input_tokens", 0), default=0)
        tokens_out = _safe_int(getattr(usage, "output_tokens", 0), default=0)
        cost_usd = self._estimate_cost(
            model_name=request.model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        return StructuredGenerationResult(
            output=payload,
            raw_text=raw_text,
            usage=StructuredGenerationUsage(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
            ),
        )
