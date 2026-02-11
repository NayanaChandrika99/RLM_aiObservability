# ABOUTME: Implements the shared structured-generation runtime loop with deterministic retry behavior.
# ABOUTME: Validates model JSON payloads against request schema and maps exhausted retries to runtime errors.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from investigator.runtime.llm_client import (
    ModelOutputInvalidError,
    RuntimeModelClient,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    StructuredGenerationUsage,
)


@dataclass
class StructuredGenerationLoopResult:
    output: dict[str, Any]
    raw_text: str
    usage: StructuredGenerationUsage
    attempt_count: int


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "number":
        return _is_number(value)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True


def _validate_payload(value: Any, schema: dict[str, Any], *, path: str = "$") -> list[str]:
    errors: list[str] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, str) and not _matches_type(value, schema_type):
        errors.append(f"{path} must be type {schema_type}.")
        return errors

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path} must be one of {enum_values}.")

    if schema_type == "object" and isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for field_name in required:
                if isinstance(field_name, str) and field_name not in value:
                    errors.append(f"{path}.{field_name} is required.")

        properties = schema.get("properties")
        if isinstance(properties, dict):
            for field_name, field_schema in properties.items():
                if field_name not in value:
                    continue
                if not isinstance(field_schema, dict):
                    continue
                errors.extend(
                    _validate_payload(
                        value[field_name],
                        field_schema,
                        path=f"{path}.{field_name}",
                    )
                )
        if schema.get("additionalProperties") is False and isinstance(properties, dict):
            allowed_keys = set(properties.keys())
            for field_name in value:
                if field_name not in allowed_keys:
                    errors.append(f"{path}.{field_name} is not allowed.")

    if schema_type == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(_validate_payload(item, item_schema, path=f"{path}[{index}]"))

    if schema_type == "string" and isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"{path} must have minLength {min_length}.")

    if schema_type == "number" and _is_number(value):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and float(value) < float(minimum):
            errors.append(f"{path} must be >= {minimum}.")
        if isinstance(maximum, (int, float)) and float(value) > float(maximum):
            errors.append(f"{path} must be <= {maximum}.")
    return errors


def _repair_request(
    request: StructuredGenerationRequest,
    *,
    previous_output: str,
    schema_errors: list[str],
) -> StructuredGenerationRequest:
    error_lines = "\n".join(f"- {item}" for item in schema_errors)
    repair_prompt = (
        f"{request.user_prompt}\n\n"
        "The previous JSON output did not satisfy the required schema.\n"
        f"Previous output:\n{previous_output}\n\n"
        f"Schema validation errors:\n{error_lines}\n\n"
        "Return only corrected JSON that matches the schema exactly."
    )
    return replace(request, user_prompt=repair_prompt)


def run_structured_generation_loop(
    *,
    client: RuntimeModelClient,
    request: StructuredGenerationRequest,
    max_attempts: int = 2,
) -> StructuredGenerationLoopResult:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")
    usage_total = StructuredGenerationUsage()
    active_request = request
    last_result: StructuredGenerationResult | None = None
    last_errors: list[str] = []

    for attempt_index in range(1, max_attempts + 1):
        result = client.generate_structured(active_request)
        usage_total.add(result.usage)
        schema_errors = _validate_payload(result.output, request.response_schema)
        if not schema_errors:
            return StructuredGenerationLoopResult(
                output=result.output,
                raw_text=result.raw_text,
                usage=usage_total,
                attempt_count=attempt_index,
            )
        last_result = result
        last_errors = schema_errors
        if attempt_index == max_attempts:
            break
        active_request = _repair_request(
            request=request,
            previous_output=result.raw_text,
            schema_errors=schema_errors,
        )

    details = "; ".join(last_errors) if last_errors else "unknown structured output validation failure"
    sample = last_result.raw_text if last_result is not None else ""
    error = ModelOutputInvalidError(
        f"Model output remained schema-invalid after {max_attempts} attempts: {details}. "
        f"Last output: {sample}"
    )
    error.usage = usage_total
    error.attempt_count = max_attempts
    raise error
