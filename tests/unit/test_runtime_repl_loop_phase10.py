# ABOUTME: Validates Phase 10 shared REPL runtime loop behavior, including tool calls and sub-LLM usage.
# ABOUTME: Ensures SUBMIT finalization, sandbox enforcement, and non-trivial subcall requirements are enforced.

from __future__ import annotations

import json
from typing import Any

from investigator.runtime.contracts import RuntimeBudget
from investigator.runtime.llm_client import (
    StructuredGenerationResult,
    StructuredGenerationUsage,
    TextGenerationResult,
)
from investigator.runtime.repl_loop import ReplLoop
from investigator.runtime.tool_registry import ToolRegistry


class _InspectionAPI:
    def list_spans(self, trace_id: str) -> list[dict[str, Any]]:
        return [{"trace_id": trace_id, "span_id": "root"}]

    def get_spans(self, trace_id: str, type: str | None = None) -> list[dict[str, Any]]:
        del type
        return self.list_spans(trace_id)

    def get_control(self, control_id: str, controls_version: str) -> dict[str, Any]:
        if not isinstance(control_id, str):
            raise TypeError("control_id must be a string")
        return {
            "control_id": control_id,
            "controls_version": controls_version,
            "severity": "high",
            "required_evidence": ["required_error_span"],
        }

    def required_evidence(self, control_id: str, controls_version: str) -> list[str]:
        control = self.get_control(control_id=control_id, controls_version=controls_version)
        return [str(item) for item in (control.get("required_evidence") or []) if str(item)]


class _FakeModelClient:
    model_provider = "openai"

    def __init__(self, *, step_outputs: list[dict[str, Any]], subquery_outputs: list[str]) -> None:
        self._step_outputs = list(step_outputs)
        self._subquery_outputs = list(subquery_outputs)
        self.calls = 0
        self.requests: list[Any] = []

    def generate_structured(self, request):  # noqa: ANN001, ANN201
        self.requests.append(request)
        self.calls += 1
        schema_name = str(getattr(request, "response_schema_name", ""))
        if schema_name == "repl_runtime_step_v1":
            payload = self._step_outputs.pop(0)
            usage = StructuredGenerationUsage(tokens_in=80, tokens_out=16, cost_usd=0.03)
            return StructuredGenerationResult(
                output=payload,
                raw_text=json.dumps(payload, sort_keys=True),
                usage=usage,
            )
        if schema_name == "repl_runtime_subquery_v1":
            answer = self._subquery_outputs.pop(0)
            payload = {"answer": answer}
            usage = StructuredGenerationUsage(tokens_in=40, tokens_out=11, cost_usd=0.01)
            return StructuredGenerationResult(
                output=payload,
                raw_text=json.dumps(payload, sort_keys=True),
                usage=usage,
            )
        raise AssertionError(f"Unexpected schema name: {schema_name}")


class _FakeTextModelClient:
    model_provider = "openai"

    def __init__(self, *, step_outputs: list[dict[str, Any]], text_outputs: list[str]) -> None:
        self._step_outputs = list(step_outputs)
        self._text_outputs = list(text_outputs)
        self.structured_schema_names: list[str] = []
        self.text_calls = 0

    def generate_structured(self, request):  # noqa: ANN001, ANN201
        schema_name = str(getattr(request, "response_schema_name", ""))
        self.structured_schema_names.append(schema_name)
        if schema_name != "repl_runtime_step_v1":
            raise AssertionError(f"Unexpected schema name: {schema_name}")
        payload = self._step_outputs.pop(0)
        usage = StructuredGenerationUsage(tokens_in=80, tokens_out=16, cost_usd=0.03)
        return StructuredGenerationResult(
            output=payload,
            raw_text=json.dumps(payload, sort_keys=True),
            usage=usage,
        )

    def generate_text(
        self,
        *,
        model_name: str,
        temperature: float | None,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
    ) -> TextGenerationResult:
        del model_name, temperature, system_prompt, user_prompt, max_output_tokens
        self.text_calls += 1
        answer = self._text_outputs.pop(0)
        usage = StructuredGenerationUsage(tokens_in=45, tokens_out=9, cost_usd=0.012)
        return TextGenerationResult(text=answer, usage=usage)


def test_repl_loop_executes_code_with_tool_and_subquery_then_submit() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Inspect spans, run one semantic subquery, then submit.",
                "code": (
                    "spans_payload = call_tool('list_spans', trace_id=trace_id)\n"
                    "label_note = llm_query('Classify likely RCA label from span status.')\n"
                    "print(label_note)\n"
                    "SUBMIT("
                    "primary_label='tool_failure',"
                    "summary=f'Repl RCA summary: {label_note}',"
                    "confidence=0.79,"
                    "remediation=['Add retry/backoff around tool calls.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=["tool_failure"],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 RCA non-trivial investigation",
        input_vars={
            "trace_id": "trace-repl",
            "evidence_seed": [
                {
                    "trace_id": "trace-repl",
                    "span_id": "root",
                    "kind": "SPAN",
                    "ref": "root",
                    "excerpt_hash": "seed",
                    "ts": "2026-02-10T00:00:00Z",
                }
            ],
        },
        budget=RuntimeBudget(max_iterations=4, max_subcalls=3),
        require_subquery_for_non_trivial=True,
    )

    assert result.status == "completed"
    assert isinstance(result.output, dict)
    assert result.output.get("primary_label") == "tool_failure"
    assert result.usage.tool_calls == 1
    assert result.usage.llm_subcalls == 1
    assert result.usage.tokens_in > 0
    assert result.usage.cost_usd > 0.0
    assert result.repl_trajectory
    assert result.repl_trajectory[0]["reasoning"]
    tool_trace = result.repl_trajectory[0]["tool_trace"]
    subquery_trace = result.repl_trajectory[0]["subquery_trace"]
    assert len(tool_trace) == 1
    assert tool_trace[0]["tool_name"] == "list_spans"
    assert len(subquery_trace) == 1
    assert "Classify likely RCA label" in subquery_trace[0]["prompt"]
    assert subquery_trace[0]["answer"] == "tool_failure"
    assert "running" in result.state_trajectory


def test_repl_loop_supports_direct_tool_alias_calls_with_positional_trace_id() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Use direct tool helper aliases and submit.",
                "code": (
                    "spans_a = list_spans(trace_id)\n"
                    "spans_b = get_spans(trace_id)\n"
                    "print(len(spans_a.get('result', [])))\n"
                    "print(len(spans_b.get('result', [])))\n"
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary='direct helper aliases worked',"
                    "confidence=0.52,"
                    "remediation=['Keep alias helpers bound in REPL runtime.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 direct tool helper alias compatibility",
        input_vars={"trace_id": "trace-repl", "evidence_seed": []},
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=False,
    )

    assert result.status == "completed"
    assert isinstance(result.output, dict)
    assert result.output.get("summary") == "direct helper aliases worked"
    assert result.repl_trajectory
    output_text = str(result.repl_trajectory[0]["output"] or "")
    assert "[Error]" not in output_text
    tool_trace = result.repl_trajectory[0]["tool_trace"]
    assert len(tool_trace) == 2
    assert tool_trace[0]["tool_name"] == "list_spans"
    assert tool_trace[1]["tool_name"] == "get_spans"
    assert tool_trace[0]["status"] == "ok"
    assert tool_trace[1]["status"] == "ok"


def test_repl_loop_prefers_text_generation_for_subqueries_when_available() -> None:
    model_client = _FakeTextModelClient(
        step_outputs=[
            {
                "reasoning": "Use llm_query and submit once.",
                "code": (
                    "label_note = llm_query('Return one label token for this trace.')\n"
                    "SUBMIT("
                    "primary_label='tool_failure',"
                    "summary=f'RCA summary: {label_note}',"
                    "confidence=0.71,"
                    "remediation=['Retry transient tool failures.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        text_outputs=["tool_failure"],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 RCA non-trivial investigation",
        input_vars={
            "trace_id": "trace-repl",
            "evidence_seed": [
                {
                    "trace_id": "trace-repl",
                    "span_id": "root",
                    "kind": "SPAN",
                    "ref": "root",
                    "excerpt_hash": "seed",
                    "ts": "2026-02-10T00:00:00Z",
                }
            ],
        },
        budget=RuntimeBudget(max_iterations=4, max_subcalls=3),
        require_subquery_for_non_trivial=True,
    )

    assert result.status == "completed"
    assert result.usage.llm_subcalls == 1
    assert model_client.text_calls == 1
    assert "repl_runtime_subquery_v1" not in model_client.structured_schema_names
    assert result.repl_trajectory
    subquery_trace = result.repl_trajectory[0]["subquery_trace"]
    assert len(subquery_trace) == 1
    assert subquery_trace[0]["mode"] == "text"
    assert subquery_trace[0]["answer"] == "tool_failure"


def test_repl_loop_blocks_non_trivial_submit_then_recovers_on_next_step() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Directly submit without subquery.",
                "code": (
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary='Submitted without semantic subquery',"
                    "confidence=0.4,"
                    "remediation=['Review instructions.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            },
            {
                "reasoning": "Use one subquery and resubmit.",
                "code": (
                    "label_note = llm_query('Return one RCA label token from evidence.')\n"
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary=f'Resubmitted with semantic subquery: {label_note}',"
                    "confidence=0.51,"
                    "remediation=['Review instructions.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            },
        ],
        subquery_outputs=["instruction_failure"],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 RCA non-trivial investigation",
        input_vars={
            "trace_id": "trace-repl",
            "evidence_seed": [
                {
                    "trace_id": "trace-repl",
                    "span_id": "root",
                    "kind": "SPAN",
                    "ref": "root",
                    "excerpt_hash": "seed",
                    "ts": "2026-02-10T00:00:00Z",
                }
            ],
        },
        budget=RuntimeBudget(max_iterations=2, max_subcalls=1),
        require_subquery_for_non_trivial=True,
    )

    assert result.status == "completed"
    assert isinstance(result.output, dict)
    assert result.output.get("primary_label") == "instruction_failure"
    assert result.usage.llm_subcalls == 1
    assert len(result.repl_trajectory) == 2
    assert "Guardrail" in result.repl_trajectory[0]["output"]
    assert "llm_query" in result.repl_trajectory[0]["output"]


def test_repl_loop_recovers_after_import_block_error() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Attempt forbidden import.",
                "code": "import os\nprint(os.getcwd())",
            },
            {
                "reasoning": "Use subquery and submit.",
                "code": (
                    "note = llm_query('Return one RCA label token from evidence.')\n"
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary=f'Completed after blocked import: {note}',"
                    "confidence=0.55,"
                    "remediation=['Remove forbidden imports.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            },
        ],
        subquery_outputs=["instruction_failure"],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 sandbox check",
        input_vars={"trace_id": "trace-repl", "evidence_seed": []},
        budget=RuntimeBudget(max_iterations=2, max_subcalls=2),
        require_subquery_for_non_trivial=True,
    )

    assert result.status == "completed"
    assert result.repl_trajectory
    assert "Import statements are blocked in REPL runtime." in result.repl_trajectory[0]["output"]
    assert result.output is not None


def test_repl_loop_fails_on_sandbox_violation_in_code() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Attempt forbidden filesystem call.",
                "code": (
                    "open('x.txt', 'w')\n"
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary='should not complete',"
                    "confidence=0.1,"
                    "remediation=['none'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 sandbox check",
        input_vars={"trace_id": "trace-repl", "evidence_seed": []},
        budget=RuntimeBudget(max_iterations=1),
        require_subquery_for_non_trivial=False,
    )

    assert result.status == "failed"
    assert result.error_code == "SANDBOX_VIOLATION"


def test_repl_loop_supports_globals_lookup_in_submit_step() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Use globals() to read available keys then submit.",
                "code": (
                    "env_keys = sorted([k for k in globals().keys() if isinstance(k, str)])\n"
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary=f'globals_count={len(env_keys)}',"
                    "confidence=0.6,"
                    "remediation=['Keep runtime helpers stable.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 RCA globals submit stability",
        input_vars={"trace_id": "trace-repl", "evidence_seed": []},
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=False,
    )

    assert result.status == "completed"
    assert isinstance(result.output, dict)
    assert str(result.output.get("summary") or "").startswith("globals_count=")
    assert result.repl_trajectory
    assert "[Error]" not in str(result.repl_trajectory[0]["output"] or "")


def test_repl_loop_autofills_controls_version_and_control_id_alias_for_control_tools() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Call get_control with alias key and rely on controls_version from context.",
                "code": (
                    "control_payload = call_tool('get_control', control='control.execution.hard_failures')\n"
                    "required_payload = call_tool('required_evidence', control='control.execution.hard_failures')\n"
                    "print(json.dumps(control_payload))\n"
                    "print(json.dumps(required_payload))\n"
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary='control tools succeeded',"
                    "confidence=0.61,"
                    "remediation=['Normalize tool args before invocation.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 policy tool arg normalization",
        input_vars={
            "trace_id": "trace-repl",
            "controls_version": "controls-v1",
            "evidence_seed": [],
        },
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=False,
    )

    assert result.status == "completed"
    assert result.repl_trajectory
    step = result.repl_trajectory[0]
    assert "[ToolError]" not in str(step.get("output") or "")
    tool_trace = step["tool_trace"]
    assert len(tool_trace) == 2
    assert tool_trace[0]["status"] == "ok"
    assert tool_trace[1]["status"] == "ok"


def test_repl_loop_includes_tool_signatures_in_prompt_context() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Submit immediately.",
                "code": (
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary='done',"
                    "confidence=0.5,"
                    "remediation=['none'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    _ = loop.run(
        objective="Phase10 prompt context completeness",
        input_vars={"trace_id": "trace-repl", "evidence_seed": []},
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=False,
    )

    assert model_client.requests
    first_request = model_client.requests[0]
    prompt_text = str(getattr(first_request, "user_prompt", ""))
    assert "\"tool_signatures\"" in prompt_text


def test_repl_loop_normalizes_control_id_when_control_dict_is_passed() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Use get_control result dict as control_id input.",
                "code": (
                    "control_payload = call_tool('get_control', control='control.execution.hard_failures')\n"
                    "control_obj = control_payload['result']\n"
                    "required_payload = call_tool('required_evidence', control_id=control_obj)\n"
                    "print(json.dumps(required_payload))\n"
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary='normalized control dict',"
                    "confidence=0.62,"
                    "remediation=['Normalize control_id dict inputs.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 control_id dict normalization",
        input_vars={"trace_id": "trace-repl", "controls_version": "controls-v1", "evidence_seed": []},
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=False,
    )

    assert result.status == "completed"
    assert result.repl_trajectory
    assert "[ToolError]" not in str(result.repl_trajectory[0]["output"] or "")


def test_repl_loop_allows_safe_json_import_for_repl_fallback_parsing() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Import json and submit parsed value.",
                "code": (
                    "import json as _json\n"
                    "payload = _json.loads('{\"score\": 7}')\n"
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary=f\"score={payload['score']}\","
                    "confidence=0.63,"
                    "remediation=['Use safe imports only.'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 safe import fallback",
        input_vars={"trace_id": "trace-repl", "evidence_seed": []},
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=False,
    )

    assert result.status == "completed"
    assert isinstance(result.output, dict)
    assert result.output.get("summary") == "score=7"


def test_repl_loop_fails_hard_on_last_iteration_execution_error_for_rca() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Trigger execution failure on final iteration.",
                "code": (
                    "print(undefined_symbol)\n"
                    "SUBMIT("
                    "primary_label='data_schema_mismatch',"
                    "summary='should not execute submit',"
                    "confidence=0.1,"
                    "remediation=['none'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 RCA fallback recovery",
        input_vars={
            "trace_id": "trace-repl",
            "allowed_labels": ["tool_failure", "data_schema_mismatch"],
            "deterministic_label_hint": "data_schema_mismatch",
            "evidence_seed": [],
        },
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=True,
    )

    assert result.status == "failed"
    assert result.output is None
    assert result.error_code == "SUBMIT_DEADLINE_REACHED"
    assert "undefined_symbol" in str(result.error_message or "")
    assert result.repl_trajectory
    assert "[Recovery] Deterministic fallback SUBMIT applied" not in str(result.repl_trajectory[0]["output"])


def test_repl_loop_fails_hard_on_last_iteration_execution_error_for_compliance() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Trigger execution failure on final iteration.",
                "code": (
                    "print(undefined_policy_var)\n"
                    "SUBMIT("
                    "pass_fail='insufficient_evidence',"
                    "confidence=0.1,"
                    "rationale='should not execute submit',"
                    "covered_requirements=[],"
                    "missing_evidence=['required_error_span'],"
                    "evidence_refs=[default_evidence],"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 compliance fallback recovery",
        input_vars={
            "trace_id": "trace-repl",
            "controls_version": "controls-v1",
            "control": {"control_id": "control.execution.hard_failures"},
            "required_evidence": ["required_error_span"],
            "default_evidence": {
                "trace_id": "trace-repl",
                "span_id": "root",
                "kind": "SPAN",
                "ref": "root",
                "excerpt_hash": "seed",
                "ts": "2026-02-10T00:00:00Z",
            },
            "evidence_seed": [],
        },
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=True,
    )

    assert result.status == "failed"
    assert result.output is None
    assert result.error_code == "SUBMIT_DEADLINE_REACHED"
    assert "undefined_policy_var" in str(result.error_message or "")


def test_repl_loop_fails_hard_when_deadline_hits_without_submit() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Do work but do not submit.",
                "code": "x = 1\nprint('no submit in this step')",
            },
            {
                "reasoning": "Still do not submit after enforcement retry.",
                "code": "x = 2\nprint('still no submit')",
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 fallback on submit deadline",
        input_vars={
            "trace_id": "trace-repl",
            "allowed_labels": ["tool_failure", "data_schema_mismatch"],
            "deterministic_label_hint": "tool_failure",
            "evidence_seed": [],
        },
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=True,
    )

    assert result.status == "failed"
    assert result.output is None
    assert result.error_code == "MODEL_OUTPUT_INVALID"
    assert "must include SUBMIT" in str(result.error_message or "")
    assert result.repl_trajectory == []


def test_repl_loop_fails_hard_on_planning_budget_exhaustion_after_progress() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "First step does not submit.",
                "code": "x = 1\nprint('first step complete')",
            },
            {
                "reasoning": "Second step would submit but planning budget should trigger first.",
                "code": (
                    "SUBMIT("
                    "primary_label='tool_failure',"
                    "summary='late submit',"
                    "confidence=0.9,"
                    "remediation=['none'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            },
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 fallback on planning budget exhaustion",
        input_vars={
            "trace_id": "trace-repl",
            "allowed_labels": ["tool_failure", "data_schema_mismatch"],
            "deterministic_label_hint": "tool_failure",
            "evidence_seed": [],
        },
        budget=RuntimeBudget(max_iterations=3, max_subcalls=1, max_tokens_total=120),
        require_subquery_for_non_trivial=True,
    )

    assert result.status == "failed"
    assert result.output is None
    assert result.error_code == "BUDGET_EXHAUSTED"
    assert "max_tokens_total reached" in str(result.error_message or "")


def test_repl_loop_includes_pre_filter_context_in_prompt_context() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Submit immediately.",
                "code": (
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary='done',"
                    "confidence=0.5,"
                    "remediation=['none'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    _ = loop.run(
        objective="Phase10 pre-filter context prompt",
        input_vars={"trace_id": "trace-repl", "evidence_seed": []},
        pre_filter_context={
            "hot_spans": [{"span_id": "root", "status_code": "ERROR"}],
            "branch_span_ids": ["root"],
            "preliminary_label": "instruction_failure",
        },
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=False,
    )

    assert model_client.requests
    first_request = model_client.requests[0]
    prompt_text = str(getattr(first_request, "user_prompt", ""))
    assert "\"pre_filter_context\"" in prompt_text
    assert "\"hot_spans\"" in prompt_text
    assert "\"branch_span_ids\"" in prompt_text


def test_repl_loop_includes_env_tips_in_prompt_context() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Submit immediately.",
                "code": (
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary='done',"
                    "confidence=0.5,"
                    "remediation=['none'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    _ = loop.run(
        objective="Phase10 env tips prompt",
        input_vars={"trace_id": "trace-repl", "evidence_seed": []},
        env_tips="Prefer branch-root evidence first and avoid repeated refetch loops.",
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=False,
    )

    assert model_client.requests
    first_request = model_client.requests[0]
    prompt_text = str(getattr(first_request, "user_prompt", ""))
    assert "\"env_tips\"" in prompt_text
    assert "Prefer branch-root evidence first" in prompt_text


def test_repl_loop_includes_non_trivial_subquery_requirement_in_prompt_context() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Submit immediately.",
                "code": (
                    "SUBMIT("
                    "primary_label='instruction_failure',"
                    "summary='done',"
                    "confidence=0.5,"
                    "remediation=['none'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            }
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    _ = loop.run(
        objective="Phase10 non-trivial subquery prompt contract",
        input_vars={"trace_id": "trace-repl", "evidence_seed": []},
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=True,
    )

    assert model_client.requests
    first_request = model_client.requests[0]
    prompt_text = str(getattr(first_request, "user_prompt", ""))
    assert "\"require_subquery_for_non_trivial\": true" in prompt_text


def test_repl_loop_regenerates_once_when_submit_required_and_first_plan_omits_submit() -> None:
    model_client = _FakeModelClient(
        step_outputs=[
            {
                "reasoning": "Inspect before finalizing.",
                "code": "print('planned step without submit')",
            },
            {
                "reasoning": "Finalize now with submit.",
                "code": (
                    "SUBMIT("
                    "primary_label='tool_failure',"
                    "summary='finalized after submit enforcement',"
                    "confidence=0.61,"
                    "remediation=['stabilize finalize behavior'],"
                    "evidence_refs=evidence_seed,"
                    "gaps=[]"
                    ")"
                ),
            },
        ],
        subquery_outputs=[],
    )
    loop = ReplLoop(
        tool_registry=ToolRegistry(inspection_api=_InspectionAPI()),
        model_client=model_client,
        model_name="gpt-5-mini",
        temperature=0.0,
    )

    result = loop.run(
        objective="Phase10 submit enforcement",
        input_vars={"trace_id": "trace-repl", "evidence_seed": []},
        budget=RuntimeBudget(max_iterations=1, max_subcalls=1),
        require_subquery_for_non_trivial=False,
    )

    assert result.status == "completed"
    assert isinstance(result.output, dict)
    assert result.output.get("summary") == "finalized after submit enforcement"
    assert len(result.repl_trajectory) == 1
    assert "SUBMIT(" in str(result.repl_trajectory[0]["code"] or "")
    assert len(model_client.requests) == 2
    second_prompt = str(getattr(model_client.requests[1], "user_prompt", ""))
    assert "\"submit_enforcement\"" in second_prompt
