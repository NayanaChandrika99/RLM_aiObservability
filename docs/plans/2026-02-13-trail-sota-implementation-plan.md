# TRAIL SOTA Experimentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the experimentation infrastructure for rapid TRAIL GAIA benchmark iteration, starting with enhanced single-pass prompt (Approach A).

**Architecture:** A new experiment runner CLI (`trail_experiment.py`) drives traces through an enhanced prompt pipeline (`trail_prompt_v2.py`), auto-scores via the official scorer, and logs results. A dev subset manifest enables fast iteration on 18 traces. Single-pass mode added to existing `trail_agent.py` uses litellm directly (same as the baseline).

**Tech Stack:** Python 3.11+, litellm, sklearn, scipy, numpy, uv, pytest

---

## Task 1: Dev Subset Manifest

**Files:**
- Create: `arcgentica/dev_subset_manifest.json`
- Test: `tests/unit/test_trail_experiment_phase11.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_trail_experiment_phase11.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_trail_experiment_phase11.py::test_dev_subset_manifest_is_valid_json -v`
Expected: FAIL with "Manifest not found"

**Step 3: Write the manifest file**

```json
{
  "subset_id": "dev18",
  "description": "18 representative GAIA traces covering all 21 TRAIL categories (easy/medium/hard mix)",
  "trace_ids": [
    "0035f455b3ff2295167a844f04d85d34",
    "0140b3f657eddf76ca82f72c49ac8e58",
    "0adc4f3b99d9564d32811e913cc9d248",
    "18efa24e637b9423f34180d1f2041d3e",
    "27a6c5ebc3311542156fdde857a0035f",
    "3205fa0cb2135fe671bf7cd0e5a26151",
    "3637c5845140adbf29c565923c20e94d",
    "396b6aa1ab86eb2e20d27582eb5eebd9",
    "41b597524173272503073a0799ac523c",
    "59365b27641e501d105b0e8f5e7c5af7",
    "672d36d8ecc4816738433c75136eb99d",
    "6e64c2e543327a7c2f2ce3d26ced94d1",
    "72877db591837666d500b459fb3cf29d",
    "7bf0addde339e4cac9dd3b772232a7e0",
    "860f9d45f2e50bfecb190bb26eff1f32",
    "876eb108c8650d4ada63a8d39aa1e96c",
    "915d2c66879657f694f88e0ed6f02cf5",
    "99f6b447779ba86b3cff2caede832d59"
  ]
}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_trail_experiment_phase11.py::test_dev_subset_manifest_is_valid_json -v`
Expected: PASS

**Step 5: Commit**

```bash
git add arcgentica/dev_subset_manifest.json tests/unit/test_trail_experiment_phase11.py
git commit -m "feat(trail): add dev18 subset manifest for experiment iteration"
```

---

## Task 2: Enhanced Prompt V2 (`trail_prompt_v2.py`)

This is the core prompt improvement. The file contains:
- `TRAIL_SINGLE_PASS_PROMPT_V2`: Full prompt with taxonomy + detection hints + location rules + examples + score guidance
- `build_single_pass_message()`: Function that combines prompt + trace content (with smart truncation)

**Files:**
- Create: `arcgentica/trail_prompt_v2.py`
- Test: `tests/unit/test_trail_experiment_phase11.py` (append tests)

### Step 1: Write the failing tests

Append to `tests/unit/test_trail_experiment_phase11.py`:

```python
from arcgentica.trail_prompt_v2 import (
    TRAIL_SINGLE_PASS_PROMPT_V2,
    build_single_pass_message,
    smart_truncate_trace,
)
from arcgentica.trail_common import TRAIL_LEAF_CATEGORIES


def test_prompt_v2_contains_all_leaf_categories() -> None:
    for category in TRAIL_LEAF_CATEGORIES:
        assert category in TRAIL_SINGLE_PASS_PROMPT_V2, f"Missing category: {category}"


def test_prompt_v2_contains_location_rules() -> None:
    prompt = TRAIL_SINGLE_PASS_PROMPT_V2
    assert "Resource Abuse" in prompt
    assert "first" in prompt.lower()
    assert "last" in prompt.lower()
    assert "span_id" in prompt


def test_prompt_v2_contains_detection_hints() -> None:
    prompt = TRAIL_SINGLE_PASS_PROMPT_V2
    assert "DETECT:" in prompt or "detect:" in prompt.lower()


def test_prompt_v2_contains_score_guidance() -> None:
    prompt = TRAIL_SINGLE_PASS_PROMPT_V2
    assert "reliability_score" in prompt
    assert "security_score" in prompt
    assert "instruction_adherence_score" in prompt
    assert "plan_opt_score" in prompt


def test_smart_truncate_preserves_small_trace() -> None:
    trace = {
        "trace_id": "abc123",
        "spans": [
            {
                "span_id": "span_1",
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "hello world"}],
                "child_spans": [],
            }
        ],
    }
    result = smart_truncate_trace(trace, max_chars=500_000)
    # Small trace should pass through unchanged
    assert result["trace_id"] == "abc123"
    assert len(result["spans"]) == 1


def test_smart_truncate_truncates_large_trace() -> None:
    # Build a trace with many large spans
    spans = []
    for i in range(50):
        spans.append({
            "span_id": f"span_{i:04d}",
            "span_name": f"op_{i}",
            "status_code": "Error" if i == 5 else "Unset",
            "status_message": "some failure" if i == 5 else "",
            "span_attributes": {},
            "logs": [{"body": "x" * 2000}],
            "child_spans": [],
        })
    trace = {"trace_id": "big_trace", "spans": spans}
    result = smart_truncate_trace(trace, max_chars=10_000)
    result_text = json.dumps(result)
    assert len(result_text) <= 15_000  # some overhead is ok
    # Error span should be preserved in full
    error_spans = [s for s in _walk_all_spans(result["spans"]) if s.get("status_code") == "Error"]
    assert len(error_spans) >= 1
    # Span index should be appended
    assert "span_index" in result


def _walk_all_spans(spans: list) -> list:
    flat = []
    for s in spans:
        flat.append(s)
        flat.extend(_walk_all_spans(s.get("child_spans") or []))
    return flat


def test_build_single_pass_message_returns_string() -> None:
    trace = {
        "trace_id": "test_trace",
        "spans": [
            {
                "span_id": "sp1",
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [],
                "child_spans": [],
            }
        ],
    }
    msg = build_single_pass_message(trace)
    assert isinstance(msg, str)
    assert "test_trace" in msg or "sp1" in msg
    # Should contain the prompt
    assert "errors" in msg.lower()
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_trail_experiment_phase11.py -k "prompt_v2 or smart_truncate or build_single_pass" -v`
Expected: FAIL with import error

### Step 3: Write `trail_prompt_v2.py`

Create `arcgentica/trail_prompt_v2.py`:

```python
# ABOUTME: Enhanced TRAIL prompts (v2) with category detection hints, location rules, and LLM-generated scores.
# ABOUTME: Includes smart trace truncation for context window overflow and single-pass message builder.

from __future__ import annotations

import json
from typing import Any

try:
    from .trail_common import TRAIL_LEAF_CATEGORIES
except ImportError:
    from trail_common import TRAIL_LEAF_CATEGORIES


TRAIL_SINGLE_PASS_PROMPT_V2 = """\
You are an expert trace analyst for the TRAIL benchmark. Analyze the LLM agent trace below and find ALL errors using the taxonomy provided.

# Trace Structure Guide

Before analyzing, understand the trace structure:
- Each span has a unique `span_id` -- use these exact IDs for error locations.
- `child_spans` contain nested execution steps within a parent span.
- `logs[].body` contains function call names, arguments, and outputs.
- Check `status_code` and `status_message` for execution errors.
- `span_name` identifies the operation type (e.g., "LiteLLMModel.__call__", "FinalAnswerTool").
- TOOL spans have `tool.name` in their attributes or `span_name`.

# Taxonomy with Detection Guidance

├── Reasoning Errors
│   ├── Hallucinations
│   │   ├── Language-only
│   │   │   DETECT: The LLM output text contains factual claims (numbers, dates, names)
│   │   │   that are not supported by any tool output or retrieved document in the trace.
│   │   │   Look for confident assertions without a preceding tool call that provided the data.
│   │   └── Tool-related (fabricating tool outputs/capabilities)
│   │       DETECT: The LLM output claims "According to the search results..." or
│   │       "I have verified using tool X..." but NO TOOL span with that tool.name exists
│   │       as a child/sibling span. Compare stated tool usage against actual TOOL spans.
│   ├── Information Processing
│   │   ├── Poor Information Retrieval (tried to find irrelevant information)
│   │   │   DETECT: Tool calls return results, but the agent uses them for a purpose
│   │   │   unrelated to the original task. Or: search queries are off-topic or too vague.
│   │   └── Tool Output Misinterpretation (wrong assumptions about tool output)
│   │       DETECT: Tool output shows value X, but the agent's next reasoning step
│   │       treats it as value Y or draws an incorrect conclusion from the data.
│   ├── Decision Making
│   │   ├── Incorrect Problem Identification (misunderstood the task)
│   │   │   DETECT: The agent's stated understanding of the task diverges from the
│   │   │   actual task description. Or: the agent solves a different problem than asked.
│   │   └── Tool Selection Errors (used the wrong tool)
│   │       DETECT: The agent uses tool A when tool B would be more appropriate for
│   │       the subtask. Or: tries to use a calculator for a web search task.
│   └── Output Generation
│       ├── Formatting Errors (code execution or output structuring errors)
│       │   DETECT: Code syntax errors, wrong output format, missing required fields,
│       │   incorrect data type in final answer, or structural issues in generated code.
│       └── Instruction Non-compliance (failed to follow task instructions)
│           DETECT: The agent ignores explicit instructions (e.g., missing <end_plan> tag,
│           not using required format, skipping mandatory steps, answering differently
│           than instructed).
├── System Execution Errors
│   ├── Configuration
│   │   ├── Tool Definition Issues (tool defined incorrectly by user)
│   │   │   DETECT: Tool description/schema in the trace doesn't match what the tool
│   │   │   actually does. Or: tool parameters are inconsistent with tool name.
│   │   └── Environment Setup Errors (permission problems, missing API keys)
│   │       DETECT: TOOL spans with status_code="Error" containing FileNotFoundError,
│   │       InterpreterError ("not permitted to evaluate"), PermissionError,
│   │       or UnboundLocalError in an internal tool path.
│   ├── API Issues
│   │   ├── Rate Limiting (like 429)
│   │   │   DETECT: HTTP 429 responses, "rate limit exceeded" messages in span logs.
│   │   ├── Authentication Errors (like 401/403)
│   │   │   DETECT: HTTP 401/403 responses, "unauthorized" or "forbidden" in span logs.
│   │   ├── Service Errors (like 500)
│   │   │   DETECT: HTTP 500 responses, "internal server error" in span logs.
│   │   └── Resource Not Found (like 404)
│   │       DETECT: HTTP 404 responses, "not found" messages, missing resource errors.
│   └── Resource Management
│       ├── Resource Exhaustion (includes memory overflow)
│       │   DETECT: MemoryError, "resource exhausted", OOM messages in span logs.
│       └── Timeout Issues (system took too long)
│           DETECT: "timed out", "timeout", deadline exceeded messages in span logs.
├── Planning and Coordination Errors
│   ├── Context Management
│   │   ├── Context Handling Failures (window overflow, state tracking, forgetting context)
│   │   │   DETECT: The agent performs an action for context/page A, but the Address/state
│   │   │   field in the preceding tool response shows it is on context/page B. Or: the
│   │   │   agent receives an error message but repeats the same mistake. Or: earlier
│   │   │   information is lost/contradicted in later reasoning.
│   │   └── Resource Abuse (called tool excessively due to memory issues)
│   │       DETECT: The same tool is called 5+ times with identical or near-identical
│   │       parameters. Count repeated tool calls in the trace.
│   └── Task Management
│       ├── Goal Deviation (deviated from the task or subtask)
│       │   DETECT: The agent's actions diverge from the stated goal. It works on
│       │   something tangential or abandons the main objective for a side quest.
│       └── Task Orchestration (subtask coordination, progress monitoring)
│           DETECT: A multi-step plan is written but the only TOOL child span is
│           final_answer -- no intermediate tools were called. Or: the agent says
│           "I will now use tool X" but no X span follows. Low ratio of actual
│           TOOL spans to planned steps.

# CRITICAL LOCATION RULES

- For ALL errors EXCEPT "Resource Abuse": use the span_id where the error FIRST appears in the trace.
- For "Resource Abuse" ONLY: use the span_id of the LAST instance of excessive tool calling.
- The location MUST be an exact `span_id` from the trace (e.g., "3bcc157b63d51414").
- Prefer the most specific (deepest child) span where the error manifests.
- Do NOT use a parent span when a child span is more precise.

# Output Format

Return strict JSON (no markdown, no extra text):

{
    "errors": [
        {
            "category": "[EXACT leaf category name from taxonomy]",
            "location": "[exact span_id from trace]",
            "evidence": "[direct quote or specific observation from the trace]",
            "description": "[what went wrong and why it matters]",
            "impact": "HIGH|MEDIUM|LOW"
        }
    ],
    "scores": [
        {
            "reliability_score": 3,
            "reliability_reasoning": "[How reliable were outputs? Hallucinations and incorrect info weigh more than formatting issues]",
            "security_score": 5,
            "security_reasoning": "[Only reduce from 5 for auth errors, data exposure, or unsafe API calls]",
            "instruction_adherence_score": 4,
            "instruction_adherence_reasoning": "[Did it follow instructions? Missing tags are LOW impact; ignoring core task is HIGH]",
            "plan_opt_score": 3,
            "plan_opt_reasoning": "[Was the plan efficient? Redundant calls, failed retries without adaptation reduce this]",
            "overall": 3.75
        }
    ]
}

# Example 1: Reasoning Errors

Given a trace where the agent claims "According to the USGS record..." but no search tool span exists:

{
    "errors": [
        {
            "category": "Instruction Non-compliance",
            "location": "98fa1dda65ab168b",
            "evidence": "The output plan ends with step 6 instead of ending with <end_plan>.",
            "description": "The system failed to append the required <end_plan> tag at the end of its plan.",
            "impact": "LOW"
        },
        {
            "category": "Tool-related",
            "location": "bc20feefb97e11e5",
            "evidence": "I have verified that the report came from the USGS database. According to the USGS record, prior to 2020 a population was reported from Biscayne Bay.",
            "description": "The system claimed to have obtained USGS information but no search_agent tool span exists in the trace. It fabricated the tool interaction.",
            "impact": "HIGH"
        },
        {
            "category": "Goal Deviation",
            "location": "bc20feefb97e11e5",
            "evidence": "The plan lists steps involving search_agent, extraction, and translation. The Thought immediately calls final_answer, omitting all intermediate steps.",
            "description": "The system deviated from its own plan by skipping all data-gathering steps and jumping to final_answer.",
            "impact": "HIGH"
        }
    ],
    "scores": [
        {
            "reliability_score": 1,
            "reliability_reasoning": "System hallucinated information and failed to execute its plan, providing an unverified answer.",
            "security_score": 5,
            "security_reasoning": "No security issues detected.",
            "instruction_adherence_score": 2,
            "instruction_adherence_reasoning": "Created a plan but ignored the fundamental instruction to use tools for data gathering.",
            "plan_opt_score": 2,
            "plan_opt_reasoning": "Initial plan was reasonable but execution was highly suboptimal due to skipping all tool calls.",
            "overall": 2.5
        }
    ]
}

# Example 2: System and Planning Errors

Given a trace where a tool throws FileNotFoundError and the agent writes a 7-step plan but only calls final_answer:

{
    "errors": [
        {
            "category": "Environment Setup Errors",
            "location": "a1b2c3d4e5f6g7h8",
            "evidence": "InterpreterError: Forbidden operation: open() is not permitted.",
            "description": "The tool environment does not allow file system operations, causing the tool call to fail with a permission error.",
            "impact": "HIGH"
        },
        {
            "category": "Task Orchestration",
            "location": "i9j0k1l2m3n4o5p6",
            "evidence": "Plan lists 7 steps but only final_answer tool span exists as child. No intermediate tool spans present.",
            "description": "The agent wrote a detailed multi-step plan but executed none of the intermediate steps, jumping straight to final_answer.",
            "impact": "HIGH"
        },
        {
            "category": "Context Handling Failures",
            "location": "q7r8s9t0u1v2w3x4",
            "evidence": "Agent action targets page 'Settings' but Address field shows current page is 'Home'.",
            "description": "The agent lost track of its navigation state, performing actions intended for a different page context.",
            "impact": "MEDIUM"
        }
    ],
    "scores": [
        {
            "reliability_score": 2,
            "reliability_reasoning": "Environment errors and incomplete execution make outputs unreliable.",
            "security_score": 5,
            "security_reasoning": "No security vulnerabilities detected.",
            "instruction_adherence_score": 1,
            "instruction_adherence_reasoning": "Failed to execute the core task workflow, skipping all planned intermediate steps.",
            "plan_opt_score": 1,
            "plan_opt_reasoning": "Plan was created but never executed, representing maximum planning waste.",
            "overall": 2.25
        }
    ]
}

# Rules

- Be EXHAUSTIVE: find ALL errors in the trace.
- Only use the final subcategories listed above (e.g., "Resource Not Found", not "API Issues").
- If no errors found, return {"errors": [], "scores": [{...with all 5s...}]}.
- Evidence must be a direct quote or specific observation from the trace content.
- Impact: HIGH = affects final answer correctness, MEDIUM = affects process quality, LOW = cosmetic/minor.
- Scores are 0-5 (integer). Overall = average of the four scores (can be float).

# Trace Data

{trace}
"""


def _walk_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested span tree into a list."""
    flat: list[dict[str, Any]] = []
    for span in spans:
        flat.append(span)
        children = span.get("child_spans") or []
        if isinstance(children, list):
            flat.extend(_walk_spans([c for c in children if isinstance(c, dict)]))
    return flat


def _span_has_error_signal(span: dict[str, Any]) -> bool:
    """Check if a span has error indicators worth preserving in full."""
    status = span.get("status_code", "")
    if isinstance(status, str) and status.lower() in ("error", "err"):
        return True
    msg = span.get("status_message", "")
    if isinstance(msg, str):
        lowered = msg.lower()
        for kw in ("error", "failed", "exception", "timeout", "traceback", "429", "401", "403", "404", "500"):
            if kw in lowered:
                return True
    logs = span.get("logs") or []
    for log in logs:
        if not isinstance(log, dict):
            continue
        body = log.get("body", "")
        body_str = json.dumps(body) if not isinstance(body, str) else body
        lowered = body_str.lower()
        for kw in ("error", "failed", "exception", "timeout", "traceback"):
            if kw in lowered:
                return True
    return False


def _truncate_span_logs(span: dict[str, Any], max_log_chars: int) -> dict[str, Any]:
    """Return a shallow copy of span with truncated logs."""
    result = dict(span)
    logs = span.get("logs")
    if not isinstance(logs, list) or not logs:
        return result
    truncated_logs = []
    for log in logs:
        if not isinstance(log, dict):
            truncated_logs.append(log)
            continue
        body = log.get("body", "")
        body_str = json.dumps(body) if not isinstance(body, str) else body
        if len(body_str) > max_log_chars:
            truncated_log = dict(log)
            truncated_log["body"] = body_str[:max_log_chars] + "...[truncated]"
            truncated_logs.append(truncated_log)
        else:
            truncated_logs.append(log)
    result["logs"] = truncated_logs
    # Recurse into child_spans
    children = span.get("child_spans")
    if isinstance(children, list) and children:
        result["child_spans"] = [
            _truncate_span_logs(c, max_log_chars) if isinstance(c, dict) else c
            for c in children
        ]
    return result


def smart_truncate_trace(
    trace: dict[str, Any],
    max_chars: int = 400_000,
    normal_log_chars: int = 200,
    error_log_chars: int = 2000,
) -> dict[str, Any]:
    """Truncate large traces while preserving error spans and span index.

    Strategy:
    1. If trace JSON fits in max_chars, return as-is.
    2. Otherwise, keep error spans in full, truncate normal span logs.
    3. Always append a span_index listing all span_ids for location reference.
    """
    trace_text = json.dumps(trace)
    if len(trace_text) <= max_chars:
        return trace

    spans = trace.get("spans", [])
    if not isinstance(spans, list):
        return trace

    flat = _walk_spans(spans)

    # Build span index (always included)
    span_index = []
    for s in flat:
        sid = s.get("span_id", "")
        sname = s.get("span_name", "")
        status = s.get("status_code", "")
        span_index.append({"span_id": sid, "span_name": sname, "status_code": status})

    def _truncate_tree(span_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for span in span_list:
            if not isinstance(span, dict):
                continue
            if _span_has_error_signal(span):
                truncated = _truncate_span_logs(span, error_log_chars)
            else:
                truncated = _truncate_span_logs(span, normal_log_chars)
            result.append(truncated)
        return result

    truncated_spans = _truncate_tree(spans)
    result = {
        "trace_id": trace.get("trace_id", ""),
        "spans": truncated_spans,
        "span_index": span_index,
    }
    return result


def build_single_pass_message(
    trace: dict[str, Any],
    max_trace_chars: int = 400_000,
) -> str:
    """Build the complete prompt message for a single-pass TRAIL analysis."""
    truncated = smart_truncate_trace(trace, max_chars=max_trace_chars)
    trace_text = json.dumps(truncated, indent=2)
    return TRAIL_SINGLE_PASS_PROMPT_V2.replace("{trace}", trace_text)
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_trail_experiment_phase11.py -k "prompt_v2 or smart_truncate or build_single_pass" -v`
Expected: PASS (all 7 tests)

### Step 5: Commit

```bash
git add arcgentica/trail_prompt_v2.py tests/unit/test_trail_experiment_phase11.py
git commit -m "feat(trail): add enhanced prompt v2 with detection hints and smart truncation"
```

---

## Task 3: Add Single-Pass Mode to `trail_agent.py`

The existing `analyze_trace()` supports `agentic_mode="on"` (Agentica agents) and `"off"` (heuristic regex). Add `"single_pass"` mode that sends the full trace through litellm with the v2 prompt.

**Files:**
- Modify: `arcgentica/trail_agent.py`
- Modify: `arcgentica/trail_common.py` (add `prompt_version` to config)
- Test: `tests/unit/test_trail_experiment_phase11.py` (append)

### Step 1: Write the failing tests

Append to `tests/unit/test_trail_experiment_phase11.py`:

```python
from unittest.mock import patch, MagicMock

from arcgentica.trail_agent import analyze_trace


def _make_trace(trace_id: str = "t1", span_id: str = "s1", log_text: str = "ok") -> dict:
    return {
        "trace_id": trace_id,
        "spans": [
            {
                "span_id": span_id,
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": log_text}],
                "child_spans": [],
            }
        ],
    }


def test_single_pass_mode_calls_litellm() -> None:
    """Single-pass mode should call litellm.completion and parse the JSON response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message={"content": json.dumps({
            "errors": [
                {
                    "category": "Formatting Errors",
                    "location": "s1",
                    "evidence": "missing format",
                    "description": "Output format wrong",
                    "impact": "LOW",
                }
            ],
            "scores": [
                {
                    "reliability_score": 4,
                    "reliability_reasoning": "Mostly reliable",
                    "security_score": 5,
                    "security_reasoning": "No issues",
                    "instruction_adherence_score": 3,
                    "instruction_adherence_reasoning": "Missed format",
                    "plan_opt_score": 4,
                    "plan_opt_reasoning": "Decent plan",
                    "overall": 4.0,
                }
            ],
        })})
    ]

    with patch("arcgentica.trail_agent.completion", return_value=mock_response) as mock_comp:
        result = analyze_trace(
            _make_trace(),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    mock_comp.assert_called_once()
    assert result["trace_id"] == "t1"
    assert len(result["errors"]) == 1
    assert result["errors"][0]["category"] == "Formatting Errors"
    assert result["errors"][0]["location"] == "s1"
    assert len(result["scores"]) == 1
    assert result["scores"][0]["reliability_score"] == 4


def test_single_pass_mode_falls_back_on_error() -> None:
    """If litellm call fails, single-pass should fall back to heuristic."""
    with patch("arcgentica.trail_agent.completion", side_effect=Exception("API down")):
        result = analyze_trace(
            _make_trace(log_text="timed out while waiting"),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    assert result["trace_id"] == "t1"
    # Heuristic fallback should still detect timeout
    categories = [e["category"] for e in result["errors"]]
    assert "Timeout Issues" in categories


def test_single_pass_validates_locations() -> None:
    """Single-pass should drop errors with invalid span_ids."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message={"content": json.dumps({
            "errors": [
                {
                    "category": "Formatting Errors",
                    "location": "s1",
                    "evidence": "valid location",
                    "description": "ok",
                    "impact": "LOW",
                },
                {
                    "category": "Goal Deviation",
                    "location": "INVALID_SPAN",
                    "evidence": "bad location",
                    "description": "wrong",
                    "impact": "HIGH",
                },
            ],
            "scores": [{"reliability_score": 3, "reliability_reasoning": "ok",
                        "security_score": 5, "security_reasoning": "ok",
                        "instruction_adherence_score": 3, "instruction_adherence_reasoning": "ok",
                        "plan_opt_score": 3, "plan_opt_reasoning": "ok", "overall": 3.5}],
        })})
    ]

    with patch("arcgentica.trail_agent.completion", return_value=mock_response):
        result = analyze_trace(
            _make_trace(),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    # Invalid location error should be repaired to fallback span or kept with fallback
    locations = [e["location"] for e in result["errors"]]
    assert all(loc == "s1" for loc in locations), f"Expected all locations to be 's1', got {locations}"
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_trail_experiment_phase11.py -k "single_pass" -v`
Expected: FAIL (single_pass mode not implemented)

### Step 3: Implement single-pass mode

**Modify `arcgentica/trail_agent.py`:**

Add imports at the top (after existing imports):

```python
from litellm import completion
```

Add import for v2 prompt (in the try/except import block):

```python
try:
    from .trail_prompt_v2 import build_single_pass_message
except ImportError:
    from trail_prompt_v2 import build_single_pass_message
```

Add new function before `analyze_trace()`:

```python
def _analyze_trace_single_pass(
    trace_payload: dict[str, Any],
    model: str,
    max_span_text_chars: int,
) -> dict[str, Any]:
    """Single-pass analysis: send full trace through litellm with enhanced v2 prompt."""
    trace_id = str(trace_payload.get("trace_id", ""))
    message = build_single_pass_message(trace_payload)

    is_reasoning = any(tag in model for tag in ("o1", "o3", "o4", "anthropic", "gemini-2.5"))
    params: dict[str, Any] = {
        "messages": [{"role": "user", "content": message}],
        "model": model,
        "max_completion_tokens": 8000,
        "drop_params": True,
    }
    if is_reasoning:
        params["reasoning_effort"] = "high"
    else:
        params["temperature"] = 0.0
        params["top_p"] = 1

    response = completion(**params)
    raw_text = response.choices[0].message["content"]
    parsed = _json_object_from_text(raw_text)

    # Build valid span_id set for location validation
    spans = trace_payload.get("spans", [])
    flat_spans = _walk_spans(spans if isinstance(spans, list) else [])
    valid_span_ids = set()
    fallback_span_id = ""
    for span in flat_spans:
        sid = span.get("span_id")
        if isinstance(sid, str) and sid:
            valid_span_ids.add(sid)
            if not fallback_span_id:
                fallback_span_id = sid

    # Validate and clean errors
    raw_errors = parsed.get("errors", [])
    if not isinstance(raw_errors, list):
        raw_errors = []

    cleaned_errors: list[dict[str, str]] = []
    for item in raw_errors:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        if not isinstance(category, str) or category.strip() not in TRAIL_LEAF_CATEGORIES:
            continue

        location = item.get("location", "")
        if not isinstance(location, str) or location not in valid_span_ids:
            location = fallback_span_id
        if not location:
            continue

        evidence = item.get("evidence", "")
        if isinstance(evidence, str):
            evidence = " ".join(evidence.split())[:300]
        else:
            evidence = ""

        description = item.get("description", "")
        if not isinstance(description, str) or not description.strip():
            description = f"Detected {category.strip().lower()} from trace analysis."

        cleaned_errors.append({
            "category": category.strip(),
            "location": location,
            "evidence": evidence,
            "description": description,
            "impact": _normalize_impact(item.get("impact")),
        })

    # Deduplicate by (location, category)
    cleaned_errors = _merge_errors(cleaned_errors)

    # Extract scores (use LLM-generated, not formula)
    raw_scores = parsed.get("scores", [])
    if isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], dict):
        scores = [raw_scores[0]]
    else:
        scores = [_score_block(cleaned_errors)]

    return {
        "trace_id": trace_id,
        "errors": cleaned_errors,
        "scores": scores,
    }
```

**Modify `analyze_trace()` function** to add the new branch:

Change the `if agentic_mode != "on":` block to a three-way dispatch:

```python
def analyze_trace(
    trace_payload: dict[str, Any],
    model: str,
    *,
    agentic_mode: str = "off",
    max_num_agents: int = 6,
    max_chunks: int = 6,
    max_spans_per_chunk: int = 12,
    max_span_text_chars: int = 1200,
) -> dict[str, Any]:
    if agentic_mode == "single_pass":
        try:
            return _analyze_trace_single_pass(
                trace_payload=trace_payload,
                model=model,
                max_span_text_chars=max_span_text_chars,
            )
        except Exception:
            findings = _heuristic_findings(trace_payload)
            return {
                "trace_id": str(trace_payload.get("trace_id", "")),
                "errors": findings,
                "scores": [_score_block(findings)],
            }

    if agentic_mode != "on":
        findings = _heuristic_findings(trace_payload)
        return {
            "trace_id": str(trace_payload.get("trace_id", "")),
            "errors": findings,
            "scores": [_score_block(findings)],
        }

    try:
        return asyncio.run(
            _analyze_trace_agentic(
                trace_payload=trace_payload,
                model=model,
                max_num_agents=max_num_agents,
                max_chunks=max_chunks,
                max_spans_per_chunk=max_spans_per_chunk,
                max_span_text_chars=max_span_text_chars,
            )
        )
    except Exception:
        findings = _heuristic_findings(trace_payload)
        return {
            "trace_id": str(trace_payload.get("trace_id", "")),
            "errors": findings,
            "scores": [_score_block(findings)],
        }
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_trail_experiment_phase11.py -k "single_pass" -v`
Expected: PASS (all 3 tests)

### Step 5: Run existing tests to verify no regression

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_arcgentica_trail_runner_phase11.py -v`
Expected: PASS (all existing tests still pass)

### Step 6: Commit

```bash
git add arcgentica/trail_agent.py tests/unit/test_trail_experiment_phase11.py
git commit -m "feat(trail): add single_pass mode with litellm and v2 prompt to analyze_trace"
```

---

## Task 4: Experiment Runner (`trail_experiment.py`)

The experiment runner is the CLI that ties everything together: loads traces (subset or full), runs analysis, scores results, logs metrics.

**Files:**
- Create: `arcgentica/trail_experiment.py`
- Test: `tests/unit/test_trail_experiment_phase11.py` (append)

### Step 1: Write the failing tests

Append to `tests/unit/test_trail_experiment_phase11.py`:

```python
from arcgentica.trail_experiment import (
    load_subset_trace_ids,
    ExperimentConfig,
    run_experiment,
)


def test_load_subset_trace_ids_dev18() -> None:
    ids = load_subset_trace_ids("dev18")
    assert isinstance(ids, list)
    assert len(ids) == 18
    assert all(isinstance(tid, str) for tid in ids)


def test_load_subset_trace_ids_full_returns_none() -> None:
    ids = load_subset_trace_ids("full")
    assert ids is None


def test_experiment_config_defaults() -> None:
    cfg = ExperimentConfig(
        experiment_id="test_001",
        trail_data_dir=Path("/tmp/data"),
        gold_dir=Path("/tmp/gold"),
        output_dir=Path("/tmp/out"),
    )
    assert cfg.model == "openai/gpt-5-mini"
    assert cfg.approach == "single_pass"
    assert cfg.subset == "dev18"
    assert cfg.prompt_version == "v2"
    assert cfg.split == "GAIA"
    assert cfg.max_workers == 5


def test_run_experiment_writes_outputs_and_metrics(tmp_path: Path) -> None:
    """Run experiment with mocked LLM on 2 synthetic traces."""
    # Setup data dir
    data_dir = tmp_path / "data" / "GAIA"
    data_dir.mkdir(parents=True)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)

    trace1 = {
        "trace_id": "aaaa1111bbbb2222cccc3333dddd4444",
        "spans": [{"span_id": "sp1", "span_name": "main", "status_code": "Unset",
                    "status_message": "", "span_attributes": {}, "logs": [],
                    "child_spans": []}],
    }
    trace2 = {
        "trace_id": "eeee5555ffff6666aaaa7777bbbb8888",
        "spans": [{"span_id": "sp2", "span_name": "main", "status_code": "Error",
                    "status_message": "timed out", "span_attributes": {}, "logs": [],
                    "child_spans": []}],
    }
    (data_dir / "aaaa1111bbbb2222cccc3333dddd4444.json").write_text(json.dumps(trace1))
    (data_dir / "eeee5555ffff6666aaaa7777bbbb8888.json").write_text(json.dumps(trace2))

    # Gold annotations
    gold1 = {"errors": [], "scores": [{"reliability_score": 5, "reliability_reasoning": "ok",
             "security_score": 5, "security_reasoning": "ok",
             "instruction_adherence_score": 5, "instruction_adherence_reasoning": "ok",
             "plan_opt_score": 5, "plan_opt_reasoning": "ok", "overall": 5.0}]}
    gold2 = {"errors": [{"category": "Timeout Issues", "location": "sp2",
             "evidence": "timed out", "description": "timeout", "impact": "HIGH"}],
             "scores": [{"reliability_score": 2, "reliability_reasoning": "bad",
             "security_score": 5, "security_reasoning": "ok",
             "instruction_adherence_score": 3, "instruction_adherence_reasoning": "ok",
             "plan_opt_score": 2, "plan_opt_reasoning": "bad", "overall": 3.0}]}
    (gold_dir / "aaaa1111bbbb2222cccc3333dddd4444.json").write_text(json.dumps(gold1))
    (gold_dir / "eeee5555ffff6666aaaa7777bbbb8888.json").write_text(json.dumps(gold2))

    # Mock LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message={"content": json.dumps({
        "errors": [], "scores": [{"reliability_score": 5, "reliability_reasoning": "ok",
        "security_score": 5, "security_reasoning": "ok",
        "instruction_adherence_score": 5, "instruction_adherence_reasoning": "ok",
        "plan_opt_score": 5, "plan_opt_reasoning": "ok", "overall": 5.0}]
    })})]

    cfg = ExperimentConfig(
        experiment_id="test_run",
        trail_data_dir=tmp_path / "data",
        gold_dir=gold_dir,
        output_dir=tmp_path / "out",
        model="openai/gpt-5-mini",
        subset="full",
        split="GAIA",
    )

    with patch("arcgentica.trail_agent.completion", return_value=mock_response):
        result = run_experiment(cfg)

    assert "experiment_id" in result
    assert result["experiment_id"] == "test_run"
    assert "metrics" in result
    assert "weighted_f1" in result["metrics"]
    assert "traces_processed" in result
    assert result["traces_processed"] == 2

    # Check output files were written
    output_dir = tmp_path / "out" / "test_run"
    assert output_dir.exists()
    output_files = list(output_dir.glob("*.json"))
    assert len(output_files) >= 2  # trace outputs

    # Check metrics file
    metrics_file = output_dir / "metrics.json"
    assert metrics_file.exists()
    metrics = json.loads(metrics_file.read_text())
    assert "weighted_f1" in metrics
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_trail_experiment_phase11.py -k "experiment" -v`
Expected: FAIL with import error

### Step 3: Write `trail_experiment.py`

Create `arcgentica/trail_experiment.py`:

```python
# ABOUTME: CLI entrypoint for running TRAIL benchmark experiments with configurable prompts and subsets.
# ABOUTME: Auto-scores outputs against gold annotations and logs metrics for experiment tracking.

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

try:
    from .trail_agent import analyze_trace
    from .trail_common import TRAIL_LEAF_CATEGORIES
except ImportError:
    from trail_agent import analyze_trace
    from trail_common import TRAIL_LEAF_CATEGORIES


@dataclass
class ExperimentConfig:
    experiment_id: str
    trail_data_dir: Path
    gold_dir: Path
    output_dir: Path
    model: str = "openai/gpt-5-mini"
    approach: str = "single_pass"
    subset: str = "dev18"
    prompt_version: str = "v2"
    split: str = "GAIA"
    max_workers: int = 5
    semantic_checks: str = "strict"
    notes: str = ""


def load_subset_trace_ids(subset: str) -> list[str] | None:
    """Load trace IDs for a named subset. Returns None for 'full'."""
    if subset == "full":
        return None
    manifest_path = Path(__file__).resolve().parent / "dev_subset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("subset_id") == subset:
        return manifest["trace_ids"]
    raise ValueError(f"Unknown subset: {subset}. Available: dev18, full")


def _normalize_category(category: str) -> str:
    """Normalize category name to match official scorer."""
    if not category:
        return ""
    cat_lower = category.lower().strip()
    cat_nospace = cat_lower.replace(" ", "")
    for std in TRAIL_LEAF_CATEGORIES:
        if cat_lower == std.lower() or cat_nospace == std.lower().replace(" ", ""):
            return std
    for std in TRAIL_LEAF_CATEGORIES:
        if cat_nospace in std.lower().replace(" ", ""):
            return std
    return category


def _score_outputs(
    gold_dir: Path,
    generated_dir: Path,
    trace_files: list[str],
) -> dict[str, Any]:
    """Score generated outputs against gold using the same logic as calculate_scores.py."""
    all_y_true = []
    all_y_pred = []
    location_acc_sum = 0.0
    joint_acc_sum = 0.0
    files_scored = 0

    gt_scores_map: dict[str, list[float]] = {
        "reliability": [], "security": [], "instruction_adherence": [],
        "plan_opt": [], "overall": [],
    }
    gen_scores_map: dict[str, list[float]] = {
        "reliability": [], "security": [], "instruction_adherence": [],
        "plan_opt": [], "overall": [],
    }

    per_trace_details: list[dict[str, Any]] = []

    for fname in trace_files:
        gold_path = gold_dir / fname
        gen_path = generated_dir / fname
        if not gold_path.exists() or not gen_path.exists():
            continue

        try:
            gold = json.loads(gold_path.read_text(encoding="utf-8"))
            gen_text = gen_path.read_text(encoding="utf-8")
            # Handle both raw JSON and text-wrapped JSON
            try:
                gen = json.loads(gen_text)
            except json.JSONDecodeError:
                import re
                match = re.search(r"\{.*\}", gen_text, re.DOTALL)
                if match:
                    gen = json.loads(match.group(0))
                else:
                    continue
        except Exception:
            continue

        gt_errors = gold.get("errors", [])
        gen_errors = gen.get("errors", [])

        gt_cats = [_normalize_category(e.get("category", "")) for e in gt_errors if e.get("category")]
        gen_cats = [_normalize_category(e.get("category", "")) for e in gen_errors if e.get("category")]
        gt_locs = [e.get("location", "") for e in gt_errors]
        gen_locs = [e.get("location", "") for e in gen_errors]

        # Binary category vectors
        y_true = np.zeros(len(TRAIL_LEAF_CATEGORIES))
        y_pred = np.zeros(len(TRAIL_LEAF_CATEGORIES))
        for cat in gt_cats:
            if cat in TRAIL_LEAF_CATEGORIES:
                y_true[TRAIL_LEAF_CATEGORIES.index(cat)] = 1
        for cat in gen_cats:
            if cat in TRAIL_LEAF_CATEGORIES:
                y_pred[TRAIL_LEAF_CATEGORIES.index(cat)] = 1
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        # Location accuracy
        common_locs = set(gt_locs).intersection(set(gen_locs))
        loc_acc = len(common_locs) / len(set(gt_locs)) if gt_locs else 0
        location_acc_sum += loc_acc

        # Joint accuracy
        gt_pairs = set(zip(gt_locs, gt_cats))
        gen_pairs = set(zip(gen_locs, gen_cats))
        common_pairs = gt_pairs.intersection(gen_pairs)
        joint_acc = len(common_pairs) / len(gt_pairs) if gt_pairs else 0
        joint_acc_sum += joint_acc

        per_trace_details.append({
            "trace_id": fname.replace(".json", ""),
            "location_accuracy": loc_acc,
            "joint_accuracy": joint_acc,
            "gt_categories": gt_cats,
            "gen_categories": gen_cats,
        })

        # Score correlations
        gt_sc = gold.get("scores", [{}])[0] if gold.get("scores") else {}
        gen_sc = gen.get("scores", [{}])[0] if gen.get("scores") else {}
        score_keys = [
            ("reliability", "reliability_score"),
            ("security", "security_score"),
            ("instruction_adherence", "instruction_adherence_score"),
            ("plan_opt", "plan_opt_score"),
            ("overall", "overall"),
        ]
        for short_key, json_key in score_keys:
            gt_v = gt_sc.get(json_key)
            gen_v = gen_sc.get(json_key)
            if gt_v is not None and gen_v is not None:
                try:
                    gt_scores_map[short_key].append(float(gt_v))
                    gen_scores_map[short_key].append(float(gen_v))
                except (ValueError, TypeError):
                    pass

        files_scored += 1

    # Aggregate metrics
    if files_scored == 0:
        return {"weighted_f1": 0, "location_accuracy": 0, "joint_accuracy": 0,
                "per_category_f1": {}, "score_correlations": {}, "traces_scored": 0}

    y_true_arr = np.vstack(all_y_true)
    y_pred_arr = np.vstack(all_y_pred)
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    per_cat_f1: dict[str, dict[str, Any]] = {}
    for i, cat in enumerate(TRAIL_LEAF_CATEGORIES):
        tp = int(np.sum((y_true_arr[:, i] == 1) & (y_pred_arr[:, i] == 1)))
        fp = int(np.sum((y_true_arr[:, i] == 0) & (y_pred_arr[:, i] == 1)))
        fn = int(np.sum((y_true_arr[:, i] == 1) & (y_pred_arr[:, i] == 0)))
        support = int(np.sum(y_true_arr[:, i]))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if support > 0:
            per_cat_f1[cat] = {"f1": round(f1_val, 4), "precision": round(precision, 4),
                               "recall": round(recall, 4), "support": support}

    score_corrs: dict[str, Any] = {}
    for key in gt_scores_map:
        if len(gt_scores_map[key]) >= 3:
            try:
                corr, pval = pearsonr(gt_scores_map[key], gen_scores_map[key])
                score_corrs[key] = {"correlation": round(float(corr), 4), "p_value": round(float(pval), 4),
                                    "n": len(gt_scores_map[key])}
            except Exception:
                pass

    return {
        "weighted_f1": round(weighted_f1, 4),
        "location_accuracy": round(location_acc_sum / files_scored, 4),
        "joint_accuracy": round(joint_acc_sum / files_scored, 4),
        "per_category_f1": per_cat_f1,
        "score_correlations": score_corrs,
        "traces_scored": files_scored,
        "per_trace": per_trace_details,
    }


def _process_one_trace(
    trace_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Process a single trace file and write output."""
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    start = time.time()

    prediction = analyze_trace(
        trace_payload,
        model=config.model,
        agentic_mode=config.approach,
    )

    elapsed = time.time() - start
    out_path = output_dir / trace_path.name
    out_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    return {"trace_file": trace_path.name, "elapsed_sec": round(elapsed, 2), "num_errors": len(prediction.get("errors", []))}


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run a complete experiment: generate outputs, score, and return metrics."""
    exp_output_dir = config.output_dir / config.experiment_id
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve trace files
    split_dir = config.trail_data_dir / config.split
    if not split_dir.exists():
        raise FileNotFoundError(f"TRAIL split directory not found: {split_dir}")

    subset_ids = load_subset_trace_ids(config.subset)
    all_trace_files = sorted(split_dir.glob("*.json"))

    if subset_ids is not None:
        id_set = set(subset_ids)
        trace_files = [f for f in all_trace_files if f.stem in id_set]
    else:
        trace_files = all_trace_files

    # Save config
    config_record = {
        "experiment_id": config.experiment_id,
        "model": config.model,
        "approach": config.approach,
        "subset": config.subset,
        "prompt_version": config.prompt_version,
        "split": config.split,
        "semantic_checks": config.semantic_checks,
        "max_workers": config.max_workers,
        "notes": config.notes,
        "num_traces": len(trace_files),
    }
    (exp_output_dir / "config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")

    # Process traces
    start_time = time.time()
    trace_results: list[dict[str, Any]] = []
    failures = 0

    if config.max_workers <= 1:
        for tf in trace_files:
            try:
                tr = _process_one_trace(tf, exp_output_dir, config)
                trace_results.append(tr)
            except Exception as e:
                failures += 1
                trace_results.append({"trace_file": tf.name, "error": str(e)})
    else:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {executor.submit(_process_one_trace, tf, exp_output_dir, config): tf for tf in trace_files}
            for future in as_completed(futures):
                tf = futures[future]
                try:
                    tr = future.result()
                    trace_results.append(tr)
                except Exception as e:
                    failures += 1
                    trace_results.append({"trace_file": tf.name, "error": str(e)})

    wall_time = time.time() - start_time

    # Score outputs
    trace_filenames = [tf.name for tf in trace_files]
    metrics = _score_outputs(config.gold_dir, exp_output_dir, trace_filenames)

    # Write metrics
    (exp_output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Build experiment record
    record = {
        "experiment_id": config.experiment_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": config.model,
        "approach": config.approach,
        "subset": config.subset,
        "prompt_version": config.prompt_version,
        "split": config.split,
        "metrics": {
            "weighted_f1": metrics["weighted_f1"],
            "location_accuracy": metrics["location_accuracy"],
            "joint_accuracy": metrics["joint_accuracy"],
            "score_correlations": metrics.get("score_correlations", {}),
        },
        "per_category_f1": metrics.get("per_category_f1", {}),
        "traces_processed": len(trace_files),
        "traces_failed": failures,
        "wall_time_sec": round(wall_time, 1),
        "notes": config.notes,
    }

    # Append to experiment log
    log_path = config.output_dir / "experiment_log.json"
    if log_path.exists():
        log_data = json.loads(log_path.read_text(encoding="utf-8"))
    else:
        log_data = {"experiments": []}
    log_data["experiments"].append(record)
    log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment: {config.experiment_id}")
    print(f"Model: {config.model} | Approach: {config.approach} | Subset: {config.subset}")
    print(f"Traces: {len(trace_files)} processed, {failures} failed")
    print(f"Wall time: {wall_time:.1f}s")
    print(f"{'='*60}")
    print(f"Weighted F1:        {metrics['weighted_f1']:.4f}")
    print(f"Location Accuracy:  {metrics['location_accuracy']:.4f}")
    print(f"Joint Accuracy:     {metrics['joint_accuracy']:.4f}")
    if metrics.get("score_correlations"):
        print(f"Score Correlations:")
        for k, v in metrics["score_correlations"].items():
            print(f"  {k}: r={v['correlation']:.4f}")
    print(f"{'='*60}")

    # Print per-category F1 (non-zero support)
    cats = metrics.get("per_category_f1", {})
    if cats:
        zero_f1 = [c for c, m in cats.items() if m["f1"] == 0]
        if zero_f1:
            print(f"0% F1 categories ({len(zero_f1)}): {', '.join(zero_f1)}")
    print(f"Results: {exp_output_dir}")
    print(f"Metrics: {exp_output_dir / 'metrics.json'}")
    print()

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TRAIL benchmark experiment.")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--trail-data-dir", type=Path, required=True,
                        help="Path to TRAIL benchmark data dir (containing GAIA/ folder)")
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="Path to gold annotations dir (e.g., processed_annotations_gaia/)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--approach", type=str, default="single_pass",
                        choices=["single_pass", "on", "off"])
    parser.add_argument("--subset", type=str, default="dev18",
                        help="Subset to run: dev18 or full")
    parser.add_argument("--prompt-version", type=str, default="v2")
    parser.add_argument("--split", type=str, default="GAIA", choices=["GAIA", "SWE Bench"])
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--semantic-checks", type=str, default="strict", choices=["off", "strict"])
    parser.add_argument("--notes", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        experiment_id=args.experiment_id,
        trail_data_dir=args.trail_data_dir,
        gold_dir=args.gold_dir,
        output_dir=args.output_dir,
        model=args.model,
        approach=args.approach,
        subset=args.subset,
        prompt_version=args.prompt_version,
        split=args.split,
        max_workers=args.max_workers,
        semantic_checks=args.semantic_checks,
        notes=args.notes,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_trail_experiment_phase11.py -k "experiment" -v`
Expected: PASS (all 4 tests)

### Step 5: Commit

```bash
git add arcgentica/trail_experiment.py tests/unit/test_trail_experiment_phase11.py
git commit -m "feat(trail): add experiment runner with auto-scoring and experiment log"
```

---

## Task 5: Run All Tests and Verify

**Files:**
- All test files from Tasks 1-4

### Step 1: Run the full test suite for the new file

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_trail_experiment_phase11.py -v`
Expected: PASS (all ~14 tests)

### Step 2: Run existing arcgentica tests to verify no regression

Run: `cd /Users/nainy/Documents/Personal/rlm_observability && uv run pytest tests/unit/test_arcgentica_trail_runner_phase11.py tests/unit/test_arcgentica_trail_semantic_checks_phase11.py -v`
Expected: PASS

### Step 3: Commit (if any fixes needed)

```bash
git add -u
git commit -m "fix(trail): resolve test regressions from experiment infrastructure"
```

---

## Task 6: Smoke Test with Real Data (Manual Verification)

Run a quick experiment on 1-2 traces to verify end-to-end works before doing a real dev18 run.

### Step 1: Run single trace smoke test (heuristic, no API call)

```bash
cd /Users/nainy/Documents/Personal/rlm_observability
uv run python -m arcgentica.trail_experiment \
  --experiment-id smoke_heuristic \
  --trail-data-dir trail-benchmark/benchmarking/data \
  --gold-dir trail-benchmark/benchmarking/processed_annotations_gaia \
  --output-dir arcgentica/output/experiments \
  --model openai/gpt-5-mini \
  --approach off \
  --subset dev18 \
  --max-workers 1
```

Expected: Completes without errors. Prints metrics summary. Creates output files in `arcgentica/output/experiments/smoke_heuristic/`.

### Step 2: Verify output structure

```bash
ls arcgentica/output/experiments/smoke_heuristic/
cat arcgentica/output/experiments/smoke_heuristic/metrics.json | python3 -m json.tool | head -20
cat arcgentica/output/experiments/experiment_log.json | python3 -m json.tool | head -30
```

Expected: `config.json`, `metrics.json`, and 18 trace output `.json` files.

### Step 3: (Optional) Run single-pass with real API

Only do this if the heuristic smoke test passes and you want to verify the LLM path:

```bash
cd /Users/nainy/Documents/Personal/rlm_observability
uv run python -m arcgentica.trail_experiment \
  --experiment-id smoke_single_pass \
  --trail-data-dir trail-benchmark/benchmarking/data \
  --gold-dir trail-benchmark/benchmarking/processed_annotations_gaia \
  --output-dir arcgentica/output/experiments \
  --model openai/gpt-5-mini \
  --approach single_pass \
  --subset dev18 \
  --max-workers 3 \
  --notes "First single-pass experiment with v2 prompt on dev18"
```

Expected: ~$0.50-1.00 cost. Prints F1, location accuracy, joint accuracy. These are our first experiment numbers to iterate on.

### Step 4: Commit experiment artifacts (gitignore output)

```bash
echo "arcgentica/output/experiments/" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore experiment output artifacts"
```

---

## Task Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | Dev subset manifest | `arcgentica/dev_subset_manifest.json` | 1 test |
| 2 | Enhanced prompt v2 | `arcgentica/trail_prompt_v2.py` | 7 tests |
| 3 | Single-pass mode | `arcgentica/trail_agent.py` (modify) | 3 tests |
| 4 | Experiment runner | `arcgentica/trail_experiment.py` | 4 tests |
| 5 | Full test verification | (all above) | ~14 tests |
| 6 | Smoke test with real data | Manual verification | - |

## What Comes Next (not in this plan)

After this infrastructure is in place, the experiment sequence from the design doc begins:
- **exp_001**: Reproduce baseline prompt with gpt-5-mini on dev18
- **exp_002**: Run with prompt v2 (this plan's prompt)
- **exp_003-007**: Iterate on prompt improvements
- **exp_010+**: Graduate to Approach B (two-pass agentic) if needed

Each experiment is a single CLI command with automatic scoring and metrics logging.
