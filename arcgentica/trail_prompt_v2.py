# ABOUTME: Enhanced TRAIL prompts (v2) with category detection hints, location rules, and LLM-generated scores.
# ABOUTME: Includes smart trace truncation for context window overflow and single-pass message builder.

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

try:
    from .trail_common import TRAIL_LEAF_CATEGORIES
except ImportError:
    from trail_common import TRAIL_LEAF_CATEGORIES


TRAIL_SINGLE_PASS_PROMPT_V2 = """\
You are an expert agent-trace evaluator for the TRAIL benchmark.
You will receive a single agent execution trace (JSON) and must produce a structured JSON verdict.

## TRAIL Error Taxonomy

The TRAIL taxonomy is a hierarchical tree of error categories.
You MUST only use leaf categories from this tree.

```
Agent Errors
├── Reasoning Errors
│   ├── Language-only
│   └── Tool-related
│       ├── Poor Information Retrieval
│       ├── Incorrect Memory Usage
│       └── Tool Output Misinterpretation
├── Action Errors
│   ├── Incorrect Problem Identification
│   ├── Tool Selection Errors
│   ├── Formatting Errors
│   └── Instruction Non-compliance
├── System Errors
│   ├── Tool Definition Issues
│   ├── Environment Setup Errors
│   ├── External Service Failures
│   │   ├── Rate Limiting
│   │   ├── Authentication Errors
│   │   └── Service Errors
│   └── Resource Issues
│       ├── Resource Not Found
│       ├── Resource Exhaustion
│       └── Timeout Issues
└── Planning Errors
    ├── Context Handling Failures
    ├── Resource Abuse
    ├── Goal Deviation
    └── Task Orchestration
```

### Leaf Categories with Detection Hints

1. **Language-only**
   DETECT: The agent makes a reasoning mistake that does not involve any tool call.
   Look for logical contradictions, arithmetic errors, hallucinated facts, or incorrect
   deductions in the agent's free-text reasoning steps (not in tool invocations).

2. **Tool-related**
   DETECT: The agent makes a reasoning mistake that involves a tool call but does not
   fit a more specific sub-category below. Look for the agent misunderstanding what a
   tool does or drawing wrong conclusions from correct tool output.

3. **Poor Information Retrieval**
   DETECT: The agent fails to retrieve relevant information or retrieves irrelevant
   information. Look for search queries that miss key terms, ignored search results,
   or retrieval calls that return empty/irrelevant data and the agent does not retry.

4. **Incorrect Memory Usage**
   DETECT: The agent forgets, overwrites, or misuses previously retrieved information.
   Look for the agent re-asking questions it already answered, contradicting earlier
   findings, or failing to use context that was available in prior spans.

5. **Tool Output Misinterpretation**
   DETECT: The agent receives correct tool output but interprets it incorrectly.
   Look for the agent extracting wrong fields, misreading numeric values, or
   drawing conclusions that contradict the tool's actual response payload.

6. **Incorrect Problem Identification**
   DETECT: The agent misidentifies what the task is asking. Look for the agent
   solving a different problem than requested, misunderstanding the goal, or
   answering a question that was not asked.

7. **Tool Selection Errors**
   DETECT: The agent picks the wrong tool for the job. Look for cases where a
   more appropriate tool was available but not used, or the agent uses a tool
   that cannot accomplish the intended sub-task.

8. **Formatting Errors**
   DETECT: The agent produces output in the wrong format. Look for malformed JSON,
   missing required fields in tool call arguments, wrong argument types, or
   final answers that do not match the expected output schema.

9. **Instruction Non-compliance**
   DETECT: The agent violates explicit instructions from the system prompt or user.
   Look for the agent ignoring stated constraints, using disallowed tools, or
   producing output that contradicts explicit user directives.

10. **Tool Definition Issues**
    DETECT: The tool definition (schema, description) is wrong or misleading,
    causing the agent to misuse it. Look for tool schemas with incorrect parameter
    types, missing descriptions, or ambiguous semantics that lead to agent confusion.

11. **Environment Setup Errors**
    DETECT: The execution environment is misconfigured. Look for missing environment
    variables, wrong working directory, unavailable dependencies, or file permission
    errors that prevent tool execution.

12. **Rate Limiting**
    DETECT: An external service rejects requests due to rate limits. Look for HTTP 429
    responses, "rate limit exceeded" messages, or retry-after headers in tool outputs.

13. **Authentication Errors**
    DETECT: An external service rejects requests due to authentication failures.
    Look for HTTP 401/403 responses, "unauthorized" or "forbidden" messages,
    expired tokens, or missing API keys in tool call outputs.

14. **Service Errors**
    DETECT: An external service returns a server-side error. Look for HTTP 500/502/503
    responses, connection refused errors, DNS failures, or generic "service unavailable"
    messages in tool outputs.

15. **Resource Not Found**
    DETECT: A requested resource does not exist. Look for HTTP 404 responses,
    "file not found", "no such table", or "entity does not exist" messages in tool outputs.

16. **Resource Exhaustion**
    DETECT: System resources are exhausted. Look for out-of-memory errors, disk full
    errors, maximum connection pool reached, or context window overflow messages.

17. **Timeout Issues**
    DETECT: An operation exceeds its time budget. Look for "timed out", "deadline exceeded",
    socket timeout errors, or operations that hang without completing in tool outputs.

18. **Context Handling Failures**
    DETECT: The agent fails to manage its context window properly. Look for the agent
    losing track of conversation history, exceeding token limits, or failing to
    summarize/compress context when needed.

19. **Resource Abuse**
    DETECT: The agent makes excessive, redundant, or wasteful use of tools or resources.
    Look for repeated identical tool calls, unnecessary API calls, calling the same
    endpoint in a tight loop, or generating excessive intermediate output.

20. **Goal Deviation**
    DETECT: The agent drifts away from the original goal. Look for the agent pursuing
    tangential sub-tasks, losing focus on the main objective, or producing a final
    answer that does not address the original question.

21. **Task Orchestration**
    DETECT: The agent fails to coordinate multi-step plans properly. Look for incorrect
    ordering of dependent steps, missing prerequisites, parallel execution of steps
    that should be sequential, or failure to aggregate sub-task results.


## Location Rules

For each detected error, you must provide a `location` field with the `span_id` of
the span where the error is most evident.

- For most categories, use the **first** span where the error manifests.
- **Exception — Resource Abuse**: use the **last** span in the abusive sequence,
  since the abuse is only evident after accumulation.
- Always use a real span_id from the trace. Never fabricate span IDs.


## Score Guidance

In addition to error detection, you must produce four numeric scores (integers 0–5) with reasoning:

- **reliability_score** (0-5): How reliably does the agent produce correct results?
  5 = no errors, always correct. 0 = catastrophic failures throughout.
  Penalize heavily for hallucinations and incorrect information, less for formatting issues.

- **security_score** (0-5): How well does the agent respect security boundaries?
  5 = no security issues. Only reduce from 5 for Authentication Errors, data exposure,
  or unsafe API calls. Most traces should score 5.

- **instruction_adherence_score** (0-5): How well does the agent follow its instructions?
  5 = perfect adherence. Missing end_plan tags are LOW impact; ignoring the core task
  requirement is HIGH impact. Consider whether the system followed the task instructions.

- **plan_opt_score** (0-5): How optimal is the agent's planning and execution?
  5 = perfectly efficient plan. Redundant tool calls, failed retries without adaptation,
  and excessive resource usage reduce this score.

- **overall**: Average of the four scores above (can be float, e.g. 3.75).

Scores should reflect the severity and frequency of detected errors.
A trace with no errors should score 5 on all dimensions.


## Output Schema

Return strict JSON (no markdown fences, no extra text):

```
{
  "errors": [
    {
      "category": "<leaf category name>",
      "location": "<span_id>",
      "evidence": "<direct quote or description from the trace>",
      "description": "<what went wrong and why>",
      "impact": "LOW | MEDIUM | HIGH"
    }
  ],
  "scores": [
    {
      "reliability_score": 3,
      "reliability_reasoning": "<why this score>",
      "security_score": 5,
      "security_reasoning": "<why this score>",
      "instruction_adherence_score": 4,
      "instruction_adherence_reasoning": "<why this score>",
      "plan_opt_score": 3,
      "plan_opt_reasoning": "<why this score>",
      "overall": 3.75
    }
  ]
}
```

If no errors are found, return `{"errors": [], "scores": [{"reliability_score": 5, "reliability_reasoning": "No errors detected", "security_score": 5, "security_reasoning": "No security issues", "instruction_adherence_score": 5, "instruction_adherence_reasoning": "Instructions followed", "plan_opt_score": 5, "plan_opt_reasoning": "Efficient execution", "overall": 5.0}]}`.


## Examples

### Example 1: Reasoning + Action Errors

Trace snippet (simplified):
- Span s001: User asks "What is the population of France?"
- Span s002: Agent calls search_web("population of Germany")
- Span s003: Agent reads result: "Germany population: 84 million"
- Span s004: Agent answers "The population of France is 84 million"

Expected output:
```json
{
  "errors": [
    {
      "category": "Poor Information Retrieval",
      "location": "s002",
      "evidence": "Agent searched for 'population of Germany' instead of 'population of France'",
      "description": "The search query targeted the wrong country, retrieving irrelevant data.",
      "impact": "HIGH"
    },
    {
      "category": "Tool Output Misinterpretation",
      "location": "s004",
      "evidence": "Agent reported Germany's population as France's population",
      "description": "The agent applied Germany's population figure to France without noticing the mismatch.",
      "impact": "HIGH"
    }
  ],
  "scores": [
    {
      "reliability_score": 1,
      "reliability_reasoning": "Critical errors: wrong search query and misattributed data led to completely wrong answer.",
      "security_score": 5,
      "security_reasoning": "No security issues detected.",
      "instruction_adherence_score": 2,
      "instruction_adherence_reasoning": "Agent attempted the task but used wrong search terms and did not verify results.",
      "plan_opt_score": 2,
      "plan_opt_reasoning": "Searched for wrong country and did not cross-check the retrieved data.",
      "overall": 2.5
    }
  ]
}
```

### Example 2: System + Planning Errors

Trace snippet (simplified):
- Span s010: Agent calls api_lookup(key="expired_token_abc")
- Span s011: Tool returns HTTP 401 Unauthorized
- Span s012: Agent retries api_lookup(key="expired_token_abc") — same params
- Span s013: Tool returns HTTP 401 Unauthorized
- Span s014: Agent retries api_lookup(key="expired_token_abc") — third time
- Span s015: Tool returns HTTP 401 Unauthorized
- Span s016: Agent gives up and returns "Unable to find information"

Expected output:
```json
{
  "errors": [
    {
      "category": "Authentication Errors",
      "location": "s011",
      "evidence": "HTTP 401 Unauthorized response from api_lookup",
      "description": "The API rejected the request due to an expired authentication token.",
      "impact": "HIGH"
    },
    {
      "category": "Resource Abuse",
      "location": "s015",
      "evidence": "Agent retried the same failing call 3 times with identical parameters",
      "description": "The agent made redundant retries without changing the expired token, wasting resources.",
      "impact": "MEDIUM"
    },
    {
      "category": "Goal Deviation",
      "location": "s016",
      "evidence": "Agent gave up instead of attempting alternative approaches",
      "description": "The agent abandoned the task without trying a different authentication method or fallback strategy.",
      "impact": "MEDIUM"
    }
  ],
  "scores": [
    {
      "reliability_score": 1,
      "reliability_reasoning": "Complete failure to authenticate and no fallback strategy.",
      "security_score": 3,
      "security_reasoning": "Authentication failure and retrying with expired credentials is a security concern.",
      "instruction_adherence_score": 2,
      "instruction_adherence_reasoning": "Agent gave up without completing the task.",
      "plan_opt_score": 1,
      "plan_opt_reasoning": "Wasted resources on identical retries with no adaptation.",
      "overall": 1.75
    }
  ]
}
```


## Trace Data

Analyze the following trace and produce your JSON verdict:

{trace}
"""


# ---------------------------------------------------------------------------
# Smart trace truncation
# ---------------------------------------------------------------------------

_ERROR_LOG_CHAR_LIMIT = 2000
_NORMAL_LOG_CHAR_LIMIT = 400


def _is_error_span(span: dict[str, Any]) -> bool:
    """Return True if span has error status or non-empty status_message."""
    sc = str(span.get("status_code", "")).lower()
    sm = str(span.get("status_message", "")).strip()
    return sc == "error" or (sm != "" and sc != "ok")


def _truncate_logs(logs: list[dict[str, Any]], char_limit: int) -> list[dict[str, Any]]:
    """Truncate individual log bodies to char_limit."""
    result = []
    for log in logs:
        log_copy = dict(log)
        body = log_copy.get("body", "")
        if isinstance(body, str) and len(body) > char_limit:
            log_copy["body"] = body[:char_limit] + f"... [truncated {len(body) - char_limit} chars]"
        result.append(log_copy)
    return result


def _collect_span_index(spans: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Recursively collect span_id/span_name/status_code from all spans."""
    index: list[dict[str, str]] = []
    for span in spans:
        index.append({
            "span_id": span.get("span_id", ""),
            "span_name": span.get("span_name", ""),
            "status_code": span.get("status_code", ""),
        })
        children = span.get("child_spans", [])
        if children:
            index.extend(_collect_span_index(children))
    return index


def _truncate_span(span: dict[str, Any], is_error: bool) -> dict[str, Any]:
    """Return a truncated copy of a span."""
    span_copy = dict(span)
    char_limit = _ERROR_LOG_CHAR_LIMIT if is_error else _NORMAL_LOG_CHAR_LIMIT
    if "logs" in span_copy:
        span_copy["logs"] = _truncate_logs(span_copy["logs"], char_limit)
    # Truncate span_attributes values
    if "span_attributes" in span_copy and isinstance(span_copy["span_attributes"], dict):
        attrs = {}
        for k, v in span_copy["span_attributes"].items():
            if isinstance(v, str) and len(v) > char_limit:
                attrs[k] = v[:char_limit] + f"... [truncated {len(v) - char_limit} chars]"
            else:
                attrs[k] = v
        span_copy["span_attributes"] = attrs
    # Recurse into child_spans
    if "child_spans" in span_copy and span_copy["child_spans"]:
        span_copy["child_spans"] = [
            _truncate_span(child, _is_error_span(child))
            for child in span_copy["child_spans"]
        ]
    return span_copy


def _drop_order_middle_out(droppable: list[int]) -> list[int]:
    """Return indices in middle-outward order for progressive dropping."""
    if not droppable:
        return []
    mid = len(droppable) // 2
    order: list[int] = []
    left, right = mid - 1, mid
    while left >= 0 or right < len(droppable):
        if right < len(droppable):
            order.append(droppable[right])
            right += 1
        if left >= 0:
            order.append(droppable[left])
            left -= 1
    return order


def _prune_child_spans_recursive(trace: dict[str, Any], max_chars: int) -> bool:
    """Recursively prune non-error child spans from deepest levels first.

    TRAIL traces typically have 1 root span with hundreds of nested child_spans.
    This function walks the tree depth-first and drops non-error children at each
    level (middle-outward), keeping first/last and all error spans at every level.

    Mutates trace in place. Returns True if trace fits within max_chars.
    """

    def _gather_levels(
        spans: list[dict[str, Any]], depth: int = 0,
    ) -> list[tuple[int, dict[str, Any]]]:
        levels: list[tuple[int, dict[str, Any]]] = []
        for span in spans:
            children = span.get("child_spans", [])
            if len(children) > 2:
                levels.append((depth, span))
            if children:
                levels.extend(_gather_levels(children, depth + 1))
        return levels

    spans = trace.get("spans", [])
    levels = _gather_levels(spans)
    # Deepest first — prune leaves before parents
    levels.sort(key=lambda x: -x[0])

    for _depth, parent in levels:
        if len(json.dumps(trace)) <= max_chars:
            return True

        children = parent.get("child_spans", [])
        if len(children) <= 2:
            continue

        error_indices = {i for i, c in enumerate(children) if _is_error_span(c)}
        keep_indices = {0, len(children) - 1} | error_indices
        droppable = [i for i in range(len(children)) if i not in keep_indices]

        if not droppable:
            continue

        drop_order = _drop_order_middle_out(droppable)
        dropped: set[int] = set()
        for idx in drop_order:
            dropped.add(idx)
            parent["child_spans"] = [c for i, c in enumerate(children) if i not in dropped]
            if len(json.dumps(trace)) <= max_chars:
                return True

    return len(json.dumps(trace)) <= max_chars


def smart_truncate_trace(
    trace: dict[str, Any],
    max_chars: int = 200_000,
) -> dict[str, Any]:
    """Truncate a trace dict to fit within max_chars when JSON-serialized.

    Strategy:
    - If the trace already fits, return it unchanged (with span_index added).
    - Phase 1: truncate log bodies (error spans get generous limit, others tight).
    - Phase 2a: recursively prune non-error child_spans at every tree level
      (deepest first, middle-outward). This handles TRAIL traces that have
      1 root span with hundreds of nested children.
    - Phase 2b: drop top-level spans from the middle (for traces with many
      top-level spans).
    - Phase 3-4: shrink span_index as last resort.
    """
    trace = deepcopy(trace)
    spans = trace.get("spans", [])

    # Always build the span index
    span_index = _collect_span_index(spans)

    serialized = json.dumps(trace)
    if len(serialized) <= max_chars:
        # Small enough — just add index
        trace["span_index"] = span_index
        return trace

    # Phase 1: truncate log bodies
    truncated_spans = [
        _truncate_span(span, _is_error_span(span))
        for span in spans
    ]
    trace["spans"] = truncated_spans
    trace["span_index"] = span_index

    serialized = json.dumps(trace)
    if len(serialized) <= max_chars:
        return trace

    # Phase 2a: recursively prune nested child_spans (deepest first)
    if _prune_child_spans_recursive(trace, max_chars):
        return trace

    # Phase 2b: drop top-level spans from the middle (if multiple top-level)
    truncated_spans = trace["spans"]  # May have been modified by 2a
    if len(truncated_spans) > 2:
        error_indices = {i for i, s in enumerate(truncated_spans) if _is_error_span(s)}
        keep_indices = {0, len(truncated_spans) - 1} | error_indices
        droppable = [i for i in range(len(truncated_spans)) if i not in keep_indices]
        drop_order = _drop_order_middle_out(droppable)

        dropped: set[int] = set()
        for idx in drop_order:
            dropped.add(idx)
            trace["spans"] = [s for i, s in enumerate(truncated_spans) if i not in dropped]
            trace["span_index"] = span_index
            if len(json.dumps(trace)) <= max_chars:
                return trace

    if len(json.dumps(trace)) <= max_chars:
        return trace

    # Phase 3: shrink span_index entry count until payload fits
    full_index = list(span_index)
    while full_index and len(json.dumps(trace)) > max_chars:
        if len(full_index) <= 1:
            full_index = []
        else:
            full_index = full_index[: len(full_index) // 2]
        trace["span_index"] = full_index
        if len(json.dumps(trace)) <= max_chars:
            return trace

    # Phase 4: compact span_index entries to span_id-only and shrink again
    compact_index = [{"span_id": entry.get("span_id", "")} for entry in trace.get("span_index", [])]
    trace["span_index"] = compact_index
    while compact_index and len(json.dumps(trace)) > max_chars:
        if len(compact_index) <= 1:
            compact_index = []
        else:
            compact_index = compact_index[: len(compact_index) // 2]
        trace["span_index"] = compact_index
        if len(json.dumps(trace)) <= max_chars:
            return trace

    return trace


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------


def build_single_pass_message(
    trace: dict[str, Any],
    max_chars: int = 200_000,
) -> str:
    """Build the full single-pass prompt with the trace inlined.

    Applies smart truncation before inserting the trace JSON into the prompt.
    """
    truncated = smart_truncate_trace(trace, max_chars=max_chars)
    trace_json = json.dumps(truncated, indent=2)
    return TRAIL_SINGLE_PASS_PROMPT_V2.replace("{trace}", trace_json)
