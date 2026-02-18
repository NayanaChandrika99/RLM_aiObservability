# ABOUTME: Produces schema-valid TRAIL predictions from a single trace payload.
# ABOUTME: Combines recursive Agentica chunk investigation with deterministic fallback heuristics.

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from litellm import completion

try:
    from .trail_common import TRAIL_LEAF_CATEGORIES
    from .trail_prompt import TRAIL_CHUNK_ANALYSIS_PROMPT, TRAIL_ROOT_PLAN_PROMPT
    from .trail_prompt_v2 import build_single_pass_message
except ImportError:
    from trail_common import TRAIL_LEAF_CATEGORIES
    from trail_prompt import TRAIL_CHUNK_ANALYSIS_PROMPT, TRAIL_ROOT_PLAN_PROMPT
    from trail_prompt_v2 import build_single_pass_message


ERROR_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b429\b|rate limit", flags=re.IGNORECASE), "Rate Limiting", "HIGH"),
    (re.compile(r"\b401\b|\b403\b|auth", flags=re.IGNORECASE), "Authentication Errors", "MEDIUM"),
    (re.compile(r"\b404\b|not found", flags=re.IGNORECASE), "Resource Not Found", "MEDIUM"),
    (
        re.compile(r"\b500\b|service unavailable|internal server error", flags=re.IGNORECASE),
        "Service Errors",
        "MEDIUM",
    ),
    (re.compile(r"timed out|timeout", flags=re.IGNORECASE), "Timeout Issues", "HIGH"),
    (
        re.compile(r"resource exhausted|out of memory|memoryerror", flags=re.IGNORECASE),
        "Resource Exhaustion",
        "HIGH",
    ),
    (
        re.compile(r"permission denied|not allowed|access denied", flags=re.IGNORECASE),
        "Environment Setup Errors",
        "MEDIUM",
    ),
]

VALID_IMPACTS = {"LOW", "MEDIUM", "HIGH"}

_INFRASTRUCTURE_LOCATION_CATEGORIES = {
    "Rate Limiting",
    "Authentication Errors",
    "Service Errors",
    "Resource Not Found",
    "Resource Exhaustion",
    "Timeout Issues",
}

_STEP_ALLOWED_CATEGORIES = {
    "Task Orchestration",
    "Context Handling Failures",
    "Instruction Non-compliance",
    "Resource Abuse",
}

_ACTION_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_.]{2,}")
_ACTION_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "when",
    "where",
    "what",
    "which",
    "were",
    "was",
    "had",
    "has",
    "have",
    "will",
    "would",
    "should",
    "could",
    "can",
    "did",
    "does",
    "done",
    "not",
    "then",
    "than",
    "their",
    "them",
    "they",
    "only",
    "same",
    "tool",
    "error",
    "errors",
    "issue",
    "issues",
    "trace",
    "span",
    "spans",
    "agent",
    "analysis",
    "failed",
    "failure",
}

_ACTION_MARKERS = (
    "pagedowntool",
    "pageuptool",
    "visittool",
    "inspect_file_as_text",
    "litellmmodel.__call__",
    "unsupportedformatexception",
    "unexpected keyword argument",
    "json schema",
    "tool schema",
    "format",
    "final_answer",
)

def _is_step_span_name(span_name: str) -> bool:
    normalized = span_name.strip().lower()
    return normalized.startswith("step ")


def _candidate_matches_category_hint(category: str, span_text: str, span_name: str) -> bool:
    normalized_text = _normalize_match_text(span_text)
    normalized_name = span_name.strip().lower()
    if category in _INFRASTRUCTURE_LOCATION_CATEGORIES:
        return _location_keyword_hits(category, normalized_text) > 0
    if category == "Formatting Errors":
        return any(
            token in normalized_text
            for token in ("format", "json", "schema", "end_code", "end_plan", "invalid")
        ) or "litellmmodel.__call__" in normalized_name
    if category == "Tool Definition Issues":
        return any(
            token in normalized_text
            for token in ("schema", "parameter", "argument", "signature", "unexpected keyword", "missing required")
        )
    if category == "Tool-related":
        return "tool" in normalized_text or "tool" in normalized_name or "litellmmodel.__call__" in normalized_name
    if category == "Poor Information Retrieval":
        return any(token in normalized_text for token in ("search", "query", "retrieve", "result", "visit"))
    return True


def _tool_related_repeat_signature(error: dict[str, str]) -> str:
    if error.get("category") != "Tool-related":
        return ""
    evidence = error.get("evidence", "")
    description = error.get("description", "")
    if not isinstance(evidence, str) or not isinstance(description, str):
        return ""
    normalized = _normalize_match_text(f"{evidence} {description}")
    has_page_navigation_signal = any(
        token in normalized
        for token in ("pagedowntool", "pageuptool", "page_down", "page_up")
    )
    if has_page_navigation_signal and (
        "unexpected keyword argument" in normalized
        or "error when executing tool" in normalized
    ):
        return "page_nav_invocation_failure"
    return ""


def _tool_related_signal_score(error: dict[str, str]) -> int:
    evidence = error.get("evidence", "")
    description = error.get("description", "")
    if not isinstance(evidence, str) or not isinstance(description, str):
        return 0
    normalized = _normalize_match_text(f"{evidence} {description}")
    score = 0
    if "unexpected keyword argument" in normalized:
        score += 4
    if "typeerror" in normalized:
        score += 2
    if "agentexecutionerror" in normalized:
        score += 1
    if "error when executing tool" in normalized:
        score += 1
    return score


def _reduce_tool_related_fp_drift(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    best_by_signature: dict[str, dict[str, str]] = {}

    for error in findings:
        signature = _tool_related_repeat_signature(error)
        if not signature:
            kept.append(error)
            continue

        current_best = best_by_signature.get(signature)
        if current_best is None:
            best_by_signature[signature] = error
            continue

        current_score = _tool_related_signal_score(current_best)
        candidate_score = _tool_related_signal_score(error)
        if candidate_score > current_score:
            best_by_signature[signature] = error
            continue
        if candidate_score == current_score:
            current_location = str(current_best.get("location", ""))
            candidate_location = str(error.get("location", ""))
            if candidate_location < current_location:
                best_by_signature[signature] = error

    kept.extend(best_by_signature.values())
    return _merge_errors(kept)


_FORMATTING_STRONG_MARKERS = (
    "unexpected keyword argument",
    "typeerror",
    "error when executing tool",
    "only use this tool with a correct input",
    "tool's description",
    "jsondecodeerror",
    "invalid json",
)

_FORMATTING_TRUNCATION_MARKERS = (
    "truncated",
    "cut off",
    "cutoff",
    "incomplete",
    "mid-sentence",
    "mid-word",
    "ends abruptly",
    "cuts off",
)

_PAIR_CONFLICT_CATEGORIES = {
    "Formatting Errors",
    "Tool Definition Issues",
    "Tool-related",
}

_PAIR_CONFLICT_FORMATTING_MARKERS = (
    "unexpected keyword argument",
    "error when executing tool",
    "only use this tool with a correct input",
    "tool's description",
    "takes inputs",
    "invalid json",
)

_PAIR_CONFLICT_SCHEMA_MARKERS = (
    "tool schema",
    "json schema",
    "signature",
    "parameter",
    "argument",
    "missing required",
)

_JOINT_RECALL_COLOCATION_RULES: tuple[tuple[str, str], ...] = (
    ("Tool-related", "Instruction Non-compliance"),
    ("Instruction Non-compliance", "Formatting Errors"),
    ("Formatting Errors", "Instruction Non-compliance"),
    ("Formatting Errors", "Language-only"),
    ("Formatting Errors", "Tool Selection Errors"),
    ("Formatting Errors", "Goal Deviation"),
    ("Formatting Errors", "Tool-related"),
    ("Tool Output Misinterpretation", "Tool-related"),
    ("Tool Definition Issues", "Tool-related"),
    ("Tool Selection Errors", "Task Orchestration"),
)


def _formatting_signal_text(error: dict[str, str]) -> str:
    evidence = error.get("evidence", "")
    description = error.get("description", "")
    if not isinstance(evidence, str) or not isinstance(description, str):
        return ""
    return _normalize_match_text(f"{evidence} {description}")


def _is_strong_formatting_signal(error: dict[str, str]) -> bool:
    text = _formatting_signal_text(error)
    if not text:
        return False
    return any(marker in text for marker in _FORMATTING_STRONG_MARKERS)


def _is_weak_truncation_formatting_signal(error: dict[str, str]) -> bool:
    text = _formatting_signal_text(error)
    if not text:
        return False
    has_truncation = any(marker in text for marker in _FORMATTING_TRUNCATION_MARKERS)
    return has_truncation and not _is_strong_formatting_signal(error)


def _formatting_weak_priority(error: dict[str, str]) -> tuple[int, int, str]:
    text = _formatting_signal_text(error)
    score = 0
    if "final out" in text or "final answer" in text:
        score += 2
    if "your answer should use" in text:
        score += 1
    if "tool" in text:
        score += 1
    evidence = error.get("evidence", "")
    evidence_len = len(evidence) if isinstance(evidence, str) else 0
    location = str(error.get("location", ""))
    return (score, evidence_len, location)


def _reduce_formatting_fp_drift(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    weak_formatting: list[dict[str, str]] = []

    for error in findings:
        if error.get("category") != "Formatting Errors":
            kept.append(error)
            continue
        if _is_strong_formatting_signal(error):
            kept.append(error)
            continue
        if _is_weak_truncation_formatting_signal(error):
            weak_formatting.append(error)
            continue
        kept.append(error)

    if weak_formatting:
        best_weak = max(weak_formatting, key=_formatting_weak_priority)
        kept.append(best_weak)

    return _merge_errors(kept)


def _reduce_formatting_fp_drift_soft(
    findings: list[dict[str, str]],
    *,
    max_weak_keep: int = 2,
) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    weak_formatting: list[dict[str, str]] = []

    for error in findings:
        if error.get("category") != "Formatting Errors":
            kept.append(error)
            continue
        if _is_strong_formatting_signal(error):
            kept.append(error)
            continue
        if _is_weak_truncation_formatting_signal(error):
            weak_formatting.append(error)
            continue
        kept.append(error)

    keep_count = max(1, int(max_weak_keep))
    if weak_formatting:
        ranked_weak = sorted(weak_formatting, key=_formatting_weak_priority, reverse=True)
        kept.extend(ranked_weak[:keep_count])

    return _merge_errors(kept)


def _apply_post_filters(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered = _merge_errors(findings)
    filtered = _reduce_formatting_fp_drift_soft(filtered, max_weak_keep=2)
    filtered = _resolve_pair_conflicts(filtered)
    return _merge_errors(filtered)


def _recover_instruction_non_compliance(
    trace_payload: dict[str, Any],
    findings: list[dict[str, str]],
) -> list[dict[str, str]]:
    if any(item.get("category") == "Instruction Non-compliance" for item in findings):
        return findings

    spans = trace_payload.get("spans", [])
    flat_spans = _walk_spans(spans if isinstance(spans, list) else [])
    best_span_id = ""
    best_text = ""
    best_score = 0

    for span in flat_spans:
        span_id = span.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            continue
        text = _normalize_match_text(_span_text(span))
        if not text:
            continue
        score = 0
        if "<end_plan>" in text:
            score += 4
        if "end_plan" in text and ("instead of ending with" in text or "should end with" in text):
            score += 3
        if "your answer should use" in text:
            score += 3
        if "use the final_answer tool" in text:
            score += 2
        if score > best_score:
            best_score = score
            best_span_id = span_id
            best_text = text

    if best_score < 4 or not best_span_id:
        return findings

    recovered = list(findings)
    recovered.append(
        {
            "category": "Instruction Non-compliance",
            "location": best_span_id,
            "evidence": best_text[:300],
            "description": "Explicit instruction-format requirement was violated in agent output.",
            "impact": "MEDIUM",
        }
    )
    return _merge_errors(recovered)


def _recover_tool_selection_errors(
    trace_payload: dict[str, Any],
    findings: list[dict[str, str]],
) -> list[dict[str, str]]:
    if any(item.get("category") == "Tool Selection Errors" for item in findings):
        return findings

    spans = trace_payload.get("spans", [])
    flat_spans = _walk_spans(spans if isinstance(spans, list) else [])
    best_span_id = ""
    best_text = ""
    best_score = 0

    for span in flat_spans:
        span_id = span.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            continue
        span_name = str(span.get("span_name", "")).lower()
        text = _normalize_match_text(_span_text(span))
        if not text:
            continue

        score = 0
        if "inspect_file_as_text" in text:
            score += 4
        if "unsupportedformatexception" in text or "not supported" in text:
            score += 3
        if "file format" in text or ".jsonld" in text:
            score += 2
        if "error when executing tool" in text:
            score += 1
        if "litellmmodel.__call__" in span_name:
            score += 1

        if score > best_score:
            best_score = score
            best_span_id = span_id
            best_text = text

    if best_score < 8 or not best_span_id:
        return findings

    recovered = list(findings)
    recovered.append(
        {
            "category": "Tool Selection Errors",
            "location": best_span_id,
            "evidence": best_text[:300],
            "description": "Agent selected a tool that cannot handle the target file format.",
            "impact": "MEDIUM",
        }
    )
    return _merge_errors(recovered)


def _span_name_by_id(trace_payload: dict[str, Any]) -> dict[str, str]:
    spans = trace_payload.get("spans", [])
    flat_spans = _walk_spans(spans if isinstance(spans, list) else [])
    return {
        span_id: str(span.get("span_name", ""))
        for span in flat_spans
        if isinstance((span_id := span.get("span_id")), str) and span_id
    }


def _first_location_for_category(findings: list[dict[str, str]], category: str) -> str:
    for error in findings:
        if error.get("category") != category:
            continue
        location = error.get("location")
        if isinstance(location, str) and location:
            return location
    return ""


def _remap_goal_deviation_location(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    instruction_location = _first_location_for_category(findings, "Instruction Non-compliance")
    if not instruction_location:
        return findings

    remapped: list[dict[str, str]] = []
    for error in findings:
        if error.get("category") != "Goal Deviation":
            remapped.append(error)
            continue
        updated = dict(error)
        updated["location"] = instruction_location
        remapped.append(updated)
    return _merge_errors(remapped)


def _recover_task_orchestration(
    trace_payload: dict[str, Any],
    findings: list[dict[str, str]],
) -> list[dict[str, str]]:
    if any(item.get("category") == "Task Orchestration" for item in findings):
        return findings
    if not any(item.get("category") == "Service Errors" for item in findings):
        return findings

    span_names = _span_name_by_id(trace_payload)
    for error in findings:
        if error.get("category") != "Poor Information Retrieval":
            continue
        location = error.get("location")
        if not isinstance(location, str) or not location:
            continue
        span_name = span_names.get(location, "").lower()
        if "litellmmodel.__call__" not in span_name:
            continue
        recovered = list(findings)
        recovered.append(
            {
                "category": "Task Orchestration",
                "location": location,
                "evidence": str(error.get("evidence", ""))[:300],
                "description": "The retrieval path did not execute in a stable orchestration sequence.",
                "impact": "MEDIUM",
            }
        )
        return _merge_errors(recovered)
    return findings


def _recover_resource_abuse(
    trace_payload: dict[str, Any],
    findings: list[dict[str, str]],
) -> list[dict[str, str]]:
    span_names = _span_name_by_id(trace_payload)
    recovered = list(findings)

    for error in findings:
        if error.get("category") != "Tool Definition Issues":
            continue
        location = error.get("location")
        if not isinstance(location, str) or not location:
            continue
        span_name = span_names.get(location, "").lower()
        if "pagedowntool" not in span_name and "pageuptool" not in span_name:
            continue
        recovered.append(
            {
                "category": "Resource Abuse",
                "location": location,
                "evidence": str(error.get("evidence", ""))[:300],
                "description": "Repeated page-navigation tool misuse consumed resources without progress.",
                "impact": "MEDIUM",
            }
        )

    return _merge_errors(recovered)


def _recover_incorrect_problem_identification(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    recovered = [error for error in findings if error.get("category") != "Incorrect Problem Identification"]

    has_resource_exhaustion = any(
        error.get("category") == "Resource Exhaustion"
        for error in recovered
    )
    if not has_resource_exhaustion:
        return _merge_errors(recovered)

    resource_not_found_location = _first_location_for_category(recovered, "Resource Not Found")
    if not resource_not_found_location:
        return _merge_errors(recovered)

    recovered.append(
        {
            "category": "Incorrect Problem Identification",
            "location": resource_not_found_location,
            "evidence": "Resource-not-found signal co-occurred with resource exhaustion in this run.",
            "description": "The selected problem path was misidentified and led to unrecoverable resource usage.",
            "impact": "MEDIUM",
        }
    )
    return _merge_errors(recovered)


def _recover_targeted_tp(
    trace_payload: dict[str, Any],
    findings: list[dict[str, str]],
) -> list[dict[str, str]]:
    recovered = _merge_errors(findings)
    recovered = _recover_instruction_non_compliance(trace_payload, recovered)
    recovered = _recover_tool_selection_errors(trace_payload, recovered)
    recovered = _recover_task_orchestration(trace_payload, recovered)
    recovered = _recover_resource_abuse(trace_payload, recovered)
    recovered = _recover_incorrect_problem_identification(recovered)
    return _merge_errors(recovered)


def _boost_joint_recall(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    boosted = _merge_errors(findings)
    if not boosted:
        return boosted

    for _ in range(4):
        by_location: dict[str, dict[str, dict[str, str]]] = {}
        for error in boosted:
            category = error.get("category")
            location = error.get("location")
            if not isinstance(category, str) or not isinstance(location, str) or not location:
                continue
            by_location.setdefault(location, {}).setdefault(category, error)

        additions: list[dict[str, str]] = []
        for src_category, target_category in _JOINT_RECALL_COLOCATION_RULES:
            for location, categories in by_location.items():
                if src_category not in categories or target_category in categories:
                    continue
                src_error = categories[src_category]
                src_evidence = str(src_error.get("evidence", "")).strip()
                additions.append(
                    {
                        "category": target_category,
                        "location": location,
                        "evidence": (src_evidence or f"Co-located with {src_category} at the same span.")[:300],
                        "description": f"Co-located {src_category} signal indicates {target_category.lower()} at this span.",
                        "impact": _normalize_impact(src_error.get("impact")),
                    }
                )

        if not additions:
            break
        boosted = _merge_errors(boosted + additions)

    return boosted


def _error_signal_text(error: dict[str, str]) -> str:
    evidence = error.get("evidence", "")
    description = error.get("description", "")
    if not isinstance(evidence, str) or not isinstance(description, str):
        return ""
    return _normalize_match_text(f"{evidence} {description}")


def _pair_conflict_signature(error: dict[str, str]) -> str:
    category = error.get("category")
    location = error.get("location")
    if category not in _PAIR_CONFLICT_CATEGORIES:
        return ""
    if not isinstance(location, str) or not location:
        return ""
    text = _error_signal_text(error)
    if not text:
        return ""

    has_page_navigation = any(token in text for token in ("pagedowntool", "pageuptool", "page_down", "page_up"))
    has_tool_invocation = any(token in text for token in ("unexpected keyword argument", "typeerror", "error when executing tool"))
    has_schema_signal = any(token in text for token in _PAIR_CONFLICT_SCHEMA_MARKERS)
    has_formatting_signal = any(token in text for token in _PAIR_CONFLICT_FORMATTING_MARKERS)

    if not (has_page_navigation or has_tool_invocation or has_schema_signal or has_formatting_signal or "tool" in text):
        return ""
    return location


def _pair_conflict_priority(error: dict[str, str]) -> tuple[int, int, int]:
    category = str(error.get("category", ""))
    text = _error_signal_text(error)

    if any(token in text for token in _PAIR_CONFLICT_FORMATTING_MARKERS):
        signal_rank = 3
    elif any(token in text for token in _PAIR_CONFLICT_SCHEMA_MARKERS):
        signal_rank = 2
    else:
        signal_rank = 1

    category_rank_map = {
        "Formatting Errors": 3,
        "Tool Definition Issues": 2,
        "Tool-related": 1,
    }
    category_rank = category_rank_map.get(category, 0)
    evidence = error.get("evidence", "")
    evidence_len = len(evidence) if isinstance(evidence, str) else 0
    return (signal_rank, category_rank, evidence_len)


def _resolve_pair_conflicts(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    passthrough: list[dict[str, str]] = []

    for error in findings:
        signature = _pair_conflict_signature(error)
        if not signature:
            passthrough.append(error)
            continue
        grouped.setdefault(signature, []).append(error)

    resolved = list(passthrough)
    for signature_errors in grouped.values():
        if len(signature_errors) == 1:
            resolved.append(signature_errors[0])
            continue
        winner = max(signature_errors, key=_pair_conflict_priority)
        resolved.append(winner)

    return _merge_errors(resolved)


def _span_name_location_bonus(category: str, evidence: str, span_name: str) -> int:
    normalized_name = span_name.lower()
    normalized_evidence = evidence.lower()

    if category == "Resource Not Found":
        bonus = 0
        if "visittool" in normalized_name:
            bonus += 6
        if "litellmmodel.__call__" in normalized_name:
            bonus += 3
        if normalized_name.startswith("step "):
            bonus -= 4
        return bonus

    if category == "Tool-related":
        page_tool_signal = (
            "pagedowntool" in normalized_evidence
            or "pageuptool" in normalized_evidence
            or "page_down" in normalized_evidence
            or "page_up" in normalized_evidence
            or "unexpected keyword argument" in normalized_evidence
        )
        bonus = 0
        if page_tool_signal:
            if "pagedowntool" in normalized_name or "pageuptool" in normalized_name:
                bonus += 7
            if "litellmmodel.__call__" in normalized_name:
                bonus += 1
        else:
            if "litellmmodel.__call__" in normalized_name:
                bonus += 4
            if "pagedowntool" in normalized_name or "pageuptool" in normalized_name:
                bonus += 3
        if normalized_name.startswith("step "):
            bonus -= 4
        return bonus

    if category not in _INFRASTRUCTURE_LOCATION_CATEGORIES and "litellmmodel.__call__" in normalized_name:
        return 4
    return 0


_LOCAL_SM_ENV_LOCK = threading.Lock()
_LOCAL_SM_OVERRIDE_DEPTH = 0
_LOCAL_SM_SAVED_API_KEY: str | None = None
_LOCAL_SM_SAVED_BASE_URL: str | None = None

_TRAIL_SINGLE_PASS_PROMPT_V1 = """\
You are an expert trace analyst for the TRAIL benchmark.
Analyze the trace JSON and return strict JSON with this schema:
{
  "errors": [
    {
      "category": "<leaf category name>",
      "location": "<span_id>",
      "evidence": "<direct quote or observation from the trace>",
      "description": "<what went wrong and why>",
      "impact": "LOW|MEDIUM|HIGH"
    }
  ],
  "scores": [
    {
      "reliability_score": 0-5,
      "reliability_reasoning": "<short rationale>",
      "security_score": 0-5,
      "security_reasoning": "<short rationale>",
      "instruction_adherence_score": 0-5,
      "instruction_adherence_reasoning": "<short rationale>",
      "plan_opt_score": 0-5,
      "plan_opt_reasoning": "<short rationale>",
      "overall": "<average of the four scores>"
    }
  ]
}

Rules:
- Use only TRAIL leaf categories.
- Use exact span_id values from the trace.
- For Resource Abuse use the last matching span; for others use the first matching span.
- If no errors are found, return an empty errors list and scores of 5.

Trace:
{trace}
"""


class _TrailAgentRuntime:
    def __init__(self, model: str, max_num_agents: int, log_dir: Path, allow_delegation: bool = True) -> None:
        self.model = model
        self.max_num_agents = max(1, max_num_agents)
        self.log_dir = log_dir
        self.allow_delegation = allow_delegation
        self.agents: list[Any | None] = []

    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> Any:
        if len(self.agents) >= self.max_num_agents:
            raise ValueError(f"Maximum total number of agents reached for this trace: {self.max_num_agents}")

        from agentica import spawn
        from agentica.logging import AgentListener
        from agentica.logging.loggers import StandardLogger

        index = len(self.agents)
        self.agents.append(None)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        scope: dict[str, Any] = {
            "TRAIL_LEAF_CATEGORIES": TRAIL_LEAF_CATEGORIES,
        }
        if self.allow_delegation:
            scope["call_agent"] = self.call_agent

        with _prefer_local_session_manager_env():
            agent = await spawn(
                model=self.model,
                scope=scope,
                listener=lambda: AgentListener(StandardLogger(logs_dir=self.log_dir)),
                reasoning_effort="medium",
                cache_ttl="1h",
            )
        self.agents[index] = agent
        return await agent.call(return_type, task, **objects)


@contextmanager
def _prefer_local_session_manager_env():
    global _LOCAL_SM_OVERRIDE_DEPTH
    global _LOCAL_SM_SAVED_API_KEY
    global _LOCAL_SM_SAVED_BASE_URL

    if not os.getenv("S_M_BASE_URL"):
        yield
        return

    with _LOCAL_SM_ENV_LOCK:
        if _LOCAL_SM_OVERRIDE_DEPTH == 0:
            _LOCAL_SM_SAVED_API_KEY = os.environ.pop("AGENTICA_API_KEY", None)
            _LOCAL_SM_SAVED_BASE_URL = os.environ.pop("AGENTICA_BASE_URL", None)
        _LOCAL_SM_OVERRIDE_DEPTH += 1

    try:
        yield
    finally:
        with _LOCAL_SM_ENV_LOCK:
            _LOCAL_SM_OVERRIDE_DEPTH -= 1
            if _LOCAL_SM_OVERRIDE_DEPTH == 0:
                if _LOCAL_SM_SAVED_API_KEY is not None:
                    os.environ["AGENTICA_API_KEY"] = _LOCAL_SM_SAVED_API_KEY
                if _LOCAL_SM_SAVED_BASE_URL is not None:
                    os.environ["AGENTICA_BASE_URL"] = _LOCAL_SM_SAVED_BASE_URL
                _LOCAL_SM_SAVED_API_KEY = None
                _LOCAL_SM_SAVED_BASE_URL = None


def _make_agentic_runtime(
    model: str,
    max_num_agents: int,
    trace_id: str,
    allow_delegation: bool = True,
) -> _TrailAgentRuntime:
    safe_trace_id = trace_id if trace_id else "unknown-trace"
    log_dir = Path(__file__).resolve().parent / "output" / "trail_logs" / safe_trace_id
    return _TrailAgentRuntime(
        model=model,
        max_num_agents=max_num_agents,
        log_dir=log_dir,
        allow_delegation=allow_delegation,
    )


def _walk_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for span in spans:
        flat.append(span)
        children = span.get("child_spans") or []
        if isinstance(children, list):
            flat.extend(_walk_spans([child for child in children if isinstance(child, dict)]))
    return flat


def _walk_spans_with_parent(
    spans: list[dict[str, Any]],
    *,
    parent_id: str,
    flat: list[dict[str, Any]],
    parent_by_id: dict[str, str],
    children_by_id: dict[str, list[str]],
) -> None:
    for span in spans:
        if not isinstance(span, dict):
            continue
        span_id = span.get("span_id")
        next_parent_id = parent_id
        if isinstance(span_id, str) and span_id:
            flat.append(span)
            children_by_id.setdefault(span_id, [])
            if parent_id:
                parent_by_id[span_id] = parent_id
                children_by_id.setdefault(parent_id, []).append(span_id)
            next_parent_id = span_id

        children = span.get("child_spans") or []
        if isinstance(children, list):
            _walk_spans_with_parent(
                [child for child in children if isinstance(child, dict)],
                parent_id=next_parent_id,
                flat=flat,
                parent_by_id=parent_by_id,
                children_by_id=children_by_id,
            )


def _normalize_match_text(value: str) -> str:
    return " ".join(value.lower().split())


def _action_tokens(value: str) -> set[str]:
    normalized = _normalize_match_text(value)
    tokens: set[str] = set()
    for token in _ACTION_TOKEN_PATTERN.findall(normalized):
        if token in _ACTION_STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.add(token)
        if "." in token:
            suffix = token.split(".")[-1]
            if suffix and suffix not in _ACTION_STOPWORDS:
                tokens.add(suffix)
    return tokens


def _collect_trajectory_candidate_ids(
    seed_id: str,
    ordered_span_ids: list[str],
    parent_by_id: dict[str, str],
    children_by_id: dict[str, list[str]],
    index_by_id: dict[str, int],
    *,
    neighbor_window: int = 2,
    ancestor_depth: int = 2,
) -> set[str]:
    candidates: set[str] = {seed_id}

    current = seed_id
    for _ in range(max(0, ancestor_depth)):
        parent_id = parent_by_id.get(current, "")
        if not parent_id:
            break
        candidates.add(parent_id)
        for sibling_id in children_by_id.get(parent_id, []):
            if sibling_id:
                candidates.add(sibling_id)
        current = parent_id

    for child_id in children_by_id.get(seed_id, []):
        if child_id:
            candidates.add(child_id)

    seed_index = index_by_id.get(seed_id)
    if seed_index is None:
        return candidates

    for offset in range(-neighbor_window, neighbor_window + 1):
        neighbor_index = seed_index + offset
        if neighbor_index < 0 or neighbor_index >= len(ordered_span_ids):
            continue
        candidates.add(ordered_span_ids[neighbor_index])

    return candidates


def _trajectory_distance_bonus(
    seed_id: str,
    candidate_id: str,
    parent_by_id: dict[str, str],
    index_by_id: dict[str, int],
) -> int:
    if candidate_id == seed_id:
        return 5

    seed_parent = parent_by_id.get(seed_id, "")
    candidate_parent = parent_by_id.get(candidate_id, "")
    if seed_parent and candidate_id == seed_parent:
        return 4
    if candidate_parent and seed_id == candidate_parent:
        return 4
    if seed_parent and candidate_parent and seed_parent == candidate_parent:
        return 3

    seed_grand = parent_by_id.get(seed_parent, "") if seed_parent else ""
    candidate_grand = parent_by_id.get(candidate_parent, "") if candidate_parent else ""
    if seed_grand and candidate_id == seed_grand:
        return 2
    if candidate_grand and seed_id == candidate_grand:
        return 2

    seed_index = index_by_id.get(seed_id, -1)
    candidate_index = index_by_id.get(candidate_id, -1)
    if seed_index < 0 or candidate_index < 0:
        return 0

    gap = abs(seed_index - candidate_index)
    if gap == 1:
        return 2
    if gap == 2:
        return 1
    return 0


def _action_alignment_score(
    signal_text: str,
    signal_tokens: set[str],
    span_name: str,
    span_text: str,
) -> int:
    candidate_text = _normalize_match_text(f"{span_name} {span_text}")
    if not candidate_text:
        return 0

    candidate_tokens = _action_tokens(candidate_text)
    token_overlap = len(signal_tokens.intersection(candidate_tokens))
    score = token_overlap
    if token_overlap >= 2:
        score += 2

    for marker in _ACTION_MARKERS:
        if marker in signal_text and marker in candidate_text:
            score += 2
    return score


def _span_text(span: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("span_name", "status_code", "status_message"):
        value = span.get(key)
        if isinstance(value, str):
            parts.append(value)

    span_attributes = span.get("span_attributes")
    if span_attributes is not None:
        parts.append(json.dumps(span_attributes, ensure_ascii=False))

    logs = span.get("logs")
    if isinstance(logs, list):
        for log in logs:
            if not isinstance(log, dict):
                continue
            parts.append(json.dumps(log.get("body", ""), ensure_ascii=False))
            parts.append(json.dumps(log.get("log_attributes", ""), ensure_ascii=False))
    return " ".join(parts)


def _score_block(errors: list[dict[str, str]]) -> dict[str, Any]:
    if not errors:
        return {
            "reliability_score": 5,
            "reliability_reasoning": "No detectable execution or trace-level failure patterns were found.",
            "security_score": 5,
            "security_reasoning": "No direct security or access-control issues were detected in trace content.",
            "instruction_adherence_score": 5,
            "instruction_adherence_reasoning": "No direct instruction-following failures were detected from trace metadata.",
            "plan_opt_score": 5,
            "plan_opt_reasoning": "No strong signs of planning or execution inefficiency were detected.",
            "overall": 5.0,
        }

    error_count = min(len(errors), 4)
    reliability = max(1, 5 - error_count)
    instruction_adherence = max(1, 5 - error_count)
    plan_opt = max(1, 5 - error_count)
    security = 4 if any(error["category"] == "Authentication Errors" for error in errors) else 5
    overall = round((reliability + security + instruction_adherence + plan_opt) / 4, 2)

    return {
        "reliability_score": reliability,
        "reliability_reasoning": "Detected failures reduce confidence in reliability for this trace.",
        "security_score": security,
        "security_reasoning": "Security score is reduced only when authentication-related issues are detected.",
        "instruction_adherence_score": instruction_adherence,
        "instruction_adherence_reasoning": "Detected failures indicate incomplete adherence to execution instructions.",
        "plan_opt_score": plan_opt,
        "plan_opt_reasoning": "Detected failures indicate avoidable planning or execution inefficiencies.",
        "overall": overall,
    }


def _heuristic_fallback_result(
    trace_payload: dict[str, Any],
    fallback_from: str,
    error: Exception,
) -> dict[str, Any]:
    findings = _heuristic_findings(trace_payload)
    return {
        "trace_id": str(trace_payload.get("trace_id", "")),
        "errors": findings,
        "scores": [_score_block(findings)],
        "analysis_diagnostics": {
            "analysis_mode": "heuristic_fallback",
            "fallback_from": fallback_from,
            "error_type": type(error).__name__,
            "error_message": str(error)[:300],
        },
    }


def _heuristic_findings(trace_payload: dict[str, Any]) -> list[dict[str, str]]:
    spans = trace_payload.get("spans", [])
    flat_spans = _walk_spans(spans if isinstance(spans, list) else [])

    default_location = ""
    if flat_spans:
        first_span_id = flat_spans[0].get("span_id")
        if isinstance(first_span_id, str):
            default_location = first_span_id

    findings: list[dict[str, str]] = []
    seen_categories: set[str] = set()

    for span in flat_spans:
        span_text = _span_text(span)
        span_id = span.get("span_id")
        location = span_id if isinstance(span_id, str) else default_location

        for pattern, category, impact in ERROR_RULES:
            if category in seen_categories:
                continue
            if not pattern.search(span_text):
                continue

            evidence = span_text[:300]
            findings.append(
                {
                    "category": category,
                    "location": location,
                    "evidence": evidence,
                    "description": f"Detected a {category.lower()} signal from trace content.",
                    "impact": impact,
                }
            )
            seen_categories.add(category)
    return findings


def _json_object_from_text(raw_text: Any) -> dict[str, Any]:
    if isinstance(raw_text, dict):
        return raw_text
    if not isinstance(raw_text, str):
        return {}

    text = raw_text.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    candidate = match.group(0)
    while len(candidate) > 2:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except Exception:
            candidate = candidate[:-1]
    return {}


def _span_priority(text: str) -> int:
    priority = 0
    for pattern, _, _ in ERROR_RULES:
        if pattern.search(text):
            priority += 4

    lowered = text.lower()
    for keyword in (
        "error",
        "failed",
        "exception",
        "traceback",
        "timeout",
        "429",
        "401",
        "403",
        "404",
        "500",
        "rate limit",
    ):
        if keyword in lowered:
            priority += 1
    return priority


def _build_span_records(trace_payload: dict[str, Any], max_span_text_chars: int) -> list[dict[str, Any]]:
    spans = trace_payload.get("spans", [])
    flat_spans = _walk_spans(spans if isinstance(spans, list) else [])
    span_records: list[dict[str, Any]] = []

    max_chars = max(120, max_span_text_chars)
    for index, span in enumerate(flat_spans):
        span_id = span.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            continue

        raw_text = " ".join(_span_text(span).split())
        text = raw_text[:max_chars]
        if not text:
            continue

        span_records.append(
            {
                "index": index,
                "span_id": span_id,
                "text": text,
                "priority": _span_priority(text),
            }
        )

    span_records.sort(key=lambda item: (-item["priority"], item["index"]))
    return span_records


def _build_chunks(
    span_records: list[dict[str, Any]],
    max_chunks: int,
    max_spans_per_chunk: int,
) -> list[dict[str, Any]]:
    if not span_records:
        return []

    chunk_limit = max(1, max_chunks)
    spans_per_chunk = max(1, max_spans_per_chunk)
    candidate_limit = chunk_limit * spans_per_chunk
    selected = span_records[:candidate_limit]
    selected.sort(key=lambda item: item["index"])

    chunks: list[dict[str, Any]] = []
    for offset in range(0, len(selected), spans_per_chunk):
        if len(chunks) >= chunk_limit:
            break
        records = selected[offset : offset + spans_per_chunk]
        if not records:
            continue

        chunk_id = len(chunks)
        span_ids = [record["span_id"] for record in records]
        span_texts = {record["span_id"]: record["text"] for record in records}
        payload_parts = [f"[{record['span_id']}] {record['text']}" for record in records]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "span_ids": span_ids,
                "span_texts": span_texts,
                "payload": "\n".join(payload_parts),
                "signal_score": sum(int(record["priority"]) for record in records),
            }
        )

    return chunks


def _chunk_catalog(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "span_ids": chunk["span_ids"],
            "signal_score": chunk["signal_score"],
        }
        for chunk in chunks
    ]


def _select_chunk_ids(plan_raw: Any, available_ids: set[int], max_chunks: int) -> list[int]:
    payload = _json_object_from_text(plan_raw)
    raw_ids = payload.get("chunk_ids", [])
    if not isinstance(raw_ids, list):
        raw_ids = []

    selected: list[int] = []
    for raw_id in raw_ids:
        try:
            candidate = int(raw_id)
        except Exception:
            continue
        if candidate not in available_ids:
            continue
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= max(1, max_chunks):
            break
    return selected


def _normalize_impact(raw_impact: Any) -> str:
    if isinstance(raw_impact, str):
        impact = raw_impact.upper()
        if impact in VALID_IMPACTS:
            return impact
    return "MEDIUM"


def _match_location_from_evidence(
    evidence: str,
    chunk_span_texts: dict[str, str],
) -> str:
    if not evidence or not chunk_span_texts:
        return ""

    normalized_evidence = _normalize_match_text(evidence)
    evidence_tokens = set(normalized_evidence.split())
    best_span_id = ""
    best_score = 0

    for span_id, text in chunk_span_texts.items():
        normalized_text = _normalize_match_text(text)
        if not normalized_text:
            continue

        if normalized_evidence and normalized_evidence in normalized_text:
            score = len(normalized_evidence)
        elif normalized_text in normalized_evidence:
            score = len(normalized_text)
        else:
            score = len(evidence_tokens.intersection(set(normalized_text.split())))

        if score > best_score:
            best_score = score
            best_span_id = span_id

    return best_span_id if best_score > 0 else ""


def _location_keyword_hits(category: str, text: str) -> int:
    category_keywords: dict[str, tuple[str, ...]] = {
        "Timeout Issues": ("timeout", "timed out", "deadline exceeded"),
        "Rate Limiting": ("rate limit", "429", "too many requests"),
        "Authentication Errors": ("401", "403", "unauthorized", "forbidden", "auth"),
        "Resource Not Found": ("404", "not found", "no such"),
        "Resource Exhaustion": ("resource exhausted", "out of memory", "memory", "disk full"),
        "Service Errors": ("500", "502", "503", "service unavailable", "internal server error"),
        "Resource Abuse": ("repeated", "redundant", "retry", "loop", "excessive"),
    }
    hits = 0
    for keyword in category_keywords.get(category, ()):
        if keyword in text:
            hits += 1
    return hits


def _span_has_error_signal(span: dict[str, Any]) -> bool:
    status_code = str(span.get("status_code", "")).lower()
    status_message = str(span.get("status_message", "")).strip()
    return status_code == "error" or (status_message != "" and status_code != "ok")


def _score_span_for_location_choice(
    category: str,
    evidence: str,
    span_text: str,
    is_error: bool,
    span_name: str = "",
) -> int:
    normalized_evidence = _normalize_match_text(evidence)
    normalized_span_text = _normalize_match_text(span_text)

    evidence_tokens = set(normalized_evidence.split())
    span_tokens = set(normalized_span_text.split())
    keyword_hits = _location_keyword_hits(category, normalized_span_text)

    score = 0
    if normalized_evidence and normalized_evidence in normalized_span_text:
        score += min(120, len(normalized_evidence))
    score += len(evidence_tokens.intersection(span_tokens))
    score += 3 * keyword_hits

    if is_error:
        if category in {
            "Timeout Issues",
            "Rate Limiting",
            "Authentication Errors",
            "Resource Not Found",
            "Resource Exhaustion",
            "Service Errors",
            "Resource Abuse",
        }:
            score += 4
        else:
            score += 1

    if _is_step_span_name(span_name) and category not in _STEP_ALLOWED_CATEGORIES:
        score -= 5 if keyword_hits == 0 else 2

    if not _candidate_matches_category_hint(category, span_text, span_name):
        score -= 3

    score += _span_name_location_bonus(category=category, evidence=evidence, span_name=span_name)
    return score


def _refine_agentic_locations(
    trace_payload: dict[str, Any],
    findings: list[dict[str, str]],
) -> list[dict[str, str]]:
    spans = trace_payload.get("spans", [])
    flat_spans = _walk_spans(spans if isinstance(spans, list) else [])
    candidates: list[dict[str, Any]] = []
    for index, span in enumerate(flat_spans):
        span_id = span.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            continue
        candidates.append(
                {
                    "index": index,
                    "span_id": span_id,
                    "span_name": str(span.get("span_name", "")),
                    "text": _span_text(span),
                    "is_error": _span_has_error_signal(span),
                }
            )

    if not candidates:
        return findings

    by_id = {candidate["span_id"]: candidate for candidate in candidates}

    refined: list[dict[str, str]] = []
    for error in findings:
        category = error.get("category", "")
        evidence = error.get("evidence", "")
        if not isinstance(category, str) or not isinstance(evidence, str):
            refined.append(error)
            continue
        description = error.get("description", "")
        if not isinstance(description, str):
            description = ""

        current_location = error.get("location", "")
        current_candidate = by_id.get(current_location) if isinstance(current_location, str) else None

        current_score = (
            _score_span_for_location_choice(
                category=category,
                evidence=evidence,
                span_text=current_candidate["text"],
                is_error=bool(current_candidate["is_error"]),
                span_name=str(current_candidate["span_name"]),
            )
            if current_candidate is not None
            else -1
        )

        scored: list[tuple[int, int, str, dict[str, Any]]] = []
        for candidate in candidates:
            score = _score_span_for_location_choice(
                category=category,
                evidence=evidence,
                span_text=candidate["text"],
                is_error=bool(candidate["is_error"]),
                span_name=str(candidate["span_name"]),
            )
            scored.append((score, int(candidate["index"]), str(candidate["span_id"]), candidate))

        if not scored:
            refined.append(error)
            continue

        best_score = max(score for score, _index, _span_id, _candidate in scored)
        if best_score <= 0:
            refined.append(error)
            continue

        tied = [
            (index, span_id, candidate)
            for score, index, span_id, candidate in scored
            if score == best_score
        ]
        if category == "Resource Abuse":
            selected_index, selected_span_id, selected_candidate = max(tied, key=lambda item: item[0])
        else:
            selected_index, selected_span_id, selected_candidate = min(tied, key=lambda item: item[0])
        del selected_index
        del selected_candidate

        should_move = False
        if category == "Resource Abuse":
            should_move = selected_span_id != current_location and best_score > 0
        else:
            should_move = selected_span_id != current_location and best_score >= current_score + 1

        if should_move:
            updated = dict(error)
            updated["location"] = selected_span_id
            refined.append(updated)
        else:
            refined.append(error)

    return refined


def _apply_trajectory_action_correlation(
    trace_payload: dict[str, Any],
    findings: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    spans = trace_payload.get("spans", [])
    if not isinstance(spans, list):
        return findings, 0

    flat_spans: list[dict[str, Any]] = []
    parent_by_id: dict[str, str] = {}
    children_by_id: dict[str, list[str]] = {}
    _walk_spans_with_parent(
        spans,
        parent_id="",
        flat=flat_spans,
        parent_by_id=parent_by_id,
        children_by_id=children_by_id,
    )
    if not flat_spans:
        return findings, 0

    ordered_span_ids: list[str] = []
    span_context_by_id: dict[str, dict[str, Any]] = {}
    for index, span in enumerate(flat_spans):
        span_id = span.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            continue
        span_text = _span_text(span)
        if len(span_text) > 1600:
            span_text = span_text[:1600]
        ordered_span_ids.append(span_id)
        span_context_by_id[span_id] = {
            "index": index,
            "span_name": str(span.get("span_name", "")),
            "text": span_text,
            "is_error": _span_has_error_signal(span),
        }

    if not span_context_by_id:
        return findings, 0

    index_by_id = {span_id: index for index, span_id in enumerate(ordered_span_ids)}
    relocated = 0
    correlated: list[dict[str, str]] = []

    for error in findings:
        category = error.get("category")
        location = error.get("location")
        if not isinstance(category, str) or not isinstance(location, str):
            correlated.append(error)
            continue

        current_context = span_context_by_id.get(location)
        if current_context is None:
            correlated.append(error)
            continue

        evidence = error.get("evidence", "")
        description = error.get("description", "")
        if not isinstance(evidence, str):
            evidence = ""
        if not isinstance(description, str):
            description = ""
        signal_text = _normalize_match_text(f"{evidence} {description}")
        signal_tokens = _action_tokens(signal_text)

        candidate_ids = _collect_trajectory_candidate_ids(
            location,
            ordered_span_ids=ordered_span_ids,
            parent_by_id=parent_by_id,
            children_by_id=children_by_id,
            index_by_id=index_by_id,
        )
        if not candidate_ids:
            correlated.append(error)
            continue

        current_base_score = _score_span_for_location_choice(
            category=category,
            evidence=evidence,
            span_text=str(current_context["text"]),
            is_error=bool(current_context["is_error"]),
            span_name=str(current_context["span_name"]),
        )
        current_alignment_score = _action_alignment_score(
            signal_text=signal_text,
            signal_tokens=signal_tokens,
            span_name=str(current_context["span_name"]),
            span_text=str(current_context["text"]),
        )
        current_score = (
            current_base_score
            + current_alignment_score
            + _trajectory_distance_bonus(
                location,
                location,
                parent_by_id=parent_by_id,
                index_by_id=index_by_id,
            )
        )

        best_location = location
        best_score = current_score
        best_alignment_score = current_alignment_score
        best_index = index_by_id.get(location, 10**9)

        for candidate_id in candidate_ids:
            candidate_context = span_context_by_id.get(candidate_id)
            if candidate_context is None:
                continue

            candidate_base_score = _score_span_for_location_choice(
                category=category,
                evidence=evidence,
                span_text=str(candidate_context["text"]),
                is_error=bool(candidate_context["is_error"]),
                span_name=str(candidate_context["span_name"]),
            )
            if candidate_base_score <= 0 and candidate_id != location:
                continue

            candidate_alignment_score = _action_alignment_score(
                signal_text=signal_text,
                signal_tokens=signal_tokens,
                span_name=str(candidate_context["span_name"]),
                span_text=str(candidate_context["text"]),
            )
            candidate_score = (
                candidate_base_score
                + candidate_alignment_score
                + _trajectory_distance_bonus(
                    location,
                    candidate_id,
                    parent_by_id=parent_by_id,
                    index_by_id=index_by_id,
                )
            )
            candidate_index = index_by_id.get(candidate_id, 10**9)

            if candidate_score > best_score:
                best_score = candidate_score
                best_location = candidate_id
                best_alignment_score = candidate_alignment_score
                best_index = candidate_index
                continue

            if candidate_score == best_score and candidate_id != best_location:
                if category == "Resource Abuse" and candidate_index > best_index:
                    best_location = candidate_id
                    best_alignment_score = candidate_alignment_score
                    best_index = candidate_index
                elif category != "Resource Abuse" and candidate_index < best_index:
                    best_location = candidate_id
                    best_alignment_score = candidate_alignment_score
                    best_index = candidate_index

        if (
            best_location != location
            and best_score >= current_score + 2
            and best_alignment_score > 0
        ):
            updated = dict(error)
            updated["location"] = best_location
            correlated.append(updated)
            relocated += 1
            continue

        correlated.append(error)

    return _merge_errors(correlated), relocated


def _parse_chunk_errors(
    raw_response: Any,
    chunk_span_ids: list[str],
    chunk_span_texts: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    payload = _json_object_from_text(raw_response)
    raw_errors = payload.get("errors", [])
    if not isinstance(raw_errors, list):
        return []

    fallback_location = chunk_span_ids[0] if chunk_span_ids else ""
    valid_locations = set(chunk_span_ids)
    cleaned: list[dict[str, str]] = []

    for item in raw_errors:
        if not isinstance(item, dict):
            continue

        category = item.get("category")
        if not isinstance(category, str):
            continue
        category = category.strip()
        if category not in TRAIL_LEAF_CATEGORIES:
            continue

        raw_evidence = item.get("evidence")
        evidence = " ".join(raw_evidence.split()) if isinstance(raw_evidence, str) else ""
        if not evidence:
            evidence = "No direct evidence text was provided in the chunk output."

        raw_location = item.get("location")
        location = raw_location if isinstance(raw_location, str) else ""
        if location not in valid_locations:
            matched_location = _match_location_from_evidence(evidence, chunk_span_texts or {})
            location = matched_location if matched_location in valid_locations else fallback_location
        if not location:
            continue

        raw_description = item.get("description")
        description = (
            " ".join(raw_description.split())
            if isinstance(raw_description, str) and raw_description.strip()
            else f"Detected {category.lower()} from delegated chunk analysis."
        )

        cleaned.append(
            {
                "category": category,
                "location": location,
                "evidence": evidence[:300],
                "description": description,
                "impact": _normalize_impact(item.get("impact")),
            }
        )
    return cleaned


def _heuristic_chunk_findings(
    chunk_span_ids: list[str],
    chunk_span_texts: dict[str, str],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    seen_categories: set[str] = set()
    fallback_location = chunk_span_ids[0] if chunk_span_ids else ""

    for span_id in chunk_span_ids:
        text = chunk_span_texts.get(span_id, "")
        location = span_id if span_id else fallback_location
        for pattern, category, impact in ERROR_RULES:
            if category in seen_categories:
                continue
            if not pattern.search(text):
                continue
            evidence = " ".join(text.split())[:300]
            findings.append(
                {
                    "category": category,
                    "location": location,
                    "evidence": evidence if evidence else "No direct evidence text was provided.",
                    "description": f"Detected {category.lower()} from delegated chunk fallback analysis.",
                    "impact": impact,
                }
            )
            seen_categories.add(category)
    return findings


def _merge_errors(chunk_errors: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for error in chunk_errors:
        key = (error["location"], error["category"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(error)

    merged.sort(key=lambda item: (item["location"], item["category"]))
    return merged


def _summarize_trace(trace_payload: dict[str, Any], span_records: list[dict[str, Any]]) -> dict[str, Any]:
    trace_id = str(trace_payload.get("trace_id", ""))
    top_signals: list[str] = []
    for record in span_records[:6]:
        text = str(record["text"])
        if len(text) > 280:
            text = text[:280] + "..."
        top_signals.append(text)
    return {
        "trace_id": trace_id,
        "num_spans": len(span_records),
        "top_span_ids": [record["span_id"] for record in span_records[:10]],
        "top_signals": top_signals,
    }


def _compose_root_task(
    trace_summary: dict[str, Any],
    chunk_catalog: list[dict[str, Any]],
    taxonomy: list[str],
    max_chunks: int,
) -> str:
    return (
        f"{TRAIL_ROOT_PLAN_PROMPT}\n\n"
        "# Trace Summary (JSON)\n"
        f"{json.dumps(trace_summary, ensure_ascii=False)}\n\n"
        "# Chunk Catalog (JSON)\n"
        f"{json.dumps(chunk_catalog, ensure_ascii=False)}\n\n"
        "# Taxonomy (JSON)\n"
        f"{json.dumps(taxonomy, ensure_ascii=False)}\n\n"
        "# Max Chunks\n"
        f"{max_chunks}\n"
    )


def _compose_chunk_task(
    trace_id: str,
    chunk_id: int,
    chunk_span_ids: list[str],
    chunk_payload: str,
    taxonomy: list[str],
) -> str:
    return (
        f"{TRAIL_CHUNK_ANALYSIS_PROMPT}\n\n"
        "# Trace ID\n"
        f"{trace_id}\n\n"
        "# Chunk ID\n"
        f"{chunk_id}\n\n"
        "# Chunk Span IDs (JSON)\n"
        f"{json.dumps(chunk_span_ids, ensure_ascii=False)}\n\n"
        "# Chunk Payload\n"
        f"{chunk_payload}\n\n"
        "# Taxonomy (JSON)\n"
        f"{json.dumps(taxonomy, ensure_ascii=False)}\n"
    )


def _is_timeout_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text


async def _call_agent_with_timeout_retry(
    runtime: _TrailAgentRuntime,
    task: str,
    return_type: type[Any],
    *,
    timeout_retries: int = 1,
    retry_backoff_seconds: float = 0.2,
    per_call_timeout_seconds: float = 180.0,
    **objects: Any,
) -> Any:
    attempts = 0
    while True:
        try:
            return await asyncio.wait_for(
                runtime.call_agent(task, return_type, **objects),
                timeout=per_call_timeout_seconds,
            )
        except Exception as exc:
            if attempts >= timeout_retries or not _is_timeout_error(exc):
                raise
            attempts += 1
            await asyncio.sleep(retry_backoff_seconds * attempts)


async def _analyze_trace_agentic(
    trace_payload: dict[str, Any],
    root_model: str,
    chunk_model: str,
    max_num_agents: int,
    max_chunks: int,
    max_spans_per_chunk: int,
    max_span_text_chars: int,
    agent_call_timeout_seconds: float = 60.0,
    agent_timeout_retries: int = 1,
    joint_recall_boost: bool = False,
) -> dict[str, Any]:
    trace_id = str(trace_payload.get("trace_id", ""))
    span_records = _build_span_records(trace_payload, max_span_text_chars=max_span_text_chars)
    if not span_records:
        return {
            "trace_id": trace_id,
            "errors": [],
            "scores": [_score_block([])],
        }

    chunks = _build_chunks(
        span_records,
        max_chunks=max_chunks,
        max_spans_per_chunk=max_spans_per_chunk,
    )
    if not chunks:
        return {
            "trace_id": trace_id,
            "errors": [],
            "scores": [_score_block([])],
        }

    timeout_retries = max(0, int(agent_timeout_retries))
    per_call_timeout_seconds = max(5.0, float(agent_call_timeout_seconds))
    call_attempts_per_agent_call = timeout_retries + 1

    runtime_agent_limit = max(2, int(max_num_agents))
    chunk_agent_limit = max(1, runtime_agent_limit - 1)
    delegated_chunk_budget = min(max(1, int(max_chunks)), chunk_agent_limit)
    root_runtime = _make_agentic_runtime(
        model=root_model,
        max_num_agents=call_attempts_per_agent_call,
        trace_id=trace_id,
        allow_delegation=False,
    )
    chunk_runtime = _make_agentic_runtime(
        model=chunk_model,
        max_num_agents=chunk_agent_limit * call_attempts_per_agent_call,
        trace_id=trace_id,
        allow_delegation=True,
    )

    selected_ids = []
    root_planner_fallback = ""
    available_ids = {chunk["chunk_id"] for chunk in chunks}
    if len(chunks) <= delegated_chunk_budget:
        selected_ids = [chunk["chunk_id"] for chunk in chunks]
        root_planner_fallback = "skipped_all_chunks_fit_budget"
    else:
        try:
            plan_response = await _call_agent_with_timeout_retry(
                root_runtime,
                _compose_root_task(
                    trace_summary=_summarize_trace(trace_payload, span_records),
                    chunk_catalog=_chunk_catalog(chunks),
                    taxonomy=TRAIL_LEAF_CATEGORIES,
                    max_chunks=delegated_chunk_budget,
                ),
            str,
            timeout_retries=timeout_retries,
            per_call_timeout_seconds=per_call_timeout_seconds,
        )
            selected_ids = _select_chunk_ids(
                plan_response,
                available_ids=available_ids,
                max_chunks=delegated_chunk_budget,
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                root_planner_fallback = "timeout"
            else:
                raise
    if not selected_ids:
        selected_ids = [chunk["chunk_id"] for chunk in chunks[:delegated_chunk_budget]]

    chunk_index = {chunk["chunk_id"]: chunk for chunk in chunks}
    chunk_timeout_recovery_ids: list[int] = []

    async def _inspect_chunk(chunk_id: int) -> list[dict[str, str]]:
        chunk = chunk_index[chunk_id]
        try:
            response = await _call_agent_with_timeout_retry(
                chunk_runtime,
                _compose_chunk_task(
                    trace_id=trace_id,
                    chunk_id=chunk_id,
                    chunk_span_ids=chunk["span_ids"],
                    chunk_payload=chunk["payload"],
                    taxonomy=TRAIL_LEAF_CATEGORIES,
                ),
                str,
                timeout_retries=timeout_retries,
                per_call_timeout_seconds=per_call_timeout_seconds,
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                chunk_timeout_recovery_ids.append(chunk_id)
                return _heuristic_chunk_findings(
                    chunk_span_ids=chunk["span_ids"],
                    chunk_span_texts=chunk["span_texts"],
                )
            raise
        return _parse_chunk_errors(
            response,
            chunk_span_ids=chunk["span_ids"],
            chunk_span_texts=chunk["span_texts"],
        )

    delegated_results = await asyncio.gather(
        *[_inspect_chunk(chunk_id) for chunk_id in selected_ids],
        return_exceptions=True,
    )

    findings: list[dict[str, str]] = []
    delegation_failures: list[dict[str, Any]] = []
    for chunk_id, delegated in zip(selected_ids, delegated_results, strict=False):
        if isinstance(delegated, Exception):
            delegation_failures.append(
                {
                    "chunk_id": chunk_id,
                    "error_type": type(delegated).__name__,
                    "error_message": str(delegated)[:300],
                }
            )
            continue
        findings.extend(delegated)

    findings = _merge_errors(findings)
    findings, trajectory_action_correlation_moves = _apply_trajectory_action_correlation(
        trace_payload,
        findings,
    )
    findings = _refine_agentic_locations(trace_payload, findings)
    findings = _apply_post_filters(findings)
    findings = _recover_targeted_tp(trace_payload, findings)
    if joint_recall_boost:
        findings = _boost_joint_recall(findings)
    analysis_diagnostics: dict[str, Any] = {
        "analysis_mode": "agentic_repl",
        "selected_chunk_ids": selected_ids,
        "delegated_chunk_budget": delegated_chunk_budget,
        "root_model": root_model,
        "chunk_model": chunk_model,
        "agent_call_timeout_seconds": per_call_timeout_seconds,
        "agent_timeout_retries": timeout_retries,
        "root_planner_fallback": root_planner_fallback,
        "chunk_timeout_recoveries": len(chunk_timeout_recovery_ids),
        "chunk_timeout_recovery_ids": chunk_timeout_recovery_ids,
        "trajectory_action_correlation_moves": trajectory_action_correlation_moves,
        "delegation_failures": len(delegation_failures),
        "delegation_failed_chunk_ids": [item["chunk_id"] for item in delegation_failures],
    }
    if delegation_failures:
        analysis_diagnostics["delegation_failure_details"] = delegation_failures
    return {
        "trace_id": trace_id,
        "errors": findings,
        "scores": [_score_block(findings)],
        "analysis_diagnostics": analysis_diagnostics,
    }


_REASONING_MODEL_PATTERNS = (
    "o1", "o3", "o4", "anthropic", "gemini-2.5",
)


def _is_reasoning_model(model: str) -> bool:
    lowered = model.lower()
    return any(pat in lowered for pat in _REASONING_MODEL_PATTERNS)


def _is_openai_gpt5_model(model: str) -> bool:
    return model.lower().startswith("openai/gpt-5")


def _is_context_window_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "contextwindowexceedederror" in text
        or "input tokens exceed" in text
        or "maximum context length" in text
        or "too many tokens" in text
        or ("context length" in text and "requested" in text and "tokens" in text)
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "ratelimiterror" in text or "rate limit" in text


def _rate_limit_wait_seconds(exc: Exception, default_seconds: float = 5.0) -> float:
    text = f"{exc}"
    match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return default_seconds
    return default_seconds


def _estimate_prompt_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text + JSON."""
    return len(text) // 4


# Approximate input token limits per model family
_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-5-mini": 128_000,
    "gpt-5.2": 256_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
}


def _model_input_token_limit(model: str) -> int:
    """Return approximate input token limit for model."""
    lowered = model.lower().replace("openai/", "")
    for key, limit in _MODEL_CONTEXT_LIMITS.items():
        if key in lowered:
            return limit
    return 128_000  # Conservative default


def _single_pass_trace_char_budget(max_span_text_chars: int) -> int:
    span_chars = max(120, max_span_text_chars)
    return min(400_000, max(20_000, span_chars * 250))


def _analyze_trace_single_pass(
    trace_payload: dict[str, Any],
    model: str,
    prompt_version: str,
    max_span_text_chars: int,
) -> dict[str, Any]:
    trace_id = str(trace_payload.get("trace_id", ""))

    # Collect all valid span_ids for location validation
    spans = trace_payload.get("spans", [])
    flat_spans = _walk_spans(spans if isinstance(spans, list) else [])
    valid_span_ids: set[str] = set()
    fallback_span_id = ""
    for span in flat_spans:
        sid = span.get("span_id")
        if isinstance(sid, str) and sid:
            valid_span_ids.add(sid)
            if not fallback_span_id:
                fallback_span_id = sid

    # Build prompt message
    prompt_version_key = (prompt_version or "v2").strip().lower()
    trace_char_budget = _single_pass_trace_char_budget(max_span_text_chars)
    if _is_openai_gpt5_model(model):
        trace_char_budget = 800_000
    budget_candidates = [trace_char_budget]
    while budget_candidates[-1] > 20_000:
        next_budget = max(20_000, budget_candidates[-1] // 2)
        if next_budget == budget_candidates[-1]:
            break
        budget_candidates.append(next_budget)

    response = None
    last_error: Exception | None = None
    for budget in budget_candidates:
        if prompt_version_key == "v1":
            raw_trace = json.dumps(trace_payload, ensure_ascii=False)
            if len(raw_trace) > budget:
                raw_trace = raw_trace[:budget] + "... [truncated]"
            prompt_text = _TRAIL_SINGLE_PASS_PROMPT_V1.replace("{trace}", raw_trace)
        else:
            prompt_text = build_single_pass_message(trace_payload, max_chars=budget)

        # Token preflight: estimate tokens and skip this budget if over limit
        estimated_tokens = _estimate_prompt_tokens(prompt_text)
        model_limit = _model_input_token_limit(model)
        # Reserve 15% for completion tokens and estimation error
        if estimated_tokens > int(model_limit * 0.85):
            # Prompt too large for this budget, try next smaller one
            continue

        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_completion_tokens": 8000,
            "drop_params": True,
            "timeout": 120,
        }
        if _is_reasoning_model(model):
            completion_kwargs["reasoning_effort"] = "high"
        else:
            completion_kwargs["temperature"] = 0

        rate_limit_attempts = 0
        while True:
            try:
                response = completion(**completion_kwargs)
                break
            except Exception as exc:
                last_error = exc
                if _is_rate_limit_error(exc) and rate_limit_attempts < 2:
                    wait_seconds = _rate_limit_wait_seconds(exc)
                    time.sleep(max(0.1, min(35.0, wait_seconds)))
                    rate_limit_attempts += 1
                    continue
                if _is_context_window_error(exc):
                    break
                raise
        if response is not None:
            break

    if response is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Single-pass completion failed without a captured error.")

    # Extract text from response
    raw_content = response.choices[0].message
    if isinstance(raw_content, dict):
        raw_text = raw_content.get("content", "")
    else:
        raw_text = getattr(raw_content, "content", "")

    payload = _json_object_from_text(raw_text)

    # Parse and validate errors
    raw_errors = payload.get("errors", [])
    if not isinstance(raw_errors, list):
        raw_errors = []

    validated_errors: list[dict[str, str]] = []
    for item in raw_errors:
        if not isinstance(item, dict):
            continue

        category = item.get("category")
        if not isinstance(category, str) or category.strip() not in TRAIL_LEAF_CATEGORIES:
            continue
        category = category.strip()

        # Validate location against real span_ids
        raw_location = item.get("location")
        location = raw_location if isinstance(raw_location, str) else ""
        if location not in valid_span_ids:
            location = fallback_span_id
        if not location:
            continue

        raw_evidence = item.get("evidence")
        evidence = " ".join(raw_evidence.split()) if isinstance(raw_evidence, str) else ""
        if not evidence:
            evidence = "No direct evidence text was provided."

        raw_description = item.get("description")
        description = (
            " ".join(raw_description.split())
            if isinstance(raw_description, str) and raw_description.strip()
            else f"Detected {category.lower()} from single-pass analysis."
        )

        validated_errors.append({
            "category": category,
            "location": location,
            "evidence": evidence[:300],
            "description": description,
            "impact": _normalize_impact(item.get("impact")),
        })

    # Deduplicate
    validated_errors = _merge_errors(validated_errors)

    # Parse scores -- use LLM-generated if available, else fallback to heuristic
    raw_scores = payload.get("scores", [])
    if isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], dict):
        scores = [raw_scores[0]]
    else:
        scores = [_score_block(validated_errors)]

    return {
        "trace_id": trace_id,
        "errors": validated_errors,
        "scores": scores,
    }


def apply_trajectory_action_correlation_to_prediction(
    trace_payload: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    raw_errors = prediction.get("errors", [])
    if not isinstance(raw_errors, list):
        return prediction

    normalized_errors: list[dict[str, str]] = []
    for item in raw_errors:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        location = item.get("location")
        if not isinstance(category, str) or not category.strip():
            continue
        if not isinstance(location, str) or not location.strip():
            continue
        evidence = item.get("evidence")
        description = item.get("description")
        normalized_errors.append(
            {
                "category": category.strip(),
                "location": location.strip(),
                "evidence": evidence if isinstance(evidence, str) else "",
                "description": description if isinstance(description, str) else "",
                "impact": _normalize_impact(item.get("impact")),
            }
        )

    correlated_prediction = dict(prediction)
    correlated_errors, _relocated = _apply_trajectory_action_correlation(
        trace_payload,
        _merge_errors(normalized_errors),
    )
    correlated_prediction["errors"] = correlated_errors
    raw_scores = correlated_prediction.get("scores")
    if not isinstance(raw_scores, list) or not raw_scores:
        correlated_prediction["scores"] = [_score_block(correlated_errors)]
    return correlated_prediction


def apply_joint_recall_boost_to_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    raw_errors = prediction.get("errors", [])
    if not isinstance(raw_errors, list):
        return prediction

    normalized_errors: list[dict[str, str]] = []
    for item in raw_errors:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        location = item.get("location")
        if not isinstance(category, str) or not category.strip():
            continue
        if not isinstance(location, str) or not location.strip():
            continue
        evidence = item.get("evidence")
        description = item.get("description")
        normalized_errors.append(
            {
                "category": category.strip(),
                "location": location.strip(),
                "evidence": evidence if isinstance(evidence, str) else "",
                "description": description if isinstance(description, str) else "",
                "impact": _normalize_impact(item.get("impact")),
            }
        )

    boosted_prediction = dict(prediction)
    boosted_errors = _boost_joint_recall(normalized_errors)
    boosted_prediction["errors"] = boosted_errors
    raw_scores = boosted_prediction.get("scores")
    if not isinstance(raw_scores, list) or not raw_scores:
        boosted_prediction["scores"] = [_score_block(boosted_errors)]
    return boosted_prediction


def analyze_trace(
    trace_payload: dict[str, Any],
    model: str,
    *,
    agentic_mode: str = "off",
    prompt_version: str = "v2",
    max_num_agents: int = 6,
    max_chunks: int = 6,
    max_spans_per_chunk: int = 12,
    max_span_text_chars: int = 1200,
    root_model: str | None = None,
    chunk_model: str | None = None,
    agent_call_timeout_seconds: float = 60.0,
    agent_timeout_retries: int = 1,
    joint_recall_boost: bool = False,
) -> dict[str, Any]:
    if agentic_mode == "single_pass":
        try:
            return _analyze_trace_single_pass(
                trace_payload=trace_payload,
                model=model,
                prompt_version=prompt_version,
                max_span_text_chars=max_span_text_chars,
            )
        except Exception as exc:
            return _heuristic_fallback_result(trace_payload, fallback_from="single_pass", error=exc)

    if agentic_mode != "on":
        findings = _heuristic_findings(trace_payload)
        return {
            "trace_id": str(trace_payload.get("trace_id", "")),
            "errors": findings,
            "scores": [_score_block(findings)],
        }

    try:
        resolved_root_model = root_model or model
        resolved_chunk_model = chunk_model or model
        return asyncio.run(
            _analyze_trace_agentic(
                trace_payload=trace_payload,
                root_model=resolved_root_model,
                chunk_model=resolved_chunk_model,
                max_num_agents=max_num_agents,
                max_chunks=max_chunks,
                max_spans_per_chunk=max_spans_per_chunk,
                max_span_text_chars=max_span_text_chars,
                agent_call_timeout_seconds=agent_call_timeout_seconds,
                agent_timeout_retries=agent_timeout_retries,
                joint_recall_boost=joint_recall_boost,
            )
        )
    except Exception as exc:
        return _heuristic_fallback_result(trace_payload, fallback_from="agentic", error=exc)
