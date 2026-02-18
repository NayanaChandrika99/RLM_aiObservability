# ABOUTME: Defines shared configuration and path helpers for ARCgentica TRAIL benchmark runs.
# ABOUTME: Keeps split iteration and output directory conventions deterministic and scorer-compatible.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TRAIL_LEAF_CATEGORIES = [
    "Language-only",
    "Tool-related",
    "Poor Information Retrieval",
    "Incorrect Memory Usage",
    "Tool Output Misinterpretation",
    "Incorrect Problem Identification",
    "Tool Selection Errors",
    "Formatting Errors",
    "Instruction Non-compliance",
    "Tool Definition Issues",
    "Environment Setup Errors",
    "Rate Limiting",
    "Authentication Errors",
    "Service Errors",
    "Resource Not Found",
    "Resource Exhaustion",
    "Timeout Issues",
    "Context Handling Failures",
    "Resource Abuse",
    "Goal Deviation",
    "Task Orchestration",
]


@dataclass(frozen=True)
class TrailRunConfig:
    trail_data_dir: Path
    split: str
    model: str
    output_dir: Path
    semantic_checks: str
    agentic_mode: str = "on"
    max_num_agents: int = 6
    max_chunks: int = 6
    max_spans_per_chunk: int = 12
    max_span_text_chars: int = 1200


def model_slug(model: str) -> str:
    return model.replace("/", "-")


def split_data_dir(config: TrailRunConfig) -> Path:
    return config.trail_data_dir / config.split


def run_output_dir(config: TrailRunConfig) -> Path:
    return config.output_dir / f"outputs_{model_slug(config.model)}-{config.split}"


def iter_trace_files(config: TrailRunConfig) -> list[Path]:
    split_dir = split_data_dir(config)
    if not split_dir.exists():
        raise FileNotFoundError(f"TRAIL split directory not found: {split_dir}")
    return sorted(split_dir.glob("*.json"))
