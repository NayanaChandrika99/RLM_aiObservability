# ABOUTME: Defines scaffold presets for RCA comparison experiments (heuristic, LLM, RLM, RLM+tips).
# ABOUTME: Each preset bundles engine flags so the CLI and comparison runner can select by name.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaffoldPreset:
    name: str
    display_name: str
    use_llm_judgment: bool
    use_repl_runtime: bool
    tips_profile: str
    fallback_on_llm_error: bool
    description: str


SCAFFOLD_PRESETS: dict[str, ScaffoldPreset] = {
    "heuristic": ScaffoldPreset(
        name="heuristic",
        display_name="Heuristic",
        use_llm_judgment=False,
        use_repl_runtime=False,
        tips_profile="none",
        fallback_on_llm_error=False,
        description="Deterministic hot-span + pattern match, zero LLM calls.",
    ),
    "llm_single_shot": ScaffoldPreset(
        name="llm_single_shot",
        display_name="LLM Single-Shot",
        use_llm_judgment=True,
        use_repl_runtime=False,
        tips_profile="none",
        fallback_on_llm_error=False,
        description="Single structured LLM call for RCA judgment.",
    ),
    "rlm": ScaffoldPreset(
        name="rlm",
        display_name="RLM",
        use_llm_judgment=True,
        use_repl_runtime=True,
        tips_profile="none",
        fallback_on_llm_error=False,
        description="REPL loop without tips.",
    ),
    "rlm_tips": ScaffoldPreset(
        name="rlm_tips",
        display_name="RLM + Tips",
        use_llm_judgment=True,
        use_repl_runtime=True,
        tips_profile="trace_rca_v1",
        fallback_on_llm_error=False,
        description="REPL loop with trace_rca_v1 tips profile.",
    ),
}


def list_scaffold_names() -> list[str]:
    return sorted(SCAFFOLD_PRESETS.keys())


def get_scaffold_preset(name: str) -> ScaffoldPreset:
    preset = SCAFFOLD_PRESETS.get(name)
    if preset is None:
        available = ", ".join(list_scaffold_names())
        raise ValueError(f"Unknown scaffold preset: {name!r}. Available: {available}")
    return preset
