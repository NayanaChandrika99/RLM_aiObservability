# ABOUTME: Validates scaffold presets are correctly defined with expected flags and names.
# ABOUTME: Ensures invalid preset names raise ValueError and the preset count matches expectations.

from __future__ import annotations

import pytest

from investigator.rca.scaffolds import (
    SCAFFOLD_PRESETS,
    get_scaffold_preset,
    list_scaffold_names,
)


def test_scaffold_preset_count_is_four() -> None:
    assert len(SCAFFOLD_PRESETS) == 4


def test_list_scaffold_names_returns_sorted() -> None:
    names = list_scaffold_names()
    assert names == sorted(names)
    assert set(names) == {"heuristic", "llm_single_shot", "rlm", "rlm_tips"}


def test_heuristic_preset_disables_llm_and_repl() -> None:
    preset = get_scaffold_preset("heuristic")
    assert preset.use_llm_judgment is False
    assert preset.use_repl_runtime is False
    assert preset.tips_profile == "none"


def test_llm_single_shot_preset_enables_llm_disables_repl() -> None:
    preset = get_scaffold_preset("llm_single_shot")
    assert preset.use_llm_judgment is True
    assert preset.use_repl_runtime is False
    assert preset.tips_profile == "none"


def test_rlm_preset_enables_llm_and_repl_no_tips() -> None:
    preset = get_scaffold_preset("rlm")
    assert preset.use_llm_judgment is True
    assert preset.use_repl_runtime is True
    assert preset.tips_profile == "none"


def test_rlm_tips_preset_enables_llm_repl_and_tips() -> None:
    preset = get_scaffold_preset("rlm_tips")
    assert preset.use_llm_judgment is True
    assert preset.use_repl_runtime is True
    assert preset.tips_profile == "trace_rca_v1"


def test_get_scaffold_preset_invalid_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown scaffold preset"):
        get_scaffold_preset("nonexistent")


def test_all_presets_have_name_matching_key() -> None:
    for key, preset in SCAFFOLD_PRESETS.items():
        assert preset.name == key


def test_all_presets_have_nonempty_description() -> None:
    for preset in SCAFFOLD_PRESETS.values():
        assert preset.description.strip()


def test_all_presets_have_nonempty_display_name() -> None:
    for preset in SCAFFOLD_PRESETS.values():
        assert preset.display_name.strip()
