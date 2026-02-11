# ABOUTME: Validates Phase 8B runtime state-machine transitions for recursive execution.
# ABOUTME: Ensures invalid transitions are rejected before runtime artifact persistence.

from __future__ import annotations

import pytest

from investigator.runtime.recursive_loop import RuntimeStateMachine, StateTransitionError


def test_runtime_state_machine_allows_budget_termination_path() -> None:
    machine = RuntimeStateMachine()

    assert machine.state == "initialized"
    machine.transition("running")
    machine.transition("terminated_budget")
    machine.transition("partial")
    assert machine.state == "partial"


def test_runtime_state_machine_rejects_invalid_transition() -> None:
    machine = RuntimeStateMachine()

    with pytest.raises(StateTransitionError):
        machine.transition("partial")


def test_runtime_state_machine_allows_terminated_budget_to_failed() -> None:
    machine = RuntimeStateMachine()
    machine.transition("running")
    machine.transition("terminated_budget")
    machine.transition("failed")

    assert machine.state == "failed"
