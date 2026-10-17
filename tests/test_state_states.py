"""Phase 4, step 1: ``AgentState`` enum + ``AgentStateError`` (states.py).

The closed-set state vocabulary the state machine (step 2), policy (step 3),
and agent loop (step 5) all share. Mirrors the ``OperationMode`` discipline
from Phase 3: ``StrEnum`` string interoperability with config and JSON, a
closed value set, and an explicit error type for invalid values — never
implicit coercion.

Strict step-1 scope: the enum and its error type only. No state machine,
transitions, recovery counters, event logging, or agent loop (those land in
subsequent phase-4 steps).
"""

from __future__ import annotations

import pytest

from ai_game_agent.state.states import AgentState, AgentStateError

# --- The closed set: exactly six states --------------------------------------


def test_agent_state_has_exactly_six_states():
    assert {s.name for s in AgentState} == {
        "IDLE",
        "COMBAT",
        "LOOTING",
        "RECOVERY",
        "AI_ANALYSIS",
        "PAUSED",
    }


def test_agent_state_values_are_the_expected_lowercase_strings():
    assert {s.name: s.value for s in AgentState} == {
        "IDLE": "idle",
        "COMBAT": "combat",
        "LOOTING": "looting",
        "RECOVERY": "recovery",
        "AI_ANALYSIS": "ai_analysis",
        "PAUSED": "paused",
    }


# --- String interop: config / JSON round-trip --------------------------------


def test_agent_state_constructs_from_its_string_value():
    assert AgentState("combat") is AgentState.COMBAT
    assert AgentState("idle") is AgentState.IDLE


def test_agent_state_is_string_comparable():
    # StrEnum: the enum value == its string, so config strings, JSON
    # payloads, and the enum all interoperate without coercion.
    assert AgentState.COMBAT == "combat"
    assert str(AgentState.COMBAT) == "combat"


# --- Error taxonomy: invalid values never coerce ------------------------------


def test_invalid_state_raises_agent_state_error():
    with pytest.raises(AgentStateError) as excinfo:
        AgentState("bogus")
    assert "bogus" in str(excinfo.value)  # message names the bad value


def test_agent_state_error_is_a_value_error():
    assert issubclass(AgentStateError, ValueError)
    with pytest.raises(ValueError):
        AgentState("bogus")  # catchable as ValueError too (typo: no coercion)


# --- Hashable / set-usable ----------------------------------------------------


def test_states_are_hashable_and_usable_in_frozenset():
    all_states = frozenset(AgentState)
    assert all_states == frozenset({
        AgentState.IDLE,
        AgentState.COMBAT,
        AgentState.LOOTING,
        AgentState.RECOVERY,
        AgentState.AI_ANALYSIS,
        AgentState.PAUSED,
    })
    assert AgentState.COMBAT in all_states
    assert len(frozenset({AgentState.IDLE, AgentState.IDLE})) == 1


# --- Package re-export (intended import path) ---------------------------------


def test_state_package_reexports_the_enum():
    from ai_game_agent.state import AgentState as PkgState
    from ai_game_agent.state import AgentStateError as PkgError

    assert PkgState is AgentState
    assert PkgError is AgentStateError
