"""Phase 4, step 2: ``GameStateMachine.decide()`` + transitions (machine.py).

Red tests for the deterministic decision core. The machine is driven by a
*stub* policy whose ``transition_signals(obs)`` returns a fixed dict of the
stable signal keys — the machine never sees zone names, so the test matrix
stays small and game-agnostic. A fake clock (``now``) is used so no real time
elapses in the watchdog tests.
"""

from __future__ import annotations

import pytest

from ai_game_agent.state.machine import (
    SIGNAL_HEALTH_LOW,
    SIGNAL_HEALTH_RECOVERED,
    SIGNAL_LOOT_PRESENT,
    SIGNAL_TARGET_HOSTILE,
    SIGNAL_TARGET_PRESENT,
    GameStateMachine,
)
from ai_game_agent.state.states import AgentState

OBS = object()  # sentinel observation; the stub policy ignores it


class StubPolicy:
    """Fixed-signal policy: the machine's only signal source (the seam)."""

    def __init__(self) -> None:
        self.signals: dict[str, object] = {}
        self.calls = 0

    def transition_signals(self, observation: object) -> dict[str, object]:
        self.calls += 1
        return dict(self.signals)


def make(signals, *, initial=AgentState.IDLE, on_transition=None):
    pol = StubPolicy()
    pol.signals = signals
    m = GameStateMachine(policy=pol, initial=initial, on_transition=on_transition)
    return m, pol


# --- §4.3 transition table (via the stable signal seam) ----------------------


def test_idle_no_actionable_signal_stays_idle():
    m, _ = make({})
    assert m.decide(OBS, 1000) is AgentState.IDLE


def test_idle_low_health_to_recovery():
    m, _ = make({SIGNAL_HEALTH_LOW: True})
    assert m.decide(OBS, 1000) is AgentState.RECOVERY


def test_idle_loot_marker_to_looting():
    m, _ = make({SIGNAL_LOOT_PRESENT: True})
    assert m.decide(OBS, 1000) is AgentState.LOOTING


def test_idle_hostile_target_to_combat():
    m, _ = make({SIGNAL_TARGET_PRESENT: True, SIGNAL_TARGET_HOSTILE: True})
    assert m.decide(OBS, 1000) is AgentState.COMBAT


def test_idle_priority_health_beats_loot_beats_hostile():
    m, _ = make({
        SIGNAL_HEALTH_LOW: True,
        SIGNAL_LOOT_PRESENT: True,
        SIGNAL_TARGET_PRESENT: True,
        SIGNAL_TARGET_HOSTILE: True,
    })
    assert m.decide(OBS, 1000) is AgentState.RECOVERY  # health floor is safety


def test_idle_priority_loot_beats_hostile():
    m, _ = make({SIGNAL_LOOT_PRESENT: True, SIGNAL_TARGET_PRESENT: True, SIGNAL_TARGET_HOSTILE: True})
    assert m.decide(OBS, 1000) is AgentState.LOOTING  # table order is the contract


def test_combat_target_lost_to_idle():
    m, _ = make({SIGNAL_TARGET_PRESENT: False}, initial=AgentState.COMBAT)
    assert m.decide(OBS, 1000) is AgentState.IDLE


def test_combat_low_health_to_recovery():
    m, _ = make({SIGNAL_HEALTH_LOW: True}, initial=AgentState.COMBAT)
    assert m.decide(OBS, 1000) is AgentState.RECOVERY


def test_combat_order_target_lost_before_health():
    m, _ = make({SIGNAL_TARGET_PRESENT: False, SIGNAL_HEALTH_LOW: True},
                initial=AgentState.COMBAT)
    assert m.decide(OBS, 1000) is AgentState.IDLE  # table order is the contract


def test_looting_cleared_to_idle():
    m, _ = make({SIGNAL_LOOT_PRESENT: False}, initial=AgentState.LOOTING)
    assert m.decide(OBS, 1000) is AgentState.IDLE


def test_looting_still_present_self_loop():
    m, _ = make({SIGNAL_LOOT_PRESENT: True}, initial=AgentState.LOOTING)
    assert m.decide(OBS, 1000) is AgentState.LOOTING


def test_recovery_health_recovered_to_idle():
    m, _ = make({SIGNAL_HEALTH_RECOVERED: True}, initial=AgentState.RECOVERY)
    assert m.decide(OBS, 1.0) is AgentState.IDLE  # now=1.0 < 5.0 timeout: no watchdog


def test_paused_is_terminal():
    m, _ = make({SIGNAL_HEALTH_RECOVERED: True, SIGNAL_LOOT_PRESENT: True},
                initial=AgentState.PAUSED)
    assert m.decide(OBS, 1000) is AgentState.PAUSED


def test_initial_ai_analysis_anchors_watchdog_clock_before_timing_out():
    # Construction has no clock; the first decision anchors the initial
    # watchdog state's entry time instead of comparing process uptime to zero.
    m, _ = make({}, initial=AgentState.AI_ANALYSIS)
    assert m.decide(OBS, 1000) is AgentState.AI_ANALYSIS
    assert m.state_entered_at == 1000
    assert m.decide(OBS, 1004.9) is AgentState.AI_ANALYSIS
    assert m.decide(OBS, 1005) is AgentState.PAUSED


# --- Self-loop idempotence ---------------------------------------------------


def test_self_loop_does_not_fire_on_transition():
    events: list = []
    m, _ = make({SIGNAL_TARGET_HOSTILE: True}, initial=AgentState.COMBAT,
                on_transition=lambda b, a, r, t: events.append((b, a, r, t)))
    assert m.decide(OBS, 1000) is AgentState.COMBAT
    assert events == []


def test_self_loop_does_not_reset_recovery_counter():
    pol = StubPolicy()
    m = GameStateMachine(policy=pol)
    pol.signals = {SIGNAL_HEALTH_LOW: True}
    m.decide(OBS, 1000)  # -> RECOVERY (entry), count 0
    m.note_recovery_attempt()  # 1
    pol.signals = {}  # no recovery signal -> self-loop RECOVERY
    assert m.decide(OBS, 1000) is AgentState.RECOVERY
    assert m.recovery_count == 1  # NOT reset by the self-loop


# --- Bounded recovery --------------------------------------------------------


def test_note_recovery_attempt_counts_1_2_3():
    m, _ = make({})
    assert m.note_recovery_attempt() == 1
    assert m.note_recovery_attempt() == 2
    assert m.note_recovery_attempt() == 3


def test_recovery_exhaustion_paused_regardless_of_observation():
    pol = StubPolicy()
    m = GameStateMachine(policy=pol)
    pol.signals = {SIGNAL_HEALTH_LOW: True}
    assert m.decide(OBS, 1000) is AgentState.RECOVERY  # entry, count 0
    assert m.note_recovery_attempt() == 1
    assert m.decide(OBS, 1000) is AgentState.RECOVERY  # self-loop, count 1
    assert m.note_recovery_attempt() == 2
    assert m.decide(OBS, 1000) is AgentState.RECOVERY  # self-loop, count 2
    assert m.note_recovery_attempt() == 3
    # Exhausted: the next decide() is PAUSED even with a "healthy" signal.
    pol.signals = {SIGNAL_HEALTH_RECOVERED: True}
    assert m.decide(OBS, 1000) is AgentState.PAUSED
    assert m.state is AgentState.PAUSED


def test_reset_recovery_zeroes_counter():
    m, _ = make({})
    m.note_recovery_attempt()
    m.note_recovery_attempt()
    m.reset_recovery()
    assert m.recovery_count == 0


def test_healthy_exit_resets_recovery_counter():
    pol = StubPolicy()
    m = GameStateMachine(policy=pol)
    pol.signals = {SIGNAL_HEALTH_LOW: True}
    m.decide(OBS, 1000)  # -> RECOVERY
    m.note_recovery_attempt()
    m.note_recovery_attempt()
    assert m.recovery_count == 2
    pol.signals = {SIGNAL_HEALTH_RECOVERED: True}
    assert m.decide(OBS, 1000) is AgentState.IDLE
    assert m.recovery_count == 0  # healthy exit resets the episode


# --- on_transition seam (step 4 refines this into a LoopEvent) ---------------


def test_enter_fires_on_transition_with_before_after_reason_now():
    events: list = []
    m, _ = make({SIGNAL_HEALTH_LOW: True},
                on_transition=lambda b, a, r, t: events.append((b, a, r, t)))
    m.decide(OBS, 1234.5)
    assert events == [
        (AgentState.IDLE, AgentState.RECOVERY, "low health on idle scan", 1234.5)
    ]


def test_loop_forces_recovery_on_observation_failure_via_enter():
    events: list = []
    m, _ = make({}, on_transition=lambda b, a, r, t: events.append((b, a, r, t)))
    # §4.3 "any -> RECOVERY: observation failure this tick" is a loop-driven
    # forced move (no observation to read), exposed through the controlled writer.
    m.enter(AgentState.RECOVERY, "observation failure this tick", 2000.0)
    assert m.state is AgentState.RECOVERY
    assert m.recovery_count == 0  # fresh recovery episode
    assert events == [
        (AgentState.IDLE, AgentState.RECOVERY, "observation failure this tick", 2000.0)
    ]


# --- Watchdog ----------------------------------------------------------------


def test_watchdog_fires_in_recovery_after_timeout():
    pol = StubPolicy()
    m = GameStateMachine(policy=pol)  # state_timeout_s = 5.0
    pol.signals = {SIGNAL_HEALTH_LOW: True}
    m.decide(OBS, 1000)  # -> RECOVERY, entered_at = 1000
    assert m.state is AgentState.RECOVERY
    assert m.decide(OBS, 1004) is AgentState.RECOVERY  # 4s < 5s: still recovering
    assert m.decide(OBS, 1005) is AgentState.PAUSED  # 5s >= 5s: forced pause


def test_watchdog_does_not_fire_in_non_watchdog_states():
    pol = StubPolicy()
    m = GameStateMachine(policy=pol)  # IDLE: not a watchdog state
    pol.signals = {}
    assert m.decide(OBS, 1000) is AgentState.IDLE
    assert m.decide(OBS, 1_000_000) is AgentState.IDLE  # indefinite: loop-bounded


# --- None-safety: unknown is never False / 0% health -------------------------


def test_none_signals_are_never_treated_as_false():
    pol = StubPolicy()
    m = GameStateMachine(policy=pol, initial=AgentState.COMBAT)
    pol.signals = {
        SIGNAL_TARGET_PRESENT: None,
        SIGNAL_TARGET_HOSTILE: None,
        SIGNAL_HEALTH_LOW: None,
    }
    assert m.decide(OBS, 1000) is AgentState.COMBAT  # unknown target != target lost

    pol2 = StubPolicy()
    m2 = GameStateMachine(policy=pol2, initial=AgentState.LOOTING)
    pol2.signals = {SIGNAL_LOOT_PRESENT: None}
    assert m2.decide(OBS, 1000) is AgentState.LOOTING  # unknown loot != cleared


# --- Policy is the only signal source ----------------------------------------


def test_policy_is_called_once_per_decide():
    pol = StubPolicy()
    m = GameStateMachine(policy=pol)
    pol.signals = {}
    m.decide(OBS, 1000)
    m.decide(OBS, 1001)
    assert pol.calls == 2


# --- Constructor validation --------------------------------------------------


def test_constructor_validates_bounds():
    with pytest.raises(ValueError):
        GameStateMachine(policy=StubPolicy(), recovery_attempts=0)
    with pytest.raises(ValueError):
        GameStateMachine(policy=StubPolicy(), state_timeout_s=0.0)
