"""Phase 4, step 2: ``GameStateMachine.decide()`` + transitions (machine.py).

Red tests for the deterministic decision core. The machine is driven by a
*stub* policy whose ``transition_signals(obs)`` returns a fixed dict of the
stable signal keys — the machine never sees zone names, so the test matrix
stays small and game-agnostic. A fake clock (``now``) is used so no real time
elapses in the watchdog tests.

Step 4 (plan §4.2) extends this file with the ``LoopEvent`` contract: the
``on_transition`` seam fires frozen ``LoopEvent`` objects (``STATE_ENTER``),
and the tests assert the exact field set, the closed ``EventType``, sequence
numbering, and the emitted reasons — all through the public callback, never
private counters.
"""

from __future__ import annotations

import dataclasses
import json
import typing

import pytest

from ai_game_agent.state.events import EventType, LoopEvent
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
    m, _ = make({
        SIGNAL_LOOT_PRESENT: True,
        SIGNAL_TARGET_PRESENT: True,
        SIGNAL_TARGET_HOSTILE: True,
    })
    assert m.decide(OBS, 1000) is AgentState.LOOTING  # table order is the contract


def test_combat_target_lost_to_idle():
    m, _ = make({SIGNAL_TARGET_PRESENT: False}, initial=AgentState.COMBAT)
    assert m.decide(OBS, 1000) is AgentState.IDLE

def test_combat_non_hostile_but_present_target_does_not_end_combat():
    m, _ = make({
        SIGNAL_TARGET_PRESENT: True,
        SIGNAL_TARGET_HOSTILE: False,
    }, initial=AgentState.COMBAT)
    assert m.decide(OBS, 1000) is AgentState.COMBAT



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
    events: list[LoopEvent] = []
    m, _ = make({SIGNAL_TARGET_HOSTILE: True}, initial=AgentState.COMBAT,
                on_transition=events.append)
    assert m.decide(OBS, 1000) is AgentState.COMBAT
    assert events == []  # self-loop: no spurious STATE_ENTER event


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


# --- on_transition seam: LoopEvent (plan §4.2) -------------------------------


def make_event(**overrides: object) -> LoopEvent:
    """A valid ``LoopEvent`` for shape tests; override any field."""
    base: dict[str, object] = {
        "ts": 1000.0,
        "seq": 0,
        "event": EventType.STATE_ENTER,
        "state_before": AgentState.IDLE,
        "state_after": AgentState.RECOVERY,
        "reason": "low health on idle scan",
        "detail": {},
    }
    base.update(overrides)
    return LoopEvent(**base)


def test_enter_fires_state_enter_event_with_before_after_reason_now():
    events: list[LoopEvent] = []
    m, _ = make({SIGNAL_HEALTH_LOW: True}, on_transition=events.append)
    m.decide(OBS, 1234.5)
    assert len(events) == 1
    ev = events[0]
    assert ev.event is EventType.STATE_ENTER
    assert ev.state_before is AgentState.IDLE
    assert ev.state_after is AgentState.RECOVERY
    assert ev.reason == "low health on idle scan"  # §4.3 table reason, verbatim
    assert ev.ts == 1234.5  # the injected now, verbatim (no wall clock)
    assert ev.seq == 0
    assert ev.detail == {}


def test_loop_forces_recovery_on_observation_failure_via_enter():
    events: list[LoopEvent] = []
    m, _ = make({}, on_transition=events.append)
    # §4.3 "any -> RECOVERY: observation failure this tick" is a loop-driven
    # forced move (no observation to read), exposed through the controlled writer.
    m.enter(AgentState.RECOVERY, "observation failure this tick", 2000.0)
    assert m.state is AgentState.RECOVERY
    assert m.recovery_count == 0  # fresh recovery episode
    assert len(events) == 1
    ev = events[0]
    assert ev.event is EventType.STATE_ENTER
    assert ev.state_before is AgentState.IDLE
    assert ev.state_after is AgentState.RECOVERY
    assert ev.reason == "observation failure this tick"
    assert ev.ts == 2000.0
    assert ev.seq == 0


def test_enter_self_loop_emits_no_event():
    events: list[LoopEvent] = []
    m, _ = make({SIGNAL_TARGET_HOSTILE: True}, initial=AgentState.COMBAT,
                on_transition=events.append)
    m.enter(AgentState.COMBAT, "still engaging hostile target", 1000.0)
    assert events == []  # idempotent writer: no spurious STATE_ENTER


# --- LoopEvent contract (plan §4.2) ------------------------------------------


def test_state_package_reexports_the_event_contract():
    from ai_game_agent.state import EventType as PkgET
    from ai_game_agent.state import LoopEvent as PkgLE

    assert PkgET is EventType
    assert PkgLE is LoopEvent


def test_event_type_is_closed_with_exactly_the_plan_values():
    assert len(EventType) == 8
    assert {e.name for e in EventType} == {
        "TICK", "STATE_ENTER", "STATE_EXIT", "ACTION_BATCH",
        "ACTION_RESULT", "RECOVERY", "STOP", "ERROR",
    }
    for e in EventType:  # closed str-enum: string interop + round-trip
        assert isinstance(e, str)
        assert EventType(str(e)) is e
    with pytest.raises(ValueError):
        EventType("bogus")  # closed: no implicit coercion


def test_loop_event_is_frozen_dataclass_with_exact_fields():
    assert LoopEvent.__dataclass_params__.frozen  # frozen=True dataclass
    fields = dataclasses.fields(LoopEvent)
    assert [f.name for f in fields] == [
        "ts", "seq", "event", "state_before", "state_after", "reason", "detail",
    ]
    hints = typing.get_type_hints(LoopEvent)
    assert hints["ts"] is float
    assert hints["seq"] is int
    assert hints["event"] is EventType
    assert hints["state_before"] == AgentState | None
    assert hints["state_after"] == AgentState | None
    assert hints["reason"] is str
    assert hints["detail"] == dict[str, object]


def test_loop_event_rejects_mutation():
    ev = make_event()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.ts = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.reason = "other"  # type: ignore[misc]


def test_loop_event_to_dict_is_json_round_trippable():
    ev = make_event(detail={"zone": "health_bar", "present": True, "attempt": 2})
    d = ev.to_dict()
    keys = set(d)
    assert keys == {"ts", "seq", "event", "state_before", "state_after", "reason", "detail"}
    assert json.loads(json.dumps(d)) == d  # JSON-serializable end to end
    assert d["event"] == "state_enter"  # enums as their string values
    assert d["state_before"] == "idle"
    assert d["state_after"] == "recovery"
    assert d["detail"] == {"zone": "health_bar", "present": True, "attempt": 2}


def test_loop_event_to_dict_defensively_copies_nested_detail():
    ev = make_event(detail={"zone": "health_bar", "nested": {"present": True}})
    d = ev.to_dict()
    # Mutating the serialized result must not leak back into the event —
    # neither at the top level nor through nested containers (deep-copy
    # isolation), even though the dataclass itself is frozen.
    d["detail"]["nested"]["present"] = False
    d["detail"]["injected"] = "mutated"
    assert ev.detail == {"zone": "health_bar", "nested": {"present": True}}
    # A fresh serialization still yields the original nested values.
    assert ev.to_dict()["detail"]["nested"] == {"present": True}


def test_machine_event_seq_is_0_based_and_monotonic():
    events: list[LoopEvent] = []
    m, pol = make({SIGNAL_HEALTH_LOW: True}, on_transition=events.append)
    m.decide(OBS, 1000)  # IDLE -> RECOVERY (seq 0)
    pol.signals = {SIGNAL_HEALTH_RECOVERED: True}
    m.decide(OBS, 1001)  # RECOVERY -> IDLE (seq 1)
    assert [e.seq for e in events] == [0, 1]


def test_machine_event_seq_is_per_instance():
    ev_a: list[LoopEvent] = []
    m_a, _ = make({SIGNAL_HEALTH_LOW: True}, on_transition=ev_a.append)
    m_a.decide(OBS, 1000)
    ev_b: list[LoopEvent] = []
    m_b, _ = make({SIGNAL_HEALTH_LOW: True}, on_transition=ev_b.append)
    m_b.decide(OBS, 1000)
    assert ev_a[0].seq == 0
    assert ev_b[0].seq == 0  # a fresh machine starts a fresh session


def test_watchdog_pause_emits_state_enter_event():
    events: list[LoopEvent] = []
    pol = StubPolicy()
    m = GameStateMachine(policy=pol, on_transition=events.append)
    pol.signals = {SIGNAL_HEALTH_LOW: True}
    m.decide(OBS, 1000)  # IDLE -> RECOVERY
    assert m.decide(OBS, 1005) is AgentState.PAUSED  # watchdog fires
    ev = events[-1]
    assert ev.event is EventType.STATE_ENTER
    assert ev.state_before is AgentState.RECOVERY
    assert ev.state_after is AgentState.PAUSED
    assert ev.reason == "state timeout (watchdog)"
    assert ev.ts == 1005


def test_recovery_exhaustion_emits_state_enter_event():
    events: list[LoopEvent] = []
    pol = StubPolicy()
    m = GameStateMachine(policy=pol, on_transition=events.append)
    pol.signals = {SIGNAL_HEALTH_LOW: True}
    m.decide(OBS, 1000)  # IDLE -> RECOVERY (seq 0)
    for _ in range(3):
        m.note_recovery_attempt()
    pol.signals = {SIGNAL_HEALTH_RECOVERED: True}
    assert m.decide(OBS, 1000) is AgentState.PAUSED  # exhausted regardless
    ev = events[-1]
    assert ev.event is EventType.STATE_ENTER
    assert ev.state_before is AgentState.RECOVERY
    assert ev.state_after is AgentState.PAUSED
    assert ev.reason == "recovery attempts exhausted; safe pause"
    assert ev.seq == 1


def test_construction_does_not_emit_event():
    # The initial state has no clock and no transition: no event may exist.
    events: list[LoopEvent] = []
    make({SIGNAL_HEALTH_LOW: True}, on_transition=events.append)
    assert events == []


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
