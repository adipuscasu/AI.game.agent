"""Phase 4, step 3: ``StatePolicy`` + WoW ``DeterministicPolicy``.

Red tests for the "Action Plan" table and the stable signal seam that the
step-2 ``GameStateMachine`` consumes. The policy is the *only* place in the
codebase that knows literal WoW zone names (``games/wow/zones.py``); the
machine and loop stay game-agnostic and read a typed ``dict`` of signals.

Contract points encoded here (from ``docs/phase-4-implementation-plan.md``
§4.4 + the step-3 review corrections):

* ``policy.actions(state, obs, mode)`` returns the §4.4 action tuple under
  an input-allowed mode and ``()`` under ``OBSERVE_ONLY`` for *every* state
  (seeing and deciding is allowed; acting is not).
* COMBAT emits the configurable key/delay rotation (default ``q`` / 120 ms /
  ``w``). RECOVERY emits ``r`` / 300 ms. LOOTING clicks the marker's actual
  bounding-box center, then the pickup key — and emits **no click at all**
  when no marker was located (never an unlocated ``MOUSE_CLICK``).
* ``transition_signals(obs)`` returns *exactly* the five keys the machine
  reads: ``health_low``, ``health_recovered``, ``loot_present``,
  ``target_present``, ``target_hostile``.
* A *per-name* detector failure (``ui:loot_glow: ...`` /
  ``templates:loot_glow: ...``) makes that presence signal ``None``
  (unknown), not ``False``. A *different* zone's error does **not** flip it.
* Health is **unknown by default** (``Observation`` carries no calibrated
  health fraction). When a ``health_fraction`` callable is supplied, a
  reading of ``0.20`` → ``health_low`` True; ``0.60`` → ``health_recovered``
  True; an invalid reading (``bool`` / non-finite / out of [0,1]) is rejected
  → both health signals ``None`` (no exception, no tick death).
* ``UiZoneHit.value`` is check evidence in [0,1] and is **never** treated as
  a WoW health percentage.
* Every action carries a non-empty ``source`` for replay provenance.
"""

from __future__ import annotations

import datetime
import math

import pytest

from ai_game_agent.actions.action import Action, ActionKind
from ai_game_agent.actions.modes import OperationMode
from ai_game_agent.games.wow.policy import DeterministicPolicy
from ai_game_agent.games.wow.zones import (
    ZONE_HEALTH_BAR,
    ZONE_LOOT_GLOW,
    ZONE_TARGET,
    ZONE_TARGET_HOSTILE,
)
from ai_game_agent.perception.observation import BBox, Observation, UiZoneHit
from ai_game_agent.state.policy import StatePolicy
from ai_game_agent.state.states import AgentState

# The exact signal keys the step-2 machine consumes — the policy must emit
# *these* and no others.
SIGNAL_KEYS = {
    "health_low",
    "health_recovered",
    "loot_present",
    "target_present",
    "target_hostile",
}

# Zone-name literals must live ONLY in games/wow/ + this test file.
ZONE_LITERALS = ('"loot_glow"', '"health_bar"', '"target_hostile"', '"target"')


def _obs(
    *,
    zones: tuple[UiZoneHit, ...] = (),
    detector_errors: tuple[str, ...] = (),
) -> Observation:
    """Build a minimal real ``Observation`` for policy testing."""
    return Observation(
        frame_width=1920,
        frame_height=1080,
        captured_at=datetime.datetime(2026, 10, 13, 12, 0, 0),
        source="test",
        templates=(),
        ui_zones=zones,
        text_regions=(),
        objects=(),
        detector_errors=detector_errors,
    )


def _zone(name: str, *, bbox: BBox | None = None, check: str = "presence") -> UiZoneHit:
    return UiZoneHit(
        zone=name,
        check=check,
        bbox=bbox or BBox(x=0, y=0, width=1, height=1),
        value=1.0,  # presence evidence; value is NOT a health fraction
    )


# --- Protocol conformance ----------------------------------------------------


def test_deterministic_policy_satisfies_state_policy_protocol():
    assert isinstance(DeterministicPolicy(), StatePolicy)


def test_policy_exposes_stable_names():
    assert DeterministicPolicy().name() == "wow"


# --- §4.4 action table (input-allowed mode) ---------------------------------


def test_combat_emits_configured_rotation():
    p = DeterministicPolicy()
    obs = _obs(zones=(_zone(ZONE_TARGET), _zone(ZONE_TARGET_HOSTILE)))
    actions = p.actions(AgentState.COMBAT, obs, OperationMode.AUTONOMOUS)
    assert tuple(a.kind for a in actions) == (
        ActionKind.KEY_PRESS,
        ActionKind.DELAY,
        ActionKind.KEY_PRESS,
    )
    assert actions[0].key == "q"
    assert actions[1].duration_ms == 120
    assert actions[2].key == "w"


def test_recovery_emits_res_hotkey_then_delay():
    p = DeterministicPolicy()
    obs = _obs()
    actions = p.actions(AgentState.RECOVERY, obs, OperationMode.AUTONOMOUS)
    assert tuple(a.kind for a in actions) == (ActionKind.KEY_PRESS, ActionKind.DELAY)
    assert actions[0].key == "r"
    assert actions[1].duration_ms == 300


def test_looting_clicks_marker_center_then_pickup_key():
    p = DeterministicPolicy()
    marker = _zone(ZONE_LOOT_GLOW, bbox=BBox(x=100, y=200, width=10, height=20))
    obs = _obs(zones=(marker,))
    actions = p.actions(AgentState.LOOTING, obs, OperationMode.AUTONOMOUS)
    assert actions[0].kind is ActionKind.MOUSE_CLICK
    # (100,200,10,20) -> center (105,210), from the hit's actual bbox.
    assert (actions[0].x, actions[0].y) == (105, 210)
    assert actions[1].kind is ActionKind.DELAY
    assert actions[2].kind is ActionKind.KEY_PRESS
    assert actions[2].key == "f"


def test_looting_with_no_marker_emits_no_click():
    """Correction #1: no located marker -> no MOUSE_CLICK at all."""
    p = DeterministicPolicy()
    obs = _obs()  # no loot_glow hit
    actions = p.actions(AgentState.LOOTING, obs, OperationMode.AUTONOMOUS)
    assert ActionKind.MOUSE_CLICK not in tuple(a.kind for a in actions)
    # The pickup still happens: DELAY + pickup key.
    assert tuple(a.kind for a in actions) == (ActionKind.DELAY, ActionKind.KEY_PRESS)
    assert actions[1].key == "f"


@pytest.mark.parametrize(
    "state",
    [AgentState.IDLE, AgentState.AI_ANALYSIS, AgentState.PAUSED],
)
def test_idle_and_terminal_states_emit_no_actions(state):
    p = DeterministicPolicy()
    obs = _obs(zones=(_zone(ZONE_TARGET), _zone(ZONE_TARGET_HOSTILE)))
    assert p.actions(state, obs, OperationMode.AUTONOMOUS) == ()


# --- Mode gating -------------------------------------------------------------


def test_observe_only_emits_no_actions_for_every_state():
    p = DeterministicPolicy()
    obs = _obs(
        zones=(_zone(ZONE_TARGET), _zone(ZONE_TARGET_HOSTILE), _zone(ZONE_LOOT_GLOW)),
    )
    for state in (
        AgentState.IDLE,
        AgentState.COMBAT,
        AgentState.LOOTING,
        AgentState.RECOVERY,
        AgentState.AI_ANALYSIS,
        AgentState.PAUSED,
    ):
        assert p.actions(state, obs, OperationMode.OBSERVE_ONLY) == ()


def test_observe_only_still_reports_signals():
    """Seeing + deciding is allowed under OBSERVE_ONLY; only acting is gated."""
    p = DeterministicPolicy()
    obs = _obs(zones=(_zone(ZONE_TARGET), _zone(ZONE_TARGET_HOSTILE)))
    signals = p.transition_signals(obs)
    assert signals["target_present"] is True
    assert signals["target_hostile"] is True


# --- transition_signals: the stable seam ------------------------------------


def test_transition_signals_returns_exactly_machine_keys():
    p = DeterministicPolicy()
    obs = _obs()
    assert set(p.transition_signals(obs).keys()) == SIGNAL_KEYS


def test_loot_present_true_when_marker_hit():
    p = DeterministicPolicy()
    obs = _obs(zones=(_zone(ZONE_LOOT_GLOW),))
    assert p.transition_signals(obs)["loot_present"] is True


def test_loot_present_false_when_no_hit_and_clean():
    p = DeterministicPolicy()
    obs = _obs()  # no loot hit, no detector error
    assert p.transition_signals(obs)["loot_present"] is False


def test_loot_present_none_when_its_detector_failed():
    """Correction #2: a per-name detector failure -> unknown (None), not False."""
    p = DeterministicPolicy()
    obs = _obs(detector_errors=("ui:loot_glow: no pixels",))
    assert p.transition_signals(obs)["loot_present"] is None


def test_loot_present_not_flipped_by_another_zone_failure():
    """Narrow: only the *named* zone's failure flips its own signal."""
    p = DeterministicPolicy()
    obs = _obs(detector_errors=("ui:health_bar: no pixels",))
    assert p.transition_signals(obs)["loot_present"] is False  # loot detector ran clean


def test_target_present_distinct_from_target_hostile():
    p = DeterministicPolicy()
    obs = _obs(zones=(_zone(ZONE_TARGET),))  # target present, NOT hostile
    signals = p.transition_signals(obs)
    assert signals["target_present"] is True
    assert signals["target_hostile"] is False


def test_target_hostile_true_when_hostile_marker_hit():
    p = DeterministicPolicy()
    obs = _obs(zones=(_zone(ZONE_TARGET), _zone(ZONE_TARGET_HOSTILE)))
    signals = p.transition_signals(obs)
    assert signals["target_present"] is True
    assert signals["target_hostile"] is True


def test_target_present_none_when_its_detector_failed():
    p = DeterministicPolicy()
    obs = _obs(detector_errors=("templates:target: match failed",))
    assert p.transition_signals(obs)["target_present"] is None


# --- Health: unknown by default, calibrated when provided -------------------


def test_health_signals_none_by_default():
    p = DeterministicPolicy()  # no health_fraction callable
    obs = _obs(zones=(_zone(ZONE_HEALTH_BAR, check="brightness"),))
    signals = p.transition_signals(obs)
    assert signals["health_low"] is None
    assert signals["health_recovered"] is None


def test_health_low_when_fraction_below_floor():
    p = DeterministicPolicy(health_fraction=lambda obs: 0.20)
    signals = p.transition_signals(_obs())
    assert signals["health_low"] is True      # 0.20 < 0.30
    assert signals["health_recovered"] is False  # 0.20 < 0.50


def test_health_recovered_when_fraction_above_floor():
    p = DeterministicPolicy(health_fraction=lambda obs: 0.60)
    signals = p.transition_signals(_obs())
    assert signals["health_low"] is False     # 0.60 >= 0.30
    assert signals["health_recovered"] is True  # 0.60 >= 0.50


def test_health_unknown_when_fraction_returns_none():
    p = DeterministicPolicy(health_fraction=lambda obs: None)
    signals = p.transition_signals(_obs())
    assert signals["health_low"] is None
    assert signals["health_recovered"] is None


@pytest.mark.parametrize("bad", [True, math.nan, float("inf"), -0.5, 1.5, "0.5"])
def test_invalid_health_reading_rejected_not_raised(bad):
    """Correction #3: a bad reading -> both health signals None, no exception."""
    p = DeterministicPolicy(health_fraction=lambda obs: bad)
    signals = p.transition_signals(_obs())
    assert signals["health_low"] is None
    assert signals["health_recovered"] is None


# --- Provenance --------------------------------------------------------------


def test_every_action_carries_policy_source():
    p = DeterministicPolicy()
    marker = _zone(ZONE_LOOT_GLOW, bbox=BBox(x=100, y=200, width=10, height=20))
    for state, obs in [
        (AgentState.COMBAT, _obs(zones=(_zone(ZONE_TARGET), _zone(ZONE_TARGET_HOSTILE)))),
        (AgentState.LOOTING, _obs(zones=(marker,))),
        (AgentState.RECOVERY, _obs()),
    ]:
        for action in p.actions(state, obs, OperationMode.AUTONOMOUS):
            assert action.source == "policy:wow"
            assert isinstance(action, Action)


# --- Constructor validation --------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"combat_keys": ()},
        {"combat_delay_ms": 0},
        {"combat_delay_ms": -1},
        {"loot_delay_ms": 0},
        {"recovery_delay_ms": 0},
        {"loot_pickup_key": ""},
        {"recovery_key": ""},
        {"health_low": 0.0},
        {"health_low": 1.5},
        {"health_recovered": 0.0},
        {"health_recovered": 2.0},
        {"health_low": 0.8, "health_recovered": 0.2},  # low must be < recovered
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DeterministicPolicy(**kwargs)


def test_constructor_accepts_valid_custom_configuration():
    p = DeterministicPolicy(
        name="wow-alt",
        combat_keys=("1", "2"),
        combat_delay_ms=250,
        loot_pickup_key="b",
        loot_delay_ms=500,
        recovery_key="t",
        recovery_delay_ms=400,
        health_low=0.40,
        health_recovered=0.60,
    )
    assert p.name() == "wow-alt"
    obs = _obs(zones=(_zone(ZONE_TARGET), _zone(ZONE_TARGET_HOSTILE)))
    actions = p.actions(AgentState.COMBAT, obs, OperationMode.AUTONOMOUS)
    assert actions[0].key == "1"
    assert actions[1].duration_ms == 250
    assert actions[2].key == "2"


# --- Zone-name literals stay inside games/wow/ -------------------------------


def test_no_zone_literal_leaks_outside_games_wow():
    """The policy module must not contain a *quoted* zone-name literal; it
    imports the constants from zones.py instead."""
    import ai_game_agent.games.wow.policy as policy_mod

    with open(policy_mod.__file__, encoding="utf-8") as fh:
        source = fh.read()
    for literal in ZONE_LITERALS:
        assert literal not in source, f"zone literal {literal} must live in zones.py"


# --- Phase 5 close-out: ai_enabled signal producer + ai_prompt --------------
# The policy owns the SIGNAL_AI_ANALYSIS_REQUIRED producer and the ai_prompt()
# method. The signal is only present in the returned dict when ai_enabled is
# True (the existing "exactly 5 keys" test above holds by default).


def test_ai_signal_absent_by_default():
    """Default ai_enabled=False -> the key is not in the returned dict."""
    p = DeterministicPolicy()
    obs = _obs()  # clean idle observation
    assert "ai_analysis_required" not in p.transition_signals(obs)


def test_ai_signal_present_when_enabled_idle():
    p = DeterministicPolicy(ai_enabled=True)
    obs = _obs()  # clean idle observation
    assert p.transition_signals(obs)["ai_analysis_required"] is True


def test_ai_signal_suppressed_when_hostile():
    p = DeterministicPolicy(ai_enabled=True)
    obs = _obs(zones=(_zone(ZONE_TARGET), _zone(ZONE_TARGET_HOSTILE)))
    assert p.transition_signals(obs)["ai_analysis_required"] is False


def test_ai_signal_suppressed_when_loot():
    p = DeterministicPolicy(ai_enabled=True)
    obs = _obs(zones=(_zone(ZONE_LOOT_GLOW),))
    assert p.transition_signals(obs)["ai_analysis_required"] is False


def test_ai_signal_absent_when_enabled_false_explicit():
    p = DeterministicPolicy(ai_enabled=False)
    obs = _obs()
    assert "ai_analysis_required" not in p.transition_signals(obs)


def test_ai_prompt_returns_string():
    p = DeterministicPolicy(ai_enabled=True)
    obs = _obs(zones=(_zone(ZONE_TARGET_HOSTILE),))
    prompt = p.ai_prompt(AgentState.AI_ANALYSIS, obs)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_ai_enabled_defaults_false():
    p = DeterministicPolicy()
    assert p.ai_enabled is False


def test_ai_enabled_true_round_trip():
    p = DeterministicPolicy(ai_enabled=True)
    assert p.ai_enabled is True
