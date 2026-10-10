"""Step 3: ``OperationMode`` + ``SafetyGuard`` (mode gate + bounded limits).

The plan's regression priority: **the ``OBSERVE_ONLY`` gate is proven first**
— it is the one mode the whole project's safety story depends on, and the
mock backend's event log must be asserted *empty* under it. All tests run
headless (no pynput, no desktop, no executor yet — the guard is a pure
validator with a small stateful batch counter and an error counter).
"""

from __future__ import annotations

import pytest

from ai_game_agent.actions import Action, ActionKind, OperationMode, SafetyGuard
from ai_game_agent.actions.base import (
    ActionError,
    ActionValidationError,
    ModeViolation,
    SafetyViolation,
)
from ai_game_agent.config import Config, InputConfig, SafetyConfig

# --- OperationMode: the closed enum the config string validates against -----


def test_mode_enum_has_exactly_four_values():
    assert {m.value for m in OperationMode} == {
        "observe_only",
        "assisted",
        "semi_autonomous",
        "autonomous",
    }


def test_mode_enum_is_string_comparable():
    # StrEnum: the enum value == its string, so config strings, JSON
    # payloads, and the enum all interoperate without coercion.
    assert OperationMode.OBSERVE_ONLY == "observe_only"
    assert OperationMode("autonomous") is OperationMode.AUTONOMOUS
    with pytest.raises(ValueError):
        OperationMode("obseve_only")  # typo: no implicit coercion


def test_mode_allows_input():
    assert not OperationMode.OBSERVE_ONLY.input_allowed
    assert OperationMode.ASSISTED.input_allowed
    assert OperationMode.SEMI_AUTONOMOUS.input_allowed
    assert OperationMode.AUTONOMOUS.input_allowed


# --- Mode gate (the safety-critical regression) ------------------------------


def test_observe_only_gate_blocks_all_action_kinds():
    guard = SafetyGuard(mode=OperationMode.OBSERVE_ONLY)
    kinds = (
        ("key_press", Action(kind=ActionKind.KEY_PRESS, key="w")),
        ("key_hold", Action(kind=ActionKind.KEY_HOLD, key="w", duration_ms=100)),
        ("mouse_move", Action(kind=ActionKind.MOUSE_MOVE, x=10, y=20)),
        ("mouse_click", Action(kind=ActionKind.MOUSE_CLICK)),
        ("mouse_down", Action(kind=ActionKind.MOUSE_DOWN, button="left")),
        ("mouse_up", Action(kind=ActionKind.MOUSE_UP, button="left")),
        ("mouse_scroll", Action(kind=ActionKind.MOUSE_SCROLL, scroll=3)),
        ("delay", Action(kind=ActionKind.DELAY, duration_ms=100)),
    )
    for name, action in kinds:
        with pytest.raises(ModeViolation) as excinfo:
            guard.check(action)
        assert name in str(excinfo.value), f"{name}: message must name the kind"
        assert "OBSERVE_ONLY" in str(excinfo.value)


def test_mode_gate_is_mode_scoped_not_backend_scoped():
    # The gate is about the *mode*, not the action or the backend: an
    # otherwise-valid action under OBSERVE_ONLY is rejected identically.
    guard = SafetyGuard(mode=OperationMode.OBSERVE_ONLY)
    with pytest.raises(ModeViolation):
        guard.check(Action(kind=ActionKind.KEY_PRESS, key="w"))
    with pytest.raises(ModeViolation):
        guard.check(Action(kind=ActionKind.MOUSE_SCROLL, scroll=1))


def test_autonomous_gate_allows():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    guard.check(Action(kind=ActionKind.KEY_PRESS, key="w"))  # no raise


def test_assisted_gate_allows_execution_but_reports_violations():
    # ASSISTED's contract: safe actions execute (the gate passes), while
    # bound violations are *reported* (raised for the executor to catch and
    # convert into a proposal/report), never silently executed.
    guard = SafetyGuard(mode=OperationMode.ASSISTED)
    guard.check(Action(kind=ActionKind.KEY_PRESS, key="w"))  # allowed
    with pytest.raises(SafetyViolation):
        guard.check(Action(kind=ActionKind.DELAY, duration_ms=10**9))  # reported


# --- Bounds: duration / delay ------------------------------------------------


def test_duration_within_bound_passes():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    guard.check(Action(kind=ActionKind.DELAY, duration_ms=5000))  # == max (default 5000)


def test_duration_over_bound_raises():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    with pytest.raises(SafetyViolation) as excinfo:
        guard.check(Action(kind=ActionKind.DELAY, duration_ms=5001))
    assert "5000" in str(excinfo.value)  # message names the bound


def test_hold_duration_over_bound_raises():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    with pytest.raises(SafetyViolation):
        guard.check(Action(kind=ActionKind.KEY_HOLD, key="w", duration_ms=5001))


def test_delay_field_is_bounded_too():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    guard.check(Action(kind=ActionKind.DELAY, duration_ms=1000, delay_ms=5000))
    with pytest.raises(SafetyViolation):
        guard.check(Action(kind=ActionKind.DELAY, duration_ms=1000, delay_ms=5001))


def test_negative_timing_rejected_by_contract_not_guard():
    # The Action contract forbids negative timing at construction; the guard
    # never sees them (defense in depth: the contract is the first line).
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.DELAY, duration_ms=-1)


def test_custom_duration_bound_from_config():
    cfg = Config(
        input=InputConfig(mode="autonomous"),
        safety=SafetyConfig(max_action_duration_ms=100),
    )
    guard = SafetyGuard.from_config(cfg)
    guard.check(Action(kind=ActionKind.DELAY, duration_ms=100))
    with pytest.raises(SafetyViolation):
        guard.check(Action(kind=ActionKind.DELAY, duration_ms=101))


def test_mode_gate_fires_before_bound_check():
    # The OBSERVE_ONLY gate is checked before any bound: an action that is
    # also over the duration bound is rejected as a *mode* violation, not a
    # bound violation — the mode is the safety story.
    cfg = Config(safety=SafetyConfig(max_action_duration_ms=1))
    guard = SafetyGuard.from_config(cfg)
    with pytest.raises(ModeViolation):
        guard.check(Action(kind=ActionKind.DELAY, duration_ms=10**9))


# --- Coordinate sanity lives in the Action contract, not the guard ------------


def test_coordinate_sanity_lives_in_the_action_contract():
    # Negative *absolute* coordinates are rejected at the Action contract
    # (the single source of truth — the guard cannot receive such an action;
    # it is unconstructable). (0, 0) is a valid absolute coordinate, and
    # relative deltas may legitimately be negative.
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_MOVE, x=-1, y=0)
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_MOVE, x=0, y=-1)

    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    guard.check(Action(kind=ActionKind.MOUSE_MOVE, x=0, y=0))
    guard.check(Action(kind=ActionKind.MOUSE_MOVE, x=-10, y=5, relative=True))
    guard.check(Action(kind=ActionKind.KEY_PRESS, key="w", x=None, y=None))


# --- Batch bound: max_consecutive_actions ------------------------------------


def test_consecutive_actions_within_bound_pass():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    for _ in range(50):  # default max is 50
        guard.check(Action(kind=ActionKind.KEY_PRESS, key="w"))


def test_consecutive_actions_over_bound_raises():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    for _ in range(50):
        guard.check(Action(kind=ActionKind.KEY_PRESS, key="w"))
    with pytest.raises(SafetyViolation) as excinfo:
        guard.check(Action(kind=ActionKind.KEY_PRESS, key="w"))  # 51st
    assert "50" in str(excinfo.value)


def test_batch_counter_resets():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    for _ in range(50):
        guard.check(Action(kind=ActionKind.KEY_PRESS, key="w"))
    guard.reset_batch()
    guard.check(Action(kind=ActionKind.KEY_PRESS, key="w"))  # new batch, no raise


# --- Backend error bound (bounded retry, no infinite loops) ------------------


def test_backend_error_counter_is_bounded():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    assert guard.backend_errors == 0
    for _ in range(3):  # default max_backend_errors is 3
        guard.record_backend_error()
    assert guard.backend_errors == 3
    assert guard.should_pause  # bounded retry exhausted -> PAUSE, never loop


def test_backend_error_counter_resets_on_success():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    guard.record_backend_error()
    guard.record_backend_error()
    guard.record_success()  # one success clears the consecutive count
    assert guard.backend_errors == 0
    assert not guard.should_pause


def test_counters_are_per_guard_instance():
    # Documented semantics: counters are per guard instance (each executor
    # run uses a fresh guard or reset_batch), never shared across instances.
    a = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    b = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    a.check(Action(kind=ActionKind.KEY_PRESS, key="w"))
    a.record_backend_error()
    assert a.batch_count == 1
    assert a.backend_errors == 1
    assert b.batch_count == 0
    assert b.backend_errors == 0


def test_custom_backend_error_bound():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS, max_backend_errors=1)
    guard.record_backend_error()
    assert guard.should_pause


# --- Construction / DI -------------------------------------------------------


def test_guard_is_immutable():
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    with pytest.raises(AttributeError):
        guard.mode = OperationMode.OBSERVE_ONLY
    with pytest.raises(AttributeError):
        guard.max_action_duration_ms = 1


def test_guard_accepts_string_mode():
    # Config strings arrive as plain str; the guard must accept them (the
    # OperationMode coercion is one-way and lossless).
    guard = SafetyGuard(mode="autonomous")
    assert guard.mode is OperationMode.AUTONOMOUS


def test_from_config_defaults():
    guard = SafetyGuard.from_config()
    assert guard.mode is OperationMode.OBSERVE_ONLY  # the shipped default
    assert guard.max_action_duration_ms == 5000
    assert guard.max_consecutive_actions == 50
    assert guard.max_backend_errors == 3


def test_from_config_carries_config_values():
    cfg = Config(
        input=InputConfig(backend="mock", mode="semi_autonomous"),
        safety=SafetyConfig(
            max_action_duration_ms=123,
            max_consecutive_actions=7,
            max_backend_errors=2,
        ),
    )
    guard = SafetyGuard.from_config(cfg)
    assert guard.mode is OperationMode.SEMI_AUTONOMOUS
    assert guard.max_action_duration_ms == 123
    assert guard.max_consecutive_actions == 7
    assert guard.max_backend_errors == 2


def test_guard_rejects_invalid_mode_string():
    # The guard coerces the mode via the OperationMode enum; a typo is a
    # ValueError at the boundary (never a silently-wrong mode).
    with pytest.raises(ValueError):
        SafetyGuard(mode="obseve_only")


def test_invalid_mode_rejected_at_config_layer():
    # A bad mode string is a config-layer error: pydantic ValidationError on
    # direct view construction (ConfigError via load_config — see
    # test_actions_config.py). The guard never sees one: from_config receives
    # already-validated Config values.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        InputConfig(mode="obseve_only")


def test_mode_set_is_in_sync_across_layers():
    # config cannot import the enum (an actions import would cycle back
    # through config via base.py), so the single-canonical-set invariant is
    # enforced behaviorally here: every OperationMode value must round-trip
    # through the config model, and a typo must not.
    from pydantic import ValidationError

    for mode in OperationMode:
        assert InputConfig(mode=mode.value).mode == mode.value
    with pytest.raises(ValidationError):
        InputConfig(mode="not_a_mode")


# --- Error taxonomy ----------------------------------------------------------


def test_mode_violation_and_safety_violation_are_action_errors():
    assert issubclass(ModeViolation, ActionError)
    assert issubclass(SafetyViolation, ActionError)
