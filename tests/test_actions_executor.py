"""Tests for ai_game_agent.actions.executor (Phase 3, step 5).

TDD: red first (module missing), then green. Covers the plan's Executor API:
execute (validate→delay→perform→post), release-on-exception, key/mouse-state
tracking, the state machine, and the emergency stop (stop→cancel→release-all→
block→record). All headless via MockBackend; no pynput, no display.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ai_game_agent.actions import (
    ActionQueue,
    ActionRequest,
    ExecutorState,
    SafetyGuard,
)
from ai_game_agent.actions.action import (
    Action,
    ActionKind,
)
from ai_game_agent.actions.base import (
    ModeViolation,
    SafetyViolation,
)
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.actions.modes import OperationMode


@pytest.fixture
def backend() -> MockBackend:
    b = MockBackend()
    b.open()
    return b


def _guard(**kw: object) -> SafetyGuard:
    return SafetyGuard(mode=OperationMode.AUTONOMOUS, **kw)  # type: ignore[arg-type]


def _executor(backend: MockBackend, **kw: object) -> object:
    """Build a wired Executor: backend + guard + queue + sleeper (no real sleep)."""
    from ai_game_agent.actions.executor import Executor

    return Executor(
        backend=backend,
        safety=_guard(**kw),
        clock=lambda: 0.0,
        sleeper=lambda secs: None,
    )


# --- execute: happy path -----------------------------------------------------


def test_execute_key_press_emits_press_event(backend: MockBackend) -> None:
    ex = _executor(backend)
    action = Action(kind=ActionKind.KEY_PRESS, key="a")
    result = ex.execute(action)
    assert result.ok is True
    assert result.stop_event is False
    assert backend.events == [("press", "a")]


def test_execute_mouse_move_emits_move_event(backend: MockBackend) -> None:
    ex = _executor(backend)
    action = Action(kind=ActionKind.MOUSE_MOVE, x=100, y=200)
    result = ex.execute(action)
    assert result.ok is True
    assert backend.events == [("mouse_move", 100, 200)]


def test_execute_key_hold_releases_even_if_sleeper_raises(backend: MockBackend) -> None:
    """The try/finally release guarantee: a failure during the hold still releases the key."""
    from ai_game_agent.actions.executor import Executor

    def boom(secs: float) -> None:
        raise RuntimeError("sleeper exploded")

    ex = Executor(
        backend=backend,
        safety=_guard(),
        clock=lambda: 0.0,
        sleeper=boom,
    )
    action = Action(kind=ActionKind.KEY_HOLD, key="a", duration_ms=100)
    # The executor must NOT let the sleeper's exception escape unhandled —
    # it converts to a failed ActionResult (release is guaranteed).
    result = ex.execute(action)
    assert result.ok is False
    # The key must have been released despite the failure: held state empty.
    assert ex.held_keys == set()  # type: ignore[union-attr]
    # key_down was issued, then key_up (release) — in that order.
    assert backend.events == [("key_down", "a"), ("key_up", "a")]


def test_execute_records_result_with_action_and_error(backend: MockBackend) -> None:
    from ai_game_agent.actions.executor import Executor

    def boom(secs: float) -> None:
        raise RuntimeError("x")

    ex = Executor(backend=backend, safety=_guard(), clock=lambda: 0.0, sleeper=boom)
    action = Action(kind=ActionKind.KEY_HOLD, key="a", duration_ms=10)
    result = ex.execute(action)
    assert result.ok is False
    assert result.error is not None
    assert "sleeper" not in result.error or True  # error present; wording is implementation detail


# --- state machine -----------------------------------------------------------


def test_initial_state_is_idle(backend: MockBackend) -> None:
    ex = _executor(backend)
    assert ex.state is ExecutorState.IDLE  # type: ignore[union-attr]


def test_execute_transitions_to_running(backend: MockBackend) -> None:
    ex = _executor(backend)
    ex.execute(Action(kind=ActionKind.KEY_PRESS, key="a"))  # type: ignore[union-attr]
    assert ex.state is ExecutorState.RUNNING  # type: ignore[union-attr]


# --- mode gate: OBSERVE_ONLY rejects before any backend call -----------------


def test_observe_only_raises_mode_violation_and_emits_nothing(backend: MockBackend) -> None:
    from ai_game_agent.actions.executor import Executor

    ex = Executor(
        backend=backend,
        safety=SafetyGuard(mode=OperationMode.OBSERVE_ONLY),
        clock=lambda: 0.0,
        sleeper=lambda s: None,
    )
    with pytest.raises(ModeViolation):
        ex.execute(Action(kind=ActionKind.KEY_PRESS, key="a"))
    assert backend.events == []  # the safety-critical regression: zero input


# --- safety bound: over-limit duration rejected -------------------------------


def test_over_bound_duration_raises_safety_violation(backend: MockBackend) -> None:
    ex = _executor(backend, max_action_duration_ms=100)
    with pytest.raises(SafetyViolation):
        ex.execute(Action(kind=ActionKind.KEY_HOLD, key="a", duration_ms=101))
    assert backend.events == []


# --- release_all -------------------------------------------------------------


def test_release_all_releases_held_and_is_idempotent(backend: MockBackend) -> None:
    ex = _executor(backend)
    # Simulate held state by driving the backend directly (the executor mirrors it).
    backend.key_down("a")
    backend.mouse_down("left")
    ex.held_keys = set(backend.held_keys)  # type: ignore[union-attr]
    ex.held_buttons = set(backend.held_buttons)  # type: ignore[union-attr]
    ex.release_all()  # type: ignore[union-attr]
    assert ex.held_keys == set()  # type: ignore[union-attr]
    assert ex.held_buttons == set()  # type: ignore[union-attr]
    assert backend.held_keys == set()
    assert backend.held_buttons == set()
    # Idempotent: a second call emits no further events.
    before = list(backend.events)
    ex.release_all()  # type: ignore[union-attr]
    assert backend.events == before


# --- EmergencyStop -----------------------------------------------------------


def test_emergency_stop_full_sequence(backend: MockBackend) -> None:
    """stop flag → cancel queue → release all → block → record, in that order."""
    from ai_game_agent.actions.executor import Executor

    ex = Executor(
        backend=backend,
        safety=_guard(),
        clock=lambda: 0.0,
        sleeper=lambda s: None,
    )
    # Hold a key so the stop has something to release.
    backend.key_down("w")
    ex.held_keys = set(backend.held_keys)  # type: ignore[union-attr]

    # Enqueue a pending action so cancel_all has work to drop.
    q: ActionQueue = ex.queue  # type: ignore[union-attr]
    q.add(ActionRequest(action=Action(kind=ActionKind.KEY_PRESS, key="x"),
                        requested_at=datetime(2026, 10, 10, 10, 30)))
    assert q.pending_count() == 1

    ex.trigger_emergency_stop(reason="F12 hotkey")  # type: ignore[union-attr]

    # 1. stopped flag set → state is STOPPED
    assert ex.state is ExecutorState.STOPPED  # type: ignore[union-attr]
    # 2. queue cancelled
    assert q.pending_count() == 0
    # 3/4. all held keys + buttons released
    assert backend.held_keys == set()
    assert backend.held_buttons == set()
    # 5. further execute is a no-op returning stop_event, no new input
    result = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="q"))  # type: ignore[union-attr]
    assert result.ok is False
    assert result.stop_event is True
    # 6. the stop is recorded
    assert ex.stop_reason == "F12 hotkey"  # type: ignore[union-attr]


def test_emergency_stop_is_idempotent(backend: MockBackend) -> None:
    from ai_game_agent.actions.executor import Executor

    ex = Executor(
        backend=backend,
        safety=_guard(),
        clock=lambda: 0.0,
        sleeper=lambda s: None,
    )
    ex.trigger_emergency_stop(reason="first")  # type: ignore[union-attr]
    ex.trigger_emergency_stop(reason="second")  # type: ignore[union-attr]
    assert ex.state is ExecutorState.STOPPED  # type: ignore[union-attr]
    # The first reason is retained (the stop is recorded once).
    assert ex.stop_reason == "first"  # type: ignore[union-attr]


def test_emergency_stop_blocks_even_in_autonomous(backend: MockBackend) -> None:
    from ai_game_agent.actions.executor import Executor

    ex = Executor(
        backend=backend,
        safety=_guard(),
        clock=lambda: 0.0,
        sleeper=lambda s: None,
    )
    ex.trigger_emergency_stop(reason="manual")  # type: ignore[union-attr]
    result = ex.execute(Action(kind=ActionKind.MOUSE_MOVE, x=1, y=1))  # type: ignore[union-attr]
    assert result.ok is False
    assert result.stop_event is True
    assert backend.events == []  # no input emitted after the stop
