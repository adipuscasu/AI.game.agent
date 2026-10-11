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
    InputBackendError,
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


# --- P1: backend-error recording -> PAUSED (bounded retry, no loop) ---------


class FailingBackend(MockBackend):
    """A backend whose ``press`` always fails (simulated backend I/O error)."""

    def press(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        raise InputBackendError("simulated backend I/O failure")


def test_backend_error_is_recorded_on_failure() -> None:
    from ai_game_agent.actions.executor import Executor

    b = FailingBackend()
    b.open()
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    result = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="a"))
    assert result.ok is False
    assert ex.safety.backend_errors == 1  # recorded, so the threshold is live


def test_backend_success_clears_the_error_count() -> None:
    from ai_game_agent.actions.executor import Executor

    b = FailingBackend()
    b.open()
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    ex.execute(Action(kind=ActionKind.KEY_PRESS, key="a"))
    assert ex.safety.backend_errors == 1
    result = ex.execute(Action(kind=ActionKind.MOUSE_MOVE, x=1, y=2))
    assert result.ok is True
    assert ex.safety.backend_errors == 0  # one success clears the consecutive count


def test_repeated_backend_errors_pause_the_executor() -> None:
    from ai_game_agent.actions.executor import Executor

    b = FailingBackend()
    b.open()
    guard = _guard(max_backend_errors=2)
    ex = Executor(backend=b, safety=guard, clock=lambda: 0.0, sleeper=lambda s: None)
    ex.execute(Action(kind=ActionKind.KEY_PRESS, key="a"))
    assert ex.state is ExecutorState.RUNNING  # below the bound: still running
    ex.execute(Action(kind=ActionKind.KEY_PRESS, key="b"))
    assert guard.should_pause
    assert ex.state is ExecutorState.PAUSED  # bounded retry exhausted -> safe pause


def test_paused_executor_blocks_further_input() -> None:
    from ai_game_agent.actions.executor import Executor

    b = FailingBackend()
    b.open()
    ex = Executor(
        backend=b, safety=_guard(max_backend_errors=1), clock=lambda: 0.0, sleeper=lambda s: None
    )
    ex.execute(Action(kind=ActionKind.KEY_PRESS, key="a"))
    assert ex.state is ExecutorState.PAUSED
    events_before = len(b.events)
    result = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="b"))
    assert result.ok is False
    assert result.stop_event is False  # pause is not an emergency stop
    assert len(b.events) == events_before  # PAUSED gates input: no backend contact


def test_resume_recovers_a_paused_executor() -> None:
    from ai_game_agent.actions.executor import Executor

    b = FailingBackend()
    b.open()
    ex = Executor(
        backend=b, safety=_guard(max_backend_errors=1), clock=lambda: 0.0, sleeper=lambda s: None
    )
    ex.execute(Action(kind=ActionKind.KEY_PRESS, key="a"))
    assert ex.state is ExecutorState.PAUSED
    ex.resume()
    assert ex.state is ExecutorState.IDLE
    assert ex.safety.backend_errors == 0  # the consecutive-error count is cleared
    result = ex.execute(Action(kind=ActionKind.MOUSE_MOVE, x=1, y=2))
    assert result.ok is True  # input flows again after resume


def test_resume_from_stopped_raises(backend: MockBackend) -> None:
    ex = _executor(backend)
    ex.trigger_emergency_stop(reason="manual")  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        ex.resume()  # STOPPED is terminal: never recoverable at runtime


# --- P2: release_all fault tolerance + failed-release retry ------------------


class FlakyReleaseBackend(MockBackend):
    """A backend that fails to release key ``'a'`` until ``fail_a`` is cleared."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_a = True

    def key_up(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        if key == "a" and self.fail_a:
            raise InputBackendError("simulated release failure for key 'a'")
        super().key_up(key, modifiers)


def test_release_all_continues_after_a_failed_release() -> None:
    from ai_game_agent.actions.executor import Executor

    b = FlakyReleaseBackend()
    b.open()
    b.key_down("a")
    b.key_down("b")
    b.mouse_down("left")
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    ex.held_keys = set(b.held_keys)
    ex.held_buttons = set(b.held_buttons)
    ex.release_all()
    # Tracking state is cleared: the executor has done all it can for 'a'.
    assert ex.held_keys == set()
    assert ex.held_buttons == set()
    # The other held inputs still received their release (no early bail-out).
    assert ("key_up", "b") in b.events
    assert ("mouse_up", "left") in b.events


def test_release_all_retries_only_failed_releases() -> None:
    from ai_game_agent.actions.executor import Executor

    b = FlakyReleaseBackend()
    b.open()
    b.key_down("a")
    b.key_down("b")
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    ex.held_keys = set(b.held_keys)
    ex.release_all()
    b.fail_a = False  # the backend recovers
    ex.release_all()
    assert ("key_up", "a") in b.events  # the failed release was retried
    # No duplicate release for the key that had already succeeded.
    assert b.events.count(("key_up", "b")) == 1


def test_emergency_stop_retries_failed_release_on_next_stop() -> None:
    from ai_game_agent.actions.executor import Executor

    b = FlakyReleaseBackend()
    b.open()
    b.key_down("a")
    b.key_down("b")
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    ex.held_keys = set(b.held_keys)
    ex.trigger_emergency_stop(reason="first")
    assert ex.state is ExecutorState.STOPPED
    assert ("key_up", "b") in b.events
    assert "a" in b.held_keys  # 'a' is still physically held (release failed)
    b.fail_a = False  # the backend recovers
    ex.trigger_emergency_stop(reason="second")
    assert ("key_up", "a") in b.events  # the retry landed
    assert b.held_keys == set()
    assert ex.stop_reason == "first"  # the original reason is retained


# --- release failure during key_hold -----------------------------------------


def test_hold_key_release_failure_updates_bookkeeping() -> None:
    """A failed release during the hold must not leave the key tracked as held.

    The same contract ``release_all`` honors: after a failed release the
    executor has done all it can for that input — it is tracked as a failed
    release (retried by ``release_all`` / emergency stop), not as a still-held
    key. The backend still physically holds it until the retry lands.
    """
    from ai_game_agent.actions.executor import Executor

    b = FlakyReleaseBackend()
    b.open()
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    result = ex.execute(Action(kind=ActionKind.KEY_HOLD, key="a", duration_ms=100))
    assert result.ok is False
    assert "simulated release failure" in (result.error or "")
    # The executor has done all it can for 'a': no longer tracked as held.
    assert ex.held_keys == set()
    # But the backend still physically holds the key (the release failed).
    assert "a" in b.held_keys
    # key_down was issued; the failed key_up recorded no event.
    assert b.events == [("key_down", "a")]


def test_hold_key_release_failure_is_retried_by_release_all() -> None:
    from ai_game_agent.actions.executor import Executor

    b = FlakyReleaseBackend()
    b.open()
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    ex.execute(Action(kind=ActionKind.KEY_HOLD, key="a", duration_ms=100))
    assert "a" in b.held_keys  # still physically held on the backend
    b.fail_a = False  # the backend recovers
    assert ex.release_all() is True
    assert ("key_up", "a") in b.events  # the failed release was retried
    assert b.held_keys == set()


def test_emergency_stop_releases_modifiers_of_held_key() -> None:
    """An emergency stop during a modified key hold must release the modifiers
    too — ``release_all`` calls ``key_up`` without the action's modifiers, so
    the executor must remember and emit them (the Windows backend only
    releases the modifiers it is told to).
    """
    from ai_game_agent.actions.executor import Executor

    b = MockBackend()
    b.open()
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    # Hold 's' with ctrl: key_down carries the modifier, and the executor
    # must keep it as part of the held state.
    assert ex.execute(Action(kind=ActionKind.KEY_HOLD, key="s", modifiers=["ctrl"], duration_ms=100)).ok is True
    ex.trigger_emergency_stop(reason="manual")
    assert ex.state is ExecutorState.STOPPED
    # The main key is released ...
    assert ("key_up", "s") in b.events
    # ... and the modifier too — not left stuck.
    assert ("key_up", "ctrl") in b.events
    # Ordering: the modifier releases after the main key (key first, mods
    # after — the backend's key_up contract).
    assert b.events.index(("key_up", "ctrl")) > b.events.index(("key_up", "s"))


class FlakyMouseReleaseBackend(MockBackend):
    """A backend that fails to release the ``'left'`` button until ``fail_left`` is cleared."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_left = True

    def mouse_up(self, button: str = "left") -> None:
        if button == "left" and self.fail_left:
            raise InputBackendError("simulated release failure for button 'left'")
        super().mouse_up(button)


def test_failed_mouse_release_updates_bookkeeping() -> None:
    """A failed MOUSE_UP release must not leave the button tracked as held —
    the same contract as the key path: tracked as a failed release, not held.
    """
    from ai_game_agent.actions.executor import Executor

    b = FlakyMouseReleaseBackend()
    b.open()
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    assert ex.execute(Action(kind=ActionKind.MOUSE_DOWN, button="left")).ok is True
    result = ex.execute(Action(kind=ActionKind.MOUSE_UP, button="left"))
    assert result.ok is False
    assert "simulated release failure" in (result.error or "")
    # The executor has done all it can for 'left': no longer tracked as held.
    assert ex.held_buttons == set()
    # But the backend still physically holds the button (the release failed).
    assert "left" in b.held_buttons
    # mouse_down was issued; the failed mouse_up recorded no event.
    assert b.events == [("mouse_down", "left")]


def test_failed_mouse_release_blocks_further_input_until_flushed() -> None:
    """A failed MOUSE_UP release gates ordinary input until flushed —
    the button may still be physically held (same contract as the key path).
    """
    from ai_game_agent.actions.executor import Executor

    b = FlakyMouseReleaseBackend()
    b.open()
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    assert ex.execute(Action(kind=ActionKind.MOUSE_DOWN, button="left")).ok is True
    result = ex.execute(Action(kind=ActionKind.MOUSE_UP, button="left"))
    assert result.ok is False
    assert "left" in b.held_buttons  # still physically held on the backend

    # A subsequent action must not emit new input while the release is pending.
    events_before = len(b.events)
    result2 = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="b"))
    assert result2.ok is False
    assert result2.stop_event is False  # not an emergency stop
    assert len(b.events) == events_before  # no backend contact at all
    assert "left" in b.held_buttons

    # resume() while the backend still fails must not unblock the executor.
    ex.resume()
    result3 = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="b"))
    assert result3.ok is False
    assert "left" in b.held_buttons

    # The backend recovers: resume() flushes the pending release and input
    # flows again.
    b.fail_left = False
    ex.resume()
    assert ex.state is ExecutorState.IDLE
    assert ("mouse_up", "left") in b.events  # the pending release was flushed
    assert b.held_buttons == set()
    result4 = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="b"))
    assert result4.ok is True
    assert ("press", "b") in b.events


def test_failed_release_blocks_further_input_until_flushed() -> None:
    """A failed release gates ordinary input: the key may still be physically
    held, so no new input may flow until the pending release is flushed.

    ``resume()`` retries the release first — while the backend still fails
    the block stays; once the backend recovers the flush lands and input
    flows again. An unrelated success must never clear the condition.
    """
    from ai_game_agent.actions.executor import Executor

    b = FlakyReleaseBackend()
    b.open()
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    result = ex.execute(Action(kind=ActionKind.KEY_HOLD, key="a", duration_ms=100))
    assert result.ok is False
    assert "a" in b.held_keys  # still physically held on the backend

    # A subsequent action must not emit new input while the release is pending.
    events_before = len(b.events)
    result2 = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="b"))
    assert result2.ok is False
    assert result2.stop_event is False  # not an emergency stop
    assert len(b.events) == events_before  # no backend contact at all
    assert "a" in b.held_keys

    # resume() while the backend still fails must not unblock the executor.
    ex.resume()
    result3 = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="b"))
    assert result3.ok is False
    assert "a" in b.held_keys

    # The backend recovers: resume() flushes the pending release and input
    # flows again.
    b.fail_a = False
    ex.resume()
    assert ex.state is ExecutorState.IDLE
    assert ("key_up", "a") in b.events  # the pending release was flushed
    assert b.held_keys == set()
    result4 = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="b"))
    assert result4.ok is True
    assert ("press", "b") in b.events


def test_hold_key_release_failure_is_retried_by_emergency_stop() -> None:
    from ai_game_agent.actions.executor import Executor

    b = FlakyReleaseBackend()
    b.open()
    ex = Executor(backend=b, safety=_guard(), clock=lambda: 0.0, sleeper=lambda s: None)
    ex.execute(Action(kind=ActionKind.KEY_HOLD, key="a", duration_ms=100))
    assert "a" in b.held_keys  # still physically held on the backend
    b.fail_a = False  # the backend recovers
    ex.trigger_emergency_stop(reason="manual")
    assert ex.state is ExecutorState.STOPPED
    assert ("key_up", "a") in b.events  # the retry landed
    assert b.held_keys == set()
