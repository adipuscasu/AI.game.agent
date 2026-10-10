"""Tests for ai_game_agent.actions.hotkey (Phase 3, step 7).

TDD: red first (module missing), then green. The plan (build-order step 7):
"``HotkeyListener`` protocol + fake-triggerable listener (mechanism) —
trigger → ``EmergencyStop`` proven headless."

The physical F12 binding cannot be tested headless — that is the
Windows-side pynput listener (step 8). What *can* be tested, and must be, is
the **mechanism**: a hotkey trigger requesting the emergency stop exactly
once (idempotent), releasing held input mid-hold, and blocking any
subsequent dispatch. The acceptance criteria signed off for this step:

1. **Idempotency** — repeated ``trigger()`` calls request the emergency stop
   only once.
2. **Mid-hold safety** — triggering the stop while a key is held releases it,
   and the executor dispatches no subsequent actions.
3. **Regression gate** — ``uv run pytest`` passes all existing and new tests
   before the step commit (checked in CI, asserted here by the suite itself).

No ``pynput`` import anywhere in this file (nor in ``hotkey.py``): the
listener contract is dependency-free so the mechanism stays testable in a
bare venv, exactly like the rest of steps 1–7.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ai_game_agent.actions import (
    ActionQueue,
    ActionRequest,
    SafetyGuard,
)
from ai_game_agent.actions.action import Action, ActionKind
from ai_game_agent.actions.hotkey import FakeHotkeyListener, HotkeyListener
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.actions.modes import OperationMode


@pytest.fixture
def backend() -> MockBackend:
    b = MockBackend()
    b.open()
    return b


def _guard() -> SafetyGuard:
    return SafetyGuard(mode=OperationMode.AUTONOMOUS)


def _executor(backend: MockBackend) -> object:
    """A wired Executor: backend + guard + no-op clock/sleeper (headless)."""
    from ai_game_agent.actions.executor import Executor

    return Executor(
        backend=backend,
        safety=_guard(),
        clock=lambda: 0.0,
        sleeper=lambda secs: None,
    )


# --- Contract ----------------------------------------------------------------


def test_fake_listener_conforms_to_protocol() -> None:
    listener = FakeHotkeyListener(on_trigger=lambda: None)
    assert isinstance(listener, HotkeyListener)


def test_default_triggered_key_is_f12() -> None:
    """The config default is ``safety.emergency_stop_keys: [F12]``."""
    requests: list[str] = []
    listener = FakeHotkeyListener(on_trigger=lambda: requests.append("stop"))
    assert listener.triggered_keys == ("f12",)
    listener.arm()
    listener.trigger("f12")
    assert requests == ["stop"]


def test_key_matching_is_case_insensitive() -> None:
    """``safety.emergency_stop_keys`` ships ``F12`` (uppercase); key systems
    (pynput, the Action contract) speak lowercase. Both spellings must fire."""
    requests: list[str] = []
    listener = FakeHotkeyListener(
        on_trigger=lambda: requests.append("stop"), triggered_keys=("F12",)
    )
    listener.arm()
    listener.trigger("f12")
    listener.arm()  # fresh cycle (see test_rearm_allows_a_new_stop_request_cycle)
    listener.trigger("F12")
    assert requests == ["stop", "stop"]


def test_trigger_ignores_keys_that_are_not_stop_keys() -> None:
    """A game key (``a``/``w``) must never be mistaken for the stop key."""
    requests: list[str] = []
    listener = FakeHotkeyListener(
        on_trigger=lambda: requests.append("stop"), triggered_keys=("F12",)
    )
    listener.arm()
    listener.trigger("a")
    listener.trigger("w")
    assert requests == []
    # The listener is still armed and the stop still fires for the real key.
    listener.trigger("f12")
    assert requests == ["stop"]


# --- AC1: idempotency ---------------------------------------------------------


def test_repeated_trigger_requests_stop_only_once() -> None:
    """AC1: repeated ``trigger()`` calls request the emergency stop once.

    Double delivery of a key press (OS auto-repeat, two threads, a flaky
    backend) must not request the stop more than once.
    """
    requests: list[str] = []
    listener = FakeHotkeyListener(
        on_trigger=lambda: requests.append("stop"), triggered_keys=("F12",)
    )
    listener.arm()
    listener.trigger("f12")
    listener.trigger("f12")
    listener.trigger("F12")
    assert requests == ["stop"]
    assert listener.trigger_count == 1


def test_repeated_trigger_does_not_double_release(backend: MockBackend) -> None:
    """The mechanism end to end: the second trigger must not re-release.

    A duplicate ``key_up`` for an already-released key is harmless in the
    mock but would be a spurious OS event on the real backend — the
    "request only once" contract is what keeps the event log clean.
    """
    ex = _executor(backend)
    listener = FakeHotkeyListener(
        on_trigger=lambda: ex.trigger_emergency_stop(reason="f12"),  # type: ignore[union-attr]
        triggered_keys=("F12",),
    )
    listener.arm()
    backend.key_down("w")
    ex.held_keys = set(backend.held_keys)  # type: ignore[union-attr]

    listener.trigger("f12")
    first = list(backend.events)
    assert ex.stopped is True  # type: ignore[union-attr]
    assert backend.held_keys == set()

    listener.trigger("f12")  # double delivery
    assert backend.events == first  # no new release events
    assert ex.stopped is True  # type: ignore[union-attr]


def test_trigger_while_disarmed_does_not_request_stop() -> None:
    """No arm, no stop: a stale trigger after ``disarm()`` (or before the
    first ``arm()``) must not fire. The listener is off until armed."""
    requests: list[str] = []
    listener = FakeHotkeyListener(
        on_trigger=lambda: requests.append("stop"), triggered_keys=("F12",)
    )
    listener.trigger("f12")  # never armed
    assert requests == []

    listener.arm()
    listener.trigger("f12")
    listener.disarm()
    assert requests == ["stop"]

    listener.trigger("f12")  # after disarm
    assert requests == ["stop"]


def test_rearm_allows_a_new_stop_request_cycle() -> None:
    """``arm()`` starts a new session: a fresh executor may be bound to a
    fresh listener arm cycle and request its own stop. Within one cycle the
    stop is still requested exactly once (AC1)."""
    requests: list[str] = []
    listener = FakeHotkeyListener(
        on_trigger=lambda: requests.append("stop"), triggered_keys=("F12",)
    )
    listener.arm()
    listener.trigger("f12")
    listener.arm()  # new session
    listener.trigger("f12")
    assert requests == ["stop", "stop"]
    assert listener.trigger_count == 2


# --- AC2: mid-hold safety ------------------------------------------------------


def test_trigger_mid_hold_releases_held_key(backend: MockBackend) -> None:
    """AC2 (part 1): the stop mid-hold releases the held key.

    The concrete AGENTS.md requirement: emergency stop must "release all
    held keyboard keys" — and it must do so even mid-hold, not at the hold's
    natural end.
    """
    ex = _executor(backend)
    listener = FakeHotkeyListener(
        on_trigger=lambda: ex.trigger_emergency_stop(reason="f12"),  # type: ignore[union-attr]
        triggered_keys=("F12",),
    )
    listener.arm()
    backend.key_down("w")
    ex.held_keys = set(backend.held_keys)  # type: ignore[union-attr]
    backend.mouse_down("left")
    ex.held_buttons = set(backend.held_buttons)  # type: ignore[union-attr]

    listener.trigger("F12")

    assert ex.stopped is True  # type: ignore[union-attr]
    assert backend.held_keys == set()
    assert backend.held_buttons == set()
    # The release was emitted by the stop, after the key_down.
    assert ("key_up", "w") in backend.events
    assert ("mouse_up", "left") in backend.events
    # The stop is recorded with a human-readable reason.
    assert ex.stop_reason == "f12"  # type: ignore[union-attr]


def test_trigger_mid_hold_blocks_subsequent_dispatch(backend: MockBackend) -> None:
    """AC2 (part 2): after the stop, the executor dispatches nothing.

    A queued action is dropped, and a later ``execute`` is a blocked no-op
    with ``stop_event=True`` — the backend sees no new input.
    """
    ex = _executor(backend)
    listener = FakeHotkeyListener(
        on_trigger=lambda: ex.trigger_emergency_stop(reason="f12"),  # type: ignore[union-attr]
        triggered_keys=("F12",),
    )
    listener.arm()

    # Pending work the stop must cancel.
    q: ActionQueue = ex.queue  # type: ignore[union-attr]
    q.add(
        ActionRequest(
            action=Action(kind=ActionKind.KEY_PRESS, key="x"),
            requested_at=datetime(2026, 10, 10, 16, 0),
        )
    )
    assert q.pending_count() == 1

    # Hold a key so the stop has something to release, then fire it.
    backend.key_down("w")
    ex.held_keys = set(backend.held_keys)  # type: ignore[union-attr]
    listener.trigger("f12")

    # 1. queued work cancelled
    assert q.pending_count() == 0
    # 2. held key released
    assert backend.held_keys == set()
    # 3. further dispatch is blocked: failed result, stop_event set, and the
    #    backend receives nothing (event log frozen at the stop).
    before = list(backend.events)
    result = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="q"))  # type: ignore[union-attr]
    assert result.ok is False
    assert result.stop_event is True
    assert backend.events == before
