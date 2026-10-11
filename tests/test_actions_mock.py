"""Step 2: ``MockBackend`` — the deterministic, CI-safe input backend.

The mock performs no OS I/O. It records every event it is asked to perform, in
order, and tracks its own held keys/buttons so tests can assert release-on-error
and the emergency stop's release-all. This mirrors Phase 1's capture
``MockBackend`` (the "test a platform I/O abstraction headless" pattern).

``open``/``close`` are lifecycle markers: events emitted before ``open()``
raise ``InputBackendError`` (fail fast, like the capture facade).
"""

from __future__ import annotations

import pytest

from ai_game_agent.actions.base import InputBackendError
from ai_game_agent.actions.mock import MockBackend


def _opened() -> MockBackend:
    m = MockBackend()
    m.open()
    return m


def test_open_close_marks_lifecycle():
    m = MockBackend()
    assert m.opened is False
    m.open()
    assert m.opened is True
    m.open()  # idempotent
    assert m.opened is True
    m.close()
    assert m.opened is False
    m.close()  # idempotent
    assert m.opened is False


def test_key_press_records_down_then_up():
    m = _opened()
    m.press("w")
    assert m.events == [("press", "w")]
    m.key_down("a")
    m.key_up("a")
    assert m.events == [("press", "w"), ("key_down", "a"), ("key_up", "a")]


def test_modifiers_are_recorded_as_tuple():
    m = _opened()
    m.press("s", ("ctrl", "shift"))
    assert m.events == [("press", "s", ("ctrl", "shift"))]


def test_mouse_click_move_down_up_events():
    m = _opened()
    m.mouse_move(100, 200)
    m.mouse_click("left", 2)
    m.mouse_down("right")
    m.mouse_up("right")
    assert m.events == [
        ("mouse_move", 100, 200),
        ("mouse_click", "left", 2),
        ("mouse_down", "right"),
        ("mouse_up", "right"),
    ]


def test_held_keys_tracked_and_released_on_key_up():
    m = _opened()
    m.key_down("w")
    assert m.held_keys == {"w"}
    m.key_up("w")
    assert m.held_keys == set()


def test_held_buttons_tracked_and_released_on_mouse_up():
    m = _opened()
    m.mouse_down("left")
    assert m.held_buttons == {"left"}
    m.mouse_up("left")
    assert m.held_buttons == set()


def test_key_up_of_unknown_key_is_a_noop_on_state():
    # key_up is a state release, not a "must have been held" assertion.
    m = _opened()
    m.key_up("w")  # nothing held
    assert m.held_keys == set()
    assert ("key_up", "w") in m.events  # still recorded (event log is total)


def test_release_all_releases_everything_and_records():
    m = _opened()
    m.key_down("w")
    m.key_down("a")
    m.mouse_down("left")
    m.release_all()
    assert m.held_keys == set()
    assert m.held_buttons == set()
    assert ("key_up", "w") in m.events
    assert ("key_up", "a") in m.events
    assert ("mouse_up", "left") in m.events


def test_release_all_is_idempotent():
    m = _opened()
    m.key_down("w")
    m.release_all()
    n = len(m.events)
    m.release_all()
    assert len(m.events) == n  # no held input left -> nothing to release


def test_events_before_open_raise_input_backend_error():
    m = MockBackend()
    with pytest.raises(InputBackendError):
        m.press("w")


def test_close_releases_held_input():
    # A leaked held key is the exact failure AGENTS.md calls out: close() must
    # release whatever is held so a half-open backend can't leave a key stuck.
    m = _opened()
    m.key_down("w")
    m.mouse_down("left")
    m.close()
    assert m.held_keys == set()
    assert m.held_buttons == set()
    assert m.opened is False
