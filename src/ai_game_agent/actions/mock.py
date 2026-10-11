"""Deterministic mock input backend (Phase 3, step 2).

The CI/test analog of Phase 1's capture ``MockBackend``. It performs **no OS
I/O**; instead it:

* records every event it is asked to perform, in order, into ``events``
  (e.g. ``("press", "s", ("ctrl", "shift"))``) so tests can assert the exact
  event sequence and order;
* tracks its own held keys and held buttons (``held_keys`` / ``held_buttons``)
  so tests can assert release-on-error and the emergency stop's release-all;
* enforces ``open()``/``close()`` lifecycle semantics: events before ``open()``
  raise :class:`InputBackendError` (fail fast, like the capture facade), and
  ``close()`` releases any held input so a half-open backend can't leave a key
  stuck.

This is what makes the whole decision/safety layer testable headless, exactly
as Phase 1's mock made capture testable.
"""

from __future__ import annotations

from ai_game_agent.actions.base import InputBackendError


class MockBackend:
    """Deterministic input backend that records events instead of emitting them.

    Conforms structurally to :class:`~ai_game_agent.actions.base.InputBackend`
    (no base class required). ``events`` is an ordered list of tuples; each
    tuple's first element names the event and the remainder are its arguments
    (``modifiers`` is kept as a tuple, not a list, so it is hashable and
    JSON-stable).
    """

    name = "mock"

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.held_keys: set[str] = set()
        self.held_buttons: set[str] = set()
        self.opened = False

    # --- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        """Mark the backend open. Idempotent."""
        self.opened = True

    def close(self) -> None:
        """Release resources and any held input. Idempotent.

        Releasing held input on close is the concrete realization of
        AGENTS.md's "release mouse buttons / release all held keyboard keys"
        safety rule: a half-open backend must never leave a key or button stuck.
        """
        self.release_all()
        self.opened = False

    def _ensure_open(self) -> None:
        if not self.opened:
            raise InputBackendError("mock backend not opened; call open() first")

    # --- keyboard -----------------------------------------------------------

    @staticmethod
    def _event(kind: str, key: str, modifiers: tuple[str, ...]) -> tuple[object, ...]:
        """Build a keyboard event, omitting ``modifiers`` when empty.

        The event log records only the *significant* arguments so an event is
        unambiguous: ``("press", "w")`` vs ``("press", "s", ("ctrl", "shift"))``.
        """
        mods = tuple(modifiers)
        return (kind, key) if not mods else (kind, key, mods)

    def key_down(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        self._ensure_open()
        self.events.append(self._event("key_down", key, modifiers))
        self.held_keys.add(key)

    def key_up(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        self._ensure_open()
        self.events.append(self._event("key_up", key, modifiers))
        self.held_keys.discard(key)

    def press(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        self._ensure_open()
        self.events.append(self._event("press", key, modifiers))

    # --- mouse --------------------------------------------------------------

    def mouse_move(self, x: int, y: int, relative: bool = False) -> None:
        self._ensure_open()
        self.events.append(("mouse_move", x, y) if not relative else ("mouse_move", x, y, True))

    def mouse_click(self, button: str = "left", clicks: int = 1) -> None:
        self._ensure_open()
        self.events.append(("mouse_click", button, clicks))

    def mouse_down(self, button: str = "left") -> None:
        self._ensure_open()
        self.events.append(("mouse_down", button))
        self.held_buttons.add(button)

    def mouse_up(self, button: str = "left") -> None:
        self._ensure_open()
        self.events.append(("mouse_up", button))
        self.held_buttons.discard(button)

    # --- release-all --------------------------------------------------------

    def release_all(self) -> None:
        """Release every held key and button, recording the release events.

        Idempotent: once nothing is held, a further call is a no-op (no events
        are emitted) so the emergency stop can call it repeatedly and the
        mock's event log stays stable.
        """
        for key in sorted(self.held_keys):
            self.events.append(("key_up", key))
        for button in sorted(self.held_buttons):
            self.events.append(("mouse_up", button))
        self.held_keys.clear()
        self.held_buttons.clear()
