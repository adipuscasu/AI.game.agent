"""Windows input backend (Phase 3, step 8): real mouse/keyboard I/O via pynput.

The :class:`InputBackend` adapter for the ``windows`` backend — the only
implementation that performs real OS input. As with the capture layer's mss
adapter (``capture.py``), pynput is **imported only when the backend is
opened**, never at module import time, so a bare venv without the ``input``
extra can import this module and construct :class:`WindowsBackend` cleanly and
only hits a missing-dependency error when it actually tries to open (the
contract ``tests/test_actions_bare.py`` asserts).

Design rules (the AGENTS.md safety story, made concrete):

* **Release-on-exception by construction.** A modifier combo is released in
  reverse if any part of the press fails, so a partial failure never leaves a
  modifier stuck. A failed ``mouse_down`` never marks the button held.
  ``close()``/``release_all()`` drain every held key and button.
* **The emergency-stop hotkey is an AI-independent decision object** owned by
  the backend (the step-7 mechanism), exposed as ``_hotkey_listener``. It is
  armed on :meth:`open`, disarmed on :meth:`close`; fires ``on_stop`` **at
  most once per arm cycle**; is inert once disarmed; matches the stop key
  case-insensitively. The pynput listener is the event *source*; the decider is
  the *decision* — so the mechanism is testable headless with a fake pynput.

Dependency rule: stdlib + this project's types at import time; ``pynput`` is
imported lazily inside the default factory only.
"""

from __future__ import annotations

from collections.abc import Callable

from ai_game_agent.actions.base import InputBackendError

__all__ = ["WindowsBackend"]


def _default_pynput_factory() -> object:
    """Lazily import real pynput (only when the backend is opened).

    Module-level so it can be the constructor default without running the
    import at class-definition time. ``create_backend`` already guards
    availability before building this backend, so in practice this import
    succeeds; direct construction in a bare venv defers the failure to
    :meth:`WindowsBackend.open`.
    """
    import pynput  # lazy: the 'input' extra is required only here

    return pynput


class _HotkeyDecider:
    """Armed/disarmed emergency-stop decision object owned by the backend.

    The step-7 mechanism (:class:`~ai_game_agent.actions.hotkey.HotkeyListener`)
    realized as the backend's internal decision object (``_hotkey_listener``).
    :meth:`handle` is the routing entry the pynput listener calls.

    * armed on ``open`` — a stop-request cycle is open;
    * disarmed on ``close`` — a stale key event must not fire a stop;
    * fires ``on_stop`` **at most once per arm cycle**, regardless of redelivery;
    * matches the stop key case-insensitively (config ships ``F12``; key
      systems speak ``f12``).
    """

    def __init__(
        self,
        on_stop: Callable[[], None] | None,
        stop_key: str | tuple[str, ...],
    ) -> None:
        self._on_stop = on_stop if on_stop is not None else (lambda: None)
        if isinstance(stop_key, (tuple, list, set)):
            self._stop_keys = {str(k).lower() for k in stop_key}
        else:
            self._stop_keys = {str(stop_key).lower()}
        self._armed = False
        self._fired = False

    @property
    def armed(self) -> bool:
        """Whether a stop-request cycle is currently open."""
        return self._armed

    def arm(self) -> None:
        """Open (or reopen) a stop-request cycle. Idempotent; re-arm resets."""
        self._armed = True
        self._fired = False

    def disarm(self) -> None:
        """Close the cycle; :meth:`handle` is inert until the next :meth:`arm`."""
        self._armed = False

    def handle(self, key: object) -> None:
        """One key event from the input layer. Fire ``on_stop`` if armed + match.

        At most once per arm cycle. ``key`` may be a pynput ``Key`` (has
        ``.name``), a pynput ``Character`` (stringifies to the char), a bare
        string, or a fake event exposing ``.name``. A non-normalizable key
        simply does not match and never raises (a stray key event must not take
        the handler down).
        """
        if not self._armed:
            return
        if isinstance(key, str):
            name = key
        elif isinstance(getattr(key, "name", None), str):
            name = key.name
        else:
            name = str(key)
        if name.lower() not in self._stop_keys:
            return
        if self._fired:
            return
        self._fired = True
        self._on_stop()


class WindowsBackend:
    """pynput-backed :class:`InputBackend` (real mouse + keyboard I/O).

    Conforms structurally to
    :class:`~ai_game_agent.actions.base.InputBackend` (no base class, mirroring
    :class:`MssBackend`). pynput is injected via ``pynput_factory`` for headless
    testing and is only imported (by the default factory) at :meth:`open` time,
    so a bare venv can import this module and construct the backend cleanly.

    Parameters
    ----------
    pynput_factory:
        Zero-arg callable returning the pynput package (or a fake exposing
        ``.keyboard`` / ``.mouse``). Defaults to importing real pynput.
    on_stop:
        Emergency-stop hook invoked by :attr:`_hotkey_listener` when the stop
        key is pressed while armed. In the real CLI this is
        ``Executor.trigger_emergency_stop``. Optional (no-op when omitted).
    stop_key:
        Emergency-stop key (``"f12"`` by default), or a tuple of keys.
        Matched case-insensitively.
    """

    name = "windows"

    def __init__(
        self,
        pynput_factory: Callable[[], object] | None = None,
        on_stop: Callable[[], None] | None = None,
        stop_key: str | tuple[str, ...] = "f12",
    ) -> None:
        self._factory = (
            pynput_factory if pynput_factory is not None else _default_pynput_factory
        )
        self._hotkey_listener = _HotkeyDecider(on_stop, stop_key)
        # Held-state source of truth (mirrors the mock/executor contract).
        self.held_keys: set[str] = set()
        self.held_buttons: set[str] = set()
        # Populated in open(); None until then.
        self._kb: object | None = None
        self._mouse: object | None = None
        self._key: object | None = None
        self._button: object | None = None
        self._listener: object | None = None
        self._opened = False

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        """Allocate pynput controllers, start the hotkey listener, arm the stop.

        Idempotent. Importing real pynput (default factory) happens here, so a
        bare venv without the ``input`` extra fails at ``open`` with a clean
        error rather than at import time.
        """
        if self._opened:
            return
        try:
            pyn = self._factory()
        except ImportError as exc:
            raise InputBackendError(
                "the 'windows' input backend needs the 'pynput' library; "
                "install the 'input' extra with 'uv sync --extra input' "
                "(or 'pip install pynput'), or set input.backend to 'mock'"
            ) from exc

        kb_mod = pyn.keyboard
        mouse_mod = pyn.mouse
        self._key = getattr(kb_mod, "Key", None)
        self._button = getattr(mouse_mod, "Button", None)
        # Prefer controllers the factory already provided (the test fake's
        # ``bind``); otherwise instantiate them through the pynput package API.
        # Real pynput exposes no pre-built controllers, so this falls through
        # to the documented ``keyboard.Controller()`` / ``mouse.Controller()`` /
        # ``keyboard.Listener(...)`` construction.
        self._kb = pyn.kb if getattr(pyn, "kb", None) is not None else kb_mod.Controller()
        self._mouse = (
            pyn.mouse_ctrl
            if getattr(pyn, "mouse_ctrl", None) is not None
            else mouse_mod.Controller()
        )
        self._listener = (
            pyn.listener
            if getattr(pyn, "listener", None) is not None
            else kb_mod.Listener(on_press=self._on_key_press)
        )
        self._listener.start()
        self._hotkey_listener.arm()
        # Test-injection hook: a fake pynput may expose ``bind`` so the test can
        # grab the live controllers/listener for assertions. Real pynput has no
        # such method, so this is a no-op in production.
        binder = getattr(pyn, "bind", None)
        if callable(binder):
            binder(self._kb, self._mouse, self._listener)
        self._opened = True

    def close(self) -> None:
        """Disarm the hotkey, stop the listener, and release held input.

        Idempotent. Releasing held input on close is the concrete realization of
        AGENTS.md's "release all held keyboard keys / release mouse buttons": a
        half-open backend must never leave a key or button stuck.
        """
        if not self._opened:
            return
        self._hotkey_listener.disarm()
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
        self.release_all()
        self._opened = False

    def release_all(self) -> None:
        """Release every held key and button. Idempotent; safe after ``close``.

        Drains :attr:`held_keys` and :attr:`held_buttons`. Open-tolerant, so
        the emergency stop (and repeated emergency stops) can call it at any
        point.
        """
        if self._kb is not None:
            for key in sorted(self.held_keys):
                try:
                    self._kb.up(self._key_obj(key))
                except Exception:
                    pass
        if self._mouse is not None:
            for button in sorted(self.held_buttons):
                try:
                    self._mouse.up(self._button_obj(button))
                except Exception:
                    pass
        self.held_keys.clear()
        self.held_buttons.clear()

    # -- hotkey routing -----------------------------------------------------

    def _on_key_press(self, key: object) -> None:
        """pynput listener callback: route every key into the stop decider."""
        self._hotkey_listener.handle(key)

    # -- key/button mapping -------------------------------------------------

    def _key_obj(self, name: str) -> object:
        """Map a canonical key name to a pynput ``Key`` (or the bare char)."""
        Key = self._key
        if Key is not None and hasattr(Key, name):
            return getattr(Key, name)
        return name  # bare character (a, w, s, ...) — pynput accepts a char

    def _button_obj(self, name: str) -> object:
        """Map a canonical button name to a pynput ``Button`` (or the name)."""
        Button = self._button
        if Button is not None and hasattr(Button, name):
            return getattr(Button, name)
        return name

    # -- keyboard -----------------------------------------------------------

    def _ensure_open(self) -> None:
        if not self._opened:
            raise InputBackendError("windows backend not opened; call open() first")

    def key_down(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        """Press ``modifiers`` (in order) then ``key``; hold everything down.

        Release-on-exception: if the key press fails after some modifiers are
        already down, the already-pressed modifiers are released (reverse
        order) so a partial failure never leaves a modifier stuck.
        """
        self._ensure_open()
        mods = tuple(modifiers)
        pressed: list[str] = []
        try:
            for m in mods:
                self._kb.press(self._key_obj(m))
                pressed.append(m)
            self._kb.press(self._key_obj(key))
        except Exception:
            for m in reversed(pressed):
                try:
                    self._kb.up(self._key_obj(m))
                except Exception:
                    pass
            raise
        self.held_keys.add(key)

    def key_up(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        """Release ``key`` first, then ``modifiers`` (reverse order)."""
        self._ensure_open()
        self._kb.up(self._key_obj(key))
        self.held_keys.discard(key)
        for m in reversed(tuple(modifiers)):
            self._kb.up(self._key_obj(m))

    def press(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        """A complete press-and-release of ``key`` (down then up)."""
        self._ensure_open()
        self.key_down(key, modifiers)
        self.key_up(key, modifiers)

    # -- mouse --------------------------------------------------------------

    def mouse_move(self, x: int, y: int, relative: bool = False) -> None:
        """Move the pointer to ``(x, y)`` (absolute), or by the delta if
        relative."""
        self._ensure_open()
        if relative:
            self._mouse.relative_move(int(x), int(y))
        else:
            self._mouse.move(int(x), int(y))

    def mouse_click(self, button: str = "left", clicks: int = 1) -> None:
        """Click ``button`` ``clicks`` time(s)."""
        self._ensure_open()
        self._mouse.click(self._button_obj(button), int(clicks))

    def mouse_down(self, button: str = "left") -> None:
        """Press and hold ``button`` down.

        A backend failure here does **not** mark the button held — only a
        successful press is tracked, so ``release_all`` can't "release" a
        button the OS never saw a matching down for.
        """
        self._ensure_open()
        self._mouse.down(self._button_obj(button))
        self.held_buttons.add(button)

    def mouse_up(self, button: str = "left") -> None:
        """Release ``button``."""
        self._ensure_open()
        self._mouse.up(self._button_obj(button))
        self.held_buttons.discard(button)

    def mouse_scroll(self, ticks: int, x: int | None = None, y: int | None = None) -> None:
        """Scroll by ``ticks`` (positive = up). Present for the executor's
        optional ``MOUSE_SCROLL`` path; not part of the InputBackend protocol."""
        self._ensure_open()
        self._mouse.scroll(0, int(ticks), x, y)
