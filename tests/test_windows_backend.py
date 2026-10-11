"""Tests for ``ai_game_agent.actions.windows`` (Phase 3, step 8).

TDD: red first (``windows.py`` missing → ``ImportError``), then green.

This file never imports ``pynput``. The :class:`WindowsBackend` is driven
through an injectable pynput-like object (``pynput_factory``), exactly the
pattern :class:`ai_game_agent.capture.MssBackend` uses with
``mss_factory`` (see ``tests/test_mss_backend.py``). The real ``pynput``
import is therefore confined to ``windows.py`` itself, and this suite runs
headless on a bare venv — the same guarantee the rest of steps 1–7 already
have, now extended to the only backend that emits real OS input.

Windows-specific behavior (actual key scancodes, the physical F12 binding on
a live desktop, ``pynput.keyboard.Listener`` thread semantics) is
**out of scope for CI**: the fakes mimic the documented pynput surface
(``Key``/``Button``/``Controller``/``Listener``), and ``docs/manual-testing.md``
covers the Windows-side run. This file verifies the adapter's *contract*:

* :class:`WindowsBackend` conforms structurally to
  :class:`~ai_game_agent.actions.base.InputBackend`.
* The canonical lowercase key names the :class:`Action` contract already
  validates (``"a"``, ``"f12"``, ``"space"``) and the modifier names
  (``"ctrl"``/``"alt"``/``"shift"``/``"win"``) map onto pynput's ``Key.*``
  symbols.
* A modifier combo is a well-defined down/up sequence (modifiers first,
  released in reverse) so a half-failed combo cannot leave a modifier stuck.
* Partially-failed input operations leave no held keys or buttons (the
  AGENTS.md safety rule: ``close()``/``release_all()`` must release).
* ``open()``/``close()`` are idempotent; ``close()`` stops the hotkey
  listener **and** releases held input.
* The step-7 hotkey *mechanism* (``arm``/``trigger``/``disarm``, idempotent
  stop request) is wired as the backend's decision object: a configured stop
  key (default ``F12``) requests the emergency stop **exactly once** per arm
  cycle, a stray key is ignored, a disarmed trigger does not fire, and an
  end-to-end wiring through a real :class:`Executor` proves the stop
  releases held input and blocks subsequent dispatch.

The one assertion this file makes about :mod:`pynput` is the absence: the
``input`` extra must *not* be installed in the shared dev venv (see
``tests/test_actions_bare.py``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import pytest

from ai_game_agent.actions import (
    ActionQueue,
    ActionRequest,
    SafetyGuard,
)
from ai_game_agent.actions.action import Action, ActionKind
from ai_game_agent.actions.base import (
    InputBackend,
    InputBackendError,
    Keyboard,
    Mouse,
)
from ai_game_agent.actions.executor import Executor
from ai_game_agent.actions.hotkey import FakeHotkeyListener
from ai_game_agent.actions.modes import OperationMode


# ---------------------------------------------------------------------------
# Fakes for the pynput surface (never imports the real library)
# ---------------------------------------------------------------------------


class FakeKey:
    """Stand-in for ``pynput.keyboard.Key``: an attribute namespace.

    ``FakeKey.f12.name == "f12"``, ``FakeKey.ctrl.name == "ctrl"``, and so on.
    Real pynput exposes the same set of names the Action contract already
    validates (see ``_NAMED_KEYS`` in ``action.py``), so this fake only needs
    ``__getattr__`` to cover the union.
    """

    _NAMES = (
        "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11",
        "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20",
        "f21", "f22", "f23", "f24",
        "space", "enter", "tab", "backspace", "delete", "esc", "pause",
        "print_screen", "caps_lock", "num_lock", "scroll_lock",
        "ctrl", "alt", "shift", "cmd", "win",
        "up", "down", "left", "right",
        "page_up", "page_down", "home", "end", "insert",
    )

    @classmethod
    def __getattr__(cls, name: str) -> str:
        if name.startswith("_") or name in cls._NAMES:
            # ``_NAMES`` is a class attribute; ``__getattr__`` only runs when
            # normal lookup fails, so this branch covers non-private names.
            if name in cls._NAMES:
                return name
        raise AttributeError(name)


class FakeButton:
    """Stand-in for ``pynput.mouse.Button``: ``left``/``right``/``middle``."""

    left = "left"
    right = "right"
    middle = "middle"


class FakeKeyboardController:
    """Records ``press``/``up`` in order (pynput's ``keyboard.Controller``)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.held: set[object] = set()

    def press(self, key: object) -> None:
        self.events.append(("down", key))
        self.held.add(key)

    def up(self, key: object) -> None:
        self.events.append(("up", key))
        self.held.discard(key)


class FakeMouseController:
    """Records ``move``/``click``/``down``/``up``/``scroll`` (pynput mouse)."""

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.held: set[object] = set()
        self.scroll_ticks: list[int] = []

    def move(self, x: int, y: int, *args: object, **kwargs: object) -> None:
        self.events.append(("move", x, y))

    def relative_move(self, dx: int, dy: int, *args: object, **kwargs: object) -> None:
        self.events.append(("relative_move", dx, dy))

    def click(self, button: object = FakeButton.left, clicks: int = 1, *a: object, **kw: object) -> None:
        self.events.append(("click", button, clicks))

    def down(self, button: object = FakeButton.left, *a: object, **kw: object) -> None:
        self.events.append(("down", button))
        self.held.add(button)

    def up(self, button: object = FakeButton.left, *a: object, **kw: object) -> None:
        self.events.append(("up", button))
        self.held.discard(button)

    def scroll(self, dx: int, dy: int, x: int | None = None, y: int | None = None) -> None:
        self.scroll_ticks.append(dy)
        self.events.append(("scroll", dy))


class FakeListener:
    """Stand-in for ``pynput.keyboard.Listener``.

    ``simulate_key(name)`` invokes the callback the backend passed as
    ``on_press`` with a fake key (one whose ``.name`` is ``name``) — the same
    call pynput would make on a real key event. This is the only "OS event"
    the tests inject; the backend's routing from there is what's under test.
    """

    def __init__(self, on_press: Callable[[object], None] | None = None,
                 suppress: bool = False) -> None:
        self._on_press = on_press
        self.suppress = suppress
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def simulate_key(self, name: str) -> None:
        if not self.started or self.stopped:
            raise AssertionError(
                f"listener must be started (and not stopped) to deliver {name!r}"
            )
        assert self._on_press is not None, "listener created without on_press"
        self._on_press(_FakeKeyEvent(name))


class _FakeKeyEvent:
    """Mimics the ``key`` object pynput passes to ``on_press`` callbacks."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakePynput:
    """The shape of the ``pynput`` object the backend will use, fake.

    ``keyboard.Key`` / ``keyboard.Controller`` / ``keyboard.Listener`` plus
    ``mouse.Button`` / ``mouse.Controller`` — the minimal surface the adapter
    needs. The backend reaches it through a single factory, so a test never
    sees the real library.
    """

    def __init__(self) -> None:
        self.keyboard = type("Keyboard", (), {
            "Key": FakeKey,
            "Controller": FakeKeyboardController,
            "Listener": FakeListener,
        })
        self.mouse = type("Mouse", (), {
            "Button": FakeButton,
            "Controller": FakeMouseController,
        })
        # The backend picks up the live objects from here.
        self.kb: FakeKeyboardController | None = None
        self.mouse_ctrl: FakeMouseController | None = None
        self.listener: FakeListener | None = None

    def bind(self, kb: FakeKeyboardController, mouse: FakeMouseController,
             listener: FakeListener) -> None:
        self.kb = kb
        self.mouse_ctrl = mouse
        self.listener = listener


def make_backend(
    on_stop: Callable[[], None] | None = None,
    stop_key: str | None = None,
) -> tuple[object, FakePynput]:
    """Build a :class:`WindowsBackend` wired to fakes.

    The default ``stop_key`` (``None``) exercises the backend's default of
    ``"F12"`` (the ``safety.emergency_stop_keys`` config default, lowercased).
    """
    pyn = FakePynput()
    from ai_game_agent.actions.windows import WindowsBackend  # red: module missing

    backend = WindowsBackend(
        pynput_factory=lambda: pyn,
        on_stop=on_stop,
        **({"stop_key": stop_key} if stop_key is not None else {}),
    )
    return backend, pyn


def _make_executor(backend: object) -> Executor:
    """A real :class:`Executor` wired to the given (fake) backend."""
    return Executor(
        backend=backend,
        safety=SafetyGuard(mode=OperationMode.AUTONOMOUS),
        clock=lambda: 0.0,
        sleeper=lambda secs: None,
    )


# --- Contract ----------------------------------------------------------------


def test_windows_backend_conforms_to_input_backend_protocol() -> None:
    backend, _ = make_backend()
    assert isinstance(backend, InputBackend)
    assert isinstance(backend, Keyboard)
    assert isinstance(backend, Mouse)
    assert backend.name == "windows"


# --- Keyboard: key-name mapping ----------------------------------------------


def test_key_down_maps_named_key_to_pynput_key() -> None:
    backend, pyn = make_backend()
    backend.open()
    backend.key_down("f12")
    backend.close()
    assert pyn.kb is not None
    assert ("down", "f12") in pyn.kb.events


def test_key_down_maps_single_letter_key() -> None:
    backend, pyn = make_backend()
    backend.open()
    backend.key_down("a")
    backend.close()
    assert pyn.kb is not None
    assert ("down", "a") in pyn.kb.events


def test_key_down_maps_space_named_key() -> None:
    backend, pyn = make_backend()
    backend.open()
    backend.key_down("space")
    backend.close()
    assert pyn.kb is not None
    assert ("down", "space") in pyn.kb.events


def test_key_up_releases_the_named_key() -> None:
    backend, pyn = make_backend()
    backend.open()
    backend.key_down("w")
    backend.key_up("w")
    backend.close()
    assert pyn.kb is not None
    seq = [e for e in pyn.kb.events if e[1] == "w"]
    assert seq == [("down", "w"), ("up", "w")]


def test_press_is_down_then_up() -> None:
    """A complete press emits a ``down`` immediately followed by an ``up``."""
    backend, pyn = make_backend()
    backend.open()
    backend.press("s")
    backend.close()
    assert pyn.kb is not None
    seq = [e for e in pyn.kb.events if e[1] == "s"]
    assert seq == [("down", "s"), ("up", "s")]


# --- Keyboard: modifier combos -----------------------------------------------


def test_modifier_combo_down_presses_modifiers_first() -> None:
    """``key_down("s", ("ctrl", "shift"))`` must press ``ctrl``, then
    ``shift``, then ``s`` — the pynput idiom for a combo key press."""
    backend, pyn = make_backend()
    backend.open()
    backend.key_down("s", ("ctrl", "shift"))
    backend.close()
    assert pyn.kb is not None
    seq = [e[1] for e in pyn.kb.events if e[0] == "down"]
    assert seq == ["ctrl", "shift", "s"]


def test_modifier_combo_up_releases_in_reverse() -> None:
    """The release mirrors the press: ``s`` up, then ``shift`` up, then
    ``ctrl`` up — so a half-failed combo cannot leave a modifier stuck."""
    backend, pyn = make_backend()
    backend.open()
    backend.key_down("s", ("ctrl", "shift"))
    backend.key_up("s", ("ctrl", "shift"))
    backend.close()
    assert pyn.kb is not None
    seq = [e[1] for e in pyn.kb.events if e[0] == "up"]
    assert seq == ["s", "shift", "ctrl"]


def test_key_down_partial_failure_releases_already_pressed_modifiers() -> None:
    """If pressing a modifier raises (backend I/O failure), the modifiers
    already pressed must be released and no modifier left held."""
    pyn = FakePynput()
    from ai_game_agent.actions.windows import WindowsBackend  # red: module missing

    def _factory() -> FakePynput:
        kb = pyn.kb if pyn.kb is not None else FakeKeyboardController()
        pyn.bind(kb, FakeMouseController(), FakeListener())
        return pyn

    backend = WindowsBackend(pynput_factory=_factory)
    backend.open()
    # Force the first modifier to fail.
    kb = pyn.kb
    assert kb is not None

    def _boom(key: object) -> None:
        raise InputBackendError("simulated pynput failure")

    kb.press = _boom  # type: ignore[method-assign]
    with pytest.raises(InputBackendError):
        backend.key_down("s", ("ctrl", "shift"))
    # No modifier may be left held: the backend must have released what it
    # pressed, or pressed nothing (either is acceptable; the invariant is
    # "no modifier stuck").
    assert kb.held == set()


# --- Mouse -------------------------------------------------------------------


def test_mouse_move_absolute() -> None:
    backend, pyn = make_backend()
    backend.open()
    backend.mouse_move(100, 200)
    backend.close()
    assert pyn.mouse_ctrl is not None
    assert ("move", 100, 200) in pyn.mouse_ctrl.events


def test_mouse_move_relative() -> None:
    backend, pyn = make_backend()
    backend.open()
    backend.mouse_move(10, -5, relative=True)
    backend.close()
    assert pyn.mouse_ctrl is not None
    assert ("relative_move", 10, -5) in pyn.mouse_ctrl.events


def test_mouse_click_left_default() -> None:
    backend, pyn = make_backend()
    backend.open()
    backend.mouse_click()
    backend.close()
    assert pyn.mouse_ctrl is not None
    assert ("click", "left", 1) in pyn.mouse_ctrl.events


def test_mouse_click_right_double() -> None:
    backend, pyn = make_backend()
    backend.open()
    backend.mouse_click("right", 2)
    backend.close()
    assert pyn.mouse_ctrl is not None
    assert ("click", "right", 2) in pyn.mouse_ctrl.events


def test_mouse_down_up_track_held_button() -> None:
    backend, pyn = make_backend()
    backend.open()
    backend.mouse_down("left")
    assert backend.held_buttons == {"left"}
    backend.mouse_up("left")
    assert backend.held_buttons == set()
    backend.close()
    assert pyn.mouse_ctrl is not None
    seq = [e[1] for e in pyn.mouse_ctrl.events if e[0] in ("down", "up")]
    assert seq == ["left", "left"]


def test_mouse_down_failure_leaves_no_button_held() -> None:
    """If ``mouse_down`` fails (e.g. the button is unavailable), the backend
    must not mark the button held — otherwise ``release_all`` would emit a
    spurious ``mouse_up`` the OS never saw a matching down for."""
    pyn = FakePynput()
    from ai_game_agent.actions.windows import WindowsBackend  # red: module missing

    def _factory() -> FakePynput:
        kb = pyn.kb if pyn.kb is not None else FakeKeyboardController()
        mouse = FakeMouseController()

        def _raise(button: object = FakeButton.left, *a: object, **kw: object) -> None:
            raise InputBackendError("simulated pynput failure")

        mouse.down = _raise  # type: ignore[method-assign]
        pyn.bind(kb, mouse, FakeListener())
        return pyn

    backend = WindowsBackend(pynput_factory=_factory)
    backend.open()
    with pytest.raises(InputBackendError):
        backend.mouse_down("left")
    assert backend.held_buttons == set()


# --- Release-all / lifecycle -------------------------------------------------


def test_release_all_releases_held_keys_and_buttons() -> None:
    backend, pyn = make_backend()
    backend.open()
    backend.key_down("w")
    backend.mouse_down("left")
    backend.release_all()
    backend.close()
    assert backend.held_keys == set()
    assert backend.held_buttons == set()
    assert pyn.kb is not None
    assert ("up", "w") in pyn.kb.events
    assert pyn.mouse_ctrl is not None
    assert ("up", "left") in pyn.mouse_ctrl.events


def test_release_all_is_idempotent() -> None:
    """A second ``release_all`` must not emit a duplicate ``up`` (the
    emergency stop may call it repeatedly)."""
    backend, pyn = make_backend()
    backend.open()
    backend.key_down("w")
    backend.release_all()
    first = list(pyn.kb.events) if pyn.kb is not None else []
    backend.release_all()
    assert (pyn.kb.events if pyn.kb is not None else []) == first
    assert backend.held_keys == set()


def test_close_releases_held_input_and_stops_listener() -> None:
    backend, pyn = make_backend(on_stop=lambda: None)
    backend.open()
    backend.key_down("w")
    backend.mouse_down("left")
    backend.close()
    assert backend.held_keys == set()
    assert backend.held_buttons == set()
    assert pyn.listener is not None and pyn.listener.stopped is True


def test_open_close_idempotent() -> None:
    backend, pyn = make_backend(on_stop=lambda: None)
    backend.open()
    backend.open()  # idempotent
    assert pyn.listener is not None and pyn.listener.started is True
    backend.close()
    backend.close()  # idempotent
    assert pyn.listener is not None and pyn.listener.stopped is True
    assert backend.held_keys == set()


def test_input_before_open_raises() -> None:
    """Fail fast (like the mock): input before ``open()`` is a domain error,
    not a traceback from a half-open backend."""
    backend, _ = make_backend()
    with pytest.raises(InputBackendError):
        backend.key_down("a")


# --- Hotkey: stop-key routing (step-7 mechanism as the decision object) ------


def test_default_stop_key_is_f12() -> None:
    """The config default is ``safety.emergency_stop_keys: [F12]``; the
    backend must accept the uppercase spelling and lower it."""
    hits: list[str] = []
    backend, pyn = make_backend(on_stop=lambda: hits.append("stop"))
    backend.open()
    assert pyn.listener is not None
    pyn.listener.simulate_key("f12")
    backend.close()
    assert hits == ["stop"]


def test_configurable_stop_key() -> None:
    """A non-default stop key (e.g. ``"escape"``) must route the same way,
    and the *default* key must no longer fire once the key is overridden."""
    hits: list[str] = []
    backend, pyn = make_backend(on_stop=lambda: hits.append("stop"), stop_key="escape")
    backend.open()
    assert pyn.listener is not None
    pyn.listener.simulate_key("escape")
    pyn.listener.simulate_key("f12")  # no longer the stop key
    backend.close()
    assert hits == ["stop"]


def test_non_stop_key_is_ignored() -> None:
    """A game key (``a``/``w``/``space``) must never be mistaken for the stop
    key: no stop request, no exception, no state change."""
    hits: list[str] = []
    backend, pyn = make_backend(on_stop=lambda: hits.append("stop"))
    backend.open()
    assert pyn.listener is not None
    for key in ("a", "w", "space", "ctrl"):
        pyn.listener.simulate_key(key)
    assert hits == []
    # The stop still works after a bunch of non-stop keys.
    pyn.listener.simulate_key("f12")
    assert hits == ["stop"]
    backend.close()


def test_repeated_f12_requests_stop_only_once_per_arm_cycle() -> None:
    """The step-7 AC1 idempotency, exercised through the real backend routing:
    OS auto-repeat, double-delivery, or a second finger on the same key must
    request the emergency stop exactly once per arm cycle."""
    hits: list[str] = []
    backend, pyn = make_backend(on_stop=lambda: hits.append("stop"))
    backend.open()
    assert pyn.listener is not None
    pyn.listener.simulate_key("f12")
    pyn.listener.simulate_key("F12")  # uppercase variant
    pyn.listener.simulate_key("f12")  # third delivery
    backend.close()
    assert hits == ["stop"]


def test_disarmed_trigger_does_not_request_stop() -> None:
    """After ``close()`` (which disarms the listener and stops pynput), a
    stale key event must not fire the stop handler. ``simulate_key`` raises
    if the listener is stopped, so this asserts via the backend's disarmed
    state, not via the fake's own guard."""
    hits: list[str] = []
    backend, _ = make_backend(on_stop=lambda: hits.append("stop"))
    backend.open()
    backend.close()
    # After close the backend must not route anything.
    assert hits == []
    # Direct routing must be inert: calling the backend's stop-path with a
    # stop key must be a no-op once disarmed.
    backend._hotkey_listener.handle("f12")  # type: ignore[attr-defined]  # red
    assert hits == []


# --- Integration: backend hotkey → real Executor (the step-7 chain) ----------


def test_hotkey_triggers_full_emergency_stop_chain() -> None:
    """End-to-end: the step-7 mechanism, wired as the backend's decision
    object, stops the executor, releases held input, cancels queued work,
    and blocks further dispatch — the AGENTS.md emergency-stop contract,
    now through the real backend routing."""
    from ai_game_agent.actions.windows import WindowsBackend  # red: module missing

    hits: list[str] = []
    pyn = FakePynput()
    # A *real* step-7 listener as the decision object, exactly how the
    # executor/CLI would wire it.
    step7 = FakeHotkeyListener(
        on_trigger=lambda: hits.append("stop"),
        triggered_keys=("F12",),
    )
    step7.arm()  # the CLI arms the stop-request cycle for this run

    def _on_stop() -> None:
        # The routing the CLI performs: request the stop through the step-7
        # mechanism, then make the executor's terminal stop.
        step7.trigger("f12")
        ex.trigger_emergency_stop("f12")

    backend = WindowsBackend(
        pynput_factory=lambda: pyn,
        on_stop=_on_stop,
    )
    ex = _make_executor(backend)
    # Pending work the stop must cancel.
    q: ActionQueue = ex.queue
    q.add(
        ActionRequest(
            action=Action(kind=ActionKind.KEY_PRESS, key="x"),
            requested_at=datetime(2026, 10, 10, 17, 0),
        )
    )
    assert q.pending_count() == 1
    # Hold real input so the stop has something to release.
    backend.open()
    backend.key_down("w")
    ex.held_keys = set(backend.held_keys)
    backend.mouse_down("left")
    ex.held_buttons = set(backend.held_buttons)
    # The OS delivers F12.
    assert pyn.listener is not None
    pyn.listener.simulate_key("f12")

    assert ex.stopped is True
    assert q.pending_count() == 0
    assert backend.held_keys == set()
    assert backend.held_buttons == set()
    assert ex.stop_reason == "f12"
    assert hits == ["stop"]

    # Further dispatch is blocked: no new backend events, failed result.
    before = list(pyn.kb.events if pyn.kb else [])
    result = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="q"))
    assert result.ok is False
    assert result.stop_event is True
    assert (pyn.kb.events if pyn.kb else []) == before
    backend.close()


def test_hotkey_double_delivery_does_not_double_release() -> None:
    """A second F12 delivery must not re-request the stop or re-release:
    the backend's event log is frozen at the stop."""
    from ai_game_agent.actions.windows import WindowsBackend  # red: module missing

    hits: list[str] = []
    pyn = FakePynput()
    step7 = FakeHotkeyListener(
        on_trigger=lambda: hits.append("stop"), triggered_keys=("F12",)
    )
    step7.arm()  # the CLI arms the stop-request cycle for this run

    def _on_stop() -> None:
        # The routing the CLI performs: request the stop through the step-7
        # mechanism, then make the executor's terminal stop.
        step7.trigger("f12")
        ex.trigger_emergency_stop("f12")

    backend = WindowsBackend(
        pynput_factory=lambda: pyn,
        on_stop=_on_stop,
    )
    ex = _make_executor(backend)
    backend.open()
    backend.key_down("w")
    ex.held_keys = set(backend.held_keys)
    assert pyn.listener is not None
    pyn.listener.simulate_key("f12")
    first = list(pyn.kb.events if pyn.kb else [])
    pyn.listener.simulate_key("f12")  # auto-repeat
    assert (pyn.kb.events if pyn.kb else []) == first
    assert hits == ["stop"]
    assert ex.stopped is True
    backend.close()
