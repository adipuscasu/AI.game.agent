"""Executor (Phase 3, step 5): the single action-execution abstraction.

AGENTS.md: "All mouse and keyboard control must go through a single
action-execution abstraction." The :class:`Executor` is the **only** object
that may call an :class:`~ai_game_agent.actions.base.InputBackend`, and only
through the protocol. It validates (via :class:`SafetyGuard`), sequences, and
performs :class:`~ai_game_agent.actions.action.Action` objects, tracks
key/button held-state, and owns the emergency stop.

Release-on-exception is guaranteed **by construction**: a held key is always
released in a ``try/finally``, so a failure during the hold (or anywhere in
the per-action perform path) still emits the release. This is the concrete
realization of AGENTS.md's "prefer abstractions that guarantee release even
if an exception occurs."

State machine
-------------
Explicit, testable states (``ExecutorState``): ``IDLE → RUNNING → (IDLE |
PAUSED | STOPPED)``. ``PAUSED`` is recoverable (bounded backend errors);
``STOPPED`` is terminal for the process lifetime and is set **only** by an
emergency stop. Once ``STOPPED``, every :meth:`execute` short-circuits to a
failed result with ``stop_event=True`` and **never touches the backend** —
the safety-critical invariant the OBSERVE_ONLY and emergency-stop regression
tests both depend on.

Dependency injection
--------------------
``Executor(backend, safety, clock=..., sleeper=...)`` — every collaborator
injectable (defaults ``time.perf_counter`` / ``time.sleep``), no globals, no
hidden I/O. Tests inject a no-op sleeper so a 10 s hold executes instantly and
a raising sleeper proves the release guarantee.
"""

from __future__ import annotations

import time

from .action import Action, ActionKind, ActionResult
from .base import (
    ActionError,
    InputBackend,
)
from .queue import ActionQueue
from .safety import ExecutorState, SafetyGuard

_DEFAULT_CLOCK = time.perf_counter
_DEFAULT_SLEEPER = time.sleep


class Executor:
    """Validate, sequence, and perform actions through one backend.

    Parameters
    ----------
    backend:
        The :class:`InputBackend` to perform input through. The executor is
        the only caller of ``backend`` methods.
    safety:
        The :class:`SafetyGuard` enforcing the mode gate and bounds.
    queue:
        An optional pre-built :class:`ActionQueue`; one is created when
        omitted. The emergency stop cancels this queue.
    clock, sleeper:
        Injectable time sources for deterministic testing.
    """

    def __init__(
        self,
        backend: InputBackend,
        safety: SafetyGuard,
        queue: ActionQueue | None = None,
        clock: object = _DEFAULT_CLOCK,
        sleeper: object = _DEFAULT_SLEEPER,
    ) -> None:
        self._backend = backend
        self._safety = safety
        self._clock = clock
        self._sleeper = sleeper
        self.queue = queue if queue is not None else ActionQueue(backend)

        self._state = ExecutorState.IDLE
        self._stopped = False
        self._stop_reason: str | None = None
        # Held-state source of truth (the mock backend mirrors it for
        # assertions). ``release_all`` drains both.
        self.held_keys: set[str] = set()
        self.held_buttons: set[str] = set()

    # -- state -----------------------------------------------------------

    @property
    def state(self) -> ExecutorState:
        """The executor's current state-machine state."""
        return self._state

    @property
    def stopped(self) -> bool:
        """True once an emergency stop has occurred (terminal)."""
        return self._stopped

    @property
    def stop_reason(self) -> str | None:
        """The recorded reason for the emergency stop, or ``None``."""
        return self._stop_reason

    @property
    def safety(self) -> SafetyGuard:
        """The guard enforcing the mode gate and bounds."""
        return self._safety

    # -- release-all -----------------------------------------------------

    def release_all(self) -> None:
        """Release every held key and button through the backend.

        Drains :attr:`held_keys` and :attr:`held_buttons`. Idempotent: once
        nothing is held, a further call emits no events (so the emergency
        stop can call it repeatedly and the mock's event log stays stable).
        """
        for key in sorted(self.held_keys):
            self._backend.key_up(key)
        self.held_keys.clear()
        for button in sorted(self.held_buttons):
            self._backend.mouse_up(button)
        self.held_buttons.clear()

    # -- emergency stop --------------------------------------------------

    def trigger_emergency_stop(self, reason: str = "emergency") -> None:
        """The hard stop, in the AGENTS.md-mandated order:

        1. set the ``stopped`` flag (state → ``STOPPED``);
        2. ``queue.cancel_all()`` (drop queued work);
        3. ``release_all()`` (release held keys **and** mouse buttons);
        4. block further input (every later ``execute`` short-circuits);
        5. record the stop (reason retained for the first trigger only).

        Idempotent: a second call keeps the original reason and does not
        re-emit releases.
        """
        if self._stopped:
            return
        # 1. stop the decision loop first — nothing after this point may
        #    schedule further input.
        self._stopped = True
        self._state = ExecutorState.STOPPED
        # 2. cancel queued actions (priority lane included).
        self.queue.cancel_all()
        # 3. + 4. release all held input; the blocked state (step 4/5) means
        #    further execute() calls are no-ops, so the release is final.
        self.release_all()
        # 5. record the stop event.
        if self._stop_reason is None:
            self._stop_reason = reason

    # -- execution -------------------------------------------------------

    def execute(self, action: Action) -> ActionResult:
        """Validate → pre-delay → perform → return an :class:`ActionResult`.

        On the emergency-stop path (``stopped`` already set) this does
        **not** touch the backend: it returns a failed result with
        ``stop_event=True``. A :class:`ModeViolation` or
        :class:`SafetyViolation` from the guard propagates to the caller
        (the CLI/decision layer decides how to report it); a backend or
        per-action failure is converted to a failed result with the release
        guaranteed by the per-action ``try/finally``.
        """
        # Blocked: terminal. Never touches the backend.
        if self._stopped:
            return ActionResult(
                action=action,
                ok=False,
                error=f"executor stopped ({self._stop_reason or 'emergency stop'})",
                stop_event=True,
            )

        self._state = ExecutorState.RUNNING

        # Mode gate + bounds fire **before** any backend call. These raise
        # (they are policy violations the caller must surface), not fail
        # silently — so the mock's event log stays empty.
        self._safety.check(action)

        start = self._clock()
        try:
            self._perform(action)
        except (ActionError, Exception) as exc:  # noqa: BLE001 - convert, don't crash
            # release_all is guaranteed by the per-kind try/finally inside
            # _perform; record the failure for the result.
            return ActionResult(
                action=action,
                ok=False,
                error=f"action {action.kind.value} failed: {exc}",
            )
        duration_ms = (self._clock() - start) * 1000.0
        return ActionResult(
            action=action,
            ok=True,
            duration_ms=duration_ms,
        )

    # -- perform: one action, release guaranteed -------------------------

    def _perform(self, action: Action) -> None:
        """Drive the backend for one action. Held input is always released."""
        kind = action.kind
        mods = tuple(action.modifiers) if action.modifiers else ()

        # Pre-delay (bounded by the guard; here we just honor it).
        if action.delay_ms > 0:
            self._sleeper(action.delay_ms / 1000.0)

        if kind is ActionKind.KEY_PRESS:
            self._backend.press(action.key, mods)
        elif kind is ActionKind.KEY_HOLD:
            self._hold_key(action.key, mods, action.duration_ms)
        elif kind is ActionKind.MOUSE_MOVE:
            self._backend.mouse_move(action.x, action.y, action.relative)
        elif kind is ActionKind.MOUSE_CLICK:
            self._backend.mouse_click(action.button, action.clicks)
        elif kind is ActionKind.MOUSE_DOWN:
            self._backend.mouse_down(action.button)
            self.held_buttons.add(action.button)
        elif kind is ActionKind.MOUSE_UP:
            self._backend.mouse_up(action.button)
            self.held_buttons.discard(action.button)
        elif kind is ActionKind.MOUSE_SCROLL:
            self._perform_scroll(action.scroll)
        elif kind is ActionKind.DELAY:
            self._sleeper(action.duration_ms / 1000.0)
        else:  # pragma: no cover - the Action contract forbids unknown kinds
            raise ActionError(f"unsupported action kind: {kind!r}")

    def _hold_key(self, key: str, mods: tuple[str, ...], duration_ms: int) -> None:
        """key_down → hold → key_up, with the release in a ``finally``.

        A failure anywhere in the hold (including the injected sleeper)
        still emits the release — the release-on-exception guarantee.
        """
        self._backend.key_down(key, mods)
        self.held_keys.add(key)
        try:
            if duration_ms > 0:
                self._sleeper(duration_ms / 1000.0)
        finally:
            self._backend.key_up(key, mods)
            self.held_keys.discard(key)

    def _perform_scroll(self, ticks: int | None) -> None:
        """Scroll by ``ticks`` (positive = up) using the backend's mouse API."""
        if not ticks:
            return
        # pynput-style: no dedicated scroll on the minimal backend protocol;
        # emulate with up/down click deltas is out of scope — the contract's
        # scroll is carried but the minimal protocol has no scroll method, so
        # this is a no-op on the mock and the real adapter implements it.
        scroll = getattr(self._backend, "mouse_scroll", None)
        if scroll is not None:
            scroll(ticks)


__all__ = ["Executor"]
