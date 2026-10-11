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
emergency stop.

* A backend I/O failure during ``execute`` is recorded on the guard
  (``record_backend_error``); a success clears the consecutive count
  (``record_success``). Once ``safety.should_pause`` is true the executor
  transitions to ``PAUSED`` — bounded retry, never an infinite loop
  (AGENTS.md).
* ``PAUSED`` **gates further input**: while paused, every :meth:`execute`
  short-circuits to a failed result and **never touches the backend**.
  :meth:`resume` is the only way back: it clears the consecutive-error
  count and returns the executor to ``IDLE``. If the backend is still
  failing it will simply pause again after ``max_backend_errors`` more
  failures. ``resume`` refuses to run from ``STOPPED`` (raises
  :class:`ValueError`).
* Once ``STOPPED``, every :meth:`execute` short-circuits to a failed result
  with ``stop_event=True`` and **never touches the backend** — the
  safety-critical invariant the OBSERVE_ONLY and emergency-stop regression
  tests both depend on.
* **Pending-release gate**: while any failed release is still unresolved
  (``_unreleased_keys`` / ``_unreleased_buttons`` non-empty), every
  :meth:`execute` short-circuits to a failed result **without touching the
  backend** — a failed release means the physical input state is uncertain
  (the key may still be held), so no unrelated input may flow and no
  unrelated success may clear the condition. ``resume()`` flushes the
  pending release first and only unblocks if the flush lands.

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
        # Releases that were attempted but whose backend call raised: kept
        # so a later ``release_all`` / ``trigger_emergency_stop`` retries
        # **only** those, without re-emitting releases that already landed.
        self._unreleased_keys: set[str] = set()
        self._unreleased_buttons: set[str] = set()

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

    def release_all(self) -> bool:
        """Release every held key and button through the backend.

        Every held input gets its own release attempt: a backend exception
        on one release does **not** stop the remaining releases — the
        emergency stop must release all held input, not bail on the first
        failure. A failed release is remembered and retried by a later call
        (``release_all`` or a repeat ``trigger_emergency_stop``); releases
        that already succeeded are never re-emitted, so a clean release
        stays idempotent (the mock's event log stays stable).

        Drains :attr:`held_keys` and :attr:`held_buttons` — cleared either
        way, since after a failed release the executor has done all it can
        for that input until the backend recovers.

        Returns:
            ``True`` if every release succeeded, ``False`` if any backend
            call raised (the failed inputs remain in the executor's
            failed-release sets for a later retry).
        """
        ok = True
        for key in sorted(self.held_keys | self._unreleased_keys):
            try:
                self._backend.key_up(key)
            except Exception:  # noqa: BLE001 - one failure must not block the rest
                self._unreleased_keys.add(key)
                ok = False
            else:
                self._unreleased_keys.discard(key)
        self.held_keys.clear()
        for button in sorted(self.held_buttons | self._unreleased_buttons):
            try:
                self._backend.mouse_up(button)
            except Exception:  # noqa: BLE001 - one failure must not block the rest
                self._unreleased_buttons.add(button)
                ok = False
            else:
                self._unreleased_buttons.discard(button)
        self.held_buttons.clear()
        return ok

    # -- emergency stop --------------------------------------------------

    def trigger_emergency_stop(self, reason: str = "emergency") -> None:
        """The hard stop, in the AGENTS.md-mandated order:

        1. set the ``stopped`` flag (state → ``STOPPED``);
        2. ``queue.cancel_all()`` (drop queued work);
        3. ``release_all()`` (release held keys **and** mouse buttons);
        4. block further input (every later ``execute`` short-circuits);
        5. record the stop (reason retained for the first trigger only).

        Idempotent: a second call keeps the original reason and re-emits
        nothing — **except** that releases which failed during the first
        call are retried (the executor is already ``STOPPED``, so the retry
        cannot emit any new input).
        """
        if self._stopped:
            # Already stopped: retry only the releases that failed on the
            # first call (a clean first release makes this a no-op).
            if self._unreleased_keys or self._unreleased_buttons:
                self.release_all()
            return
        # 1. stop the decision loop first — nothing after this point may
        #    schedule further input.
        self._stopped = True
        self._state = ExecutorState.STOPPED
        # 2. cancel queued actions (priority lane included).
        self.queue.cancel_all()
        # 3. + 4. release all held input (fault-tolerant: a failed release
        #    is remembered and retried on the next call); the blocked state
        #    means further execute() calls are no-ops, so the release is
        #    final.
        self.release_all()
        # 5. record the stop event.
        if self._stop_reason is None:
            self._stop_reason = reason

    def resume(self) -> None:
        """Recover a ``PAUSED`` executor: clear the error count, go ``IDLE``.

        ``PAUSED`` is the recoverable safe state (bounded backend errors
        exhausted — AGENTS.md's "transition to a safe paused state"); the
        caller invokes this once the backend is worth retrying. If the
        backend is still failing it will simply pause again after
        ``max_backend_errors`` more failures, so this cannot create an
        unbounded retry loop.

        Any pending (failed) releases are flushed **first**: while a key or
        button release is unresolved the physical input state is uncertain,
        so the executor must not unblock. If the flush fails the executor
        goes to ``PAUSED`` (still blocked — the pending-release gate in
        :meth:`execute` applies too); if it succeeds the error count clears
        and input flows again.

        Raises:
            ValueError: if the executor is ``STOPPED`` — an emergency stop
                is terminal for the process lifetime and is never undone
                at runtime.
        """
        if self._stopped:
            raise ValueError(
                "cannot resume from STOPPED: an emergency stop is terminal "
                "for the process lifetime"
            )
        if self.release_all():
            # All releases landed (or there was nothing pending): the
            # physical input state is settled, so a success is a real
            # success — clear the consecutive-error count and unblock.
            self._safety.record_success()
            self._state = ExecutorState.IDLE
        else:
            # At least one release still failed: stay in a safe blocked
            # state until a later resume() (or emergency stop) flushes it.
            self._state = ExecutorState.PAUSED

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

        # Paused: bounded backend errors exhausted. Gates further input
        # (no backend contact) until the caller recovers via resume().
        if self._state is ExecutorState.PAUSED:
            return ActionResult(
                action=action,
                ok=False,
                error=(
                    f"executor paused after {self._safety.backend_errors} "
                    "consecutive backend errors; call resume() to continue"
                ),
            )

        # Pending release: a failed release leaves the physical input state
        # uncertain (the key/button may still be held). Ordinary actions are
        # blocked — with or without backend contact — until the pending
        # release is flushed via resume() or trigger_emergency_stop(). No
        # unrelated action may clear the condition while input is unresolved.
        if self._unreleased_keys or self._unreleased_buttons:
            return ActionResult(
                action=action,
                ok=False,
                error=(
                    "executor has an unresolved key/mouse release "
                    "(physical input state uncertain); call resume() to flush "
                    "it before further input"
                ),
            )

        self._state = ExecutorState.RUNNING

        # Mode gate + bounds fire **before** any backend call. These raise
        # (they are policy violations the caller must surface), not fail
        # silently — so the mock's event log stays empty.
        self._safety.check(action)

        start = self._clock()
        try:
            self._perform(action)
        except Exception as exc:  # noqa: BLE001 - convert, don't crash
            # Bounded-retry contract (AGENTS.md): a backend failure is
            # recorded on the guard, and once the bound is exhausted the
            # executor pauses (``PAUSED`` is recoverable, unlike the
            # terminal ``STOPPED``) — never an infinite retry loop.
            self._safety.record_backend_error()
            if self._safety.should_pause:
                self._state = ExecutorState.PAUSED
            # release_all is guaranteed by the per-kind try/finally inside
            # _perform; record the failure for the result.
            return ActionResult(
                action=action,
                ok=False,
                error=f"action {action.kind.value} failed: {exc}",
            )
        duration_ms = (self._clock() - start) * 1000.0
        # A success clears the consecutive-error count (the count tracks
        # *consecutive* failures only).
        self._safety.record_success()
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
            self._release_button(action.button)
        elif kind is ActionKind.MOUSE_SCROLL:
            self._perform_scroll(action.scroll)
        elif kind is ActionKind.DELAY:
            self._sleeper(action.duration_ms / 1000.0)
        else:  # pragma: no cover - the Action contract forbids unknown kinds
            raise ActionError(f"unsupported action kind: {kind!r}")

    def _hold_key(self, key: str, mods: tuple[str, ...], duration_ms: int) -> None:
        """key_down → hold → key_up, with the release in a ``finally``.

        A failure anywhere in the hold (including the injected sleeper)
        still emits the release — the release-on-exception guarantee. A
        failed release attempt is handled exactly like the explicit
        release path (see :meth:`_release_key`): the key is tracked as a
        failed release for a later retry, never left as "still held".
        """
        self._backend.key_down(key, mods)
        self.held_keys.add(key)
        try:
            if duration_ms > 0:
                self._sleeper(duration_ms / 1000.0)
        finally:
            self._release_key(key, mods)

    def _release_button(self, button: str) -> None:
        """Emit one mouse-button release, updating bookkeeping either way.

        The button/mouse counterpart of :meth:`_release_key`: after a
        failed release the button moves from :attr:`held_buttons` to the
        failed-release set (retried by ``release_all`` /
        ``trigger_emergency_stop``), never left as "still held", and the
        exception propagates for the normal error accounting.
        """
        try:
            self._backend.mouse_up(button)
        except Exception:  # noqa: BLE001 - record, then let it propagate
            self._unreleased_buttons.add(button)
            raise
        finally:
            self.held_buttons.discard(button)

    def _release_key(self, key: str, mods: tuple[str, ...]) -> None:
        """Emit one key release, updating bookkeeping whether it lands or fails.

        The held-state bookkeeping is updated **either way**: after a
        failed release the executor has done all it can for that input,
        so it moves from :attr:`held_keys` to the failed-release set —
        where ``release_all`` / ``trigger_emergency_stop`` retry exactly
        the releases that failed, without re-emitting ones that landed.
        The exception still propagates (``execute`` records it as a
        backend error for the bounded-retry contract).
        """
        try:
            self._backend.key_up(key, mods)
        except Exception:  # noqa: BLE001 - record, then let it propagate
            self._unreleased_keys.add(key)
            raise
        finally:
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
