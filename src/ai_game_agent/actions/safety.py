"""Safety layer (Phase 3, step 3): the mode gate + bounded action limits.

The AI-independent safety layer (AGENTS.md: "Safety mechanisms are mandatory",
"Safety mechanisms are … independent of the AI"). A :class:`SafetyGuard` is a
small, frozen, dependency-free validator that the executor calls
**before** it performs an action. It never touches a backend, a queue, or an
input library — it checks the :class:`~ai_game_agent.actions.action.Action`
against the mode and the configured bounds and raises a classified error:

* :class:`~ai_game_agent.actions.base.ModeViolation` — any input attempt
  under ``OBSERVE_ONLY``. This is the safety-critical regression: the mode
  gate fires *before* any bound check, so an over-long hold under
  ``OBSERVE_ONLY`` is still a mode violation, and the backend's event log is
  asserted empty by the executor tests.
* :class:`~ai_game_agent.actions.base.SafetyViolation` — a bound violation
  (duration/delay over ``max_action_duration_ms``, a negative coordinate, or
  the batch over ``max_consecutive_actions``).
* :class:`ActionQueueError` is raised by the queue (step 4), not the guard.

The guard is the single place "is this action legal right now?" is answered,
which keeps the mode gate and the bounds in one testable unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ai_game_agent.actions.action import Action, ActionKind
from ai_game_agent.actions.base import ModeViolation, SafetyViolation
from ai_game_agent.actions.modes import OperationMode
from ai_game_agent.config import Config


class ExecutorState(Enum):
    """The executor's explicit, testable state machine (Phase 3 step 4 owns it).

    Declared here because the guard is the only other component that needs to
    observe it (PAUSED gates further input, STOPPED is terminal). Keeping the
    enum in one place avoids a second copy in ``executor.py``.
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class SafetyGuard:
    """Validates a proposed :class:`Action` against the mode + configured bounds.

    All limits come from config (``SafetyConfig`` + ``InputConfig.mode``) so a
    YAML edit changes the guard's behavior without touching code — the
    config-first pattern Phase 2 established for perception. The guard is
    **not** immutable once constructed (it tracks a per-batch action count and
    a backend-error count); the *configured* bounds are, however, set once at
    construction and never mutated afterwards.

    ``mode`` is the only field that gates input: under ``OBSERVE_ONLY`` every
    :meth:`check` raises :class:`ModeViolation` before any bound is even
    evaluated.
    """

    mode: OperationMode
    max_action_duration_ms: int = 5000
    max_consecutive_actions: int = 50
    max_backend_errors: int = 3

    # Mutable, per-execution state (the guard is one per executor batch).
    _batch_count: int = field(default=0, init=False, repr=False)
    _backend_error_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, OperationMode):
            # ``OperationMode("x")`` raises ValueError for an unknown string;
            # coerce so a plain str from config/config-view lands on the enum.
            self.mode = OperationMode(self.mode)
        if not (self.max_action_duration_ms > 0):
            raise ValueError("max_action_duration_ms must be > 0")
        if not (self.max_consecutive_actions > 0):
            raise ValueError("max_consecutive_actions must be > 0")
        if not (self.max_backend_errors > 0):
            raise ValueError("max_backend_errors must be > 0")

    # -- Mode gate (safety-critical; fires before any bound check) -----------

    def check_mode(self, action: Action) -> None:
        """Raise :class:`ModeViolation` under ``OBSERVE_ONLY``; else no-op.

        Split out from :meth:`check` so the executor (and tests) can assert
        the mode gate independently of the bound checks — this is the one
        method the project's safety story depends on.
        """
        if not self.mode.input_allowed:
            raise ModeViolation(
                f"OBSERVE_ONLY mode: action {action.kind.value!r} is not allowed "
                f"(no input may be emitted); promote the mode or use ASSISTED "
                f"to propose it."
            )

    def check(self, action: Action) -> None:
        """Validate one action: mode gate, then bounds, then batch count.

        Raises:
            ModeViolation: under ``OBSERVE_ONLY`` (always, for any action).
            SafetyViolation: a bound or batch-count violation.
        """
        # Mode gate first: a bad action under OBSERVE_ONLY is a *mode*
        # violation, not a bound violation — the mode is the safety story.
        self.check_mode(action)

        # Duration/delay bounds (any timing field, any kind that carries one).
        if action.duration_ms is not None and action.duration_ms > self.max_action_duration_ms:
            raise SafetyViolation(
                f"action {action.kind.value!r} duration_ms={action.duration_ms} "
                f"exceeds max_action_duration_ms={self.max_action_duration_ms}"
            )
        if action.delay_ms is not None and action.delay_ms > self.max_action_duration_ms:
            raise SafetyViolation(
                f"action {action.kind.value!r} delay_ms={action.delay_ms} "
                f"exceeds max_action_duration_ms={self.max_action_duration_ms}"
            )

        # Coordinate sanity: screen-space coordinates must be non-negative.
        # (The action contract already forbids *non-int* coordinates; this
        # catches a negative value that is in-range for an int.)
        if action.kind in (ActionKind.MOUSE_MOVE, ActionKind.MOUSE_CLICK):
            if action.x is not None and action.x < 0:
                raise SafetyViolation(
                    f"action {action.kind.value!r} x={action.x} is negative"
                )
            if action.y is not None and action.y < 0:
                raise SafetyViolation(
                    f"action {action.kind.value!r} y={action.y} is negative"
                )

        # Batch bound: max_consecutive_actions per executor run.
        if self._batch_count >= self.max_consecutive_actions:
            raise SafetyViolation(
                f"batch already executed {self._batch_count} actions; "
                f"max_consecutive_actions={self.max_consecutive_actions} "
                f"(split the batch or raise the bound)"
            )

    def allow(self, action: Action) -> None:
        """Record that an action passed the gate (increment the batch count).

        The executor calls :meth:`check` (raise-on-violation) then
        :meth:`allow` (record-on-success). Splitting the two keeps the check
        pure — it can be called in ASSISTED to *propose* without consuming a
        slot — while the count still enforces the batch bound over the
        actions actually executed.
        """
        self._batch_count += 1

    def reset_batch(self) -> None:
        """Start a new batch (clear the consecutive-action count)."""
        self._batch_count = 0

    # -- Bounded backend-error retry (no infinite loops) ---------------------

    def record_backend_error(self) -> None:
        """Count a backend I/O failure; :attr:`should_pause` flips at the bound."""
        self._backend_error_count += 1

    def record_success(self) -> None:
        """A successful action clears the consecutive-error count."""
        self._backend_error_count = 0

    @property
    def backend_errors(self) -> int:
        return self._backend_error_count

    @property
    def should_pause(self) -> bool:
        """True once ``max_backend_errors`` consecutive failures have occurred.

        The executor transitions to ``PAUSED`` (recoverable, distinct from
        the terminal ``STOPPED``) rather than retrying forever — AGENTS.md:
        "Do not create infinite retry loops. If recovery fails, transition to
        a safe paused state."
        """
        return self._backend_error_count >= self.max_backend_errors

    # -- Construction from config ---------------------------------------------

    @classmethod
    def from_config(cls, config: Config | None = None) -> SafetyGuard:
        """Build a guard from a :class:`~ai_game_agent.config.Config`.

        ``config`` may be ``None`` (all safe defaults) — this is the path the
        bare-venv import contract and the ``act --demo`` CLI use.
        """
        config = config if config is not None else Config()
        return cls(
            mode=OperationMode(config.input.mode),
            max_action_duration_ms=config.safety.max_action_duration_ms,
            max_consecutive_actions=config.safety.max_consecutive_actions,
            max_backend_errors=config.safety.max_backend_errors,
        )


__all__ = ["ExecutorState", "SafetyGuard"]
