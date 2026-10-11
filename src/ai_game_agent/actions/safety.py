"""Safety layer (Phase 3, step 3): the mode gate + bounded action limits.

The AI-independent safety layer (AGENTS.md: "Safety mechanisms are mandatory").
A :class:`SafetyGuard` is a small, frozen, dependency-free validator that the
executor calls **before** performing an action. It never touches a backend,
queue, or input library — it checks the
:class:`~ai_game_agent.actions.action.Action` against the operating mode and
the configured bounds and raises a classified error:

* :class:`~ai_game_agent.actions.base.ModeViolation` — any input attempt
  under ``OBSERVE_ONLY``. This is the safety-critical regression: the mode
  gate fires **before** any bound check, so an over-long hold under
  ``OBSERVE_ONLY`` is still a mode violation, and the executor tests assert
  the backend's event log is empty.
* :class:`~ai_game_agent.actions.base.SafetyViolation` — a bound violation
  (duration/delay over ``max_action_duration_ms``, or the batch over
  ``max_consecutive_actions``).

Two methods, deliberately distinct:

* :meth:`SafetyGuard.check_mode` — the mode gate alone. Independently
  testable; consumes no batch slot. This is the method the project's safety
  story depends on.
* :meth:`SafetyGuard.check` — the complete action-admission policy: mode
  gate, then bounds, then the batch bound. Only an action that passes
  **every** check consumes a batch slot; a rejected action never does.

Counter semantics and threading
-------------------------------
The batch and backend-error counters live in a private mutable state object
(kept off the frozen guard) and are **per guard instance**: each executor
run should use a fresh guard or call :meth:`reset_batch` before a new batch.
The guard is **not thread-safe**; the executor runs a single decision loop
(single-threaded assumption), so no synchronization is added here — if a
future executor admits concurrent ``check()`` callers, it must own the
lock, not the guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ai_game_agent.actions.action import Action
from ai_game_agent.actions.base import ModeViolation, SafetyViolation
from ai_game_agent.actions.modes import OperationMode
from ai_game_agent.config import Config


class ExecutorState(Enum):
    """The executor's explicit, testable state (owned by step 4's Executor).

    Declared here because the guard is the only other component that needs to
    observe it (PAUSED gates further input; STOPPED is terminal). Keeping the
    enum in one place avoids a second copy in ``executor.py``.
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class _GuardState:
    """Mutable per-guard counters (isolated from the frozen guard)."""

    batch_count: int = 0
    backend_errors: int = 0


@dataclass(frozen=True)
class SafetyGuard:
    """Validates a proposed :class:`Action` against the mode + configured bounds.

    All bounds come from config (``SafetyConfig`` + ``InputConfig.mode``) so
    a YAML edit changes guard behavior without touching code — the
    config-first pattern Phase 2 established for perception. The guard is
    **frozen**: mode and bounds are set once at construction and can never be
    mutated (assignment raises). The only mutable state is the per-guard
    counter object (see module docstring for the semantics).

    ``mode`` is the only field that gates input: under ``OBSERVE_ONLY``,
    every :meth:`check` raises :class:`ModeViolation` before any bound is
    even evaluated.
    """

    mode: OperationMode
    max_action_duration_ms: int = 5000
    max_consecutive_actions: int = 50
    max_backend_errors: int = 3

    _state: _GuardState = field(
        default_factory=_GuardState, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.mode, OperationMode):
            # ``OperationMode("x")`` raises ValueError for an unknown string —
            # a typo never silently lands on the wrong mode.
            object.__setattr__(self, "mode", OperationMode(self.mode))
        # The guard cannot silently operate with missing/invalid limits:
        # enforce the same positivity contract SafetyConfig declares.
        if not (self.max_action_duration_ms > 0):
            raise ValueError("max_action_duration_ms must be > 0")
        if not (self.max_consecutive_actions > 0):
            raise ValueError("max_consecutive_actions must be > 0")
        if not (self.max_backend_errors > 0):
            raise ValueError("max_backend_errors must be > 0")

    # -- Mode gate (safety-critical; fires before any bound check) -----------

    def check_mode(self, action: Action) -> None:
        """Raise :class:`ModeViolation` under ``OBSERVE_ONLY``; else no-op.

        The mode gate alone: independently testable, and it consumes no batch
        slot. This is the one method the project's safety story depends on.
        """
        if not self.mode.input_allowed:
            raise ModeViolation(
                f"OBSERVE_ONLY mode: action {action.kind.value!r} is not "
                f"allowed (no input may be emitted); promote the mode or use "
                f"ASSISTED to propose it."
            )

    def check(self, action: Action) -> None:
        """Complete action-admission policy: mode gate, bounds, batch bound.

        A batch slot is consumed **only** when every check passes; a rejected
        action never consumes one.

        Raises:
            ModeViolation: under ``OBSERVE_ONLY`` (any action).
            SafetyViolation: a bound or batch-count violation.
        """
        self.check_mode(action)

        # Timing bounds (the Action contract already guarantees both are
        # non-negative ints; zero for kinds that do not carry them).
        if action.duration_ms > self.max_action_duration_ms:
            raise SafetyViolation(
                f"action {action.kind.value!r} duration_ms="
                f"{action.duration_ms} exceeds "
                f"max_action_duration_ms={self.max_action_duration_ms}"
            )
        if action.delay_ms > self.max_action_duration_ms:
            raise SafetyViolation(
                f"action {action.kind.value!r} delay_ms={action.delay_ms} "
                f"exceeds max_action_duration_ms="
                f"{self.max_action_duration_ms}"
            )

        # Batch bound: max_consecutive_actions admitted per run. (Coordinate
        # sanity lives in the Action contract, not here — single source of
        # truth.)
        if self._state.batch_count >= self.max_consecutive_actions:
            raise SafetyViolation(
                f"batch already executed {self._state.batch_count} actions; "
                f"max_consecutive_actions={self.max_consecutive_actions} "
                f"(split the batch or raise the bound)"
            )

        # Every check passed: admit and consume the slot.
        self._state.batch_count += 1

    def reset_batch(self) -> None:
        """Start a new batch (clear the consecutive-action count)."""
        self._state.batch_count = 0

    # -- Bounded backend-error retry (no infinite loops) ---------------------

    def record_backend_error(self) -> None:
        """Count a backend I/O failure; :attr:`should_pause` flips at the bound."""
        self._state.backend_errors += 1

    def record_success(self) -> None:
        """A successful action clears the consecutive-error count."""
        self._state.backend_errors = 0

    @property
    def batch_count(self) -> int:
        """Actions admitted in the current batch (since the last reset)."""
        return self._state.batch_count

    @property
    def backend_errors(self) -> int:
        """Current consecutive-backend-error count (since the last success)."""
        return self._state.backend_errors

    @property
    def should_pause(self) -> bool:
        """True once ``max_backend_errors`` consecutive failures have occurred.

        The executor transitions to ``PAUSED`` (recoverable, distinct from
        the terminal ``STOPPED``) rather than retrying forever — AGENTS.md:
        "Do not create infinite retry loops. If recovery fails, transition to
        a safe paused state."
        """
        return self._state.backend_errors >= self.max_backend_errors

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
