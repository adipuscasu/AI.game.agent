"""GameStateMachine (Phase 4, step 2) — the deterministic decision core.

Maps ``(current state, policy signals)`` to the next state and enforces the
phase-4 invariants that are *independent* of any specific game:

* **Watchdog.** While in a ``watchdog_states`` state (``RECOVERY`` and the
  ``AI_ANALYSIS`` slot) for ``>= state_timeout_s`` — measured at ``decide()``
  time via the injected ``now`` — force ``PAUSED`` (reason
  ``"state timeout (watchdog)"``). The *only* escape from a stuck RECOVERY
  without a health reading, and the seam Phase 5's AI_ANALYSIS hangs off.
  It does **not** fire in non-watchdog states (IDLE/COMBAT/LOOTING persist
  indefinitely; the loop is bounded by ``max_ticks`` and the hotkey).
* **Bounded recovery.** ``note_recovery_attempt()`` counts consecutive
  RECOVERY attempts; once ``recovery_attempts`` pass without a healthy exit,
  the next ``decide()`` returns ``PAUSED`` — no infinite retry.
* **Self-loop idempotence.** A ``decide()`` that returns the current state is
  *not* a transition: no ``on_transition`` callback, no counter change,
  ``state_entered_at`` preserved.
* **No I/O, no side effects.** Reads only the policy-produced ``signals``
  dict.

Step-2 scope: transitions, recovery counter, watchdog, and the raw
``on_transition`` seam. Step 4 refines the callback into a ``LoopEvent``;
until then the callback is the raw ``(before, after, reason, now)`` tuple
(§4.3).

The machine is game-agnostic: it consumes only the stable signal keys the
policy promises — ``health_low``, ``loot_present``, ``target_hostile``,
``health_recovered`` — never zone names. A value of ``None`` means
unknown/not-detected and is *never* treated as ``False`` (or 0% health).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_game_agent.state.states import AgentState

__all__ = [
    "GameStateMachine",
    "OnTransition",
    "SIGNAL_HEALTH_LOW",
    "SIGNAL_HEALTH_RECOVERED",
    "SIGNAL_LOOT_PRESENT",
    "SIGNAL_TARGET_HOSTILE",
    "SIGNAL_TARGET_PRESENT",
]

# Stable signal keys: the machine's consumption seam (step 3 produces these).
SIGNAL_HEALTH_LOW = "health_low"
SIGNAL_HEALTH_RECOVERED = "health_recovered"
SIGNAL_LOOT_PRESENT = "loot_present"
SIGNAL_TARGET_HOSTILE = "target_hostile"
SIGNAL_TARGET_PRESENT = "target_present"

Signals = dict[str, object]
OnTransition = Callable[[AgentState, AgentState, str, float], None]

# Stable, test-asserted, logged verbatim.
REASON_LOW_HEALTH_IDLE = "low health on idle scan"
REASON_LOOT_MARKER = "loot marker visible"
REASON_HOSTILE_TARGET = "hostile target on frame"
REASON_TARGET_LOST = "target lost, exiting combat"
REASON_HEALTH_FLOOR_COMBAT = "health floor breached in combat"
REASON_LOOT_CLEARED = "loot cleared"
REASON_HEALTH_RECOVERED = "health recovered above floor"
REASON_RECOVERY_EXHAUSTED = "recovery attempts exhausted; safe pause"
REASON_STATE_TIMEOUT = "state timeout (watchdog)"
REASON_NO_ACTIONABLE_SIGNAL = "no actionable zone signal"


class GameStateMachine:
    """Deterministic state-transition core (no I/O, no Action creation).

    The injected ``policy`` (duck-typed; must expose
    ``transition_signals(observation) -> dict``) is the sole source of
    signal values — the GameAdapter boundary. The machine only enforces
    shape + invariants, keeping the test matrix small and the replay
    contract exact.
    """

    def __init__(
        self,
        *,
        policy: Any,
        initial: AgentState = AgentState.IDLE,
        recovery_attempts: int = 3,
        state_timeout_s: float = 5.0,
        watchdog_states: frozenset[AgentState] = frozenset(
            {AgentState.RECOVERY, AgentState.AI_ANALYSIS}
        ),
        on_transition: OnTransition | None = None,
    ) -> None:
        if recovery_attempts < 1:
            raise ValueError("recovery_attempts must be >= 1")
        if state_timeout_s <= 0:
            raise ValueError("state_timeout_s must be > 0")
        if initial not in set(AgentState):
            raise ValueError(f"unknown initial state: {initial!r}")
        if not watchdog_states <= set(AgentState):
            raise ValueError("watchdog_states must be a subset of AgentState")

        self.policy = policy
        self.state: AgentState = initial
        self.recovery_attempts = recovery_attempts
        self.state_timeout_s = state_timeout_s
        self.watchdog_states = frozenset(watchdog_states)
        self.on_transition = on_transition
        # The constructor has no clock input. None means the initial state's
        # entry time has not yet been anchored; the first decide(now) anchors
        # it instead of incorrectly treating process uptime as state duration.
        self.state_entered_at: float | None = None
        self._recovery_count: int = 0

    # ---- Public decision / bookkeeping API ----------------------------------

    def decide(self, observation: Any, now: float) -> AgentState:
        """Choose the next state from the current state + policy signals.

        No I/O. No Action creation. The injected ``now`` is the only time
        source (deterministic watchdog tests). Returns the (possibly
        unchanged) state.
        """
        # 1) Watchdog hard-override — only in watchdog states, checked before
        #    any per-state logic (it is the "only escape" from a stuck state).
        in_watchdog = self.state in self.watchdog_states
        if in_watchdog and self.state_entered_at is None:
            # The initial state was supplied before a clock existed. Start its
            # timeout at the first decision, never at an arbitrary zero epoch.
            self.state_entered_at = now
        timed_out = (
            self.state_entered_at is not None
            and now >= self.state_entered_at + self.state_timeout_s
        )
        if in_watchdog and timed_out:
            self._enter(AgentState.PAUSED, REASON_STATE_TIMEOUT, now)
            return AgentState.PAUSED

        # 2) Per-state dispatch over the policy's typed signals.
        signals = self.policy.transition_signals(observation)
        nxt, reason = self._next(self.state, signals)

        if nxt is self.state:
            return self.state  # self-loop idempotence: no event, no counter change

        self._enter(nxt, reason, now)
        return nxt

    def enter(self, new_state: AgentState, reason: str, now: float) -> None:
        """Controlled state writer (public).

        The loop uses this for forced moves ``decide()`` cannot express from
        an observation — e.g. a capture/perception failure this tick
        ("observation failure this tick" -> RECOVERY). Idempotent on
        self-loops (no event, no state change).
        """
        self._enter(new_state, reason, now)

    def note_recovery_attempt(self) -> int:
        """Count one RECOVERY attempt; returns the new 1-based count."""
        self._recovery_count += 1
        return self._recovery_count

    def reset_recovery(self) -> None:
        """Zero the RECOVERY attempt counter (healthy exit from RECOVERY)."""
        self._recovery_count = 0

    @property
    def recovery_count(self) -> int:
        """Current RECOVERY attempt count (0-based; read-only)."""
        return self._recovery_count

    def recovery_exhausted(self) -> bool:
        """True once the counter has reached ``recovery_attempts``."""
        return self._recovery_count >= self.recovery_attempts

    # ---- Internal ------------------------------------------------------------

    def _enter(self, new_state: AgentState, reason: str, now: float) -> None:
        if new_state is self.state:
            return  # self-loop: no event, no counter change, clock preserved
        before = self.state
        self.state = new_state
        self.state_entered_at = now
        # Recovery-episode management: entering RECOVERY starts a fresh
        # episode (counter 0); a healthy exit out of RECOVERY (to a
        # non-watchdog state) resets it.
        if new_state is AgentState.RECOVERY:
            self._recovery_count = 0
        elif before is AgentState.RECOVERY and new_state not in self.watchdog_states:
            self._recovery_count = 0
        if self.on_transition is not None:
            self.on_transition(before, new_state, reason, now)

    def _next(self, state: AgentState, signals: Signals) -> tuple[AgentState, str]:
        """Return ``(next_state, reason)`` for a genuine change, or
        ``(state, ...)`` for a self-loop. Table order follows plan §4.3."""
        if state is AgentState.IDLE:
            if self._is_true(signals, SIGNAL_HEALTH_LOW):
                return AgentState.RECOVERY, REASON_LOW_HEALTH_IDLE
            if self._is_true(signals, SIGNAL_LOOT_PRESENT):
                return AgentState.LOOTING, REASON_LOOT_MARKER
            if (
                self._is_true(signals, SIGNAL_TARGET_PRESENT)
                and self._is_true(signals, SIGNAL_TARGET_HOSTILE)
            ):
                return AgentState.COMBAT, REASON_HOSTILE_TARGET
            return AgentState.IDLE, REASON_NO_ACTIONABLE_SIGNAL
        if state is AgentState.COMBAT:
            # A present but non-hostile target is not the same as no target.
            # Only an explicit target_present=False ends combat; None means
            # detection is unknown and must preserve the current state.
            if self._is_false(signals, SIGNAL_TARGET_PRESENT):
                return AgentState.IDLE, REASON_TARGET_LOST
            if self._is_true(signals, SIGNAL_HEALTH_LOW):
                return AgentState.RECOVERY, REASON_HEALTH_FLOOR_COMBAT
            return AgentState.COMBAT, "still engaging hostile target"
        if state is AgentState.LOOTING:
            if self._is_false(signals, SIGNAL_LOOT_PRESENT):
                return AgentState.IDLE, REASON_LOOT_CLEARED
            return AgentState.LOOTING, "loot marker still present"
        if state is AgentState.RECOVERY:
            # Exhaustion wins "regardless of the observation."
            if self._recovery_count >= self.recovery_attempts:
                return AgentState.PAUSED, REASON_RECOVERY_EXHAUSTED
            if self._is_true(signals, SIGNAL_HEALTH_RECOVERED):
                return AgentState.IDLE, REASON_HEALTH_RECOVERED
            return AgentState.RECOVERY, "recovery in progress"
        # PAUSED (terminal-ish) and AI_ANALYSIS (phase-5 slot): stable; the
        # watchdog above is what can move them.
        return state, "state stable"

    @staticmethod
    def _is_true(signals: Signals, key: str) -> bool:
        """True only on an explicit ``True`` — ``None``/``False`` never match."""
        return signals.get(key) is True

    @staticmethod
    def _is_false(signals: Signals, key: str) -> bool:
        """True only on an explicit ``False`` — ``None``/``True`` never match."""
        return signals.get(key) is False
