"""Structured event log (Phase 4, step 4, plan §4.2).

One frozen :class:`LoopEvent` per tick/decision, append-only per session.
This is the artifact that answers *"why did the agent do this?"* and that
the step-8 replay tests assert against.

Scope (step 4): the event model — a closed :class:`EventType` vocabulary and
the frozen, JSON-serializable :class:`LoopEvent`. Emission of
``STATE_ENTER`` events is wired into :class:`GameStateMachine` (``machine.py``);
``TICK``/``ACTION_*``/``RECOVERY``/``STOP``/``ERROR`` are produced by the
agent loop (step 5) and the executor, not by the decision core.

Design notes:

* :class:`EventType` is a closed ``str, Enum`` (``StrEnum``) — the same
  discipline as :class:`AgentState` and ``OperationMode``: string
  interoperability with config and JSON, a closed value set, and an implicit
  coercion to an unknown value is impossible (``EventType("bogus")`` raises).
* :class:`LoopEvent` is a frozen dataclass (immutable, hashable) so an event
  is a value that can be logged, replayed, and compared without mutation.
* ``seq`` is 0-based and assigned by the machine that emits the event — not
  by the receiving callback — so the machine owns a well-defined session
  sequence.
* ``detail`` is always a ``dict[str, JSON-compatible]``; ``to_dict()`` yields
  a plain, ``json.dumps``-able mapping (enums as their string values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ai_game_agent.state.states import AgentState

__all__ = ["EventType", "LoopEvent"]


class EventType(StrEnum):
    """Closed event vocabulary (plan §4.2).

    Every tick, every transition, every executed action batch, every error,
    and every stop writes ≥1 :class:`LoopEvent` naming one of these. The
    set is closed: an unknown value is a hard error, never an implicit value.
    """

    TICK = "tick"
    STATE_ENTER = "state_enter"
    STATE_EXIT = "state_exit"
    ACTION_BATCH = "action_batch"
    ACTION_RESULT = "action_result"
    RECOVERY = "recovery"
    STOP = "stop"
    ERROR = "error"

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"invalid event type: {value!r}")


@dataclass(frozen=True)
class LoopEvent:
    """One immutable, append-only event in the session audit trail.

    ``ts``      — monotonic-safe epoch seconds (injected clock; deterministic).
    ``seq``     — 0-based, per-machine session sequence (assigned by the machine).
    ``event``   — the closed :class:`EventType`.
    ``state_before`` / ``state_after`` — the transition endpoints; either may
                  be ``None`` (e.g. a ``TICK`` with no state change, or a
                  ``STOP`` emitted before the loop starts).
    ``reason``  — human-readable "why" (the AGENTS.md log requirement).
    ``detail``  — structured, JSON-compatible payload (zone values, action
                  names, error class, attempt index). Always a plain dict.
    """

    ts: float
    seq: int
    event: EventType
    state_before: AgentState | None
    state_after: AgentState | None
    reason: str
    detail: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """A plain, ``json.dumps``-able mapping of this event.

        Enums are rendered as their string values so the result round-trips
        through JSON unchanged (``json.loads(json.dumps(ev.to_dict()))``).
        """
        return {
            "ts": self.ts,
            "seq": self.seq,
            "event": self.event.value,
            "state_before": None if self.state_before is None else self.state_before.value,
            "state_after": None if self.state_after is None else self.state_after.value,
            "reason": self.reason,
            "detail": dict(self.detail),
        }
