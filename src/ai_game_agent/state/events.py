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
* :class:`LoopEvent` is a frozen dataclass (immutable field *bindings*) so an
  event can be logged, replayed, and compared without rebinding a field. Note
  ``frozen=True`` does not make the value deep-immutable: the ``detail`` dict
  is still a mutable container a caller may edit in place, and the presence of
  that unhashable dict field means :class:`LoopEvent` itself is **not**
  hashable. The audit-trail guarantee that matters — *serializing must not
  corrupt the source* — is provided by :meth:`LoopEvent.to_dict` deep-copying
  ``detail``.
* ``seq`` ownership (plan-internal, resolved): this step follows the
  step-4 contract bullet — 0-based, monotonically increasing *per machine
  instance*, assigned by the machine, not the callback. The plan's global
  note and the §4.2 sketch assign *session-wide* sequencing to the agent
  loop (step 5), which owns the authoritative, session-wide sequence across
  all event kinds; the machine's per-instance counter is provisional and
  test-scoped until then.
* ``detail`` is always a ``dict[str, JSON-compatible]``; ``to_dict()``
  yields a plain, ``json.dumps``-able mapping (enums as their string values),
  deep-copying ``detail`` so mutating the serialized result never leaks back
  into the event. Validating that ``detail`` values are in fact
  JSON-compatible is the step-8 record/replay obligation, not step 4's.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum

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
    ``seq``     — 0-based, monotonically increasing per machine instance
                  (assigned by the machine); session-wide authoritative
                  sequencing is owned by the agent loop (step 5 — see the
                  module docstring).
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

    def to_dict(self) -> dict[str, object]:
        """A plain, ``json.dumps``-able mapping of this event.

        ``detail`` is deep-copied: mutating the returned mapping (top level
        or nested) never leaks back into the event — closing the gap
        ``frozen=True`` cannot cover, since it freezes the field binding, not
        the dict the field points at. Enums are rendered as their string
        values so the result round-trips through JSON unchanged
        (``json.loads(json.dumps(ev.to_dict()))``).
        """
        return {
            "ts": self.ts,
            "seq": self.seq,
            "event": self.event.value,
            "state_before": None if self.state_before is None else self.state_before.value,
            "state_after": None if self.state_after is None else self.state_after.value,
            "reason": self.reason,
            "detail": deepcopy(self.detail),
        }
