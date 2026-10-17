"""AgentLoopReport (Phase 4, step 5) — the per-session summary.

One frozen, JSON-serializable snapshot of a completed (or stopped)
``AgentLoop.run()``: how many ticks ran, which state transitions fired, how
many actions were attempted / ok / failed, which errors occurred, the final
machine state, *why* the loop stopped, and how long it ran.

Mirrors the ``Observation`` contract: frozen field *bindings* + a ``to_dict()``
that yields a plain, ``json.dumps``-able mapping (enums as their string
values; nested ``LoopEvent``s via their own ``to_dict``, which deep-copies
``detail``). This is the artifact the CLI ``run`` subcommand (step 7) prints
and the step-8 replay tests diff against.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_game_agent.state import AgentState, LoopEvent

__all__ = ["AgentLoopReport"]


@dataclass(frozen=True)
class AgentLoopReport:
    """A per-session summary of one ``AgentLoop.run()`` call.

    ``ticks``              — how many full ``tick()`` iterations completed.
    ``transitions``        — every ``STATE_ENTER`` event the machine fired.
    ``actions_attempted``  — total :class:`Action`s dispatched to the executor.
    ``actions_ok``         — how many of those returned ``ok=True``.
    ``actions_failed``     — how many returned ``ok=False`` (or raised).
    ``errors``             — every ``RECOVERY`` / ``ERROR`` event recorded.
    ``final_state``        — the machine state when the loop exited.
    ``stopped_reason``     — ``"max_ticks"`` / ``"stop_requested"`` /
                            ``"paused"`` / ``"capture_start_failed"``.
    ``wall_time_s``        — wall-clock duration of the ``run()`` call.
    """

    ticks: int
    transitions: tuple[LoopEvent, ...]
    actions_attempted: int
    actions_ok: int
    actions_failed: int
    errors: tuple[LoopEvent, ...]
    final_state: AgentState
    stopped_reason: str | None
    wall_time_s: float

    def to_dict(self) -> dict[str, object]:
        """A plain, ``json.dumps``-able mapping of this report."""
        return {
            "ticks": self.ticks,
            "transitions": [e.to_dict() for e in self.transitions],
            "actions_attempted": self.actions_attempted,
            "actions_ok": self.actions_ok,
            "actions_failed": self.actions_failed,
            "errors": [e.to_dict() for e in self.errors],
            "final_state": self.final_state.value,
            "stopped_reason": self.stopped_reason,
            "wall_time_s": self.wall_time_s,
        }
