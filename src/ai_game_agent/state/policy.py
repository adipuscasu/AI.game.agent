"""The ``StatePolicy`` contract (Phase 4, step 3).

The *Action Plan* seam between the state machine and the game adapter
(``docs/phase-4-implementation-plan.md`` §4.4). The state machine
(:mod:`ai_game_agent.state.machine`) and the agent loop (a later step)
depend only on this Protocol — neither knows anything about a specific game.

Game-agnostic rule: nothing in the ``state`` or ``agent`` layers may contain
game-specific literals (zone names, key bindings, …). Those live in the game
adapter package (e.g. :mod:`ai_game_agent.games.wow`). This module is that
rule made concrete: it declares the seam's *shape* and references no game at
all.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_game_agent.actions.action import Action
from ai_game_agent.actions.modes import OperationMode
from ai_game_agent.perception.observation import Observation
from ai_game_agent.state.states import AgentState

__all__ = ["StatePolicy"]


@runtime_checkable
class StatePolicy(Protocol):
    """What the agent loop needs from a game-specific action policy."""

    def name(self) -> str:
        """A stable identifier (e.g. ``"wow"``) for logging and replay."""
        ...

    def actions(
        self,
        state: AgentState,
        observation: Observation,
        mode: OperationMode,
    ) -> tuple[Action, ...]:
        """The validated action plan to execute while in ``state``.

        Must return ``()`` when ``mode.input_allowed`` is ``False``
        (``OBSERVE_ONLY``) — the policy is the first line of that hard gate
        (the executor and SafetyGuard back it up). Returned actions are fully
        validated (``Action`` enforces this at construction) and carry a
        non-empty ``source`` for provenance.
        """
        ...

    def transition_signals(self, observation: Observation) -> dict[str, object]:
        """Signals for :meth:`ai_game_agent.state.machine.GameStateMachine.decide`.

        A dict of stable signal names to ``bool | None``. ``None`` means
        "unknown this tick" (e.g. the detector for that signal failed, or no
        calibrated reading was available) — distinct from ``False`` (the
        detector ran cleanly and reported absence).
        """
        ...
