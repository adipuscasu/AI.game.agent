"""Operating modes (Phase 3, step 3).

The mode the executor runs in. ``OperationMode`` is the single source of
truth for the mode string — the config layer validates ``input.mode`` against
its values, and the :class:`~ai_game_agent.actions.safety.SafetyGuard` gate
switches on it.

Phase-3 scope (docs/phase-3-implementation-plan.md §5.6):

* ``OBSERVE_ONLY``      — never emits input (hard gate; built + regression-tested).
* ``ASSISTED``          — propose; execute only on approval (mechanism only).
* ``SEMI_AUTONOMOUS``   — execute safe actions; gate unknown (mechanism only).
* ``AUTONOMOUS``        — full execute path.
"""

from __future__ import annotations

from enum import StrEnum


class OperationMode(StrEnum):
    """The four operating modes (AGENTS.md Human-in-the-Loop)."""

    OBSERVE_ONLY = "observe_only"
    ASSISTED = "assisted"
    SEMI_AUTONOMOUS = "semi_autonomous"
    AUTONOMOUS = "autonomous"

    @property
    def input_allowed(self) -> bool:
        """Whether the mode may emit OS input at all.

        ``OBSERVE_ONLY`` is the only mode for which this is ``False``; it is
        the one mode the whole project's safety story depends on. The other
        three are all "input allowed" — *how* they gate individual actions is
        the SafetyGuard's job, not the mode's.
        """
        return self is not OperationMode.OBSERVE_ONLY


__all__ = ["OperationMode"]
