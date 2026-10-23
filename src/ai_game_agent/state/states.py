"""Agent states (Phase 4, step 1).

The closed set of states the state machine (step 2), policy (step 3), and
agent loop (step 5) all share. ``AgentState`` is the single source of truth
for the state vocabulary — same discipline as ``OperationMode`` (Phase 3):
``StrEnum`` for string interoperability with config and JSON, a closed value
set, and an explicit error type for invalid values instead of implicit
coercion.

Step-1 scope: the enum and its error type only. Transition rules, recovery
counters, event logging, and the agent loop land in subsequent phase-4
steps.
"""

from __future__ import annotations

from enum import StrEnum


class AgentStateError(ValueError):
    """Raised when an invalid agent state is requested."""


class AgentState(StrEnum):
    """The six agent states (phase-4 plan §4.1).

    * ``IDLE``        — no known signal; the default resting state.
    * ``COMBAT``      — a hostile target is present; the combat plan applies.
    * ``LOOTING``     — a loot marker is present; the loot plan applies.
    * ``RECOVERY``    — a known-but-unhandled situation; bounded retry, then
      ``PAUSED``.
    * ``AI_ANALYSIS`` — Phase 5 slot; Phase 4 never *enters* it — it falls
      through to ``RECOVERY``/``PAUSED`` instead.
    * ``PAUSED``      — terminal-ish safe state: executor stopped, loop exits.
    """

    IDLE = "idle"
    COMBAT = "combat"
    LOOTING = "looting"
    RECOVERY = "recovery"
    AI_ANALYSIS = "ai_analysis"
    PAUSED = "paused"

    @classmethod
    def _missing_(cls, value: object) -> AgentState:
        raise AgentStateError(f"invalid agent state: {value!r}")


__all__ = ["AgentState", "AgentStateError"]
