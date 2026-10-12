"""The agent state layer (Phase 4).

Public re-exports so callers can do::

    from ai_game_agent.state import AgentState, AgentStateError

Step-1 scope: the ``AgentState`` enum and ``AgentStateError`` from
:mod:`ai_game_agent.state.states` only. The machine, events, and policy
modules land in subsequent phase-4 steps and are re-exported here as they
are added.
"""

from ai_game_agent.state.states import AgentState, AgentStateError

__all__ = ["AgentState", "AgentStateError"]
