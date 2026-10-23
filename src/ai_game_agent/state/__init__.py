"""The agent state layer (Phase 4).

Public re-exports so callers can do::

    from ai_game_agent.state import AgentState, AgentStateError

Step-1 scope: the ``AgentState`` enum and ``AgentStateError`` from
:mod:`ai_game_agent.state.states`. Step-2 scope: the ``GameStateMachine``
decision core from :mod:`ai_game_agent.state.machine`. Step-3 scope: the
``StatePolicy`` contract from :mod:`ai_game_agent.state.policy`. Step-4 scope:
the ``LoopEvent`` / ``EventType`` audit-trail model from
:mod:`ai_game_agent.state.events` (plan §4.2).
"""

from ai_game_agent.state.events import EventType, LoopEvent
from ai_game_agent.state.machine import GameStateMachine
from ai_game_agent.state.policy import StatePolicy
from ai_game_agent.state.states import AgentState, AgentStateError

__all__ = [
    "AgentState",
    "AgentStateError",
    "EventType",
    "GameStateMachine",
    "LoopEvent",
    "StatePolicy",
]
