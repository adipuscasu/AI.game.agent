"""The agent orchestration layer (Phase 4, step 5).

Public re-exports so callers can do::

    from ai_game_agent.agent import AgentLoop, AgentLoopReport
"""

from ai_game_agent.agent.loop import AgentLoop, TickResult
from ai_game_agent.agent.report import AgentLoopReport

__all__ = ["AgentLoop", "AgentLoopReport", "TickResult"]
