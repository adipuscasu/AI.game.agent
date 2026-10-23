"""World of Warcraft adapter (Phase 4, step 3).

The only package in the repository that knows literal WoW zone names — and
those literals live in :mod:`ai_game_agent.games.wow.zones`, not here or
anywhere else.

Public re-exports so a later config-wiring step can do::

    from ai_game_agent.games.wow import DeterministicPolicy
"""

from ai_game_agent.games.wow.policy import DeterministicPolicy
from ai_game_agent.games.wow.zones import (
    ZONE_HEALTH_BAR,
    ZONE_LOOT_GLOW,
    ZONE_TARGET,
    ZONE_TARGET_HOSTILE,
)

__all__ = [
    "DeterministicPolicy",
    "ZONE_HEALTH_BAR",
    "ZONE_LOOT_GLOW",
    "ZONE_TARGET",
    "ZONE_TARGET_HOSTILE",
]
