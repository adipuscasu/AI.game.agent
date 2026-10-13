"""Canonical WoW zone-name literals (Phase 4, step 3).

The *only* place in the repository where these strings appear as literals.
Every other module (state, agent, tests) references them through these
constants. Renaming a zone in a config file is a one-line change here — no
other code is touched, and no zone literal leaks into the generic layers.
"""

from __future__ import annotations

ZONE_HEALTH_BAR = "health_bar"
ZONE_LOOT_GLOW = "loot_glow"
ZONE_TARGET = "target"
ZONE_TARGET_HOSTILE = "target_hostile"

__all__ = [
    "ZONE_HEALTH_BAR",
    "ZONE_LOOT_GLOW",
    "ZONE_TARGET",
    "ZONE_TARGET_HOSTILE",
]
