"""Game adapters (Phase 4, step 3).

Game-specific code lives here, one sub-package per game (``wow`` is the
initial adapter). The dependency direction is one-way: a game adapter may
import the generic core contracts (``state``, ``actions``, ``perception``),
but the core never imports from ``games``. Zone-name literals and other
game-specific strings are confined to the game sub-package (for WoW,
``games/wow/zones.py``); the generic layers reference only stable signals
and abstractions.
"""
