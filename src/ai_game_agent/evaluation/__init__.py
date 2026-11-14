"""EH4: versioned perception-evaluation dataset and reproducible scoring.

This package is deliberately game-agnostic: it knows the ``Observation``
perception contract (what a detection *is*) and a generic
``signals -> (state, reason)`` decision seam, but no specific game. Game
adapters plug in their state machine at evaluation time.
"""
