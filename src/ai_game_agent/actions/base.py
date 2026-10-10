"""Action-layer error hierarchy (Phase 3, step 1).

Dependency-free. Per AGENTS.md this maps to the ``ACTION_ERROR`` /
``INPUT_ERROR`` error classes. Every action-layer error inherits from
:class:`ActionError` so callers can catch a single type, while subclasses let
a specific layer (validation, queue, backend, safety, mode) classify the
failure precisely.

The rest of the action layer (protocols + factory) is added in step 2/3 by
extending this module; step 1 intentionally ships only the errors and the
contract they guard.
"""

from __future__ import annotations

__all__ = [
    "ActionError",
    "ActionQueueError",
    "ActionValidationError",
    "InputBackendError",
    "ModeViolation",
    "SafetyViolation",
]


class ActionError(Exception):
    """Base class for all action-layer errors (the ``ACTION_ERROR`` class)."""


class ActionValidationError(ActionError):
    """An action failed contract validation (bad field set for its kind).

    Raised at construction / deserialization time, before any backend is
    touched. This is the "the model proposes; application code validates it"
    gate from AGENTS.md.
    """


class ActionQueueError(ActionError):
    """The action queue rejected an action (bound exceeded, queue full)."""


class InputBackendError(ActionError):
    """The input backend I/O failed (the ``INPUT_ERROR`` class).

    Includes "the backend dependency is not installed" (fail-fast with a clear
    extra name) and runtime backend I/O failures.
    """


class SafetyViolation(ActionError):
    """An action would exceed a configured safety bound.

    In ``ASSISTED`` mode this is *reported*, not executed; in
    ``AUTONOMOUS`` / ``SEMI_AUTONOMOUS`` it pauses the batch and is recorded.
    """


class ModeViolation(ActionError):
    """Input was attempted in a mode that forbids it (e.g. ``OBSERVE_ONLY``).

    Always raised and never "recovered": it is a contract violation, logged as
    a safety event. A missing/invalid mode fails *closed* (raises this) rather
    than defaulting to an input-enabled mode.
    """
