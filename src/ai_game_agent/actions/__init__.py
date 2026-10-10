"""The action-execution layer (Phase 3).

Public re-exports so callers can do::

    from ai_game_agent.actions import Action, ActionKind, Executor

The contract (:mod:`ai_game_agent.actions.action`), error hierarchy
(:mod:`ai_game_agent.actions.base`), queue, executor, safety, and modes are
dependency-free (stdlib + the project's own types) so importing this package
in a bare venv is always safe. Real OS input (pynput) is imported lazily by
the Windows adapter only when the ``windows`` backend is selected.
"""

from ai_game_agent.actions.action import (
    SCHEMA_VERSION,
    Action,
    ActionKind,
    ActionLog,
    ActionResult,
    validate_action,
)
from ai_game_agent.actions.base import (
    ActionError,
    ActionQueueError,
    ActionValidationError,
    InputBackend,
    InputBackendError,
    Keyboard,
    ModeViolation,
    Mouse,
    SafetyViolation,
    create_backend,
)
from ai_game_agent.actions.executor import Executor
from ai_game_agent.actions.modes import OperationMode
from ai_game_agent.actions.queue import ActionQueue, ActionRequest
from ai_game_agent.actions.safety import ExecutorState, SafetyGuard

__all__ = [
    "Action",
    "ActionError",
    "ActionKind",
    "ActionLog",
    "ActionQueueError",
    "ActionResult",
    "ActionValidationError",
    "Executor",
    "ExecutorState",
    "InputBackend",
    "InputBackendError",
    "Keyboard",
    "ModeViolation",
    "Mouse",
    "OperationMode",
    "SCHEMA_VERSION",
    "SafetyGuard",
    "SafetyViolation",
    "ActionQueue",
    "ActionRequest",
    "create_backend",
    "validate_action",
]
