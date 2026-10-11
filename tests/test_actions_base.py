"""Step 2: input protocols, error hierarchy, and the ``create_backend`` factory.

All assertions run headless (no pynput, no desktop). The ``windows`` factory
path is tested for its *fail-fast error contract* — on a box without the
``input`` extra (true of CI/dev) it must raise a clean ``InputBackendError``
naming the missing extra, not a traceback. This mirrors Phase 1's
``create_backend('mss')`` behavior on a bare venv.
"""

from __future__ import annotations

import importlib.util

import pytest

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
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.config import InputConfig

# --- Error hierarchy (AGENTS.md ACTION_ERROR / INPUT_ERROR) -----------------


def test_error_hierarchy_subclasses_action_error():
    for cls in (
        ActionValidationError,
        ActionQueueError,
        InputBackendError,
        SafetyViolation,
        ModeViolation,
    ):
        assert issubclass(cls, ActionError), f"{cls.__name__} must subclass ActionError"


def test_action_error_is_an_exception():
    assert issubclass(ActionError, Exception)


# --- Protocols (structural, runtime_checkable) ------------------------------


def test_protocols_are_structural_and_mock_conforms():
    m = MockBackend()
    assert isinstance(m, InputBackend)
    assert isinstance(m, Keyboard)
    assert isinstance(m, Mouse)


def test_mock_backend_name():
    assert MockBackend().name == "mock"


# --- Factory ----------------------------------------------------------------


def test_create_backend_mock_returns_mock_backend():
    b = create_backend(InputConfig(backend="mock"))
    assert isinstance(b, MockBackend)
    assert isinstance(b, InputBackend)


def test_create_backend_normalizes_backend_name():
    # A stray case/whitespace in the config must not break selection.
    b = create_backend(InputConfig(backend="  MOCK "))
    assert isinstance(b, MockBackend)


def test_create_backend_unknown_raises_action_error():
    with pytest.raises(ActionError) as excinfo:
        create_backend(InputConfig(backend="doesnotexist"))
    assert "doesnotexist" in str(excinfo.value)


def test_create_backend_windows_without_pynput_fails_fast():
    if importlib.util.find_spec("pynput") is not None:
        pytest.skip("pynput present; windows path is covered by Windows manual testing")
    with pytest.raises(InputBackendError) as excinfo:
        create_backend(InputConfig(backend="windows"))
    msg = str(excinfo.value).lower()
    assert "input" in msg, "error must name the missing 'input' extra"
    assert "pynput" in msg, "error must name the missing library"
