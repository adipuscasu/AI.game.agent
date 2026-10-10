"""Tests for ai_game_agent.actions.queue (Phase 3, step 4).

TDD: red first (module missing), then green. Covers the plan's
ActionQueue API: add / pop_next / clear, duplicate rejection, and
idempotent clear.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError  # noqa: F401  (kept for parity with config tests)

from ai_game_agent.actions import ActionQueue, ActionQueueError, ActionRequest
from ai_game_agent.actions.action import Action, ActionKind, ActionValidationError
from ai_game_agent.actions.mock import MockBackend


def _req(kind: ActionKind, **action_kwargs: object) -> ActionRequest:
    """Build an ActionRequest from a kind + arbitrary action kwargs."""
    action = Action(kind=kind, **action_kwargs)  # type: ignore[arg-type]
    return ActionRequest(action=action, requested_at=datetime(2026, 10, 10, 10, 30))


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend()


def test_add_and_pop_next_executes_in_order(backend: MockBackend) -> None:
    q = ActionQueue(backend)
    r1 = _req(ActionKind.MOUSE_MOVE, x=10, y=20)
    r2 = _req(ActionKind.MOUSE_CLICK, button="left", clicks=1)
    q.add(r1)
    q.add(r2)
    assert q.pending_count() == 2

    first = q.pop_next()
    assert first is not None and first.action.kind is ActionKind.MOUSE_MOVE
    second = q.pop_next()
    assert second is not None and second.action.kind is ActionKind.MOUSE_CLICK
    assert q.pending_count() == 0
    assert q.pop_next() is None


def test_add_rejects_duplicate_pending_kind(backend: MockBackend) -> None:
    q = ActionQueue(backend)
    first = _req(ActionKind.MOUSE_CLICK, button="left", clicks=1)
    second = _req(ActionKind.MOUSE_CLICK, button="left", clicks=2)
    q.add(first)
    with pytest.raises(ActionQueueError, match="duplicate"):
        q.add(second)
    # The first request stays pending, unchanged.
    assert q.pending_count() == 1
    assert q.pop_next() is first


def test_clear_is_idempotent(backend: MockBackend) -> None:
    q = ActionQueue(backend)
    q.add(_req(ActionKind.MOUSE_MOVE, x=1, y=2))
    assert q.clear() == 1
    assert q.pending_count() == 0
    assert q.clear() == 0  # second clear is a no-op, not an error


def test_action_payload_validation_fails_fast() -> None:
    # Invalid params are rejected at Action construction (fail fast),
    # before anything reaches the queue.
    with pytest.raises(ActionValidationError):
        _req(ActionKind.MOUSE_MOVE)  # missing required x / y


def test_action_request_is_immutable() -> None:
    req = _req(ActionKind.MOUSE_MOVE, x=1, y=2)
    with pytest.raises((TypeError, AttributeError)):
        req.action = Action(kind=ActionKind.MOUSE_CLICK, button="left")  # type: ignore[call-arg]


def test_pending_kinds_reflect_fifo_order(backend: MockBackend) -> None:
    q = ActionQueue(backend)
    q.add(_req(ActionKind.KEY_PRESS, key="a"))
    q.add(_req(ActionKind.MOUSE_MOVE, x=1, y=2))
    assert q.pending_kinds() == ("key_press", "mouse_move")
