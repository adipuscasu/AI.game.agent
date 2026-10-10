"""Tests for the action contract (Phase 3, step 1).

The contract (``Action``, ``ActionResult``, ``ActionLog``) must be:

* immutable (frozen dataclasses),
* JSON-serializable via ``to_dict()`` / ``to_json()`` with plain values only,
* validated at construction (an invalid action is unconstructable),
* versioned (``SCHEMA_VERSION``),
* importable in a bare venv (stdlib only).

This is the Phase-3 analog of Phase 1's ``Frame`` and Phase 2's
``Observation``. Every action type the project supports — keyboard, mouse
move, click, scroll, hold, and delay — is part of this contract.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from ai_game_agent.actions import (
    SCHEMA_VERSION,
    Action,
    ActionError,
    ActionKind,
    ActionLog,
    ActionResult,
    ActionValidationError,
)

# ---------------------------------------------------------------------------
# Contract basics
# ---------------------------------------------------------------------------


def test_all_action_kinds_defined() -> None:
    assert {k.value for k in ActionKind} == {
        "key_press",
        "key_hold",
        "mouse_move",
        "mouse_click",
        "mouse_down",
        "mouse_up",
        "mouse_scroll",
        "delay",
    }


def test_valid_key_press_constructs() -> None:
    action = Action(kind=ActionKind.KEY_PRESS, key="f")
    assert action.key == "f"
    assert action.modifiers == ()
    assert action.schema_version == SCHEMA_VERSION


def test_valid_key_press_with_modifiers() -> None:
    action = Action(kind=ActionKind.KEY_PRESS, key="v", modifiers=("ctrl",))
    assert action.modifiers == ("ctrl",)


def test_valid_key_hold_constructs() -> None:
    action = Action(kind=ActionKind.KEY_HOLD, key="w", duration_ms=250)
    assert action.duration_ms == 250


def test_valid_mouse_move_constructs() -> None:
    action = Action(kind=ActionKind.MOUSE_MOVE, x=100, y=200)
    assert (action.x, action.y) == (100, 200)
    assert action.relative is False


def test_valid_mouse_click_constructs() -> None:
    action = Action(kind=ActionKind.MOUSE_CLICK, button="right", clicks=2)
    assert action.button == "right"
    assert action.clicks == 2


def test_valid_mouse_scroll_constructs() -> None:
    action = Action(kind=ActionKind.MOUSE_SCROLL, scroll=-3)
    assert action.scroll == -3


def test_valid_delay_constructs() -> None:
    action = Action(kind=ActionKind.DELAY, duration_ms=100)
    assert action.duration_ms == 100


# ---------------------------------------------------------------------------
# Validation: required / forbidden fields per kind
# ---------------------------------------------------------------------------


def test_key_press_requires_key() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_PRESS)


def test_key_press_forbids_mouse_fields() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_PRESS, key="w", x=1, y=2)


def test_key_press_forbids_hold_duration() -> None:
    # A timed press is a KEY_HOLD; the two kinds must stay distinct.
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_PRESS, key="w", duration_ms=100)


def test_key_hold_requires_positive_duration() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_HOLD, key="w", duration_ms=0)


def test_mouse_move_requires_both_coords() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_MOVE, x=10)
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_MOVE, y=10)


def test_mouse_move_rejects_negative_absolute_coords() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_MOVE, x=-1, y=10)


def test_mouse_move_relative_allows_negative_deltas() -> None:
    action = Action(
        kind=ActionKind.MOUSE_MOVE, x=-30, y=12, relative=True
    )
    assert (action.x, action.y) == (-30, 12)


def test_mouse_move_forbids_extra_clicks() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_MOVE, x=1, y=2, clicks=2)


def test_mouse_click_requires_positive_clicks() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_CLICK, clicks=0)


def test_mouse_click_rejects_invalid_button() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_CLICK, button="up")


def test_mouse_click_forbids_relative_position() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_CLICK, x=1, y=2, relative=True)


def test_mouse_down_requires_known_button() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_DOWN, button="middle2")


def test_mouse_up_rejects_position() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_UP, x=5, y=5)


def test_mouse_scroll_requires_nonzero_scroll() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_SCROLL)
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_SCROLL, scroll=0)


def test_mouse_scroll_rejects_position() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_SCROLL, scroll=1, x=1, y=2)


def test_delay_requires_positive_duration() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.DELAY, duration_ms=0)


def test_delay_forbids_input_fields() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.DELAY, duration_ms=10, key="w")


# ---------------------------------------------------------------------------
# Validation: field-level rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["", "A", "a b", "ctrl+shift", "F12"],
)
def test_invalid_key_name_rejected(key: str) -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_PRESS, key=key)


def test_named_keys_accepted() -> None:
    for key in ("a", "z", "0", "9", "f1", "f12", "space", "enter", "tab", "page_up"):
        Action(kind=ActionKind.KEY_PRESS, key=key)


@pytest.mark.parametrize("modifiers", [("bogus",), ("ctrl", "CTRL"), ("w",)])
def test_invalid_modifier_rejected(modifiers: tuple[str, ...]) -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_PRESS, key="v", modifiers=modifiers)


def test_duplicate_modifiers_rejected() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_PRESS, key="v", modifiers=("ctrl", "ctrl"))


def test_modifier_cannot_be_the_key() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_PRESS, key="shift", modifiers=("shift",))


def test_negative_duration_rejected() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_HOLD, key="w", duration_ms=-1)


def test_negative_delay_rejected() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_PRESS, key="w", delay_ms=-5)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_out_of_bounds_rejected(confidence: float) -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_PRESS, key="w", confidence=confidence)


def test_confidence_bounds_accepted() -> None:
    assert Action(kind=ActionKind.KEY_PRESS, key="w", confidence=0.0).confidence == 0.0
    assert Action(kind=ActionKind.KEY_PRESS, key="w", confidence=1.0).confidence == 1.0


def test_empty_source_rejected() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.KEY_PRESS, key="w", source="")


def test_bool_rejected_as_int_coord() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_MOVE, x=True, y=1)  # type: ignore[arg-type]


def test_bool_rejected_as_clicks() -> None:
    with pytest.raises(ActionValidationError):
        Action(kind=ActionKind.MOUSE_CLICK, clicks=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_action_is_frozen() -> None:
    action = Action(kind=ActionKind.KEY_PRESS, key="w")
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.key = "a"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_action_to_dict_has_plain_values() -> None:
    action = Action(
        kind=ActionKind.KEY_HOLD,
        key="w",
        modifiers=("ctrl", "shift"),
        x=None,
        y=None,
        relative=False,
        button="left",
        clicks=1,
        scroll=None,
        duration_ms=250,
        delay_ms=10,
        source="test",
        confidence=0.9,
    )
    data = action.to_dict()
    # Every value must be a JSON plain type (str / int / float / bool / None /
    # list) — no enums, no tuples, no dataclasses.
    serialized = json.dumps(data)
    assert isinstance(serialized, str)
    assert data["kind"] == "key_hold"
    assert data["modifiers"] == ["ctrl", "shift"]


def test_action_to_json_round_trips_via_from_dict() -> None:
    action = Action(
        kind=ActionKind.MOUSE_SCROLL,
        scroll=-3,
        source="replay",
        confidence=0.5,
    )
    restored = Action.from_dict(json.loads(action.to_json()))
    assert restored == action


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ActionValidationError):
        Action.from_dict({"kind": "key_press", "key": "w", "rogue_field": 1})


def test_from_dict_accepts_enum_or_string_kind() -> None:
    via_enum = Action.from_dict({"kind": ActionKind.KEY_PRESS, "key": "w"})
    via_str = Action.from_dict({"kind": "key_press", "key": "w"})
    assert via_enum == via_str


def test_from_dict_requires_kind() -> None:
    with pytest.raises(ActionValidationError):
        Action.from_dict({"key": "w"})


def test_from_dict_rejects_non_dict() -> None:
    with pytest.raises(ActionValidationError):
        Action.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_action_result_to_dict_and_json() -> None:
    action = Action(kind=ActionKind.KEY_PRESS, key="w")
    result = ActionResult(action=action, ok=False, error="blocked", duration_ms=1.5)
    data = result.to_dict()
    assert data["ok"] is False
    assert data["error"] == "blocked"
    assert data["stop_event"] is False
    json.loads(result.to_json())


def test_action_result_round_trips() -> None:
    action = Action(kind=ActionKind.MOUSE_CLICK, x=5, y=6)
    result = ActionResult(action=action, ok=True, duration_ms=3.0)
    restored = ActionResult.from_dict(json.loads(result.to_json()))
    assert restored == result


def test_action_log_to_dict_and_json() -> None:
    log = ActionLog(
        results=(
            ActionResult(action=Action(kind=ActionKind.KEY_PRESS, key="a"), ok=True),
            ActionResult(
                action=Action(kind=ActionKind.MOUSE_CLICK),
                ok=False,
                error="safety violation",
            ),
        ),
        stopped=True,
        stop_reason="emergency stop (f12)",
    )
    data = log.to_dict()
    assert data["stopped"] is True
    assert data["stop_reason"] == "emergency stop (f12)"
    assert len(data["results"]) == 2
    parsed = json.loads(log.to_json())
    assert parsed["results"][1]["ok"] is False


def test_action_log_normalizes_list_to_tuple() -> None:
    log = ActionLog(
        results=[ActionResult(action=Action(kind=ActionKind.DELAY, duration_ms=1), ok=True)]
    )
    assert isinstance(log.results, tuple)


def test_action_log_round_trips() -> None:
    log = ActionLog(
        results=(
            ActionResult(
                action=Action(kind=ActionKind.KEY_PRESS, key="w", modifiers=("ctrl",)),
                ok=True,
                duration_ms=2.0,
            ),
        )
    )
    restored = ActionLog.from_dict(json.loads(log.to_json()))
    assert restored == log


def test_schema_version_defaults() -> None:
    assert SCHEMA_VERSION == 1
    assert Action(kind=ActionKind.DELAY, duration_ms=1).schema_version == SCHEMA_VERSION
    assert ActionLog(results=()).schema_version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


def test_error_hierarchy() -> None:
    assert issubclass(ActionValidationError, ActionError)
    assert issubclass(ActionError, Exception)
    with pytest.raises(ActionError):
        raise ActionValidationError("contract violation")


# ---------------------------------------------------------------------------
# Bare-venv import contract (stdlib only)
# ---------------------------------------------------------------------------


def test_package_imports_contract() -> None:
    import ai_game_agent.actions as actions

    assert actions.Action is Action
    assert actions.ActionKind is ActionKind
    assert actions.ActionLog is ActionLog
    assert actions.ActionResult is ActionResult
    assert actions.ActionValidationError is ActionValidationError
    assert actions.SCHEMA_VERSION == SCHEMA_VERSION
