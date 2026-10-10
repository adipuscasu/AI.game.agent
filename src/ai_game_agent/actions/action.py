"""The action contract (Phase 3, step 1).

This is the Phase-3 analog of Phase 1's ``Frame`` and Phase 2's
``Observation``: the stable, dependency-free types that the action layer
(Phase 3), the state machine (Phase 4), the AI output parsing (Phase 5), and
the assisted-approval mechanism (Phase 6) all program against.

Contract rules:

* ``Action``, ``ActionResult``, and ``ActionLog`` are **frozen** dataclasses —
  immutable and hashable.
* Every one is **JSON-serializable** via ``to_dict()`` / ``to_json()`` with
  plain values only (str / int / float / bool / None / list) and
  **reconstructible** via ``from_dict()``.
* An invalid action is **unconstructable**: ``Action(...)`` and
  ``Action.from_dict(...)`` both raise :class:`ActionValidationError`. There is
  no way to hold an action whose field set is wrong for its kind.
* Coordinates are **screen space** (the action layer operates on the real
  desktop), unlike ``Observation`` which is frame space.
* Nothing here imports a keyboard/mouse library. The contract, validation, and
  serialization are stdlib-only so they import and test in a bare venv.

The per-kind validation table (``_KIND_RULES``) is the single source of truth
for "which fields each action kind requires / forbids." The executor, the CLI
``--script`` loader, and replay all validate against these same rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from ai_game_agent.actions.base import ActionValidationError

__all__ = [
    "Action",
    "ActionKind",
    "ActionLog",
    "ActionResult",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = 1  # bump if the serialized shape changes meaningfully

# Field names (not values) each kind may *carry*. Everything else must be its
# default. This is the "forbidden" half of the per-kind contract.
#
# NOTE: a field may be *carried* yet still be value-invalid (e.g. ``duration_ms``
# negative, ``key`` not a real key name); the field-name table plus the
# value-level checks in ``_validate_field_values`` together define the contract.
_KIND_FIELDS: dict[str, frozenset[str]] = {
    "key_press": frozenset({"key", "modifiers", "delay_ms", "source", "confidence"}),
    "key_hold": frozenset(
        {"key", "modifiers", "duration_ms", "delay_ms", "source", "confidence"}
    ),
    "mouse_move": frozenset(
        {"x", "y", "relative", "delay_ms", "source", "confidence"}
    ),
    "mouse_click": frozenset(
        {"x", "y", "button", "clicks", "delay_ms", "source", "confidence"}
    ),
    "mouse_down": frozenset({"button", "delay_ms", "source", "confidence"}),
    "mouse_up": frozenset({"button", "delay_ms", "source", "confidence"}),
    "mouse_scroll": frozenset({"scroll", "delay_ms", "source", "confidence"}),
    "delay": frozenset({"duration_ms", "delay_ms", "source", "confidence"}),
}

# Fields whose *absence* (i.e. left at the default) is illegal for the kind.
_KIND_REQUIRED: dict[str, tuple[str, ...]] = {
    "key_press": ("key",),
    "key_hold": ("key",),
    "mouse_move": ("x", "y"),
    "mouse_click": (),
    "mouse_down": (),
    "mouse_up": (),
    "mouse_scroll": ("scroll",),
    "delay": (),
}

# Defaults — the values a field must hold when the kind does not use it.
_DEFAULTS = {
    "key": None,
    "modifiers": (),
    "x": None,
    "y": None,
    "relative": False,
    "button": "left",
    "clicks": 1,
    "scroll": None,
    "duration_ms": 0,
    "delay_ms": 0,
    "source": "user",
    "confidence": None,
}

_VALID_BUTTONS = frozenset({"left", "right", "middle"})
_VALID_MODIFIERS = frozenset({"ctrl", "alt", "shift", "win"})

# Named keys (lowercase) accepted by pynput's key system. Single letters and
# digits are additionally accepted by their shape, so only the *named* keys
# (and the digits check) are listed here.
_NAMED_KEYS = frozenset(
    {
        "space", "enter", "tab", "backspace", "delete", "insert", "home",
        "end", "page_up", "page_down", "up", "down", "left", "right",
        "esc", "caps_lock", "num_lock", "scroll_lock", "print_screen",
        "pause", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
        "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20",
        "f21", "f22", "f23", "f24",
        "comma", "period", "slash", "semicolon", "apostrophe", "bracket_left",
        "bracket_right", "backslash", "minus", "equal", "grave", "numpad_0",
        "numpad_1", "numpad_2", "numpad_3", "numpad_4", "numpad_5", "numpad_6",
        "numpad_7", "numpad_8", "numpad_9", "numpad_multiply",
        "numpad_add", "numpad_subtract", "numpad_decimal", "numpad_divide",
    }
)


def _is_int(value: object) -> bool:
    """True for a real int (``bool`` is excluded — it subclasses ``int``)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_key_name(key: object) -> bool:
    """True for a pynput-compatible key name: strict lowercase.

    Single letters/digits are valid by shape; everything else must be a named
    key. Uppercase (``"A"``, ``"F12"``) is rejected: the backend speaks
    lowercase key names, so the contract is strict rather than lenient.
    """
    if not isinstance(key, str) or not key:
        return False
    if key != key.lower():
        return False
    if len(key) == 1:
        return key.isalpha() or key.isdigit()
    return key in _NAMED_KEYS


class ActionKind(StrEnum):
    """The set of discrete, deterministic input operations.

    Every kind the project supports is part of the contract, including scroll
    (required for the OBSERVE_ONLY zero-event regression test) and timed holds.
    """

    KEY_PRESS = "key_press"       # key (or modifiers+key) down then up
    KEY_HOLD = "key_hold"         # key down, hold for duration_ms, up
    MOUSE_MOVE = "mouse_move"     # move to (x, y); absolute or relative
    MOUSE_CLICK = "mouse_click"   # button, clicks, optional (x, y)
    MOUSE_DOWN = "mouse_down"     # button down (bounded hold)
    MOUSE_UP = "mouse_up"         # button up
    MOUSE_SCROLL = "mouse_scroll" # scroll by N ticks (positive = up)
    DELAY = "delay"               # wait duration_ms, then continue


@dataclass(frozen=True)
class Action:
    """One discrete, validated, deterministic input operation.

    Coordinates, when present, are in **screen space** (the action layer
    drives the real desktop, unlike ``Observation`` which is frame space).
    ``None`` fields are unused by the kind; the kind-specific field set is
    enforced at construction and at ``from_dict`` so an invalid action cannot
    be held, queued, or executed.
    """

    kind: ActionKind
    # keyboard
    key: str | None = None
    modifiers: tuple[str, ...] = ()
    # mouse
    x: int | None = None
    y: int | None = None
    relative: bool = False
    button: str = "left"
    clicks: int = 1
    scroll: int | None = None
    # timing (all bounded — see SafetyGuard, later step)
    duration_ms: int = 0
    delay_ms: int = 0
    # provenance (replayability: "why did the agent do this?")
    source: str = "user"
    confidence: float | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_action(self)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dict of JSON plain values (no enums/tuples/dataclasses)."""
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "key": self.key,
            "modifiers": list(self.modifiers),
            "x": self.x,
            "y": self.y,
            "relative": self.relative,
            "button": self.button,
            "clicks": self.clicks,
            "scroll": self.scroll,
            "duration_ms": self.duration_ms,
            "delay_ms": self.delay_ms,
            "source": self.source,
            "confidence": self.confidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> Action:
        """Reconstruct an :class:`Action` from a plain dict (``to_dict`` inverse).

        Raises:
            ActionValidationError: if ``data`` is not a dict, lacks a valid
                ``kind``/required field, carries an unknown key, or sets a field
                the kind forbids.
        """
        if not isinstance(data, dict):
            raise ActionValidationError(
                f"Action.from_dict expects a dict, got {type(data).__name__}"
            )
        unknown = set(data) - set(_ALL_FIELDS)
        if unknown:
            raise ActionValidationError(
                f"Action has unknown field(s): {sorted(unknown)}"
            )
        raw_kind = data.get("kind")
        if raw_kind is None:
            raise ActionValidationError("Action is missing required field 'kind'")
        kind = raw_kind.value if isinstance(raw_kind, ActionKind) else raw_kind
        if not isinstance(kind, str) or kind not in _KIND_FIELDS:
            raise ActionValidationError(f"unknown Action kind: {kind!r}")
        # Rebuild with defaults, overriding from the payload. ``tuple`` fields
        # accept lists from JSON.
        kwargs: dict[str, object] = {}
        for name in _ALL_FIELDS:
            if name not in data:
                continue
            value = data[name]
            if name in ("modifiers",):
                if not isinstance(value, (list, tuple)):
                    raise ActionValidationError(
                        f"field 'modifiers' must be a list of strings, got {value!r}"
                    )
                value = tuple(str(m) for m in value)
            if name == "kind":
                value = ActionKind(kind)
            kwargs[name] = value
        try:
            return cls(**kwargs)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ActionValidationError(f"invalid Action payload: {exc}") from exc


@dataclass(frozen=True)
class ActionResult:
    """Outcome of executing one :class:`Action`. Always JSON-serializable.

    ``stop_event`` is ``True`` when an emergency stop intervened (the action
    did not complete). ``error`` is a human-readable reason for ``ok=False``
    and never contains secrets.
    """

    action: Action
    ok: bool
    error: str | None = None
    duration_ms: float = 0.0
    stop_event: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise ActionValidationError(
                f"ActionResult.action must be an Action, got {type(self.action).__name__}"
            )
        if not isinstance(self.ok, bool):
            raise ActionValidationError("ActionResult.ok must be a bool")
        if not isinstance(self.stop_event, bool):
            raise ActionValidationError("ActionResult.stop_event must be a bool")
        if self.error is not None and not isinstance(self.error, str):
            raise ActionValidationError("ActionResult.error must be a str or None")
        if not (self.duration_ms >= 0.0):
            raise ActionValidationError(
                f"ActionResult.duration_ms must be >= 0, got {self.duration_ms}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.to_dict(),
            "ok": self.ok,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "stop_event": self.stop_event,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ActionResult:
        if not isinstance(data, dict):
            raise ActionValidationError(
                f"ActionResult.from_dict expects a dict, got {type(data).__name__}"
            )
        if "action" not in data:
            raise ActionValidationError(
                "ActionResult is missing required field 'action'"
            )
        unknown = set(data) - {"action", "ok", "error", "duration_ms", "stop_event"}
        if unknown:
            raise ActionValidationError(
                f"ActionResult has unknown field(s): {sorted(unknown)}"
            )
        action = Action.from_dict(data["action"])
        return cls(
            action=action,
            ok=bool(data.get("ok", False)),
            error=data.get("error"),
            duration_ms=float(data.get("duration_ms", 0.0)),
            stop_event=bool(data.get("stop_event", False)),
        )


@dataclass(frozen=True)
class ActionLog:
    """A complete, replayable record of a batch of :class:`ActionResult`.

    This is the Phase-3 analog of Phase 1's frame recording and Phase 2's
    ``Observation``: the artifact the ``act`` CLI prints and replay feeds back
    through queue -> executor without a real backend.

    ``results`` is a tuple (immutable). ``stopped`` is ``True`` when an
    emergency stop occurred during the batch; ``stop_reason`` carries the
    human-readable reason (e.g. the triggering hotkey or a safety bound).
    """

    results: tuple[ActionResult, ...] = ()
    stopped: bool = False
    stop_reason: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
        if not isinstance(self.stopped, bool):
            raise ActionValidationError("ActionLog.stopped must be a bool")
        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise ActionValidationError("ActionLog.stop_reason must be a str or None")
        for result in self.results:
            if not isinstance(result, ActionResult):
                raise ActionValidationError(
                    "ActionLog.results must contain only ActionResult"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "results": [r.to_dict() for r in self.results],
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ActionLog:
        if not isinstance(data, dict):
            raise ActionValidationError(
                f"ActionLog.from_dict expects a dict, got {type(data).__name__}"
            )
        unknown = set(data) - {
            "schema_version",
            "results",
            "stopped",
            "stop_reason",
        }
        if unknown:
            raise ActionValidationError(
                f"ActionLog has unknown field(s): {sorted(unknown)}"
            )
        raw_results = data.get("results", [])
        if not isinstance(raw_results, (list, tuple)):
            raise ActionValidationError("ActionLog.results must be a list")
        results = tuple(ActionResult.from_dict(r) for r in raw_results)
        return cls(
            results=results,
            stopped=bool(data.get("stopped", False)),
            stop_reason=data.get("stop_reason"),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_ALL_FIELDS = (
    "kind", "key", "modifiers", "x", "y", "relative", "button", "clicks",
    "scroll", "duration_ms", "delay_ms", "source", "confidence", "schema_version",
)


def _validate_action(action: Action) -> None:
    """Enforce the per-kind field contract + field-level value rules.

    Raises :class:`ActionValidationError` on the first violation.
    """
    if not isinstance(action.kind, ActionKind):
        raise ActionValidationError(
            f"Action.kind must be an ActionKind, got {action.kind!r}"
        )
    kind = action.kind.value

    # 1. Field-name contract: the kind must carry *only* its allowed fields,
    #    with the others at their defaults.
    allowed = _KIND_FIELDS[kind]
    for name, default in _DEFAULTS.items():
        if name in allowed:
            continue
        value = getattr(action, name)
        if value != default:
            raise ActionValidationError(
                f"action kind {kind!r} must not set field {name!r} "
                f"(got {value!r}); use a kind that uses it"
            )

    # 2. Required fields must be present (not ``None``). Zero is a *value*,
    #    not an absence: (0, 0) is the screen origin, a legal mouse target.
    #    (``key=""`` and ``scroll=0`` remain illegal — the value-level rules
    #    below reject them with their specific messages.)
    for name in _KIND_REQUIRED[kind]:
        if getattr(action, name) is None:
            raise ActionValidationError(
                f"action kind {kind!r} requires field {name!r} to be set"
            )

    # 3. Field-level value rules (shared across kinds).
    _validate_field_values(action, kind)


def _validate_field_values(action: Action, kind: str) -> None:
    # key
    if action.key is not None and not _valid_key_name(action.key):
        raise ActionValidationError(f"invalid key name: {action.key!r}")

    # modifiers
    if action.modifiers:
        if len(set(action.modifiers)) != len(action.modifiers):
            raise ActionValidationError(
                f"duplicate modifiers: {action.modifiers!r}"
            )
        for mod in action.modifiers:
            # Strict lowercase: the backend (pynput) speaks lowercase modifier
            # names, so a mixed-case modifier is a contract violation.
            if mod != mod.lower() or mod not in _VALID_MODIFIERS:
                raise ActionValidationError(f"invalid modifier: {mod!r}")
            if action.key is not None and mod == action.key.lower():
                raise ActionValidationError(
                    f"modifier {mod!r} cannot also be the key"
                )
        # A key action with a modifier that is itself the key is nonsensical;
        # a non-key kind must not carry modifiers at all.
        if kind not in ("key_press", "key_hold"):
            raise ActionValidationError(
                f"action kind {kind!r} must not carry modifiers"
            )

    # button
    if action.button not in _VALID_BUTTONS:
        raise ActionValidationError(f"invalid mouse button: {action.button!r}")

    # clicks
    if not _is_int(action.clicks) or action.clicks < 1:
        raise ActionValidationError(
            f"clicks must be an int >= 1, got {action.clicks!r}"
        )

    # coords
    for name in ("x", "y"):
        value = getattr(action, name)
        if value is None:
            continue
        if not _is_int(value):
            raise ActionValidationError(
                f"field {name!r} must be an int or None, got {value!r}"
            )
        if not action.relative and value < 0:
            raise ActionValidationError(
                f"absolute coordinate {name!r} must be >= 0, got {value}"
            )

    # scroll
    if action.scroll is not None:
        if not _is_int(action.scroll) or action.scroll == 0:
            raise ActionValidationError(
                f"scroll must be a non-zero int or None, got {action.scroll!r}"
            )

    # relative is only meaningful for mouse_move (and mouse_scroll position);
    # it must be a bool.
    if not isinstance(action.relative, bool):
        raise ActionValidationError(
            f"relative must be a bool, got {action.relative!r}"
        )

    # duration / delay
    for name in ("duration_ms", "delay_ms"):
        value = getattr(action, name)
        if not _is_int(value) or value < 0:
            raise ActionValidationError(
                f"field {name!r} must be a non-negative int, got {value!r}"
            )

    # duration is only used by key_hold / delay.
    if kind not in ("key_hold", "delay") and action.duration_ms != 0:
        raise ActionValidationError(
            f"action kind {kind!r} must not set duration_ms (got "
            f"{action.duration_ms}); use a timed kind"
        )
    if kind == "delay" and action.duration_ms <= 0:
        raise ActionValidationError(
            "a DELAY action must have a positive duration_ms"
        )
    if kind == "key_hold" and action.duration_ms <= 0:
        raise ActionValidationError(
            "a KEY_HOLD action must have a positive duration_ms"
        )

    # source
    if not isinstance(action.source, str) or not action.source:
        raise ActionValidationError(
            f"source must be a non-empty string, got {action.source!r}"
        )

    # confidence
    if action.confidence is not None:
        if (
            isinstance(action.confidence, bool)
            or not isinstance(action.confidence, (int, float))
            or not (0.0 <= action.confidence <= 1.0)
        ):
            raise ActionValidationError(
                f"confidence must be in [0.0, 1.0] or None, got {action.confidence!r}"
            )


def validate_action(action: object) -> Action:
    """Public helper: return ``action`` if valid, else raise.

    Lets the executor / CLI / replay validate an action without re-instantiating
    it. An already-valid :class:`Action` passes through unchanged.
    """
    if not isinstance(action, Action):
        raise ActionValidationError(
            f"validate_action expects an Action, got {type(action).__name__}"
        )
    return action
