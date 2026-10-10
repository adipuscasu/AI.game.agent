"""Action-layer error hierarchy + input protocols + backend factory (Phase 3, step 2).

Dependency-free. Per AGENTS.md this maps to the ``ACTION_ERROR`` /
``INPUT_ERROR`` error classes. Every action-layer error inherits from
:class:`ActionError` so callers can catch a single type, while subclasses let
a specific layer (validation, queue, backend, safety, mode) classify the
failure precisely.

The :class:`Keyboard` / :class:`Mouse` / :class:`InputBackend` protocols are
structural (``typing.Protocol``) so conformance requires no base class — the
same decoupling ``perception/base.py`` uses for its detector protocols. Only
the :class:`~ai_game_agent.actions.executor.Executor` (step 4) is allowed to
talk to an :class:`InputBackend`; queue, safety, and modes program against
these protocols and never call a keyboard/mouse library directly.

The :func:`create_backend` factory mirrors ``capture.create_backend``: a
config/CLI decision selects the backend, and the platform-specific
``WindowsBackend`` (pynput) is imported **only when selected** — a bare venv
without the ``input`` extra never imports pynput and gets a clean
:class:`InputBackendError` naming the extra instead of an import traceback.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_game_agent.config import InputConfig

__all__ = [
    "ActionError",
    "ActionQueueError",
    "ActionValidationError",
    "InputBackend",
    "InputBackendError",
    "Keyboard",
    "ModeViolation",
    "Mouse",
    "SafetyViolation",
    "create_backend",
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


# ---------------------------------------------------------------------------
# Input backend protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Keyboard(Protocol):
    """Keyboard input surface.

    ``key`` is a canonical, lowercase, single-token name (``"a"``, ``"f12"``,
    ``"space"``) — the same names :class:`Action` validates against. ``modifiers``
    is a tuple of the same names (e.g. ``("ctrl", "shift")``). Implementations
    decide how to map these to real keys; tests inject fakes or the mock.
    """

    def key_down(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        """Press ``key`` (with ``modifiers``) and hold it down."""
        ...

    def key_up(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        """Release ``key`` (with ``modifiers``)."""
        ...

    def press(self, key: str, modifiers: tuple[str, ...] = ()) -> None:
        """A complete press-and-release of ``key`` (down then up)."""
        ...


@runtime_checkable
class Mouse(Protocol):
    """Mouse input surface.

    Method names are flat (``mouse_move`` / ``mouse_click`` / ``mouse_down`` /
    ``mouse_up``) rather than nested under a sub-object, matching
    :class:`~ai_game_agent.actions.mock.MockBackend` and the executor's single
    flat call surface. ``x``/``y`` are *screen-space* absolute coordinates
    (the action layer operates on the real desktop); ``relative`` moves by the
    delta instead. ``button`` is one of ``left``/``right``/``middle``;
    ``clicks`` is a positive integer.
    """

    def mouse_move(self, x: int, y: int, relative: bool = False) -> None:
        """Move the pointer to ``(x, y)`` (absolute), or by the delta if ``relative``."""
        ...

    def mouse_click(self, button: str = "left", clicks: int = 1) -> None:
        """Click ``button`` ``clicks`` time(s)."""
        ...

    def mouse_down(self, button: str = "left") -> None:
        """Press and hold ``button`` down."""
        ...

    def mouse_up(self, button: str = "left") -> None:
        """Release ``button``."""
        ...


@runtime_checkable
class InputBackend(Keyboard, Mouse, Protocol):
    """A complete input backend: keyboard + mouse + lifecycle.

    ``open()`` prepares the backend (allocate resources) and is idempotent;
    ``close()`` releases resources, is idempotent, and **must release any
    held keys/buttons** so a half-open backend cannot leave input stuck
    (the exact failure AGENTS.md's safety section calls out).

    ``release_all()`` releases every held key and button immediately and is
    the hook the emergency stop and the executor's error path call. It is
    idempotent so it can be invoked repeatedly (emergency stop is terminal and
    may fire more than once).
    """

    name: str

    def open(self) -> None:
        """Prepare the backend (allocate resources). Idempotent."""
        ...

    def close(self) -> None:
        """Release backend resources and any held input. Idempotent."""
        ...

    def release_all(self) -> None:
        """Release all held keys and buttons. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_backend(input_cfg: InputConfig) -> InputBackend:
    """Factory: return the input backend named in ``input_cfg``.

    Mirrors :func:`ai_game_agent.capture.create_backend`: the backend is a
    config/CLI decision, not a code branch, and the platform backend (pynput)
    is imported lazily only when selected. A bare venv without the ``input``
    extra therefore stays importable and gets a clean error, never a traceback.

    Args:
        input_cfg: :class:`InputConfig` (``input_cfg.backend`` selects).

    Returns:
        A :class:`MockBackend` for ``backend == "mock"`` or a
        :class:`WindowsBackend` for ``backend == "windows"``.

    Raises:
        InputBackendError: if the ``windows`` backend is selected but pynput
            (the ``input`` extra) is not installed — fail fast with the extra
            name, mirroring Phase 1's ``mss`` availability check.
        ActionError: if the backend name is unknown.
    """
    name = (input_cfg.backend or "").strip().lower()
    if name == "mock":
        from ai_game_agent.actions.mock import MockBackend

        return MockBackend()
    if name == "windows":
        if not _pynput_available():
            raise InputBackendError(
                "the 'windows' input backend needs the 'pynput' library; "
                "install the 'input' extra with 'uv sync --extra input' "
                "(or 'pip install pynput'), or set input.backend to 'mock'"
            )
        from ai_game_agent.actions.windows import WindowsBackend

        return WindowsBackend()
    raise ActionError(f"unknown input backend: {input_cfg.backend!r}")


def _pynput_available() -> bool:
    """Return True if pynput can be imported (the ``input`` extra is installed)."""
    try:
        import pynput  # noqa: F401

        return True
    except ImportError:
        return False
