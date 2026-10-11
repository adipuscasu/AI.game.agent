"""Hotkey listener (Phase 3, step 7): the emergency-stop mechanism.

The AI-independent mechanism behind the emergency-stop hotkey (AGENTS.md:
"the agent must have an emergency-stop mechanism independent of the AI").
This module is the **contract + fake** — the part of step 7 the plan calls
"``HotkeyListener`` protocol + fake-triggerable listener (mechanism) —
trigger → ``EmergencyStop`` proven headless."

Why a protocol and a fake instead of the pynput listener now:

* The physical key binding (an OS-global keyboard hook) cannot be tested
  headless — it is the Windows-side pynput listener (step 8), which routes
  every key press into :meth:`HotkeyListener.trigger`.
* The **mechanism** — armed/disarmed state, stop-key matching, and
  "request the stop exactly once per arm cycle" — is pure logic, and it is
  the part the safety story actually depends on: a double-delivered key
  press must not double-release, and a stale trigger must not stop an
  already-stopped run.

Contract (:class:`HotkeyListener`, structural / ``runtime_checkable``)
--------------------------------------------------------------------------------
* :meth:`arm` — start a stop-request cycle (idempotent; re-arming starts a
  fresh cycle, so a new run can bind a new cycle).
* :meth:`disarm` — end the cycle; triggers until the next ``arm`` are
  ignored (a stale key event must not fire a stop).
* :meth:`trigger` — one key-press event from the input layer. If the
  listener is armed and the key matches a configured stop key, the
  ``on_trigger`` callback (wired to
  ``Executor.trigger_emergency_stop``) is invoked **exactly once per arm
  cycle**, regardless of how many times the key is redelivered.

Dependency rule: this module is **stdlib-only**. ``pynput`` is imported by
nothing here — the same bare-venv import guarantee the rest of steps 1–7
hold (the ``input`` extra is only required by the step-8 Windows adapter).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class HotkeyListener(Protocol):
    """The emergency-stop hotkey mechanism, independent of any input library.

    An implementation owns an ``on_trigger`` callback (typically
    ``Executor.trigger_emergency_stop``) and decides, from :meth:`trigger`
    events, when that callback fires. The input layer (e.g. the pynput
    global keyboard hook) is the *event source*; this protocol is the
    *decision* — so it is testable with a fake event source, in any
    environment, with no OS input involved.
    """

    def arm(self) -> None:
        """Start a stop-request cycle (idempotent)."""
        ...

    def disarm(self) -> None:
        """End the cycle; :meth:`trigger` is ignored until the next ``arm``."""
        ...

    def trigger(self, key: str) -> None:
        """One key-press event; fire ``on_trigger`` if armed and it is a stop key.

        At most once per arm cycle (idempotent against key redelivery).
        """
        ...


class FakeHotkeyListener:
    """Fake-triggerable :class:`HotkeyListener` for headless tests.

    Conforms structurally to :class:`HotkeyListener`. The test drives the
    event source by calling :meth:`trigger` directly — the same entry point
    the real (pynput) listener uses, so the mechanism under test is
    identical.

    Semantics (the acceptance criteria for step 7):

    * **Idempotency** — repeated stop-key triggers in one arm cycle invoke
      ``on_trigger`` exactly once (AC1).
    * **Arm/disarm** — a trigger is honored only while armed (no stale
      stops); :meth:`arm` reopens a cycle (AC2's "no subsequent dispatch"
      is the executor's job, proven in ``tests/test_actions_hotkey.py``).
    * **Matching** — stop keys compare case-insensitively (config ships
      ``F12``; key systems speak lowercase ``f12``).
    """

    name = "fake"

    def __init__(
        self,
        on_trigger: Callable[[], None],
        triggered_keys: tuple[str, ...] = ("f12",),
    ) -> None:
        self._on_trigger = on_trigger
        # Normalize once so matching is case-insensitive by construction.
        self.triggered_keys = tuple(k.lower() for k in triggered_keys)
        self._armed = False
        self._stop_requested = False
        # How many times ``on_trigger`` was actually invoked (total, across
        # arm cycles). Distinct from ``armed``: a cycle may be armed without
        # a stop key ever being pressed.
        self.trigger_count = 0

    @property
    def armed(self) -> bool:
        """Whether a stop-request cycle is currently open."""
        return self._armed

    def arm(self) -> None:
        """Open (or reopen) a stop-request cycle.

        Re-opening is deliberate: each executor run arms its own cycle, and
        within that cycle the stop is requested at most once. Calling
        ``arm()`` twice without a trigger in between is a harmless no-op.
        """
        self._armed = True
        self._stop_requested = False

    def disarm(self) -> None:
        """Close the cycle. Triggers until the next :meth:`arm` are ignored."""
        self._armed = False

    def trigger(self, key: object) -> None:
        """One key-press event from the input layer.

        ``key`` may be a plain string (this fake's contract) or a pynput
        ``Key`` instance (the real adapter's); both stringify to the key
        name, so :func:`str` normalization covers both. A non-string that
        cannot be normalized to a stop key simply does not match — it never
        raises (a stray key event must not take the listener down).
        """
        if not self._armed:
            return
        if isinstance(key, str):
            name = key.lower()
        else:
            name = str(key).lower()
        if name not in self.triggered_keys:
            return
        if self._stop_requested:
            # AC1: the stop was already requested this cycle — a redelivered
            # key (OS auto-repeat, duplicate delivery) must not request it
            # again, and must not re-release anything.
            return
        self._stop_requested = True
        self.trigger_count += 1
        self._on_trigger()


__all__ = ["FakeHotkeyListener", "HotkeyListener"]
