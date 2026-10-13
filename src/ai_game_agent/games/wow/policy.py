"""WoW deterministic action policy (Phase 4, step 3).

Implements the ``StatePolicy`` protocol with the step-3 "Action Plan" table
(``docs/phase-4-implementation-plan.md`` §4.4). This is the *only* place in
the codebase that knows WoW-specific facts: the zone names (imported from
:mod:`ai_game_agent.games.wow.zones`) and the key bindings (constructor
args, defaulting to the config's ``q`` / ``f`` / ``r``).

Design invariants (see AGENTS.md):

* The policy **proposes** validated :class:`Action` objects; it never calls a
  keyboard/mouse API itself. Execution is the executor's job.
* Under an input-disallowed mode (``OBSERVE_ONLY``) it returns ``()`` — the
  first line of the OBSERVE_ONLY hard gate.
* Zone-name literals appear only in :mod:`...zones`; this module imports the
  constants, so renaming a zone is a one-file change.
* Signal names are imported from the (game-agnostic) state machine, not
  re-spelled here — keeping the policy free of *any* zone/signal string
  literal that could drift from the contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ai_game_agent.actions.action import Action, ActionKind
from ai_game_agent.actions.modes import OperationMode
from ai_game_agent.games.wow.zones import ZONE_LOOT_GLOW, ZONE_TARGET, ZONE_TARGET_HOSTILE
from ai_game_agent.perception.observation import BBox, Observation
from ai_game_agent.state.machine import (
    SIGNAL_HEALTH_LOW,
    SIGNAL_HEALTH_RECOVERED,
    SIGNAL_LOOT_PRESENT,
    SIGNAL_TARGET_HOSTILE,
    SIGNAL_TARGET_PRESENT,
)
from ai_game_agent.state.states import AgentState

__all__ = ["DeterministicPolicy"]


# --- small module-level helpers (stateless, easy to test) -------------------


def _find_bbox(observation: Observation, name: str) -> BBox | None:
    """Return the bounding box for a named zone/template hit, else ``None``.

    A semantic UI state can surface as either a ``UiZoneHit`` (``ui_zones``)
    or a ``TemplateHit`` (``templates``); exact-name match, no substring.
    """
    for hit in observation.ui_zones:
        if hit.zone == name:
            return hit.bbox
    for hit in observation.templates:
        if hit.template == name:
            return hit.bbox
    return None


def _detector_failed(observation: Observation, name: str) -> bool:
    """True if the detector that reports ``name`` errored this tick.

    The pipeline records a per-detector failure as ``"{layer}:{name}: {reason}"``
    where ``layer`` is ``ui`` or ``templates`` for a zone/template. A *named*
    failure for a *different* zone (or a nameless ``objects:`` failure) does
    **not** count as a failure for ``name`` — the presence signal must stay
    narrow.
    """
    prefixes = (f"ui:{name}:", f"templates:{name}:")
    return any(err.startswith(p) for err in observation.detector_errors for p in prefixes)


def _presence(observation: Observation, name: str) -> bool | None:
    """Presence signal: ``True`` if present, ``None`` if unknown, else ``False``.

    * present this tick  -> ``True``
    * not present, but that detector failed this tick -> ``None`` (unknown)
    * not present, detector ran clean -> ``False`` (genuinely absent)
    """
    if _find_bbox(observation, name) is not None:
        return True
    if _detector_failed(observation, name):
        return None
    return False


def _valid_fraction(value: object) -> bool:
    """A usable health reading: a real (non-bool) number, finite, in [0, 1]."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    as_float = float(value)
    return math.isfinite(as_float) and 0.0 <= as_float <= 1.0


def _valid_delay_ms(value: object) -> bool:
    """A usable delay: a real (non-bool) positive integer of milliseconds."""
    return (isinstance(value, int) and not isinstance(value, bool) and value > 0)


def _valid_key(value: object) -> bool:
    return isinstance(value, str) and value != ""


# --- the policy -------------------------------------------------------------


class DeterministicPolicy:
    """WoW action policy: the §4.4 action table + the stable signal seam.

    All action parameters (keys, delays, thresholds) are constructor args so
    a later config-wiring step can pass ``cfg.agent.*`` straight through.
    Defaults match the project's standard bindings (``q`` / ``f`` / ``r``).
    """

    def __init__(
        self,
        *,
        name: str = "wow",
        combat_keys: tuple[str, ...] = ("q", "w"),
        combat_delay_ms: int = 120,
        loot_pickup_key: str = "f",
        loot_delay_ms: int = 200,
        recovery_key: str = "r",
        recovery_delay_ms: int = 300,
        health_low: float = 0.30,
        health_recovered: float = 0.50,
        health_fraction: Callable[[Observation], float | None] | None = None,
    ) -> None:
        if not _valid_key(name):
            raise ValueError(f"policy name must be a non-empty string, got {name!r}")
        if not (isinstance(combat_keys, (list, tuple)) and combat_keys):
            raise ValueError("combat_keys must be a non-empty list/tuple")
        for key in combat_keys:
            if not _valid_key(key):
                raise ValueError(f"combat key must be a non-empty string, got {key!r}")
        if not _valid_key(loot_pickup_key):
            raise ValueError(
                f"loot_pickup_key must be a non-empty string, got {loot_pickup_key!r}"
            )
        if not _valid_key(recovery_key):
            raise ValueError(
                f"recovery_key must be a non-empty string, got {recovery_key!r}"
            )
        if not _valid_delay_ms(combat_delay_ms):
            raise ValueError(f"combat_delay_ms must be a positive int, got {combat_delay_ms!r}")
        if not _valid_delay_ms(loot_delay_ms):
            raise ValueError(f"loot_delay_ms must be a positive int, got {loot_delay_ms!r}")
        if not _valid_delay_ms(recovery_delay_ms):
            raise ValueError(
                f"recovery_delay_ms must be a positive int, got {recovery_delay_ms!r}"
            )
        if not (0.0 < health_low <= 1.0):
            raise ValueError(f"health_low must be in (0, 1], got {health_low!r}")
        if not (0.0 < health_recovered <= 1.0):
            raise ValueError(
                f"health_recovered must be in (0, 1], got {health_recovered!r}"
            )
        if not (health_low < health_recovered):
            raise ValueError(
                f"health_low must be < health_recovered, got {health_low!r} !< {health_recovered!r}"
            )

        self._name = name
        self._combat_keys = tuple(combat_keys)
        self._combat_delay_ms = combat_delay_ms
        self._loot_pickup_key = loot_pickup_key
        self._loot_delay_ms = loot_delay_ms
        self._recovery_key = recovery_key
        self._recovery_delay_ms = recovery_delay_ms
        self._health_low = health_low
        self._health_recovered = health_recovered
        self._health_fraction = health_fraction

    # -- StatePolicy protocol ------------------------------------------------

    def name(self) -> str:
        return self._name

    def actions(
        self,
        state: AgentState,
        observation: Observation,
        mode: OperationMode,
    ) -> tuple[Action, ...]:
        # First line of the OBSERVE_ONLY hard gate: seeing + deciding is fine,
        # acting is not. (The executor and SafetyGuard back this up.)
        if not mode.input_allowed:
            return ()
        if state is AgentState.COMBAT:
            return self._combat_actions()
        if state is AgentState.LOOTING:
            return self._looting_actions(observation)
        if state is AgentState.RECOVERY:
            return self._recovery_actions()
        # IDLE / AI_ANALYSIS / PAUSED: nothing to do deterministically.
        return ()

    def transition_signals(self, observation: Observation) -> dict[str, object]:
        signals: dict[str, object] = {
            SIGNAL_LOOT_PRESENT: _presence(observation, ZONE_LOOT_GLOW),
            SIGNAL_TARGET_PRESENT: _presence(observation, ZONE_TARGET),
            SIGNAL_TARGET_HOSTILE: _presence(observation, ZONE_TARGET_HOSTILE),
        }
        signals.update(self._health_signals(observation))
        return signals

    # -- per-state action builders (private) --------------------------------

    def _combat_actions(self) -> tuple[Action, ...]:
        actions: list[Action] = []
        for index, key in enumerate(self._combat_keys):
            if index > 0:
                actions.append(self._delay(self._combat_delay_ms))
            actions.append(self._key(key))
        return tuple(actions)

    def _looting_actions(self, observation: Observation) -> tuple[Action, ...]:
        actions: list[Action] = []
        bbox = _find_bbox(observation, ZONE_LOOT_GLOW)
        if bbox is not None:
            # Click the marker's actual bounding-box center — never a hard-coded
            # point. No located marker => no click at all (correction #1).
            actions.append(self._click(bbox.x + bbox.width // 2, bbox.y + bbox.height // 2))
        actions.append(self._delay(self._loot_delay_ms))
        actions.append(self._key(self._loot_pickup_key))
        return tuple(actions)

    def _recovery_actions(self) -> tuple[Action, ...]:
        return (self._key(self._recovery_key), self._delay(self._recovery_delay_ms))

    # -- health signals (calibrated source is optional) ---------------------

    def _health_signals(self, observation: Observation) -> dict[str, object]:
        # Unknown by default: ``Observation`` carries no calibrated health
        # fraction (``UiZoneHit.value`` is [0,1] *evidence*, not a health %).
        if self._health_fraction is None:
            return {SIGNAL_HEALTH_LOW: None, SIGNAL_HEALTH_RECOVERED: None}
        reading = self._health_fraction(observation)
        if not _valid_fraction(reading):
            # A bad reading is rejected (signal -> None), never raised: an
            # exception here would kill the whole tick (correction #3).
            return {SIGNAL_HEALTH_LOW: None, SIGNAL_HEALTH_RECOVERED: None}
        return {
            SIGNAL_HEALTH_LOW: reading < self._health_low,
            SIGNAL_HEALTH_RECOVERED: reading >= self._health_recovered,
        }

    # -- action builders (private) ------------------------------------------

    def _key(self, key: str) -> Action:
        return Action(kind=ActionKind.KEY_PRESS, key=key, source=self._src())

    def _delay(self, ms: int) -> Action:
        return Action(kind=ActionKind.DELAY, duration_ms=ms, source=self._src())

    def _click(self, x: int, y: int) -> Action:
        return Action(kind=ActionKind.MOUSE_CLICK, x=x, y=y, source=self._src())

    def _src(self) -> str:
        return f"policy:{self._name}"
