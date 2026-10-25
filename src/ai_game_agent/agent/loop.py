"""AgentLoop (Phase 4, step 5) — the orchestration.

The single decision loop tying the committed pieces together:

    grab -> observe -> decide -> act -> pace

Game-agnostic by construction: it depends only on the ``state`` / ``actions``
/ ``capture`` / ``perception`` layers and the injected ``StatePolicy``; no
zone names, key bindings, or game literals appear here (those live in the
game adapter, ``ai_game_agent.games.wow``).

Safety story (AGENTS.md):

* **OBSERVE_ONLY is a hard gate** — under that mode the loop never calls
  ``Executor.execute()``; the policy already returns ``()`` when
  ``mode.input_allowed`` is ``False`` and the loop additionally gates the
  dispatch on the same flag, so the guard is never even consulted and no
  input can be emitted (defense in depth).
* **Pipeline failures are classified, not fatal** — a ``CaptureError`` from
  ``grab()`` or any exception from ``observe()`` is routed to bounded
  RECOVERY via the state machine, recorded as a ``RECOVERY`` event, and once
  ``recovery_attempts`` pass without a healthy exit the machine moves to
  ``PAUSED`` and the loop exits. The loop itself never raises out of
  ``tick()``.
* **Release-all on every exit path** — ``run()`` calls
  ``executor.release_all()`` and ``capture.stop()`` in ``finally`` so one
  cleanup failure cannot skip the other.

Sequencing: the loop owns the authoritative, session-wide, 1-based ``seq``
across all event kinds (``TICK``, ``ACTION_*``, ``RECOVERY``, ``STOP``,
``ERROR``, and the machine's ``STATE_ENTER``). The machine's per-instance
0-based counter is provisional; the loop renumbers each ``STATE_ENTER`` it
receives via its ``on_transition`` seam.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace as _replace

from ai_game_agent.actions.action import Action, ActionResult
from ai_game_agent.actions.executor import Executor
from ai_game_agent.actions.hotkey import HotkeyListener
from ai_game_agent.actions.modes import OperationMode
from ai_game_agent.agent.report import AgentLoopReport
from ai_game_agent.capture import CaptureError
from ai_game_agent.perception import Observation
from ai_game_agent.state import (
    AgentState,
    EventType,
    GameStateMachine,
    LoopEvent,
    StatePolicy,
)

__all__ = ["AgentLoop", "TickResult"]


@dataclass(frozen=True)
class TickResult:
    """Outcome of one :meth:`AgentLoop.tick` (for ``--once`` and replay tests).

    ``observation`` may be ``None`` when the tick failed before perception
    (e.g. a ``CaptureError``); ``state`` is the machine state after the tick;
    ``actions`` / ``results`` are the dispatched batch (empty on the failure
    path or under ``OBSERVE_ONLY``); ``error`` is a human-readable reason
    when the tick was interrupted by a pipeline failure.
    """

    observation: Observation | None = None
    state: AgentState = AgentState.IDLE
    actions: tuple[Action, ...] = ()
    results: tuple[ActionResult, ...] = ()
    error: str | None = None


class AgentLoop:
    """Blocking observe-decide-act loop with an authoritative session audit trail.

    All collaborators are injected (no globals, no hidden I/O); the clock
    (``now_fn``) is injectable for deterministic watchdog tests, and the
    pacing is owned by the capture facade (``wait_next_frame``), never by a
    second independent sleep in this loop (plan §4.6: "do not add a second
    independent sleep on top of the capture facade, which would double-pace
    the loop").
    """

    def __init__(
        self,
        *,
        capture: object,
        perception: object,
        executor: Executor,
        machine: GameStateMachine,
        policy: StatePolicy,
        mode: OperationMode,
        max_ticks: int | None,
        now_fn: Callable[[], float] = time.time,
        stop_event: threading.Event | None = None,
        hotkey: HotkeyListener | None = None,
        on_event: Callable[[LoopEvent], None] | None = None,
        recorder: Callable[[LoopEvent], None] | None = None,
        ai: object | None = None,
    ) -> None:
        if max_ticks is not None and max_ticks < 1:
            raise ValueError("max_ticks must be >= 1 or None")
        self._capture = capture
        self._perception = perception
        self._executor = executor
        self._machine = machine
        self._policy = policy
        self._mode = mode
        self._max_ticks = max_ticks
        self._now = now_fn
        self._stop_event = stop_event
        self._hotkey = hotkey
        self._on_event = on_event
        self._recorder = recorder
        # Phase 5 close-out: optional AI provider (duck-typed AIProvider).
        # The loop never imports ai_game_agent.ai — it only calls
        # provider.decide(prompt) -> AIDecision. None (default) keeps
        # AI_ANALYSIS behaving exactly as today (no dispatch, watchdog fires).
        self._ai = ai
        # Report accumulators (so a lone tick() also works, not just run()).
        self._seq = 0
        self._ticks = 0
        self._transitions: list[LoopEvent] = []
        self._actions_attempted = 0
        self._actions_ok = 0
        self._actions_failed = 0
        self._errors: list[LoopEvent] = []
        self._stopped_reason: str | None = None
        self._start_wall: float | None = None
        # Own the machine's STATE_EMIT seam: renumber its provisional
        # per-instance seq into the session-wide 1-based stream and fan it
        # out to the subscriber (on_event) and the report.
        self._machine.on_transition = self._on_state_event

    # ---- event plumbing (session-wide sequencing lives here) ---------------

    def _next_seq(self) -> int:
        """Allocate the next session-wide 1-based sequence number."""
        self._seq += 1
        return self._seq

    def _emit(self, event: LoopEvent) -> None:
        """Fan one event out to the subscriber and the recorder (if any)."""
        if self._on_event is not None:
            self._on_event(event)
        if self._recorder is not None:
            self._recorder(event)

    def _event(
        self,
        *,
        event: EventType,
        ts: float,
        state_before: AgentState | None = None,
        state_after: AgentState | None = None,
        reason: str = "",
        detail: dict[str, object] | None = None,
    ) -> LoopEvent:
        """Build one :class:`LoopEvent` with a fresh session ``seq``, not yet emitted."""
        return LoopEvent(
            ts=ts,
            seq=self._next_seq(),
            event=event,
            state_before=state_before,
            state_after=state_after,
            reason=reason,
            detail=dict(detail) if detail else {},
        )

    def _on_state_event(self, raw: LoopEvent) -> None:
        """Renumber a machine ``STATE_ENTER`` into the session seq; record + emit."""
        ev = _replace(raw, seq=self._next_seq())
        self._transitions.append(ev)
        self._emit(ev)

    # ---- one iteration ------------------------------------------------------

    def tick(self) -> TickResult:
        """One full grab -> observe -> decide -> act iteration.

        Exposed for ``--once`` mode and replay tests (plan §4.6). A pipeline
        failure (``CaptureError`` / perceive exception) is classified, routed
        to bounded RECOVERY -> PAUSED, and recorded — never raised out of
        this method. Under ``OBSERVE_ONLY`` no ``executor.execute()`` call is
        made at all (defense in depth on top of the policy's own gate).
        """
        now = self._now()
        try:
            frame = self._capture.grab()
        except CaptureError as exc:
            return self._on_pipeline_error(exc, "capture", now)
        try:
            observation = self._perception.observe(frame)
        except Exception as exc:  # noqa: BLE001 - classify, never kill the loop
            return self._on_pipeline_error(exc, "perception", now)

        self._machine.decide(observation, now=now)
        state = self._machine.state
        if state is AgentState.AI_ANALYSIS and self._ai is not None:
            # AI seam: the policy owns the prompt (game-agnostic loop only
            # passes the string to the provider). A provider failure is
            # classified, routed to bounded RECOVERY, never raised out of
            # tick() — same contract as capture/perception.
            prompt_fn = getattr(self._policy, "ai_prompt", None)
            prompt = prompt_fn(state, observation) if prompt_fn else ""
            try:
                decision = self._ai.decide(prompt)
                actions = (decision.action,)
            except Exception as exc:  # noqa: BLE001 - classify, never kill the loop
                return self._on_pipeline_error(exc, "ai", now)
        else:
            actions = self._policy.actions(state, observation, self._mode)
        results: tuple[ActionResult, ...] = ()
        if self._mode.input_allowed and actions:
            results = self._dispatch(actions, now)

        zones = {h.zone: h.value for h in observation.ui_zones}
        self._emit(
            self._event(
                event=EventType.TICK,
                ts=now,
                state_before=state,
                state_after=state,
                reason=f"tick @ {state.value}",
                detail={"zones": zones, "state": state.value},
            )
        )
        return TickResult(
            observation=observation, state=state, actions=actions, results=results
        )

    # ---- action dispatch ----------------------------------------------------

    def _dispatch(
        self, actions: tuple[Action, ...], now: float
    ) -> tuple[ActionResult, ...]:
        """Execute an action batch; record each result; stop on pause/stop.

        ``actions_attempted`` counts only the actions actually dispatched to
        the executor (one per ``execute()`` call), not the full batch size —
        so a mid-batch stop does not inflate the counter with undelivered
        actions (matches the ``AgentLoopReport`` docstring).
        """
        self._emit(
            self._event(
                event=EventType.ACTION_BATCH,
                ts=now,
                state_before=self._machine.state,
                state_after=self._machine.state,
                reason=f"dispatching {len(actions)} action(s)",
                detail={"count": len(actions)},
            )
        )
        results: list[ActionResult] = []
        for action in actions:
            if self._executor.stopped or self._executor.state.name == "PAUSED":
                break
            self._actions_attempted += 1  # F3: count only dispatched actions
            try:
                result = self._executor.execute(action)
            except Exception as exc:  # noqa: BLE001 - record, don't crash the batch
                result = ActionResult(
                    action=action, ok=False, error=f"{type(exc).__name__}: {exc}"
                )
            if result.ok:
                self._actions_ok += 1
            else:
                self._actions_failed += 1
            err = result.error
            self._emit(
                self._event(
                    event=EventType.ACTION_RESULT,
                    ts=now,
                    state_before=self._machine.state,
                    state_after=self._machine.state,
                    reason="ok" if result.ok else f"failed: {err}",
                    detail={"kind": action.kind.value, "ok": result.ok, "error": err},
                )
            )
            results.append(result)
        return tuple(results)

    # ---- pipeline failure handling -----------------------------------------

    def _on_pipeline_error(
        self, exc: Exception, stage: str, now: float
    ) -> TickResult:
        """Route a capture/perception failure to bounded RECOVERY -> PAUSED."""
        before = self._machine.state
        if before is not AgentState.RECOVERY:
            self._machine.enter(AgentState.RECOVERY, reason=f"{stage} failed", now=now)
        attempt = self._machine.note_recovery_attempt()
        reason = f"{stage} failed (attempt {attempt}): {exc}"
        ev = self._event(
            event=EventType.RECOVERY,
            ts=now,
            state_before=before,
            state_after=self._machine.state,
            reason=reason,
            detail={"stage": stage, "error": str(exc), "attempt": attempt},
        )
        self._errors.append(ev)
        self._emit(ev)
        if self._machine.recovery_exhausted():
            self._machine.enter(
                AgentState.PAUSED,
                reason="recovery attempts exhausted; safe pause",
                now=now,
            )
        return TickResult(state=self._machine.state, error=reason)

    # ---- the loop -----------------------------------------------------------

    def run(self) -> AgentLoopReport:
        """Blocking loop: grab -> observe -> decide -> act, until stop/limit.

        Exits on (in priority order): ``executor.stopped``
        (``"emergency_stop"``), ``stop_event`` (``"stop_requested"``), the
        machine entering ``PAUSED`` (``"paused"``), ``max_ticks`` reached
        (``"max_ticks"``), or a capture-start failure
        (``"capture_start_failed"``). ``emergency_stop`` and
        ``stop_requested`` are always highest priority. When the final
        permitted tick both exhausts the tick budget AND lands the machine
        in ``PAUSED`` (e.g. recovery exhausted on the last tick),
        ``"paused"`` is reported because it is the more actionable
        diagnosis — a routine budget limit must not obscure a
        safety-related pause.

        Cleanup runs in ``finally`` on every path — including exceptions
        from ``tick()`` — in the order ``release_all()`` → ``disarm()`` →
        ``capture.stop()``; each step is wrapped in its own ``try/except``
        so one cleanup failure cannot skip the others. The original
        exception (if any) propagates; it is never converted into a success
        report.

        When a ``hotkey`` listener is provided it is armed inside the
        ``try`` (so an arm failure still triggers cleanup) and disarmed in
        ``finally``; a trigger during the loop sets the ``stop_event`` and
        fires the executor's emergency stop, so the loop exits immediately
        after the tick in flight — before pacing — with
        ``stopped_reason == "emergency_stop"``.
        """
        self._reset_report()
        try:
            self._capture.start()
        except Exception:  # noqa: BLE001 - can't start: hard stop, report it
            # Defensive cleanup: nothing was armed or dispatched, but a
            # partially-opened backend must still be released (the real
            # facade's stop() is idempotent; release_all() is a no-op with
            # no held inputs).
            self._cleanup()
            return self._finish_report("capture_start_failed")
        try:
            if self._hotkey is not None:
                self._hotkey.arm()
            while True:
                # emergency_stop has highest priority: the executor owns the
                # stop; the loop only observes and reports.
                if self._executor.stopped:
                    self._stopped_reason = "emergency_stop"
                    break
                if self._stop_event is not None and self._stop_event.is_set():
                    self._stopped_reason = "stop_requested"
                    break
                if self._max_ticks is not None and self._ticks >= self._max_ticks:
                    self._stopped_reason = "max_ticks"
                    break
                self.tick()
                self._ticks += 1
                # P2 (review of 1848b9e) / P3 (review of 924bc29): a stop
                # that fires DURING the tick must not pay a
                # wait_next_frame() before the loop notices it — recheck the
                # stop conditions here in the documented priority order
                # (emergency stop first, then stop_event), and exit without
                # pacing when one is set. These take precedence over a
                # same-tick PAUSED entry so the reported reason always honors
                # the highest-priority stop.
                if self._executor.stopped:
                    self._stopped_reason = "emergency_stop"
                    break
                if self._stop_event is not None and self._stop_event.is_set():
                    self._stopped_reason = "stop_requested"
                    break
                if self._machine.state is AgentState.PAUSED:
                    self._stopped_reason = "paused"
                    break
                self._capture.wait_next_frame()  # F2: pace before the next tick
        finally:
            # F1: cleanup runs on every exit path (break, exception, …);
            # the original exception (if any) propagates after cleanup.
            self._cleanup()
        return self._finish_report(self._stopped_reason)

    def _cleanup(self) -> None:
        """release_all() → disarm() → capture.stop().

        Each step is wrapped in its own ``try/except`` so one cleanup
        failure cannot skip the others, and a cleanup failure must never
        mask the original exception (``finally``) or the
        ``capture_start_failed`` report. All three are idempotent on a
        clean state, so calling this on every exit path is safe.
        """
        try:
            self._executor.release_all()
        except Exception:  # noqa: BLE001 - belt-and-braces; must not mask the stop
            pass
        if self._hotkey is not None:
            try:
                self._hotkey.disarm()
            except Exception:  # noqa: BLE001 - disarm must not mask capture.stop
                pass
        try:
            self._capture.stop()
        except Exception:  # noqa: BLE001 - one cleanup failure must not skip this
            pass

    # ---- report assembly ----------------------------------------------------

    def _reset_report(self) -> None:
        """Zero the accumulators so each ``run()`` starts from a clean slate."""
        self._ticks = 0
        self._transitions = []
        self._actions_attempted = 0
        self._actions_ok = 0
        self._actions_failed = 0
        self._errors = []
        self._stopped_reason = None
        self._start_wall = self._now()

    def _finish_report(self, stopped_reason: str | None) -> AgentLoopReport:
        """Assemble the :class:`AgentLoopReport` from the accumulators."""
        base = self._start_wall if self._start_wall is not None else self._now()
        return AgentLoopReport(
            ticks=self._ticks,
            transitions=tuple(self._transitions),
            actions_attempted=self._actions_attempted,
            actions_ok=self._actions_ok,
            actions_failed=self._actions_failed,
            errors=tuple(self._errors),
            final_state=self._machine.state,
            stopped_reason=stopped_reason,
            wall_time_s=self._now() - base,
        )
