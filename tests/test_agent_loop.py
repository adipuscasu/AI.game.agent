"""Step 5 reds — AgentLoop.tick() + run() (plan §4.6).

TDD red: the ``ai_game_agent.agent`` package does not exist yet, so collecting
this module fails with::

    ModuleNotFoundError: No module named 'ai_game_agent.agent'

That is the *correct* red reason (a missing module, not a typo). Every other
import here resolves against the committed tree.

These tests are the spec for the green: they wire the *real* executor,
safety guard, state machine, and WoW policy, drive the (not-yet-built)
AgentLoop, and assert the Step-5 contract:

* OBSERVE_ONLY performs grab + observe + decide but **zero**
  executor.execute() calls (the action MockBackend's ``events`` stays empty).
* ``run(max_ticks=N)`` returns an AgentLoopReport with ``ticks == N`` and
  ``stopped_reason == "max_ticks"``.
* ``run()`` with ``stop_event`` pre-set returns ``ticks == 0`` and
  ``stopped_reason == "stop_requested"`` (before any grab).
* ``run()`` with a grab that raises CaptureError routes to RECOVERY, then
  PAUSED once ``recovery_attempts`` is exhausted (``stopped_reason == "paused"``).
* The session-wide event ``seq`` is 1-based and strictly increasing.
* A TICK event's ``detail`` carries the zone names so a log reader can
  reconstruct the decision without re-running perception.
"""

from __future__ import annotations

import datetime
import threading

import pytest

from ai_game_agent.actions import Executor, OperationMode, SafetyGuard
from ai_game_agent.actions.hotkey import FakeHotkeyListener
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.agent import AgentLoop, AgentLoopReport
from ai_game_agent.capture import CaptureError
from ai_game_agent.games.wow import (
    ZONE_LOOT_GLOW,
    ZONE_TARGET,
    ZONE_TARGET_HOSTILE,
    DeterministicPolicy,
)
from ai_game_agent.perception import BBox, Observation, UiZoneHit
from ai_game_agent.state import AgentState, EventType, GameStateMachine, LoopEvent

TS = datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC)
_BB = BBox(x=10, y=10, width=20, height=10)


def _observation(
    *, loot: bool = False, target: bool = False, hostile: bool = False
) -> Observation:
    zones: tuple[UiZoneHit, ...] = ()
    if loot:
        zones = (UiZoneHit(zone=ZONE_LOOT_GLOW, check="presence", bbox=_BB, value=1.0),)
    if target:
        zones = zones + (UiZoneHit(zone=ZONE_TARGET, check="presence", bbox=_BB, value=1.0),)
    if hostile:
        zones = zones + (
            UiZoneHit(zone=ZONE_TARGET_HOSTILE, check="presence", bbox=_BB, value=1.0),
        )
    return Observation(
        frame_width=128,
        frame_height=72,
        captured_at=TS,
        source="test",
        templates=(),
        ui_zones=zones,
        text_regions=(),
        objects=(),
        detector_errors=(),
    )


class _FakeCapture:
    """Duck-typed capture: the loop only needs start/stop/grab/wait_next_frame."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.grab_count = 0
        self.wait_count = 0
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def grab(self):
        self.grab_count += 1
        if self.fail:
            raise CaptureError("scripted capture failure (test)")
        return object()  # the fake perception ignores the frame payload

    def wait_next_frame(self) -> float:
        self.wait_count += 1
        return 0.0


class _FakePerception:
    def __init__(
        self,
        observation: Observation | None = None,
        on_observe: object | None = None,
    ) -> None:
        self._obs = observation if observation is not None else _observation()
        self._on_observe = on_observe
        self.observe_count = 0

    def observe(self, frame):
        self.observe_count += 1
        if self._on_observe is not None:
            self._on_observe()  # P2 seam: fire a stop DURING the tick
        return self._obs


class _StubDecisionPolicy:
    """Supplies machine.decide() signals; the loop's *action* policy is real."""

    def __init__(self, signals: dict[str, object] | None = None) -> None:
        self._signals = dict(signals or {})

    def transition_signals(self, observation):
        return dict(self._signals)


class _RaisingSignalPolicy:
    """StatePolicy whose transition_signals raises.

    Forces an exception out of ``machine.decide()`` (and therefore out of
    ``tick()`` / ``run()``) — the failure seam for the F1 regression test.
    """

    def transition_signals(self, observation):
        raise RuntimeError("simulated decision failure (F1 regression)")


@pytest.fixture
def action_backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
def make_loop(action_backend: MockBackend):
    def _make(
        *,
        mode: OperationMode = OperationMode.AUTONOMOUS,
        capture: _FakeCapture | None = None,
        perception: _FakePerception | None = None,
        machine: GameStateMachine | None = None,
        executor: Executor | None = None,
        hotkey: FakeHotkeyListener | None = None,
        max_ticks: int | None = None,
        stop_event: threading.Event | None = None,
        on_event=None,
        recovery_attempts: int = 3,
    ) -> AgentLoop:
        machine = machine or GameStateMachine(
            policy=_StubDecisionPolicy(),
            initial=AgentState.IDLE,
            recovery_attempts=recovery_attempts,
        )
        executor = executor or Executor(
            backend=action_backend,
            safety=SafetyGuard(mode=mode),
        )
        return AgentLoop(
            capture=capture or _FakeCapture(),
            perception=perception or _FakePerception(),
            executor=executor,
            machine=machine,
            policy=DeterministicPolicy(),
            mode=mode,
            max_ticks=max_ticks,
            now_fn=lambda: 0.0,
            stop_event=stop_event,
            hotkey=hotkey,
            on_event=on_event,
        )

    return _make


# ---------------------------------------------------------------------------
# Step-5 contract reds
# ---------------------------------------------------------------------------


def test_observe_only_performs_zero_executor_calls(make_loop, action_backend) -> None:
    machine = GameStateMachine(policy=_StubDecisionPolicy(), initial=AgentState.COMBAT)
    loop = make_loop(mode=OperationMode.OBSERVE_ONLY, machine=machine)
    loop.tick()
    # The safety story: OBSERVE_ONLY may see + decide, but must never emit input.
    assert action_backend.events == []
    assert machine.state is AgentState.COMBAT


def test_run_max_ticks_report(make_loop) -> None:
    loop = make_loop(max_ticks=3)
    report = loop.run()
    assert isinstance(report, AgentLoopReport)
    assert report.ticks == 3
    assert report.stopped_reason == "max_ticks"
    assert isinstance(report.final_state, AgentState)


def test_run_pre_stopped_returns_zero_ticks(make_loop) -> None:
    stop = threading.Event()
    stop.set()
    loop = make_loop(max_ticks=10, stop_event=stop)
    report = loop.run()
    assert report.ticks == 0
    assert report.stopped_reason == "stop_requested"


def test_capture_error_routes_to_recovery_then_paused(make_loop) -> None:
    capture = _FakeCapture(fail=True)
    machine = GameStateMachine(
        policy=_StubDecisionPolicy(),
        initial=AgentState.IDLE,
        recovery_attempts=2,
    )
    loop = make_loop(capture=capture, machine=machine, max_ticks=10)
    report = loop.run()
    assert machine.state is AgentState.PAUSED
    assert report.final_state is AgentState.PAUSED
    assert report.stopped_reason == "paused"


def test_session_seq_is_one_based_and_increasing(make_loop) -> None:
    events: list[LoopEvent] = []
    loop = make_loop(max_ticks=2, on_event=events.append)
    loop.run()
    seqs = [e.seq for e in events]
    assert seqs, "expected at least one event across two ticks"
    assert seqs[0] == 1, f"session seq must be 1-based, got {seqs[0]}"
    assert all(b > a for a, b in zip(seqs, seqs[1:], strict=False)), (
        f"seq must be strictly increasing: {seqs}"
    )


def test_tick_event_carries_zone_detail(make_loop) -> None:
    obs = _observation(loot=True, target=True, hostile=True)
    events: list[LoopEvent] = []
    loop = make_loop(perception=_FakePerception(obs), max_ticks=1, on_event=events.append)
    loop.run()
    ticks = [e for e in events if e.event is EventType.TICK]
    assert ticks, "expected at least one TICK event"
    zones = ticks[0].detail.get("zones")
    assert isinstance(zones, dict)
    assert ZONE_LOOT_GLOW in zones
    assert ZONE_TARGET in zones
    assert ZONE_TARGET_HOSTILE in zones


# ---------------------------------------------------------------------------
# Phase 4 close-out regression tests — lock the working-tree fixes in
# AgentLoop.run() before they are committed:
#
# * F1 — cleanup runs on every exit path (even when tick() raises), in the
#   mandated order: release_all -> disarm -> capture.stop;
# * F2 — pacing is owned by the capture facade: exactly one
#   wait_next_frame() per completed tick, and none after an exit where no
#   tick completed;
# * F3 — actions_attempted counts only the actions actually dispatched
#   (a mid-batch emergency stop must not count the undelivered ones).
# ---------------------------------------------------------------------------


def test_cleanup_runs_in_order_when_tick_raises(make_loop, action_backend) -> None:
    """F1: an exception escaping tick() must propagate out of run() (never a
    success report), and cleanup must still run in the required order:
    release_all -> disarm -> capture.stop.

    Pre-fix, release_all() sat inside the same try as the loop body, so an
    exception in tick() skipped it (only disarm + capture.stop ran, from the
    finally). The shared ``order`` list proves both occurrence and sequence,
    including that disarm runs.
    """
    machine = GameStateMachine(
        policy=_RaisingSignalPolicy(),
        initial=AgentState.IDLE,
        recovery_attempts=3,
    )
    executor = Executor(
        backend=action_backend,
        safety=SafetyGuard(mode=OperationMode.AUTONOMOUS),
    )
    order: list[str] = []

    _orig_release_all = executor.release_all

    def recording_release_all() -> bool:
        order.append("release_all")
        return _orig_release_all()

    executor.release_all = recording_release_all  # type: ignore[method-assign]

    hotkey = FakeHotkeyListener(on_trigger=lambda: None)
    _orig_disarm = hotkey.disarm

    def recording_disarm() -> None:
        order.append("disarm")
        _orig_disarm()

    hotkey.disarm = recording_disarm  # type: ignore[method-assign]

    capture = _FakeCapture()
    _orig_stop = capture.stop

    def recording_stop() -> None:
        order.append("capture.stop")
        _orig_stop()

    capture.stop = recording_stop  # type: ignore[method-assign]

    loop = make_loop(
        capture=capture,
        machine=machine,
        executor=executor,
        hotkey=hotkey,
        max_ticks=5,
    )

    with pytest.raises(RuntimeError, match="simulated decision failure"):
        loop.run()
    # The exception propagated (not swallowed into a report), and cleanup ran
    # in the mandated order — all three steps, including disarm:
    assert order == ["release_all", "disarm", "capture.stop"], (
        f"cleanup order wrong on the exception path: {order}"
    )


def test_pacing_at_tick_boundary(make_loop, action_backend) -> None:
    """F2: the loop paces via the capture facade — exactly one
    wait_next_frame() per completed tick (plan §4.6) — and never paces after
    an exit where no tick completed.

    Scenario 1 (normal): 3 ticks -> exactly 3 pacing calls.
    Scenario 2 (boundary): an emergency stop fired before any tick -> the
    loop exits without pacing at all (wait_count == 0).

    Pre-fix, run() never called wait_next_frame() at all (no throttle), so
    scenario 1 fails with wait_count == 0.
    """
    # Scenario 1: one pacing call per completed tick.
    capture = _FakeCapture()
    loop = make_loop(capture=capture, max_ticks=3)
    report = loop.run()
    assert report.ticks == 3
    assert report.stopped_reason == "max_ticks"
    assert capture.wait_count == 3, (
        f"expected wait_next_frame() once per tick (3), got {capture.wait_count}"
    )

    # Scenario 2: emergency stop before any tick -> no pacing at all.
    boundary_capture = _FakeCapture()
    executor = Executor(
        backend=action_backend,
        safety=SafetyGuard(mode=OperationMode.AUTONOMOUS),
    )
    executor.trigger_emergency_stop("pre-run stop")
    loop2 = make_loop(capture=boundary_capture, executor=executor, max_ticks=5)
    report2 = loop2.run()
    assert report2.ticks == 0
    assert report2.stopped_reason == "emergency_stop"
    assert boundary_capture.wait_count == 0, (
        "must not pace after an exit where no tick completed "
        f"(got {boundary_capture.wait_count})"
    )


def test_actions_attempted_counts_only_dispatched_actions(
    make_loop, action_backend
) -> None:
    """F3: a mid-batch emergency stop must not inflate actions_attempted with
    undelivered actions.

    COMBAT dispatches a 3-action batch (key, delay, key). Stopping the
    executor from the first execute() call means only 1 action was actually
    dispatched — the report must say 1, not the batch size of 3.

    Pre-fix, the counter was seeded with len(batch) up front, so a mid-batch
    stop over-counted (actions_attempted == 3).
    """
    action_backend.open()
    machine = GameStateMachine(
        policy=_StubDecisionPolicy(),
        initial=AgentState.COMBAT,
        recovery_attempts=3,
    )
    executor = Executor(
        backend=action_backend,
        safety=SafetyGuard(mode=OperationMode.AUTONOMOUS),
    )
    calls = {"n": 0}
    _orig_execute = executor.execute

    def execute_and_stop(action):
        calls["n"] += 1
        result = _orig_execute(action)
        executor.trigger_emergency_stop("mid-batch stop")
        return result

    executor.execute = execute_and_stop  # type: ignore[method-assign]

    loop = make_loop(machine=machine, executor=executor, max_ticks=1)
    report = loop.run()
    assert executor.stopped is True
    assert report.stopped_reason == "emergency_stop"
    assert calls["n"] == 1, f"expected exactly 1 execute() call, got {calls['n']}"
    assert report.actions_attempted == 1, (
        f"actions_attempted must count only dispatched actions (1), "
        f"got {report.actions_attempted}"
    )


# ---------------------------------------------------------------------------
# P2 close-out (review of 1848b9e): a stop that fires DURING a tick must
# not pay a wait_next_frame() before the loop notices it. The executor
# releases held input the instant it is triggered, but the loop's own
# cleanup (disarm + capture.stop) and its exit still ran a full frame
# period late. Now: after each completed tick, recheck the stop conditions
# (emergency_stop first, then stop_event) BEFORE pacing — same priority
# order as the top-of-loop checks.
# ---------------------------------------------------------------------------


def test_emergency_stop_during_tick_exits_without_pacing(
    make_loop, action_backend
) -> None:
    """P2-A: the emergency-stop hotkey fires inside the tick (during
    perception); run() must exit right after that tick with
    stopped_reason == "emergency_stop" and ZERO pacing calls.

    IDLE + empty observation dispatches no actions, so the discriminator
    is purely the pacing: pre-fix, run() paced once after the tick
    (wait_count == 1) and only saw the stop at the next iteration top.
    """
    executor = Executor(
        backend=action_backend,
        safety=SafetyGuard(mode=OperationMode.AUTONOMOUS),
    )
    hotkey = FakeHotkeyListener(
        on_trigger=lambda: executor.trigger_emergency_stop("hotkey test")
    )
    # The listener is armed by run() before the first tick, so this fires.
    perception = _FakePerception(on_observe=lambda: hotkey.trigger("f12"))
    capture = _FakeCapture()
    loop = make_loop(
        capture=capture,
        perception=perception,
        executor=executor,
        hotkey=hotkey,
        max_ticks=10,
    )

    report = loop.run()
    assert report.ticks == 1, f"expected 1 tick before the stop, got {report.ticks}"
    assert report.stopped_reason == "emergency_stop", (
        f"stopped_reason should be emergency_stop, got {report.stopped_reason}"
    )
    assert capture.wait_count == 0, (
        "must not pace after an emergency stop fired during tick() "
        f"(got wait_count={capture.wait_count})"
    )
    assert executor.stopped is True
    assert capture.stopped is True, "capture.stop() must run in cleanup"


def test_stop_event_during_tick_exits_without_pacing(
    make_loop, action_backend
) -> None:
    """P2-B: stop_event is set inside the tick (during perception); run()
    must exit right after that tick with stopped_reason ==
    "stop_requested" and ZERO pacing calls — the same responsiveness
    contract as the emergency-stop path, at its own priority level.
    """
    stop_event = threading.Event()
    perception = _FakePerception(on_observe=stop_event.set)
    capture = _FakeCapture()
    loop = make_loop(
        capture=capture,
        perception=perception,
        stop_event=stop_event,
        max_ticks=10,
    )

    report = loop.run()
    assert report.ticks == 1, f"expected 1 tick before the stop, got {report.ticks}"
    assert report.stopped_reason == "stop_requested", (
        f"stopped_reason should be stop_requested, got {report.stopped_reason}"
    )
    assert capture.wait_count == 0, (
        "must not pace after stop_event fired during tick() "
        f"(got wait_count={capture.wait_count})"
    )
    assert capture.stopped is True, "capture.stop() must run in cleanup"
