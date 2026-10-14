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
        return 0.0


class _FakePerception:
    def __init__(self, observation: Observation | None = None) -> None:
        self._obs = observation if observation is not None else _observation()
        self.observe_count = 0

    def observe(self, frame):
        self.observe_count += 1
        return self._obs


class _StubDecisionPolicy:
    """Supplies machine.decide() signals; the loop's *action* policy is real."""

    def __init__(self, signals: dict[str, object] | None = None) -> None:
        self._signals = dict(signals or {})

    def transition_signals(self, observation):
        return dict(self._signals)


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
        executor = Executor(
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
