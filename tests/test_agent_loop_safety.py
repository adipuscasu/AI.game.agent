"""Step 6 red — AgentLoop safety integration: hotkey stops the loop (plan §Step 6).

TDD red: the current (Step-5) ``AgentLoop`` does not yet:

* accept a ``hotkey`` (``HotkeyListener``) kwarg in its constructor;
* detect ``executor.stopped`` during ``run()`` and report
  ``stopped_reason == "emergency_stop"``.

Every import here resolves against the committed tree; the two red tests
fail with either ``TypeError`` (hotkey kwarg) or ``AssertionError``
(stopped_reason), which is the correct red reason.

The three spec-lock tests (release_all on PAUSED exit, capture.stop()
in finally even when release_all raises, and per-action batch isolation)
are expected to pass against the Step-5 implementation — they lock the
behaviour Step 6 must not regress.
"""

from __future__ import annotations

import datetime
import threading

import pytest

from ai_game_agent.actions import Executor, OperationMode, SafetyGuard
from ai_game_agent.actions.hotkey import FakeHotkeyListener
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.agent import AgentLoop
from ai_game_agent.capture import CaptureError
from ai_game_agent.games.wow import ZONE_TARGET_HOSTILE, DeterministicPolicy
from ai_game_agent.perception import BBox, Observation, UiZoneHit
from ai_game_agent.state import AgentState, GameStateMachine

TS = datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC)
_BB = BBox(x=10, y=10, width=20, height=10)


def _observation(*, hostile: bool = False) -> Observation:
    zones: tuple[UiZoneHit, ...] = ()
    if hostile:
        zones = (UiZoneHit(zone=ZONE_TARGET_HOSTILE, check="presence", bbox=_BB, value=1.0),)
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


class _StubSignals:
    """StatePolicy stub: returns empty signals (IDLE stays IDLE)."""

    def name(self) -> str:
        return "stub"

    def actions(self, state, observation, mode) -> tuple:
        return ()

    def transition_signals(self, observation) -> dict:
        return {}


class _FakeCapture:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.started = False
        self.stopped = False
        self.stop_calls = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self.stop_calls += 1

    def grab(self):
        if self.fail:
            raise CaptureError("scripted capture failure (test)")
        return object()

    def wait_next_frame(self) -> float:
        return 0.0


class _FakePerception:
    def __init__(self, *, hostile: bool = False) -> None:
        self._obs = _observation(hostile=hostile)

    def observe(self, frame):
        return self._obs


def _make_loop(
    *,
    capture: object,
    perception: object,
    executor: Executor,
    machine: GameStateMachine,
    mode: OperationMode = OperationMode.AUTONOMOUS,
    max_ticks: int | None = 2,
    stop_event: threading.Event | None = None,
    hotkey: FakeHotkeyListener | None = None,
) -> AgentLoop:
    kwargs: dict[str, object] = dict(
        capture=capture,
        perception=perception,
        executor=executor,
        machine=machine,
        policy=DeterministicPolicy(),
        mode=mode,
        max_ticks=max_ticks,
        now_fn=lambda: 0.0,
        stop_event=stop_event,
    )
    if hotkey is not None:
        kwargs["hotkey"] = hotkey
    return AgentLoop(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Red test 1 — hotkey kwarg is accepted (Step-5 loop raises TypeError)
# ---------------------------------------------------------------------------


def test_hotkey_kwarg_accepted_by_loop() -> None:
    """Constructing AgentLoop with a hotkey= keyword must not raise TypeError."""
    backend = MockBackend()
    executor = Executor(backend=backend, safety=SafetyGuard(mode=OperationMode.AUTONOMOUS))
    machine = GameStateMachine(
        policy=_StubSignals(),
        initial=AgentState.IDLE,
        recovery_attempts=3,
    )
    hotkey = FakeHotkeyListener(on_trigger=lambda: None)
    loop = _make_loop(
        capture=_FakeCapture(),
        perception=_FakePerception(),
        executor=executor,
        machine=machine,
        hotkey=hotkey,
    )
    assert loop is not None


# ---------------------------------------------------------------------------
# Red test 2 — emergency stop during run() exits with stopped_reason=="emergency_stop"
# ---------------------------------------------------------------------------


def test_emergency_stop_exits_run_with_reason() -> None:
    """Triggering FakeHotkeyListener during run() must cause run() to exit
    with stopped_reason == "emergency_stop" and executor.stopped is True."""
    backend = MockBackend()
    executor = Executor(backend=backend, safety=SafetyGuard(mode=OperationMode.AUTONOMOUS))
    machine = GameStateMachine(
        policy=_StubSignals(),
        initial=AgentState.IDLE,
        recovery_attempts=3,
    )
    stop_event = threading.Event()
    hotkey = FakeHotkeyListener(
        on_trigger=lambda: (stop_event.set(), executor.trigger_emergency_stop()),
    )
    loop = _make_loop(
        capture=_FakeCapture(),
        perception=_FakePerception(),
        executor=executor,
        machine=machine,
        max_ticks=10,
        stop_event=stop_event,
        hotkey=hotkey,
    )
    # Trigger the hotkey before calling run(); the loop must detect the
    # executor's stopped flag on its first check and exit immediately.
    hotkey.arm()
    hotkey.trigger("f12")
    report = loop.run()
    assert executor.stopped is True
    assert report.stopped_reason == "emergency_stop"
    assert report.ticks <= 1


# ---------------------------------------------------------------------------
# Spec-lock 1 — release_all is called on PAUSED exit
# ---------------------------------------------------------------------------


def test_paused_exit_calls_release_all() -> None:
    """When the loop exits via PAUSED (recovery exhausted), executor.release_all()
    must have been called (belt-and-braces, even if the executor already
    auto-released in its own finally blocks).

    We patch executor.release_all to a recorder and assert it was called.
    """
    backend = MockBackend()
    executor = Executor(backend=backend, safety=SafetyGuard(mode=OperationMode.AUTONOMOUS))
    machine = GameStateMachine(
        policy=_StubSignals(),
        initial=AgentState.RECOVERY,
        recovery_attempts=1,
    )
    capture = _FakeCapture(fail=True)

    release_all_called = {"count": 0}
    original_release_all = executor.release_all

    def recording_release_all():
        release_all_called["count"] += 1
        return original_release_all()

    executor.release_all = recording_release_all  # type: ignore[method-assign]

    loop = _make_loop(
        capture=capture,
        perception=_FakePerception(),
        executor=executor,
        machine=machine,
        max_ticks=5,
    )
    report = loop.run()
    assert report.stopped_reason == "paused"
    assert release_all_called["count"] >= 1, (
        f"release_all not called on PAUSED exit (calls={release_all_called['count']})"
    )


# ---------------------------------------------------------------------------
# Spec-lock 2 — capture.stop() is called even when executor.release_all raises
# ---------------------------------------------------------------------------


def test_capture_stop_called_even_when_release_all_raises() -> None:
    """If executor.release_all() raises, capture.stop() must still be called
    in the outer finally block."""
    backend = MockBackend()
    executor = Executor(backend=backend, safety=SafetyGuard(mode=OperationMode.AUTONOMOUS))

    def raising_release_all() -> bool:
        raise RuntimeError("simulated release failure")

    executor.release_all = raising_release_all  # type: ignore[method-assign]

    machine = GameStateMachine(
        policy=_StubSignals(),
        initial=AgentState.IDLE,
        recovery_attempts=3,
    )
    capture = _FakeCapture()
    loop = _make_loop(
        capture=capture,
        perception=_FakePerception(),
        executor=executor,
        machine=machine,
        max_ticks=1,
    )
    report = loop.run()
    assert report.ticks == 1
    assert capture.stopped is True, "capture.stop() was not called after release_all raised"


# ---------------------------------------------------------------------------
# Spec-lock 3 — per-action batch isolation (no blind retries)
# ---------------------------------------------------------------------------


def test_batch_stops_when_executor_stops_mid_batch() -> None:
    """If the executor stops mid-batch (emergency stop fired), the loop must
    stop dispatching further actions in the batch — no blind retry."""
    backend = MockBackend()
    backend.open()
    executor = Executor(backend=backend, safety=SafetyGuard(mode=OperationMode.AUTONOMOUS))
    machine = GameStateMachine(
        policy=_StubSignals(),
        initial=AgentState.COMBAT,
        recovery_attempts=3,
    )

    original_execute = executor.execute
    call_count = {"n": 0}

    def execute_and_stop(action):
        call_count["n"] += 1
        result = original_execute(action)
        executor.trigger_emergency_stop("mid-batch stop")
        return result

    executor.execute = execute_and_stop  # type: ignore[method-assign]

    loop = _make_loop(
        capture=_FakeCapture(),
        perception=_FakePerception(hostile=True),
        executor=executor,
        machine=machine,
        max_ticks=1,
    )
    loop.tick()
    assert executor.stopped is True
    # Only one action was executed before the stop; the rest of the batch
    # (COMBAT produces multiple actions) must not have been dispatched.
    assert call_count["n"] == 1, (
        f"expected exactly 1 execute() call, got {call_count['n']} — batch was not halted"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
