"""EH2 reds — make agent-loop failures observable without compromising safe shutdown.

TDD red (Phase 7, work package EH2): the loop's RECOVERY path currently
records only ``str(exc)`` (``detail={"stage", "error", "attempt"}``), so an
operational ``CaptureError`` and an unexpected programming ``RuntimeError``
collapse into one opaque message — the exception *type*, the formatted
traceback, and the ``__cause__`` / ``__context__`` chain are all lost.
Separately, a failing cleanup step (``release_all`` / ``disarm`` /
``capture.stop``) is swallowed by a bare ``except Exception: pass`` and
leaves zero observable evidence, and a throwing event sink inside cleanup
would mask the primary exception.

These tests are the spec for the green:

* RECOVERY events keep the existing fields (``stage`` / ``error`` /
  ``attempt``) and gain exception identity: ``exc_type``, ``exc_repr``,
  explicit ``exc_cause`` and implicit ``exc_context`` where applicable, and
  a formatted, **length-capped** traceback that contains **no local
  variables**.
* The RECOVERY *routing* and bounded-recovery safety contract are unchanged:
  operational and unexpected failures both route to RECOVERY (never out of
  ``tick()``), and now remain distinguishable in the audit trail.
* A failed cleanup step is recorded as an ``EventType.ERROR`` event
  (step name + exception identity + capped traceback) without skipping the
  remaining steps, without masking the primary exception, and without
  perturbing the ``capture_start_failed`` report.
* Recording cleanup evidence is best-effort and non-throwing: if the event
  sink itself raises, the remaining cleanup steps still run and the
  primary exception is still the one that propagates.

Every test is headless (mock backend, duck-typed capture/perception, no
desktop session).
"""

from __future__ import annotations

import datetime
import re
import threading

import pytest

from ai_game_agent.actions import (
    Executor,
    OperationMode,
    SafetyGuard,
)
from ai_game_agent.actions.hotkey import FakeHotkeyListener
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.agent import AgentLoop
from ai_game_agent.capture import CaptureError
from ai_game_agent.games.wow import DeterministicPolicy
from ai_game_agent.perception import Observation
from ai_game_agent.state import (
    AgentState,
    EventType,
    GameStateMachine,
    LoopEvent,
)

_TS = datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC)

_PERCEPTION_MSG = "simulated perception failure (EH2)"


def _observation() -> Observation:
    return Observation(
        frame_width=128,
        frame_height=72,
        captured_at=_TS,
        source="test",
        templates=(),
        ui_zones=(),
        text_regions=(),
        objects=(),
        detector_errors=(),
    )


class _FakeCapture:
    """Duck-typed capture: the loop only needs start/stop/grab/wait_next_frame."""

    def __init__(self, *, fail_grab: bool = False, fail_stop: bool = False) -> None:
        self.fail_grab = fail_grab
        self.fail_stop = fail_stop
        self.grab_count = 0
        self.wait_count = 0
        self.stopped = False
        self.stop_call_count = 0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.stop_call_count += 1
        self.stopped = True
        if self.fail_stop:
            raise OSError("simulated capture.stop failure (EH2)")

    def grab(self):
        self.grab_count += 1
        if self.fail_grab:
            raise CaptureError("simulated capture failure (EH2)")
        return _frame_like()

    def wait_next_frame(self) -> float:
        self.wait_count += 1
        return 0.0


def _frame_like() -> object:
    """A grab return the loop's non-AI path never image-encodes."""
    from ai_game_agent.capture import Frame

    return Frame(width=1, height=1, pixels=b"\x00" * 3, captured_at=_TS, source="test")


class _ScriptedPerception:
    """Perception whose observe() raises a scripted exception (or sequence)."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.observe_count = 0

    def observe(self, frame):
        self.observe_count += 1
        raise self._exc


class _StubDecisionPolicy:
    """StatePolicy: no transition signals (the loop's action policy is real)."""

    def transition_signals(self, observation):
        return {}


class _RaisingSignalPolicy:
    """StatePolicy whose transition_signals raises: forces an exception out of
    ``machine.decide()`` (the F1 primary-exception seam)."""

    def transition_signals(self, observation):
        raise RuntimeError("simulated decision failure (EH2 primary)")


@pytest.fixture
def action_backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
def make_loop(action_backend: MockBackend):
    def _make(
        *,
        mode: OperationMode = OperationMode.AUTONOMOUS,
        capture: _FakeCapture | None = None,
        perception: object | None = None,
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
            perception=perception or _ScriptedPerception(RuntimeError("unused")),
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


def _recovery_events(events: list[LoopEvent]) -> list[LoopEvent]:
    return [e for e in events if e.event is EventType.RECOVERY]


def _error_events(events: list[LoopEvent]) -> list[LoopEvent]:
    return [e for e in events if e.event is EventType.ERROR]


class _PassingPerception:
    """Perception that returns a fixed observation (no failure)."""

    def observe(self, frame):
        return _observation()


# ---------------------------------------------------------------------------
# 1. RECOVERY events preserve the failed exception's identity, chain, and a
#    formatted (length-capped, no-locals) traceback — while keeping the
#    existing fields and the existing RECOVERY routing.
# ---------------------------------------------------------------------------


def test_recovery_event_preserves_exception_identity(make_loop) -> None:
    """A perception failure's RECOVERY event must carry the exception's
    type, repr, a formatted traceback containing the raise site, and keep
    the pre-existing stage/error/attempt fields verbatim. The failure must
    still route to bounded RECOVERY (never out of tick())."""
    perception = _ScriptedPerception(RuntimeError(_PERCEPTION_MSG))
    events: list[LoopEvent] = []
    loop = make_loop(perception=perception, on_event=events.append)

    result = loop.tick()

    # Routing unchanged: bounded RECOVERY on the failing tick, not PAUSED.
    assert result.state is AgentState.RECOVERY
    assert loop._machine.state is AgentState.RECOVERY

    recovery = _recovery_events(events)
    assert len(recovery) == 1
    ev = recovery[0]

    # Pre-existing fields are preserved verbatim (backward compatibility).
    assert ev.detail.get("stage") == "perception"
    assert ev.detail.get("error") == _PERCEPTION_MSG
    assert ev.detail.get("attempt") == 1

    # EH2: exception identity is retained.
    assert ev.detail.get("exc_type") == "RuntimeError"
    assert ev.detail.get("exc_repr") == f"RuntimeError({_PERCEPTION_MSG!r})"

    tb = ev.detail.get("exc_traceback")
    assert isinstance(tb, str) and tb, "the RECOVERY event must carry a traceback"
    assert "Traceback" in tb
    assert "RuntimeError" in tb
    assert _PERCEPTION_MSG in tb

    # Unchained failure: no cause/context fields (nothing to preserve).
    assert "exc_cause" not in ev.detail
    assert "exc_context" not in ev.detail

    # The reason (logs / TickResult.error) names the exception type.
    assert "perception failed" in result.error
    assert "RuntimeError" in result.error
    assert _PERCEPTION_MSG in result.error


def test_recovery_event_records_explicit_cause_and_context(make_loop) -> None:
    """An explicit `raise ... from cause` inside an active exception handler
    — so BOTH `__cause__` (the explicit `from`) and `__context__` (the
    exception being handled) are set to the same inner failure. Both
    `exc_cause` and `exc_context` must be preserved independently."""
    cause = LookupError("inner cause (EH2)")

    def observe(frame):
        try:
            raise cause
        except LookupError:
            raise ValueError("outer failure (EH2)") from cause

    perception = type("_P", (), {"observe": staticmethod(observe)})()
    events: list[LoopEvent] = []
    loop = make_loop(perception=perception, on_event=events.append)

    result = loop.tick()
    assert result.state is AgentState.RECOVERY  # routing unchanged

    ev = _recovery_events(events)[0]
    assert ev.detail.get("exc_type") == "ValueError"
    assert ev.detail.get("exc_cause") == repr(cause)
    assert ev.detail.get("exc_context") == repr(cause)

    # The formatted traceback renders the explicit chain (both frames/types).
    tb = ev.detail["exc_traceback"]
    assert "ValueError" in tb and "LookupError" in tb
    assert "inner cause (EH2)" in tb


def test_recovery_event_records_explicit_cause_no_active_context(make_loop) -> None:
    """An explicit `raise ... from cause` performed OUTSIDE an active
    exception handler: `__cause__` is set and `__context__` is None. The
    RECOVERY event must record `exc_cause` and must NOT fabricate
    `exc_context` from the cause — each field mirrors its actual Python
    attribute independently. The formatted traceback still preserves the
    complete explicit chain."""

    class CauseError(Exception):
        """Inner explicit cause (EH2)."""

    class WrapperError(Exception):
        """Outer wrapper (EH2)."""

    cause = CauseError("explicit cause (EH2)")

    def observe(frame):
        raise WrapperError("wrapper failure (EH2)") from cause

    perception = type("_P", (), {"observe": staticmethod(observe)})()
    events: list[LoopEvent] = []
    loop = make_loop(perception=perception, on_event=events.append)

    result = loop.tick()
    assert result.state is AgentState.RECOVERY  # routing unchanged

    ev = _recovery_events(events)[0]
    assert ev.detail.get("exc_type") == "WrapperError"
    assert ev.detail.get("exc_cause") == repr(cause)
    assert "exc_context" not in ev.detail, (
        "no active handler at raise time: __context__ is None, so "
        "exc_context must not be fabricated from the cause"
    )

    # The formatted traceback preserves the complete explicit chain.
    tb = ev.detail["exc_traceback"]
    assert "WrapperError" in tb and "CauseError" in tb
    assert "explicit cause (EH2)" in tb


def test_recovery_event_records_implicit_context_only(make_loop) -> None:
    """An implicit chain (no `from`): `__cause__` is None and `__context__`
    carries the inner exception — the implicit context must be preserved
    and `exc_cause` must not be fabricated."""
    cause_repr = "LookupError('inner context (EH2)')"

    def observe(frame):
        try:
            raise LookupError("inner context (EH2)")
        except LookupError:
            raise ValueError("outer failure (EH2)")  # noqa: B904 - implicit chain (no `from`) by design

    perception = type("_P", (), {"observe": staticmethod(observe)})()
    events: list[LoopEvent] = []
    loop = make_loop(perception=perception, on_event=events.append)

    result = loop.tick()
    assert result.state is AgentState.RECOVERY  # routing unchanged

    ev = _recovery_events(events)[0]
    assert ev.detail.get("exc_type") == "ValueError"
    assert "exc_cause" not in ev.detail, "no explicit `from`: cause must stay absent"
    assert ev.detail.get("exc_context") == cause_repr
    assert "inner context (EH2)" in ev.detail["exc_traceback"]


def test_recovery_traceback_is_length_capped_and_has_no_locals(make_loop) -> None:
    """Tracebacks are formatted without local variables and length-capped;
    small exceptions keep their full traceback; the capped form still ends
    with the exception line (the raise site / message tail survives)."""
    secret_local = "SECRET_LOCAL_MARKER_7f3a9c"
    big_msg = "EH2-cap-marker " + "x" * 20_000

    def observe(frame):
        local = secret_local  # noqa: F841 - intentionally unused: a locals-printing formatter would leak this
        raise RuntimeError(big_msg)

    perception = type("_P", (), {"observe": staticmethod(observe)})()
    events: list[LoopEvent] = []
    loop = make_loop(perception=perception, on_event=events.append)
    loop.tick()
    ev = _recovery_events(events)[0]
    tb = ev.detail["exc_traceback"]

    # Length-capped at _TB_CAP (4000).
    assert len(tb) <= 4_000, f"traceback must be length-capped at _TB_CAP (got {len(tb)} chars)"
    assert "truncated" in tb, "the capped form must mark the truncation"
    # The tail (exception line / message tail) survives the cap.
    assert big_msg[-20:] in tb
    # No local variables are printed.
    assert secret_local not in tb
    assert "local" not in tb.split("\n")[-1]

    # A small, normal exception keeps its *full* traceback (no false cap).
    small: list[LoopEvent] = []
    small_loop = make_loop(
        perception=_ScriptedPerception(RuntimeError(_PERCEPTION_MSG)),
        on_event=small.append,
    )
    small_loop.tick()
    small_tb = _recovery_events(small)[0].detail["exc_traceback"]
    assert "truncated" not in small_tb
    assert _PERCEPTION_MSG in small_tb
    assert small_tb.rstrip().endswith("RuntimeError: " + _PERCEPTION_MSG)


def test_operational_and_unexpected_errors_stay_distinguishable(make_loop) -> None:
    """A `CaptureError` (operational) and a `RuntimeError` (unexpected
    programming error) both route to bounded RECOVERY — but their RECOVERY
    events must remain distinguishable: distinct stages AND distinct
    exception types (the current single opaque `str(exc)` collapses them)."""
    events_cap: list[LoopEvent] = []
    loop_cap = make_loop(capture=_FakeCapture(fail_grab=True), on_event=events_cap.append)
    res_cap = loop_cap.tick()
    ev_cap = _recovery_events(events_cap)[0]

    events_rt: list[LoopEvent] = []
    loop_rt = make_loop(
        perception=_ScriptedPerception(RuntimeError(_PERCEPTION_MSG)),
        on_event=events_rt.append,
    )
    res_rt = loop_rt.tick()
    ev_rt = _recovery_events(events_rt)[0]

    # Routing unchanged for both (fail-safe, bounded).
    assert res_cap.state is AgentState.RECOVERY
    assert res_rt.state is AgentState.RECOVERY

    # Distinguishable in the audit trail.
    assert ev_cap.detail.get("stage") == "capture"
    assert ev_rt.detail.get("stage") == "perception"
    assert ev_cap.detail.get("exc_type") == "CaptureError"
    assert ev_rt.detail.get("exc_type") == "RuntimeError"


# ---------------------------------------------------------------------------
# 2. Cleanup failures become observable ERROR events without compromising
#    safe shutdown: order preserved, primary exception preserved, and the
#    recording itself is best-effort / non-throwing.
# ---------------------------------------------------------------------------


def _primary_failure_loop(make_loop, *, capture, hotkey=None, on_event=None):
    """A loop whose `machine.decide()` raises (F1 seam): the primary
    exception escapes `run()` and cleanup runs in `finally`.

    Perception must *pass* (perception precedes decide) — only the machine
    signal policy raises."""
    machine = GameStateMachine(
        policy=_RaisingSignalPolicy(), initial=AgentState.IDLE, recovery_attempts=3
    )
    return make_loop(
        capture=capture,
        perception=_PassingPerception(),
        machine=machine,
        hotkey=hotkey,
        on_event=on_event,
    )


def test_cleanup_failure_is_recorded_and_primary_propagates(make_loop) -> None:
    """Primary exception + a failing `capture.stop()`: the primary still
    propagates, cleanup order is preserved, and the failed step is recorded
    as an ERROR event (step name + exception identity + traceback)."""
    capture = _FakeCapture(fail_stop=True)
    events: list[LoopEvent] = []
    loop = _primary_failure_loop(make_loop, capture=capture, on_event=events.append)

    with pytest.raises(RuntimeError, match=re.escape("simulated decision failure (EH2 primary)")):
        loop.run()

    # The failing step was still attempted (not skipped).
    assert capture.stopped is True
    assert capture.stop_call_count == 1

    errors = _error_events(events)
    assert len(errors) == 1, "exactly one ERROR event for one failing cleanup step"
    ev = errors[0]
    assert ev.detail.get("step") == "capture.stop"
    assert ev.detail.get("exc_type") == "OSError"
    assert "simulated capture.stop failure (EH2)" in ev.detail.get("error", "")
    assert "OSError" in ev.detail.get("exc_traceback", "")

    # No ERROR event for the two steps that succeeded.
    steps = {e.detail.get("step") for e in errors}
    assert steps == {"capture.stop"}


def test_cleanup_event_recording_failure_is_swallowed(make_loop) -> None:
    """The event sink itself raising during cleanup: the remaining steps
    must still execute and the PRIMARY exception must remain the one that
    propagates (a broken sink must not mask it)."""
    hotkey = FakeHotkeyListener(on_trigger=lambda: None)

    def failing_disarm() -> None:
        raise OSError("simulated disarm failure (EH2)")

    hotkey.disarm = failing_disarm  # type: ignore[method-assign]

    capture = _FakeCapture()  # capture.stop succeeds and is recorded

    def broken_sink(event: LoopEvent) -> None:
        raise OSError("event bus down (EH2)")

    loop = _primary_failure_loop(make_loop, capture=capture, hotkey=hotkey)
    loop._on_event = broken_sink  # the event sink is broken

    with pytest.raises(RuntimeError, match=re.escape("simulated decision failure (EH2 primary)")):
        loop.run()

    # The step AFTER the failing one still ran (recording did not block it).
    assert capture.stopped is True, "capture.stop must run after a broken-sink step"

    # The failing step's evidence is retained (recorded before the broken
    # sink could reject it) even though the sink never saw it.
    errors = _error_events(loop._errors)
    assert len(errors) == 1
    assert errors[0].detail.get("step") == "disarm"
    assert errors[0].detail.get("exc_type") == "OSError"


def test_capture_start_failed_retains_cleanup_evidence(make_loop) -> None:
    """`capture.start()` failure + a failing `capture.stop()`: the report
    still says `capture_start_failed`, and the failed cleanup step is
    retained in the report's error trail (both pieces of evidence)."""
    capture = _FakeCapture(fail_stop=True)
    capture.start = lambda: (_ for _ in ()).throw(CaptureError("simulated start failure (EH2)"))
    loop = make_loop(capture=capture)

    report = loop.run()  # must NOT raise
    assert report.stopped_reason == "capture_start_failed"
    assert report.ticks == 0

    errors = _error_events(list(report.errors))
    assert len(errors) == 1
    ev = errors[0]
    assert ev.detail.get("step") == "capture.stop"
    assert ev.detail.get("exc_type") == "OSError"
    assert "simulated capture.stop failure (EH2)" in ev.detail.get("error", "")


def test_clean_run_records_no_error_events(make_loop) -> None:
    """Over-emission guard: a clean run and a primary-failure run whose
    cleanup fully succeeds must record NO ERROR events (only a genuinely
    failing cleanup step earns one)."""
    # Clean run.
    report = make_loop(max_ticks=1).run()
    assert _error_events(list(report.errors)) == []

    # Primary failure, fully successful cleanup.
    capture = _FakeCapture()
    events: list[LoopEvent] = []
    loop = _primary_failure_loop(make_loop, capture=capture, on_event=events.append)
    with pytest.raises(RuntimeError, match="simulated decision failure"):
        loop.run()
    assert capture.stopped is True
    assert _error_events(events) == []
    assert _error_events(list(loop._errors)) == []


def test_clean_run_with_failing_cleanup_is_reported_and_delivered(make_loop) -> None:
    """A run that completes *normally* — no primary exception, no RECOVERY —
    but whose cleanup fails: ``run()`` must still return a report (not
    raise), and the failed cleanup steps must be retained in
    ``AgentLoopReport.errors`` **and** delivered through the normal event
    path (``on_event``) — the evidence exists even though the tick
    pipeline never failed.

    Covers both failing steps (``disarm`` then ``capture.stop``); the
    succeeding ``release_all`` records nothing."""

    def failing_disarm() -> None:
        raise OSError("simulated disarm failure (EH2 clean run)")

    hotkey = FakeHotkeyListener(on_trigger=lambda: None)
    hotkey.disarm = failing_disarm  # type: ignore[method-assign]
    capture = _FakeCapture(fail_stop=True)
    events: list[LoopEvent] = []
    loop = make_loop(
        capture=capture,
        perception=_PassingPerception(),
        hotkey=hotkey,
        max_ticks=1,
        on_event=events.append,
    )

    report = loop.run()  # must NOT raise: the run itself completed normally

    # The run is clean: one tick, stopped by the budget, never entered
    # RECOVERY, so no RECOVERY events exist either.
    assert report.ticks == 1
    assert report.stopped_reason == "max_ticks"
    assert report.final_state is AgentState.IDLE
    assert _recovery_events(list(report.errors)) == []

    # Both failing steps were attempted (not skipped by each other).
    assert capture.stop_call_count == 1
    assert capture.stopped is True

    # Both failing steps are retained in the report, in cleanup order;
    # the succeeding release_all recorded nothing.
    errors = _error_events(list(report.errors))
    assert [e.detail.get("step") for e in errors] == ["disarm", "capture.stop"]
    disarm_ev, stop_ev = errors
    assert disarm_ev.detail.get("exc_type") == "OSError"
    assert "simulated disarm failure (EH2 clean run)" in disarm_ev.detail.get("error", "")
    assert stop_ev.detail.get("exc_type") == "OSError"
    assert "simulated capture.stop failure (EH2)" in stop_ev.detail.get("error", "")
    assert "OSError" in stop_ev.detail.get("exc_traceback", "")

    # Both were delivered through the normal event path — and the report
    # retains the very same event objects the subscriber saw.
    delivered = _error_events(events)
    assert len(delivered) == 2
    assert delivered[0] is disarm_ev
    assert delivered[1] is stop_ev


# ---------------------------------------------------------------------------
# 7. EH3 F1 reds — a cleanup step whose failure is *itself* unstringifiable
#    must still be recorded best-effort; the diagnostic path must be fully
#    isolated so it can neither abort the remaining cleanup steps nor mask
#    the primary exception.
# ---------------------------------------------------------------------------


def test_cleanup_step_with_raising_exception_repr_is_recorded_and_primary_propagates(make_loop) -> None:  # noqa: E501
    """Primary exception + a failing ``disarm()`` whose exception type
    raises in ``__str__``/``__repr__``: the diagnostic path must be fully
    best-effort — the ``capture.stop`` step after it must still run, and
    the PRIMARY exception must remain the one that propagates (no
    diagnostic exception may mask it). At least one ERROR event must be
    retained for the disarm step (best-effort step name)."""

    class _RaisingDisarm(Exception):
        pass

    def failing_disarm() -> None:
        raise _RaisingDisarm("disarm blew up (EH3)")

    # Make str()/repr() raise: the EH2 diagnostic does
    # f"...{exc}" / str(exc) / repr(exc) inside its except block.
    _RaisingDisarm.__str__ = lambda self: (_ for _ in ()).throw(
        RuntimeError("diagnostic clock failure (EH3)")
    )
    _RaisingDisarm.__repr__ = _RaisingDisarm.__str__

    hotkey = FakeHotkeyListener(on_trigger=lambda: None)
    hotkey.disarm = failing_disarm  # type: ignore[method-assign]
    capture = _FakeCapture()  # capture.stop must still run after the failing step

    loop = _primary_failure_loop(make_loop, capture=capture, hotkey=hotkey)

    with pytest.raises(RuntimeError, match=re.escape("simulated decision failure (EH2 primary)")):
        loop.run()

    # The step AFTER the failing one still ran (the diagnostic could not abort it).
    assert capture.stopped is True, "capture.stop must run after a raising-diagnostic step"
    assert capture.stop_call_count == 1

    # At least one ERROR event retained for the failing step (best-effort).
    errors = _error_events(loop._errors)
    assert len(errors) >= 1, "a raising diagnostic must still leave a step-named ERROR event"
    steps = {e.detail.get("step") for e in errors}
    assert "disarm" in steps, f"the failing step must be identified: {steps!r}"

    # No diagnostic exception leaked into the recorded events.
    for e in errors:
        assert "diagnostic clock failure (EH3)" not in (e.reason or "")


# ---------------------------------------------------------------------------
# 8. EH4 — the pipeline failure-reporting path must be *fully* best-effort
#    too (F1), plus the two regression locks the EH3 review flagged as
#    missing: a cleanup *clock* failure (F2) and an *isolated* __repr__
#    failure (F3).
#
#    Central invariant: an exception-reporting failure must never become a
#    new operational failure, prevent recovery, interrupt safe shutdown,
#    or replace the original failure.
# ---------------------------------------------------------------------------


class _EH4ClockDisarm(Exception):
    """Ordinary exception: only the *clock* is broken, not the exception."""

    def __str__(self) -> str:
        return "disarm broke (EH4 clock)"


def test_cleanup_clock_failure_is_best_effort_and_primary_propagates(make_loop) -> None:
    """The clock failing *during cleanup diagnostic generation* must not
    mask the primary exception or abort the remaining cleanup steps.

    The first ``now()`` call (the tick's own timestamp) succeeds so the run
    reaches the intended cleanup path; every subsequent call — the
    diagnostic's timestamp — raises. The failing step's evidence must still
    be retained (step name, best-effort), the ``capture.stop`` step after it
    must still run, and the PRIMARY exception must remain the one that
    propagates (the clock exception must not replace it)."""
    hotkey = FakeHotkeyListener(on_trigger=lambda: None)

    def failing_disarm() -> None:
        raise _EH4ClockDisarm()

    hotkey.disarm = failing_disarm  # type: ignore[method-assign]
    capture = _FakeCapture()  # capture.stop must still run

    calls = {"n": 0}

    # Call budget: 1 = run()'s _reset_report()/_start_wall, 2 = the tick's
    # own timestamp (tick: now = self._now()). Both must SUCCEED so the run
    # reaches the intended cleanup path. Call 3 onward = the cleanup
    # diagnostic's timestamp — and only there does the clock fail.
    def flaky_clock() -> float:
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("diagnostic clock failure (EH4)")
        return 0.0

    loop = _primary_failure_loop(make_loop, capture=capture, hotkey=hotkey)
    loop._now = flaky_clock  # the tick's timestamp succeeds; the diagnostic's fails

    with pytest.raises(RuntimeError, match=re.escape("simulated decision failure (EH2 primary)")):
        loop.run()

    # The clock failure did not prevent the following step.
    assert capture.stopped is True, "capture.stop must run after a clock-failing diagnostic"
    assert capture.stop_call_count == 1

    # Fallback evidence for the failing step is retained (best-effort).
    errors = _error_events(loop._errors)
    assert len(errors) >= 1, "a clock-failing diagnostic must still leave a step-named ERROR event"
    assert "disarm" in {e.detail.get("step") for e in errors}

    # No diagnostic clock exception leaked into any recorded event.
    for e in errors:
        assert "diagnostic clock failure (EH4)" not in (e.reason or "")
        assert "diagnostic clock failure (EH4)" not in str(e.detail)


class _EH4ReprOnlyDisarm(Exception):
    """``__str__`` succeeds; ``__repr__`` raises — isolates the ``repr()``
    seam inside the diagnostic from the ``str`` seam (the combined EH3 test
    cannot prove the repr line is protected, because str() is invoked
    first)."""

    def __str__(self) -> str:
        return "disarm broke (EH4 repr-only)"

    def __repr__(self) -> str:
        raise RuntimeError("repr diagnostic failure (EH4)")


def test_cleanup_step_with_raising_repr_only_is_isolated(make_loop) -> None:
    """A failing cleanup step whose exception type has a healthy ``__str__``
    but a raising ``__repr__``: the diagnostic must actually *reach* the
    ``repr()`` operation (inside ``_exception_detail``) without it
    escaping — the ``capture.stop`` step after it must still run, the
    PRIMARY exception must remain the one that propagates, and the retained
    evidence must identify the failing step while leaking no diagnostic
    exception."""
    hotkey = FakeHotkeyListener(on_trigger=lambda: None)

    def failing_disarm() -> None:
        raise _EH4ReprOnlyDisarm()

    hotkey.disarm = failing_disarm  # type: ignore[method-assign]
    capture = _FakeCapture()  # capture.stop must still run after the failing step

    loop = _primary_failure_loop(make_loop, capture=capture, hotkey=hotkey)

    with pytest.raises(RuntimeError, match=re.escape("simulated decision failure (EH2 primary)")):
        loop.run()

    # The step AFTER the failing one still ran (the repr failure did not escape).
    assert capture.stopped is True, "capture.stop must run after a repr-failing diagnostic"
    assert capture.stop_call_count == 1

    # Fallback/primary evidence retains the failing step name.
    errors = _error_events(loop._errors)
    assert len(errors) >= 1
    assert "disarm" in {e.detail.get("step") for e in errors}

    # No diagnostic exception leaked into the recorded events.
    for e in errors:
        assert "repr diagnostic failure (EH4)" not in (e.reason or "")
        assert "repr diagnostic failure (EH4)" not in str(e.detail)


class _EH4RaisingReprPerception(Exception):
    """``__str__`` succeeds (a recognizable message); ``__repr__`` raises a
    secondary exception — the pipeline-diagnostic failure seam (F1/F4)."""

    def __str__(self) -> str:
        return "perception broke (EH4 pipeline)"

    def __repr__(self) -> str:
        raise RuntimeError("repr diagnostic failure (EH4)")


class _EH4RaisingStrPerception(Exception):
    """``__str__`` raises (``__repr__`` left working) — isolates the
    pipeline's ``str(exc)`` diagnostic failure seam, which no other EH4
    pipeline test exercises: there ``__repr__`` (or the cause/context
    ``repr``) is what fails."""

    def __str__(self) -> str:
        raise RuntimeError("str diagnostic failure (EH4)")

    def __repr__(self) -> str:
        return "<EH4RaisingStrPerception: str broken>"


def test_pipeline_repr_failure_routes_to_recovery_without_leaking(make_loop) -> None:
    """A perception failure whose exception type raises in ``__repr__``:
    ``tick()`` must still *return* the bounded RECOVERY result (never leak
    the diagnostic exception), the machine must be in RECOVERY, and the
    RECOVERY event must be retained with the stage, attempt, and the
    original exception type — explicitly flagging that detailed diagnostics
    were unavailable, without fabricating message/cause/context content.
    """
    events: list[LoopEvent] = []
    loop = make_loop(
        perception=_ScriptedPerception(_EH4RaisingReprPerception()),
        on_event=events.append,
        recovery_attempts=2,  # one attempt: stay in RECOVERY, not exhausted
    )

    result = loop.tick()  # must RETURN, not raise the diagnostic exception

    # Bounded RECOVERY routing preserved.
    assert result.state is AgentState.RECOVERY
    assert loop._machine.state is AgentState.RECOVERY

    recovery = _recovery_events(events)
    assert len(recovery) == 1
    ev = recovery[0]

    # Stage/attempt preserved; the ORIGINAL exception type is identifiable.
    assert ev.detail.get("stage") == "perception"
    assert ev.detail.get("attempt") == 1
    assert ev.detail.get("exc_type") == "_EH4RaisingReprPerception"

    # The fallback explicitly identifies that detailed diagnostics were
    # unavailable (spec: it must say so, not just be absent).
    assert "unavailable" in (ev.reason or "").lower()
    assert "perception" in (ev.reason or "")
    assert "attempt 1" in (ev.reason or "")

    # No secondary diagnostic exception appears as the reported failure.
    assert "repr diagnostic failure (EH4)" not in (ev.reason or "")
    assert "repr diagnostic failure (EH4)" not in str(ev.detail)
    assert "repr diagnostic failure (EH4)" not in (result.error or "")

    # No fabricated cause/context information.
    assert "exc_cause" not in ev.detail
    assert "exc_context" not in ev.detail


class _EH4BrokenReprCause(Exception):
    """Explicit ``__cause__`` whose ``__repr__`` raises."""

    def __repr__(self) -> str:
        raise RuntimeError("cause repr failure (EH4)")


def test_pipeline_broken_cause_repr_does_not_leak(make_loop) -> None:
    """An explicit ``__cause__`` whose ``__repr__`` raises: the cause
    formatting inside ``_exception_detail`` is an independent operation and
    must be contained; routing, evidence, and no-leak guarantees as above.
    """
    exc = RuntimeError("perception broke (EH4 cause)")
    exc.__cause__ = _EH4BrokenReprCause("cause payload")
    events: list[LoopEvent] = []
    loop = make_loop(
        perception=_ScriptedPerception(exc),
        on_event=events.append,
        recovery_attempts=2,
    )

    result = loop.tick()  # must return, not leak the cause-repr failure

    assert result.state is AgentState.RECOVERY
    assert loop._machine.state is AgentState.RECOVERY

    recovery = _recovery_events(events)
    assert len(recovery) == 1
    ev = recovery[0]

    assert ev.detail.get("stage") == "perception"
    assert ev.detail.get("attempt") == 1
    assert ev.detail.get("exc_type") == "RuntimeError"
    assert "unavailable" in (ev.reason or "").lower()

    # No leak, and no fabricated cause field.
    assert "cause repr failure (EH4)" not in (ev.reason or "")
    assert "cause repr failure (EH4)" not in str(ev.detail)
    assert "cause repr failure (EH4)" not in (result.error or "")
    assert "exc_cause" not in ev.detail


class _EH4BrokenReprContext(Exception):
    """Implicit ``__context__`` whose ``__repr__`` raises."""

    def __repr__(self) -> str:
        raise RuntimeError("context repr failure (EH4)")


def test_pipeline_broken_context_repr_does_not_leak(make_loop) -> None:
    """An implicit ``__context__`` whose ``__repr__`` raises: the context
    field is formatted independently of the cause field and must be
    contained; routing, evidence, and no-leak guarantees as above."""
    exc = RuntimeError("perception broke (EH4 context)")
    exc.__context__ = _EH4BrokenReprContext("context payload")
    events: list[LoopEvent] = []
    loop = make_loop(
        perception=_ScriptedPerception(exc),
        on_event=events.append,
        recovery_attempts=2,
    )

    result = loop.tick()  # must return, not leak the context-repr failure

    assert result.state is AgentState.RECOVERY
    assert loop._machine.state is AgentState.RECOVERY

    recovery = _recovery_events(events)
    assert len(recovery) == 1
    ev = recovery[0]

    assert ev.detail.get("stage") == "perception"
    assert ev.detail.get("attempt") == 1
    assert ev.detail.get("exc_type") == "RuntimeError"
    assert "unavailable" in (ev.reason or "").lower()

    # No leak, and no fabricated context field.
    assert "context repr failure (EH4)" not in (ev.reason or "")
    assert "context repr failure (EH4)" not in str(ev.detail)
    assert "context repr failure (EH4)" not in (result.error or "")
    assert "exc_context" not in ev.detail


def test_pipeline_diag_failure_still_exhausts_bounded_recovery(make_loop) -> None:
    """The bounded-recovery contract must hold even when diagnostics fail:
    with a single allowed attempt, a diagnostic-failing pipeline failure
    must still exhaust recovery and land the machine in PAUSED — with the
    RECOVERY event retained (not lost to the diagnostic failure)."""
    events: list[LoopEvent] = []
    loop = make_loop(
        perception=_ScriptedPerception(_EH4RaisingReprPerception()),
        on_event=events.append,
        recovery_attempts=1,  # one attempt: exhausted -> PAUSED
    )

    result = loop.tick()  # must return, not leak the diagnostic exception

    assert result.state is AgentState.PAUSED
    assert loop._machine.state is AgentState.PAUSED

    recovery = _recovery_events(events)
    assert len(recovery) == 1
    assert recovery[0].detail.get("stage") == "perception"
    assert recovery[0].detail.get("attempt") == 1


def test_pipeline_str_failure_routes_to_recovery_without_leaking(make_loop) -> None:
    """A perception failure whose exception type raises in ``__str__`` (while
    ``__repr__`` works): isolates the pipeline's ``str(exc)`` diagnostic seam,
    which no other EH4 pipeline test covers. ``tick()`` must still *return* the
    bounded RECOVERY result (never leak the diagnostic exception), the machine
    must be in RECOVERY (attempts not exhausted), and the RECOVERY event must be
    retained with the stage, attempt, and the original exception type —
    explicitly flagging that detailed diagnostics were unavailable, and the
    diagnostic exception must not appear as the reported failure."""
    events: list[LoopEvent] = []
    loop = make_loop(
        perception=_ScriptedPerception(_EH4RaisingStrPerception()),
        on_event=events.append,
        recovery_attempts=2,  # one attempt: stay in RECOVERY, not exhausted
    )

    result = loop.tick()  # must RETURN, not raise the diagnostic exception

    # Bounded RECOVERY routing preserved.
    assert result.state is AgentState.RECOVERY
    assert loop._machine.state is AgentState.RECOVERY

    # Exactly the expected RECOVERY event is retained.
    recovery = _recovery_events(events)
    assert len(recovery) == 1
    ev = recovery[0]

    # Stage/attempt preserved; the ORIGINAL exception type is identifiable.
    assert ev.detail.get("stage") == "perception"
    assert ev.detail.get("attempt") == 1
    assert ev.detail.get("exc_type") == "_EH4RaisingStrPerception"

    # The fallback explicitly identifies that detailed diagnostics were
    # unavailable (spec: it must say so, not just be absent).
    assert "unavailable" in (ev.reason or "").lower()
    assert "perception" in (ev.reason or "")
    assert "attempt 1" in (ev.reason or "")

    # The diagnostic exception does not appear in the event or result error.
    assert "str diagnostic failure (EH4)" not in (ev.reason or "")
    assert "str diagnostic failure (EH4)" not in str(ev.detail)
    assert "str diagnostic failure (EH4)" not in (result.error or "")

    # No fabricated cause/context information.
    assert "exc_cause" not in ev.detail
    assert "exc_context" not in ev.detail


def test_pipeline_event_sink_failure_does_not_defeat_recovery(make_loop) -> None:
    """The implementation guards ``_emit(ev)`` so a failing event sink cannot
    interrupt recovery. A perception failure with a *normal* exception (the
    diagnostic path succeeds) plus an ``on_event`` callback that raises when
    the RECOVERY event is delivered must not leak out of ``tick()``; with a
    single allowed recovery attempt the attempt is exhausted and the machine
    reaches PAUSED, while the RECOVERY event remains retained in
    ``loop._errors`` — the sink failure neither loses the event nor replaces
    the original pipeline failure."""
    events: list[LoopEvent] = []

    def _sink_raises(event: LoopEvent) -> None:
        if event.event is EventType.RECOVERY:
            raise RuntimeError("sink diagnostic failure (EH4)")
        events.append(event)

    loop = make_loop(
        perception=_ScriptedPerception(
            RuntimeError("perception pipeline failure (EH4 sink test)")
        ),
        on_event=_sink_raises,
        recovery_attempts=1,  # one attempt: exhausted -> PAUSED
    )

    result = loop.tick()  # must return; the sink exception must not escape

    assert result.state is AgentState.PAUSED
    assert loop._machine.state is AgentState.PAUSED

    # The RECOVERY event remains retained despite the callback failure.
    retained = [e for e in loop._errors if e.event is EventType.RECOVERY]
    assert len(retained) == 1
    ev = retained[0]

    # It still carries the expected stage/attempt/type information.
    assert ev.detail.get("stage") == "perception"
    assert ev.detail.get("attempt") == 1
    assert ev.detail.get("exc_type") == "RuntimeError"

    # It IS the original pipeline failure (the sink did not replace it).
    assert "perception pipeline failure (EH4 sink test)" in (ev.reason or "")
    assert "perception pipeline failure (EH4 sink test)" in (result.error or "")

    # The sink exception does not appear as diagnostic content.
    assert "sink diagnostic failure (EH4)" not in (ev.reason or "")
    assert "sink diagnostic failure (EH4)" not in str(ev.detail)
    assert "sink diagnostic failure (EH4)" not in (result.error or "")

