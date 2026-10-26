"""Phase 6 assisted-mode approval gate — contract against the *real* API.

Every name imported below exists in the committed tree (verified against the
source, not assumed):

* ``Action`` / ``ActionKind`` from ``ai_game_agent.actions``. (The draft
  imported a non-existent ``ActionType`` — the kind enum is ``ActionKind``.)
* ``ApprovalDecision`` / ``AutoApprover`` / ``TerminalApprover`` /
  ``Approver`` from ``ai_game_agent.actions`` (the approver protocol has no
  ``ScriptedApprover`` and no free ``parse_approval_reply`` function — the
  fail-safe parsing lives *inside* ``TerminalApprover`` and is tested through
  the approver with a monkeypatched ``sys.stdin``).
* ``Executor`` / ``SafetyGuard`` — the executor has **no** ``approver``
  parameter and no ``approver`` attribute; the gate is loop-owned.
* ``AgentLoop`` — the approval gate (``_gate_ai_actions``) is a **loop**
  seam, consulted only for AI-proposed actions in ``ASSISTED`` mode.
  Rejection → forced ``machine.enter(PAUSED)`` + an ``APPROVAL`` audit
  event; the batch is not dispatched.
* ``AIDecision`` from ``ai_game_agent.ai.schema`` (the AI seam's typed
  decision; the draft's free ``ai_decision(...)`` helper does not exist).
* ``GameStateMachine`` / ``AgentState`` / ``EventType`` from
  ``ai_game_agent.state``.
* ``DeterministicPolicy`` from ``ai_game_agent.games.wow`` (the draft's
  ``WowPolicy`` is a non-existent alias).
* ``MockBackend`` from ``ai_game_agent.actions.mock`` (the draft imported
  it from ``conftest``, where it does not live).
* ``Observation`` / ``BBox`` / ``UiZoneHit`` from
  ``ai_game_agent.perception`` (the draft's ``MockPerception`` /
  ``MockClock`` do not exist anywhere in the tree).

The loop is built exactly as ``tests/test_agent_loop.py`` does — real
executor, real safety guard, real state machine, real policy, duck-typed
capture/perception/AI fakes — so these tests pin the *implemented* assisted
contract rather than a planned one.
"""

from __future__ import annotations

import datetime
import io
import threading

import pytest

from ai_game_agent.actions import (
    Action,
    ActionKind,
    ApprovalDecision,
    AutoApprover,
    Executor,
    ModeViolation,
    OperationMode,
    SafetyGuard,
    TerminalApprover,
)
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.agent import AgentLoop
from ai_game_agent.ai.schema import AIDecision
from ai_game_agent.games.wow import ZONE_TARGET_HOSTILE, DeterministicPolicy
from ai_game_agent.perception import BBox, Observation, UiZoneHit
from ai_game_agent.state import AgentState, EventType, GameStateMachine

TS = datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC)
_BB = BBox(x=10, y=10, width=20, height=10)


# ---------------------------------------------------------------------------
# Fakes — duck-typed, mirroring tests/test_agent_loop.py (no conftest deps)
# ---------------------------------------------------------------------------


def _hostile_observation() -> Observation:
    """An observation carrying the hostile-target zone (opaque to decide())."""
    return Observation(
        frame_width=128,
        frame_height=72,
        captured_at=TS,
        source="test",
        templates=(),
        ui_zones=(
            UiZoneHit(zone=ZONE_TARGET_HOSTILE, check="presence", bbox=_BB, value=1.0),
        ),
        text_regions=(),
        objects=(),
        detector_errors=(),
    )


class _FakeCapture:
    def __init__(self) -> None:
        self.grab_count = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def grab(self):
        self.grab_count += 1
        return object()

    def wait_next_frame(self) -> float:
        return 0.0


class _FakePerception:
    def __init__(self, observation: Observation | None = None) -> None:
        self._obs = observation if observation is not None else _hostile_observation()

    def observe(self, frame):
        return self._obs


class _StubDecisionPolicy:
    """Machine.decide() signal source: hostile target => COMBAT."""

    def __init__(self, signals: dict[str, object] | None = None) -> None:
        self._signals = dict(signals or {})

    def transition_signals(self, observation):
        return dict(self._signals)


class _FakeAI:
    """Duck-typed AIProvider: decide() returns a fixed decision."""

    def __init__(
        self, decision: AIDecision | None = None, error: Exception | None = None
    ) -> None:
        self._decision = decision
        self._error = error
        self.decide_calls = 0
        self.last_prompt = None

    def decide(self, prompt, **kwargs):
        self.decide_calls += 1
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        return self._decision


class _ScriptedApprover:
    """Duck-typed Approver: pops decisions from a tuple, then rejects.

    (The tree ships no ``ScriptedApprover``; this is the local test seam,
    matching the repo's convention of per-module fake approvers.)
    """

    name = "scripted"

    def __init__(self, decisions: tuple[ApprovalDecision, ...]) -> None:
        self._decisions = list(decisions)
        self.calls = 0

    def request_approval(self, action, **kwargs) -> ApprovalDecision:
        self.calls += 1
        if self._decisions:
            return self._decisions.pop(0)
        return ApprovalDecision.rejected("scripted-exhausted")


def _ai_decision() -> AIDecision:
    return AIDecision(
        action=Action(kind=ActionKind.KEY_PRESS, key="e", source="ai"),
        confidence=0.9,
    )


def _loop(
    *,
    mode: OperationMode,
    approver=None,
    backend: MockBackend | None = None,
    ai: _FakeAI | None = None,
    machine: GameStateMachine | None = None,
    max_ticks: int | None = None,
    on_event=None,
) -> AgentLoop:
    backend = backend or MockBackend()
    executor = Executor(backend=backend, safety=SafetyGuard(mode=mode))
    machine = machine or GameStateMachine(
        policy=_StubDecisionPolicy({"target_hostile": True}),
        initial=AgentState.AI_ANALYSIS,
        recovery_attempts=3,
        state_timeout_s=5.0,
    )
    # Test convention (mirrors tests/test_agent_loop.py): the loop does not
    # own the input backend's lifecycle, so the test opens it up front.
    if isinstance(getattr(backend, "opened", None), bool) and not backend.opened:
        backend.open()
    return AgentLoop(
        capture=_FakeCapture(),
        perception=_FakePerception(),
        executor=executor,
        machine=machine,
        policy=DeterministicPolicy(),
        mode=mode,
        approver=approver,
        ai=ai,
        max_ticks=max_ticks,
        now_fn=lambda: 0.0,
        on_event=on_event,
    )


# ---------------------------------------------------------------------------
# 1) Gate fires only for AI-proposed actions in ASSISTED mode
# ---------------------------------------------------------------------------


def test_assisted_approval_allows_ai_action():
    """ASSISTED + approving approver → the AI action dispatches (ok)."""
    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
    approver = AutoApprover()
    events: list[object] = []
    loop = _loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=ai,
        max_ticks=2,
        on_event=events.append,
    )
    report = loop.run()
    # The AI is consulted at most once per AI_ANALYSIS episode, so across
    # two ticks exactly one action is dispatched (the episode persists).
    assert report.actions_ok == 1, f"expected 1 ok, got {report.actions_ok}: {report.errors}"
    assert report.actions_attempted == 1
    assert ai.decide_calls >= 1
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals, "expected an APPROVAL audit event"
    assert approvals[0].detail["approved"] is True
    assert approvals[0].detail["kind"] == "key_press"


def test_assisted_rejection_halts_episode_and_blocks_dispatch():
    """ASSISTED + rejecting approver → machine PAUSED, nothing dispatched."""
    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
    approver = _ScriptedApprover(
        (ApprovalDecision.rejected("human said no"),)
    )
    events: list[object] = []
    loop = _loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=ai,
        max_ticks=3,
        on_event=events.append,
    )
    report = loop.run()
    assert loop._machine.state is AgentState.PAUSED
    assert report.final_state is AgentState.PAUSED
    assert report.actions_ok == 0, f"rejected action still ran: {report.actions_ok}"
    assert report.actions_attempted == 0, "rejected action was dispatched"
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals, "expected an APPROVAL audit event on rejection"
    assert approvals[0].detail["approved"] is False
    assert "rejected" in approvals[0].reason


def test_raising_approver_is_classified_as_rejection():
    """An approver that raises must NOT crash the loop; it must reject."""

    class _RaisingApprover:
        name = "raising"

        def request_approval(self, action, **kwargs):
            raise RuntimeError("approver blew up")

    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
    events: list[object] = []
    loop = _loop(
        mode=OperationMode.ASSISTED,
        approver=_RaisingApprover(),
        backend=backend,
        ai=ai,
        max_ticks=3,
        on_event=events.append,
    )
    report = loop.run()  # must not raise
    assert loop._machine.state is AgentState.PAUSED
    assert report.actions_ok == 0
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals and approvals[0].detail["approved"] is False


def test_gate_does_not_fire_for_deterministic_actions_in_assisted():
    """ASSISTED without an AI decision must NOT consult the approver.

    The gate is AI-proposal-scoped: deterministic COMBAT actions flow
    through the executor exactly as in AUTONOMOUS, and a rejecting
    approver has no effect on them.
    """
    backend = MockBackend()
    approver = _ScriptedApprover((ApprovalDecision.rejected("no"),))
    events: list[object] = []
    machine = GameStateMachine(
        policy=_StubDecisionPolicy({"target_hostile": True}),
        initial=AgentState.COMBAT,
        recovery_attempts=3,
        state_timeout_s=5.0,
    )
    loop = _loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=None,  # no AI → policy.actions() drives the batch in COMBAT
        machine=machine,
        max_ticks=2,
        on_event=events.append,
    )
    report = loop.run()
    # COMBAT dispatches a 3-action batch (q, delay, w) per tick -> 2 ticks
    # dispatch all 6; the approver must never have been consulted.
    assert report.actions_ok == 6, f"deterministic actions blocked: {report.actions_ok}"
    assert approver.calls == 0, "approver consulted for a non-AI action"
    assert not [e for e in events if e.event is EventType.APPROVAL]


def test_gate_is_bypassed_in_autonomous_mode_even_with_approver():
    """The gate is a no-op outside ASSISTED — the other modes are unchanged."""

    class _SpyApprover:
        name = "spy"
        calls = 0

        def request_approval(self, action, **kwargs):
            _SpyApprover.calls += 1
            return ApprovalDecision.approved("spy-approved")

    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
    events: list[object] = []
    loop = _loop(
        mode=OperationMode.AUTONOMOUS,
        approver=_SpyApprover(),
        backend=backend,
        ai=ai,
        max_ticks=2,
        on_event=events.append,
    )
    report = loop.run()
    assert report.actions_ok == 1, f"AUTONOMOUS path changed: {report.actions_ok}"
    assert report.actions_attempted == 1
    assert _SpyApprover.calls == 0, "approver consulted in AUTONOMOUS mode — gate leaked"
    assert not [e for e in events if e.event is EventType.APPROVAL]


def test_assisted_without_approver_dispatches_ai_action():
    """ASSISTED + no approver → the gate is a no-op; the AI action runs.

    Absence of an approver is "no gate to apply" (the loop returns True
    when ``self._approver is None``); the input safety guarantee remains
    the SafetyGuard at the executor, not a default rejection here.
    """
    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
    events: list[object] = []
    loop = _loop(
        mode=OperationMode.ASSISTED,
        approver=None,
        backend=backend,
        ai=ai,
        max_ticks=2,
        on_event=events.append,
    )
    report = loop.run()
    assert report.actions_ok == 1
    assert not [e for e in events if e.event is EventType.APPROVAL]


# ---------------------------------------------------------------------------
# 2) The executor is unmodified: no approver seam on it
# ---------------------------------------------------------------------------


def test_executor_has_no_approver_seam():
    """The approval gate is loop-owned; the executor exposes no approver.

    Guards against a regression that re-locates the gate into
    ``Executor.execute`` (which would bypass the loop's forced PAUSED
    safe-halt and the ``APPROVAL`` audit event).
    """
    backend = MockBackend()
    backend.open()  # test convention: the executor does not own lifecycle
    ex = Executor(backend=backend, safety=SafetyGuard(mode=OperationMode.ASSISTED))
    assert not hasattr(ex, "approver"), (
        "Executor gained an approver seam — the gate must stay loop-owned"
    )
    # The executor runs an action with no approval concept at all.
    result = ex.execute(Action(kind=ActionKind.KEY_PRESS, key="e", source="ai"))
    assert result.ok


# ---------------------------------------------------------------------------
# 3) TerminalApprover fail-safe parsing (through the approver; stdin
#    monkeypatched — the tree has no free parse_approval_reply function)
# ---------------------------------------------------------------------------


@pytest.fixture
def quiet_stdout(monkeypatch):
    """TerminalApprover prints its prompt to stdout; silence it."""
    monkeypatch.setattr("sys.stdout", io.StringIO())


def test_terminal_approver_affirmatives_approve(quiet_stdout, monkeypatch):
    for reply in ("y", "Y", "yes", "YES", "approve", "ok", "go", "1"):
        monkeypatch.setattr("sys.stdin", io.StringIO(reply + "\n"))
        approver = TerminalApprover(timeout_s=0.5)
        decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
        assert decision.is_approved, f"{reply!r} should approve, got {decision}"


def test_terminal_approver_negatives_reject(quiet_stdout, monkeypatch):
    for reply in ("n", "N", "no", "NO", "reject", ""):
        monkeypatch.setattr("sys.stdin", io.StringIO(reply + "\n"))
        approver = TerminalApprover(timeout_s=0.5)
        decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
        assert not decision.is_approved, f"{reply!r} should reject, got {decision}"


def test_terminal_approver_unparsable_rejects(quiet_stdout, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("maybe later\n"))
    approver = TerminalApprover(timeout_s=0.5)
    decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
    assert not decision.is_approved
    assert decision.reason == "unparsable-reply"


def test_terminal_approver_eof_rejects(quiet_stdout, monkeypatch):
    """An empty stream (EOF) is a rejection — the fail-safe default."""
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    approver = TerminalApprover(timeout_s=0.5)
    decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
    assert not decision.is_approved
    assert decision.reason == "no-input-channel"


def test_terminal_approver_none_timeout_uses_default() -> None:
    """An explicit ``timeout_s=None`` means "use the default" — not a crash.

    Regression guard: the default ASSISTED run path (``run --mode assisted``
    with neither ``--approve-all`` nor ``--approve-timeout``) constructs
    ``TerminalApprover(timeout_s=None)``. ``None`` is the module's existing
    "use the instance default" sentinel (see ``request_approval``), so the
    constructor must map it to ``DEFAULT_TIMEOUT_S`` rather than raising
    ``TypeError`` on ``None < 0``.
    """
    approver = TerminalApprover(timeout_s=None)  # type: ignore[arg-type]
    assert approver._timeout_s == TerminalApprover.DEFAULT_TIMEOUT_S


def test_terminal_approver_keyboard_interrupt_rejects(quiet_stdout, monkeypatch):
    """A KeyboardInterrupt during the prompt is a rejection, not a crash."""

    class _KbdIntStdin:
        def readline(self):
            raise KeyboardInterrupt()

    monkeypatch.setattr("sys.stdin", _KbdIntStdin())
    approver = TerminalApprover(timeout_s=0.5)
    decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
    assert not decision.is_approved


def test_terminal_approver_timeout_rejects(quiet_stdout, monkeypatch):
    """A timeout on the prompt is a rejection (the configured bound).

    Uses a real-fd pipe (the realistic terminal/pipe path) and a writer
    thread that delivers an affirmative reply only *after* the deadline:
    the approver must reject with ``timeout`` and must not honour the
    late reply.
    """
    import os
    import time

    class _PipeStdin:
        def __init__(self, reply: bytes, delay: float) -> None:
            self.r, self.w = os.pipe()

            def _write() -> None:
                time.sleep(delay)
                try:
                    os.write(self.w, reply)
                except OSError:
                    pass
                finally:
                    os.close(self.w)

            threading.Thread(target=_write, daemon=True).start()

        def fileno(self):
            return self.r

        def readline(self):
            buf = b""
            while True:
                c = os.read(self.r, 1)
                if not c:
                    break
                buf += c
                if c == b"\n":
                    break
            return buf.decode()

    monkeypatch.setattr("sys.stdin", _PipeStdin(b"y\n", delay=0.3))
    approver = TerminalApprover(timeout_s=0.1)
    decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
    assert not decision.is_approved
    assert decision.reason == "timeout"


def test_terminal_approver_timeout_when_select_rejects_fd(quiet_stdout, monkeypatch):
    """A timeout must still reject when ``select`` refuses the fd.

    On Windows ``select.select`` only accepts sockets and raises
    ``OSError [WinError 10038]`` for pipes / console fds — so the approver
    must not crash on the real Windows production path (an ASSISTED-mode
    prompt on a console would raise here), and must fall back to a bounded
    read that still times out (fail-safe rejection, never an approval).

    The contract (timeout → rejection, no crash) is OS-independent, so we
    pin the Windows behaviour from the Linux host by making ``select.select``
    raise the same ``OSError``. This is the regression guard for the
    cross-platform read path.
    """
    import os
    import select
    import time

    def _select_rejects(*args, **kwargs):
        raise OSError(
            10038, "An operation was attempted on something that is not a socket"
        )

    class _PipeStdin:
        def __init__(self, reply: bytes, delay: float) -> None:
            self.r, self.w = os.pipe()

            def _write() -> None:
                time.sleep(delay)
                try:
                    os.write(self.w, reply)
                except OSError:
                    pass
                finally:
                    os.close(self.w)

            threading.Thread(target=_write, daemon=True).start()

        def fileno(self):
            return self.r

        def readline(self):  # pragma: no cover - the bounded path reads the raw fd
            buf = b""
            while True:
                c = os.read(self.r, 1)
                if not c:
                    break
                buf += c
                if c == b"\n":
                    break
            return buf.decode()

    monkeypatch.setattr(select, "select", _select_rejects)
    monkeypatch.setattr("sys.stdin", _PipeStdin(b"y\n", delay=0.3))
    approver = TerminalApprover(timeout_s=0.1)
    decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
    assert not decision.is_approved
    assert decision.reason == "timeout"


def _action() -> Action:
    return Action(kind=ActionKind.KEY_PRESS, key="e", source="ai")


# ---------------------------------------------------------------------------
# 4) F12 emergency stop still wins while an approver would block
# ---------------------------------------------------------------------------


def test_f12_emergency_stop_wins_while_approver_would_block():
    """A stop fired while a prompt is pending must not be delayed.

    The approver blocks on a gate the test never signals (so it *would*
    hang without the stop); the executor's emergency stop is then raised
    and the run must exit with the stop reason, with the AI action never
    dispatched.
    """
    backend = MockBackend()
    gate = threading.Event()

    class _BlockingApprover:
        name = "blocking"

        def __init__(self) -> None:
            self.blocked = threading.Event()

        def request_approval(self, action, **kwargs):
            self.blocked.set()
            gate.wait(timeout=1.5)
            return ApprovalDecision.approved("too late")

    approver = _BlockingApprover()
    ai = _FakeAI(decision=_ai_decision())
    loop = _loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=ai,
        max_ticks=10,
    )
    executor = loop._executor

    result_box: dict[str, object] = {}

    def _run() -> None:
        try:
            result_box["report"] = loop.run()
        except BaseException as exc:  # pragma: no cover - surface test failures
            result_box["exc"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    assert approver.blocked.wait(timeout=3.0), (
        "approver never entered (loop not in the ASSISTED AI path)"
    )
    # Fire the emergency stop while the prompt is still pending.
    executor.trigger_emergency_stop("f12")
    gate.set()  # unblock the approver so the thread can exit promptly
    t.join(timeout=6.0)
    assert "exc" not in result_box, f"run raised: {result_box.get('exc')!r}"
    report = result_box["report"]
    assert executor.stopped
    assert report.stopped_reason == "emergency_stop", f"got {report.stopped_reason!r}"
    # The AI action must NOT have been dispatched (stop wins over approval).
    assert ("press", "e") not in [ev[:2] for ev in backend.events]


# ---------------------------------------------------------------------------
# 5) OBSERVE_ONLY hard gate: no action, no approval, executor untouched
# ---------------------------------------------------------------------------


def test_observe_only_dispatches_nothing_even_with_ai_and_approver():
    """OBSERVE_ONLY is a hard gate: the loop never dispatches, so neither
    the executor nor the approver ever see the AI decision."""
    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
    events: list[object] = []
    loop = _loop(
        mode=OperationMode.OBSERVE_ONLY,
        approver=AutoApprover(),
        backend=backend,
        ai=ai,
        max_ticks=2,
        on_event=events.append,
    )
    report = loop.run()
    assert report.actions_ok == 0
    assert report.actions_attempted == 0
    # No APPROVAL event: the gate is consulted only when the loop is about
    # to dispatch, and OBSERVE_ONLY never dispatches.
    assert not [e for e in events if e.event is EventType.APPROVAL]
    # The AI may still be consulted (seeing is fine), but its decision must
    # never reach the executor.
    assert ai.decide_calls >= 1
    assert backend.events == [], f"OBSERVE_ONLY leaked input: {backend.events}"


def test_observe_only_safety_guard_blocks_direct_execute():
    """SafetyGuard is the last line: a direct execute() in OBSERVE_ONLY
    raises ModeViolation before touching the backend."""
    backend = MockBackend()
    ex = Executor(backend=backend, safety=SafetyGuard(mode=OperationMode.OBSERVE_ONLY))
    with pytest.raises(ModeViolation):
        ex.execute(Action(kind=ActionKind.KEY_PRESS, key="e", source="user"))
    assert backend.events == []


# ---------------------------------------------------------------------------
# 6) APPROVAL event detail shape (audit-trail contract)
# ---------------------------------------------------------------------------


def test_approval_event_detail_contract():
    """Every APPROVAL event carries the audit fields the spec names:
    kind, approved, reason, channel (and confidence when known)."""
    backend = MockBackend()

    class _ChannelApprover:
        name = "channel-x"

        def request_approval(self, action, **kwargs):
            return ApprovalDecision.approved("yes")

    ai = _FakeAI(decision=_ai_decision())
    events: list[object] = []
    loop = _loop(
        mode=OperationMode.ASSISTED,
        approver=_ChannelApprover(),
        backend=backend,
        ai=ai,
        max_ticks=2,
        on_event=events.append,
    )
    loop.run()
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals
    detail = approvals[0].detail
    for key in ("kind", "approved", "reason", "channel"):
        assert key in detail, f"APPROVAL detail missing {key!r}: {detail}"
    assert detail["kind"] == "key_press"
    assert detail["approved"] is True
    assert detail["channel"] == "channel-x"


# ---------------------------------------------------------------------------
# 7) PR #6 review findings — timeout contract, zero-timeout poll, F12 cancel,
#    full-action prompt
# ---------------------------------------------------------------------------


def test_terminal_approver_timeout_contract_none_zero_positive_negative():
    """The timeout contract is explicit: ``None`` → 30 s default, ``0`` is a
    legal (non-blocking poll) value, positive values are accepted verbatim,
    and negative values are rejected.

    Regression guard: the CLI used to reject ``0`` at the argument level
    ("must be a positive number"), contradicting the approver accepting it;
    the help text said "no timeout by default" while the default was 30 s.
    """
    assert TerminalApprover(timeout_s=None)._timeout_s == (
        TerminalApprover.DEFAULT_TIMEOUT_S
    )
    assert TerminalApprover.DEFAULT_TIMEOUT_S == 30.0
    assert TerminalApprover(timeout_s=0)._timeout_s == 0.0
    assert TerminalApprover(timeout_s=5.0)._timeout_s == 5.0
    with pytest.raises(ValueError):
        TerminalApprover(timeout_s=-1)


def _late_reply_stdin(delay: float, reply: bytes = b"y\n"):
    """A real-fd stdin whose reply arrives only after ``delay`` seconds.

    Same shape as the existing timeout tests: a pipe with a writer thread.
    """
    import os
    import time

    class _PipeStdin:
        def __init__(self) -> None:
            self.r, self.w = os.pipe()

            def _write() -> None:
                time.sleep(delay)
                try:
                    os.write(self.w, reply)
                except OSError:
                    pass
                finally:
                    os.close(self.w)

            threading.Thread(target=_write, daemon=True).start()

        def fileno(self):
            return self.r

        def readline(self):
            buf = b""
            while True:
                c = os.read(self.r, 1)
                if not c:
                    break
                buf += c
                if c == b"\n":
                    break
            return buf.decode()

    return _PipeStdin()


def test_terminal_approver_zero_timeout_is_non_blocking_poll(
    quiet_stdout, monkeypatch
):
    """``timeout=0`` must be a non-blocking poll, not an unbounded read.

    Regression guard: the old ``if select is not None and timeout > 0`` guard
    skipped the deadline entirely for ``timeout=0`` and fell through to a
    bare ``sys.stdin.readline()`` — an unbounded hang on a live terminal.
    After the fix, ``0`` polls once: an unbuffered reply (arriving after
    1 s) is NOT honoured and the request rejects with ``"timeout"``
    promptly.
    """
    import time

    monkeypatch.setattr("sys.stdin", _late_reply_stdin(delay=1.0))
    approver = TerminalApprover(timeout_s=0)
    t0 = time.monotonic()
    decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
    elapsed = time.monotonic() - t0
    assert not decision.is_approved
    assert decision.reason == "timeout"
    assert elapsed < 0.5, (
        f"timeout=0 blocked for {elapsed:.2f}s (expected <0.1s non-blocking poll)"
    )


def test_terminal_approver_zero_timeout_honours_buffered_reply(
    quiet_stdout, monkeypatch
):
    """A reply already buffered on stdin is honoured by the ``0`` poll
    (the poll only rejects when *nothing* is available)."""
    import os

    r, w = os.pipe()
    os.write(w, b"y\n")
    os.close(w)

    class _FdStdin:
        def fileno(self):
            return r

        def readline(self):
            buf = b""
            while True:
                c = os.read(r, 1)
                if not c:
                    break
                buf += c
                if c == b"\n":
                    break
            return buf.decode()

    monkeypatch.setattr("sys.stdin", _FdStdin())
    approver = TerminalApprover(timeout_s=0)
    decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
    assert decision.is_approved, f"buffered reply ignored: {decision}"
    assert decision.reason == "user-approved"


def test_terminal_approver_cancel_event_interrupts_pending_wait(
    quiet_stdout, monkeypatch
):
    """Setting the ``cancel_event`` during a pending wait must unblock the
    approver promptly with an ``"interrupted"`` rejection.

    Regression guard: without cancellation support the approver would sit
    in ``select`` for the full 5 s even after F12 fired, delaying the
    emergency-stop reaction.
    """
    import time

    monkeypatch.setattr("sys.stdin", _late_reply_stdin(delay=5.0))
    cancel = threading.Event()
    approver = TerminalApprover(timeout_s=5.0, cancel_event=cancel)

    result_box: dict[str, object] = {}

    def _call() -> None:
        result_box["decision"] = approver.request_approval(
            _action(), reasoning="r", confidence=0.5
        )

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    time.sleep(0.3)
    cancel.set()
    t.join(timeout=2.0)
    assert not t.is_alive(), "approver did not return after cancel_event was set"
    decision = result_box.get("decision")
    assert decision is not None
    assert not decision.is_approved
    assert decision.reason == "interrupted"


def test_approval_prompt_shows_modifiers(quiet_stdout, monkeypatch):
    """The prompt must surface modifiers: ``ctrl+shift+a`` is a different
    action from bare ``a``, and the approver must see the difference.

    Regression guard: the old prompt only listed six fields and omitted
    ``modifiers``, ``relative``, ``clicks`` — an approver could not tell
    which of those variants was proposed.
    """
    captured = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    action = Action(
        kind=ActionKind.KEY_PRESS, key="a", modifiers=("ctrl", "shift"), source="ai"
    )
    TerminalApprover(timeout_s=0.5).request_approval(
        action, reasoning="r", confidence=0.5
    )
    out = captured.getvalue()
    assert "ctrl" in out, f"modifiers missing from prompt: {out!r}"
    assert "shift" in out, f"modifiers missing from prompt: {out!r}"


def test_approval_prompt_shows_relative_flag_and_clicks(
    quiet_stdout, monkeypatch
):
    """``relative`` mouse moves and multi-clicks differ from the defaults
    and must be visible in the prompt."""
    captured = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    move = Action(
        kind=ActionKind.MOUSE_MOVE, x=10, y=20, relative=True, source="ai"
    )
    TerminalApprover(timeout_s=0.5).request_approval(
        move, reasoning="r", confidence=0.5
    )
    out = captured.getvalue()
    assert "relative" in out, f"relative flag missing from prompt: {out!r}"

    captured2 = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured2)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    click = Action(
        kind=ActionKind.MOUSE_CLICK, x=5, y=6, button="left", clicks=3, source="ai"
    )
    TerminalApprover(timeout_s=0.5).request_approval(
        click, reasoning="r", confidence=0.5
    )
    out2 = captured2.getvalue()
    assert "clicks" in out2, f"clicks missing from prompt: {out2!r}"
