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

import contextlib
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


def test_f12_interrupts_real_approver_pending_wait_in_loop(quiet_stdout, monkeypatch):
    """F12 must interrupt the *real* ``TerminalApprover`` pending wait at the
    loop level — not merely the executor's refusal to dispatch afterwards.

    Regression guard (finding 3): the earlier loop-level test used a fake
    blocking approver that the test released with a manual gate, so it only
    proved the executor dropped the late approval. It never proved the real
    approver's pending ``stdin`` wait is interrupted by the emergency-stop
    signal and the run exits on its own. This test wires the real approver
    with a ``cancel_event`` exactly as ``__main__`` does for F12 (the F12
    handler sets the approver's cancel event *and* triggers the executor
    stop), over a stdin that will never answer, and fires the stop while the
    wait is pending — with no manual release of any kind.
    """
    import os

    # Pipe whose write end stays open and is never written to: the prompt
    # would otherwise wait out the full (10 s) timeout.
    r, w = os.pipe()

    class _NeverAnsweringStdin:
        def fileno(self):
            return r

        def readline(self):  # pragma: no cover - the cancel path must return first
            raise AssertionError("cancel path must unblock the wait before reading stdin")

    monkeypatch.setattr("sys.stdin", _NeverAnsweringStdin())

    cancel = threading.Event()
    approver = TerminalApprover(timeout_s=10.0, cancel_event=cancel)

    entered = threading.Event()
    _orig = approver.request_approval

    def _mark(action, **kwargs):
        entered.set()
        return _orig(action, **kwargs)

    approver.request_approval = _mark

    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
    events: list[object] = []
    loop = _loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=ai,
        max_ticks=10,
        on_event=events.append,
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
    assert entered.wait(timeout=3.0), (
        "approver never entered (loop not in the ASSISTED AI path)"
    )
    # The prompt is pending inside the real approver; nothing dispatched yet.
    assert ("press", "e") not in [ev[:2] for ev in backend.events]
    # Fire the emergency stop in the production order (__main__._f12):
    # set the approver's cancel event, then trigger the executor stop.
    cancel.set()
    executor.trigger_emergency_stop("f12")
    t.join(timeout=3.0)
    assert not t.is_alive(), (
        "run did not exit after F12 — the real approver's pending wait was not interrupted"
    )
    assert "exc" not in result_box, f"run raised: {result_box.get('exc')!r}"
    report = result_box["report"]
    assert executor.stopped
    assert report.stopped_reason == "emergency_stop", f"got {report.stopped_reason!r}"
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals, "expected an APPROVAL audit event"
    assert approvals[0].detail["approved"] is False
    assert approvals[0].detail["reason"] == "interrupted"
    # The AI action must NOT have been dispatched (stop wins over approval).
    assert report.actions_attempted == 0
    assert ("press", "e") not in [ev[:2] for ev in backend.events]
    os.close(w)
    os.close(r)


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
    assert report.actions_attempted == 0
    assert report.actions_ok == 0
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals and approvals[0].detail["approved"] is False
    # The audit trail must preserve the approver's actual failure reason,
    # not the generic malformed-decision tag (P3, review of c57d5d1).
    assert approvals[0].detail["reason"] == (
        "approver-error: RuntimeError: approver blew up"
    )


# ---------------------------------------------------------------------------
# P2 (review) — the gate must not treat a *malformed* decision as approval.
#
# The old gate read ``approved = getattr(decision, "is_approved", False)`` and
# trusted the truthiness of whatever the approver returned. A broken approver
# returning an object with a *truthy* non-bool ``is_approved`` (e.g. the string
# ``"no"``) would be counted as granted and the AI action would dispatch —
# violating the central invariant that an affirmative approval decision is the
# only path to execution. Only a genuine :class:`ApprovalDecision` with an
# affirmative outcome may approve; anything else is a rejection.
# ---------------------------------------------------------------------------


class _MalformedApprover:
    """Duck-typed approver that returns a fixed, malformed payload."""

    name = "malformed"

    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.calls = 0

    def request_approval(self, action, **kwargs):
        self.calls += 1
        return self._payload


def test_malformed_decision_truthy_string_is_not_approved():
    """``is_approved="no"`` (a truthy string) must be a rejection, not approval."""
    from types import SimpleNamespace

    payload = SimpleNamespace(is_approved="no", reason="not actually approved")
    approver = _MalformedApprover(payload)
    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
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
    assert report.actions_ok == 0, (
        f"malformed decision dispatched an action: {report.actions_ok}"
    )
    assert report.actions_attempted == 0
    assert approver.calls >= 1
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals, "expected an APPROVAL audit event on the malformed decision"
    assert approvals[0].detail["approved"] is False
    assert approvals[0].detail["reason"], "a rejection must carry a reason"


def test_malformed_decision_truthy_bool_is_not_approved():
    """``is_approved=True`` on a non-``ApprovalDecision`` object must reject.

    This is the trap the old gate fell into: ``True`` is truthy, so the action
    dispatched. Only a real :class:`ApprovalDecision` may authorize execution.
    The malformed object carries no reason, so the audit trail must tag it with
    the distinct ``invalid-approval-decision`` reason.
    """
    from types import SimpleNamespace

    payload = SimpleNamespace(is_approved=True)  # no ``reason`` attribute
    approver = _MalformedApprover(payload)
    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
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
    assert report.actions_ok == 0, "truthy-bool malformed decision dispatched an action"
    assert report.actions_attempted == 0
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals
    assert approvals[0].detail["approved"] is False
    assert approvals[0].detail["reason"] == "invalid-approval-decision"


def test_approval_decision_with_invalid_outcome_is_not_approved():
    """A genuine :class:`ApprovalDecision` with an out-of-vocabulary ``outcome``
    is a rejection — approval derives from the validated outcome, not truthiness.
    """
    # ``DecisionOutcome`` is a Literal annotation; the dataclass does not enforce
    # it at runtime, so a malformed outcome can exist and must be treated as "no".
    payload = ApprovalDecision(outcome="sneaky", reason="sneaky-outcome")
    approver = _MalformedApprover(payload)
    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
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
    assert report.actions_ok == 0
    assert report.actions_attempted == 0
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals
    assert approvals[0].detail["approved"] is False
    assert approvals[0].detail["reason"] == "invalid-approval-decision"


# ---------------------------------------------------------------------------
# P2/P3 (fresh review) — the gate must validate the *reason* type too.
#
# ``QueueApprover._valid_decision`` requires a string reason, but the loop
# gate only checked the outcome vocabulary. An approver returning
# ``ApprovalDecision(outcome="approved", reason=object())`` would pass the
# gate and authorize execution with a non-JSON-serializable audit reason.
# Both outcomes with a non-string reason must be treated as malformed and
# rejected with the distinct ``invalid-approval-decision`` tag.
# ---------------------------------------------------------------------------


def test_approved_decision_with_non_string_reason_is_rejected():
    """A non-string ``reason`` must not authorize execution, even if approved."""
    payload = ApprovalDecision(outcome="approved", reason=object())
    approver = _MalformedApprover(payload)
    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
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
    assert report.actions_ok == 0, (
        "approved decision with non-string reason dispatched an action"
    )
    assert report.actions_attempted == 0
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals
    assert approvals[0].detail["approved"] is False
    assert approvals[0].detail["reason"] == "invalid-approval-decision"


def test_rejected_decision_with_non_string_reason_is_rejected():
    """A non-string ``reason`` is malformed even when the outcome is rejected."""
    payload = ApprovalDecision(outcome="rejected", reason=object())
    approver = _MalformedApprover(payload)
    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
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
    assert report.actions_ok == 0
    assert report.actions_attempted == 0
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals
    assert approvals[0].detail["approved"] is False
    assert approvals[0].detail["reason"] == "invalid-approval-decision"


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


def test_assisted_without_approver_is_fail_closed():
    """ASSISTED + AI + no approver configured → the AI action is REFUSED.

    F2 (readiness): the Phase-6 invariant is that an affirmative approval
    decision is the *only* path to AI-action execution. "No approver
    configured" is not the same as "everyone approved" — it means nobody
    can answer, so the gate must fail CLOSED: reject, enter PAUSED, emit the
    APPROVAL audit event with a meaningful reason, and dispatch nothing.
    (Previously this path returned True — fail-open — which let AI actions
    execute with no human in the loop.)
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
    assert loop._machine.state is AgentState.PAUSED, (
        "AI action executed without an approver — must pause"
    )
    assert report.final_state is AgentState.PAUSED
    assert report.actions_attempted == 0, f"AI action dispatched: {report.actions_attempted}"
    assert report.actions_ok == 0
    assert ("press", "e") not in [ev[:2] for ev in backend.events]
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals, "expected an APPROVAL audit event on the no-approver rejection"
    assert approvals[0].detail["approved"] is False
    assert approvals[0].detail["reason"] == "no-approver-configured"


def test_autonomous_without_approver_still_dispatches_ai_action():
    """AUTONOMOUS + AI + no approver → the action still dispatches.

    The fail-closed rule is ASSISTED-scoped (the Phase-6 contract): AUTONOMOUS
    mode executes every action by definition and must not be altered when
    no approver is present. Control case for the fail-closed gate.
    """
    backend = MockBackend()
    ai = _FakeAI(decision=_ai_decision())
    events: list[object] = []
    loop = _loop(
        mode=OperationMode.AUTONOMOUS,
        approver=None,
        backend=backend,
        ai=ai,
        max_ticks=2,
        on_event=events.append,
    )
    report = loop.run()
    assert report.actions_ok == 1, f"AUTONOMOUS no-approver path changed: {report.actions_ok}"
    assert report.actions_attempted == 1
    assert ("press", "e") in [ev[:2] for ev in backend.events]
    assert not [e for e in events if e.event is EventType.APPROVAL], (
        "no approver exists, so no APPROVAL audit event may be emitted"
    )


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


def test_bounded_read_does_not_accumulate_worker_threads(quiet_stdout, monkeypatch):
    """Repeated timed-out prompts on the select-unavailable fallback path
    must not accumulate blocked worker threads.

    The bounded read must release the main thread within the configured
    deadline *and* leave no per-prompt worker behind: a worker blocked in
    ``readline()`` with no reply would live (daemon, but blocked) for the
    whole process lifetime, so N timed-out prompts must not leave N live
    threads.

    Regression guard: the bounded read used to spawn one daemon worker per
    prompt and abandon it on timeout — an unbounded thread accumulation on
    exactly the Windows console path this fallback exists for.
    """
    import os
    import select

    def _select_rejects(*args, **kwargs):
        raise OSError(10038, "An operation was attempted on something that is not a socket")

    # Write end stays open and is never written to: every prompt times out.
    r, w = os.pipe()

    class _NeverAnsweringStdin:
        def fileno(self):
            return r

        def readline(self):  # pragma: no cover - a worker would block here forever
            buf = b""
            while True:
                c = os.read(r, 1)
                if not c:
                    break
                buf += c
                if c == b"\n":
                    break
            return buf.decode()

    monkeypatch.setattr(select, "select", _select_rejects)
    monkeypatch.setattr("sys.stdin", _NeverAnsweringStdin())
    approver = TerminalApprover(timeout_s=0.1)

    def _approve_read_threads() -> int:
        return sum(1 for t in threading.enumerate() if t.name == "approve-read")

    before = _approve_read_threads()
    for _ in range(5):
        decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
        assert not decision.is_approved
        assert decision.reason == "timeout"
    os.close(w)
    os.close(r)
    after = _approve_read_threads()
    assert after <= before, (
        f"bounded read accumulated worker threads: {before} -> {after} "
        "over 5 timed-out prompts (a per-prompt worker is still blocked on readline)"
    )


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


def test_approval_event_detail_includes_action_and_reasoning():
    """F3 (P2): the APPROVAL audit event must record the **proposed action**
    and its **reasoning**, not just the outcome. Without the action's
    serialized form, a reviewer cannot answer "what did the agent try to do
    when it was approved/rejected?" from the audit trail alone.
    """
    backend = MockBackend()

    class _ChannelApprover:
        name = "channel-x"

        def request_approval(self, action, **kwargs):
            return ApprovalDecision.approved("yes")

    ai = _FakeAI(
        decision=AIDecision(
            action=Action(kind=ActionKind.KEY_PRESS, key="e", source="ai"),
            reasoning="target detected at (10,10)",
            confidence=0.9,
        )
    )
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
    assert "action" in detail, f"APPROVAL detail missing 'action': {detail!r}"
    assert "reasoning" in detail, f"APPROVAL detail missing 'reasoning': {detail!r}"
    assert detail["reasoning"] == "target detected at (10,10)"
    action_dict = Action(kind=ActionKind.KEY_PRESS, key="e", source="ai").to_dict()
    assert detail["action"] == action_dict


def test_approval_event_detail_carries_audit_counters():
    """P2 (review #2): the phase-6 plan (``docs/phase-6-implementation-plan.md``
    §2) states the approval counters are *recorded on each APPROVAL event*.
    The event detail must therefore carry the cumulative
    ``approvals_requested / approvals_granted / approvals_rejected`` so the
    audit trail reflects the running totals — not just the per-event outcome.
    """
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
        max_ticks=1,
        on_event=events.append,
    )
    loop.run()
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals
    detail = approvals[0].detail
    for key in ("approvals_requested", "approvals_granted", "approvals_rejected"):
        assert key in detail, f"APPROVAL detail missing counter {key!r}: {detail}"
    assert detail["approvals_requested"] == 1
    assert detail["approvals_granted"] == 1
    assert detail["approvals_rejected"] == 0


def test_approval_counters_track_successive_decisions():
    """P2 (review #2): approve → reject → approve must increment the counters
    monotonically so an audit reader can reconstruct the running totals from
    the event stream (1/1/0, 2/1/1, 3/2/1). Drives the gate directly (rather
    than ``run()``) so the three decisions are each consulted; a real run
    halts on the first rejection, which would prevent the third from firing.
    """
    approver = _ScriptedApprover(
        (
            ApprovalDecision.approved("ok"),
            ApprovalDecision.rejected("nope"),
            ApprovalDecision.approved("ok"),
        )
    )
    ai = _FakeAI(decision=_ai_decision())
    events: list[object] = []
    loop = _loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        ai=ai,
        on_event=events.append,
    )
    actions = (_action(),)
    # Drive the gate three times: approve, reject, approve.
    assert loop._gate_ai_actions(actions, now=0.0) is True
    assert loop._gate_ai_actions(actions, now=0.0) is None  # rejection halts
    assert loop._gate_ai_actions(actions, now=0.0) is True
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert len(approvals) == 3, f"expected 3 APPROVAL events, got {len(approvals)}"
    expected = [(1, 1, 0), (2, 1, 1), (3, 2, 1)]
    for ev, (req, granted, rejected) in zip(approvals, expected, strict=True):
        assert (ev.detail["approvals_requested"],
                ev.detail["approvals_granted"],
                ev.detail["approvals_rejected"]) == (req, granted, rejected), (
            f"counter mismatch at event {ev.seq}: {ev.detail}")


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


def test_terminal_approver_timeout_rejects_non_finite(quiet_stdout):
    """F2 (P2): ``nan`` and ``inf`` must be rejected, not just ``-inf`` and
    ``-1``. Pre-fix, ``nan < 0`` and ``inf < 0`` are both ``False`` in Python
    (NaN comparisons are always False, ``inf > 0``), so only ``-inf`` and
    negative numbers were caught — a ``nan`` timeout would slip through to
    ``deadline = now + nan`` (always NaN → the deadline check never fires →
    the loop blocks forever on the ``sleep`` branch), and ``inf`` would
    become an unbounded wait (defeating the "bounded durations" safety
    contract). Both must raise ``ValueError`` at the same seam as ``-1``.
    """
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            TerminalApprover(timeout_s=bad)


def test_auto_approver_timeout_rejects_non_finite(quiet_stdout):
    """F2 (P2): the same non-finite rejection applies to
    :class:`AutoApprover`'s ``request_approval`` timeout seam (a CI/CI-like
    path that could also pass ``nan``/``inf``)."""
    action = _action()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            AutoApprover().request_approval(action, timeout_s=bad)


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


def _straddling_stdin(first: bytes, first_at: float, second: bytes, second_at: float):
    """Real-fd stdin pipe: write ``first`` at ``first_at`` s, then ``second``
    at ``second_at`` s, holding the write end open. ``readline`` works on this
    object (the old select->readline() path would demonstrably approve a
    straddling reply), so it isolates the fd reader's deadline enforcement
    from the read mechanism.
    """
    import os
    import time

    class _StraddleStdin:
        def __init__(self) -> None:
            self.r, self.w = os.pipe()

            def _write() -> None:
                time.sleep(first_at)
                try:
                    os.write(self.w, first)
                    time.sleep(max(0.0, second_at - first_at))
                    os.write(self.w, second)
                except OSError:
                    pass
                finally:
                    os.close(self.w)

            threading.Thread(target=_write, daemon=True).start()

        def fileno(self):
            return self.r

        def readline(self):  # the old path blocks here until the late \\n
            buf = b""
            while True:
                c = os.read(self.r, 1)
                if not c:
                    break
                buf += c
                if c == b"\n":
                    break
            return buf.decode()

    return _StraddleStdin()


def test_fd_straddling_newline_after_deadline_is_timeout(quiet_stdout, monkeypatch):
    """A reply whose *newline* lands after the deadline is not approved, even
    if its first byte arrived early.

    ``y`` at ~0.05 s, ``\\n`` at ~0.30 s, deadline 0.10 s. The old
    select->readline() hand-off approved it (readline() blocked until the
    late newline). The raw-fd reader must time out instead: a line is
    honoured only if its terminating newline is consumed within the window.
    """
    monkeypatch.setattr(
        "sys.stdin", _straddling_stdin(b"y", first_at=0.05, second=b"\n", second_at=0.30)
    )
    decision = TerminalApprover(timeout_s=0.1).request_approval(
        _action(), reasoning="r", confidence=0.5
    )
    assert not decision.is_approved
    assert decision.reason == "timeout"


def test_fd_lone_byte_without_newline_does_not_hang(quiet_stdout, monkeypatch):
    """A lone ``y`` with no newline and an open writer must not hang.

    The old readline() path blocked forever (or hit the guarded fake). The
    raw-fd reader must return a ``timeout`` in ~``timeout_s`` and never call
    ``readline`` -- the guarded fake raises if it is.
    """
    import os
    import time

    r, w = os.pipe()

    def _write() -> None:
        time.sleep(0.05)
        try:
            os.write(w, b"y")  # no newline; keep w open
        except OSError:
            pass

    threading.Thread(target=_write, daemon=True).start()

    class _NoReadline:
        def fileno(self):
            return r

        def readline(self):  # pragma: no cover
            raise AssertionError("readline() must not be used on the fd path")

    monkeypatch.setattr("sys.stdin", _NoReadline())

    started = time.monotonic()
    decision = TerminalApprover(timeout_s=0.3).request_approval(
        _action(), reasoning="r", confidence=0.5
    )
    elapsed = time.monotonic() - started
    os.close(r)
    os.close(w)

    assert elapsed < 2.0, f"approver hung: took {elapsed:.2f}s"
    assert not decision.is_approved
    assert decision.reason == "timeout"


def test_fd_reply_split_across_poll_slices_is_approved(quiet_stdout, monkeypatch):
    """A reply whose bytes arrive in separate poll slices is approved, not lost.

    ``y`` at ~0.05 s and ``\\n`` ~0.08 s later, deadline 0.30 s — both
    within the window. Each ``os.read`` sees only its own byte, so the
    reader must carry the partial line across poll slices. The old
    per-slice buffer discarded the ``y`` and read the lone ``\\n`` as a
    blank line, misclassifying a valid approval as ``user-rejected``.
    """
    import os

    r, w = os.pipe()

    def _write() -> None:
        time.sleep(0.05)
        try:
            os.write(w, b"y")
            time.sleep(0.08)
            os.write(w, b"\n")
        except OSError:
            pass
        finally:
            os.close(w)

    threading.Thread(target=_write, daemon=True).start()

    class _SplitStdin:
        def fileno(self):
            return r

        def readline(self):  # pragma: no cover
            raise AssertionError("readline() must not be used on the fd path")

    monkeypatch.setattr("sys.stdin", _SplitStdin())
    decision = TerminalApprover(timeout_s=0.3).request_approval(
        _action(), reasoning="r", confidence=0.5
    )
    assert decision.is_approved, f"split reply lost: {decision}"
    assert decision.reason == "user-approved"
    os.close(r)


def test_fd_eof_is_no_input_channel(quiet_stdout, monkeypatch):
    """A stdin whose writer has closed (real-fd EOF) is ``no-input-channel``,
    not ``timeout``: there is no channel left to answer, and the request
    must not wait out the deadline on a dead pipe.
    """
    import os
    import time

    r, w = os.pipe()
    os.close(w)  # immediate EOF on the read end, no data

    class _EofStdin:
        def fileno(self):
            return r

    monkeypatch.setattr("sys.stdin", _EofStdin())

    started = time.monotonic()
    decision = TerminalApprover(timeout_s=0.5).request_approval(
        _action(), reasoning="r", confidence=0.5
    )
    elapsed = time.monotonic() - started
    assert not decision.is_approved
    assert decision.reason == "no-input-channel", f"got {decision.reason!r}"
    assert elapsed < 0.4, f"EOF should return promptly, took {elapsed:.2f}s"
    os.close(r)


def test_fd_continuous_stream_timeout_zero_is_bounded(quiet_stdout, monkeypatch):
    """timeout=0 on a *continuously* readable stream must stay bounded.

    Regression guard for the ``_read_until_newline`` liveness defect:
    the helper's ``while True`` loop read one byte at a time with no
    deadline or byte budget, so a producer that keeps supplying bytes
    without a newline could monopolize the reader, bypass the
    ``timeout=0`` "single poll" contract, and grow the partial buffer
    without bound.  With a bounded budget the poll must return promptly
    as a timeout rejection (no approval — no newline was seen) and the
    reader must be released.
    """
    import os
    import time

    r, w = os.pipe()

    def _pump() -> None:
        # Keep the pipe buffer full of newline-free bytes so every
        # os.read succeeds and an unbounded loop would never return.
        chunk = b"x" * 65536
        try:
            while True:
                try:
                    os.write(w, chunk)
                except OSError:
                    break
        finally:
            os.close(w)

    t = threading.Thread(target=_pump, daemon=True)
    t.start()

    class _StreamStdin:
        def fileno(self):
            return r

        def readline(self):  # pragma: no cover - fd path must be taken
            raise AssertionError("fd path must handle the continuous stream")

    monkeypatch.setattr("sys.stdin", _StreamStdin())

    started = time.monotonic()
    decision = TerminalApprover(timeout_s=0).request_approval(
        _action(), reasoning="r", confidence=0.5
    )
    elapsed = time.monotonic() - started
    os.close(r)
    assert not decision.is_approved
    assert decision.reason == "timeout", f"got {decision.reason!r}"
    assert elapsed < 1.0, f"timeout=0 poll was not bounded: {elapsed:.2f}s"


def test_fd_continuous_stream_timeout_positive_is_bounded(quiet_stdout, monkeypatch):
    """timeout>0 on a *continuously* readable stream must respect the deadline.

    Regression guard: the same unbounded helper also defeated the
    ``timeout > 0`` path, because the caller only checked the deadline
    *after* the helper returned — and a continuous stream made the
    helper never return.  The deadline must therefore be enforced
    inside the read loop as well, so the wait ends within the deadline
    (plus a small scheduling margin) with a timeout rejection.
    """
    import os
    import time

    r, w = os.pipe()

    def _pump() -> None:
        chunk = b"x" * 65536
        try:
            while True:
                try:
                    os.write(w, chunk)
                except OSError:
                    break
        finally:
            os.close(w)

    t = threading.Thread(target=_pump, daemon=True)
    t.start()

    class _StreamStdin:
        def fileno(self):
            return r

        def readline(self):  # pragma: no cover - fd path must be taken
            raise AssertionError("fd path must handle the continuous stream")

    monkeypatch.setattr("sys.stdin", _StreamStdin())

    started = time.monotonic()
    decision = TerminalApprover(timeout_s=0.1).request_approval(
        _action(), reasoning="r", confidence=0.5
    )
    elapsed = time.monotonic() - started
    os.close(r)
    assert not decision.is_approved
    assert decision.reason == "timeout", f"got {decision.reason!r}"
    # Deadline 0.1 s; allow generous scheduling/pipe-buffer margin but
    # not the unbounded spin the defect allowed.
    assert elapsed < 1.5, f"continuous stream defeated the deadline: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# 7b) Finding 3 — TerminalApprover cancellation consistency at the
#     approval-acceptance boundary.
#
# The contract: a cancellation already signalled (F12 ``cancel_event`` set)
# must result in an ``"interrupted"`` rejection, *never* an approval — on
# both the ``timeout=0`` single poll and the ``timeout>0`` wait, and on
# both the fd and the console reader. The ``timeout>0`` and console paths
# already check the cancel signal at their acceptance boundary; Finding 3 is
# that the ``timeout=0`` poll used to commit a buffered reply without the
# check, so a stop that fired before the commit could be overridden by a
# ``y`` already sitting in the pipe.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _buffered_reply_fd(reply: bytes = b"y\n"):
    """Context manager yielding a real-fd stdin pipe whose reply is *already*
    fully buffered.

    The writer emits the complete line (through its ``\\n``) and the write
    end is closed *before* the approver polls, so a ``timeout=0`` single poll
    or a ``timeout>0`` acceptance boundary both find the reply present. This
    is exactly the Finding 3 scenario: the decision to approve or interrupt is
    made purely on the cancel signal, with no timing dependency (no late
    ``y`` races, no ``time.sleep``).

    The reader must use the raw fd (``readline`` is guarded to fail loudly
    if the fd path regresses back onto the line-buffered read).

    Ownership is unambiguous: both ends are guaranteed closed on every exit
    path — success or exception — as soon as ``os.pipe()`` succeeds, including
    a failure of ``os.write`` in setup (see the regression test below). The
    original setup exception is re-raised unmasked: close errors during
    cleanup cannot replace it.
    """
    import os

    r, w = os.pipe()
    try:
        os.write(w, reply)  # fully buffered on the read end
    except BaseException:
        # A setup failure must not leak either descriptor; re-raise the
        # original exception (close errors below must not mask it).
        with contextlib.suppress(OSError):
            os.close(r)
        with contextlib.suppress(OSError):
            os.close(w)
        raise
    try:
        os.close(w)  # writer done; buffered bytes remain readable on r
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(r)
        raise

    class _FdStdin:
        def fileno(self):
            return r

        def readline(self):  # pragma: no cover - the fd path reads the raw fd
            raise AssertionError("readline() must not be used on the fd path")

    try:
        yield _FdStdin()
    finally:
        os.close(r)  # the read end must not leak until process exit


def test_buffered_reply_fd_closes_descriptors_on_setup_failure(
    monkeypatch,
):
    """Regression: if ``os.write`` raises *after* ``os.pipe()`` succeeded
    (a setup failure), the helper must close **both** descriptors and let the
    original setup exception propagate unmasked.

    Before the fix, ``_buffered_reply_fd`` ran ``os.write(w, reply)`` and
    ``os.close(w)`` *before* entering its ``try/finally``. A write failure
    therefore skipped the ``finally`` entirely and leaked both ``r`` and ``w``
    (and the happy-path close of ``w`` was also the only thing releasing the
    write end, so it leaked too). This test injects a write failure and pins:

    * both descriptors are closed (``os.fstat`` -> ``EBADF`` on a leaked fd
      would succeed instead), and
    * the *original* setup exception survives — a close error during cleanup
      must not replace it.
    """
    import errno
    import os

    seen_fds: list[int] = []

    real_pipe = os.pipe

    def tracking_pipe():
        r, w = real_pipe()
        seen_fds.extend((r, w))
        return r, w

    def failing_write(fd: int, data: bytes) -> int:  # noqa: ARG001
        raise OSError(errno.EBADF, "injected setup failure")

    monkeypatch.setattr(os, "pipe", tracking_pipe)
    monkeypatch.setattr(os, "write", failing_write)

    with pytest.raises(OSError, match="injected setup failure") as excinfo:
        with _buffered_reply_fd(b"y\n"):
            pass  # pragma: no cover - setup raised, body never entered

    assert isinstance(excinfo.value, OSError)
    assert "injected setup failure" in str(excinfo.value)
    assert excinfo.value.errno == errno.EBADF

    # Both descriptors must be closed: a still-open (leaked) fd would be
    # stat-able, a closed one raises EBADF.
    for fd in seen_fds:
        with pytest.raises(OSError) as err:
            os.fstat(fd)
        assert err.value.errno == errno.EBADF, f"fd {fd} leaked (not closed)"


def test_fd_timeout_zero_buffered_y_with_cancel_set_is_interrupted(
    quiet_stdout, monkeypatch
):
    """Finding 3 (core): ``timeout=0`` must not commit a buffered approval
    once cancellation has already been requested.

    Regression guard: the ``timeout=0`` single poll used to hand a fully
    buffered ``y`` straight to the decision without checking ``cancel_event``,
    so a proposal could be *approved* even though the F12 stop had already
    fired. The poll's acceptance boundary must check the cancel signal first
    and return an ``"interrupted"`` rejection (never an approval).

    Deterministic: the reply is pre-buffered on a real pipe and the cancel
    signal is set *before* the commit — no timing involved.
    """
    with _buffered_reply_fd(b"y\n") as fake_stdin:
        monkeypatch.setattr("sys.stdin", fake_stdin)
        cancel = threading.Event()
        cancel.set()  # emergency stop already requested before the decision commit
        decision = TerminalApprover(
            timeout_s=0, cancel_event=cancel
        ).request_approval(_action(), reasoning="r", confidence=0.5)
        assert not decision.is_approved
        assert decision.reason == "interrupted", f"got {decision.reason!r}"


def test_fd_timeout_zero_buffered_y_without_cancel_is_approved(
    quiet_stdout, monkeypatch
):
    """Finding 3 companion: with cancellation *clear*, a buffered ``y`` under
    ``timeout=0`` is still honoured (the cancel check must not swallow a
    legitimate fast reply). Pins the legitimate path the fix must not break."""
    with _buffered_reply_fd(b"y\n") as fake_stdin:
        monkeypatch.setattr("sys.stdin", fake_stdin)
        decision = TerminalApprover(
            timeout_s=0, cancel_event=threading.Event()  # clear, not set
        ).request_approval(_action(), reasoning="r", confidence=0.5)
        assert decision.is_approved, f"buffered reply ignored: {decision!r}"
        assert decision.reason == "user-approved"


def test_fd_timeout_positive_buffered_y_with_cancel_set_is_interrupted(
    quiet_stdout, monkeypatch
):
    """Finding 3: the ``timeout>0`` path already checks cancellation at the
    acceptance boundary — pin that a buffered ``y`` with a *set* cancel signal
    still rejects as ``"interrupted"`` (stop wins over approval). This is the
    consistent contract the ``timeout=0`` fix must match, on the same fd seam.

    Deterministic: the buffered reply is present from t=0, and the cancel
    signal is checked before any read, so no scheduling margin is needed.
    """
    with _buffered_reply_fd(b"y\n") as fake_stdin:
        monkeypatch.setattr("sys.stdin", fake_stdin)
        cancel = threading.Event()
        cancel.set()
        decision = TerminalApprover(
            timeout_s=1.0, cancel_event=cancel
        ).request_approval(_action(), reasoning="r", confidence=0.5)
        assert not decision.is_approved
        assert decision.reason == "interrupted", f"got {decision.reason!r}"


def test_console_timeout_zero_buffered_y_with_cancel_set_is_interrupted(
    quiet_stdout
):
    """Finding 3 (console path): a ``timeout=0`` poll that finds a fully
    buffered ``y`` must still lose when cancellation is already set — the
    console path checks ``cancel_event`` before returning a completed line.

    Deterministic: a scripted reader reports the key as pending from t=0;
    the cancel signal is set before the first slice, so the wait unblocks on
    the very first probe (no ``time.sleep`` flakiness).
    """
    reader = _FakeConsoleReader(("y", "\r"))
    cancel = threading.Event()
    cancel.set()
    decision = TerminalApprover(
        timeout_s=0, cancel_event=cancel, reader=reader
    ).request_approval(_action(), reasoning="r", confidence=0.5)
    assert not decision.is_approved
    assert decision.reason == "interrupted", f"got {decision.reason!r}"


def test_console_timeout_positive_buffered_y_with_cancel_set_is_interrupted(
    quiet_stdout
):
    """Finding 3 (console path): the ``timeout>0`` loop already checks
    ``cancel_event`` each slice — pin that a pending ``y`` with cancellation
    set still rejects as ``"interrupted"`` (stop wins over approval), keeping
    the two timeout branches and both reader seams consistent."""
    reader = _FakeConsoleReader(("y", "\r"))
    cancel = threading.Event()
    cancel.set()
    decision = TerminalApprover(
        timeout_s=1.0, cancel_event=cancel, reader=reader
    ).request_approval(_action(), reasoning="r", confidence=0.5)
    assert not decision.is_approved
    assert decision.reason == "interrupted", f"got {decision.reason!r}"


def test_console_timeout_zero_pending_reply_without_cancel_is_approved(
    quiet_stdout
):
    """Console ``timeout=0`` must be a genuine non-blocking probe.

    Regression: the console reader polled the deadline *before* ``kbhit()``,
    so with ``timeout=0`` it returned ``timeout`` on the first slice without
    ever probing a pending key — contradicting the documented "single
    non-blocking probe" contract and diverging from the fd path, which honours
    an already-buffered reply on the ``timeout=0`` poll. A reply already
    available from t=0 must therefore approve when cancellation is clear.
    """
    reader = _FakeConsoleReader(("y", "\r"))
    decision = TerminalApprover(
        timeout_s=0,
        cancel_event=threading.Event(),  # clear, not set
        reader=reader,
    ).request_approval(_action(), reasoning="r", confidence=0.5)
    assert decision.is_approved, (
        f"console timeout=0 ignored an available reply: {decision!r}"
    )
    assert decision.reason == "user-approved", f"got {decision.reason!r}"


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


def test_approval_prompt_shows_duration_delay_and_scroll(quiet_stdout, monkeypatch):
    """``duration_ms`` (a 500 ms KEY_HOLD is a different action from a tap),
    ``delay_ms`` (a delayed action vs. an immediate one), and ``scroll``
    (a non-zero scroll amount is a different action from a click) deviate
    from the defaults and must be visible in the prompt."""
    captured = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    hold = Action(kind=ActionKind.KEY_HOLD, key="w", duration_ms=500, source="ai")
    TerminalApprover(timeout_s=0.5).request_approval(
        hold, reasoning="r", confidence=0.5
    )
    out = captured.getvalue()
    assert "duration_ms" in out, f"duration_ms missing from prompt: {out!r}"
    assert "500" in out, f"duration value missing from prompt: {out!r}"

    captured2 = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured2)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    delayed = Action(kind=ActionKind.KEY_PRESS, key="e", delay_ms=250, source="ai")
    TerminalApprover(timeout_s=0.5).request_approval(
        delayed, reasoning="r", confidence=0.5
    )
    out2 = captured2.getvalue()
    assert "delay_ms" in out2, f"delay_ms missing from prompt: {out2!r}"
    assert "250" in out2, f"delay value missing from prompt: {out2!r}"

    captured3 = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured3)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    scrolled = Action(kind=ActionKind.MOUSE_SCROLL, scroll=3, source="ai")
    TerminalApprover(timeout_s=0.5).request_approval(
        scrolled, reasoning="r", confidence=0.5
    )
    out3 = captured3.getvalue()
    assert "scroll" in out3, f"scroll missing from prompt: {out3!r}"
    assert "3" in out3, f"scroll value missing from prompt: {out3!r}"


# ---------------------------------------------------------------------------
# 7b) Terminal injection hardening (Review P1, most important outstanding)
#
# The model's ``reasoning`` is untrusted and is printed into the approval
# prompt — the human's security boundary between an AI proposal and actual
# input execution. It must not be able to overwrite/obscure the prompt via
# ANSI control sequences, carriage returns, newlines, or any other
# non-printable control character, and it must stay bounded. The original
# reasoning value is never mutated; only its presentation is sanitized.
# ---------------------------------------------------------------------------


def _rendered_prompt(monkeypatch, reasoning: str) -> str:
    """Render one approval prompt to a captured stdout and return it."""
    captured = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    action = Action(kind=ActionKind.KEY_PRESS, key="a", source="ai")
    TerminalApprover(timeout_s=0.5).request_approval(
        action, reasoning=reasoning, confidence=0.5
    )
    return captured.getvalue()


def test_prompt_escapes_ansi_escapes_in_reasoning(quiet_stdout, monkeypatch):
    """A raw ESC (the start of any ANSI CSI/OSC sequence) must never reach the
    terminal; it is replaced by a visible ``\\x1b`` escape. The payload text
    stays visible (transparency) but inert."""
    out = _rendered_prompt(
        monkeypatch, "\x1b[2J\x1b[HFAKE APPROVE \x1b[31mnow\x1b[0m"
    )
    assert "\x1b" not in out, f"raw ESC reached the terminal: {out!r}"
    assert "\\x1b" in out, f"ESC must be visibly escaped: {out!r}"
    assert "FAKE APPROVE" in out, "payload text should stay visible, just inert"


def test_prompt_escapes_carriage_return_in_reasoning(quiet_stdout, monkeypatch):
    """A raw CR would move the cursor to column 0 and let following text
    overwrite the prompt; it must be escaped, not emitted."""
    out = _rendered_prompt(monkeypatch, "REAL\r\nOVERWRITE APPROVAL")
    assert "\r" not in out, f"raw CR reached the terminal: {out!r}"
    assert "\\x0d" in out, f"CR must be visibly escaped: {out!r}"


def test_prompt_escapes_newline_in_reasoning(quiet_stdout, monkeypatch):
    """A raw newline in the reasoning must not open a new prompt line; it is
    escaped so the reasoning stays on a single visual line."""
    out = _rendered_prompt(monkeypatch, "line one\nline two")
    assert "\\x0a" in out, f"newline must be visibly escaped: {out!r}"
    reasoning_line = [ln for ln in out.split("\n") if "reasoning:" in ln][0]
    assert "line one" in reasoning_line and "line two" in reasoning_line, (
        f"reasoning split across lines: {reasoning_line!r}"
    )


def test_prompt_truncates_overlong_reasoning(quiet_stdout, monkeypatch):
    """An overlong reasoning is bounded so it cannot flood the terminal.
    Truncation must not strip the escaping applied to control chars."""
    evil = ("x\x1b") * 5000
    out = _rendered_prompt(monkeypatch, evil)
    reasoning_line = [ln for ln in out.split("\n") if "reasoning:" in ln][0]
    assert len(reasoning_line) < 2000, (
        f"reasoning not bounded: {len(reasoning_line)} chars"
    )
    assert "\x1b" not in out, f"raw ESC leaked past truncation: {out!r}"


def test_prompt_sanitization_does_not_mutate_reasoning(quiet_stdout, monkeypatch):
    """Sanitizing is presentation-only: the caller's reasoning value is
    unchanged after the approver returns."""
    evil = "\x1b[31m\r\nkeep"
    captured = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    reasoning = evil
    decision = TerminalApprover(timeout_s=0.5).request_approval(
        Action(kind=ActionKind.KEY_PRESS, key="a", source="ai"),
        reasoning=reasoning,
        confidence=0.5,
    )
    assert reasoning == evil, "the approver must not mutate the reasoning value"
    assert decision.is_approved is False


# ---------------------------------------------------------------------------
# 8) Interactive Windows console input (msvcrt kbhit/getwch path)
#
# On a Windows console (CONIN$) ``select`` rejects the fd (WinError 10038)
# and the ``os.read`` bounded fallback is either refused (``set_blocking``)
# or blocks until Enter — so neither enforces the prompt deadline and an
# F12 cancel can be bypassed by a late ``y``. The approver must instead
# read the console key-by-key through an injectable reader protocol
# (``kbhit``/``getwch``), keeping the deadline, cancellation and fail-safe
# semantics. Tests inject a scripted fake reader (deterministic, no console
# required); the live-CONIN$ checks run only on a real Windows console.
# ---------------------------------------------------------------------------

import sys  # noqa: E402
import time  # noqa: E402

import ai_game_agent.actions.approver as _approver_mod  # noqa: E402


class _FakeConsoleReader:
    """Duck-typed ``ConsoleInputReader``: a scripted queue of pending keys.

    Enter is modelled as ``"\\r"`` (what ``msvcrt.getwch()`` returns for
    the Enter key). ``available_after`` hides the keys from ``kbhit()``
    until that many seconds have passed (a human who types late);
    ``eof`` makes ``getwch()`` raise ``EOFError`` (Ctrl-Z) and
    ``kbdint`` makes ``kbhit()`` raise ``KeyboardInterrupt`` (Ctrl-C).
    """

    def __init__(
        self,
        keys: tuple[str, ...] = (),
        *,
        available_after: float = 0.0,
        eof: bool = False,
        kbdint: bool = False,
    ) -> None:
        self._keys = list(keys)
        self._t0 = time.monotonic()
        self._available_after = available_after
        self._eof = eof
        self._kbdint = kbdint
        self.kbhit_calls = 0

    def kbhit(self) -> bool:
        self.kbhit_calls += 1
        if self._kbdint:
            raise KeyboardInterrupt
        if (time.monotonic() - self._t0) < self._available_after:
            return False
        return bool(self._keys) or self._eof

    def getwch(self) -> str:
        if self._eof:
            raise EOFError
        return self._keys.pop(0)


def test_console_reader_affirmative_reply_approves(quiet_stdout):
    reader = _FakeConsoleReader(("y", "\r"))
    approver = TerminalApprover(timeout_s=1.0, reader=reader)
    decision = approver.request_approval(_action(), reasoning="r", confidence=0.5)
    assert decision.is_approved, f"console 'y' must approve: {decision}"
    assert decision.reason == "user-approved"


def test_console_reader_word_replies(quiet_stdout):
    approver = TerminalApprover(timeout_s=1.0, reader=_FakeConsoleReader(("yes", "\r")))
    assert approver.request_approval(_action()).is_approved

    approver = TerminalApprover(timeout_s=1.0, reader=_FakeConsoleReader(("no", "\r")))
    decision = approver.request_approval(_action())
    assert not decision.is_approved
    assert decision.reason == "user-rejected"


def test_console_reader_blank_line_rejects(quiet_stdout):
    approver = TerminalApprover(timeout_s=1.0, reader=_FakeConsoleReader(("\r",)))
    decision = approver.request_approval(_action())
    assert not decision.is_approved
    assert decision.reason == "user-rejected"


def test_console_reader_garbage_rejects(quiet_stdout):
    approver = TerminalApprover(
        timeout_s=1.0, reader=_FakeConsoleReader(("m", "a", "y", "b", "e", "\r"))
    )
    decision = approver.request_approval(_action())
    assert not decision.is_approved
    assert decision.reason == "unparsable-reply"


def test_console_reader_eof_rejects(quiet_stdout):
    """Ctrl-Z (EOF) mid-wait is a fail-safe rejection, never an approval."""
    approver = TerminalApprover(timeout_s=1.0, reader=_FakeConsoleReader(eof=True))
    decision = approver.request_approval(_action())
    assert not decision.is_approved
    assert decision.reason == "no-input-channel"


def test_console_reader_keyboard_interrupt_rejects(quiet_stdout):
    """Ctrl-C (KeyboardInterrupt) mid-wait is a rejection, not a crash."""
    approver = TerminalApprover(timeout_s=1.0, reader=_FakeConsoleReader(kbdint=True))
    decision = approver.request_approval(_action())
    assert not decision.is_approved


def test_console_reader_timeout_rejects_within_deadline(quiet_stdout):
    """No key press must time out with reason ``"timeout"`` — the deadline
    is actually polled (several probes), not one-shot."""
    reader = _FakeConsoleReader()
    approver = TerminalApprover(timeout_s=0.3, reader=reader)
    t0 = time.monotonic()
    decision = approver.request_approval(_action())
    elapsed = time.monotonic() - t0
    assert not decision.is_approved
    assert decision.reason == "timeout"
    assert elapsed < 1.0, f"console wait ran {elapsed:.2f}s past a 0.3s deadline"
    assert reader.kbhit_calls > 1, "deadline loop never re-polled the reader"


def test_console_reader_cancel_interrupts_pending_wait(quiet_stdout):
    """A set ``cancel_event`` (F12) must unblock a pending console wait
    promptly with an ``"interrupted"`` rejection."""
    cancel = threading.Event()
    approver = TerminalApprover(
        timeout_s=10.0, cancel_event=cancel, reader=_FakeConsoleReader()
    )

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
    assert not t.is_alive(), "cancel_event did not unblock the console wait"
    decision = result_box.get("decision")
    assert decision is not None
    assert not decision.is_approved
    assert decision.reason == "interrupted"


def test_console_reader_late_reply_after_deadline_is_timeout(quiet_stdout):
    """A reply typed *after* the deadline must not be honoured (the P1
    bypass: late ``y`` must never approve a timed-out proposal)."""
    approver = TerminalApprover(
        timeout_s=0.2, reader=_FakeConsoleReader(("y", "\r"), available_after=0.5)
    )
    decision = approver.request_approval(_action())
    assert not decision.is_approved
    assert decision.reason == "timeout"


def test_console_reader_deadline_honours_pending_reply_at_first_probe(quiet_stdout):
    """F1 companion: a reply already pending at the **first** probe (t <
    deadline) is still honoured — the deadline check does not swallow a
    legitimate fast reply. Pre-fix and post-fix both approve, so this pins
    the *legitimate* path that F1's fix must not regress."""

    class _ClockReader:
        def __init__(self) -> None:
            self._keys = ["y", "\r"]

        def kbhit(self) -> bool:
            return bool(self._keys)

        def getwch(self) -> str:
            return self._keys.pop(0)

    reader = _ClockReader()
    # Clock fixed at t=0, deadline at 1.0: the first probe is well before the
    # deadline, so the pending reply is honoured.
    approver = TerminalApprover(timeout_s=1.0, reader=reader, now_fn=lambda: 0.0)
    decision = approver.request_approval(_action())
    assert decision.is_approved, (
        f"a pending reply at the first probe must approve, got {decision!r}"
    )
    assert decision.reason == "user-approved"


def test_console_reader_late_y_after_cancel_is_interrupted(quiet_stdout):
    """A ``y`` typed after F12 must still lose: stop wins over approval."""
    cancel = threading.Event()
    approver = TerminalApprover(
        timeout_s=10.0,
        cancel_event=cancel,
        reader=_FakeConsoleReader(("y", "\r"), available_after=0.5),
    )

    result_box: dict[str, object] = {}

    def _call() -> None:
        result_box["decision"] = approver.request_approval(_action())

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    time.sleep(0.3)
    cancel.set()
    t.join(timeout=2.0)
    assert not t.is_alive(), "cancel_event did not unblock the console wait"
    decision = result_box.get("decision")
    assert decision is not None
    assert not decision.is_approved
    assert decision.reason == "interrupted"


def test_console_reader_zero_timeout_is_single_probe(quiet_stdout):
    """``timeout=0`` on the console path is a non-blocking poll, not an
    unbounded wait: a reply arriving at 0.3 s is not honoured."""
    approver = TerminalApprover(
        timeout_s=0, reader=_FakeConsoleReader(("y", "\r"), available_after=0.3)
    )
    t0 = time.monotonic()
    decision = approver.request_approval(_action())
    elapsed = time.monotonic() - t0
    assert not decision.is_approved
    assert decision.reason == "timeout"
    assert elapsed < 0.5, f"timeout=0 blocked for {elapsed:.2f}s"


def test_auto_detects_console_reader_for_tty_stdin(quiet_stdout, monkeypatch):
    """An interactive (``isatty()``) stdin selects the console reader —
    pinning the dispatch decision without reading a real console."""

    class _TtyStdin:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", _TtyStdin())
    sentinel = _FakeConsoleReader()
    monkeypatch.setattr(_approver_mod, "MsvcrtConsoleReader", lambda: sentinel)
    approver = TerminalApprover(timeout_s=0.5)
    assert approver._console_reader() is sentinel


def test_injected_reader_wins_over_auto_detect(quiet_stdout):
    injected = _FakeConsoleReader(("y", "\r"))
    approver = TerminalApprover(timeout_s=1.0, reader=injected)
    assert approver._console_reader() is injected


def test_no_console_reader_for_non_tty_stdin(quiet_stdout, monkeypatch):
    """Piped / redirected stdin keeps the existing fd read paths."""
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    approver = TerminalApprover(timeout_s=0.5)
    assert approver._console_reader() is None


def _drain_console() -> None:
    """Consume stray keystrokes so the live checks start from a clean buffer."""
    import msvcrt

    while msvcrt.kbhit():
        msvcrt.getwch()


@pytest.mark.skipif(
    sys.platform != "win32" or not sys.stdin.isatty(),
    reason="needs a live Windows console (CONIN$)",
)
def test_live_windows_console_timeout_rejects(quiet_stdout):
    """Real CONIN$ path: no key press → ``"timeout"`` rejection within the
    deadline (previously this path hung or was refused outright)."""
    _drain_console()
    t0 = time.monotonic()
    decision = TerminalApprover(timeout_s=0.5).request_approval(_action())
    elapsed = time.monotonic() - t0
    assert not decision.is_approved
    assert decision.reason == "timeout"
    assert elapsed < 2.0, f"live console wait exceeded deadline: {elapsed:.2f}s"


@pytest.mark.skipif(
    sys.platform != "win32" or not sys.stdin.isatty(),
    reason="needs a live Windows console (CONIN$)",
)
def test_live_windows_console_cancel_interrupts(quiet_stdout):
    """Real CONIN$ path: F12 (cancel_event) during the prompt unblocks the
    wait promptly with ``"interrupted"`` — no 30 s sit-out, no late-``y``
    bypass (previously the bounded fallback blocked on a line-buffered
    read and ignored the cancel until Enter)."""
    _drain_console()
    cancel = threading.Event()
    approver = TerminalApprover(timeout_s=30.0, cancel_event=cancel)

    result_box: dict[str, object] = {}

    def _call() -> None:
        result_box["d"] = approver.request_approval(_action())

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    time.sleep(0.3)
    cancel.set()
    t.join(timeout=3.0)
    assert not t.is_alive(), "cancel did not unblock the live console wait"
    decision = result_box.get("d")
    assert decision is not None
    assert not decision.is_approved
    assert decision.reason == "interrupted"

def test_console_reader_survives_clock_jump_past_deadline(quiet_stdout):
    """P2 (review #2): if the clock advances *past* the deadline between the
    top-of-loop deadline check and the ``time.sleep`` at the bottom of the
    same iteration, ``deadline - now`` is negative and the unclamped
    ``time.sleep`` raises ``ValueError`` — aborting the approval prompt at
    the security boundary instead of letting the loop continue (and
    eventually time out, the fail-safe). The ``max(0.0, ...)`` clamp guards
    exactly this window.

    The jump happens inside ``kbhit`` (which returns False, so the loop
    falls through to the sleep line) — the only sequence that puts the
    clock past the deadline *after* the top-of-loop check but *before* the
    sleep. Pre-fix the sleep raises; post-fix it clamps to zero and the
    loop continues to the next probe, where the deadline check fires
    (a timeout — the fail-safe; the pending reply is never an approval).
    """
    state = {"now": 0.5}

    def clock() -> float:
        return state["now"]

    class _Reader:
        """First ``kbhit``: jump the clock past the deadline, return False
        (force the sleep path). Subsequent ``kbhit``: keep returning False
        so the deadline check fires on the next probe."""

        def __init__(self) -> None:
            self._jumped = False

        def kbhit(self) -> bool:
            if not self._jumped:
                self._jumped = True
                state["now"] = 5.0  # jump past the deadline (1.0)
            return False

        def getwch(self) -> str:
            return "y"  # never reached on the failure path

    reader = _Reader()
    approver = TerminalApprover(timeout_s=1.0, reader=reader, now_fn=clock)
    decision = approver.request_approval(_action())
    # Pre-fix: the unclamped sleep raises ValueError — the test errors.
    # Post-fix: the sleep clamps to zero, the loop continues, and the
    # deadline check fires on the next probe (a timeout rejection, never
    # an approval, never a crash).
    assert not decision.is_approved
    assert decision.reason == "timeout"
