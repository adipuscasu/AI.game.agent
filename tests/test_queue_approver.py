"""Phase 6 remote/web approval — the transport-agnostic ``QueueApprover``.

The Phase 6 plan (``docs/phase-6-implementation-plan.md`` §3.2 and §7) names
exactly this seam: a non-terminal ``Approver`` for remote/web approval that
"slots in here without touching the loop". The loop stays dependent on the
``Approver`` *protocol* only; a future web handler is a thin adapter over
:meth:`QueueApprover.pending` (poll for an outstanding request) and
:meth:`QueueApprover.respond` (submit a decision).

Contract (matching the existing ``Approver`` protocol the loop programs
against):

* ``request_approval(action, *, reasoning, confidence, timeout_s)`` **blocks**
  until the request is settled and returns an ``ApprovalDecision``. It never
  raises for a "no".
* ``respond(request_id, decision) -> bool`` — the human/remote side submits an
  outcome for a specific request from another thread. Returns ``True`` only
  when *this* call settled the request; a malformed / unknown / replayed /
  late (past-deadline) / cancelled payload is a no-op (``False``) and never
  produces an approval.
* ``pending() -> tuple[PendingRequest, ...]`` — an immutable snapshot of the
  *outstanding* (unsettled) requests; settled requests are excluded.

Fail-safe rule (the repo invariant every approver honours):

> **A valid, explicitly affirmative decision is the only path to execution.**
> Timeout, cancellation, unknown/replayed/forged id, malformed payload — all
> are rejections, never an approval, and never raise.

Settlement precedence (checked under the lock on both the waiting and
responding sides — first-settle-wins, then immutable):
``cancel_event`` → ``"interrupted"``; then deadline → ``"timeout"``; then a
valid response settles as that decision. So a set ``cancel_event`` is
authoritative even over a valid response, and a late response can never
approve a timed-out request.

These tests pin that contract. The loop-integration tests
(``test_assisted_queue_*`` / ``test_queue_gate_*``) prove the gate stays
loop-owned: an approving ``QueueApprover`` lets the AI action dispatch with an
``APPROVAL`` audit event tagged ``channel == "queue"``; a rejecting / timed-out
/ cancelled one forces ``PAUSED`` and dispatches nothing; and the gate is a
no-op outside ``ASSISTED`` or for deterministic (non-AI) actions.

Concurrency tests use ``threading.Event`` + bounded waits (no unbounded
sleeps) and always join worker threads in ``finally`` so no thread leaks and
no test hangs on failure.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

import pytest
import test_assisted_approval as tai  # self-contained fakes + loop factory

from ai_game_agent.actions import (
    Action,
    ActionKind,
    ApprovalDecision,
    Approver,
    AutoApprover,
    OperationMode,
    PendingRequest,
    QueueApprover,
    TerminalApprover,
)
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.state import AgentState, EventType

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _action() -> Action:
    return tai._action()


def _ai_decision():
    return tai._ai_decision()


def _wait_pending(
    approver: QueueApprover, want_id: str | None = None, timeout: float = 3.0
) -> PendingRequest | None:
    """Bounded poll for an outstanding request. Returns ``None`` on timeout.

    Polls on a bounded slice (no busy-spin, no unbounded sleep) so a failed
    test cannot hang the suite; the caller asserts ``is not None``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for pr in approver.pending():
            if want_id is None or pr.request_id == want_id:
                return pr
        time.sleep(0.005)
    for pr in approver.pending():
        if want_id is None or pr.request_id == want_id:
            return pr
    return None


def _outstanding_ids(approver: QueueApprover) -> set[str]:
    return {p.request_id for p in approver.pending()}


def _wait_outstanding(
    approver: QueueApprover, n: int, timeout: float = 3.0
) -> set[str] | None:
    """Bounded wait until at least ``n`` requests are outstanding (unsettled).

    Returns the id set once the count is reached, or ``None`` on timeout. Used
    to remove a start-time race: two worker threads started back-to-back are not
    guaranteed to have both registered their slot by the time the main thread
    reads ``pending()`` (thread-scheduling dependent). A single read after
    ``start()`` is therefore racy; poll on bounded slices instead so the suite
    is deterministic and never hangs.
    """
    deadline = time.monotonic() + timeout
    while True:
        ids = _outstanding_ids(approver)
        if len(ids) >= n:
            return ids
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# 1) Protocol conformance + channel identity
# ---------------------------------------------------------------------------


def test_queue_approver_satisfies_approver_protocol():
    """Liskov: a ``QueueApprover`` is substitutable for the other approvers.

    The loop depends only on the ``Approver`` protocol, so this is what lets
    the remote/web channel "slot in" without touching the loop (plan §3.2).
    """
    assert isinstance(QueueApprover(), Approver)


def test_queue_approver_name_is_channel_for_audit():
    """``name`` is recorded as the approval channel in the audit event; a
    remote decision must be auditable as a queue-channel decision."""
    assert QueueApprover().name == "queue"


def test_queue_approver_extends_protocol_without_leaking_it():
    """``respond()``/``pending()`` are extensions on the concrete class; the
    ``Approver`` protocol and the other approvers must NOT gain them (the
    loop programs against the protocol only)."""
    assert not hasattr(Approver, "respond")
    assert not hasattr(Approver, "pending")
    assert not hasattr(TerminalApprover, "respond")
    assert not hasattr(AutoApprover, "respond")


# ---------------------------------------------------------------------------
# 2) Core request/response contract (the four settle outcomes)
# ---------------------------------------------------------------------------


def test_request_visible_in_pending_while_waiting():
    """While a request is outstanding (unsettled) it appears in ``pending()``
    with the request data (id, action, reasoning, confidence)."""
    approver = QueueApprover(timeout_s=30.0)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(
            _action(), reasoning="r", confidence=0.7, timeout_s=30.0
        )),
        daemon=True,
    )
    t.start()
    try:
        pr = _wait_pending(approver)
        assert pr is not None, "request did not appear in pending()"
        assert isinstance(pr, PendingRequest)
        assert pr.action.kind is ActionKind.KEY_PRESS
        assert pr.reasoning == "r"
        assert pr.confidence == 0.7
        # A poll is read-only: it does not consume or settle the request.
        assert pr.request_id in _outstanding_ids(approver)
    finally:
        # Settle so the worker thread can't linger on the 30 s default.
        if pr is not None:
            approver.respond(pr.request_id, ApprovalDecision.rejected("done"))
        t.join(timeout=2.0)


def test_approved_response_wakes_caller_and_returns_approved():
    """A valid affirmative response settles the request and wakes the
    blocking caller, which returns an *approved* decision."""
    approver = QueueApprover(timeout_s=30.0)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t.start()
    try:
        pr = _wait_pending(approver)
        assert pr is not None
        assert approver.respond(pr.request_id, ApprovalDecision.approved("remote ok")) is True
    finally:
        t.join(timeout=2.0)
    d = box.get("d")
    assert d is not None, "caller was not woken"
    assert d.is_approved is True
    assert d.reason == "remote ok"
    assert approver.pending() == ()


def test_rejected_response_preserves_reason():
    """A valid negative response settles the request as rejected and preserves
    the human's reason (fail-safe: a "no" is a rejection, never an approval)."""
    approver = QueueApprover(timeout_s=30.0)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t.start()
    try:
        pr = _wait_pending(approver)
        assert pr is not None
        assert approver.respond(pr.request_id, ApprovalDecision.rejected("remote no")) is True
    finally:
        t.join(timeout=2.0)
    d = box.get("d")
    assert d is not None
    assert d.is_approved is False
    assert d.reason == "remote no"


def test_unknown_request_id_returns_false():
    """A never-issued id is a no-op (``False``) — never an approval, no crash."""
    approver = QueueApprover(timeout_s=30.0)
    assert approver.respond("no-such-id", ApprovalDecision.approved("forged")) is False
    assert approver.pending() == ()


def test_replayed_id_returns_false():
    """Replaying an already-settled id cannot re-approve: the slot is single-use
    (removed on settle), so the replay is a no-op."""
    approver = QueueApprover(timeout_s=30.0)
    pr = _wait_pending(approver, timeout=0.0)  # nothing outstanding yet
    # Issue and settle one request, then replay its id.
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t.start()
    try:
        p = _wait_pending(approver)
        assert p is not None
        assert approver.respond(p.request_id, ApprovalDecision.approved("ok")) is True
        # Replay the same id with an approval: must be a no-op (slot is gone).
        assert approver.respond(p.request_id, ApprovalDecision.approved("replay")) is False
        assert pr is None or pr.request_id != p.request_id  # nothing outstanding before
    finally:
        t.join(timeout=2.0)
    assert box.get("d") is not None and box["d"].is_approved
    assert approver.pending() == ()


def test_malformed_decision_returns_false_without_settling():
    """A malformed payload (unknown outcome / non-string reason / not an
    ``ApprovalDecision``) is a no-op: it must not settle the slot, so the
    request stays outstanding until a valid decision or the deadline."""
    approver = QueueApprover(timeout_s=30.0)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t.start()
    try:
        p = _wait_pending(approver)
        assert p is not None
        bad = [
            ApprovalDecision(outcome="maybe", reason="x"),      # unknown outcome
            ApprovalDecision(outcome="approved", reason=123),   # non-str reason
            object(),                                           # not an ApprovalDecision
        ]
        for b in bad:
            assert approver.respond(p.request_id, b) is False, f"{b!r} must not settle"
            # Still outstanding: a malformed payload never settles.
            assert p.request_id in _outstanding_ids(approver)
        # A valid rejection then settles it for good.
        assert approver.respond(p.request_id, ApprovalDecision.rejected("no")) is True
        assert approver.pending() == ()
    finally:
        t.join(timeout=2.0)


def test_timeout_removes_request_and_late_response_cannot_approve():
    """No response before the deadline → ``"timeout"`` rejection and the slot
    is removed from ``pending()``; a late response afterwards is a no-op and
    cannot approve the timed-out proposal (the deadline is validated under the
    same lock that settles)."""
    approver = QueueApprover(timeout_s=0.1)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=0.1)),
        daemon=True,
    )
    t.start()
    p = _wait_pending(approver)
    assert p is not None
    t.join(timeout=2.0)
    d = box.get("d")
    assert d is not None, "request_approval did not return (hung)"
    assert d.is_approved is False
    assert d.reason == "timeout"
    # Removed from the outstanding set; a late approve is a no-op.
    assert approver.pending() == ()
    assert approver.respond(p.request_id, ApprovalDecision.approved("late")) is False


def test_cancel_returns_interrupted_and_removes_request():
    """A set ``cancel_event`` (F12) unblocks a pending wait promptly (bounded,
    not waiting out the timeout) and settles it as ``"interrupted"`` — never an
    approval — then removes it from ``pending()``.

    ``cancel_event`` is set *before* the request starts, so the waiting loop
    sees it on its first slice and breaks immediately.
    """
    cancel = threading.Event()
    cancel.set()
    approver = QueueApprover(timeout_s=30.0, cancel_event=cancel)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t.start()
    t.join(timeout=2.0)
    d = box.get("d")
    assert d is not None, "cancel did not unblock the wait"
    assert d.is_approved is False
    assert d.reason == "interrupted"
    assert approver.pending() == ()


def test_cancel_set_after_wait_start_unblocks_promptly():
    """A ``cancel_event`` set *after* the wait begins also unblocks it (within
    the waiter's bounded re-check slice, not the full 30 s timeout)."""
    cancel = threading.Event()
    approver = QueueApprover(timeout_s=30.0, cancel_event=cancel)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t.start()
    started = _wait_pending(approver, timeout=2.0)
    assert started is not None, "request did not become outstanding"
    t0 = time.monotonic()
    cancel.set()
    t.join(timeout=3.0)
    elapsed = time.monotonic() - t0
    d = box.get("d")
    assert d is not None, "cancel did not unblock the wait"
    assert d.reason == "interrupted"
    assert elapsed < 5.0, f"cancel did not bound the wait (elapsed {elapsed:.2f}s)"
    assert approver.pending() == ()


# ---------------------------------------------------------------------------
# 3) Cancellation of several outstanding requests at once
# ---------------------------------------------------------------------------


def test_cancel_resolves_all_outstanding_requests():
    """Two simultaneous outstanding requests: one cancel settles *both* as
    ``"interrupted"`` (an emergency stop halts the episode — the safe
    behaviour) and removes both from ``pending()``."""
    cancel = threading.Event()
    approver = QueueApprover(timeout_s=30.0, cancel_event=cancel)
    box1: dict[str, Any] = {}
    box2: dict[str, Any] = {}
    t1 = threading.Thread(
        target=lambda: box1.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t2 = threading.Thread(
        target=lambda: box2.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t1.start()
    t2.start()
    try:
        ids = _wait_outstanding(approver, 2)
        assert ids is not None, "expected two outstanding requests"
        cancel.set()
    finally:
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)
    assert box1.get("d") is not None and box1["d"].reason == "interrupted"
    assert box2.get("d") is not None and box2["d"].reason == "interrupted"
    assert approver.pending() == ()


# ---------------------------------------------------------------------------
# 4) Independent correlation of concurrent requests
# ---------------------------------------------------------------------------


def test_concurrent_requests_are_independently_correlated():
    """Two concurrent requests are independent slots: settling one never
    settles the other, and each is resolved by its own id."""
    approver = QueueApprover(timeout_s=30.0)
    box1: dict[str, Any] = {}
    box2: dict[str, Any] = {}
    t1 = threading.Thread(
        target=lambda: box1.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t2 = threading.Thread(
        target=lambda: box2.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t1.start()
    t2.start()
    try:
        ids = _wait_outstanding(approver, 2)
        assert ids is not None, "expected two outstanding requests"
        p1, p2 = sorted(ids)
        # Settle p1 approved; p2 must remain outstanding.
        assert approver.respond(p1, ApprovalDecision.approved("y1")) is True
        assert p2 in _outstanding_ids(approver)
        assert p1 not in _outstanding_ids(approver)
        # Settle p2 rejected; nothing remains.
        assert approver.respond(p2, ApprovalDecision.rejected("n2")) is True
        assert approver.pending() == ()
    finally:
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)
    results = {box1.get("d"), box2.get("d")}
    assert any(d is not None and d.is_approved for d in results)
    assert any(d is not None and not d.is_approved for d in results)


def test_concurrent_respond_first_settle_wins():
    """Two concurrent ``respond()`` calls for the *same* request: exactly one
    settles (``True``), the other is a no-op (``False``) — first-settle-wins
    under the lock. Whichever wins, a valid decision settles it (never lost)."""
    approver = QueueApprover(timeout_s=30.0)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t.start()
    p = _wait_pending(approver)
    assert p is not None
    results: list[bool] = []
    lock = threading.Lock()

    def _respond(decision: ApprovalDecision) -> None:
        ok = approver.respond(p.request_id, decision)
        with lock:
            results.append(ok)

    rt1 = threading.Thread(
        target=_respond, args=(ApprovalDecision.approved("first"),)
    )
    rt2 = threading.Thread(
        target=_respond, args=(ApprovalDecision.rejected("second"),)
    )
    try:
        rt1.start()
        rt2.start()
    finally:
        rt1.join(timeout=2.0)
        rt2.join(timeout=2.0)
        t.join(timeout=2.0)
    assert sorted(results) == [False, True], f"expected exactly one settle, got {results}"
    assert box.get("d") is not None
    assert approver.pending() == ()


def test_timeout_on_one_request_does_not_affect_the_other():
    """A timeout settles only its own request; the other stays outstanding and
    is still approvable."""
    approver = QueueApprover(timeout_s=30.0)
    box1: dict[str, Any] = {}
    box2: dict[str, Any] = {}
    t1 = threading.Thread(
        target=lambda: box1.update(d=approver.request_approval(_action(), timeout_s=0.15)),
        daemon=True,
    )
    t2 = threading.Thread(
        target=lambda: box2.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t1.start()
    t2.start()
    try:
        ids = _wait_outstanding(approver, 2)
        assert ids is not None, "expected two outstanding requests"
        t1.join(timeout=2.0)  # request 1 times out
        assert box1.get("d") is not None and box1["d"].reason == "timeout"
        remaining = _outstanding_ids(approver)
        assert len(remaining) == 1, f"request 2 should still be outstanding: {remaining}"
        rid = next(iter(remaining))
        assert approver.respond(rid, ApprovalDecision.approved("y2")) is True
    finally:
        t2.join(timeout=2.0)
    assert box2.get("d") is not None and box2["d"].is_approved
    assert approver.pending() == ()


# ---------------------------------------------------------------------------
# 5) Injected-ID safety: no overwrite / no corruption of other requests
# ---------------------------------------------------------------------------


def test_duplicate_injected_id_cannot_overwrite_an_outstanding_request():
    """A misconfigured id factory that yields the *same* id twice must not let
    the second request overwrite the first outstanding one (that would let a
    later ``respond()`` settle the wrong slot and corrupt it). The second
    registration is refused as a fail-safe rejection; the first is untouched.
    """

    def factory() -> str:
        return "same-id"

    approver = QueueApprover(timeout_s=30.0, id_factory=factory)
    box1: dict[str, Any] = {}
    t1 = threading.Thread(
        target=lambda: box1.update(
            d=approver.request_approval(_action(), reasoning="first", timeout_s=30.0)
        ),
        daemon=True,
    )
    t1.start()
    p1 = _wait_pending(approver, want_id="same-id")
    assert p1 is not None, "first request did not become outstanding"
    assert p1.reasoning == "first"

    box2: dict[str, Any] = {}
    t2 = threading.Thread(
        target=lambda: box2.update(
            d=approver.request_approval(_action(), reasoning="second", timeout_s=30.0)
        ),
        daemon=True,
    )
    t2.start()
    try:
        # The second request must NOT have overwritten the first: "first" is
        # still outstanding under "same-id".
        outstanding = [p for p in approver.pending() if p.request_id == "same-id"]
        assert outstanding and any(p.reasoning == "first" for p in outstanding), (
            f"second request overwrote the first: {[(p.reasoning) for p in outstanding]}"
        )
        # The second registration was refused (a rejection, never an approval).
        # It must not still be outstanding under a fresh id either.
        assert box2.get("d") is not None and not box2["d"].is_approved, (
            "second request must be a fail-safe rejection, not an approval"
        )
        # The rejection reason must describe the *actual* failure — a duplicate
        # correlation id — and must not be mislabeled as a missing input
        # channel. (Finding #1: the audit trail must be able to tell a
        # misconfigured/malicious id factory apart from "no stdin".)
        assert box2["d"].reason == "duplicate-request-id", (
            f"duplicate id was rejected with the wrong reason: {box2['d'].reason!r}"
        )
        # Approve the *first*; the second must not be approved by it.
        assert approver.respond("same-id", ApprovalDecision.approved("ok")) is True
    finally:
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
    assert box1.get("d") is not None and box1["d"].is_approved, "first must be approved"
    assert box2.get("d") is not None and not box2["d"].is_approved, "second must not be approved"
    assert approver.pending() == ()


def test_valid_uuid_ids_never_collide():
    """With the default ``uuid4`` factory, distinct requests always get distinct
    ids (the normal path never hits the collision guard)."""
    approver = QueueApprover(timeout_s=30.0)
    ids = {str(approver._id_factory()) for _ in range(200)}
    assert len(ids) == 200, "uuid4 factory must yield unique ids"


def test_assisted_queue_duplicate_id_records_correct_audit_reason():
    """When the injected ``id_factory`` yields an id that is already
    outstanding, the ASSISTED gate must (a) force a safe halt, dispatch
    nothing, and (b) record the *semantically correct* rejection reason in
    the ``APPROVAL`` audit event — not ``no-input-channel``.

    This is the audit-level half of Finding #1: the loop records the
    approver's ``reason`` verbatim into the event detail, so if the approver
    mislabels the failure the audit trail is wrong too.
    """

    def factory() -> str:
        return "dup-id"

    backend = MockBackend()
    ai = tai._FakeAI(decision=_ai_decision())
    approver = QueueApprover(timeout_s=10.0, id_factory=factory)
    events: list[object] = []
    loop = tai._loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=ai,
        max_ticks=3,
        on_event=events.append,
    )

    # Pre-register an outstanding request under "dup-id" so the gate's own
    # ``request_approval`` (which reuses the same fixed factory) hits the
    # collision guard. Deterministic: we wait for the first to be outstanding
    # before running the loop, so the gate always sees a duplicate.
    holder: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: holder.update(
            d=approver.request_approval(
                _action(), reasoning="first", timeout_s=10.0
            )
        ),
        daemon=True,
    )
    t.start()
    try:
        p = _wait_pending(approver, want_id="dup-id", timeout=5.0)
        assert p is not None, "pre-registered request did not become outstanding"
        report = loop.run()
    finally:
        # Settle the pre-registered request so its worker exits cleanly, then
        # join (no leaked thread, no lingering outstanding slot).
        approver.respond("dup-id", ApprovalDecision.approved("ok"))
        t.join(timeout=3.0)

    assert report.actions_ok == 0, "a duplicate-id request must not dispatch"
    assert report.actions_attempted == 0, "a duplicate-id request was dispatched"
    assert loop._machine.state is AgentState.PAUSED, "duplicate id must force a safe halt"
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals, "expected an APPROVAL audit event on the duplicate rejection"
    ev = approvals[0]
    assert ev.detail["approved"] is False
    assert ev.detail["channel"] == "queue"
    # The reason the loop recorded must be the accurate one, not no-input-channel.
    assert ev.detail["reason"] == "duplicate-request-id", (
        f"audit recorded wrong rejection reason: {ev.detail['reason']!r}"
    )
    assert "no-input-channel" not in ev.detail["reason"]
    assert approver.pending() == ()


# ---------------------------------------------------------------------------
# 6) Deterministic race precedence (cancel > response; deadline > late response)
# ---------------------------------------------------------------------------


def test_set_cancel_wins_over_a_valid_response():
    """A *pre-set* ``cancel_event`` is authoritative over a valid response:
    ``respond()`` after the cancel is a no-op (``False``) and the request
    settled as ``"interrupted"`` — a responder cannot bypass an emergency stop."""
    cancel = threading.Event()
    cancel.set()
    approver = QueueApprover(timeout_s=30.0, cancel_event=cancel)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t.start()
    p = _wait_pending(approver, timeout=2.0)
    t.join(timeout=2.0)
    d = box.get("d")
    assert d is not None and d.reason == "interrupted"
    # A valid response arriving now is a no-op (cancel already settled it).
    if p is not None:
        assert approver.respond(p.request_id, ApprovalDecision.approved("too late")) is False
    assert approver.pending() == ()


def test_deadline_wins_over_a_late_response():
    """Deadline expiry is authoritative over a late response: a response that
    arrives after the deadline is a no-op (``False``) and the request settled
    as ``"timeout"`` — even if the waiter had not yet awakened (the deadline is
    validated under the same lock that settles)."""
    approver = QueueApprover(timeout_s=0.1)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=0.1)),
        daemon=True,
    )
    t.start()
    p = _wait_pending(approver, timeout=2.0)
    assert p is not None
    late: dict[str, Any] = {}

    def _late() -> None:
        time.sleep(0.3)  # well past the 0.1 s deadline
        late["ok"] = approver.respond(p.request_id, ApprovalDecision.approved("late"))

    lt = threading.Thread(target=_late, daemon=True)
    try:
        lt.start()
        t.join(timeout=2.0)
    finally:
        lt.join(timeout=2.0)
    d = box.get("d")
    assert d is not None and d.reason == "timeout"
    assert late.get("ok") is False, "a late response must not settle after the deadline"
    assert approver.pending() == ()


def test_response_wins_over_a_late_cancel():
    """Conversely, a *valid* response that wins the settle race is honoured; a
    cancel that arrives *after* is a no-op for that request (the answer already
    arrived). The fail-safe protects the unanswered request, not a settled one."""
    cancel = threading.Event()
    approver = QueueApprover(timeout_s=30.0, cancel_event=cancel)
    box: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: box.update(d=approver.request_approval(_action(), timeout_s=30.0)),
        daemon=True,
    )
    t.start()
    p = _wait_pending(approver)
    assert p is not None
    assert approver.respond(p.request_id, ApprovalDecision.approved("answered")) is True
    cancel.set()  # arrives after the valid response: a no-op for this request
    t.join(timeout=2.0)
    d = box.get("d")
    assert d is not None and d.is_approved and d.reason == "answered"
    assert approver.pending() == ()
    assert approver.respond(p.request_id, ApprovalDecision.approved("replay")) is False


# ---------------------------------------------------------------------------
# 7) PendingRequest immutability
# ---------------------------------------------------------------------------


def test_pending_request_is_frozen_and_exposes_exactly_the_contract_fields():
    """``PendingRequest`` is a frozen dataclass with *exactly* ``request_id``,
    ``action``, ``reasoning``, ``confidence`` — no decision field (the decision
    is produced only on ``respond()``), no ``is_outstanding``."""
    fields = [f.name for f in dataclasses.fields(PendingRequest)]
    assert fields == ["request_id", "action", "reasoning", "confidence"]
    assert PendingRequest.__dataclass_params__.frozen is True


def test_pending_request_is_genuinely_immutable():
    """Mutating a ``PendingRequest`` field raises (frozen) — a remote client
    may hold the snapshot safely; the approver cannot mutate it behind the
    client's back."""
    pr = PendingRequest(
        request_id="r", action=_action(), reasoning="r", confidence=0.5
    )
    for attr in ("request_id", "action", "reasoning", "confidence"):
        try:
            setattr(pr, attr, "changed")
        except dataclasses.FrozenInstanceError:
            pass
        else:  # pragma: no cover - frozen must prevent this
            raise AssertionError(f"PendingRequest.{attr} is mutable")


# ---------------------------------------------------------------------------
# 8) Existing approvers and loop gating are unchanged
# ---------------------------------------------------------------------------


def test_terminal_approver_name_and_protocol_unchanged():
    """``TerminalApprover`` keeps its channel name and remains an ``Approver``
    (existing behaviour unchanged). No real stdin I/O here — that path is
    covered by the existing TerminalApprover tests; this only pins the channel
    name and protocol conformance the loop relies on."""
    approver = TerminalApprover(timeout_s=0.2)
    assert approver.name == "terminal"
    assert isinstance(approver, Approver)
    # The loop's gate contract depends only on these two attributes.
    assert callable(approver.request_approval)


def test_auto_approver_name_and_behavior_unchanged():
    """``AutoApprover`` keeps its channel name and always-approves contract."""
    approver = AutoApprover()
    assert approver.name == "auto"
    d = approver.request_approval(_action())
    assert d.is_approved is True


def test_queue_approver_default_timeout_is_finite():
    """The per-call default timeout is finite and non-negative (a misconfigured
    negative/inf timeout is a ``ValueError``, not an unbounded wait)."""
    assert QueueApprover().DEFAULT_TIMEOUT_S == 30.0
    import pytest

    with pytest.raises(ValueError):
        QueueApprover(timeout_s=-1)
    with pytest.raises(ValueError):
        QueueApprover(timeout_s=float("inf"))


# ---------------------------------------------------------------------------
# 9) Loop integration — the gate stays loop-owned; channel is "queue"
# ---------------------------------------------------------------------------


def test_assisted_queue_approve_dispatches_ai_action():
    """ASSISTED + an approving remote approver → the AI action dispatches with
    an ``APPROVAL`` audit event tagged ``channel == "queue"``. Proves the
    remote channel is a drop-in for the loop (which depends on the protocol
    only); the human responds from another thread."""
    backend = MockBackend()
    ai = tai._FakeAI(decision=_ai_decision())
    approver = QueueApprover(timeout_s=10.0)
    events: list[object] = []
    loop = tai._loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=ai,
        max_ticks=2,
        on_event=events.append,
    )

    def _respond() -> None:
        p = _wait_pending(approver, timeout=5.0)
        assert p is not None, "loop did not issue an approval request"
        assert approver.respond(p.request_id, ApprovalDecision.approved("remote ok")) is True

    rt = threading.Thread(target=_respond, daemon=True)
    rt.start()
    try:
        report = loop.run()
    finally:
        rt.join(timeout=3.0)
    assert report.actions_ok == 1, f"expected 1 ok, got {report.actions_ok}: {report.errors}"
    assert ai.decide_calls >= 1
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals, "expected an APPROVAL audit event"
    assert approvals[0].detail["approved"] is True
    assert approvals[0].detail["channel"] == "queue"
    assert approvals[0].detail["kind"] == "key_press"


def test_assisted_queue_reject_halts_episode():
    """ASSISTED + a rejecting remote approver → PAUSED, nothing dispatched, and
    the rejection reason is preserved in the audit detail."""
    backend = MockBackend()
    ai = tai._FakeAI(decision=_ai_decision())
    approver = QueueApprover(timeout_s=10.0)
    events: list[object] = []
    loop = tai._loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=ai,
        max_ticks=3,
        on_event=events.append,
    )

    def _respond() -> None:
        p = _wait_pending(approver, timeout=5.0)
        assert p is not None
        assert approver.respond(p.request_id, ApprovalDecision.rejected("remote no")) is True

    rt = threading.Thread(target=_respond, daemon=True)
    rt.start()
    try:
        report = loop.run()
    finally:
        rt.join(timeout=3.0)
    assert report.actions_ok == 0, f"rejected action still ran: {report.actions_ok}"
    assert report.actions_attempted == 0, "rejected action was dispatched"
    assert loop._machine.state is AgentState.PAUSED
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals, "expected an APPROVAL audit event on rejection"
    assert approvals[0].detail["approved"] is False
    assert approvals[0].detail["channel"] == "queue"
    assert approvals[0].detail["reason"] == "remote no"
    assert approver.pending() == ()


def test_assisted_queue_timeout_halts_episode():
    """ASSISTED + a remote approver that never answers → PAUSED (timeout). The
    human opens the page but never clicks approve/reject; the deadline elapses
    and the safe-halt fires with no input emitted."""
    backend = MockBackend()
    ai = tai._FakeAI(decision=_ai_decision())
    approver = QueueApprover(timeout_s=0.3)
    events: list[object] = []
    loop = tai._loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=ai,
        max_ticks=3,
        on_event=events.append,
    )
    report = loop.run()  # no responder: the 0.3 s deadline elapses and halts
    assert report.actions_ok == 0
    assert report.actions_attempted == 0
    assert loop._machine.state is AgentState.PAUSED
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals, "expected an APPROVAL audit event on timeout"
    assert approvals[0].detail["approved"] is False
    assert approvals[0].detail["reason"] == "timeout"
    assert approver.pending() == ()


def test_assisted_queue_cancel_halts_episode():
    """ASSISTED + a remote approver + a set ``cancel_event`` → PAUSED
    (interrupted). The F12 emergency stop while a remote approval is pending:
    the loop halts safely and dispatches nothing."""
    backend = MockBackend()
    ai = tai._FakeAI(decision=_ai_decision())
    cancel = threading.Event()
    cancel.set()  # pre-set: the wait sees it on its first slice
    approver = QueueApprover(timeout_s=30.0, cancel_event=cancel)
    events: list[object] = []
    loop = tai._loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=ai,
        max_ticks=3,
        on_event=events.append,
    )
    report = loop.run()
    assert report.actions_ok == 0
    assert report.actions_attempted == 0
    assert loop._machine.state is AgentState.PAUSED
    approvals = [e for e in events if e.event is EventType.APPROVAL]
    assert approvals, "expected an APPROVAL audit event on cancel"
    assert approvals[0].detail["approved"] is False
    assert approvals[0].detail["reason"] == "interrupted"


def test_queue_gate_does_not_fire_for_deterministic_actions():
    """ASSISTED + QueueApprover must NOT gate deterministic (non-AI) actions.
    The gate is AI-proposal-scoped; the queue approver is never consulted for a
    deterministic COMBAT batch, and nothing is queued in ``pending()``."""
    backend = MockBackend()
    approver = QueueApprover(timeout_s=10.0)
    machine = tai.GameStateMachine(
        policy=tai._StubDecisionPolicy({"target_hostile": True}),
        initial=AgentState.COMBAT,
        recovery_attempts=3,
        state_timeout_s=5.0,
    )
    loop = tai._loop(
        mode=OperationMode.ASSISTED,
        approver=approver,
        backend=backend,
        ai=None,  # no AI → the deterministic policy drives the batch
        machine=machine,
        max_ticks=2,
    )
    report = loop.run()
    assert report.actions_ok >= 1, f"deterministic actions blocked: {report.actions_ok}"
    assert approver.pending() == (), "queue approver consulted for non-AI actions"


def test_queue_gate_is_noop_outside_assisted():
    """The gate is a no-op outside ASSISTED — the queue is never touched, and
    the existing AUTONOMOUS path is unchanged (the AI action still dispatches)."""
    backend = MockBackend()
    ai = tai._FakeAI(decision=_ai_decision())
    approver = QueueApprover(timeout_s=10.0)
    loop = tai._loop(
        mode=OperationMode.AUTONOMOUS,
        approver=approver,
        backend=backend,
        ai=ai,
        max_ticks=2,
    )
    report = loop.run()
    assert report.actions_ok == 1, f"AUTONOMOUS path changed: {report.actions_ok}"
    assert approver.pending() == (), "queue approver consulted in AUTONOMOUS mode"


# ---------------------------------------------------------------------------
# 10) No unbounded growth: settled slots are removed (regression for the audit
#     finding "completed requests removed so storage does not grow")
# ---------------------------------------------------------------------------


def test_settled_slots_are_removed_no_unbounded_growth():
    """Every settled request (approved, rejected, timed-out) is removed from
    the internal slot store, so repeated approvals do not accumulate slots and
    replayed ids become unresolvable no-ops."""
    approver = QueueApprover(timeout_s=0.05)
    for i in range(5):
        box: dict[str, Any] = {}
        t = threading.Thread(
            target=lambda b=box: b.update(
                d=approver.request_approval(_action(), timeout_s=0.05)
            ),
            daemon=True,
        )
        t.start()
        t.join(timeout=2.0)
        assert box.get("d") is not None, f"request {i} did not return"
    assert approver.pending() == ()
    assert len(approver._slots) == 0, (
        f"settled slots not removed (unbounded growth): {len(approver._slots)}"
    )


# ---------------------------------------------------------------------------
# 11) Exception safety: an injected clock that raises after registration must
#     not leak the slot into pending() (audit Finding #2)
# ---------------------------------------------------------------------------


def test_clock_raising_after_registration_does_not_leak_slot():
    """Finding #2: if the injected ``now_fn`` raises *after* the request has
    been registered in ``_slots`` (but before the request is settled),
    ``request_approval`` must not exit with the slot still outstanding.

    The clock is allowed to fail on the *first* call (the deadline computation,
    which happens before registration) — that is a pre-registration failure with
    nothing to clean up. The contract under test is the post-registration path:
    the slot is registered, a later clock call inside the wait loop raises, and
    the exception propagates. On that path the slot must be removed from the
    store so it is neither visible via ``pending()`` nor resolvable via a
    later ``respond()`` — otherwise it lingers outstanding indefinitely.
    """
    calls = {"n": 0}

    def raising_now() -> float:
        calls["n"] += 1
        if calls["n"] >= 2:
            # First call (deadline, pre-registration) succeeded; every call
            # from inside the wait loop raises — the post-registration path.
            raise RuntimeError("injected clock failed")
        return 0.0

    approver = QueueApprover(timeout_s=30.0, now_fn=raising_now)
    with pytest.raises(RuntimeError, match="injected clock failed"):
        approver.request_approval(_action(), timeout_s=30.0)

    # The post-registration cleanup must have run: the slot is gone, so the
    # request is neither outstanding nor resolvable.
    assert approver.pending() == (), (
        f"slot leaked into pending() after the clock raised: "
        f"{[p.request_id for p in approver.pending()]!r}"
    )
    assert len(approver._slots) == 0, (
        f"slot leaked into the internal store: {list(approver._slots)!r}"
    )
    # And a late response for the leaked id is a no-op (nothing to settle).
    assert approver.respond("any-id", ApprovalDecision.approved("ok")) is False

def test_concurrent_notify_all_wakes_correct_waiter():
    """P2 (review #3): with two concurrent ``request_approval`` waiters
    outstanding, ``respond()`` must wake the *correct* waiter (the one whose
    request was just settled) — not an arbitrary one. With ``notify`` (not
    ``notify_all``) the wrong waiter could be woken first, re-check, find its
    slot unsettled, and go back to sleep, while the just-settled waiter is
    still sleeping (delayed by one full slice). ``notify_all`` wakes every
    waiter so the settled one returns immediately.

    This test settles the *second* request (started last) while the first is
    still waiting, asserts the second prompt returns promptly with the correct
    decision, and that the first is still outstanding (not accidentally
    settled by the wrong-wake).
    """
    approver = QueueApprover(timeout_s=30.0)
    box1: dict[str, Any] = {}
    box2: dict[str, Any] = {}
    t1 = threading.Thread(
        target=lambda: box1.update(
            d=approver.request_approval(_action(), reasoning="first", timeout_s=30.0)
        ),
        daemon=True,
    )
    t2 = threading.Thread(
        target=lambda: box2.update(
            d=approver.request_approval(_action(), reasoning="second", timeout_s=30.0)
        ),
        daemon=True,
    )
    t1.start()
    t2.start()
    try:
        ids = _wait_outstanding(approver, 2)
        assert ids is not None, "expected two outstanding requests"
        # Identify which id belongs to the "second" request (started last).
        p_first = next(p for p in approver.pending() if p.reasoning == "first")
        p_second = next(p for p in approver.pending() if p.reasoning == "second")
        # Settle the *second* (later-registered) request while the first is
        # still waiting. With ``notify`` (bug) the wrong waiter could be woken
        # and the second prompt would be delayed by one slice; with
        # ``notify_all`` (fix) the second prompt returns immediately.
        t0 = time.monotonic()
        assert approver.respond(p_second.request_id, ApprovalDecision.approved("ok2")) is True
        # Bound the wait: the correct waiter must be woken promptly (well under
        # the 30 s timeout, and well under several slices). 5.0 s is generous
        # enough to avoid CI flakiness while catching a real hang.
        t2.join(timeout=5.0)
        elapsed = time.monotonic() - t0
        assert not t2.is_alive(), f"second prompt did not return (hung); elapsed {elapsed:.2f}s"
        assert box2.get("d") is not None and box2["d"].is_approved, (
            f"second decision wrong: {box2.get('d')!r}"
        )
        assert box2["d"].reason == "ok2"
        # The first request must still be outstanding (not accidentally settled
        # by waking the wrong waiter).
        assert p_first.request_id in _outstanding_ids(approver), (
            "first request was accidentally settled — wrong waiter was woken"
        )
        # Settle the first now.
        assert approver.respond(p_first.request_id, ApprovalDecision.rejected("no1")) is True
    finally:
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
    assert box1.get("d") is not None and not box1["d"].is_approved
    assert box1["d"].reason == "no1"
    assert approver.pending() == ()
