"""Assisted-approval mechanism (Phase 6).

AGENTS.md §13 ("Human-in-the-Loop Mode") and the high-level plan's
Phase 6 target are the same flow::

    Observe -> Analyze -> Suggest action -> Human approval -> Execute

The :class:`Approver` protocol is that "human approval" step, abstracted
from the transport so the loop never imports ``input()`` or a terminal
library. The loop (the only object that dispatches actions) consults the
approver **before** executing an AI-proposed action when running under
``ASSISTED``; a rejection halts the episode safely (see
:mod:`ai_game_agent.agent.loop`).

Two implementations ship:

* :class:`TerminalApprover` — the default human channel. Blocks on ``stdin``
  with a clear prompt and a timeout, and treats non-interactivity / timeout
  as a **rejection** (fail-safe: an unanswered proposal is never executed).
* :class:`AutoApprover` — a non-blocking policy stub for tests, CI, and
  "trust the AI" runs; always approves.

Design notes:

* **Dependency-free.** Only stdlib (``typing``, ``time``, ``sys``,
  ``dataclasses``). Importing this module in a bare venv is always safe,
  matching the rest of the ``actions`` package.
* **Fail-safe.** Every path that cannot obtain an affirmative answer
  (EOF on stdin, ``KeyboardInterrupt``, timeout, unparsable reply) returns
  ``Rejected``. The only way an action executes is an explicit
  affirmative answer — never an ambiguous one.
* **No I/O on import.** The terminal read happens inside
  :meth:`TerminalApprover.request_approval`, not at module load.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
from typing import Literal, Protocol, runtime_checkable

__all__ = [
    "Approver",
    "AutoApprover",
    "ApprovalDecision",
    "DecisionOutcome",
    "RejectionReason",
    "TerminalApprover",
]

#: The two outcomes a single approval request can have.
DecisionOutcome = Literal["approved", "rejected"]


@dataclasses.dataclass(frozen=True)
class ApprovalDecision:
    """The outcome of one approval request.

    ``outcome`` is the closed vocabulary the loop programs against;
    ``reason`` carries a human/audit-trail explanation (e.g. ``"timeout"``,
    ``"no-input-channel"``, ``"user-rejected"``).
    """

    outcome: DecisionOutcome
    reason: str = ""

    @classmethod
    def approved(cls, reason: str = "approved") -> ApprovalDecision:
        return cls(outcome="approved", reason=reason)

    @classmethod
    def rejected(cls, reason: str = "rejected") -> ApprovalDecision:
        return cls(outcome="rejected", reason=reason)

    @property
    def is_approved(self) -> bool:
        return self.outcome == "approved"

    @property
    def is_rejected(self) -> bool:
        return self.outcome == "rejected"


#: Machine-readable reason tags for rejections (audit-friendly).
RejectionReason = Literal[
    "user-rejected",
    "no-input-channel",
    "timeout",
    "interrupted",
    "unparsable-reply",
]


@runtime_checkable
class Approver(Protocol):
    """Human-in-the-loop approval channel (Phase 6).

    The loop calls :meth:`request_approval` exactly once per AI-proposed
    action when running under ``ASSISTED``. Implementations must be
    thread-confined (one approval in flight) and must **never raise** for a
    "no" — they return :meth:`ApprovalDecision.rejected`. A raised exception
    is reserved for genuine misconfiguration.

    Contract:
        * An affirmative answer returns :meth:`ApprovalDecision.approved`.
        * Anything else — a "no", a blank line, EOF, a timeout, a
          ``KeyboardInterrupt``, an unparsable reply — returns
          :meth:`ApprovalDecision.rejected` with a :data:`RejectionReason`.
        * ``request_approval`` blocks for at most ``timeout_s`` seconds
          (the implementation owns the clock; tests inject ``now_fn``).
    """

    #: Human-readable channel name (e.g. ``"terminal"``, ``"auto"``) for
    #: the approval event's audit detail.
    name: str

    def request_approval(
        self,
        action: object,
        *,
        reasoning: str = "",
        confidence: float | None = None,
        timeout_s: float | None = None,
    ) -> ApprovalDecision:
        """Ask the human to approve one proposed action.

        Args:
            action: The :class:`~ai_game_agent.actions.action.Action` being
                proposed. Typed as ``object`` so this module stays
                dependency-free; the loop passes an ``Action``.
            reasoning: The model's rationale (from ``AIDecision.reasoning``),
                surfaced in the prompt.
            confidence: Model-reported confidence in ``[0, 1]`` (optional),
                surfaced in the prompt.
            timeout_s: Per-call timeout override. ``None`` falls back to the
                implementation default.

        Returns:
            An :class:`ApprovalDecision`.

        Raises:
            ValueError: If the caller passes a negative ``timeout_s``.
        """
        ...


class AutoApprover:
    """A non-interactive approver that always approves.

    Use in tests, CI, and explicit "trust the AI" runs. It is the reference
    implementation of the protocol and the cheapest way to exercise the
    assisted gate without a human.
    """

    name: str = "auto"

    def __init__(self, reason: str = "auto-approved (AutoApprover)") -> None:
        self._reason = reason

    def request_approval(
        self,
        action: object,
        *,
        reasoning: str = "",
        confidence: float | None = None,
        timeout_s: float | None = None,
    ) -> ApprovalDecision:
        if timeout_s is not None and timeout_s < 0:
            raise ValueError(f"timeout_s must be >= 0, got {timeout_s!r}")
        return ApprovalDecision.approved(self._reason)


class TerminalApprover:
    """The default human channel: block on ``stdin`` for a y/N answer.

    Fail-safe by construction:

    * A non-interactive stdin (``EOFError`` / ``input`` immediately
      raises) is a **rejection**, not an error — an agent running
      headless must never act on an unacknowledged proposal.
    * A timeout (the prompt is shown, then we wait up to ``timeout_s``)
      is a **rejection**.
    * ``KeyboardInterrupt`` is a **rejection** (the user is bailing).
    * Blank / unparsable replies are **rejections** (ambiguous = no).

    Only an explicit affirmative (``y`` / ``yes`` / ``approve`` / ``ok``,
    case-insensitive) approves.
    """

    name: str = "terminal"

    #: Default per-call timeout. The plan's [Execute]/[Reject]/[Pause]
    #: implies a human is watching; a generous default avoids a hung
    #: prompt swallowing a CI run, while still giving a human time to answer.
    DEFAULT_TIMEOUT_S: float = 30.0

    def __init__(
        self,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        now_fn: time | None = None,  # type: ignore[valid-type]
    ) -> None:
        if timeout_s < 0:
            raise ValueError(f"timeout_s must be >= 0, got {timeout_s!r}")
        self._timeout_s = timeout_s
        # Injectable clock for deterministic timeout tests.
        self._now = now_fn if now_fn is not None else time.monotonic

    def request_approval(
        self,
        action: object,
        *,
        reasoning: str = "",
        confidence: float | None = None,
        timeout_s: float | None = None,
    ) -> ApprovalDecision:
        timeout = self._timeout_s if timeout_s is None else timeout_s
        if timeout < 0:
            raise ValueError(f"timeout_s must be >= 0, got {timeout!r}")

        prompt = self._build_prompt(action, reasoning=reasoning, confidence=confidence)
        print(prompt, end="", flush=True)

        # Read one line from stdin with a soft timeout. On non-interactive
        # stdin the read raises EOFError immediately -> reject (fail-safe).
        line, timed_out = self._read_line(timeout)
        if line is None:
            reason = "timeout" if timed_out else "no-input-channel"
            return ApprovalDecision.rejected(reason)
        reply = line

        normalized = reply.strip().lower()
        if normalized in {"y", "yes", "approve", "ok", "go", "1"}:
            return ApprovalDecision.approved("user-approved")
        if normalized in {"n", "no", "reject", "0", ""}:
            return ApprovalDecision.rejected("user-rejected")
        return ApprovalDecision.rejected("unparsable-reply")

    # ---- internals ---------------------------------------------------------

    def _build_prompt(
        self,
        action: object,
        *,
        reasoning: str,
        confidence: float | None,
    ) -> str:
        """Render the proposal in a compact, one-glance-able form."""
        kind = getattr(action, "kind", None)
        kind = getattr(kind, "value", kind) if kind is not None else "<action>"
        lines = [
            "\u26a0\ufe0f  AI proposes an action (ASSISTED mode) \u2014 approve to execute:",
            f"    action: {kind}",
        ]
        # Surface the key identifying fields for the common kinds, without
        # importing the Action contract here (the module stays dependency-free).
        for field in ("key", "x", "y", "button", "scroll", "duration_ms"):
            value = getattr(action, field, None)
            if value is not None:
                lines.append(f"    {field}: {value}")
        if reasoning:
            lines.append(f"    reasoning: {reasoning}")
        if confidence is not None:
            lines.append(f"    confidence: {confidence:.2f}")
        lines.append("    approve this action? [y/N] ")
        return "\n".join(lines)

    def _read_line(self, timeout: float) -> tuple[str | None, bool]:
        """Read one line from ``sys.stdin`` with a soft timeout.

        Returns ``(line, timed_out)``: ``line`` is the reply without a
        trailing newline, or ``None`` when no line is available (EOF,
        timeout, or interrupt); ``timed_out`` is ``True`` only when the
        deadline expired with no input (as opposed to an immediate EOF).
        ``KeyboardInterrupt`` is swallowed here so the caller can classify
        it as a rejection (the agent must not crash).
        """
        fd = None
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, ValueError, OSError):
            fd = None

        if fd is None:
            # Not a real file descriptor (e.g. a StringIO in tests): read
            # directly, no timeout (tests either pre-buffer or EOF).
            try:
                line = sys.stdin.readline()
            except (EOFError, KeyboardInterrupt, AttributeError):
                # AttributeError: stdin is None or has no readline (detached
                # session) — a fail-safe rejection, never an approval.
                return (None, False)
            return (line.rstrip("\n") if line else None, False)

        try:
            import select  # Unix: works on any fd; Windows: sockets only.
        except ImportError:  # pragma: no cover
            select = None

        if select is not None and timeout > 0:
            try:
                ready, _, _ = select.select([fd], [], [], timeout)
            except OSError:
                # Windows: ``select`` only accepts sockets and raises for
                # pipes / console fds (WinError 10038). Fall back to a
                # bounded threaded read — the fail-safe still applies.
                return self._read_line_bounded(timeout)
            if not ready:
                return (None, True)
        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt, AttributeError):
            # AttributeError: stdin is None / has no readline — a fail-safe
            # rejection, never an approval.
            return (None, False)
        return (line.rstrip("\n") if line else None, False)

    def _read_line_bounded(self, timeout: float) -> tuple[str | None, bool]:
        """Bounded read for platforms where ``select`` cannot be used.

        Windows ``select`` rejects non-socket fds (pipes / console) with
        ``OSError`` (WinError 10038), so the prompt path must not crash
        there. Read one line on a daemon worker thread and wait for it for
        at most ``timeout`` seconds on the main thread. A reply that only
        arrives *after* the deadline is a **timeout** rejection (fail-safe)
        — it is not honoured, matching the ``select`` path. EOF / interrupt
        / no-readline are fail-safe rejections. The worker is a daemon, so
        it never blocks process exit; the main thread is always released by
        the deadline (bounded durations, per the safety contract).
        """
        done = threading.Event()
        result: dict[str, str | None] = {}

        def _do_read() -> None:
            try:
                line = sys.stdin.readline()
            except (EOFError, KeyboardInterrupt, AttributeError):
                line = None
            result["line"] = line
            done.set()

        worker = threading.Thread(target=_do_read, daemon=True, name="approve-read")
        worker.start()
        if not done.wait(timeout):
            # Deadline expired with no reply: a timeout rejection (fail-safe).
            # The worker is left to finish (daemon); its late reply is ignored.
            return (None, True)
        line = result["line"]
        return (line.rstrip("\n") if line else None, False)


def _io_error_stub() -> type:  # pragma: no cover - defensive import shim
    return IOError
