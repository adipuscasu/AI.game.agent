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
  Timeout contract: ``None`` → instance default (30 s); ``0`` → non-blocking
  poll (honours a reply already buffered, rejects immediately otherwise);
  positive → wait that many seconds; negative → ``ValueError``. An injected
  ``cancel_event`` (``threading.Event``), when set while a wait is pending,
  unblocks it immediately and is classified as an ``"interrupted"`` rejection.
* :class:`AutoApprover` — a non-blocking policy stub for tests, CI, and
  "trust the AI" runs; always approves.

Design notes:

* **Dependency-free.** Only stdlib (``typing``, ``time``, ``sys``,
  ``dataclasses``; ``msvcrt`` imported lazily, only on Windows consoles).
  Importing this module in a bare venv is always safe, matching the rest
  of the ``actions`` package.
* **Console-aware.** On a Windows console (``CONIN$``) the reply is read
  key-by-key through :class:`ConsoleInputReader` (``kbhit``/``getwch``)
  under the same deadline/cancel slices as the fd paths, because
  ``select`` rejects non-socket fds and a line-buffered ``os.read``
  cannot be deadline-bounded there.
* **Fail-safe.** Every path that cannot obtain an affirmative answer
  (EOF on stdin, ``KeyboardInterrupt``, timeout, unparsable reply) returns
  ``Rejected``. The only way an action executes is an explicit
  affirmative answer — never an ambiguous one.
* **No I/O on import.** The terminal read happens inside
  :meth:`TerminalApprover.request_approval`, not at module load.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import threading
import time
from typing import Literal, Protocol, runtime_checkable

__all__ = [
    "Approver",
    "AutoApprover",
    "ApprovalDecision",
    "ConsoleInputReader",
    "DecisionOutcome",
    "MsvcrtConsoleReader",
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
class ConsoleInputReader(Protocol):
    """One key at a time from an interactive console (Windows ``CONIN$``).

    The reference implementation is :class:`MsvcrtConsoleReader`; tests
    inject a scripted fake of this shape so the approval loop stays
    deterministic on hosts without a desktop console.
    """

    def kbhit(self) -> bool:
        """Whether a key is pending — a non-blocking probe, never blocks."""
        ...

    def getwch(self) -> str:
        """The next key (call only when :meth:`kbhit` is true).

        May raise ``EOFError`` (Ctrl-Z) or ``KeyboardInterrupt`` (Ctrl-C);
        the approval loop treats both as fail-safe rejections.
        """
        ...


class MsvcrtConsoleReader:
    """A :class:`ConsoleInputReader` backed by ``msvcrt`` (Windows only).

    ``msvcrt.kbhit()`` reports pending keys without blocking and
    ``msvcrt.getwch()`` returns the next key, so a reply can be assembled
    one key at a time under a deadline — exactly what ``select`` (which
    rejects non-socket fds with WinError 10038) and a non-blocking
    ``os.read`` (line-buffered ``CONIN$`` blocks until Enter) cannot do
    on a Windows console.
    """

    def __init__(self) -> None:
        import msvcrt  # Windows-only module: ImportError on other hosts

        self._msvcrt = msvcrt

    def kbhit(self) -> bool:
        return self._msvcrt.kbhit()

    def getwch(self) -> str:
        return self._msvcrt.getwch()


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
        if timeout_s is not None and (
            not math.isfinite(timeout_s) or timeout_s < 0
        ):
            raise ValueError(
                f"timeout_s must be a finite value >= 0, got {timeout_s!r}"
            )
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
        timeout_s: float | None = DEFAULT_TIMEOUT_S,
        now_fn: time | None = None,  # type: ignore[valid-type]
        cancel_event: threading.Event | None = None,
        reader: ConsoleInputReader | None = None,
    ) -> None:
        # ``None`` is the module's "use the instance default" sentinel (the
        # same one ``request_approval`` accepts), not an error: the default
        # ASSISTED run path constructs the approver without a flag.
        self._timeout_s = (
            self.DEFAULT_TIMEOUT_S if timeout_s is None else float(timeout_s)
        )
        if not math.isfinite(self._timeout_s) or self._timeout_s < 0:
            raise ValueError(
                f"timeout_s must be a finite value >= 0, got {timeout_s!r}"
            )
        # Injectable clock for deterministic timeout tests.
        self._now = now_fn if now_fn is not None else time.monotonic
        # Optional cancellation signal (e.g. the F12 emergency stop). When
        # set while a wait is pending, the wait is unblocked immediately and
        # classified as an "interrupted" rejection — never an approval.
        self._cancel_event = cancel_event
        # Optional console input reader (kbhit/getwch). An explicit
        # injection always wins; otherwise an interactive Windows console
        # is auto-detected at read time (see _console_reader).
        self._reader = reader

    def request_approval(
        self,
        action: object,
        *,
        reasoning: str = "",
        confidence: float | None = None,
        timeout_s: float | None = None,
    ) -> ApprovalDecision:
        timeout = self._timeout_s if timeout_s is None else timeout_s
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError(
                f"timeout_s must be a finite value >= 0, got {timeout!r}"
            )

        prompt = self._build_prompt(action, reasoning=reasoning, confidence=confidence)
        print(prompt, end="", flush=True)

        # Read one line from stdin with a soft timeout. On non-interactive
        # stdin the read raises EOFError immediately -> reject (fail-safe).
        line, timed_out = self._read_line(timeout)
        if line is None:
            if self._cancel_event is not None and self._cancel_event.is_set():
                # Emergency stop (F12) interrupted the wait: classify it
                # distinctly from a plain timeout for the audit trail.
                return ApprovalDecision.rejected("interrupted")
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
        # Surface every field that deviates from the Action default, without
        # importing the Action contract here (the module stays dependency-free).
        # Defaults mirror ``_DEFAULTS`` in actions/action.py: showing only the
        # non-default values keeps the prompt concise while ensuring no
        # meaningful parameter (modifiers, relative, clicks, ...) is hidden.
        _FIELDS = (
            ("key", None),
            ("modifiers", ()),
            ("x", None),
            ("y", None),
            ("relative", False),
            ("button", "left"),
            ("clicks", 1),
            ("scroll", None),
            ("duration_ms", 0),
            ("delay_ms", 0),
        )
        for field, default in _FIELDS:
            value = getattr(action, field, None)
            if value == default:
                continue
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

        A ``timeout`` of ``0`` is a non-blocking poll (one ``select`` probe),
        never an unbounded read. A cancellable ``cancel_event`` routes the
        wait through :meth:`_read_line_cancellable`. An interactive console
        (Windows ``CONIN$``) routes through :meth:`_read_line_console`,
        whose key-polling loop enforces the deadline and checks the cancel
        signal itself.
        """
        reader = self._console_reader()
        if reader is not None:
            # Console input is line-buffered and blocking: neither select
            # (WinError 10038) nor a non-blocking os.read can enforce a
            # deadline on it, so the reader's kbhit loop owns the wait.
            return self._read_line_console(timeout, reader)
        if self._cancel_event is not None:
            return self._read_line_cancellable(timeout)
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

        if select is not None:
            # ``select.select(fd, timeout=0)`` is a non-blocking poll: a
            # buffered reply is honoured, an empty pipe returns immediately.
            # (The old ``timeout > 0`` guard turned ``timeout=0`` into an
            # unbounded ``readline`` — a hang on a live terminal.)
            try:
                ready, _, _ = select.select([fd], [], [], timeout)
            except OSError:
                # Windows: ``select`` only accepts sockets and raises for
                # pipes / console fds (WinError 10038). Fall back to a
                # bounded fd poll — the fail-safe still applies.
                return self._read_line_bounded(timeout, fd)
            if not ready:
                return (None, True)
        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt, AttributeError):
            # AttributeError: stdin is None / has no readline — a fail-safe
            # rejection, never an approval.
            return (None, False)
        return (line.rstrip("\n") if line else None, False)

    def _console_reader(self) -> ConsoleInputReader | None:
        """Resolve the console reader for the current ``sys.stdin``.

        An explicitly injected ``reader`` always wins. Otherwise the reader
        is used only when stdin is an interactive console (``isatty()``)
        and ``msvcrt`` is importable (Windows): pipe / redirect fds keep
        the existing ``select`` / ``os.read`` paths, and non-Windows hosts
        have no ``msvcrt`` (ImportError → ``None``).
        """
        if self._reader is not None:
            return self._reader
        stdin = sys.stdin
        if stdin is None:
            return None
        try:
            if not stdin.isatty():
                return None
        except (AttributeError, ValueError, OSError):
            return None
        try:
            return MsvcrtConsoleReader()
        except (ImportError, OSError, ValueError):
            # ``msvcrt`` is Windows-only; elsewhere the fd paths apply.
            return None

    def _read_line_console(
        self, timeout: float, reader: ConsoleInputReader
    ) -> tuple[str | None, bool]:
        """Read one reply from an interactive console, key by key.

        Polls the reader's pending-key probe in the same deadline slices as
        :meth:`_read_line_bounded`: a reply that *completes* (Enter
        pressed) before the deadline is honoured; a reply that does not is
        a **timeout** rejection (fail-safe — a half-typed line is never an
        approval, and a late ``y`` after the deadline cannot bypass it); a
        set ``cancel_event`` unblocks the wait on the next slice, so an
        emergency stop stays responsive. ``timeout=0`` is a single
        non-blocking probe. No worker threads are involved (bounded
        durations, safety contract).
        """
        cancel = self._cancel_event
        buf = ""
        deadline = self._now() + timeout
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    return (None, True)
                # Evaluate the deadline **before** consuming a pending key.
                # A reply still pending once the deadline has passed is a
                # timeout rejection (fail-safe): a late ``y`` typed at or past
                # the deadline must not be honoured, and a half-typed line is
                # never an approval. (A reply already buffered *before* the
                # deadline is still honoured — e.g. the ``timeout_s=0`` poll.)
                if self._now() >= deadline:
                    return (None, True)
                if reader.kbhit():
                    try:
                        ch = reader.getwch()
                    except (EOFError, KeyboardInterrupt, OSError, ValueError):
                        # Ctrl-Z / Ctrl-C / a broken console handle: a
                        # fail-safe rejection — never an approval, never a
                        # crash.
                        return (None, False)
                    buf += ch
                    if ch in "\r\n":
                        break
                    continue
                time.sleep(min(0.01, deadline - self._now()))
        except KeyboardInterrupt:
            return (None, False)
        return (buf.rstrip("\r\n"), False)

    def _read_line_cancellable(
        self, timeout: float
    ) -> tuple[str | None, bool]:
        """Read one line with a cancellable deadline.

        Polls ``select`` in 50 ms slices, checking ``self._cancel_event``
        before every slice, so an emergency stop (F12) unblocks the wait
        within one slice instead of waiting for the full timeout. A
        ``timeout`` of ``0`` performs a single non-blocking poll (a
        buffered reply is still honoured). ``select``-unavailable platforms
        fall back to :meth:`_read_line_bounded` (bounded, but not
        cancellation-responsive — the fail-safe still applies).
        """
        fd = None
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, ValueError, OSError):
            fd = None

        if fd is None:
            # No real fd (e.g. StringIO in tests): read directly.
            if self._cancel_event.is_set():
                return (None, True)
            try:
                line = sys.stdin.readline()
            except (EOFError, KeyboardInterrupt, AttributeError):
                return (None, False)
            return (line.rstrip("\n") if line else None, False)

        try:
            import select
        except ImportError:  # pragma: no cover - Windows
            select = None
        if select is None:
            return self._read_line_bounded(timeout, fd)

        deadline = self._now() + timeout
        POLL_S = 0.05
        while True:
            if self._cancel_event.is_set():
                return (None, True)
            poll = POLL_S if timeout > 0 else 0.0
            try:
                ready, _, _ = select.select([fd], [], [], poll)
            except OSError:
                # Windows: ``select`` rejects non-socket fds (WinError
                # 10038). Bounded fallback — the fail-safe still applies.
                return self._read_line_bounded(timeout, fd)
            if ready:
                break
            if self._now() >= deadline:
                return (None, True)

        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt, AttributeError):
            return (None, False)
        return (line.rstrip("\n") if line else None, False)

    def _read_line_bounded(self, timeout: float, fd: int) -> tuple[str | None, bool]:
        """Bounded read for platforms where ``select`` cannot be used.

        Windows ``select`` rejects non-socket fds (pipes / console) with
        ``OSError`` (WinError 10038), so the prompt path must not crash
        there. The read polls the fd in non-blocking slices **on the main
        thread** up to the deadline: a buffered reply is honoured, a reply
        that only arrives *after* the deadline is a **timeout** rejection
        (fail-safe — matches the ``select`` path), and nothing is left
        behind: the main thread is always released by the deadline and no
        blocked per-prompt worker accumulates (bounded durations, safety
        contract). A set ``cancel_event`` unblocks the poll on the next
        slice, so the emergency stop stays responsive on this path too.

        A ``timeout`` of ``0`` is a single non-blocking probe. If the fd
        cannot be made non-blocking (and ``select`` cannot watch it
        either), the read degrades to an immediate ``"timeout"``
        rejection — the fail-safe still holds (never approve, never
        hang, never spawn a worker); only a late reply is not honoured.
        """
        import os

        try:
            os.set_blocking(fd, False)
        except (OSError, ValueError, AttributeError, NotImplementedError):
            # The fd cannot be polled non-blocking and ``select`` cannot
            # watch it: the fail-safe is an immediate timeout rejection
            # (never approve, never hang, never spawn a worker).
            return (None, True)

        cancel = self._cancel_event
        buf = b""
        deadline = self._now() + timeout
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    return (None, True)
                try:
                    c = os.read(fd, 1)
                except BlockingIOError:
                    c = b""  # no data yet (EAGAIN) — wait until the deadline
                except (OSError, ValueError):
                    # fd broken / closed under us — a fail-safe rejection.
                    return (None, False)
                if c:
                    buf += c
                    if c == b"\n":
                        break
                    continue
                # ``c == b""`` on a non-blocking pipe means "no data now"
                # (or EOF with the writer closed): wait until the deadline.
                if self._now() >= deadline:
                    return (None, True)
                time.sleep(min(0.01, deadline - self._now()))
        finally:
            try:
                os.set_blocking(fd, True)
            except (OSError, ValueError, AttributeError, NotImplementedError):
                pass
        line = buf.decode("utf-8", "replace")
        return (line.rstrip("\n"), False)



def _io_error_stub() -> type:  # pragma: no cover - defensive import shim
    return IOError
