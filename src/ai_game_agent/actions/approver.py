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

* **Dependency-free.** Only stdlib (``os``, ``sys``, ``typing``, ``time``,
  ``dataclasses``, ``threading``, ``math``; ``msvcrt`` imported lazily, only
  on Windows consoles). Importing this module in a bare venv is always
  safe, matching the rest of the ``actions`` package.
* **Console-aware.** On a Windows console (``CONIN$``) the reply is read
  key-by-key through :class:`ConsoleInputReader` (``kbhit``/``getwch``),
  because a line-buffered read cannot be deadline-bounded there; the fd
  paths use a deadline-strict non-blocking read (``select`` is not used
  at all, so the WinError 10038 socket-only limitation never applies).
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

        A ``timeout`` of ``0`` is a single bounded non-blocking poll (a
        fully buffered reply is still honoured), never an unbounded read.
        The fd path routes through :meth:`_read_line_fd`, a deadline-strict
        non-blocking read that never calls ``readline`` and never spawns a
        worker thread. An interactive console (Windows ``CONIN$``) routes
        through :meth:`_read_line_console`, whose key-polling loop enforces
        the deadline and checks the cancel signal itself.
        """
        reader = self._console_reader()
        if reader is not None:
            # Console input is line-buffered and blocking: a raw read cannot
            # be deadline-bounded there, so the reader's kbhit loop owns
            # the wait.
            return self._read_line_console(timeout, reader)
        fd = None
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, ValueError, OSError):
            fd = None

        if fd is None:
            # Not a real file descriptor (e.g. a StringIO in tests): read
            # directly, no timeout (tests either pre-buffer or EOF).
            if self._cancel_event is not None and self._cancel_event.is_set():
                return (None, True)
            try:
                line = sys.stdin.readline()
            except (EOFError, KeyboardInterrupt, AttributeError):
                # AttributeError: stdin is None or has no readline (detached
                # session) — a fail-safe rejection, never an approval.
                return (None, False)
            return (line.rstrip("\n") if line else None, False)
        return self._read_line_fd(timeout, fd)

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
        :meth:`_read_line_fd`: a reply that *completes* (Enter
        pressed) before the deadline is honoured; a reply that does not is
        a **timeout** rejection (fail-safe — a half-typed line is never an
        approval, and a late ``y`` after the deadline cannot bypass it); a
        set ``cancel_event`` unblocks the wait on the next slice, so an
        emergency stop stays responsive. ``timeout=0`` is a single
        non-blocking probe. No worker threads are involved (bounded
        durations, safety contract).
        """
        cancel = self._cancel_event

        if timeout <= 0:
            # A genuine non-blocking probe — the documented contract, and the
            # behaviour the fd path already honours on its ``timeout<=0``
            # poll (see :meth:`_read_line_fd`). It never blocks or sleeps, so
            # ``timeout=0`` stays a single-shot probe: a reply already
            # available is consumed and honoured; a pending key that is not
            # yet available is a single non-blocking timeout. Cancellation
            # already signalled wins (the acceptance boundary), so a pending
            # stop classifies as ``"interrupted"`` rather than a timeout.
            if cancel is not None and cancel.is_set():
                return (None, True)
            try:
                buf = ""
                while True:
                    if cancel is not None and cancel.is_set():
                        return (None, True)
                    if not reader.kbhit():
                        # A key (or the terminating Enter) is not yet
                        # buffered. A non-blocking probe must never block on
                        # ``getwch()`` waiting for it, and a half-typed line is
                        # never an approval — so this is a single-shot timeout,
                        # not a wait.
                        return (None, True)
                    ch = reader.getwch()
                    buf += ch
                    if ch in "\r\n":
                        break
            except (EOFError, KeyboardInterrupt, OSError, ValueError):
                # Ctrl-Z / Ctrl-C / a broken console handle: a fail-safe
                # rejection — never an approval, never a crash.
                return (None, False)
            # Acceptance boundary: a cancel that landed while the reply was
            # being consumed wins (fail-safe; never commit a buffered approval
            # after a stop). Mirrors the fd ``timeout<=0`` boundary.
            if cancel is not None and cancel.is_set():
                return (None, True)
            return (buf.rstrip("\r\n"), False)

        # timeout > 0: bounded wait, re-polling in ~10 ms slices.
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

    def _read_line_fd(
        self, timeout: float, fd: int
    ) -> tuple[str | None, bool]:
        """Read one reply from a real fd; deadline-strict and cancel-responsive.

        The fd is driven non-blocking and read byte-by-byte with ``os.read``
        (never ``readline``, so a lone byte cannot hang; no worker threads).
        The partial line is carried across poll slices, so a reply whose
        bytes arrive in more than one slice is still reassembled.
        A reply is honoured **only** if its terminating newline is consumed
        within the allowed window — a line that completes after the deadline
        is a timeout rejection (fail-safe: a late ``y`` cannot bypass the
        deadline).

        * ``timeout <= 0``: a single bounded non-blocking poll — a line
          already fully buffered (through its newline) is honoured; an
          incomplete line, an EAGAIN read, or EOF is a timeout / no-channel
          rejection. Never blocks and never spins.
        * ``timeout > 0``: re-polls in ~10 ms slices until the deadline, and
          a set ``cancel_event`` unblocks the wait on the next slice (and at
          the acceptance boundary) — classified by the caller as
          ``"interrupted"``.
        * EOF (the writer closed the stream) returns promptly as a
          no-channel rejection: no more data can ever arrive, so waiting
          out the deadline is pointless.

        Reads are non-blocking, so an "empty" read on a live pipe means
        "not yet" and is re-polled; the deadline bounds the wait. The fd's
        blocking mode is captured and restored; a restore failure is
        swallowed (fail-closed) so it can never mask the result.
        """
        import os

        def _blocking_mode(fd: int) -> bool | None:
            try:
                return os.get_blocking(fd)
            except (OSError, ValueError, AttributeError, NotImplementedError):
                return None

        prev = _blocking_mode(fd)
        if prev is None:
            # The fd cannot even be probed: no usable channel — the caller
            # classifies this as a rejection (fail-safe).
            return (None, False)
        try:
            os.set_blocking(fd, False)
        except (OSError, ValueError, AttributeError, NotImplementedError):
            return (None, False)

        cancel = self._cancel_event
        now = self._now

        # Hard cap on how much of a not-yet-complete line we are willing to
        # buffer. A real approval reply is tiny; this bounds the partial
        # buffer's memory growth so a continuous, newline-free stream cannot
        # balloon it (and cannot spin the reader past the deadline).
        max_partial = 64 * 1024

        def _cancelled() -> bool:
            return cancel is not None and cancel.is_set()

        def _read_until_newline(
            partial: bytearray,
            deadline: float | None = None,
        ) -> tuple[bytes | None, str]:
            """Consume available bytes into ``partial`` until a newline.

            Bounded two ways: ``deadline`` (when given) stops the loop once
            the approval deadline is reached, and ``max_partial`` caps the
            bytes buffered, so a continuously-readable stream can neither
            monopolize the reader nor grow the buffer without bound. With
            neither bound active (``deadline`` is ``None``) the loop is
            still bounded by the byte cap plus ``BlockingIOError``/EOF on an
            idle pipe.

            Returns ``(line, "line")`` once a full line (through its
            ``\\n``) is assembled — ``line`` holds the bytes and
            ``partial`` is reset; ``(None, "eof")`` when the writer closed
            the stream before a newline; ``(None, "incomplete")`` when no
            more data is available yet (EAGAIN), the deadline was reached,
            or the byte cap was hit — ``partial`` keeps the bytes consumed
            so far and is carried into the next slice; ``(None, "error")``
            on a hard fd error.
            """
            while True:
                if deadline is not None and now() >= deadline:
                    return None, "incomplete"
                if len(partial) >= max_partial:
                    return None, "incomplete"
                try:
                    b = os.read(fd, 1)
                except BlockingIOError:
                    return None, "incomplete"
                except (OSError, ValueError):
                    return None, "error"
                if not b:
                    return None, "eof"
                partial.extend(b)
                if b == b"\n":
                    line = bytes(partial)
                    del partial[:]
                    return line, "line"

        def _done(line: bytes) -> tuple[str | None, bool]:
            return (line.decode("utf-8", "replace").rstrip("\n"), False)

        try:
            partial = bytearray()
            if timeout <= 0:
                line, status = _read_until_newline(partial)
                if status == "line" and line is not None:
                    # Acceptance boundary (Finding 3): a cancel already
                    # signalled wins — never commit a buffered approval after
                    # a stop. Mirrors the ``timeout > 0`` boundary above, so
                    # both branches classify the same way: return ``(None, True)``
                    # and let the caller classify a set ``cancel_event`` as an
                    # ``"interrupted"`` rejection (the fail-safe direction).
                    if not _cancelled():
                        return _done(line)
                    return (None, True)
                if status == "eof":
                    return (None, False)  # dead channel, not a timeout
                return (None, True)  # incomplete -> timeout (single poll)

            deadline = now() + timeout
            while True:
                if _cancelled():
                    return (None, True)
                line, status = _read_until_newline(partial, deadline)
                if status == "error":
                    return (None, False)
                if status == "eof":
                    # Stream exhausted: no more data can arrive, so fail fast
                    # with a no-channel rejection (the caller classifies it
                    # as ``"no-input-channel"``) instead of waiting out the
                    # deadline on a dead pipe.
                    return (None, False)
                if line is not None:
                    # Acceptance boundary: a line that completed at or past
                    # the deadline, or after a cancel, is a timeout — never
                    # an approval (fail-safe; the cancel event wins).
                    if now() < deadline and not _cancelled():
                        return _done(line)
                    return (None, True)
                remaining = deadline - now()
                if remaining <= 0:
                    return (None, True)
                time.sleep(min(0.01, remaining))
        finally:
            try:
                os.set_blocking(fd, prev)
            except (OSError, ValueError, AttributeError, NotImplementedError):
                pass  # fail-closed: a restore failure must not mask the result
