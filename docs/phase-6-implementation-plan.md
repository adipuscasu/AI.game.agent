# Phase 6: Assisted Mode — Implementation Plan

**Status:** implemented; §3.1, §4, §6 corrected on 2026-11-06 to the shipped fail-closed contract (see PR #7, commit `af3891b`) — earlier wording described an intended fail-open design that never shipped
**Branch:** `feature/phase-6`
**Date:** 2026-10-25
**Prerequisite:** Phase 5 complete (AI provider seam in `agent/loop.py`; `AIDecision` schema; `ai/` package; full suite green)

---

## 1. Purpose

Phase 5 let the loop *consult* a local AI provider when the deterministic
policy hit the `AI_ANALYSIS` state, and dispatch the validated
`AIDecision.action`. What it did **not** do is ask a human before executing
that AI-proposed input. Phase 6 adds the missing human-in-the-loop gate:

```text
Observe
   ↓
Analyze          (deterministic policy +, in AI_ANALYSIS, the AI provider)
   ↓
Suggest action   (the AI-proposed Action, validated by Action.from_dict / SafetyGuard)
   ↓
Human approval   (NEW — ASSISTED mode + an Approver)
   ↓
Execute
```

Goal (from `docs/high-level-development-plan.md` § Phase 6):

> Validate AI decisions without fully autonomous control.

The load-bearing invariant, stated as the fail-safe rule that every
implementation detail must satisfy:

> **An affirmative answer is the only path to execution.** Every other
> outcome — no, blank, EOF, timeout, no input channel, unparsable — is a
> rejection, never an approval.

## 2. What changed (the contract)

| Surface | Change |
| --- | --- |
| `actions/approver.py` | **New.** `Approver` protocol, `ApprovalDecision` (with `approved()` / `rejected()`), `AutoApprover`, `TerminalApprover`. |
| `actions/__init__.py` | Exports `Approver`, `ApprovalDecision`, `AutoApprover`, `TerminalApprover`. |
| `state/events.py` | `EventType.APPROVAL` added (closed vocabulary: 8 → 9 values). |
| `agent/loop.py` | `AgentLoop.__init__(…, approver=None)`. In `ASSISTED` mode, AI-proposed actions are held on the approver; a rejection is a **safe halt** (machine → `PAUSED`, an `APPROVAL` audit event, zero input). Approval counters (`approvals_requested / approvals_granted / approvals_rejected`) are tracked **inside the loop** and recorded on each `APPROVAL` event; they are deliberately **not** added to `AgentLoopReport`, keeping the report shape stable for the CLI and the replay diffs. |
| `__main__.py` | `run` wires an approver **only** in `ASSISTED` mode (default `TerminalApprover`; `--approve-all` → `AutoApprover`; `--approve-timeout N`). |
| `tests/test_assisted_approval.py` | **New.** 17 tests: gate contract + fail-safe approver semantics + no-op outside `ASSISTED`. |
| `tests/test_state_machine.py` | `APPROVAL` added to the closed `EventType` set (8 → 9). |
| `tests/test_agent_loop.py` | Phase-5 `AI_ANALYSIS` tests unchanged — the gate is a no-op outside `ASSISTED`. |

## 3. Design decisions (and the reasons)

### 3.1 The gate only engages in `ASSISTED` mode

`SafetyGuard` and `OperationMode` already know about `assisted` (Phase 2
planned it; `modes.py` defines the mode and its gate). The *gate itself*
was never implemented. Phase 6 implements it **as an approver**, and scopes
it to `ASSISTED` so that:

- `OBSERVE_ONLY` / `SEMI_AUTONOMOUS` / `AUTONOMOUS` dispatch exactly as in
  Phase 4/5 (byte-for-byte; their tests are unchanged and still pass).
- `ASSISTED` **requires a configured approver** for any AI-proposed action
  to run. With no approver injected, the gate **rejects** the proposal —
  it emits an `APPROVAL` audit event with reason `no-approver-configured`,
  forces the machine into `PAUSED` (a safe halt), and dispatches nothing.
  Absence of an approver is never treated as blanket approval (fail-closed,
  not fail-open).

This is the smallest surface that adds the feature without touching any
working path.

> **Historical note (superseded).** An earlier draft of this section — and
> the original Phase-6 design intent — described `ASSISTED` without an
> approver as *falling back to the pre-gate Phase-5 behavior* (i.e. the
> action dispatched anyway, "no surprise prompts"). That fail-open design
> **did not ship**. The implemented contract — pinned by
> `test_assisted_without_approver_is_fail_closed` and the shared safe-halt
> contract test in `tests/test_assisted_approval.py`, and corrected in PR #7
> (`af3891b`) — is fail-closed as described above.

### 3.2 Pluggable `Approver` protocol — one seam, three implementations

```python
class Approver(Protocol):
    name: str
    def request_approval(self, action, *, reasoning="", confidence=None,
                         timeout_s=None) -> ApprovalDecision: ...
```

- `TerminalApprover` — interactive `y/N` on stdin. Default for a real
  assisted run.
- `AutoApprover` — always approves. For CI / non-interactive runs
  (`--approve-all`).
- `QueueApprover` — transport-agnostic queue for remote/web approval;
  implemented and shipped. The loop is unchanged; a web client is a thin
  adapter over :meth:`QueueApprover.pending` and :meth:`QueueApprover.respond`.

The loop depends on the *protocol*, not a concrete class — so the
interactive, automated, and remote channels are all the same
`approver` argument, and the loop's logic is written once.

### 3.3 Rejection is a **safe halt**, not "try again"

A rejected proposal is not retried in the same episode. The loop forces the
machine into `PAUSED` (a *forced* `machine.enter`, not an observation-driven
`decide()`), which:

- makes `run()` exit on the `paused` stop reason (the same terminal-ish
  state it already knows),
- releases every held input (the loop's `finally` block runs
  `executor.release_all()` → `backend.release_all()`),
- and — because a *new* `AI_ANALYSIS` episode (state leaves and re-enters)
  is the only thing that re-triggers a consult — the next analysis is a
  fresh decision, not a re-prompt of the rejected one.

This respects the project's "deterministic over AI" and "safe by default"
posture: the *human* is the authority, and "no" (or silence) is treated the
same as "no".

### 3.4 Fail-safe approver semantics (the core test target)

`TerminalApprover` treats **every** non-affirmative answer as a rejection:

| Input | Outcome |
| --- | --- |
| `y` / `yes` / `approve` / `ok` / `go` / `1` (case-insensitive) | **approved** |
| `n` / `no` / `reject` / blank | rejected (`user-rejected`) |
| any other text (e.g. `maybe`) | rejected (`unparsable-reply`) |
| EOF / no stdin | rejected (`no-input-channel`) |
| timeout (fd path) | rejected (`timeout`) |
| `KeyboardInterrupt` | rejected (`interrupted`) |

The `_read_line` helper catches `AttributeError` as well, so a `None` stdin
or a detached session (no `readline`) is a clean rejection, **not** a crash —
that is exactly the fail-safe scenario the rule names. The timeout path uses
a **deadline-strict non-blocking read** (`_read_line_fd`) on a real file
descriptor: the fd is driven non-blocking and read byte-by-byte with
`os.read` — never `readline` (so a lone byte cannot hang), and no `select` or
worker thread is involved (so the Windows `select` socket-only limitation and
thread accumulation never apply). A reply is honoured only if its newline is
consumed within the window, and the cancel signal is checked each poll slice.
An interactive Windows console instead uses the key-by-key console reader.
`timeout_s=None` falls back to the approver's configured
`TerminalApprover.DEFAULT_TIMEOUT_S`, currently 30 s, rather than waiting
forever; `timeout_s=0` is a valid **non-blocking poll** (a single bounded
probe that honours an already-buffered reply and otherwise rejects);
negative or non-finite values are rejected with `ValueError` at the call site.
The deadline and cancellation behaviour stay fail-safe: a late reply never
approves a timed-out proposal, and a set cancel signal (e.g. the F12 emergency
stop) wins over a pending approval.

### 3.5 `APPROVAL` is an audit event, not a state

`EventType.APPROVAL` is recorded (with `approved`, `channel`, `reason`,
`action`, `reasoning`, `confidence` in `detail`) so a log reader can
reconstruct *what* was proposed, *by whom* (the AI), *who* decided
(the approver channel), and *why* — without re-running perception or the
AI. It is deliberately **not** an `AgentState`; approval is a decision
about an action, not a phase of the loop.

## 4. The approval flow (code)

`AgentLoop._gate_ai_actions(actions, now, reasoning, confidence)` returns
`True` (proceed to dispatch), `False` (unused), or `None` (rejection — the
caller halts the episode before dispatch). Shipped contract:

```python
if self._mode is not OperationMode.ASSISTED:
    return True                          # gate is ASSISTED-scoped
if not actions:
    return True                          # defensive no-op
if self._approver is None:
    # Fail-closed: nobody can answer, so nobody approved.
    self._approvals_requested += 1
    self._reject_and_halt(actions[0], now=now,
                          reason="no-approver-configured", channel="none", ...)
    return None                          # PAUSED; nothing dispatched

self._approvals_requested += 1
try:
    decision = self._approver.request_approval(actions[0], ...)
except Exception as exc:                 # an errored approver is a rejection
    decision = ApprovalDecision.rejected(reason=f"approver-error: {exc}")

# Approval is derived from the *validated* decision, never from truthiness:
a valid affirmative decision  -> APPROVAL event (approved), return True
anything else (rejection, malformed, out-of-vocabulary) ->
    _reject_and_halt(...) -> PAUSED; return None
```

The loop calls `_dispatch` **only if** the gate returned `True` — a `None`
return is the "halt before dispatch" signal (machine → `PAUSED`, held inputs
released, `APPROVAL` audit event recorded). Deterministic (non-AI) actions
never reach the gate; only actions produced by the AI decision path (the
`ai_proposed` flag set at the AI seam in `agent/loop.py`, i.e. state is
`AI_ANALYSIS` and the provider returns a decision) are gated. The gate does
not read `Action.source`; it keys off the AI seam.

> **Historical note (superseded).** An earlier version of this section
> showed the guard as `if (... or self._approver is None ...): return
> actions`, i.e. a missing approver was treated as *no gate — dispatch as
> before*. That fail-open path was never the shipped behavior; the current
> implementation rejects, safe-halts, and records the outcome as shown
> above.

## 5. CLI

```bash
# assisted, interactive y/N (the default for ASSISTED):
ai-game-agent run --mode assisted --ai --once

# assisted, auto-approve (CI / non-interactive):
ai-game-agent run --mode assisted --ai --once --approve-all

# assisted, bounded wait for the answer:
ai-game-agent run --mode assisted --ai --approve-timeout 10
```

The approver is built **only** when `mode is ASSISTED`; `--approve-all` and
`--approve-timeout` are inert in every other mode.

## 6. Verification

- **Original Phase-6 verification (historical close-out):** `pytest` full
  suite green — **732 passed** (no skips recorded at Phase 6), and
  `ruff check .` clean. This is the result of the Phase-6 implementation
  itself, as recorded in §2 and the §2 test table.
- **Current repository baseline (later, for reference only):** **848 passed /
  2 skipped**. This count is a *later* baseline from after additional phases
  were added (the 2 skips are environment-dependent — they require a live
  Windows console, `CONIN$`). It is **not** the original Phase-6 result, and
  this documentation pass did not rerun the test suite; the figure is
  quoted from the existing baseline record.
- `ruff check .` — clean across the repo.
- `tests/test_assisted_approval.py` — 76 tests (17 at Phase-6 time of
  writing) covering: approved action executes; rejection is a safe halt
  that blocks dispatch; a raising approver is classified as a rejection;
  the gate never fires for deterministic actions or outside `ASSISTED`;
  **`ASSISTED` with no approver is fail-closed** — `test_assisted_without_approver_is_fail_closed`
  asserts the AI proposal is *refused*: machine in
  `PAUSED`, zero actions dispatched, exactly one `APPROVAL` audit event
  with `approved is False` and `reason == "no-approver-configured"`
  (earlier drafts claimed this path "still dispatches (back-compat)" — that
  was never the shipped contract); every rejection path (no-approver,
  explicit rejection, raising approver, malformed decision) shares one
  safe-halt contract; the executor has no approver seam; the fail-safe
  approver table (affirmatives approve; negatives, unparsable, EOF,
  `KeyboardInterrupt`, and timeout all reject); F12 emergency stop wins
  while the approver would block; `OBSERVE_ONLY` dispatches nothing even
  with AI + approver present; and the `APPROVAL` event detail contract.
- `tests/test_state_machine.py` — `EventType` closed set now includes
  `APPROVAL` (9 values).

## 7. Out of scope (still to do)

- The web/remote client UI for `QueueApprover` (poll `pending()`, call
  `respond()`). The `Approver` protocol and the `QueueApprover` transport
  are both implemented; only the client layer remains.
- Persisting the audit `APPROVAL` events to the `--record` jsonl is already
  free (the recorder hook writes every emitted `LoopEvent`); only the
  reader side is a UX task.
- Multi-action batches in one AI decision: the current gate approves or
  rejects the whole AI action batch as one decision (one prompt); per-action
  approval is a possible refinement.

## 8. Follow-ups from the `1bbbb76` final review (deferred, non-blocking)

The Phase 6 approval-input fix (`1bbbb76`) passed final review (**PASS WITH
CAVEATS**; CI green: 777 passed / 2 skipped). Two low findings were
*deliberately not* folded into that commit — they are pre-existing / orthogonal,
and the reviewer's call was to keep the liveness + cancellation fix clean. They
are recorded here so they are not lost. (These are the *final-review* findings;
distinct from the "Finding 3" used in the `1bbbb76` commit message for the
timeout=0 cancel-commit fix.)

- **Console `timeout=0` does not probe as documented** (Low — a real doc/impl
  discrepancy). **RESOLVED.** The console reader now honours a reply already
  pending at the `timeout=0` slice, matching the fd path's single
  non-blocking probe; a pending `kbhit()` reply approves when cancellation is
  clear and rejects as `"interrupted"` when a stop is set. Regression-pinned by
  `test_console_timeout_zero_pending_reply_without_cancel_is_approved` (the
  companion `test_fd_timeout_zero_buffered_y_without_cancel_is_approved` covers
  the fd seam). No code change needed here to mark this — the implementation
  and the test are already in the tree.
- **`_buffered_reply_fd` test helper leaks the read fd** (Low — test hygiene).
  **RESOLVED.** `tests/test_assisted_approval.py::_buffered_reply_fd` is now a
  `contextlib.contextmanager`: the write end is closed at setup and the read
  end is closed in a `finally`, so both descriptors are released on every exit
  path (success or exception). The three fd regression tests adopt the
  `with _buffered_reply_fd(...) as fake_stdin:` form; the helper's API and the
  tests' assertions are unchanged.
- **TOCTOU at the acceptance boundary** (Informational — not a regression).
  Both fd acceptance paths check cancellation and then return the line; a stop
  could in theory land in between. Same as the pre-existing `timeout>0` path.
  The independent F12 executor emergency-stop is the operational safeguard. Do
  **not** expand the commit to add locking / a new synchronization protocol;
  design it separately only if an absolute "cancellation always wins" guarantee
  is ever required.
