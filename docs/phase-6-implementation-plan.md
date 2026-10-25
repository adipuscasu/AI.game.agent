# Phase 6: Assisted Mode — Implementation Plan

**Status:** implemented
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
- `ASSISTED` with **no** approver injected (the default CLI for other modes,
  and existing tests) falls back to today's Phase-5 behavior — no surprise
  prompts.

This is the smallest surface that adds the feature without touching any
working path.

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
- (A future `QueueApprover` for remote/web approval slots in here without
  touching the loop.)

The loop depends on the *protocol*, not a concrete class — so the
interactive, automated, and (later) remote channels are all the same
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
| `y` / `yes` / `approve` / `ok` / `go` (case-insensitive) | **approved** |
| `n` / `no` / `reject` / blank | rejected (`user-rejected`) |
| any other text (e.g. `maybe`) | rejected (`unparsable-reply`) |
| EOF / no stdin | rejected (`no-input-channel`) |
| timeout (fd path) | rejected (`timeout`) |
| `KeyboardInterrupt` | rejected (`interrupted`) |

The `_read_line` helper catches `AttributeError` as well, so a `None` stdin
or a detached session (no `readline`) is a clean rejection, **not** a crash —
that is exactly the fail-safe scenario the rule names. The timeout path uses
`select.select` on a real file descriptor; `timeout_s=None` means "wait
forever" (a human may take their time) while `timeout_s <= 0` is rejected
with `ValueError` at the call site (so `0` can never silently mean "no
timeout").

### 3.5 `APPROVAL` is an audit event, not a state

`EventType.APPROVAL` is recorded (with `approved`, `channel`, `reason`,
`action`, `reasoning`, `confidence` in `detail`) so a log reader can
reconstruct *what* was proposed, *by whom* (the AI), *who* decided
(the approver channel), and *why* — without re-running perception or the
AI. It is deliberately **not** an `AgentState`; approval is a decision
about an action, not a phase of the loop.

## 4. The approval flow (code)

`AgentLoop._gate_ai_actions(actions, now)`:

```python
if (self._mode is not OperationMode.ASSISTED
        or self._approver is None
        or not any(a.source == "ai" for a in actions)):
    return actions                       # no gate: dispatch as before

if any(a.source == "ai" for a in actions) is False:
    return actions

for action in actions:
    decision = self._approver.request_approval(...)
    self._approvals_requested += 1
    if not decision.is_approved:
        self._approvals_rejected += 1
        self._emit(LoopEvent(event=EventType.APPROVAL, ..., detail={...}))
        self._machine.enter(AgentState.PAUSED,
                            reason="assisted approval rejected; safe pause", now=now)
        return None                      # the caller emits nothing & halts
    self._approvals_granted += 1
    self._emit(LoopEvent(event=EventType.APPROVAL, ..., detail={...approved: True...}))
return actions
```

The loop then calls `_dispatch` **only if** the gate returned actions — a
`None` return is the "halt before dispatch" signal. Deterministic
(non-AI) actions in an `ASSISTED` episode still dispatch immediately; only
`source == "ai"` actions are gated.

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

- `pytest` — full suite green: **732 passed**.
- `ruff check .` — clean across the repo.
- `tests/test_assisted_approval.py` — 17 new tests covering: approved
  action executes; rejection is a safe halt that blocks dispatch; a
  raising approver is classified as a rejection; the gate never fires
  for deterministic actions or outside `ASSISTED`; `ASSISTED` with no
  approver still dispatches (back-compat); the executor has no approver
  seam; the fail-safe approver table (affirmatives approve; negatives,
  unparsable, EOF, `KeyboardInterrupt`, and timeout all reject);
  F12 emergency stop wins while the approver would block;
  `OBSERVE_ONLY` dispatches nothing even with AI + approver present;
  and the `APPROVAL` event detail contract.
- `tests/test_state_machine.py` — `EventType` closed set now includes
  `APPROVAL` (9 values).

## 7. Out of scope (still to do)

- A non-terminal `Approver` (remote/web approval) — the protocol is
  ready for it; the implementation is not written.
- Persisting the audit `APPROVAL` events to the `--record` jsonl is already
  free (the recorder hook writes every emitted `LoopEvent`); only the
  reader side is a UX task.
- Multi-action batches in one AI decision: the current gate approves or
  rejects the whole AI action batch as one decision (one prompt); per-action
  approval is a possible refinement.
