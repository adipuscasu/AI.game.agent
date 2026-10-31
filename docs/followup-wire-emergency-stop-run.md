# Follow-up — wire the emergency stop into the `run` CLI

**Status:** **closed / done** (opened after the Phase 4 close-out commit
`c84f90f`; resolved by `5981a1b` + `aacdc0d`, carried into `feature/phase-6`).
**Branch:** `feature/phase-4` (work completed; merged forward into
`feature/phase-6`).
This ticket is **closed**: the wiring it requested is implemented, tested, and
documented. The "Why" bullets below are the rationale **at the time of
opening** — they no longer describe the tree (see the resolution above them).

## Resolution (verified against the tree, 2026-10-31)

- `run` creates a live `threading.Event` (`stop_event`) and an `_f12` callback
  that both `stop_event.set()` **and** call
  `executor.trigger_emergency_stop("f12")`. The callback is handed to
  `create_backend(..., on_stop=_f12)` (F12 fires it via the backend's
  `_HotkeyDecider`), and the same `stop_event` is passed into `AgentLoop`. `run`
  calls `backend.open()` before dispatch and `close()` in `finally` (non-observe
  modes), so the listener is actually armed.
- `MockBackend` exposes a deterministic `trigger_emergency_stop(reason)` and an
  `on_stop` seam, so the stop path is testable headless (no pynput / OS input).
- **Evidence:** `5981a1b` (feat: wire F12 emergency-stop into run CLI and
  release all held input) and `aacdc0d` (fix: open input backend in run before
  dispatch, close in finally) — both are ancestors of `feature/phase-6`.
- **Tests:** `tests/test_stop_wiring.py` — 8 passed (mock trigger releases held
  input; `create_backend` wires `on_stop`; `run` passes a real `stop_event`; the
  full F12 chain; in-flight stop releases; `run` reports `emergency_stop` /
  `stop_requested`).
- **Docs kept honest:** `docs/phase-4-loop.md` §5.3–§5.5 and
  `docs/manual-testing.md` §5.4–§5.5 describe F12 as wired and **Ctrl+C as an
  abort** (there is still **no** SIGINT handler in `src/` — do not document a
  clean Ctrl+C stop that does not exist).

## Why (rationale at the time of opening)

- `_run` builds `AgentLoop(...)` with neither `stop_event=` nor `hotkey=`, so
  both are `None` and the loop's `stop_requested` / `emergency_stop` paths are
  unreachable from the CLI.
- `create_backend` builds `WindowsBackend()` with `on_stop=None` (no-op), so the
  F12 `_HotkeyDecider` never calls anything meaningful.
- `run` never calls `backend.open()`, so the pynput listener isn't even started.
- `MockBackend` has no `trigger_emergency_stop` and no hotkey.
- There is **no** SIGINT handler and **no** `KeyboardInterrupt` catch in `src/`;
  Ctrl+C today is an abort (stderr traceback, no stdout report, non-zero exit).

So the stop *mechanism* is implemented and tested in isolation, but not wired
into the running CLI. This ticket is the wiring.

## Scope (keep it bounded)

1. **Stop token.** Pass a `threading.Event` stop token into `AgentLoop` from the
   `run` path and check it where the loop already checks `executor.stopped`
   (top of each tick). Reuse the existing `stop_event` parameter; do not invent a
   second mechanism.
2. **F12 → emergency stop.** Wire the Windows backend's `on_stop` to
   `executor.trigger_emergency_stop(reason)` so a real F12 press stops *future*
   actions and releases held input. `run` must call `backend.open()` in the
   non-`observe_only` path (as `act` does) so the listener actually arms, and
   disarm/`close()` in `finally`.
3. **Testable mock trigger.** Give `MockBackend` an explicit, deterministic
   `trigger_emergency_stop()` (and/or expose a no-op stop source) so the
   stop-path tests run headless in a bare venv — no pynput, no OS input.
4. **In-progress action behavior.** Define and test what happens when a stop
   fires while an action is mid-`_sleeper` / holding a key: held input must be
   released (the `trigger_emergency_stop` contract already releases held keys /
   mouse buttons); assert the release, the `STOPPED` state, and that further
   `execute()` calls are blocked.
5. **Ctrl+C is separate.** If a clean partial report on Ctrl+C is desired, add a
   deliberate `KeyboardInterrupt` handler in the CLI entrypoint (catch, trigger
   the same stop path, print the report, exit). **Do not conflate** SIGINT with
   the F12/emergency-stop mechanism, and do not let one mask the other. If it is
   not added, the docs' "Ctrl+C is an abort" statement stays.

## Acceptance (definition of done)

- A `threading.Event` and the F12 hotkey both produce `stopped_reason`
  `stop_requested` / `emergency_stop` from the `run` CLI (at least via the mock
  path in CI).
- Held input is released on every stop path (mock + windows, tested).
- In-progress action: stop during a held key/button releases it (regression test).
- Ctrl+C behavior is either a clean report (handled + tested) or the docs remain
  "abort, non-zero, no report" — never a false "clean stop" claim.
- `observe_only` still emits zero OS input; `OBSERVE_ONLY`/stop semantics unchanged.
- TDD: each stop path has a failing test written before the wiring.
- Full suite green in a bare venv + `ruff check src tests` clean.
- `docs/phase-4-loop.md` §5 and `docs/manual-testing.md` §5 updated to reflect
  whatever is actually wired (remove the "not wired" wording only for what now
  works).

## Explicitly out of scope

- No new signal handling beyond the single deliberate Ctrl+C handler (item 5).
- No changes to the deterministic policy, state machine transitions, or capture.
- No new dependencies (pynput is already the existing `input` extra).

## Related

- Mechanism already present: `Executor.trigger_emergency_stop`,
  `AgentLoop(hotkey=, stop_event=)`, `WindowsBackend` `_HotkeyDecider`,
  `MockBackend.release_all`.
- Contract to honor: AGENTS.md safety (bounded durations, guaranteed release,
  emergency stop independent of the AI) and the `act` CLI's open/close lifecycle.
