# Phase 4 — the `run` loop, its config, and its stop semantics

This is the Phase 4 reference for the **observe-decide-act loop** behind the
`run` subcommand, the config that drives it, and — most importantly — an
**honest** account of how a running loop is actually stopped.

Everything in this file is verified against the current `feature/phase-4` tree,
not against the plan. Where a mechanism is *implemented and tested* but *not
wired into the CLI*, that gap is called out explicitly instead of being
described as if it worked.

---

## 1. What `run` does

`run` is the generic, **game-agnostic** agent loop (Phase 4 §4.4). It is not a
WoW script. Each tick is:

```text
capture.grab()            → frame
perception.observe(frame) → observation
machine.decide(obs, now)  → state          (deterministic state machine)
policy.actions(state, ...)→ Action[]       (the game-adapter boundary)
executor.execute(action)  → results        (via the single executor)
```

The loop is game-agnostic; the game-specific policy comes from
`games/wow` via `DeterministicPolicy.from_config`. The state machine
(`GameStateMachine`) supplies the transitions, bounded recovery, and watchdog
timeout. The executor is the *only* path to input.

## 2. Run it (headless, no display)

The cleanest deterministic invocation uses the mock capture and mock input
backends, so it runs in CI / a bare venv with no desktop and no OS input:

```bash
# One tick, then stop (== --max-ticks 1). Prints the report JSON.
uv run python -m ai_game_agent run --once \
    --backend mock --mode observe_only --config config/default.yaml

# Run for N ticks, pretty-printed.
uv run python -m ai_game_agent run --max-ticks 5 \
    --backend mock --mode observe_only --pretty
```

`--mode observe_only` (the default) means **zero OS input** is emitted, which
is the safe default. See the stop-semantics section (§5) for how to stop a
loop that is allowed to act.

## 3. Flags (verified against the parser)

| Flag | Meaning |
| --- | --- |
| `--once` | Run exactly one tick then stop (alias for `--max-ticks 1`). |
| `--max-ticks N` | Stop after N ticks. Default: run until stop / `PAUSED`. Must be ≥ 1. |
| `--mode` | `observe_only \| assisted \| semi_autonomous \| autonomous`. Default: `input.mode` from config, else `observe_only`. |
| `--backend` | Input backend (`mock \| windows`). Default: `input.backend` from config. |
| `--confirm` | Required for `--backend windows` in a non-`observe_only` mode (real OS input). No-op for `mock`. |
| `--record` | Record each `LoopEvent` to a `.jsonl` file under `--record-dir`. |
| `--record-dir` | Directory for the recorded event log (default: `recordings`). |
| `--no-templates` | Disable template matching **and** UI-zone checks. |
| `--no-ocr` | Disable OCR. |
| `--no-objects` | Disable color-blob object detection. |
| `--pretty` | Pretty-print the report JSON (`indent=2`) instead of a single sorted line. |
| `--config` | YAML config file (default: `config/default.yaml`). |

## 4. Config that drives the loop

The `agent:` block (`config/default.yaml`) parameterizes the deterministic
policy **and** the state machine's two loop knobs:

```yaml
agent:
  name: wow
  combat_keys: [q, w]
  combat_delay_ms: 120
  loot_pickup_key: f
  loot_delay_ms: 200
  recovery_key: r
  recovery_delay_ms: 300
  health_low: 0.30
  health_recovered: 0.50
  # Loop knobs (Phase 4 close-out) — read by GameStateMachine:
  state_timeout_s: 5.0   # watchdog: max seconds in a non-terminal state
  recovery_attempts: 3   # bounded recovery budget before PAUSED
```

**The two loop knobs (verified defaults + validation):**

| Key | Type | Default | Validation |
| --- | --- | --- | --- |
| `state_timeout_s` | float | `5.0` | must be a positive number; **bool rejected** |
| `recovery_attempts` | int | `3` | must be a positive int; **bool rejected** |

Both are validated identically in the pydantic model and in the frozen
`AgentConfig` facade (each raises `ValueError` on a bad value). They are
threaded from the config into `GameStateMachine(policy=..., recovery_attempts=...,
state_timeout_s=...)` on **both** the live `run` path and the `replay` path, so a
recorded replay reproduces the same recovery budget and watchdog timeout as the
session that produced it.

`health_low` / `health_recovered` are **thresholds only** — the calibrated
health reading is a runtime callable, never stored in config.

## 5. Stop semantics (verified against the tree — read this)

This is the section the plan could overstate and the code does not. Three
different "stop" paths exist in the code, but **only some are reachable from the
`run` CLI today.**

### 5.1 The clean, report-producing stops (the ones you use)

`AgentLoop.run()` exits cleanly and prints a report when one of these happens:

| `stopped_reason` | Triggered by | Reachable via `run` CLI? |
| --- | --- | --- |
| `max_ticks` | `--once` / `--max-ticks N` reached | **Yes** |
| `paused` | machine entered `PAUSED` (recovery budget exhausted) | **Yes** (on a real pipeline failure) |
| `capture_start_failed` | `capture.start()` raised before the loop began | **Yes** |

On all three, `capture.stop()` runs in `finally`, `executor.release_all()` is
called as belt-and-braces, and a full report is printed to stdout. This is the
normal, expected way to end a loop.

### 5.2 Ctrl+C / SIGINT — this is an **abort**, not a clean stop

**There is no SIGINT handler and no `KeyboardInterrupt` catch anywhere in
`src/`.** This was verified by grep (zero matches) *and* empirically by sending
SIGINT to a running loop:

- the default Python handler raises `KeyboardInterrupt`, which **propagates**
  out of `loop.run()` and out of `main()` unhandled;
- **no report is printed to stdout** (stdout stays empty);
- `KeyboardInterrupt` + a traceback go to **stderr**;
- the process exits **non-zero** (observed `130` under bash).

Because the interrupt unwinds *through* the loop's `finally`, the best-effort
cleanup there (`capture.stop()` and `executor.release_all()`) still runs — but
that is a side effect of exception unwinding, **not** a designed clean-stop
contract. Do **not** document or rely on Ctrl+C as a "clean partial report."
It is an abort. For a deterministic, report-producing stop, use `--once` or
`--max-ticks N`.

### 5.3 Emergency stop / the F12 hotkey — **wired into `run`** (non-observe modes)

All three pieces are implemented, tested, **and now connected to the `run`
subcommand**:

- `Executor.trigger_emergency_stop(reason)` is implemented (sets `stopped`,
  cancels the queue, releases all held input, blocks further input, idempotent).
- `WindowsBackend` has an armed/disarmed `_HotkeyDecider` that fires an
  `on_stop` callback (F12 by default) — implemented and tested with a fake
  pynput. The listener is **armed on `open()`** and **disarmed on `close()`**.
- `MockBackend` exposes a deterministic `trigger_emergency_stop()` and an
  `on_stop` seam (see §5.4), so the stop path is testable headless.

How the CLI wires them together:

- `_run` creates a live `threading.Event` (`stop_event`) and an `_f12`
  callback that does **two things**: `stop_event.set()` **and**
  `executor.trigger_emergency_stop("f12")`. The callback reads the executor
  lazily from a holder, because the executor is built *after* the backend —
  no construction-order cycle.
- That callback is handed to `create_backend(..., on_stop=_f12)`, which
  forwards it to the backend's `_HotkeyDecider`, so a physical F12 press fires
  it.
- `_run` passes the same `stop_event` into `AgentLoop`, which checks
  `executor.stopped` (→ `emergency_stop`) and `stop_event` (→
  `stop_requested`) with the highest priority — at the top of the loop and in
  the post-tick recheck.
- `_run` calls `backend.open()` before the loop and `backend.close()` in
  `finally`, **mirroring the `act` path** — so the pynput listener actually
  arms on entry and disarms (releasing any held input) on exit.

**Net effect:** in a non-observe `run`, **F12 stops the loop** — it sets the
stop event, triggers the executor's terminal stop, and releases every held
input. Both `emergency_stop` and `stop_requested` are reachable
`stopped_reason` values from the `run` CLI: the F12 path reports
`emergency_stop`; a programmatic `stop_event.set()` reports `stop_requested`.

**One genuine limitation:** under `OBSERVE_ONLY` the CLI deliberately leaves
the backend **closed** (the `act` path does the same) so that zero OS input is
emitted. For the `windows` backend that means the pynput hotkey listener is
**not armed** in `observe_only`, so a physical F12 is not captured there. The
`stop_event` gate, however, is still passed to the loop in every mode, so a
programmatic `stop_event.set()` still produces a clean `stop_requested` stop. The
`mock` backend has no listener dependency, so the headless stop-path tests (mock
`trigger_emergency_stop`) run in every mode.

### 5.4 The mock backend **does** expose an emergency-stop trigger

`MockBackend` (`actions/mock.py`) exposes `open`/`close`, the keyboard/mouse
verbs, `release_all`, an `events` log, **and** a deterministic
`trigger_emergency_stop(reason)`. That method releases all held keys/buttons,
sets `stopped`, and fires its `on_stop` callback at most once — the same
contract the `windows` backend's F12 handler upholds.

This is what lets the stop path be exercised **headless, in a bare venv** — no
pynput, no OS input — so CI can pin both `emergency_stop` and `stop_requested`
from the `run` CLI (see `tests/test_stop_wiring.py`). For a *physical* F12
source you still need the `windows` backend, whose hotkey listener is armed by
`open()` (and therefore only armed in non-observe `run` modes, per §5.3).

## 6. The report JSON

`run` prints `AgentLoopReport.to_dict()` to stdout. Keys:

| Key | Meaning |
| --- | --- |
| `ticks` | full `tick()` iterations completed. |
| `transitions` | every `STATE_ENTER` event fired by the machine. |
| `actions_attempted` | total `Action`s dispatched to the executor. |
| `actions_ok` | how many returned `ok=True`. |
| `actions_failed` | how many returned `ok=False` (or raised). |
| `errors` | every `RECOVERY` / `ERROR` event recorded. |
| `final_state` | machine state when the loop exited. |
| `stopped_reason` | one of `max_ticks` / `stop_requested` / `emergency_stop` / `paused` / `capture_start_failed` (see §5). |
| `wall_time_s` | wall-clock duration of the `run()` call. |

Default is a single compact sorted line; `--pretty` pretty-prints it. Both are
valid JSON.

## 7. Replay workflow

`replay` re-derives the state sequence from a recorded `.jsonl` event log — it
feeds recorded observations back through perception → state machine → decision
**without touching a real application.** This is the primary regression-testing
mechanism for perception and decision logic (AGENTS.md).

```bash
# Record a run's event log:
uv run python -m ai_game_agent run --max-ticks 50 \
    --backend mock --mode observe_only --record --record-dir recordings

# Replay it (re-derive the state sequence):
uv run python -m ai_game_agent replay <recorded>.jsonl \
    --config config/default.yaml
```

The replay path uses the **same** `agent.recovery_attempts` /
`agent.state_timeout_s` knobs as the live path, so a replay reproduces the same
bounded-recovery budget and watchdog timeout as the recorded session.
