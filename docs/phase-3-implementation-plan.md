# Phase 3 Implementation Plan — Input Control

**Date:** 2026-10-09
**Status:** Proposed — not yet implemented. Branch `feature/phase-3` exists at
        the Phase 2 merge (`24e8015`); no Phase 3 code is present yet.
**Scope:** Add the action-execution layer: a deterministic `Action` contract,
        an `ActionQueue`, an `Executor` with key/mouse-state tracking and
        bounded timing, a `SafetyGuard` with an AI-independent emergency stop
        and guaranteed release of all held keys/buttons, and the four operating
        modes. Everything up to and including the real Windows input backend is
        platform-independent and CI-testable; the Windows backend itself is the
        thin, optional, headless-untestable adapter (the Phase 1 `mss` analog).

---

## 1. Goal (from the high-level plan, §15)

> Reliably perform deterministic actions.

Concretely, Phase 3 delivers:

* **A stable `Action` contract** — the Phase-3 analog of Phase 1's `Frame`
  and Phase 2's `Observation`: frozen, immutable, JSON-serializable,
  versioned.
* **An `ActionQueue`** — bounded, ordered, cancellable, with a priority
  lane for safety-critical actions (stop, release-all).
* **An `Executor`** — validates actions, sequences them, tracks key/button
  state, enforces bounded timing, and guarantees release of any held key or
  button even when an exception is raised mid-action.
* **A `SafetyGuard` + `EmergencyStop`** — an AI-independent hard stop that
  cancels the queue, releases all held input, blocks further input, and
  records the stop event.
* **Operating modes** — `OBSERVE_ONLY` / `ASSISTED` / `SEMI_AUTONOMOUS` /
  `AUTONOMOUS`, with the `OBSERVE_ONLY` "never emit input" hard gate built
  and regression-tested now.

Explicitly **not** in Phase 3 (deferred to their phases):

* Game state machine, decision engine, planning → Phases 4 and 6–7.
* The human-approval *UX* (the `[Execute] / [Reject] / [Pause]` flow) →
  Phase 6. Phase 3 ships the propose/approve *mechanism* only.
* AI interpretation of what to act on → Phase 5. Phase 3 executes *given*
  actions; it does not decide them.
* Live-game tuning → always, per the headless/CI rule; Phase 3 ships with a
  deterministic mock backend and a documented, replaceable Windows adapter.

---

## 2. Architecture fit

Phase 1 produced `Frame` + a `CaptureBackend` protocol with a deterministic
`MockBackend` (CI) and a real `MssBackend` (Windows). Phase 2 produced
`Observation` + a `Perception` pipeline of injected detectors. Phase 3 closes
the loop at the bottom of the high-level plan's core loop:

```text
Phase 1 (exists)                 Phase 2 (exists)               Phase 3 (new)
─────────────────                ─────────────────              ───────────────
Screen ─► Capture ─► Frame ─►  Perception ─► Observation ─►  Decision (P4/P5)
                                                                │
                                                                ▼ structured action
                                                          ┌──────────────────────┐
                                                          │  Action              │
                                                          │   ↓ validate         │
                                                          │  ActionQueue         │
                                                          │   ↓ sequence + time  │
                                                          │  Executor           │
                                                          │   (key/mouse state, │
                                                          │    release-on-error)│
                                                          │   ↓ guarded         │
                                                          │  SafetyGuard        │
                                                          │   ↓ backend         │
                                                          │  InputBackend       │
                                                          │   ├── Mock  (CI)     │
                                                          │   └── Windows (real)│
                                                          └──────────────────────┘
```

Responsibility rules (per AGENTS.md "Keep the Architecture Modular"):

* **Perception reads frames and never touches input.** Phase 3 never imports
  `perception`, and `actions` never imports `capture` or `perception`.
* **Only the `Executor` (via `InputBackend`) may touch OS input.** No code in
  `queue`, `safety`, `modes`, or the CLI may call a keyboard/mouse library
  directly — they program against the protocol.
* **The contract is dependency-free.** `action.py`, `base.py`, `safety.py`,
  and `modes.py` import only the project's own types (and the stdlib), so the
  whole decision/safety layer is importable — and its behavior testable — in a
  bare venv with no `pynput`/`pyautogui` installed.
* **The platform backend is optional and lazy.** `windows.py` (pynput /
  SendInput) is imported only when the `windows` backend is actually selected,
  exactly like `MssBackend` behind `create_backend`. A missing extra fails
  fast with a clean error naming the extra — never at import time.
* **Deterministic logic, not AI.** Sequencing, key-state tracking, bounded
  timing, cancellation, and the emergency stop are all plain, testable code.
  AI (Phase 5) only ever *proposes* an `Action`; it never emits OS input.

The single most important reuse: **Phase 1 already solved "how do we test a
platform I/O abstraction headless" with `MockBackend` + `create_backend(config)`
+ an availability check. Phase 3 does the same for input.** The mock backend
records every event it is asked to perform, which is what makes the executor,
queue, safety, and stop logic deterministic and CI-runnable without a desktop
session.

---

## 3. Module layout

New package `src/ai_game_agent/actions/`, nested under the existing top-level
package (installed name and test imports keep their shape). It mirrors the
`perception/` subpackage and the `actions/` logical folder named in AGENTS.md:

```text
src/ai_game_agent/
    __init__.py
    __main__.py            # + `act` subcommand (section 9)
    capture.py             # unchanged (Phase 1)
    config.py              # + InputConfig / ActionConfig (section 8)
    perception/            # unchanged (Phase 2)
    actions/
        __init__.py        # re-exports: Action, ActionKind, ActionResult,
                           # ActionLog, Executor, ActionQueue, SafetyGuard,
                           # EmergencyStop, OperationMode, InputError, ...
        action.py          # Action + ActionKind + ActionResult + ActionLog
                           #   (the contract; dependency-free; to_dict/to_json;
                           #    SCHEMA_VERSION)
        base.py            # Protocols: Keyboard, Mouse, InputBackend,
                           #   InputEvent; InputError hierarchy; create_backend
        mock.py            # MockBackend — deterministic, records events (CI)
        windows.py         # WindowsBackend (pynput/SendInput) + hotkey
                           #   listener — optional "input" extra, lazy import
        queue.py           # ActionQueue — bounded, priority lane, cancel
        executor.py        # Executor — validate, sequence, key/mouse-state,
                           #   bounded timing, release-on-error, state machine
        safety.py          # SafetyGuard + EmergencyStop — release-all, stop
                           #   event recording, mode gate, bounded limits
        modes.py           # OperationMode enum (OBSERVE_ONLY / ASSISTED /
                           #   SEMI_AUTONOMOUS / AUTONOMOUS)
```

New test files (mirrors the `tests/test_*.py` naming):

```text
tests/
    test_action.py           # Action/ActionResult/ActionLog contract
    test_actions_base.py     # Protocols + InputError hierarchy + create_backend
    test_actions_config.py   # InputConfig / ActionConfig (config-first)
    test_actions_mock.py     # MockBackend event recording + held-state
    test_actions_queue.py    # ActionQueue order/priority/bounds/cancel
    test_actions_executor.py # sequencing, key-state, timing, release-on-error
    test_actions_safety.py   # guard + emergency stop + mode gate
    test_actions_modes.py    # mode enum + OBSERVE_ONLY hard gate
    test_actions_bare.py     # bare-venv import contract (no "input" extra)
    e2e/test_e2e_act.py      # `act` CLI subprocess e2e (JSON, exit codes)
```

---

## 4. The contract: `Action` / `ActionResult` / `ActionLog`

The stable types downstream code (Phase 4 state machine, Phase 5 VLM output,
Phase 6 assisted approval, replay) programs against. Dependency-free, frozen,
tuple collections, JSON-serializable — the exact shape of Phase 1 `Frame` and
Phase 2 `Observation`.

```python
# actions/action.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = 1  # bump if the serialized shape changes meaningfully


class ActionKind(str, Enum):
    KEY_PRESS = "key_press"        # key (or modifier+key) down then up
    KEY_HOLD  = "key_hold"         # key down, hold for duration_ms, up
    MOUSE_MOVE = "mouse_move"      # absolute or relative move to (x, y)
    MOUSE_CLICK = "mouse_click"    # button, clicks, optional (x, y)
    MOUSE_DOWN = "mouse_down"      # button down (bounded hold)
    MOUSE_UP   = "mouse_up"        # button up
    DELAY      = "delay"           # wait duration_ms, then continue


@dataclass(frozen=True)
class Action:
    """One discrete, validated, deterministic input operation.

    Coordinates, if present, are in *screen space* (the action layer operates
    on the real desktop, unlike ``Observation`` which is frame space). ``None``
    fields are simply not used by a given ``kind`` — the executor validates
    the required/forbidden field set per kind before touching a backend.
    """
    kind: ActionKind
    # keyboard
    key: str | None = None
    modifiers: tuple[str, ...] = ()        # e.g. ("ctrl", "shift")
    # mouse
    x: int | None = None
    y: int | None = None
    relative: bool = False
    button: str = "left"                   # left | right | middle
    clicks: int = 1
    # timing (all bounded — see SafetyGuard)
    duration_ms: int = 0                   # holds / delays
    delay_ms: int = 0                      # pre-execution delay
    # provenance (replayability: "why did the agent do this?")
    source: str = "user"
    confidence: float | None = None
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class ActionResult:
    """Outcome of executing one Action. Always JSON-serializable."""
    action: Action
    ok: bool
    error: str | None = None
    duration_ms: float = 0.0
    stop_event: bool = False               # true if an emergency stop intervened


@dataclass(frozen=True)
class ActionLog:
    """A complete, replayable record of a batch of actions.

    This is the Phase-3 analog of Phase 1's recording and Phase 2's
    ``Observation`` — the artifact the `act` CLI prints and replay feeds back
    through queue → executor without a real backend.
    """
    results: tuple[ActionResult, ...]
    stopped: bool = False                  # an emergency stop occurred
    stop_reason: str | None = None
    schema_version: int = SCHEMA_VERSION
```

Every one of these must support `to_dict()` / `to_json()` and round-trip
through `json.dumps` with plain values (ISO timestamps, no enums-as-objects).
`Action` exposes `validate()` (or a module-level `validate_action`) that raises
a typed `ActionValidationError` for a kind whose required fields are missing or
whose forbidden fields are set — validation happens *before* any backend call,
per AGENTS.md "the model proposes an action; application code validates it."

---

## 5. Components

### 5.1 Protocols + factory (`base.py`)

Dependency-free `typing.Protocol`s the executor programs against (structural,
no base class required — same as `perception/base.py`):

```python
class Keyboard(Protocol):
    def key_down(self, key: str, modifiers: tuple[str, ...] = ()) -> None: ...
    def key_up(self, key: str, modifiers: tuple[str, ...] = ()) -> None: ...
    def press(self, key: str, modifiers: tuple[str, ...] = ()) -> None: ...

class Mouse(Protocol):
    def move(self, x: int, y: int, relative: bool = False) -> None: ...
    def click(self, button: str = "left", clicks: int = 1) -> None: ...
    def down(self, button: str = "left") -> None: ...
    def up(self, button: str = "left") -> None: ...

class InputBackend(Protocol):
    name: str
    def open(self) -> None: ...
    def close(self) -> None: ...
    def key_down(self, key: str, modifiers: tuple[str, ...] = ()) -> None: ...
    def key_up(self, key: str, modifiers: tuple[str, ...] = ()) -> None: ...
    def press(self, key: str, modifiers: tuple[str, ...] = ()) -> None: ...
    def mouse_move(self, x: int, y: int, relative: bool = False) -> None: ...
    def mouse_click(self, button: str = "left", clicks: int = 1) -> None: ...
    def mouse_down(self, button: str = "left") -> None: ...
    def mouse_up(self, button: str = "left") -> None: ...
```

The error hierarchy (AGENTS.md `ACTION_ERROR` / `INPUT_ERROR`), defined once in
`base.py` and raised by the other modules:

```python
class ActionError(Exception): ...        # base — the ACTION_ERROR class
class ActionValidationError(ActionError): ...   # bad field set for a kind
class ActionQueueError(ActionError): ...
class InputBackendError(ActionError): ...  # INPUT_ERROR — backend I/O failed
class SafetyViolation(ActionError): ...    # a bounded limit would be exceeded
class ModeViolation(ActionError): ...      # e.g. OBSERVE_ONLY attempted input
```

`create_backend(config)` mirrors `capture.create_backend`: a factory that
returns `MockBackend()` for `backend == "mock"`, `WindowsBackend()` for
`"windows"` (fail-fast `InputBackendError` naming the missing `input` extra if
the lib is absent, mirroring `_mss_available()`), and `ActionError` for unknown
names. This keeps the CLI importable in a bare venv and makes "which backend"
a config/CLI decision, not a code branch.

### 5.2 `MockBackend` (`mock.py`)

The deterministic, CI backend (the `capture.MockBackend` analog). It performs
no OS I/O; it **records** every event it is asked to perform into an ordered
`events: list[tuple[str, ...]]` (e.g. `("key_down", "w")`, `("key_up", "w")`,
`("mouse_click", "left")`) and tracks its own held keys/buttons so tests can
assert release-on-exception and the emergency stop's release-all. It also
supports an injectable `clock` and `sleeper` so hold/delay timing is testable
without real wall-clock sleeps (the dev env already has `freezegun`).

This is what makes the whole decision/safety layer testable headless:
`tests/test_actions_executor.py` asserts *exactly which events, in which
order*, and that a key held across a raised exception is still released.

### 5.3 `ActionQueue` (`queue.py`)

Bounded FIFO with a **priority lane** and cancellation:

* `enqueue(action) -> QueueHandle` — rejects (or, with `SafetyGuard`,
  downgrades) actions that would exceed `max_consecutive_actions`.
* `put_urgent(action)` — the priority lane. The emergency stop and
  release-all go here so they always preempt queued work.
* `get() -> Action | None`, `cancel(handle)`, `cancel_all()`, `clear()`,
  `pending` (property), `__len__`.
* Bounded: a hard `max_size` (from config); `enqueue` past the bound raises
  `ActionQueueError` (no unbounded growth — AGENTS.md "bounded durations / no
  infinite loops").

Pure logic, no backend, trivially testable.

### 5.4 `Executor` (`executor.py`)

The single action-execution abstraction (AGENTS.md "All mouse and keyboard
control must go through a single action-execution abstraction"). It is the
only object that may call an `InputBackend`, and only through the protocol:

* `execute(action) -> ActionResult` — validate → pre-delay → perform →
  post → return. For holds, the release is **guaranteed by construction**:

  ```python
  def _hold(self, action: Action) -> None:
      self._backend.key_down(action.key, action.modifiers)
      self._held_keys.add(action.key)
      try:
          self._sleeper(action.duration_ms / 1000.0)
      finally:
          self._backend.key_up(action.key, action.modifiers)
          self._held_keys.discard(action.key)
  ```

  The `try/finally` is the concrete realization of AGENTS.md's "prefer
  abstractions that guarantee release even if an exception occurs." A
  `KeyError` from the backend inside `sleeper` still releases the key.

* `enqueue(action) -> QueueHandle`, `run(batch) -> ActionLog` — sequence a
  batch through the queue, enforcing `max_action_duration_ms` per action and
  `max_consecutive_actions` per batch (both from config, via `SafetyGuard`).
* **Key/mouse-state tracking** — `held_keys: set[str]`, `held_buttons: set[str]`
  are the executor's source of truth (the mock backend mirrors them for
  assertions). `release_all()` iterates both and calls the backend `key_up` /
  `mouse_up` for each, clearing the sets.
* **Timing + cancellation + timeouts** — an injected `clock`/`sleeper`
  (defaults `time.perf_counter` / `time.sleep`); a per-action deadline derived
  from `max_action_duration_ms`; a `stopped` flag that, once set, makes every
  subsequent `execute` short-circuit to `ok=False, stop_event=True` without
  touching the backend.
* **State machine** — explicit, testable states: `IDLE → RUNNING → (IDLE |
  PAUSED | STOPPED)`. `PAUSED` (recoverable, e.g. after a bounded number of
  backend errors) and `STOPPED` (emergency) are distinct; entering `STOPPED`
  is terminal for the process lifetime and recorded. This is the Phase-3
  sibling of the Phase-4 state machine, applied to execution rather than game
  state.
* Dependency injection: `Executor(backend, safety, clock=..., sleeper=...)` —
  every collaborator injectable, no globals, no hidden I/O.

### 5.5 `SafetyGuard` + `EmergencyStop` (`safety.py`)

The AI-independent safety layer (AGENTS.md "Safety mechanisms are mandatory").

* `SafetyGuard` — validates a proposed `Action` against the configured bounds
  before it runs: `duration_ms <= max_action_duration_ms`, `delay_ms` bounded,
  key/mouse name in the known set, coordinate sanity. A violation raises
  `SafetyViolation` (or, in `ASSISTED` mode, is *reported* rather than
  executed — see §5.6). Enforces `max_consecutive_actions` across a batch.
* `EmergencyStop` — the hard stop. A single `trigger(reason)` call, idempotent,
  does all six things AGENTS.md requires, in this order:
  1. set the executor's `stopped` flag (stop the decision loop);
  2. `queue.cancel_all()` (cancel queued actions);
  3. `executor.release_all()` (release all held keys);
  4. release mouse buttons (inside `release_all`);
  5. mark the executor so further `execute` is a no-op that records
     `stop_event=True` (prevent further input);
  6. record a `StopEvent` (timestamp + reason + held-state snapshot) into the
     `ActionLog.stop_reason` / log (record the stop event).
* The **mode gate** lives here: `guard.check_mode(action)` raises `ModeViolation`
  for any input attempt under `OBSERVE_ONLY`. This is the one mode the whole
  project's safety story depends on, so it is built and regression-tested in
  Phase 3 (not Phase 6).

The `hotkey` that *fires* the stop on Windows (F12) is a `HotkeyListener`
protocol + a fake-triggerable implementation for tests; the `pynput`-based
listener is the Windows adapter in §5.7. The mechanism (stop on trigger) is
tested headless against the fake; only the physical key binding is
Windows-only.

### 5.6 Operating modes (`modes.py`)

```python
class OperationMode(str, Enum):
    OBSERVE_ONLY      = "observe_only"      # never emits input (hard gate)
    ASSISTED          = "assisted"          # propose; execute only on approval
    SEMI_AUTONOMOUS   = "semi_autonomous"   # execute safe actions; gate unknown
    AUTONOMOUS        = "autonomous"        # full execute path
```

Phase-3 scope of each:

* `OBSERVE_ONLY` — **fully built + regression-tested.** Any `execute`/`run`
  raises `ModeViolation` and the mock backend's `events` list is asserted
  empty. This is the safe default the CLI ships with.
* `AUTONOMOUS` — the normal `Executor` path (validate → perform).
* `ASSISTED` / `SEMI_AUTONOMOUS` — the **mechanism** only: `run()` in these
  modes returns a `PendingProposal` (the would-be `ActionLog`) instead of
  executing, and a separate `approve(proposal)` performs it. The human-approval
  *UX* (interactive `[Execute]/[Reject]/[Pause]`, Phase 6) and the
  "safe vs. unknown action" classifier (Phase 7) are **out of scope** here.
  Implementing the propose/approve split now keeps Phase 6 from having to
  retrofit it.

This is a deliberate scoping decision and is called out so the phase does not
accidentally absorb Phases 6–7.

### 5.7 `WindowsBackend` + hotkey listener (`windows.py`)

The real, platform, optional-adapter — the `MssBackend` analog. Backed by
`pynput` (see §6), providing `Keyboard`/`Mouse` via `pynput.keyboard` /
`pynput.mouse`, plus a `pynput.keyboard.Listener` that maps the configured
emergency-stop key (default `F12`) to `EmergencyStop.trigger()`. Rules:

* Imported **only** when the `windows` backend is selected, via
  `create_backend` (lazy) — a bare venv without the `input` extra never
  imports `pynput`, and `tests/test_actions_bare.py` proves that.
* `open()`/`close()` mirror the backend protocol; `close()` releases all held
  input (a leaked held key is the exact failure AGENTS.md calls out).
* The Windows backend itself is **not** exercised by the CI suite (no
  Windows display in CI). It is covered by a bare-import contract test, an
  availability/error-path test (absent lib → clean `InputBackendError` naming
  the extra), and manual testing on the user's Windows PC
  (`docs/manual-testing.md` §Phase 3).

---

## 6. New dependencies

One new optional extra, declared in `pyproject.toml` and **not** a hard
dependency (the core + contract + mock + safety all run with zero extras):

```toml
# Phase 3 input control. Real OS input on the target Windows machine.
# NOT a hard dependency: the action *contract*, queue, executor, and safety
# are dependency-free and CI-tested via MockBackend. pynput is imported only
# when the "windows" backend is selected (actions/windows.py, lazily).
input = [
    # Cross-platform keyboard/mouse + a global Listener for the emergency
    # hotkey. Chosen over PyDirectInput (Windows-only, no listener) and
    # PyAutoGUI (higher-level wrapper) — see §13.
    "pynput>=1.7,<2",
]
```

Notes / constraints:

* The **dev box is WSL2/Linux and shares `/workspace` with the user's Windows
  PC** (9p/drvfs mount). A venv created for one OS can be broken by the
  other's package manager pruning platform-specific wheels. The `input` extra
  (and any platform binary) must therefore be installed **only on the Windows
  box**, never into a venv the Linux dev/CI side also uses. This mirrors the
  `shared-mount-cross-os-venv` lesson. `uv sync --extra input` is a
  Windows-only step.
* `freezegun` is already in the `dev` extra and is used for deterministic
  hold/delay timing tests (no real sleeps).
* No other new dependencies. The queue, executor, safety, and contract are
  stdlib + the project's own types.

---

## 7. Testing strategy (TDD, per AGENTS.md)

Every component ships test-first. All tests deterministic, independent, and
runnable in CI **without a desktop session and without a live game** — the
`MockBackend` is the stand-in for the Windows backend, exactly as it is for
Phase 1's `MssBackend`.

### 7.1 Unit tests

| File | Covers |
|---|---|
| `test_action.py` | `Action` field validation per `kind` (required present / forbidden absent); `to_dict`/`to_json` round-trip; `schema_version`; `ActionResult`/`ActionLog` JSON-serializable; immutability |
| `test_actions_base.py` | `InputError` hierarchy subclassing; `create_backend("mock")` returns a `MockBackend`; unknown name → `ActionError`; `windows` with lib absent → `InputBackendError` naming the extra (bare venv) |
| `test_actions_config.py` | `InputConfig`/`ActionConfig` defaults + `extra="forbid"` fail-fast; `safety` bounds carried into the guard (config-first, mirrors Phase 2 step 2) |
| `test_actions_mock.py` | `MockBackend` records events in order; held keys/buttons tracked and released; `open`/`close` semantics |
| `test_actions_queue.py` | FIFO order; priority lane preempts; bounded `max_size` → `ActionQueueError`; `cancel`/`cancel_all`; idempotent re-cancellation |
| `test_actions_executor.py` | sequencing a batch into the expected event log; **release-on-exception** (a raised `sleeper` still emits `key_up`); key/mouse-state tracking; per-action timeout → `PAUSED`; `stopped` flag short-circuits further input; `run()` respects `max_consecutive_actions` |
| `test_actions_safety.py` | guard rejects over-bound durations/keys (→ `SafetyViolation`); **emergency stop** order: stop flag → cancel queue → release all → block → record; idempotent double-trigger; stop mid-hold releases the held key |
| `test_actions_modes.py` | `OBSERVE_ONLY` → `ModeViolation` **and** `mock.events == []` (the critical regression); `AUTONOMOUS` executes; `ASSISTED`/`SEMI_AUTONOMOUS` return a proposal without executing until `approve()` |
| `test_actions_bare.py` | the `actions` package imports and the executor/safety run in a venv with **no `input` extra** (bare-import contract, mirrors `test_dev_extra.py`) |

### 7.2 CLI / e2e (`e2e/test_e2e_act.py`)

Subprocess tests spawning the real entry point (the `analyze` e2e pattern):

| Test | Command | Assertions |
|---|---|---|
| `test_act_prints_actionlog_json` | `act --backend mock --demo --mode autonomous` | exit 0; stdout is a single JSON object; `json.loads` succeeds; `results` is an array; `stopped == false` |
| `test_act_observe_only_emits_nothing` | `act --backend mock --demo --mode observe_only` | exit 0; `results == []`; `stopped == false` (proof the gate is end-to-end, not just unit) |
| `test_act_error_routing` | `act --backend doesnotexist` | exit 1; stderr starts with `error:`; stdout empty (same contract as Phase 1/2) |
| `test_act_config_file` | `act --backend mock --config <tmp>.yaml` (tmp config with an `input:`/`safety:` block) | exit 0; config-driven bounds take effect (e.g. a bounded-duration action is rejected → `stopped == true` + `stop_reason` set) |

### 7.3 Definition of done (phase-level)

* `uv run pytest` green (unit + e2e).
* `uv run ruff check src tests` clean.
* **Bare-venv import contract:** `uv run ai-game-agent act --backend mock --demo`
  prints a valid `ActionLog` JSON with the `input` extra **not** installed.
* **`OBSERVE_ONLY` regression:** a test asserts the mock backend receives zero
  events under `OBSERVE_ONLY` (unit *and* e2e).
* **Emergency-stop regression:** a test proves stop mid-hold releases the held
  key and that a subsequent `execute` is a no-op.
* README updated: Phase 3 section, `input` extra install, `act` usage.
* `config/default.yaml` carries the `input:` block and the existing `safety:`
  block is wired (not just documented).
* `docs/manual-testing.md` gains a Phase 3 section (Windows `act` run + F12
  emergency-stop check).

---

## 8. Configuration

Wire the already-present `safety:` block and add an `input:` block to
`config/default.yaml`. All optional; a bare config still runs the contract and
the mock backend (Phase 3 must not require the `input` extra).

```yaml
# Input control (Phase 3). The backend is selected here or via --backend.
# "mock" is the deterministic, CI/backend-free default (records events, no
# OS input). "windows" needs the "input" extra (pynput) and a real desktop.
input:
  backend: mock            # mock | windows
  # Emergency-stop hotkey (Windows). Default F12, per the safety block.
  emergency_stop_keys:
    - F12
  # Default operating mode. OBSERVE_ONLY is the safe default: the shipped
  # CLI emits no OS input unless explicitly promoted.
  mode: observe_only       # observe_only | assisted | semi_autonomous | autonomous

# Safety (Phase 3) — existing block, now wired into SafetyGuard/Executor.
safety:
  max_action_duration_ms: 5000        # a single hold/delay must be under this
  max_consecutive_actions: 50         # a batch must stay under this
  # Bounded retry: after this many backend errors the executor PAUSES
  # (never an infinite loop).
  max_backend_errors: 3
```

Add the matching typed models in `config.py` (`_InputConfig` + a `SafetyConfig`
field for `max_backend_errors`, or a new `safety.max_backend_errors`), keeping
the existing frozen / `extra="forbid"` / public-view pattern so a typo fails
fast at load. `OperationMode` (in `actions/modes.py`) is the enum the config
string is validated against.

---

## 9. CLI: `act` subcommand

Add `act` to `__main__.py`, mirroring how `analyze` wires perception: build
the backend + guard + executor **first** (fail fast, no half-open resource),
then run. The `--backend mock` path is the headless/CI default and is the one
the e2e suite exercises; `--backend windows` requires the `input` extra and a
real desktop and is manual-test territory.

```text
ai-game-agent act --backend mock --demo --mode autonomous
ai-game-agent act --backend mock --script my_actions.json --mode assisted
ai-game-agent act --backend windows --demo --mode autonomous --confirm
```

Subcommand surface:

* `--backend` — `mock` | `windows` (default from `input.backend`, then `mock`).
* `--mode` — one of the four `OperationMode` values (default from
  `input.mode`, then `observe_only`).
* `--demo` — run a small, documented built-in action sequence (a few key
  presses + a hold + a click + a delay). Makes `act` demoable out of the box,
  the Phase-2 `analyze`-demo analog.
* `--script PATH` — load an `ActionLog`-shaped JSON file (a list of `Action`
  dicts) and run it. This is the replay surface: a recorded decision's action
  sequence can be re-run through queue → executor headless.
* `--record` / `--record-dir` — save the resulting `ActionLog` as JSON under
  `recordings/actions/` (the Phase-1 recording analog) for later replay.
* `--confirm` — required for `--backend windows` in non-`OBSERVE_ONLY` modes:
  a typed confirmation before any real input, with the emergency-stop hotkey
  armed and announced.
* `--pretty` — indented JSON out (matches `analyze`).

stdout carries the `ActionLog` JSON (or, in `ASSISTED`, the proposal); stderr
carries `error:` / usage — the same stream-routing contract the e2e tests
assert for Phase 1/2.

---

## 10. Error handling

Typed, classified, bounded — per AGENTS.md:

* **`ActionValidationError`** — a `kind` with missing required / present
  forbidden fields. Raised in `validate_action` before any backend call.
* **`SafetyViolation`** — an action exceeds a configured bound (duration,
  batch size, unknown key). In `AUTONOMOUS`/`SEMI_AUTONOMOUS` this pauses the
  batch and is recorded; in `ASSISTED` it is reported, not executed.
* **`InputBackendError`** — the backend I/O failed (missing lib, OS error).
  Bounded retry: after `safety.max_backend_errors` failures the executor
  transitions to `PAUSED` and stops the batch (no infinite retry loop).
* **`ModeViolation`** — input attempted under `OBSERVE_ONLY`. Always raised;
  never "recovered" (it is a contract violation, logged as a safety event).
* **Emergency stop** — the terminal, recorded path. `StopEvent` (timestamp,
  reason, held-state snapshot) is written to the log and surfaced in the
  `ActionLog.stop_reason`.

No errors are silently ignored: every `ActionResult` carries `ok` + `error`,
and every batch is either `ok` or carries its failure/stop reason. This makes
the AGENTS.md question *"Why did the agent perform this action?"* answerable
from the `ActionLog` alone.

---

## 11. Out of scope (deliberately)

* **Deciding what to act on** — Phase 4 (state machine) and Phase 5 (AI)
  propose actions; Phase 3 only executes them. No perception→action wiring.
* **The assisted-approval UX** — Phase 6. Phase 3 ships the propose/approve
  *mechanism* (`PendingProposal` / `approve()`), not the interactive
  `[Execute]/[Reject]/[Pause]` flow or the safe/unknown action classifier.
* **Autonomous-loop orchestration** — the continuous
  observe→decide→act→observe loop and inactivity timeouts are Phase 7.
* **Multi-monitor / relative-mouse coordinate mapping beyond the raw API** —
  the contract carries absolute or relative coordinates; the Windows backend
  passes them through. Game-specific coordinate logic is Phase 4+.
* **Yielding to the game / input injection into a specific window** — the
  backend uses global desktop input (SendInput); targeting a specific window
  is a later concern, not Phase 3.

---

## 12. Suggested build order (each step = red → green → refactor)

1. `Action` + `ActionKind` + `ActionResult` + `ActionLog` + `validate_action`
   + `InputError` hierarchy (no deps).
2. `base.py` protocols + `MockBackend` (event recording) + `create_backend`
   factory + `InputConfig`/`ActionConfig` in `config.py` + `default.yaml`
   `input:` block (config tests first).
3. `OperationMode` enum + `SafetyGuard` (mode gate + bounded limits) — with
   fakes. `OBSERVE_ONLY` gate proven first (the safety-critical regression).
4. `ActionQueue` — order, priority lane, bounds, cancellation.
5. `Executor` — sequencing, key/mouse-state, **release-on-exception**
   (try/finally), timing via injectable clock/sleeper, `stopped` short-circuit,
   `IDLE/RUNNING/PAUSED/STOPPED` state machine. (The meat; all against
   `MockBackend`.)
6. `EmergencyStop` — idempotent trigger; stop→cancel→release-all→block→record
   order; stop-mid-hold regression.
7. `HotkeyListener` protocol + fake-triggerable listener (mechanism) — trigger
   → `EmergencyStop` proven headless.
8. `windows.py` — `WindowsBackend` (pynput) + real hotkey listener + `input`
   extra in `pyproject.toml` + lazy import + bare-venv import-contract test.
9. `act` CLI subcommand + `ActionLog` JSON out + `--demo`/`--script`/
   `--record`/`--confirm` + e2e suite (JSON shape, exit codes, error routing,
   `OBSERVE_ONLY` no-op, config wiring).
10. README Phase 3 section + `config/default.yaml` example + docs cross-links +
    `docs/manual-testing.md` Windows section.

Steps 1–7 need **no new dependencies** and are fully CI-runnable in a bare
venv. Step 8 adds the `input` extra (Windows-only install). Step 9 is
e2e/CLI. This ordering keeps the package importable and its contract tested
long before `pynput` is ever imported — the same property Phase 2 held for
`vision`/`ocr`.

---

## 13. Open questions (resolve before step 8)

* **Input library choice** — *recommend `pynput`.* Cross-platform, maintained,
  and ships a global keyboard `Listener` (needed for the F12 emergency-stop)
  plus a mouse API. Alternatives: `PyDirectInput` (lower latency, better for
  gaming input, but Windows-only and no listener — the stop hotkey would need
  a second lib) and `PyAutoGUI` (higher-level, cross-platform, built on the
  other two, but a heavier wrapper). Decision affects only `windows.py` and
  the `input` extra; the contract/executor are unaffected. **Needs the
  user's sign-off** since it names the dependency.
* **Cross-OS venv hazard** — *confirm.* The dev box (WSL2/Linux) and the user's
  Windows PC share `/workspace` on a 9p/drvfs mount; one `.venv` cannot serve
  both OSes safely (each side's package manager prunes the other's platform
  wheels — the `shared-mount-cross-os-venv` lesson). The `input` extra must be
  installed **only on the Windows box**, never into a venv the Linux dev/CI
  side shares. Confirm the Windows box is where real input + the F12 listener
  will actually run.
* **Scope of `ASSISTED`/`SEMI_AUTONOMOUS` in Phase 3** — *recommend: mechanism
  only.* Phase 3 ships the propose/approve split (`PendingProposal` /
  `approve()`) so Phase 6 does not retrofit it, but the interactive approval
  UX and the safe/unknown classifier stay in Phases 6–7. Confirm the user is
  okay with Phase 3 not including the approval UI.
* **Emergency-stop hotkey in CI** — *recommend: mechanism + fake now, pynput
  listener as the Windows adapter.* The physical F12 binding cannot be tested
  headless; the "trigger → stop → release-all → block → record" behavior is
  tested against a fake listener. Confirm this split is acceptable.
* **Action script format** — *recommend: `ActionLog`-shaped JSON file
  (`--script`) + a built-in `--demo`.* A list of `Action` dicts is
  human-readable, replayable, and already what `act` prints. Confirm no
  preference for a richer/typed format (e.g. a named "recipe").

---

## 14. Status & remaining work

Phase 3 has no code yet (branch `feature/phase-3` sits at the Phase 2 merge).
Pre-existing Phase-3-adjacent surface already in the tree, to be *wired* not
re-invented:

* `config/default.yaml` already carries a `safety:` block (emergency_stop_keys,
  max_action_duration_ms, max_consecutive_actions) — currently dead surface.
* `SafetyConfig` pydantic model + public `SafetyConfig` view already exist in
  `config.py` — Phase 3 adds `max_backend_errors` and wires the block into
  `SafetyGuard`/`Executor`.
* `docs/high-level-development-plan.md` §9 (Action Executor), §10 (Safety),
  and §13 (Human-in-the-Loop) are the design intent Phase 3 realizes.

Remaining work, item by item (all steps 1–10 above), to be tracked as the
`feature/phase-3` build proceeds.
