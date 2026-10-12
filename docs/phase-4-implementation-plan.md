# Phase 4: The Deterministic Game-Play Loop — Implementation Plan

**Status:** plan
**Branch:** `feature/phase-4`
**Date:** 2026-10-11
**Prerequisite:** Phase 3 complete (`Executor` + `SafetyGuard` + `MockBackend` + `act` command; full suite green)

---

## 1. Purpose


> **Contract validation note (2026-10-12).** Implement against the live Phase 1–3 APIs: `Capture` in `capture.py` with `start()`/`stop()` lifecycle; `wait_next_frame()` and `config.fps` for pacing; `CaptureError` for capture failures; `Observation.ui_zones`, `templates`, `objects`, and `detector_errors`; actual `ActionKind` values and `duration_ms`; `threading.Event` for cancellation; and the existing multi-layer config wiring. Use a scripted capture fake where the real API has no queued-frame fixture. Nest cleanup so capture and executor resources are closed even if cleanup itself raises. Event sequence numbers are session-wide and assigned by the loop. Detector errors are data, not exceptions, and hit coordinates are frame-space after capture region/scale.


Phase 3 can *see* (Phase 2: `perception.analyze`) and can *act* (Phase 3:
`actions.act`) — but in two one-shot CLI invocations that a human drives.
Phase 4 closes the loop: a **deterministic controller** that repeatedly
observes, classifies the game state, chooses and executes the state's action
sequence, and recovers itself when things go wrong — **with no AI/VLM in the
decision path**.

This is the project's core architecture principle made concrete:

```text
Fast deterministic processing
            │
            ▼
       Known state?
        /       \
      yes        no
       │          │
       ▼          ▼
 State Machine    AI   ← Phase 5, out of scope here
       │          │
       └────┬─────┘
            ▼
       Action Plan
```

The AI branch is explicitly **deferred**: Phase 4 ships a complete, playable,
fully testable loop where every decision is deterministic code. Phase 5 will
mount an `AI_ANALYSIS` state that calls the Ollama VLM; the seams for that
(watchdog timeouts, `paused` safety state, the AI state slot) are built in this
phase on purpose.

### 1.1 What "deterministic" buys us here

- **CI-safe.** No Ollama, no desktop, no model download: the loop runs against
  the real `Capture` facade with a scripted test fake implementing the actual capture contract, plus `actions.MockBackend` input in the test suite, which is
  the project's stated "primary mechanism for regression testing."
- **Replayable.** Every decision is a pure function of `(Observation, state,
  clock, attempt-counts)` — replay tests can feed recorded observations and
  assert the exact action sequence.
- **Auditable.** Every transition carries a reason string into the event log
  ("why did the agent do this?"), satisfying the logging requirement.

### 1.2 What is NOT in Phase 4

| Deferred | Why |
| --- | --- |
| Ollama / VLM / LLM integration | Phase 5. The `AI_ANALYSIS` state is defined as a *slot* (watchdog + `paused` fallback); the model call is a stubbed strategy object so Phase 5 swaps in a real client behind the same interface. |
| Episodic/persistent memory | Phase 6. Phase 4 writes a session event log (`jsonl`) that is the *input format* Phase 6 will consume. |
| Target selection / combat heuristics beyond state entry-conditions | Kept minimal (click the loot marker when present, click the primary ability when in combat) — the real value is the loop skeleton, not game skill. |
| Multi-window / multi-monitor | Single full-screen `monitor_index=0` capture (Phase 2 contract). |
| `semi_autonomous` semantics | Mode plumbing exists (Phase 3); the loop's default mode stays `observe_only` until the user promotes it — Phase 4 does not change the safety defaults. |

---

## 2. Inputs (Phase 2/3 contracts Phase 4 consumes)

| Contract | Source | What Phase 4 uses |
| --- | --- | --- |
| `Observation` (`perception/observation.py`) | frozen dataclass | `templates`, `ui_zones`, `objects`, `detector_errors`, and the other fields defined by the frozen `Observation` dataclass. Consume real hit collections (`TemplateHit`, `UiZoneHit`, `ObjectHit`), not a nonexistent `get_zone()`/`UiZoneHit` / `TemplateHit` / `ObjectHit` API. |
| `Perception.observe(frame) -> Observation` (`perception/pipeline.py`) | pure pipeline | The per-tick perception call. Constructed once, reused across the loop (detector state is per-frame, so reuse is safe). |
| `Capture` / `CaptureBackend` (`capture.py`) | `start()`/`stop()` (or context manager), `grab()`, `wait_next_frame()`, and `config.fps` | Start before the loop and always stop in nested `finally` cleanup; there is no per-frame `release()` contract. |
| `Capture` (`capture.py`) | The existing `MockBackend` returns a synthetic frame; it is not a queued-frame test fixture. Use a scripted test fake implementing `start()`/`stop()`/`grab()`/`wait_next_frame()` and counters; do not rely on undocumented queue/failure constructor arguments. |
| `Executor` + `Action` / `ActionResult` / `ActionLog` (`actions/*`) | Phase 3 | One executor instance for the whole loop (hotkey armed, backend open). `execute()` returns `ActionResult(ok=...)`. |
| `MockBackend` (`actions/mock.py`) | `events` list, `held_keys`, `held_buttons` | Test double asserting the exact emitted input sequence. |
| `SafetyGuard` (`actions/safety.py`) | mode gate, bounded errors, release-all on stop | Unchanged; Phase 4 respects the same `paused`/stop semantics the guard enforces. |
| `HotkeyListener` / `FakeHotkeyListener` (`actions/hotkey.py`) | `arm`/`disarm`/`trigger` | The loop's `run(stop_event)` must **cooperate with the F12 hotkey**: the hotkey callback (registered via the executor's backend) must be able to stop the loop, not just cancel one action. |
| Config (`config.py`) | `_PerceptionConfig`, `_InputConfig`, `_SafetyConfig` | Phase 4 adds one new frozen model, `_AgentConfig` (see §4.4), alongside them. |
| CLI (`__main__.py`) | `parse_args`, `_build_perception`, `_build_capture`, `_build_executor`, `_resolve_actions`, `_record_action_log` | Phase 4 adds one subcommand, `run`, reusing all four builders verbatim. |
| Stop primitive | `threading.Event` | Use as the loop's cooperative stop primitive; no project-specific stop-token abstraction exists. |

---

## 3. New module map (Phase 4)

```text
src/ai_game_agent/
    state/
        __init__.py          # exports: AgentState, GameStateMachine, state transitions, recovery
        states.py            # AgentState enum + AgentStateError (mirrors OperationMode style)
        events.py            # LoopEvent dataclass + EventType enum (structured event log)
        machine.py           # GameStateMachine: transitions, attempts, watchdog, recovery
        policy.py            # StatePolicy Protocol + DeterministicPolicy (the "Action Plan" table)
    agent/
        __init__.py          # exports: AgentLoop, AgentLoopReport
        loop.py              # AgentLoop: the run/tick orchestration (owns capture+perception+executor)
        report.py            # AgentLoopReport: per-session summary (ticks, transitions, results, errors)
    games/
        wow/
            __init__.py      # exports the WoW zone-name → state mapping used by the default policy
            zones.py         # ZONE_HEALTH_BAR, ZONE_LOOT_GLOW, ZONE_TARGET, ... canonical zone names
            policy.py        # wow_policy: maps WoW zone results to deterministic state choices
    config.py                # + _AgentConfig (new frozen model), + AgentConfig on Config
    __main__.py              # + `run` subcommand (loop driver), + `--once` single-tick mode for testing

tests/
    conftest.py              # + agent_config fixture, + zone_result()/observation() factory helpers
    test_state_states.py     # step 1
    test_state_machine.py    # steps 2 + 4 (transitions, recovery, watchdog)
    test_state_policy.py     # step 3 (per-state action plans, mode gating)
    test_agent_loop.py       # step 5 (tick integration: mock capture + mock input)
    test_agent_loop_safety.py# step 6 (stop/hotkey/error paths)
    test_run_cli.py          # step 7 (CLI: --once, --max-ticks, --record, --dry-run)
    test_replay.py           # step 8 (record → replay assertions)
```

New public API is additive: nothing from Phases 2/3 changes signature. The
`state/` and `agent/` packages are imported lazily by `__main__.py` exactly
like the `actions` package (keeping the CLI cold-start contract).

---

## 4. Data & configuration design

### 4.1 `AgentState` (states.py)

```python
class AgentState(StrEnum):
    IDLE = "idle"
    COMBAT = "combat"
    LOOTING = "looting"
    RECOVERY = "recovery"
    AI_ANALYSIS = "ai_analysis"   # slot for Phase 5; Phase 4 never *enters* it — it
    #                             # falls through to RECOVERY/PAUSED instead
    PAUSED = "paused"            # terminal-ish safe state: executor stopped, loop exits

class AgentStateError(ValueError):
    ...
```

Use `StrEnum`, matching the project's `OperationMode` style: config
validation is closed-set, `extra="forbid"` applies to the config model, and
`str(state)` round-trips through JSON.

### 4.2 `LoopEvent` (events.py) — the audit trail

One frozen dataclass per tick/decision, append-only per session:

```python
@dataclass(frozen=True)
class LoopEvent:
    ts: float                      # monotonic-safe epoch seconds
    seq: int                       # 0-based session-wide; assigned by AgentLoop
    event: "EventType"             # TICK, STATE_ENTER, STATE_EXIT, ACTION_BATCH,
                                   # ACTION_RESULT, RECOVERY, STOP, ERROR
    state_before: AgentState | None
    state_after: AgentState | None
    reason: str                    # human-readable "why" (log requirement)
    detail: dict                   # structured payload (zone values, action names,
                                   # error class, attempt index) — JSON-serializable
```

`EventType` is a closed `str, Enum`. Every transition, every executed action
batch, every error, and every stop writes ≥1 event with a `reason`. This is
the artifact that answers "why did the agent do this?" and that step 8's
replay tests assert against.

### 4.3 `GameStateMachine` (machine.py) — the deterministic core

```python
class GameStateMachine:
    def __init__(self, *,
                 initial: AgentState = AgentState.IDLE,
                 recovery_attempts: int = 3,
                 state_timeout_s: float = 5.0,
                 watchdog_states: frozenset[AgentState] = frozenset(
                     {AgentState.RECOVERY, AgentState.AI_ANALYSIS}),
                 policy: StatePolicy,
                 on_transition: Callable[[AgentState, AgentState, str, float], None] | None = None):
        ...

    state: AgentState                       # current
    def decide(self, observation: Observation, now: float) -> AgentState:
        """Use injected policy.transition_signals(observation) plus internal
        attempt/timeout state and now to choose the next state.

        No I/O. No Action creation. This keeps the test matrix small and the
        replay contract exact.
        """
        ...
    def enter(self, new_state: AgentState, reason: str, now: float) -> None:
        """Record transition, reset per-state attempt counter, notify callback."""
        ...
    def note_recovery_attempt(self) -> int:
        """Bounded recovery: increments a counter; returns the new count.
        Callers must stop retrying once count >= recovery_attempts."""
        ...
    def reset_recovery(self) -> None:
        ...
```

Transition table (Phase 4's deterministic core, WoW-flavored but expressed
entirely through the **generic** `Observation.zone` API — the table itself
lives in `policy.py` per game; the machine only enforces the *shape*):

| From | Observation condition (via `policy`) | To | Reason (logged) |
| --- | --- | --- | --- |
| IDLE | `health_bar.percent < 0.30` | RECOVERY | "low health on idle scan" |
| IDLE | `loot_glow.present` | LOOTING | "loot marker visible" |
| IDLE | `target.present and target.hostile` | COMBAT | "hostile target on frame" |
| IDLE | none of the above | IDLE | "no actionable zone signal" |
| COMBAT | `target.present` is false (target died/lost) | IDLE | "target lost, exiting combat" |
| COMBAT | `health_bar.percent < 0.30` | RECOVERY | "health floor breached in combat" |
| LOOTING | `loot_glow.present` is false (picked up) | IDLE | "loot cleared" |
| RECOVERY | `health_bar.percent >= 0.50` | IDLE | "health recovered above floor" |
| RECOVERY | recovery attempts exhausted | PAUSED | "recovery attempts exhausted; safe pause" |
| any | capture or perception raised this tick | RECOVERY | "observation failure this tick" |

**Rules the machine enforces (independent of the policy):**
1. **Watchdog timeout.** If the machine has stayed in any
   `watchdog_states` state for ≥ `state_timeout_s` (measured at `decide()`
   time via the injected `now`), it forces `PAUSED` with reason
   `"state timeout (watchdog)"`. This is the *only* way Phase 4 escapes a
   stuck RECOVERY loop without a health reading, and it is the exact seam
   Phase 5's `AI_ANALYSIS` will hang off: a VLM call that never returns a
   decision trips the same watchdog.
2. **Bounded recovery.** `note_recovery_attempt()` is a counter, not a
   loop: after `recovery_attempts` (default 3) consecutive RECOVERY entries
   without a healthy exit, the machine goes `PAUSED`. No infinite retry —
   the project's explicit rule.
3. **Self-loop idempotence.** `decide()` returning the *same* state as current
   is not a transition: no event, no counter reset. Only genuine changes go
   through `enter()`.
4. **No I/O, no side effects.** The machine never touches the executor or capture. It asks its injected
   policy to interpret the observation, then returns a state and emits a
   transition descriptor; the loop owns session-wide event sequencing.


**Signal and health heuristics (required, not implied by detector output).** `Observation` has no health percentage, `present` property, `get_zone()`, or `ZoneResult`. Define a small typed signal object returned by `StatePolicy.transition_signals()` with `health_fraction: float | None`, `loot_present: bool | None`, `target_present: bool | None`, `target_hostile: bool | None`, and a detector-health/degradation flag. `None` means unknown/not configured/not detected; it must not be silently treated as `False` or as 0% health. A `UiZoneHit.value` is the normalized value of a configured brightness/color/presence check, not a WoW health fraction by definition. Phase 4 must either (a) configure a documented deterministic health-bar measurement whose output is calibrated to `[0, 1]`, or (b) leave health as unknown and avoid health-threshold transitions; it must not pretend a generic UI-zone value is health percent. Likewise, target hostility must not be inferred from simple target-zone presence unless the configuration explicitly defines that check. The policy owns these game-specific heuristics and their tests.

**Detector degradation policy.** A detector listed in `Observation.detector_errors` failed for that frame, but the `Observation` itself remains valid. Define required vs optional detectors in the WoW adapter. One optional-detector error or an empty hit collection is not automatically a recovery event. On a required detector error, preserve any independent usable signals and mark affected signals unknown. Configure and test a bounded consecutive-failure threshold (default 3 consecutive ticks for a required detector); reset the counter after a successful observation of that detector. At threshold, emit a structured error/recovery event and enter `RECOVERY`; if recovery attempts are exhausted or the state watchdog expires, enter `PAUSED`. Capture-level `CaptureError` is a separate failure path and must not be conflated with detector errors.


### 4.4 `StatePolicy` (policy.py) — the "Action Plan" table

```python
class StatePolicy(Protocol):
    """Maps (state, observation) -> the concrete Action[] to run this tick.

    The policy is where game knowledge lives (wow/policy.py). The machine and
    loop stay game-agnostic — this is the GameAdapter boundary from AGENTS.md,
    realized before `games/` even exists.
    """
    def name(self) -> str: ...
    def actions(self, state: AgentState, observation: Observation,
                mode: OperationMode) -> tuple[Action, ...]: ...
    def transition_signals(self, observation: Observation) -> dict[str, object]: ...
        # Typed, documented generic signal contract consumed by the machine;
        # game-specific zone names stay inside games/wow/policy.py.
```

`DeterministicPolicy` (in `games/wow/policy.py`) implements the table:

- `IDLE`: `()` — wait for the next tick. (Phase 4 does not move the camera;
  that's a Phase 5/6 concern.)
- `COMBAT`: `(Action(kind=ActionKind.KEY_PRESS, key="q"), Action(kind=ActionKind.DELAY, duration_ms=120),
  Action(kind=ActionKind.KEY_PRESS, key="w"))` — a deterministic GCD-ish rotation. (Keys are
  configurable via `agent.combat_keys` in config; defaults are WoW-friendly
  but the loop does not hard-code them.)
- `LOOTING`: `(Action(kind=ActionKind.MOUSE_CLICK, button="left", clicks=1,
  x=<hit center x>, y=<hit center y>), Action(kind=ActionKind.DELAY, duration_ms=200),
  Action(kind=ActionKind.KEY_PRESS, key="f"))` — click the loot marker, wait, pick up.
  The click coordinates must come from the selected hit's actual coordinate fields. These are frame-space coordinates after capture region/scale, not inherently absolute desktop coordinates.
- `RECOVERY`: `(Action(kind=ActionKind.KEY_PRESS, key="r"), Action(kind=ActionKind.DELAY, duration_ms=300))`
  — re-use the "res" hotkey the Phase 2/3 demo already defined.
- `AI_ANALYSIS`: `()` in Phase 4 (the slot is defined; the action list stays
  empty until Phase 5 fills it).
- `PAUSED`: `()` — no input.

**Mode gating.** `policy.actions(state, obs, mode)` consults `mode`:
under `OBSERVE_ONLY` it returns `()` for every state (the loop still
classifies states and writes events — *seeing* and *deciding* is allowed,
*acting* is not). Under `ASSISTED`/`SEMI_AUTONOMOUS`/`AUTONOMOUS` it returns
the full plan. The **executor's `SafetyGuard`** then re-checks the mode at
dispatch — defense in depth, identical to Phase 3's `act` behavior.

### 4.5 `_AgentConfig` (config.py) — one new frozen model

```python
class _AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # The loop
    tick_hz: float = Field(default=2.0, gt=0, le=60.0)     # 2 Hz ≈ 500ms/tick
    max_ticks: int | None = Field(default=None, gt=0)      # None = run until stopped
    state_timeout_s: float = Field(default=5.0, gt=0)      # watchdog (machine)
    recovery_attempts: int = Field(default=3, gt=0)        # bounded recovery
    # WoW-flavored defaults, overridable per game later
    combat_keys: tuple[str, ...] = ("q", "w")
    # Used only when a calibrated health-fraction signal is available.
    health_low: float = Field(default=0.30, gt=0.0, le=1.0)
    health_recovered: float = Field(default=0.50, gt=0.0, le=1.0)
    required_detector_failures: int = Field(default=3, gt=0)
    required_detector_failures: int = Field(default=3, gt=0)
    loot_pickup_key: str = "f"
    # Event log
    record_dir: str = "recordings"
```

Wire the new frozen agent settings through the existing configuration chain (`_AgentConfig` → `_Config` → public `Config` → `AgentConfig`); adding a field to the public facade alone is insufficient. Keep default construction backward-compatible. `default.yaml` and
`development.yaml` each gain an `agent:` block (documented, not required —
Pydantic fills defaults).

`extra="forbid"` means a typo (`tick_hzz`) is a schema error at load, matching
the Phase 2/3 config contract.

### 4.6 `AgentLoop` (agent/loop.py) — the orchestration

```python
class AgentLoop:
    def __init__(self, *,
                 capture: Capture,
                 perception: Perception,
                 executor: Executor,
                 machine: GameStateMachine,
                 policy: StatePolicy,
                 mode: OperationMode,
                 tick_hz: float,
                 max_ticks: int | None,
                 now_fn: Callable[[], float] = time.time,   # injectable clock
                 stop_event: threading.Event | None = None,
                 on_event: Callable[[LoopEvent], None] | None = None):
        ...

    def run(self) -> AgentLoopReport:
        """Blocking loop: grab → observe → decide → act → sleep, until
        stop_event, max_ticks, or PAUSED. Returns the report.

        Invariants:
        - One capture grab per tick; released in `finally` (Phase 2 contract).
        - One Perception call per tick; `detector_errors` are observation data, not raised exceptions.
        - Executor.execute called only for non-empty action lists.
        - Every tick writes ≥1 LoopEvent (TICK + possibly STATE_*/ACTION_*).
        - Any exception from capture/perception is classified (PERCEPTION_ERROR
          / ACTION_ERROR / STATE_ERROR) and routed to RECOVERY, not propagated.
        """
        ...

    def tick(self) -> TickResult:
        """One full iteration — exposed for --once mode and replay tests."""
        ...
```

`AgentLoopReport` (report.py): frozen dataclass with
`ticks: int`, `transitions: tuple[LoopEvent, ...]`,
`actions_attempted: int`, `actions_ok: int`, `actions_failed: int`,
`errors: tuple[LoopEvent, ...]`, `final_state: AgentState`,
`stopped_reason: str | None`, `wall_time_s: float`. JSON-serializable via a
`to_dict()` (mirroring `Observation.to_dict()`).

**Cooperative stop.** `run()` checks `stop_event.is_set()` at the
top of each tick and before each action dispatch. The Phase 3 emergency-stop
path is owned by `Executor.trigger_emergency_stop()` and its configured
hotkey callback; do not invent a separate `HotkeyListener` injection point
unless the existing backend exposes one. Wire a loop-stop callback through
the actual callback seam. A stop requested during `Executor.execute()` may
interrupt that action via the executor's emergency-stop path; otherwise stop
before the next action and never dispatch further actions.

**Clock and pacing.** Inject a clock for deterministic watchdog tests. In production, use `Capture.wait_next_frame()` and `capture.config.fps` for pacing; `fps <= 0` disables pacing. Do not add a second `time.sleep(1 / tick_hz)` on top of the capture facade, which would double-pace the loop. If Phase 4 requires an independent tick rate, define how it maps to `capture.config.fps` and ensure only one pacing mechanism is active.

---

## 5. Step-by-step TDD sequence

Every step is: **red** (failing test written and observed to fail for the
*right* reason) → **green** (minimum code) → **refactor**. The project
mandates TDD; no production code lands without its red test.

### Step 1 — `AgentState` + `AgentStateError` (states.py)

**Contract (tests first):**
- `AgentState` is a `str, Enum` with exactly the 6 values in §4.1.
- `AgentState("idle")` round-trips; `AgentState("bogus")` raises `AgentStateError`
  (a `ValueError` subclass) — same discipline as `OperationMode` (Phase 3).
- `str(AgentState.COMBAT) == "combat"`.
- `AgentState` is hashable and usable in `frozenset`.

**Red test:** `tests/test_state_states.py` — import `from
ai_game_agent.state.states import AgentState, AgentStateError`; the import
itself fails (module does not exist) — *correct red reason: missing module,
not a typo*.

**Green:** add `state/__init__.py` + `states.py`. Re-export from
`state/__init__.py`.

**Verify:** `pytest tests/test_state_states.py -v` green; full suite still green.

---

### Step 2 — `GameStateMachine.decide()` + transitions (machine.py)

**Contract (tests first):**
- `GameStateMachine.decide(obs, now)` returns the state per the §4.3 table,
  driven by a *stub* `policy.transition_signals(obs)` that returns the zone
  values. (The machine does not know zone names — it consumes a `dict` of
  signals. This keeps the machine game-agnostic.)
- `enter()` fires `on_transition` with a `LoopEvent` carrying
  `event=STATE_ENTER`, `state_before`, `state_after`, and `reason`.
- Self-loops (decide returns current state) do **not** fire a transition
  callback and do **not** reset the recovery counter.
- `note_recovery_attempt()` returns 1, 2, 3; after the 3rd call, the next
  `decide()` returns `PAUSED` (recovery exhausted), *regardless* of the
  observation.
- `reset_recovery()` zeroes the counter (called on a healthy exit from
  RECOVERY).
- **Watchdog:** if `decide()` is called with `now` ≥
  `state_entered_at + state_timeout_s` while in a `watchdog_states` state, it
  returns `PAUSED` with reason `"state timeout (watchdog)"`.
- **Watchdog does NOT fire** in non-watchdog states (IDLE/COMBAT/LOOTING can
  persist indefinitely without a timeout — the loop is bounded by `max_ticks`
  and the hotkey, not by a watchdog on them).

**Red test:** `tests/test_state_machine.py` — construct the machine with a
stub `policy` that returns fixed signals; assert each row of the §4.3 table
and the watchdog/recovery rules. Use a fake clock (`now = 1000.0` etc.) so no
real time elapses.

**Green:** implement `machine.py`. Keep it ~120 lines: `decide()` is a
dispatch on `self.state` + the signals dict; `enter()` is the only writer of
`self.state`.

**Refactor:** pull the transition table into a private `_TRANSITIONS` dict so
the code reads as data, not nested `if`.

**Verify:** full suite green.

---

### Step 3 — `StatePolicy` + WoW `DeterministicPolicy` (policy.py, games/wow/)

**Contract (tests first):**
- `policy.actions(AgentState.COMBAT, obs, mode=OperationMode.AUTONOMOUS)`
  returns the §4.4 action tuple (exact keys/delays from config).
- `policy.actions(any_state, obs, mode=OperationMode.OBSERVE_ONLY)` returns
  `()` for every state.
- `policy.actions(AgentState.LOOTING, obs, mode=...)` uses the zone's marker
  `cx`/`cy` for the click, not a hard-coded point.
- `policy.transition_signals(obs)` returns a dict with keys
  `health_low`, `loot_present`, `target_hostile`, `health_recovered` — the
  *exact* keys the machine's `decide()` consumes (this is the seam; the test
  asserts the key set is stable so the machine can rely on it).
- `wow_policy` reads zone names from `games/wow/zones.py` constants — the
  policy is the *only* place in the codebase that knows the literal strings
  `"health_bar"`, `"loot_glow"`, etc.

**Red test:** `tests/test_state_policy.py` + a small `observation_factory`
helper in `conftest.py` that builds an `Observation` with specific zone
results (reusing Phase 2's `UiZoneHit` / `TemplateHit` / `ObjectHit` constructors).

**Green:** `games/wow/zones.py` (constants), `games/wow/policy.py`
(`DeterministicPolicy`), `state/policy.py` (the `Protocol`).

**Verify:** full suite green; no literal zone-name string appears outside
`games/wow/` and the test files.

---

### Step 4 — `LoopEvent` + `GameStateMachine` recovery/watchdog integration (events.py + machine.py)

This step closes §4.2 (the event model) and wires the machine's
`on_transition` callback to emit `LoopEvent`s. The machine already fires
callbacks in step 2; this step defines the *shape* it fires.

**Contract (tests first):**
- `LoopEvent` is frozen, JSON-serializable via a `to_dict()`, and its `detail`
  dict is always a `dict[str, JSON-compatible]`.
- Every `enter()` call in step 2's tests now produces a `LoopEvent` with
  `event=STATE_ENTER`, `state_before`/`state_after` set, and `reason` matching
  the §4.3 table's reason strings.
- `seq` is monotonically increasing per machine instance (0-based), assigned
  by the machine, not by the callback.

**Red test:** extend `tests/test_state_machine.py` with event-shape
assertions (the step-2 tests used a bare `list.append` callback; they now
assert the `LoopEvent` fields).

**Green:** `state/events.py`; update `machine.py` to build `LoopEvent`s.

**Verify:** full suite green.

---

### Step 5 — `AgentLoop.tick()` + `run()` (loop.py, report.py)

**Contract (tests first):**
- `tick()` performs one capture grab, one `perception.observe()` call, one `machine.decide()`, and at most one `executor.execute()` call per planned action. Record each result before deciding whether to continue.
- `tick()` under `OBSERVE_ONLY` performs the grab + perception + decide but
  **zero** `executor.execute()` calls (the policy returns `()` under that
  mode, so the loop's dispatch loop has nothing to dispatch — asserted via
  `MockBackend.events == []`).
- `tick()` records a `TICK` event with `detail={"zones": {...}}` (the zone
  names + their `present` flags) so a log reader can reconstruct the decision
  without re-running perception.
- `run()` with `max_ticks=3` and a 3-frame scripted capture fake returns an
  `AgentLoopReport` with `ticks == 3`, `final_state` matching the last
  `decide()` output, and `stopped_reason == "max_ticks"`.
- `run()` with `stop_event.is_set()` before the first tick returns immediately with
  `ticks == 0` and `stopped_reason == "stop_requested"`.
- `run()` with a a scripted fake that raises `CaptureError` from `grab()` (capture-error path)
  routes to RECOVERY, and after `recovery_attempts` ticks, `PAUSED`.
- `run()` exits on `PAUSED` (the machine's terminal state) with
  `stopped_reason == "paused"` and the executor released (hotkey disarmed,
  backend closed).

**Red test:** `tests/test_agent_loop.py` — the import of `ai_game_agent.agent`
fails (module missing).

**Green:** `agent/loop.py` + `agent/report.py`. The `tick()` body is:

```python
def tick(self) -> TickResult:
    frame = self._capture.grab()
    try:
        observation = self._perception.observe(frame)
    except Exception as exc:
        # PERCEPTION_ERROR: route to RECOVERY, record, return
        ...
    next_state = self._machine.decide(observation, now=self._now())
    if next_state is not self._machine.state:
        self._machine.enter(next_state, reason=..., now=self._now())
    actions = self._policy.actions(self._machine.state, observation, self._mode)
    results = tuple(self._executor.execute(a) for a in actions)
    # emit ACTION_BATCH + per-action ACTION_RESULT events
    return TickResult(observation=observation, state=self._machine.state,
                      actions=actions, results=results)
```

**Refactor:** extract the "route observation failure to RECOVERY" path into
a private `_on_observation_error(exc)` so `tick()` stays linear.

**Verify:** full suite green; `MockBackend.events` in the COMBAT test shows
the exact key/delay sequence from the policy.

---

### Step 6 — Safety integration: hotkey stops the loop, PAUSED releases everything (loop_safety)

**Contract (tests first):**
- Triggering the `FakeHotkeyListener` (F12) during a `run()` loop causes
  `stop_event` to be set, the loop to exit within the current tick, and the
  report's `stopped_reason == "emergency_stop"`.
- On `PAUSED` exit, `executor.release_all()` is called (asserted via
  `MockBackend.held_keys == []` and `held_buttons == []` after a COMBAT tick
  that leaves a key held — though the Phase 3 executor auto-releases, the
  test asserts the *loop* calls `release_all` as a belt-and-braces measure).
- On `PAUSED` exit, `capture.release()` is called (asserted via
  the scripted fake's lifecycle assertions).
- Each action dispatch is isolated: record a returned `ActionResult(ok=False)` and continue only when policy permits; if `execute()` raises `ActionError` or the executor reports stopped/paused, record the failure, stop dispatching the batch, and apply the bounded recovery/safe-pause policy. Do not retry a failed action blindly.

**Red test:** `tests/test_agent_loop_safety.py`.

**Green:** wire loop cancellation only through the actual Phase 3 callback seam. In nested `finally` blocks call `executor.release_all()` / the executor's documented shutdown path and `Capture.stop()` so one cleanup failure cannot skip the other.

**Verify:** full suite green.

---

### Step 7 — CLI `run` subcommand (__main__.py)

**Contract (tests first, subprocess):**
- `ai-game-agent run --once --backend mock --mode observe_only` exits 0,
  prints an `AgentLoopReport` JSON with `ticks == 1`, `actions_attempted ==
  0` (observe-only policy), and a valid `final_state`.
- `ai-game-agent run --once --backend mock --mode autonomous` with a
  mock frame that triggers COMBAT (via a config override or a fixture frame)
  prints a report with `actions_attempted >= 1` and `actions_ok >= 1`.
- `ai-game-agent run --max-ticks 3 --backend mock --mode autonomous` exits
  0 with `ticks == 3`.
- `ai-game-agent run --record --once` writes a `.jsonl` event log under
  `recordings/` (one line per `LoopEvent`) and the report references the
  file path.
- `ai-game-agent run --mode autonomous --backend windows --confirm` is
  rejected on Linux (the Phase 3 `--confirm` gate + `InputBackendError` for
  missing pynput) — **unchanged from Phase 3**, asserted to still fire.
- `ai-game-agent run` (no `--once`, no `--max-ticks`) runs until F12 or
  Ctrl-C; the test asserts it *starts* (spawns, waits 200ms, confirms the
  process is alive, sends SIGINT, asserts exit 0 with a partial report).

**Red test:** `tests/test_run_cli.py`.

**Green:** add the `run` subcommand to `parse_args()`, reusing
`_build_perception`, `_build_capture`, `_build_executor`, and adding
`_build_agent(cfg)` (constructs `GameStateMachine` + `DeterministicPolicy`
from `cfg.agent`). The handler `_run(args, cfg)` wires `AgentLoop` and calls
`loop.run()`, printing the report JSON.

**Verify:** full suite green; `--help` output includes `run`.

---

### Step 8 — Replay: record → replay (test_replay.py)

**Contract (tests first):**
- A `run --record --max-ticks 3` session writes a `.jsonl` file with exactly
  3+ `LoopEvent` lines (one `TICK` per tick + any `STATE_*` events).
- A `replay` subcommand (or a `--replay <file>` flag on `run`) reads the
  `.jsonl`, feeds the recorded *observations* (not the events — the events
  carry the zone values, which are enough to reconstruct the `Observation`
  for the policy's `transition_signals`) back through
  `machine.decide()`, and asserts the **same** state sequence as the original
  run. (This is the project's "primary mechanism for regression testing":
  the replay does not re-capture or re-perceive; it feeds recorded zone
  signals into the same machine+policy and asserts the same decisions.)
- The replay test is the **regression test** for the transition table: if a
  future edit changes a transition, the replay assertion fails.

**Red test:** `tests/test_replay.py`.

**Green:** add `--replay` to the `run` subcommand (or a separate `replay`
subcommand — decide at implementation; the test asserts the CLI contract).

**Verify:** full suite green; the replay assertion passes on the recorded
log from step 7's test.

---

### Tick result and action-failure contract

`TickResult` is a concrete frozen dataclass (not an undefined placeholder) with at least: `observation: Observation | None`, `state_before: AgentState`, `state_after: AgentState`, `actions: tuple[Action, ...]`, `results: tuple[ActionResult, ...]`, `errors: tuple[str, ...]`, and `stop_requested: bool`. A capture failure may produce `observation=None`; preserve partial results when a later action fails. The loop catches only documented capture/perception/action exceptions at the boundary, emits an error event, and routes through recovery. Do not swallow arbitrary `BaseException` or turn every exception into a retry.

**Action dispatch is per-action, not all-or-nothing.** Before each action, check the stop event and executor stopped/paused state. Execute one action, immediately log its `ActionResult`, then decide whether to continue. A returned `ok=False`, raised `ActionError`, safety rejection, emergency stop, or cancellation must have explicit handling; no unlogged failure and no further batch actions after stop/safety failure. Do not infer a new executor API: use the actual `ActionResult`, `Executor.stopped`, `stop_reason`, `release_all()`, and `trigger_emergency_stop()` contracts.

## 6. Dependency policy (no new third-party deps)

Phase 4 adds **zero** new runtime dependencies. Everything it needs is
already in the package:

| Need | Source (existing) |
| --- | --- |
| Typed game state | new `state/` package (pure Python) |
| State machine + watchdog | new `state/machine.py` (pure Python, `enum`, `dataclasses`) |
| Policy/action-plan | new `state/policy.py` + `games/wow/policy.py` (consumes Phase 2 `Action` types) |
| Loop orchestration | new `agent/loop.py` (consumes `Capture`, Phase 2 `Perception`, and Phase 3 `Executor`) |
| Stop token | `threading.Event` |
| Event log | new `state/events.py` (`dataclasses`, `json`) |
| Config | `config.py` `_AgentConfig` (Pydantic, already a dep) |
| CLI | `__main__.py` `argparse` (stdlib) |

**Explicitly NOT added:** no `pynput` (Phase 3's `input` extra is unchanged;
the `windows` backend still requires it on the Windows box), no Ollama client
(Phase 5), no new CV libs (Phase 2's `cv` extra is unchanged), no scheduler
(`time.sleep` + `threading.Event` are sufficient at 2 Hz).

The `pyproject.toml` `[project] dependencies` block is **untouched** by
Phase 4. The `[project.optional-dependencies]` block is **untouched**. CI's
bare-venv contract (no `input` extra) continues to hold because Phase 4's
loop tests use scripted capture fake + `MockBackend` exclusively.

---

## 7. Verification gates (Definition of Done, per step)

A step is complete when:

1. **Red was real** — the failing test failed for the missing-behavior
   reason (import error for a new module, or an assertion failure for a new
   method), not for a typo or environment issue.
2. **Green is minimal** — no speculative code beyond what the red test
   requires.
3. **Full suite green** — `uv run pytest -q` passes with no new
   `xfail`/`skip` and no new warnings treated as errors.
4. **No new public API breaks** — Phases 2/3 tests are untouched and pass.
5. **Config is externalized** — any new tunable (tick rate, timeout,
   recovery attempts, combat keys) lives in `config/default.yaml` under
   `agent:`, not in code.
6. **Logging is sufficient** — every transition, action batch, and error in
   a `run` session produces a `LoopEvent` with a human-readable `reason`.
7. **Safety is intact** — `OBSERVE_ONLY` emits zero input (asserted in step
   5/6); the F12 hotkey stops the loop (step 6); `PAUSED` releases all held
   input (step 6).
8. **No secrets, no machine-specific config** — the new config block is
   committed with safe defaults.

---

## 8. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| **Watchdog timeout vs. legitimate long RECOVERY.** A real low-health recovery (drinking a potion, waiting for a healer) may take > 5 s. | `state_timeout_s` is config (`agent.state_timeout_s`, default 5.0). The watchdog fires on *stuck-in-RECOVERY* (no state change), not on *being in RECOVERY*: if `decide()` keeps returning RECOVERY because health is still < 30%, the watchdog fires at 5 s and pauses — which is the *correct* safe behavior (the agent shouldn't sit in a death-spiral loop). The operator can raise the timeout or the `recovery_attempts` to give it more room. |
| **Policy/machine coupling drift.** If the policy's `transition_signals` key set changes, the machine silently breaks. | Step 3's test asserts the exact key set; step 2's machine tests use a stub policy with the same key set. A change to either side fails the other's test — the seam is contract-tested from both sides. |
| **Loop leaks a capture frame on exception.** | `run()` guarantees `Capture.stop()` in nested `finally` cleanup, including if executor cleanup raises. Tests assert lifecycle closure on normal exit and failure. |
| **Hotkey stops the executor but not the loop.** | Step 6 wires the loop's stop callback through the actual Phase 3 callback seam; the emergency-stop path itself remains owned by `Executor`. The test asserts the loop exits, not just the action. |
| **OBSERVE_ONLY still executes input via a policy bug.** | The policy returns `()` under `OBSERVE_ONLY` (step 3), *and* the executor's `SafetyGuard` re-checks the mode at dispatch (Phase 3). Defense in depth; step 5 asserts zero `MockBackend.events` under `OBSERVE_ONLY`. |
| **`tick_hz` too high → CPU/VRAM spike.** | Default is 2 Hz (500 ms/tick), well within the `le=60.0` config bound. The config validator caps it. At 2 Hz with Phase 2's detector pipeline, the loop is perception-bound, not CPU-bound. |
| **Phase 5 (AI) needs a different loop shape.** | The `AI_ANALYSIS` state slot + watchdog + `paused` fallback are the exact seams Phase 5 will use: the AI strategy object replaces the empty `policy.actions(AI_ANALYSIS, ...)` tuple, and the same watchdog bounds its runtime. No loop refactor needed. |

---

## 9. Acceptance criteria (Phase 4 complete)

1. `uv run ai-game-agent run --once --backend mock --mode observe_only`
   exits 0, prints a valid `AgentLoopReport` JSON with `ticks == 1` and
   `actions_attempted == 0`.
2. `uv run ai-game-agent run --max-ticks 5 --backend mock --mode autonomous`
   exits 0, prints a report with `ticks == 5`, and (with a mock frame that
   triggers COMBAT) `actions_ok >= 1`.
3. The F12 hotkey (via `FakeHotkeyListener` in tests; real pynput hotkey on
   Windows) stops the loop, releases all held input, and exits 0.
4. A `run --record` session produces a `.jsonl` event log; a
   `run --replay <log>` re-derives the same state sequence.
5. `uv run pytest -q` is green with no new `xfail`/`skip`, on Linux in CI
   (bare venv, no `input` extra, no Ollama, no desktop).
6. No new third-party runtime dependency; `pyproject.toml` dependencies
   block unchanged.
7. `OBSERVE_ONLY` (the default mode) emits **zero** OS input across the
   entire loop — asserted by test.
8. Every state transition, action batch, and error in a session is logged
   as a `LoopEvent` with a human-readable `reason`; a reader can answer
   "why did the agent do this?" from the log alone.
9. The `state/` and `agent/` packages contain **no** WoW-specific zone-name
   literals — those live only in `games/wow/`. The loop is
   game-agnostic; the policy is the game-adapter boundary.
10. `docs/` gains a short `phase-4-loop.md` (or a section in
    `manual-testing.md`) with the `run` command's flags, the config block,
    and the replay workflow.

---

## 10. What this plan deliberately does NOT do

- **No AI/VLM/LLM call.** The `AI_ANALYSIS` state is a defined slot with an
  empty action list; the loop never enters it in Phase 4 (the transition
  table has no *to* `AI_ANALYSIS` edge — Phase 5 adds the edge and the
  strategy). This is the project's "deterministic first" principle: the loop
  is fully playable and testable before a single model is invoked.
- **No persistent memory.** The `.jsonl` event log is a session artifact, not
  a database. Phase 6 will define the episodic-memory schema on top of it.
- **No target selection / pathing / AoE logic.** Phase 4's COMBAT policy is a
  fixed key rotation; LOOTING is a single click + pickup key. The *skeleton*
  (loop → state → policy → executor) is the deliverable; game skill is
  Phase 5/6 content that plugs into the same policy interface.
- **No multi-game support beyond the WoW adapter.** `games/wow/` is the only
  policy; the architecture supports `games/cod/`, `games/minecraft/`, etc.,
  but Phase 4 ships one.
- **No `semi_autonomous` UX.** The mode is accepted by the guard (Phase 3);
  Phase 4 does not add a human-approval prompt for `ASSISTED` (that's the
  `--confirm` CLI gate, already in Phase 3). A richer HIL UX is a Phase 5/6
  concern.
- **No performance optimization.** 2 Hz is a correctness target, not a
  performance target. Profiling and optimization are a post-Phase-4 concern
  (the project's "measure, don't guess" rule applies then).

---

## 11. Suggested commit sequence (TDD red → green per step)

| # | Commit (draft message) | Step |
| --- | --- | --- |
| 1 | `feat(state): add AgentState enum + AgentStateError` | 1 |
| 2 | `test(state): add transition-table tests (red)` | 2a |
| 3 | `feat(state): implement GameStateMachine.decide/enter/recovery/watchdog` | 2b |
| 4 | `feat(state): add LoopEvent + wire machine callbacks to events` | 4 |
| 5 | `feat(games): add WoW zone constants + DeterministicPolicy` | 3 |
| 6 | `test(agent): add AgentLoop tick/run integration tests (red)` | 5a |
| 7 | `feat(agent): implement AgentLoop.tick/run + AgentLoopReport` | 5b |
| 8 | `feat(agent): wire hotkey stop + PAUSED release-all (safety)` | 6 |
| 9 | `feat(cli): add run subcommand (--once, --max-ticks, --record, --replay)` | 7 |
| 10 | `test(cli): add run-subcommand subprocess tests` | 7b |
| 11 | `test(replay): add record → replay regression tests` | 8 |
| 12 | `docs: add phase-4 loop usage + config reference` | — |

Each commit is independently green (the suite passes at every commit), and
each `test(...)` commit lands **before** its `feat(...)` commit — the TDD
red → green ordering, visible in history.
