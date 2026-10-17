# Manual testing

Run every command from the **repository root** in a **Windows terminal**.
PowerShell is assumed — the `$LASTEXITCODE` checks are PowerShell. In `cmd`
(Command Prompt), use `echo %ERRORLEVEL%` instead.

`uv` manages the environment, so each `uv run …` line is self-contained and
picks up the right interpreter and installed extras.

Phase 1 exercises the capture backend (`capture` / `observe`). Phase 2
exercises the perception pipeline (`analyze`). Phase 3 exercises input
control (`act`). Phases 1–2 are safe to run headless with `--backend mock`
(the mock backend emits a deterministic 128×72 frame and needs no display);
Phase 3's mock backend likewise records events without touching your
keyboard, but the **windows** backend drives real mouse/keyboard input —
read §4.3 before running it.

---

## 1. Phase 1 — mock capture (no display needed)

```powershell
# Single mock screenshot → expect "saved: ...\*.png" and exit 0
uv run python -m ai_game_agent capture --backend mock --out shots
$LASTEXITCODE   # → 0

# Region + scale pipeline: 32×16 region scaled 0.5 → expect a 16×8 PNG
uv run python -m ai_game_agent capture --backend mock --region 10,20,32,16 --scale 0.5 --out shots

# Observe loop, no pacing, 5 frames → expect "frames=5 fps=..."
uv run python -m ai_game_agent observe --backend mock --frames 5 --fps 0

# Recording + rotation: 10 frames, keep max 3 → exactly 3 PNGs must remain in rec/
uv run python -m ai_game_agent observe --backend mock --frames 10 --fps 0 --record --record-max-files 3 --out rec
```

## 2. Phase 1 — real capture (mss backend, needs a display)

```powershell
uv run python -m ai_game_agent capture --backend mss --out shots
uv run python -m ai_game_agent capture --backend mss --region 0,0,3840,2160 --out shots
uv run python -m ai_game_agent observe --backend mss --frames 30 --fps 30
```

> **Note (this PC — 2 monitors):** By default `capture` grabs the *primary*
> monitor only (3840 × 2160), which is what you usually want — no black bands
> around the secondary. To grab a different monitor use `--monitor`
> (1 = primary, 2+ = the (n-1)th secondary, 0 = the whole 5760×2160 virtual
> screen):
>
> ```powershell
> uv run python -m ai_game_agent capture --backend mss --monitor 2 --out shots   # secondary (Dell U2412M, 1920×1200)
> uv run python -m ai_game_agent capture --backend mss --monitor 0 --out shots   # whole virtual screen
> ```
>
> If a *primary*-monitor capture is also fully black, disable DP HDR in the
> Windows display settings — HDR mode can make screen capture render black.

> **mss 10.x compatibility:** `MssBackend` works with mss 10+ (verified on
> 10.2.0): `grab()` passes `monitors[<selector>]` explicitly (the monitor
> argument is required in mss 10) and handles both the dict-style `shot.size`
> (mss 9) and the `Size` object (mss 10+). If you see
> `MSS.grab() missing 1 required positional argument: 'monitor'` or
> `tuple indices must be integers`, the running code predates that fix —
> update the package, not your command.

---

## 3. Phase 2 — perception (`analyze`)

`analyze` captures one frame and runs it through the perception pipeline
(template matching → UI-zone checks → OCR → color-blob object detection),
printing the resulting `Observation` as JSON on **stdout**.

Contract you can rely on while testing by hand:

- **stdout** carries only the JSON observation (or nothing).
- **stderr** carries `error:` lines and usage text.
- Exit code is **0** on success, **1** on a setup/config/backend failure.
- A detector that *fails at runtime* (e.g. Tesseract binary not found, template
  file missing) does **not** crash the run — the observation is still emitted
  with the failure recorded in its `detector_errors` list.

Shared flags with `capture`/`observe`: `--backend` (default `mss`), `--monitor`,
`--region`, `--scale`, `--fps`. `analyze` adds `--config`, `--no-templates`,
`--no-ocr`, `--no-objects`, and `--pretty`.

### 3.1 Install the optional extras

The core CLI imports and runs with no extras. Five extras cover the full
manual-testing flow:

- `dev` — test tooling (pytest, ruff, …) for the §3.5 / §4.5 cross-checks.
- `capture` — mss + Pillow. Backs the **real** `--backend mss` capture steps.
- `vision` — OpenCV + numpy. Backs **template matching** and **object blobs**.
- `ocr` — pytesseract + Pillow. Backs **OCR** (and, separately, a Tesseract
  binary).
- `input` — pynput. Backs the **windows** input backend of `act` (§4).

> **`uv sync` prunes the venv to *exactly* the extras you list.** Omitting
> `capture` here uninstalls `mss` (you will see a ` - mss==…` line in the
> sync output), and then every `--backend mss` step fails with
> `error: the 'mss' backend is not installed`. List all five:

```powershell
# One-shot: dev + capture + perception + input extras into the same .venv
uv sync --extra dev --extra capture --extra vision --extra ocr --extra input
```

OCR needs the Tesseract **binary** on `PATH` as well — the Python package alone
is not enough:

```powershell
# Option A — winget (Windows 10/11), then open a NEW terminal:
winget install --id UB-Mannheim.TesseractOCR

# Option B — download the installer from the Tesseract wiki
# (https://github.com/UB-Mannheim/tesseract/wiki), install, open a NEW terminal.

# Verify either way:
tesseract --version
```

### 3.2 Smoke test (headless, no extras needed)

With the default config no detectors are selected, so this needs neither a
display nor any extras. The mock frame is 128×72.

```powershell
# One-frame perception → JSON on stdout, exit 0
uv run python -m ai_game_agent analyze --backend mock
$LASTEXITCODE   # → 0

# Same, pretty-printed (multi-line, still valid JSON)
uv run python -m ai_game_agent analyze --backend mock --pretty

# All subsystems disabled → all four hit lists are empty
uv run python -m ai_game_agent analyze --backend mock --no-templates --no-ocr --no-objects
```

Expect a single JSON object with these keys: `schema_version`, `frame_width`
(128), `frame_height` (72), `captured_at`, `source`, `templates`, `ui_zones`,
`text_regions`, `objects`, `detector_errors`. With the default config all four
hit lists (`templates`, `ui_zones`, `text_regions`, `objects`) are `[]` and
`detector_errors` is `[]`.

### 3.3 Per-subsystem checks

Each subsystem is driven by the `perception:` block in `config/default.yaml`
(or a `--config` file). The mock frame is a solid dark color (RGB 10, 20, 30),
so it produces **no** real template / OCR / blob hits — those require real
screen content. Test these against `--backend mss` on a region where you know
the target is present.

`config/manual-test.yaml` ships in the repo (contents below); tweak it to
enable one subsystem at a time so the output is unambiguous. Then point
`analyze` at it:

```yaml
perception:
  enabled: true
  template_threshold: 0.8
  # Template matching (needs "vision"). Paths are repo-relative.
  templates:
    target_frame: assets/templates/target_frame.png   # committed 64x36 sample
  # UI-zone checks (needs "vision"). check: presence | brightness | color_present
  ui_zones:
    - name: action_bar
      x: 40
      y: 56
      width: 48
      height: 12
      check: brightness
  # OCR (needs "ocr" extra + Tesseract binary).
  ocr:
    enabled: true
    regions:
      - {name: health_bar, x: 10, y: 20, width: 100, height: 20}
  # Color-blob objects (needs "vision").
  objects:
    enabled: true
    colors:
      - {name: loot_glow, rgb: [255, 200, 0], tolerance: 40}
```

```powershell
# Full pipeline (all enabled subsystems run against the live region)
uv run python -m ai_game_agent analyze --backend mss --region 0,0,1280,720 --config config/manual-test.yaml

# Template matching only → look for a hit in `templates`
uv run python -m ai_game_agent analyze --backend mss --region 0,0,1280,720 --config config/manual-test.yaml --no-ocr --no-objects

# OCR only → look for a hit in `text_regions`
uv run python -m ai_game_agent analyze --backend mss --region 0,0,1280,720 --config config/manual-test.yaml --no-templates --no-objects

# Object blobs only → look for a hit in `objects`
uv run python -m ai_game_agent analyze --backend mss --region 0,0,1280,720 --config config/manual-test.yaml --no-templates --no-ocr

# Pretty-print the whole observation for reading
uv run python -m ai_game_agent analyze --backend mss --region 0,0,1280,720 --config config/manual-test.yaml --pretty
```

Pull one field at a time (PowerShell):

```powershell
# Only the runtime errors (empty array means everything succeeded)
uv run python -m ai_game_agent analyze --backend mss --config config/manual-test.yaml |
  ConvertFrom-Json | Select-Object -ExpandProperty detector_errors

# Only the template hits
uv run python -m ai_game_agent analyze --backend mss --config config/manual-test.yaml |
  ConvertFrom-Json | Select-Object -ExpandProperty templates
```

### 3.4 Error routing (the contract the e2e suite asserts)

None of these should print a traceback — expect a clean `error:` line on
**stderr**, **empty stdout**, and exit **1**:

```powershell
# Malformed config file → exit 1  (create a deliberately broken file first)
"perception: [not, a, dict]" | Set-Content config\bad.yaml
uv run python -m ai_game_agent analyze --backend mock --config config\bad.yaml
$LASTEXITCODE   # → 1   (stderr: "error: invalid configuration in config/bad.yaml: …")

# Unknown backend → exit 1
uv run python -m ai_game_agent analyze --backend doesnotexist
$LASTEXITCODE   # → 1   (stderr: "error: unknown capture backend: 'doesnotexist'")
```

Runtime detector failures must be *reported*, not fatal. For example, with
`ocr.enabled: true` but no Tesseract binary installed, `analyze` still exits
**0** and puts a string like `"ocr: …"` into `detector_errors` — verify the
JSON still emits:

```powershell
uv run python -m ai_game_agent analyze --backend mock --config config/manual-test.yaml |
  ConvertFrom-Json | Select-Object -ExpandProperty detector_errors
```

### 3.5 Phase 2 test suite (optional cross-check)

The manual steps above mirror the automated suites; run them to confirm your
environment agrees with CI.

```powershell
# Perception unit tests
uv run pytest tests/test_observation.py tests/test_perception_config.py tests/test_pipeline.py tests/test_template_matcher.py tests/test_ocr.py -q

# The `analyze` e2e contract (stream routing, exit codes, --config)
uv run pytest tests/e2e/test_e2e_cli.py -k analyze -v

---

## 4. Phase 3 — input control (`act`)

`act` runs input actions (keys, mouse, holds, scroll, delays) through the
`Executor` and prints the resulting `ActionLog` as JSON on **stdout**.
It is **observe-only by default**: with the shipped config
(`input.mode: observe_only`) the CLI emits zero OS input — the log prints
with `results == []`.

Contract you can rely on while testing by hand:

- **stdout** carries only the JSON ActionLog (nothing else).
- **stderr** carries `error:` lines and log lines.
- Exit code is **0** on success — including a batch *stopped* by a safety
  bound (a bounded stop is a recorded, successful stop: `stopped: true`
  with a `stop_reason`, not a CLI error); **1** on a setup/config/policy
  failure (unknown backend, missing `--confirm`, unparseable `--do` spec,
  empty `--script`).
- A mode/safety violation is recorded as a failed result for that action
  (`ok: false`) and **halts the batch**; a per-action backend failure is
  recorded and the batch continues.

Flags: `--do` (repeatable; `;`-separated within one flag), `--script`
(file), `--demo` (built-in 8-action sequence — the **default** when neither
`--do` nor `--script` is given), `--mode`, `--backend` (default `mock`
from config), `--record`, `--record-dir` (default `recordings`),
`--confirm`, `--pretty`, `--config`.

### 4.1 Install the `input` extra (real-input backend only)

The **mock** backend (the default) needs no extras. The **windows** backend
drives real mouse/keyboard input and needs pynput, installed **on the
Windows target machine**:

```powershell
uv sync --extra input
```

> **Shared-venv caveat:** if this `.venv` is shared across OSes (drvfs/9p),
> don't re-run `uv sync` on the other OS without `--extra input` — the sync
> prunes the venv to the extras you list and would uninstall pynput on the
> Windows side (see the §3.1 note).

### 4.2 Smoke test (mock backend, headless, no extras needed)

The mock backend records the events instead of touching your keyboard, so
every step below is safe on any machine.

```powershell
# Default: config mode observe_only → the gate holds, zero input. results == []
uv run python -m ai_game_agent act
$LASTEXITCODE   # → 0   (JSON: {"results": [], "stopped": false, ...})

# Full-contract demo (all 8 action kinds) executed against the mock
uv run python -m ai_game_agent act --backend mock --mode autonomous --demo --pretty
# expect: 8 results, all "ok": true, "stopped": false

# A few explicit actions (repeatable; ';' splits a batch inside one flag)
uv run python -m ai_game_agent act --backend mock --mode autonomous --do key:a --do "hold:w:100;scroll:3"
$LASTEXITCODE   # → 0   (3 results, all "ok": true)

# Record the ActionLog to ./recordings (timestamped file name)
uv run python -m ai_game_agent act --backend mock --mode autonomous --demo --record
ls recordings
# then replay the recorded log (ActionLog shape is accepted by --script)
uv run python -m ai_game_agent act --backend mock --mode autonomous --script recordings\<the-file>.json --pretty
```

Hand-written action script (a JSON list of action dicts — `--do` aside,
the same shape `Action.to_dict()` emits):

```powershell
@'
[
  {"kind": "key_press", "key": "a"},
  {"kind": "mouse_move", "x": 100, "y": 200},
  {"kind": "mouse_click", "button": "left", "clicks": 1}
]
'@ | Set-Content scripts\smoke.json
uv run python -m ai_game_agent act --backend mock --mode autonomous --script scripts\smoke.json --pretty
$LASTEXITCODE   # → 0   (3 results, all "ok": true)
```

### 4.3 Windows backend (real mouse/keyboard — read before running)

**Before the first run:**

1. Install the extra (§4.1).
2. **Focus a scratch window first** (e.g. Notepad). The demo presses
   `Ctrl+Shift+A`, holds `w` for 150 ms, double right-clicks at absolute
   (320, 180), does a relative move, a left press/release, and scrolls —
   you want those effects to land somewhere harmless.
3. Know the panic key: **F12** (`safety.emergency_stop_keys`, default).
   Press it any time a run misbehaves: the executor stops, pending actions
   are cancelled, any held key/button is released (mid-hold included), and
   no further input is dispatched for that run.
4. `--confirm` is required on every non-`observe_only` `windows` run — it
   is the explicit acknowledgment that real OS input will occur (a no-op
   for `mock`; not needed at all under `observe_only`).
5. Know the bounds (the `safety:` block): a hold longer than
   `max_action_duration_ms` (5000 ms) is *rejected* and stops the batch;
   `max_consecutive_actions` (50) and `max_backend_errors` (3) make the
   executor **pause**, never loop.

```powershell
# 1. Smallest real action: one 'a' keystroke in the focused scratch window
uv run python -m ai_game_agent act --backend windows --mode autonomous --confirm --do key:a
$LASTEXITCODE   # → 0   (expect "a" typed; log: 1 result, "ok": true, "stopped": false)

# 2. F12 drill: hold 'w' for 4 s (under the 5 s bound) and press F12 mid-hold
uv run python -m ai_game_agent act --backend windows --mode autonomous --confirm --do hold:w:4000 --pretty
# mid-hold: press F12 → the held key must release immediately, and the log
# shows "stop_event": true on the hold's result and "stopped": true with a stop_reason

# 3. Full 8-action demo, assisted mode, recorded for replay/audit
uv run python -m ai_game_agent act --backend windows --mode assisted --confirm --demo --record --pretty
# expect: mouse travel + double right-click + wheel scroll in the scratch window,
# "a" + typed "w", and 8 "ok": true results; the ActionLog is in ./recordings
```

> **Mode note:** in this CLI the hard mode gate currently rejects only
> `observe_only`; `assisted` and `semi_autonomous` dispatch through the same
> executor path as `autonomous`. Phase-3 scope: the propose/approve split is
> the guard's mechanism (see `SafetyGuard`), not yet an interactive prompt —
> `--confirm` is the acknowledgment for all real-input runs.

### 4.4 Error routing (the contract the e2e suite asserts)

None of these should print a traceback — expect a clean `error:` line on
**stderr**, an **empty stdout**, and exit **1**:

```powershell
# Unknown backend → exit 1
uv run python -m ai_game_agent act --backend doesnotexist
$LASTEXITCODE   # → 1   (stderr: "error: …")

# Real-input backend in an input-allowed mode without --confirm → exit 1
uv run python -m ai_game_agent act --backend windows --mode autonomous --demo
$LASTEXITCODE   # → 1   (stderr: "error: the 'windows' backend … requires --confirm …")

# Unparseable --do spec → exit 1
uv run python -m ai_game_agent act --do bogus:xyz
$LASTEXITCODE   # → 1   (stderr: "error: act: unknown --do kind 'bogus' …")

# --script with no actions → exit 1
"[]" | Set-Content scripts\empty.json
uv run python -m ai_game_agent act --script scripts\empty.json
$LASTEXITCODE   # → 1   (stderr: "error: act: --script contains no actions")

# --backend windows under observe_only → no --confirm needed, no input, exit 0
# (works even without the input extra — the backend is never opened)
uv run python -m ai_game_agent act --backend windows --mode observe_only
$LASTEXITCODE   # → 0   (JSON: {"results": [], "stopped": false, ...})
```

A *bounded stop* is **not** an error: with a tight safety bound the run still
exits **0** and the log records the stop (mirror of the e2e `--config` test):

```powershell
@'
input:
  backend: mock
  mode: autonomous
safety:
  max_action_duration_ms: 100
'@ | Set-Content config\act-bound.yaml
# The demo's 150 ms hold exceeds the 100 ms bound → rejected, batch halts
uv run python -m ai_game_agent act --config config\act-bound.yaml --pretty
$LASTEXITCODE   # → 0
# log: "stopped": true, "stop_reason": "safety bound: …",
# the rejected hold is a result with "ok": false
```

### 4.5 Phase 3 test suite (optional cross-check)

The manual steps above mirror the automated suites; run them to confirm your
environment agrees with CI.

```powershell
# Action-layer unit tests (contract, executor, queue, guard, hotkey, mock, windows)
uv run pytest tests/test_action.py tests/test_actions_base.py tests/test_actions_config.py tests/test_actions_executor.py tests/test_actions_hotkey.py tests/test_actions_mock.py tests/test_actions_modes.py tests/test_actions_queue.py tests/test_windows_backend.py -q

# The `act` CLI contract (stream routing, exit codes, observe-only zero-input)
uv run pytest tests/e2e/test_e2e_act.py -v
```

And the whole suite must stay green in a **bare venv** (no `vision` / `ocr` /
`input` extras) — the cv2/pynput-dependent modules skip instead of failing:

```powershell
uv run pytest -q
```
```

## 5. Phase 4 — the `run` loop and its stop semantics

The `run` subcommand is the generic, game-agnostic observe-decide-act loop
(Phase 4 §4.4). See `docs/phase-4-loop.md` for the full reference; the steps
here are the manual smoke tests plus the **honest** stop semantics you should
verify against the live tree.

### 5.1 Headless run (mock capture + mock input, no display, no OS input)

```powershell
# One tick, clean report, exit 0 (observe_only emits zero OS input)
uv run python -m ai_game_agent run --once --backend mock --mode observe_only
# -> report JSON on stdout: "stopped_reason": "max_ticks", "final_state": "idle"
$LASTEXITCODE   # -> 0

# N ticks, pretty-printed (indent=2; still valid JSON)
uv run python -m ai_game_agent run --max-ticks 5 --backend mock --mode observe_only --pretty
```

`--max-ticks N` (or `--once`) is the **deterministic, report-producing** way to
end a loop. This is the clean stop — not Ctrl+C (see §5.3).

### 5.2 The two loop config knobs

`config/default.yaml` `agent:` block now carries `state_timeout_s` (default
`5.0`, watchdog seconds in a non-terminal state) and `recovery_attempts`
(default `3`, bounded recovery budget before `PAUSED`). Both are validated
(positive, **bool rejected**) and threaded to `GameStateMachine` on both the
live `run` and the `replay` path. A bad value must fail loudly:

```powershell
# Put "recovery_attempts: 0" (or "state_timeout_s: true") in a scratch config,
# then expect a clean "error: ..." + exit 1, not a traceback:
uv run python -m ai_game_agent run --once --backend mock --config config\bad.yaml
$LASTEXITCODE   # -> 1
```

### 5.3 Ctrl+C / SIGINT — an **abort**, not a clean stop (verified)

**There is no SIGINT handler and no `KeyboardInterrupt` catch in `src/`.** So
Ctrl+C is an abort, and the honest behavior is:

- a `KeyboardInterrupt` traceback goes to **stderr**;
- **no report is printed to stdout** (stdout stays empty);
- the process exits **non-zero** (130 under bash).

The loop's `finally` cleanup (`capture.stop()`, `executor.release_all()`) still
runs, but only as a side effect of exception unwinding — it is **not** a
designed clean-stop contract. Do **not** rely on Ctrl+C for a "clean partial
report." Use `--once` / `--max-ticks` for a deterministic, report-producing
stop.

### 5.4 F12 emergency stop / hotkey — implemented + tested, **not wired into `run`**

The machinery exists and is tested: `Executor.trigger_emergency_stop()`, the
`AgentLoop(hotkey=, stop_event=)` hooks, and `WindowsBackend`'s armed F12
`_HotkeyDecider`. **But the `run` CLI does not wire it:**

- `create_backend` builds `WindowsBackend()` with `on_stop=None` (no-op);
- `_run` passes **neither** `hotkey=` nor `stop_event=` to `AgentLoop`;
- `run` never calls `backend.open()` (that's the `act` path), so the pynput
  listener isn't even started.

**Therefore, in the `run` CLI as it stands, F12 does not stop the loop.**
`stop_requested` and `emergency_stop` are valid report values in code but
**unreachable from the `run` CLI today.** This is a known wiring gap (a
follow-up), documented here so the manual tests match the tree. A programmatic
caller can still wire it by passing a `HotkeyListener` (with `on_trigger`
calling `executor.trigger_emergency_stop()`) plus a `threading.Event`.

### 5.5 Mock backend has no emergency-stop trigger

`MockBackend` exposes `open`/`close`, the input verbs, `release_all`, and an
`events` log — but **no** `trigger_emergency_stop` and **no** hotkey. The mock
is for deterministic, backend-free runs (CI, replay), not for exercising the
real stop path. For a real stop source you need the `windows` backend (and per
§5.4 it is not yet wired into `run`).

### 5.6 Replay workflow

```powershell
# Record a run's event log (one .jsonl line per LoopEvent):
uv run python -m ai_game_agent run --max-ticks 50 --backend mock --mode observe_only --record --record-dir recordings

# Replay it — re-derive the state sequence without touching a real app:
uv run python -m ai_game_agent replay <recorded>.jsonl --config config\default.yaml
```

Replay uses the same `agent.recovery_attempts` / `agent.state_timeout_s` knobs
as the live path, so it reproduces the same bounded-recovery budget and
watchdog timeout as the recorded session.

### 5.7 Phase 4 test suite (optional cross-check)

```powershell
# Loop, safety, config, and the run/replay CLI contracts
uv run pytest tests/test_agent_loop.py tests/test_agent_loop_safety.py tests/test_agent_config.py tests/test_run_cli.py tests/test_replay.py -q

# And the whole suite must stay green (CI must not need a desktop or a live game):
uv run pytest -q
```
