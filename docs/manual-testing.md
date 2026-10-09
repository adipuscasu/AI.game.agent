# Manual testing

Run every command from the **repository root** in a **Windows terminal**.
PowerShell is assumed — the `$LASTEXITCODE` checks are PowerShell. In `cmd`
(Command Prompt), use `echo %ERRORLEVEL%` instead.

`uv` manages the environment, so each `uv run …` line is self-contained and
picks up the right interpreter and installed extras.

Phase 1 exercises the capture backend (`capture` / `observe`). Phase 2
exercises the perception pipeline (`analyze`). Both are safe to run headless
with `--backend mock` (the mock backend emits a deterministic 128×72 frame and
needs no display).

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

The core CLI imports and runs with no extras. Four extras cover the full
manual-testing flow:

- `dev` — test tooling (pytest, ruff, …) for the §3.5 cross-checks.
- `capture` — mss + Pillow. Backs the **real** `--backend mss` capture steps.
- `vision` — OpenCV + numpy. Backs **template matching** and **object blobs**.
- `ocr` — pytesseract + Pillow. Backs **OCR** (and, separately, a Tesseract
  binary).

> **`uv sync` prunes the venv to *exactly* the extras you list.** Omitting
> `capture` here uninstalls `mss` (you will see a ` - mss==…` line in the
> sync output), and then every `--backend mss` step fails with
> `error: the 'mss' backend is not installed`. List all four:

```powershell
# One-shot: dev + capture + perception extras into the same .venv
uv sync --extra dev --extra capture --extra vision --extra ocr
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

Create `config/manual-test.yaml` (enable one subsystem at a time so the output
is unambiguous), then point `analyze` at it:

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
```
