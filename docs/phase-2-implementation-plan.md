# Phase 2 Implementation Plan — Basic Computer Vision

**Date:** 2026-09-27
**Status:** Approved
**Scope:** Convert Phase 1 screenshots into structured information: template
matching, UI-region detection, OCR, and (optional) basic object detection.

---

## 1. Goal (from the high-level plan, §15)

> Convert screenshots into useful structured information.

Concretely, Phase 2 delivers:

* **Template matching** — locate known UI icons/shapes in a frame.
* **UI detection** — region-of-interest scanning of configured UI zones
  (e.g. "the action bar lives at `x,y,w,h`; classify what's there").
* **OCR** — read text in configured regions (quest text, item names, error
  messages).
* **Basic object detection** — cheap pixel-level heuristics (color/bright spot
  detection) as a placeholder for a real detector, kept behind an interface so
  a YOLO/VLM detector can be added later without rework.

Explicitly **not** in Phase 2 (deferred to their phases):

* VLM/LLM interpretation → Phase 5.
* Game state machine, decision engine, planning → Phases 4–7.
* Any mouse/keyboard input → Phase 3.
* Live-game tuning → always, per the headless/CI rule; Phase 2 ships with
  synthetic assets and documented extension points.

---

## 2. Architecture fit

Phase 1 produced `Frame` (row-major RGB `bytes` + dimensions + timestamp).
Phase 2 consumes `Frame` and emits a new first-class type: **`Observation`**.

```text
Phase 1 (exists)                    Phase 2 (new)
─────────────────                   ───────────────────────────────
Screen ─► Capture ─► Frame  ─►  Perception (pipeline)
                                  ├── TemplateMatcher     (OpenCV)
                                  ├── UiRegionDetector    (OpenCV / heuristics)
                                  ├── OcrEngine           (injectable)
                                  └── ObjectDetector      (injectable)
                                        │
                                        ▼
                                  Observation  ─►  (Phase 4: State Manager)
```

Responsibility rules (per AGENTS.md "Keep the Architecture Modular"):

* Perception **reads frames, never touches input, config-IO, or AI**.
* Every detector depends on an abstract interface, not a concrete engine.
* Detectors are pure functions of `(Frame, parameters)` → result: no global
  state, no hidden I/O, trivially testable.
* `perception` imports from `capture` (for `Frame`) but **never** the reverse.
  `capture` stays perception-agnostic.

---

## 3. Module layout

New package `src/ai_game_agent/perception/` (mirrors the logical `perception/`
folder in AGENTS.md, nested under the existing top-level package so the
installed name and test imports don't change shape):

```text
src/ai_game_agent/
    __init__.py
    __main__.py            # + `analyze` subcommand (section 9)
    capture.py             # unchanged
    config.py              # + PerceptionConfig (section 8)
    perception/
        __init__.py        # re-exports: Perception, Observation, errors
        observation.py     # Observation + sub-models (the contract)
        pipeline.py        # Perception: composes detectors, builds Observation
        base.py            # detector/OCR protocol interfaces + Detection types
        template.py        # TemplateMatcher (OpenCV)
        ui.py              # UiRegionDetector (region-based heuristics)
        ocr.py             # OcrEngine protocol + TesseractEngine (lazy)
        objects.py         # ObjectDetector protocol + ColorBlobsDetector
```

Design rules:

* **Lazy heavy imports** — `cv2` and any OCR engine are imported inside the
  functions that need them (same pattern as Phase 1's `to_pil()` / `mss`), so
  the package imports cleanly in a bare venv and tests of `observation.py`
  need zero third-party deps.
* **Interface-first** — `TemplateMatcher`, `OcrEngine`, and `ObjectDetector`
  are protocols/ABCs. The `Perception` pipeline is constructed with
  instances (dependency injection); tests inject fakes.

---

## 4. The contract: `Observation`

The single output type of Phase 2. Everything downstream (state machine in
Phase 4, VLM context in Phase 5) consumes this, so it must be stable,
validated, and serializable.

```python
# perception/observation.py
@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    width: int
    height: int

@dataclass(frozen=True)
class TemplateHit:
    template: str          # logical name from config, e.g. "target_frame"
    bbox: BBox
    confidence: float      # 0.0–1.0 (normalized match score)

@dataclass(frozen=True)
class TextRegion:
    text: str
    bbox: BBox
    confidence: float

@dataclass(frozen=True)
class ObjectHit:
    kind: str              # e.g. "loot_glow", "hostile_marker"
    bbox: BBox
    confidence: float

@dataclass(frozen=True)
class Observation:
    frame_width: int
    frame_height: int
    captured_at: datetime
    source: str
    templates: tuple[TemplateHit, ...]
    text_regions: tuple[TextRegion, ...]
    objects: tuple(ObjectHit, ...)
    detector_errors: tuple[str, ...]   # per-detector failures, never fatal
```

Rules:

* **Frozen dataclasses, tuples for collections** — mirrors Phase 1's
  `Frame` style; hashable, no accidental mutation.
* **`detector_errors`** — one detector failing (e.g. OCR engine missing)
  must not fail the pipeline; the error is recorded as a string
  (`"ocr: tesseract not installed"`) and the rest of the observation stands.
  Matches AGENTS.md error classification (`PERCEPTION_ERROR` is logged, not
  raised through the pipeline).
* **`to_dict()` / `to_json()`** for logging, replay, and later AI context.
  `to_dict()` must be JSON-serializable (ISO timestamps, plain values).
* Coordinates are in **frame space** (after Phase 1 region/scale), not screen
  space — keeps the contract independent of capture configuration.
* **`schema_version`** — this field is an open question (see §13, "leaning
  yes") and is built in step 1 (the `Observation` dataclass above is shown
  without it). Resolve §13 before implementing step 1 so the field is present
  from the first commit rather than back-filled into an already-stable contract.

---

## 5. Detectors

### 5.1 Template matching (`template.py`)

* Interface:

  ```python
  class TemplateMatcher(Protocol):
      def match(self, frame: Frame, template: str) -> TemplateHit | None: ...
  ```

* `CvTemplateMatcher`:
  * templates are PNG files referenced by *logical name* in config
    (`templates: {"target_frame": "assets/templates/target_frame.png"}`) —
    config never hard-codes game-specific names into code;
  * uses `cv2.matchTemplate` with `TM_CCOEFF_NORMED`; threshold from config
    (`perception.template_threshold`, default `0.8`);
  * frame → `cv2` array via `np.frombuffer(frame.pixels, uint8).reshape(h, w, 3)`
    (zero-copy) then `cv2.cvtColor(..., COLOR_RGB2BGR)` — `Frame.pixels` is
    documented as RGB, OpenCV is BGR; this conversion is the **only** place
    the layout matters;
  * largest-match score → confidence; top-left → `BBox`.
* Template files are loaded once at matcher construction (cache keyed by name);
  a missing/corrupt template file raises `PerceptionError` **at construction
  time** (fail fast), not per-frame.

### 5.2 UI region detection (`ui.py`)

Deterministic heuristics over *configured* UI zones — the "UI detection" item
of the phase goal without any ML:

* Config declares zones with a *check*:

  ```yaml
  perception:
    ui_zones:
      - name: action_bar
        x: 640
        y: 940
        width: 640
        height: 120
        check: brightness            # mean-luma threshold over the zone
  ```

* Checks are a small closed set (extensible): `presence` (template matched in
  zone), `brightness` (mean-luma threshold), `color_present` (fraction of
  pixels inside a color distance). Unknown check names must fail at config
  load (`extra="forbid"` + a validated enum-like field), not silently disable
  the zone.
* Output: zone name + which check fired + confidence → emitted as a
  `TemplateHit` with `template=f"ui:{zone_name}"` so the `Observation` shape
  stays stable (no new field for Phase 2.5 needs).

Rationale: AGENTS.md's preferred order is
*deterministic pixel/UI detection → template matching → OCR → detection → VLM*.
Zone checks are the cheapest rung and cover most WoW HUD questions
("is the target frame present?", "is a bar draining?").

### 5.3 OCR (`ocr.py`)

* Interface:

  ```python
  class OcrEngine(Protocol):
      def read(self, frame: Frame, region: BBox) -> TextRegion | None: ...
  ```

* OCR is applied **only to configured text regions** (never the whole frame) —
  AGENTS.md: "crop to relevant regions".
* Engine choice (dependency policy: evaluate before adding):

  | Candidate | Weight | Notes |
  |---|---|---|
  | `pytesseract` + Tesseract binary | small Python wheel; **external binary** | classic, permissive (Apache), but a system binary is a packaging friction on Windows |
  | `easyocr` | pulls PyTorch | high-quality, very heavy; conflicts with "keep dependencies minimal" |
  | `paddleocr` | pulls PaddlePaddle | same objection |

  **Decision:** default engine = `pytesseract` behind the `OcrEngine`
  protocol, packaged as the optional extra `ocr`. Tests never run a real
  engine (a `FakeOcrEngine` returns fixed results), so CI stays clean either
  way. If `pytesseract` proves impractical on the target machine, swapping in
  another engine is a one-file change — which is precisely the point of the
  protocol.
* If the engine/`tesseract` binary is missing at runtime: record
  `"ocr: <reason>"` in `detector_errors`, return no text regions, continue.

### 5.4 Basic object detection (`objects.py`)

Deliberately a **placeholder interface + one cheap implementation**:

* `ColorBlobsDetector` — finds connected components whose mean color is within
  a distance of a configured reference color (e.g. the gold "loot glow",
  red hostile markers). Pure OpenCV (`cv2.inRange` + `cv2.connectedComponents`),
  deterministic, no model weights.
* The protocol (`ObjectDetector.match(frame) -> list[ObjectHit]`) is the real
  deliverable: when a real YOLO/VLM detector arrives (Phase 5 territory), it
  plugs into `Perception` without touching the pipeline.

### 5.5 The pipeline (`pipeline.py`)

```python
class Perception:
    def __init__(self, config, template_matcher=None, ui_detector=None,
                 ocr_engine=None, object_detector=None) -> None:
        ...  # defaults lazily construct real implementations

    def observe(self, frame: Frame) -> Observation:
        ...
```

* Runs detectors in a fixed order (templates → ui → ocr → objects),
  collecting hits; wraps each detector call in `try/except Exception`,
  recording failures in `detector_errors` and continuing (bounded, no retries).
* Any detector may be `None` (disabled) — the pipeline simply skips it.
  This gives the config `enabled: true/false` per subsystem.
* `Perception.observe` is **pure** given its inputs: same frame + config ⇒
  same observation (modulo clock, which comes from the frame).

---

## 6. New dependencies

| Extra | Packages | Why optional |
|---|---|---|
| `vision` | `opencv-python-headless>=4.9`, `numpy` (transitive) | Needed for real template matching / blob detection; tests use fakes + a tiny `Frame` → numpy helper, and a `FakeTemplateMatcher` where OpenCV isn't installed |
| `ocr` | `pytesseract>=0.3.10` | Only if OCR is enabled at runtime; tests inject a fake engine |

Rules:

* `opencv-python-headless` (not `opencv-python`): no GUI dependency, smaller,
  identical API for our purposes.
* `numpy` **is** directly imported by the `vision` code path (§5.1 uses
  `np.frombuffer` to build the frame array, and `cv2.matchTemplate` requires
  numpy arrays — there is no cv2-only way around it). Declare it explicitly in
  the `vision` extra rather than relying on opencv to pin it transitively.
  Non-vision modules must not import numpy, keeping the core zero-extra.
* Core (`dependencies`) stays as-is: `pydantic`, `PyYAML`. Phase 2 is fully
  importable and its contract-testable with zero extras installed.

`pyproject.toml`:

```toml
[project.optional-dependencies]
vision = ["opencv-python-headless>=4.9", "numpy>=1.26"]
ocr = ["pytesseract>=0.3.10"]
```

`uv sync --extra dev --extra vision --extra ocr` in the dev README snippet.

---

## 7. Testing strategy (TDD, per AGENTS.md)

Every detector ships test-first. All tests must run headless, deterministic,
in CI:

### 7.1 Unit tests

| File | Covers |
|---|---|
| `tests/test_observation.py` | `Observation` invariants, `to_dict`/`to_json` round-trip, JSON serializability, tuple immutability, empty observations |
| `tests/test_template_matcher.py` | `CvTemplateMatcher`: exact-match finds the planted template at the right `BBox`; threshold rejects noise; `confidence` in `[0,1]`; missing template file → `PerceptionError` at construction. Uses small synthetic PNGs (256×144, drawn with PIL) — no real game assets |
| `tests/test_ui_zones.py` | Each zone check: presence fires only when the template is in-zone (not out-of-zone); brightness flips across the threshold; color_present with/without the color |
| `tests/test_ocr.py` | `FakeOcrEngine` integration through the pipeline; missing-engine path populates `detector_errors` and yields no `TextRegion`; region cropping is passed through (fake asserts it saw the cropped size) |
| `tests/test_objects.py` | `ColorBlobsDetector`: planted gold blob detected with correct bbox bounds; absent color → no hits; `ObjectDetector` protocol compliance |
| `tests/test_pipeline.py` | `Perception`: all-on composition yields a full observation; one detector raising → recorded in `detector_errors`, others still present; disabled detectors skipped; determinism (two calls, same frame ⇒ equal observations) |

### 7.2 CLI / e2e

| Test | Command | Assertions |
|---|---|---|
| `test_analyze_prints_json` | `analyze --backend mock` | exit 0; stdout is a single JSON object; `json.loads` succeeds; `templates`/`text_regions`/`objects` are arrays; `frame_width == 128`, `frame_height == 72` (mock defaults) |
| `test_analyze_error_routing` | `analyze --backend doesnotexist` | exit 1; stderr starts with `error:`; stdout empty (same contract as Phase 1) |
| `test_analyze_disabled_subsystems` | `analyze --backend mock --no-ocr --no-objects` | exit 0; JSON has empty `text_regions` and `objects` |
| `test_analyze_config_file` | `analyze --backend mock --config <tmp>.yaml` (tmp config with a `perception:` block) | exit 0; config-driven perception settings take effect (e.g. a disabled subsystem is absent) — proves the `--config` wiring of §9 end-to-end |

Mock-backend notes: the plain `MockBackend` returns a flat RGB frame, so
template/OCR hits will be empty — that's fine, the *contract* is asserted
(JSON shape, routing, exit code). Detectors with planted content are covered
at unit level with synthetic frames.

### 7.3 Definition of done (phase-level)

* `uv run pytest` green (including new unit + e2e tests).
* `uv run ruff check src tests` clean.
* `uv run ai-game-agent analyze --backend mock` prints valid JSON in a bare
  dev environment (no `vision`/`ocr` extras installed).
* README updated: Phase 2 section, extras install, `analyze` usage.
* `config/default.yaml` carries a commented example `perception:` block.

---

## 8. Configuration

New `perception:` block in `config/default.yaml` (all optional — empty block
means "perception disabled", preserving Phase 1 behavior exactly):

```yaml
perception:
  enabled: true
  template_threshold: 0.8
  templates:                      # logical name -> PNG path (repo-relative)
    # target_frame: assets/templates/target_frame.png
  ui_zones: []                    # see plan §5.2 for the shape
  ocr:
    enabled: false                # default off: needs the `ocr` extra + tesseract
    engine: pytesseract
    regions: []                   # [{x, y, width, height, name}]
  objects:
    enabled: false
    # colors: [{name: loot_glow, rgb: [255, 200, 0], tolerance: 40}]
```

`config.py` additions (following the existing pattern exactly):

* `_PerceptionConfig` (pydantic, `frozen=True, extra="forbid"`);
* nested `_TemplateConfig`, `_UiZoneConfig`, `_OcrConfig`, `_ObjectConfig`;
* `PerceptionConfig(_FrozenConfig)` public immutable view;
* `Config.perception: _PerceptionConfig` with `default_factory`.

`extra="forbid"` is non-negotiable (M3 lesson from Phase 1): a typo in the
YAML must fail at load, not silently disable a detector.

---

## 9. CLI: `analyze` subcommand

```
ai-game-agent analyze [--backend NAME] [--config PATH]
                      [--no-templates] [--no-ocr] [--no-objects]
                      [--pretty]
```

* Grabs **one frame** via the existing `Capture` facade (respecting
  `capture.region`/`scale`), runs `Perception.observe(frame)`, prints
  `observation.to_json()` to **stdout**. No file is written, so there is no
  `--out` — stdout is the only output.
* **Config wiring (new in Phase 2):** the existing `load_config()` in
  `config.py` is currently unused by the CLI (`capture`/`observe` build
  `CaptureConfig` purely from flags). `analyze` is the first command to load a
  config file: `--config PATH` (default: `load_config()`'s own default, the
  repo-root `config/default.yaml` — reuse that resolution, do not invent a
  CWD-based default) supplies the `perception:` block and, for the
  capture step, any `capture.region`/`scale` the user set there. Without this,
  the `perception:` block would be inert and §8's example config untestable
  end-to-end.
* Errors follow the Phase 1 contract: `error: ...` on stderr, exit 1,
  stdout empty. The `Perception` construction (template file loading!)
  must be inside `main()`'s `try/except` — the known-red lesson from the
  Phase 1 e2e plan §6 (test 10) applies directly here: a missing template
  PNG must surface as `error: ...`, not a traceback.
* **Note:** today `main()` catches only `(ValueError, CaptureError, OSError)`.
  `analyze` routes two more exception classes through `main()`: `ConfigError`
  (bad YAML) — already defined in `config.py` but not yet imported into or
  caught by `__main__.py` — and `PerceptionError` (missing/corrupt template),
  which is new. Neither is a subclass of `ValueError`, so the catch tuple in
  `__main__.py` must be extended to include them (or both must be made
  subclasses of a shared base). Add a regression e2e test:
  `analyze --config <bad.yaml>` ⇒ exit 1, `error:` on stderr, empty stdout.
* `--pretty` → `json.dumps(..., indent=2)` for human inspection.
* Logging of the observation itself (with timestamps and config hash) goes
  to the log file, never stdout — stdout is machine-readable contract.

---

## 10. Error handling

New exception: `PerceptionError` (subclass of `Exception`, distinct from
`CaptureError` — different subsystem, per AGENTS.md error taxonomy).

| Situation | Behavior |
|---|---|
| Template file missing at construction | `PerceptionError` → `error: ...` exit 1 (fail fast, not per-frame) |
| `cv2` not installed but `vision` features enabled | `PerceptionError` at `CvTemplateMatcher` construction → clean CLI error naming the missing extra |
| OCR engine/binary missing at runtime | non-fatal: `detector_errors` entry, pipeline continues |
| A detector raises mid-`observe` | non-fatal: caught, recorded in `detector_errors`, other detectors still run |
| Region in config out of frame bounds | `PerceptionError` (same validation style as `Frame.region`) |

No silent catches anywhere: every `except` either records into
`detector_errors` or re-raises as `PerceptionError`.

---

## 11. Out of scope (deliberately)

* VLM/LLM analysis → Phase 5 (the `Observation` shape is designed to be a
  clean input to it).
* YOLO or any model-weight-based detection → the `ObjectDetector` protocol is
  the seam; a real implementation is a later, separately-scoped task.
* Live World of Warcraft asset capture → needs the game running; we design
  against synthetic assets and document how to add real ones
  (`assets/templates/<name>.png` + a config line).
* Performance optimization of the CV pipeline → §16 of the high-level plan,
  post-prototype.
* Multi-frame temporal detection (e.g. "health bar is *draining*") → needs
  Phase 4 state to be meaningful.

---

## 12. Suggested build order (each step = red → green → refactor)

1. `Observation` + `BBox` + sub-models + `to_dict`/`to_json` (no deps).
2. `PerceptionConfig` in `config.py` + `default.yaml` block (config tests first).
3. `base.py` protocols + `Perception` pipeline with **fake** detectors (pipeline tests green with no OpenCV at all).
4. `template.py` — `CvTemplateMatcher` + synthetic-asset tests.
5. `ui.py` — zone checks + tests.
6. `objects.py` — `ColorBlobsDetector` + tests.
7. `ocr.py` — protocol + `pytesseract` adapter + fake-engine tests.
8. `analyze` CLI subcommand + e2e tests.
9. README + config example + `docs/` cross-links.

Steps 1–3 need **no new dependencies**; 4–7 add `vision`; 7 adds `ocr`.
This ordering means the package is importable and its contract is tested
long before any heavy dependency is required.

---

## 13. Open questions (resolve before step 4)

* **Template asset policy**: do we commit a small set of synthetic sample
  templates under `assets/templates/` (recommended: yes, tiny PNGs, MIT), or
  keep the repo asset-free and generate them in-test? Leaning: commit, so
  `analyze` is demoable out of the box.
* **OCR default**: keep `pytesseract` as the only engine in Phase 2 and
  defer the "real" engine decision (easyocr vs paddleocr vs a VLM reading
  text in Phase 5) to the Phase 5 review? Leaning: yes.
* **`Observation` versioning**: add an explicit `schema_version: int = 1`
  field now (cheap) so Phase 4/5 consumers can guard against shape drift?
  Leaning: yes.
