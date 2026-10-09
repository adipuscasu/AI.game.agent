# Phase 2 (Perception) — To-Do

> Verified against the working tree on **2026-10-03** (branch `feature/phase-2`,
> HEAD `0c03ec1` "work in progress"). Companion to
> [`phase-2-implementation-plan.md`](phase-2-implementation-plan.md); §14 of that
> plan is stale and should be updated to match this document.

## Summary

| Step | Item | State |
|---|---|---|
| 1 | `Observation` + sub-models + `to_dict`/`to_json` + `schema_version` | ✅ Done (`perception/observation.py`, `tests/test_observation.py`) |
| 2 | `PerceptionConfig` + `default.yaml` `perception:` block | ✅ Done (`config.py`, `config/default.yaml`, `tests/test_perception_config.py`) |
| 3 | `base.py` protocols + `Perception` pipeline (fake detectors) | ✅ Done (`perception/base.py`, `perception/pipeline.py`, `tests/test_pipeline.py`) |
| 4 | `CvTemplateMatcher` | ✅ Done (`perception/template.py`, `tests/test_template_matcher.py`) |
| 5 | `ui.py` — zone checks | ⚠️ Code done, **tests broken** (`ui.py` 162 LOC; `tests/test_ui_zones.py` fails to collect) |
| 6 | `objects.py` — `ColorBlobsDetector` | ⚠️ Code done, **tests broken** (`objects.py` 110 + `color_blobs.py` 89 LOC; `tests/test_objects.py` has 1 error, 1 failure) |
| 7 | `ocr.py` — `pytesseract` adapter | ❌ Not started — no `ocr.py`, no `tests/test_ocr.py` |
| 8 | `analyze` CLI subcommand + e2e tests | ❌ Not started — `__main__.py` has only `capture`/`observe` |
| 9 | README + config example + `docs/` cross-links | ⚠️ Partial — `default.yaml` example block done; README has no Phase 2 section |
| — | §13 "Template asset policy" decision | ❌ Pending — gates the `analyze` out-of-the-box demo only |

Test suite status at time of writing: **2 failed, 134 passed, 1 skipped, 1 error**
(with `mss` not installed, see Housekeeping); `tests/test_ui_zones.py` fails at
collection and blocks the whole run.

## Remaining work

### 1. Fix the step 5–6 tests (blocking; do first)

The test files landed in commit `0c03ec1` but do not pass — and
`test_ui_zones.py` fails at collection, which interrupts the entire suite.

**`tests/test_ui_zones.py` (collection error):**
- `NameError: name 'TemplateHit' is not defined` — annotations
  (`List[TemplateHit]`, `list[BBox]`) reference names that are only defined as
  mocks *after* their first use; the file also defines `MockObservation` twice.
- The file hand-rolls mock classes instead of importing the real models from
  `ai_game_agent.perception.observation`; per the plan it should test the real
  `UiRegionDetector` from `perception/ui.py` (zone checks `presence`,
  `brightness`, `color_present`; emits `TemplateHit(template=f"ui:{zone_name}")`).

**`tests/test_objects.py` (1 error, 1 failure):**
- ERROR: `test_color_blobs_detector_success` — `@patch(...cv2.inRange)` decorators
  inject `mock_findContours` but **no such fixture exists** (typo/leftover; the
  patch list should match the decorators exactly).
- FAILED: `test_color_blobs_detector_config_error` — constructs
  `PerceptionConfig({"object_detection": {"color_blobs": {}}})`, i.e. passes a
  dict where the first positional arg (`enabled: bool`) is expected →
  `pydantic ValidationError`. Should use keyword args and the config fields that
  actually exist (`objects_enabled`, `object_colors`).
- Note: `test_objects.py` and `test_ui_zones.py` both reference `cv2` and mock
  internals; verify they exercise `perception/ui.py` and
  `perception/objects.py`/`color_blobs.py` against the **real** protocols in
  `perception/base.py`, not private helpers.

**Definition of done:** `python -m pytest -q` is green with the `vision` extra
installed (or the affected tests skip cleanly when it isn't, per the repo's
optional-dependency convention).

### 2. Step 7 — `perception/ocr.py` (TDD)

- `TesseractEngine` implementing the `OcrEngine` protocol from `perception/base.py`;
  the pipeline already accepts an `ocr_engine` injection point.
- Lazy `pytesseract` import; `ocr` is an extra (`pyproject.toml`) — missing
  package or missing Tesseract binary must land in `detector_errors` as
  `"ocr: …"`, **never** fail the pipeline.
- `tests/test_ocr.py` **first** (red), using a `FakeOcrEngine` per the plan;
  config side (`_OcrConfig`, `ocr_regions`) already exists in `config.py`.
- Wire into the `Perception` constructor alongside `ui_detector`.

### 3. Step 8 — `analyze` CLI subcommand (TDD, per plan §7.2 / §12)

In `src/ai_game_agent/__main__.py`:
- [ ] `analyze`: one-frame capture via the existing `Capture` facade →
      `Perception.observe(frame)` → print `observation.to_json()` to stdout
      (stream routing: stdout = JSON only, stderr = `error:`/usage).
- [ ] `--config PATH` (default: `load_config()`'s own default) — first CLI
      command to load a config file.
- [ ] Flags: `--no-templates`, `--no-ocr`, `--no-objects`, `--pretty`.
- [ ] Extend `main()`'s catch tuple — currently
      `(ValueError, CaptureError, OSError)` — to also catch `ConfigError` and
      `PerceptionError` (neither is a `ValueError` subclass); import them.
- [ ] e2e suite (currently absent): `test_analyze_prints_json`,
      `test_analyze_error_routing`, `test_analyze_disabled_subsystems`,
      `test_analyze_config_file`, plus the §9 regression
      (`analyze --config <bad.yaml>` ⇒ exit 1, `error:` on stderr, empty stdout).
- [ ] Depends on: real detector wiring (steps 5–6 fixed, step 7) and the §13
      decision below for the demoable case.

### 4. Step 9 — README + docs cross-links (last)

- [ ] README Phase 2 section (pipeline diagram, `analyze` usage, optional
      extras `vision`/`ocr`, config reference).
- [ ] Cross-links between `README.md`, `docs/phase-2-implementation-plan.md`,
      and this file.

### 5. Decide §13 "Template asset policy" (one line; gates only the demo)

- [ ] Decide how template assets are sourced/validated
      (repo-committed PNGs vs user-supplied vs generated).
- [ ] Only blocker for the `analyze` out-of-the-box demo; everything above can
      proceed without it. One asset already exists:
      `assets/templates/target_frame.png`.

## Recommended order

1. Fix step 5–6 tests (suite is currently un-runnable due to the collection
   error).
2. Step 7 (`ocr.py`) — TDD.
3. Step 8 (`analyze` CLI + e2e) — after 7 and the §13 decision.
4. Step 9 (README/docs) — once `analyze` is stable.
5. Refresh §14 of `phase-2-implementation-plan.md` to match reality
   (it still says steps 5–6 "not started" and omits `color_blobs.py`).

## Housekeeping observed

- **Dirty working tree is line-ending noise only.** All ~37 modified files show
  identical +N/−N counts and `git diff -w` / `--ignore-all-space` are empty;
  `git ls-files --eol` reports `i/lf w/crlf` with no clean attributes. The
  working copy was checked out/edited on Windows (venv is a Windows venv:
  `C:\Users\adipu\...` Python 3.14). Fix with `core.autocrlf=input` + a
  `git add -u` re-normalization pass, or add `.gitattributes` forcing LF.
- **`mss` not installed in this environment** →
  `tests/test_mss_backend.py::test_create_backend_mss_returns_mss_backend`
  fails here by design (it exercises the `CaptureError`-when-missing path only
  if the test itself mocks availability; currently it expects a real backend).
  Not a Phase 2 bug; install the extra or mark the test accordingly.
- `pyproject.toml` already declares `ocr = ["pytesseract", "Pillow"]` and
  `vision = ["opencv-python-headless", "numpy"]`; no dependency work needed for
  steps 7–8.
- `assets/templates/` exists with one placeholder asset
  (`target_frame.png`, 3 bytes — verify it is a valid PNG or replace it).

## Verification baseline (2026-10-03)

```
python3 -m pytest -q --ignore=tests/test_ui_zones.py
# 2 failed, 134 passed, 1 skipped, 1 error
#   FAILED tests/test_mss_backend.py::test_create_backend_mss_returns_mss_backend
#   FAILED tests/test_objects.py::test_color_blobs_detector_config_error
#   ERROR  tests/test_objects.py::test_color_blobs_detector_success
#   SKIPPED tests/e2e/test_e2e_cli.py:154 (console script not on PATH)

python3 -m pytest -q
# interrupted: tests/test_ui_zones.py collection NameError (TemplateHit)
```

Environment: Python 3.11.2, pydantic 2.x, PyYAML, pytest, freezegun, Pillow,
numpy, opencv-python-headless 5.0.0, no `mss`, no `pytesseract`.
