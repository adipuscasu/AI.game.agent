# Code Review — `ai-game-agent` (Phase 1: Screen Observation)

Scope: `src/ai_game_agent/` (`capture.py`, `config.py`, `__main__.py`) and `tests/`.
Every finding below was verified against the current tree on **Windows / PowerShell**
with `uv` (Python 3.11+). Line numbers reference the state of the tree at review time.

## How to reproduce the verification

Original state at review time:

```powershell
uv run ruff check src tests     # -> 5 errors, all in tests/test_recorder.py
uv run pytest -q                # -> 1 failed, 75 passed, 1 skipped
```

Current state (after the fixes in this review were addressed):

```powershell
uv run ruff check src tests     # -> All checks passed!
uv run pytest -q                # -> 77 passed, 1 skipped
```

(The one skip is the environment-blocked e2e console-script test, not a failure.)

---

## Resolution (2026-09-27)

All four findings were addressed in the tree:

| ID | Status | What was done |
|----|--------|---------------|
| **H1** | ✅ Fixed (test) | `test_recorder_rotation_ignores_unrelated_pngs` now writes `screenshot.png` to disk (`unrelated.write_bytes(b"\x00")`) and the unused `base` line was dropped. Test passes. |
| **M1** | ✅ Fixed | The four dead imports (`pytest`, `Capture`, `MockBackend`, `CaptureConfig`) were removed from `tests/test_recorder.py`; dropping `base` also cleared the F841. `ruff check src tests` is clean. |
| **N1** | ✅ Documented | `--fps` is now documented as a **ceiling, not a guarantee** in the module docstring, the `observe --fps` help text, and an inline comment at the pacing loop. Behavior is unchanged (deliberate). |
| **N2** | ✅ Fixed (source) | `main()` now catches `OSError` alongside `(ValueError, CaptureError)`, so genuine I/O failures during recording surface as a clean `error: …` line with exit code 1 instead of a raw traceback. Covered by the new regression test `test_cli_capture_oserror_is_clean_error`. |

---

## Verdict

The Phase 1 capture layer is **in good shape**: it is layered, deterministic,
headless-testable, and the security-relevant paths (config validation, file
claiming, rotation isolation) are correctly implemented. The one failing test is a
**test bug, not a source bug**, and the lint errors are all dead imports in one test
file. No production-code defects were found.

---

## Findings

### 🔴 H1 — One failing test is a test bug (`tests/test_recorder.py`)

`test_recorder_rotation_ignores_unrelated_pngs` fails, but the source is correct.

`tests/test_recorder.py:67-78`:

```python
base = "frame_20260101T000000000000"          # F841: assigned, never used
unrelated = tmp_path / "screenshot.png"       # Path object, never written to disk
(tmp_path / "reference.png").write_bytes(b"\x00")
(tmp_path / "frame_not_a_timestamp.png").write_bytes(b"\x00")
...
assert unrelated.exists()                     # FAILS: file was never created
```

`screenshot.png` is only constructed as a `Path`; it is never written, so
`unrelated.exists()` is `False`. The other two unrelated files *are* written and
are correctly preserved by rotation — which is exactly what the source is designed to
do (`Recorder._rotate()` filters via `_FRAME_FILE_RE`, `capture.py:28`, and therefore
never deletes unrelated PNGs).

**Fix** (test, not source): write the file so it matches its own intent:

```python
unrelated = tmp_path / "screenshot.png"
unrelated.write_bytes(b"\x00")
```

### 🟠 M1 — 5 lint errors, all in `tests/test_recorder.py`

```
F401  pytest            imported but unused   tests/test_recorder.py:3
F401  Capture           imported but unused   tests/test_recorder.py:5
F401  MockBackend       imported but unused   tests/test_recorder.py:5
F401  CaptureConfig     imported but unused   tests/test_recorder.py:6
F841  base              assigned but unused   tests/test_recorder.py:67
```

All are in a single file and are auto-fixable:

```powershell
uv run ruff check src tests --fix    # clears the 4 F401s; F841 needs the H1 fix above
```

Note the coupling: fixing H1 by *removing* the unused `base` line also clears the F841.

### 🟡 N1 — Achieved FPS is structurally below target (`src/ai_game_agent/__main__.py`)

The `observe` loop sleeps *after* the work, so each iteration costs
`grab() + record() + sleep(1/fps)` — the measured rate is therefore a little under the
requested one:

`__main__.py` `_observe`:

```python
for _ in range(max(0, args.frames)):
    cap.grab()                                  # work
    meter.record(time.perf_counter())
    cap.wait_next_frame()                       # sleep(1/fps)
```

A probe with `fps=30` and ~1 ms of work measured **28.4 fps** (expected: `30 / (1 + 30·t_work)`).
This is not a correctness bug — `FpsMeter` honestly reports the real rate — but it is a
worth-noting design fact: the CLI never *reaches* the requested FPS, it asymptotically
approaches it from below. If hitting the target matters, either subtract the measured
work time from the sleep, or document that `--fps` is a ceiling, not a guarantee.

### 🟡 N2 — `OSError` during recording surfaces as an unhandled traceback

`main()` only converts `(ValueError, CaptureError)` into a clean `error:` line:

`__main__.py` `main`:

```python
try:
    return int(args.func(args))
except (ValueError, CaptureError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
```

`save_frame` (which `Recorder.record` calls) opens/claims/writes real files. A
permission error, full disk, or a read-only target raises `OSError`, which is
**not** in the caught set, so the CLI exits with a raw traceback rather than a clean
`error:` message. Config and validation errors are fine (verified: pydantic
`ValidationError` is a subclass of `ValueError`, so `main()` catches it). The only
uncaught path is genuine I/O failure. Consider catching `OSError` alongside, or
wrapping the write in `save_frame` into a `CaptureError`.

---

## Verified-correct (checked, not assumed)

- **Rotation never deletes unrelated files.** `Recorder._rotate()` filters with
  `_FRAME_FILE_RE` (`capture.py:28`), so only `frame_<timestamp>[-N].png` files are
  candidates. Covered by `test_recorder_rotation_prefers_oldest_on_suffix_collision`
  and the (currently broken) `test_recorder_rotation_ignores_unrelated_pngs`.
- **Collision-safe writes.** `save_frame` claims the name with
  `os.O_CREAT | os.O_EXCL` and appends `-N` on `FileExistsError`, closing the
  exists-then-write TOCTOU window.
- **Config fail-fast.** `_CaptureConfig` uses `extra="forbid"` so typos like
  `backends:` raise instead of silently defaulting; `logging.level` is validated
  against the allowed set; `region` is validated (`width`/`height > 0`).
- **`fps = 0` is an intentional, handled feature** (headless/CI "no pacing"):
  documented at `config.py:46`, bounded `ge=0` at `config.py:47`, and short-circuited
  in `Capture.wait_next_frame` (`capture.py`, `if self.config.fps <= 0: return 0.0`).
- **Artifact dirs are git-ignored.** `.gitignore` covers `recordings/`, `screenshots/`,
  `shots/`, and `logs/`; `git ls-files shots logs recordings` returns nothing.
- **Headless determinism.** `MockBackend` precomputes its pixel buffer once; `Clock`
  is injectable; `Frame` is a frozen dataclass of raw `bytes`, so pixel math needs no
  image library and `to_pil()` defers the optional Pillow import.

## Blocked in this environment (not a code defect)

- The one **skipped** test is the e2e CLI test:
  `tests/e2e/test_e2e_cli.py:174` — skipped because
  `[WinError 4551] An Application Control policy has blocked this file`, i.e. the
  `uv run` console entry-point cannot be executed here. It is not a test failure.

---

## Strengths

- **Clean layering.** `CaptureBackend` interface → `MockBackend`/`MssBackend` →
  `Capture` facade → `Recorder`. Tests inject a mock backend and clock, so the whole
  pipeline runs headless.
- **Backend-agnostic frame model.** `Frame` carries raw RGB `bytes` plus dims/timestamp;
  `region()` and `resize()` are pure-Python (no PIL required), and `to_pil()` keeps
  Pillow optional.
- **Thoughtful edge cases.** Numeric (not lexicographic) ordering of collision suffixes
  in `_rotation_key`; mss 9-vs-≥10 `shot.size` dict/Size handling; `extra="forbid"`
  config.
- **Testable by construction.** `MockBackend`, injectable `clock`, and `FpsMeter` are
  all seams that let the suite assert exact behavior without a display.

## Suggested order of fixes

1. Fix the H1 test (write `screenshot.png`) — this also removes the F841.
2. `uv run ruff check src tests --fix` for the four dead imports.
3. Optionally broaden `main()`'s `except` to include `OSError` (N2).
4. Optionally document/adjust the FPS-ceiling behavior (N1).

---

## Phase 2 — current state (updated 2026-10-03, HEAD `15fba73`)

> Supersedes the transient review at `0c03ec1`. Verified with `python3 -m pytest -q`
> and by importing each module against the real package.

**Verdict:** Phase 2 steps 1–4 are green; the three 🔴 criticals from the prior
review (broken `test_pipeline.py`, API-mismatch `test_objects.py`, stray root
`perception/` drafts) are **resolved**. Full suite: **137 passed, 1 failed** — the
single failure is `tests/test_mss_backend.py` failing because the optional `mss`
extra is not installed in this environment (should skip, not fail).

### Still open

| # | Sev | Where | Finding |
|---|-----|-------|---------|
| 1 | 🟠 | `perception/objects.py` | Broken at runtime: `super().__init__(config)` on a `Protocol`; `BBox`/`DetectedObject` used without import; `self.config.get(...)` on a frozen config; `ObjectHit(object_type=…, detection_source=…)` vs real `(kind, bbox, confidence)`; reads `frame.image` (real `Frame` → `pixels`/`to_image()`); defines `run()` not `match()`; `print()` logging |
| 2 | 🟠 | `perception/color_blobs.py` | Import error: imports nonexistent `BlobHit`; uses `frame.data`; returns `List[Observation]`. Dead code — delete or fix |
| 3 | 🟡 | `assets/templates/target_frame.png` | Not a valid PNG (`PIL.UnidentifiedImageError`); blocks the template/step-8 demo |
| 4 | 🟡 | `tests/test_mss_backend.py` | Fails instead of skipping when `mss` is absent — add a `pytest.skip` guard |
| 5 | ⚪ | repo (39 files) | CRLF flip; no `.gitattributes` / `core.autocrlf=input` |

### Not started (roadmap)

Step 7 `perception/ocr.py` + `tests/test_ocr.py`; step 8 `analyze` subcommand
(`__main__.py` still only `capture`/`observe`; the `except` tuple lacks
`ConfigError`/`PerceptionError`); step 9 README Phase 2 section. `pyproject.toml`
already ships the `vision` + `ocr` extras, so the dependency scaffolding is in place.

### Suggested order

1. Fix `objects.py` to the `ObjectDetector` protocol (`match(frame) -> list[ObjectHit]`, `ObjectHit(kind, bbox, confidence)`, no `print`) — or delete `color_blobs.py`.
2. Replace `target_frame.png` with a valid synthetic PNG.
3. `pytest.skip` guard in `tests/test_mss_backend.py`.
4. Write `tests/test_objects.py` red→green.
5. `core.autocrlf=input` (or add `.gitattributes` forcing LF).
6. Proceed to steps 7–9.

---

## Phase 2 — re-review (updated 2026-10-04, HEAD `1f93ade`)

> Supersedes the 2026-10-03 section above. Verified with `uv run pytest -q`,
> `uv run ruff check src tests`, `uv lock --check`, and by running the `analyze`
> subcommand in both the real (cv2/numpy present) and the bare (cv2/numpy
> import-blocked) environments.

**Verdict:** Phase 2 is **functionally complete and green.** All five open
items from the 2026-10-03 review are resolved (the `mss` skip-guard was fixed
in `1f93ade`). Full suite **205 passed**, ruff clean, lockfile in sync.
The branch is ready to merge.

### Resolution of the 2026-10-03 open items

| # | Sev (10-03) | Where | Status (10-04) |
|---|-------------|-------|----------------|
| 1 | 🟠 | `perception/objects.py` | ✅ **Resolved** — `objects.py` removed; `perception/color_blobs.py` is now the real `ColorBlobsDetector` (commit `e73a495`). |
| 2 | 🟠 | `perception/color_blobs.py` | ✅ **Resolved** — import errors gone; implements the `ObjectDetector` protocol and is covered by 21 tests in `tests/test_objects.py`. |
| 3 | 🟡 | `assets/templates/target_frame.png` | ✅ **Resolved** — replaced with a valid 64×36 PNG (opens in PIL; matches at confidence 1.0 under `CvTemplateMatcher`). |
| 4 | 🟡 | `tests/test_mss_backend.py` | ✅ **Resolved** (`1f93ade`) — `pytest.importorskip("mss")` added; the test now skips cleanly on a bare venv. |
| 5 | ⚪ | repo CRLF | ✅ **Resolved** — all text files normalized to LF; `.gitattributes` added (`text=auto eol=lf`, `*.ps1` kept CRLF, image assets `binary`) so it can't recur. |

### Not-started items (10-03) — all now complete

| Item | Status |
|------|--------|
| Step 7 `perception/ocr.py` + `tests/test_ocr.py` | ✅ `TesseractEngine` behind the `OcrEngine` protocol, 14 tests (commit `a7adc1a`). |
| Step 8 `analyze` subcommand | ✅ Wired end-to-end in `__main__.py`; `--no-templates` / `--no-ocr` / `--no-objects` toggles; clean `error:` + exit-1 on `ConfigError`/`PerceptionError`/`OSError`. |
| Step 9 README Phase 2 section | ✅ `analyze` subcommand + perception pipeline documented (commit `babb684`). |

### Still open (re-verified 2026-10-04)

| # | Sev | Where | Finding |
|---|-----|-------|---------|
| 1 | ⚪ | `__main__.py:312` | `--no-templates` also disables UI-zone checks (they share one flag). Documented in the help text and covered by `test_no_templates_flag_disables_ui_detector`, but the flag name doesn't signal it. Naming-clarity only, not a bug. |

### Re-review — verified correct (checked, not assumed)

- **Bare-env `analyze` works.** With `cv2`/`numpy` import-blocked, `main(["analyze", "--backend", "mock"])` returns rc=0 and emits valid JSON with `schema_version==1` and `detector_errors==[]`. The lazy `color_blobs` import (commit `7fda6af`) means the CLI loads and degrades cleanly without the `vision` extra.
- **Real-detector `analyze` works.** With the `vision` extra, a config enabling a template + a brightness UI zone + an object color returns rc=0, detects the object (`kind=red_blob`), and reports no `detector_errors`.
- **Missing template file is a clean error.** `templates.ghost: /tmp/does_not_exist.png` → `error: template file not found: …`, exit 1, no traceback.
- **Config fail-fast.** An unknown UI-zone `check:` value raises `ConfigError` (pydantic `extra="forbid"` + validator) rather than silently passing.
- **No dead code left.** `_default_backend` and the unused `ocr.engine` knob are gone; `Pillow` is now declared explicitly in the `ocr` extra (it was transitive-only).
- **Whitespace is stable.** `.gitattributes` enforces LF; the previously-churned 14 files now show `+0 −0` vs `develop`; the whole-branch diff shrank from **48 files / +9393** to **35 files / +4934 −8**.

### Verification snapshot (2026-10-04, HEAD `1f93ade`)

```
uv run pytest -q                            -> 205 passed
uv run ruff check src tests                 -> All checks passed!
uv lock --check                             -> in sync (23 packages)
analyze (bare, cv2/numpy blocked)           -> rc=0, valid JSON, detector_errors=[]
analyze (real: template + ui_zone + objects) -> rc=0, object detected, no errors
```

### Suggested order

1. (Optional) Rename or further document the `--no-templates` / UI-zone coupling for discoverability.
2. Merge `feature/phase-2` into `develop`.

## Phase 2 — third review (2026-10-06, PR #2 head)

> External re-review of the current head of PR #2 (snapshot taken at head
> `1e0bfad`, 23 commits). Verdict: **Request changes — 1 remaining P1**
> (`perception.enabled` not honored during CLI detector construction), plus
> two non-blocking items (stale PR description; no `--no-record`).

### Resolution of the third review

| # | Sev | Where | Finding | Resolution |
|---|-----|-------|---------|------------|
| 1 | 🔴 P1 | `__main__.py` `_build_perception` | `perception.enabled: false` still constructed detectors (template matcher, OCR engine, color-blob detector) before the `Perception` pipeline could short-circuit; a configured template or object color could still require the optional `vision` extra even though the master switch said off. | ✅ Fixed in `0b4b867` — `_build_perception` now returns `Perception(perception)` immediately when `cfg.perception.enabled` is False, before any detector is constructed. No optional dependency is touched and a misconfigured template path can no longer turn "perception disabled" into a setup error. |
| 2 | 🟠 P2 | PR description | Still reported 15 commits / 37 files / 194 tests while the PR had grown to 23 commits. | ⚠️ Cleanup — PR body refresh required (needs a GitHub write credential not present in this environment). |
| 3 | 🟡 P3 | `observe` CLI | `capture.record.enabled: true` could not be overridden off for a single invocation (two-state sentinel). | ✅ Fixed in `5c5ffa3` — `--no-record` added alongside `--record`; both are `None` sentinels so precedence is flag (`--record`/`--no-record`) → `capture.record.enabled` → `False`. |

### Regression tests added

- `test_cli_analyze_perception_disabled_skips_optional_detectors` (in-process):
  config with `perception.enabled: false` + a missing template + object
  colors; asserts `analyze` exits 0 with an empty observation and no
  `detector_errors`. **Verified red** against pre-fix code `1e0bfad`
  (`error: template file not found: missing-template.png`, exit 1).
- `test_cli_analyze_perception_disabled_works_in_bare_env` (subprocess,
  `meta_path` blocker): same shape of config with **every** detector
  subsystem enabled (templates, `ui_zones`, OCR regions, object colors)
  while `cv2`/`numpy`/`PIL`/`pytesseract` are import-blocked — the
  reviewer's stronger invariant: `perception.enabled=false` means the CLI
  requires no vision/OCR dependency at all. **Verified red** against
  `1e0bfad` (exit 1 with `cv2` blocked), green after the fix.

### Import-hygiene hardening (post-review, same P1)

The reviewer's suggested "further improvement" was applied: `_build_perception`
was returning a detector-free `Perception` when disabled, but `__main__.py`
still imported the three detector implementation modules (`template`, `ocr`,
`ui`) at **module scope**, so `perception.enabled:false` skipped *construction*
yet still *imported* them. All three imports were moved into the selecting
branch of `_build_perception` (the same lazy pattern already used for
`color_blobs`), so the contract is now:

**`perception.enabled: false` → no perception implementation is imported,
instantiated, or executed.**

- `test_cli_analyze_perception_disabled_imports_no_impl_modules` (subprocess):
  after a disabled-perception `analyze`, asserts none of `perception.template`/
  `perception.ocr`/`perception.ui`/`perception.color_blobs` is in `sys.modules`.
  **Verified red** against the module-scope-import version (leaked all three),
  green after moving the imports.
- No test or source referenced the `__main__` re-exports of the three names
  (checked before moving), so nothing depended on them.

> Note on the reviewer's suggested extra assertion: the Observation JSON has
> **no `ocr` key** — it is `text_regions`. The empty-observation assertions
> already in `…_works_in_bare_env` use the correct key.

### Verification snapshot (2026-10-06, local head after import-hygiene fix)

```
pytest -q                                   -> 216 passed
ruff check src tests                        -> All checks passed!
analyze disabled-perception (bare env)      -> rc=0, empty observation, detector_errors=[]
```

### Suggested order

1. Refresh the PR description (commits/files/tests) to match the current head.
2. Merge `feature/phase-2` into `develop` once CI is green on the final head.

## Phase 2 — fourth review (2026-10-06, PR #2 head)

> External re-review of the current head of PR #2. Verdict: **Request changes —
> 1 P1 + 1 P2**, with the earlier import-hygiene concern carried over as the
> P1. Re-verified against the live tree before acting (the review snapshot lagged
> the head).

### Resolution of the fourth review

| # | Sev | Where | Finding | Resolution |
|---|-----|-------|---------|------------|
| 1 | 🔴 P1 | `__main__.py` module scope | The import-hygiene test asserts `perception.enabled:false` must not import `perception.template`/`ocr`/`ui`, yet `__main__.py` was reported to import those three at module top. | ✅ **Already fixed** — the live head (commit `da0bf9a`, "import perception impls only when their subsystem is selected") already moved all three into the selecting branch of `_build_perception`, behind the `enabled` master switch. `test_cli_analyze_perception_disabled_imports_no_impl_modules` passes against the live tree. No re-fix needed. |
| 2 | 🟠 P2 | `__main__.py` `build_parser` / `_observe` | `--record` and `--no-record` wrote to two independent argparse destinations, so `--record --no-record` evaluated as `args.record if args.record is not None else args.no_record` → `True`. **Both** flag orders enabled recording, contradicting the documented "last flag wins". | ✅ **Fixed in this commit** — both flags now share the single `record` destination (`store_const` True/False, default `None`), so argparse applies them in command-line order and the last flag wins: `--record --no-record` → off, `--no-record --record` → on, no flag → `None` → config. `_observe` simplified to `record_enabled = args.record`. |
| 3 | 🟡 Minor | PR description | Still reported 15 commits / 37 files / 194 tests while the PR had grown to 30 commits. | ⚠️ Cleanup — PR body refresh requires a GitHub write credential not present in this environment. |

### Regression tests added (red → green)

- `test_cli_observe_record_no_record_last_flag_wins`: `--record --no-record`
  with config `record.enabled: true` must record **nothing** (the last flag,
  `--no-record`, wins). **Verified red** against the pre-fix code (3 frames were
  written), green after the fix.
- `test_cli_observe_no_record_record_last_flag_wins`: `--no-record --record`
  must record **3 frames** (the last flag, `--record`, wins). The two orders
  now produce opposite results, matching the documented CLI contract.

### Verification snapshot (2026-10-06, this commit)

```
pytest -q                                   -> 225 passed
ruff check src/ai_game_agent/__main__.py tests/test_cli.py -> All checks passed!
--record --no-record  -> record = False   (last flag wins)
--no-record --record  -> record = True    (last flag wins)
(no flag)             -> record = None    (falls back to config)
```

### Suggested order

1. Refresh the PR description (commits/files/tests) to match the current head.
2. Merge `feature/phase-2` into `develop` once CI is green on the final head.
