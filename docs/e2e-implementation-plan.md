# E2E Implementation Plan — Phase 1 (Screen Observation)

**Date:** 2026-09-27
**Status:** Proposed (pending review)
**Scope:** End-to-end tests for the Phase 1 CLI contract (`ai-game-agent capture` / `observe`).

---

## 1. Why E2E, and why not Playwright

The existing suite (`tests/test_*.py`) is unit/integration level: it imports the
package and calls `main([...])` in-process. That verifies logic but does **not**
verify the observable contract a user or CI job actually experiences:

* the installed entry point resolves (`pyproject.toml` `[project.scripts]`)
* the module import path works from a *separate* Python process
* argument parsing, exit codes, and stdout/stderr routing
* files on disk (PNG validity, sizes, rotation counts)

**Playwright is not the right tool.** Playwright drives browser UIs; Phase 1 is a
headless CLI + library with no web surface. E2E here means **black-box
subprocess tests**. (Playwright becomes relevant only if a later phase ships a
browser-based monitoring UI.)

## 2. Design principles (per AGENTS.md)

| Requirement | How it is met |
|---|---|
| Headless / no desktop session | `--backend mock` everywhere |
| Deterministic | `--fps 0` (no pacing), synthetic mock frames |
| CI-safe | stdlib for the test harness (`subprocess`, `os`, `shutil`, `re`, `pathlib`); **no new dependencies** |
| Runtime deps | Pillow (`PIL`) is required for PNG assertions — it is already a `dev` extra, and the subprocess inherits the uv venv, so it must be present (run under `uv run`) |
| TDD | Tests are the specification of the CLI contract; any failure is a product bug, not a test bug |
| Independent tests | Each test uses its own `tmp_path` output directory |

## 3. Invocation mechanism

```python
REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root

def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "ai_game_agent", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
```

Key details:

1. **`python -m ai_game_agent`** (not the installed script) as the default —
   works without a build/install step.
2. **`PYTHONPATH=<repo>/src`** is mandatory: pytest's `pythonpath = ["src"]`
   applies only to the pytest *process*, not to child subprocesses.
3. **`cwd=REPO_ROOT`** so relative config/default paths resolve identically to
   a user running from the repo root.
4. **`timeout=30`** — a hung CLI must fail the test loudly, never hang CI.
5. **`sys.executable`** — uses the same interpreter as the test run (the uv
   venv), so dependency versions match.

### Optional stronger test: the installed console script

One test uses the real `ai-game-agent` executable from PATH
(`shutil.which("ai-game-agent")`), `pytest.skipif` when absent. This is the
only test that validates the `[project.scripts]` entry point end to end.

## 4. Test matrix

File: `tests/e2e/test_e2e_cli.py` — all tests marked `@pytest.mark.e2e`.

| # | Test | Command (all use `--backend mock`) | Assertions (the contract) |
|---|------|-------------------------------------|---------------------------|
| 1 | `test_capture_produces_png` | `capture --out shots` | exit 0; stdout matches `saved: .*\.png`; exactly one `*.png` in output dir; PIL opens it with size **128×72** (mock backend default) |
| 2 | `test_capture_region_and_scale_pipeline` | `capture --region 10,20,32,16 --scale 0.5 --out shots` | exit 0; PNG size is exactly **16×8** — proves region extraction → resize → PNG persistence composes correctly end to end |
| 3 | `test_observe_reports_fps` | `observe --frames 5 --fps 0` | exit 0; stdout matches `frames=5 fps=[0-9.]+`; exit fast (no pacing) |
| 4 | `test_observe_record_rotation` | `observe --frames 10 --fps 0 --record --record-max-files 3 --out rec` | exit 0; **exactly 3** PNGs remain in `rec/` (oldest rotated out at the boundary) |
| 5 | `test_invalid_region_reports_error` | `capture --region 1,2,3 --out shots` | exit ≠ 0; stderr contains `error:`; **stdout is empty** (errors must not leak to stdout); no PNG created |
| 6 | `test_invalid_scale_reports_error` | `capture --scale 0 --out shots` | exit ≠ 0; stderr contains `error:`; stdout empty; no PNG created |
| 7 | `test_no_command_prints_help` | *(no subcommand)* | exit 2; help/`usage:` text on **stderr** (argparse convention); nothing written to CWD |
| 8 | `test_console_script_smoke` *(optional, skipped if not on PATH)* | `ai-game-agent capture --backend mock --out shots` | exit 0; one PNG in output dir — validates the installed entry point |
| 9 | `test_region_exceeds_frame_reports_error` | `capture --region 0,0,9999,9999 --out shots` | exit 1; stderr contains `error:`; stdout empty; no PNG — covers the `CaptureError` (perception) path in `Frame.region()`, distinct from the `ValueError` paths in tests 5–6 |
| 10 | `test_unknown_backend_reports_error` | `capture --backend doesnotexist --out shots` | exit 1; stderr contains `error:`; stdout empty; no PNG — covers `create_backend`'s rejection branch. **Expected red on first run:** today the `CaptureError` is raised inside `_capture`'s `Capture(cfg)` construction, *before* `main()`'s `try/except`, so it surfaces as an unhandled traceback (no `error:` prefix). The fix is in the product (see §6, step 2), not the test |

### Assertion style rules

* **Exit codes** are the primary success signal, never stdout text alone.
* **Stream routing** is part of the contract: results on stdout, errors on stderr.
  Tests 5–7 and 9–10 pin this down so a future refactor can't silently move error output.
* **Error-class coverage**: tests 5–6 exercise the `ValueError` path (CLI parsing),
  while tests 9–10 exercise the `CaptureError` path (`Frame.region` / `create_backend`).
  Both map to the same `error:` prefix and exit 1, but they are distinct code paths
  (`PERCEPTION_ERROR` vs. argument validation) and must stay independently tested.
* **Disk state** is asserted via `glob("*.png")` counts + PIL `Image.open(...).size`,
  never by parsing filenames.
* Regex assertions use `re.search` with anchors (`^frames=5 `) to stay strict but
  tolerant of trailing whitespace/CR (Windows `text=True`).
* No test sleeps, reads wall-clock time, or depends on execution order.

## 5. Project wiring

### 5.1 `pyproject.toml` — marker (already done)

```toml[tool.pytest.ini_options]
markers = [
    "e2e: end-to-end CLI tests that spawn the real entry point as a subprocess",
]
```

This lets the layers run independently:

```powershell
uv run pytest -m e2e        # only E2E
uv run pytest -m "not e2e"  # unit + integration only
uv run pytest               # everything
```

### 5.2 Directory layout

```
tests/
    test_capture.py        # unit (existing)
    test_cli.py            # in-process CLI (existing)
    ...
    e2e/
        __init__.py        # (optional; not needed for pytest)
        test_e2e_cli.py    # new — subprocess black-box tests
```

`testpaths = ["tests"]` already discovers the new folder — no config change needed.

> **Note (product, not test):** in `observe`, `Capture` is constructed with
> `record_enabled=record` *and* a separate `Recorder` is created to actually
> write frames. Only the `Recorder` writes on the `observe` path today; the
> `Capture.record_enabled` flag is effectively unused there. This is intentional
> and out of scope for these tests — do not "fix" it while writing the suite.

### 5.3 README

Add an "End-to-end tests" subsection under *Running the Unit Tests* with the
`uv run pytest -m e2e` invocation and a one-line explanation that these spawn
the real CLI as a subprocess.

## 6. TDD workflow

Phase 1 is already green, so these tests act as **regression guards for the
CLI contract** rather than driving new code. The red→green loop still applies:

1. Write `tests/e2e/test_e2e_cli.py` (red = "not yet passing as a suite").
2. Run `uv run pytest -m e2e`; any failure is a genuine product/contract bug
   (import path, stream routing, rotation boundary, exit codes) — fix the
   **product**, not the test, unless the assertion itself misstates the
   intended contract.
   * **Known-red case:** test 10 will fail initially. `Capture(cfg)` is
     constructed inside `_capture`, so `create_backend`'s `CaptureError`
     escapes `main()`'s `try/except` and prints a traceback instead of
     `error: ...`. The product fix is to construct the `Capture` *inside*
     the `try` in `main()` (or move `create_backend` into the guarded
     section) so the error class is caught and reported uniformly. Do not
     weaken the assertion to a traceback; the clean `error:` prefix is the
     intended contract for every failure mode.
3. Re-run full suite + `ruff check src tests`; both must be green.
4. Commit with `test: add E2E CLI contract tests for Phase 1`.

## 7. Failure interpretation guide

| Symptom | Likely cause |
|---------|--------------|
| `ModuleNotFoundError: ai_game_agent` in subprocess | missing `PYTHONPATH` in `run_cli` |
| exit 1, `No module named 'PIL'` | subprocess venv lacks the `capture`/dev extras — run under `uv run` |
| stdout empty, error on stderr | correct behavior for error tests; wrong for success tests |
| PNG size off (e.g., 32×16 instead of 16×8) | region/scale pipeline bug in `Capture.grab()` |
| 4 files in `rec/` instead of 3 | rotation boundary off-by-one in `Recorder` |
| Count varies near the boundary on re-run | two frames shared a same-microsecond timestamp and `save_frame` appended `-1`, `-2` collision suffixes — check the dir for `-N` files before assuming an off-by-one |
| exit 1, `error: unknown capture backend` | `--backend` value typo or a backend name not in `create_backend` — expected for test 10 |

## 8. Out of scope (deliberately)

* Real-`mss` capture E2E — requires a display; violates the headless/CI rule.
  The `mss` path is covered by the fake-screen integration tests already present.
* Performance/wall-clock assertions on `fps` values — non-deterministic; only
  the `frames=N` count and format are asserted.
* The `observe` double-configuration of recording (see note in §5.2) — a
  cleanup item, not an E2E concern.
* Windows-specific shell behaviors — tests use direct process spawning
  (`subprocess.run` with an argument list, no shell), so they are portable.
