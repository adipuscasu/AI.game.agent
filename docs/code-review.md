# Code Review — Phase 1 (Screen Observation)

**Date:** 2026-09-27
**Scope:** Uncommitted changes implementing Phase 1 of the high-level development plan:
`README.md`, `pyproject.toml`, `uv.lock`, `src/ai_game_agent/capture.py`,
`src/ai_game_agent/__main__.py`, and the new test files
(`test_mss_backend.py`, `test_recorder.py`, `test_fps_meter.py`, `test_save_frame.py`, `test_cli.py`).

**Verdict:** Solid, well-structured implementation. Tests are headless and deterministic,
the backend abstraction is clean, and the layering respects `AGENTS.md`. There is
**one real bug** (a documented CLI feature that crashes at runtime) and a handful of
minor issues.

---

## 🔴 Bug

### 1. `--region` CLI option crashes at runtime (documented in README)

`README.md` and the `__main__.py` docstring both advertise:

```powershell
ai-game-agent capture --backend mss --region 0,0,1920,1080
```

But `_build_capture()` passes the raw string straight into the config:

```python
cfg = CaptureConfig(
    backend=args.backend,
    region=args.region,   # <- a string like "0,0,1920,1080"
    ...
)
```

In `config.py`, the `_coerce_region` validator only handles `None`, a `_Region`
instance, a dict, and objects with `.x/.width` attributes. A **string matches none
of these**, so it is returned as-is and pydantic v2 (which does not perform
string→model coercion) raises a `ValidationError`. There is no test covering
`--region`, so this gap went uncaught.

**Fix:** parse `"x,y,w,h"` into a `Region` in `_build_capture()` (or add a string
branch to `_coerce_region`), and add a regression test:

```python
def _parse_region(value: str | None) -> Region | None:
    if value is None:
        return None
    x, y, w, h = (int(v) for v in value.split(","))
    return Region(x, y, w, h)
```

**Test to add (red first):**

```python
def test_cli_capture_with_region(tmp_path):
    out = tmp_path / "shots"
    rc = main(["capture", "--backend", "mock", "--region", "10,20,32,16", "--out", str(out)])
    assert rc == 0
    from PIL import Image
    assert Image.open(next(out.glob("*.png"))).size == (32, 16)
```

---

## 🟡 Minor issues

### 2. `Recorder._rotate` uses lexicographic sort — breaks after >9 same-microsecond collisions

`save_frame()` appends `-1`, `-2`, … suffixes on timestamp collision. Lexicographically,
`frame_X-10.png` sorts *before* `frame_X-2.png`, so rotation can delete a newer file
before an older one once a timestamp collides more than 9 times.

**Risk:** Low (requires >10 frames captured within the same microsecond).

**Suggested fix:** sort by `Path.stat().st_mtime_ns` (or parse the numeric suffix)
instead of plain lexicographic order.

### 3. Untyped `clock` parameter in `Capture.__init__`

```python
def __init__(self, config: CaptureConfig, backend: CaptureBackend | None = None, clock=None):
```

`clock` has no annotation while `AGENTS.md` asks for type hints. It should be:

```python
clock: Callable[[float], float] | None = None
```

### 4. Dead code in `MockBackend`

`self._frame_index` is incremented in `grab()` but never read. Either remove it,
or use it to vary synthetic frames (e.g., shift the pixel color per frame) so
replay fixtures are more meaningful.

### 5. Redundant import in `MockBackend.grab()`

The inner `from datetime import datetime` shadows the module-level import at the
top of `capture.py`. Delete the inner line.

### 6. `fps=0` workaround is a slight design smell

`CaptureConfig` enforces `fps > 0` (pydantic `Field(gt=0)`), so the CLI does
`fps=max(args.fps, 1)` plus a `_noop_clock` to disable pacing. It works, but
reading `cap.config.fps` back gives `1`, not the intended "no pacing" value —
the two sources of truth disagree.

**Suggested fix (preferred):** relax the model to `fps: int = Field(default=30, ge=0)`
and treat `0` as "no pacing" in `wait_next_frame()` (which it already does).
This removes the `_noop_clock`, the `max(args.fps, 1)` clamp, and the divergence.

---

## ✅ What's good

- **TDD followed** — 22 new tests across five test files; full suite is 38 passing
  with `ruff check src tests` clean.
- **Clean dependency injection** — `mss_factory`, `clock`, and `backend` are all
  injectable, so the test suite runs headless and stays deterministic
  (AGENTS.md requirement: "Tests must be deterministic, independent, and
  runnable in CI without a desktop session or a live game").
- **`MssBackend`** correctly uses `bytes(shot.rgb)` (stride-stripped RGB) and
  validates frame size against `width * height * 3` — good defensive choice.
- **Error containment** — `Capture.grab()` wraps all backend exceptions into
  `CaptureError` (the PERCEPTION_ERROR class per AGENTS.md) while re-raising
  `CaptureError` unchanged; `save_frame()` uses collision-safe timestamped naming.
- **Config** — `config/default.yaml` already documents the schema; `Pillow` and
  `mss` are correctly gated behind the `capture`/`dev` extras so the core stays
  dependency-light (AGENTS.md: "Keep dependencies minimal").
- **Backend swap is trivial** — `create_backend()` is the single factory; adding
  DXGI or a replay backend later requires no changes to `Capture`, `Recorder`,
  or the CLI.

---

## Suggested follow-up work (in priority order)

1. Fix the `--region` crash with a regression test (TDD: red → green).
2. Annotate `Capture.clock` and remove the dead `_frame_index` / redundant import.
3. Decide on the `fps=0` semantics and consolidate (model-level `ge=0` preferred).
4. Harden `Recorder._rotate` against numeric-suffix ordering.
