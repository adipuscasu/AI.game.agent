# Code Review — Phase 2 Uncommitted Changes

> Reviewed on **2026-10-03** (branch `feature/phase-2`, HEAD `0c03ec1`).
> Scope: all uncommitted files in the working tree:
>
> | File | Status |
> |---|---|
> | `tests/test_objects.py` | modified |
> | `tests/test_pipeline.py` | modified |
> | `docs/phase2-to-do.md` | new (untracked) |
> | `perception/pipeline.py` | new (untracked, repo root) |
> | `perception/objects.py` | new (untracked, repo root) |
>
> All claims below were verified against the real package in
> `src/ai_game_agent/` (config, capture, `perception/base.py`,
> `perception/observation.py`, `perception/pipeline.py`,
> `perception/objects.py`) and with `py_compile` syntax checks.

## Summary

| # | Severity | File | Finding |
|---|----------|------|---------|
| 1 | 🔴 Critical | `tests/test_pipeline.py` | Rewrite deletes 403 lines of passing tests; replacement test fails at import time |
| 2 | 🔴 Critical | `tests/test_objects.py` | Tests an API that does not exist (fixture `KeyError`, `detect()` vs `run()`) |
| 3 | 🔴 Critical | `perception/` (root) | Stray duplicate drafts in the wrong location; `objects.py` has a SyntaxError |
| 4 | 🟠 High | `src/ai_game_agent/perception/objects.py` | Committed implementation is broken at runtime (undefined names, wrong `ObjectHit` fields) |
| 5 | 🟡 Low | `docs/phase2-to-do.md` | Suite-state description is stale relative to the working tree |
| 6 | 🟡 Low | both test files | LF→CRLF line-ending noise |

**Bottom line:** the uncommitted test rewrites target a different, imagined
API than the one implemented in `src/ai_game_agent/`. The net effect is a
loss of passing coverage and a suite that fails at collection. Recommended
action: restore `tests/test_pipeline.py`, delete the stray root `perception/`
directory, fix the real `objects.py` to the `ObjectDetector` protocol, then
rewrite `tests/test_objects.py` against it (TDD red→green per AGENTS.md).

---

## 🔴 1. `tests/test_pipeline.py` — replacement test cannot run

The diff removes 403 lines of working tests (composition, per-detector error
isolation, disabled-detector skipping, determinism — all passing per
`docs/phase2-to-do.md`) and replaces them with a single test that fails at
collection:

- **Nonexistent imports** — `from ai_game_agent.core.frame import Frame` and
  `from ai_game_agent.core.config import PerceptionConfig`. There is no
  `core` package anywhere in `src/ai_game_agent/` → `ModuleNotFoundError`
  before any test runs.
- **Patching names that don't exist** — the test patches
  `ai_game_agent.perception.pipeline.TemplateDetector`, `UIZoneDetector`,
  and `OcrDetector`. The real `Perception` (see
  `src/ai_game_agent/perception/pipeline.py`) has no such attributes;
  detectors are injected as keyword dependencies `template_matcher`,
  `ui_detector`, `ocr_engine`, `object_detector`.
- **Asserting a nonexistent attribute** — `observation.hits`. The real
  `Observation` (`perception/observation.py`) exposes `templates`,
  `text_regions`, `objects`, `detector_errors`; there is no `hits` field.
- **Asserting a nonexistent type** — `"UIZoneHit" in types_found`. No such
  class exists in the codebase.
- **Wrong `Frame` constructor** — `Frame(data=..., width=..., height=...)`.
  The real `Frame` (`capture.py`) is a dataclass requiring `width`,
  `height`, `pixels: bytes`, `captured_at`, `source`.
- **Test hygiene** — prints success banners to stdout.

**Recommendation:** `git checkout -- tests/test_pipeline.py` to restore the
passing suite.

## 🔴 2. `tests/test_objects.py` — tests a different API than the one that exists

- **Fixture crashes at setup** — `mock_config` passes `object_colors` entries
  with HSV keys (`h_min`, `h_max`, `s_min`, …) and no `rgb`. The real
  `PerceptionConfig.__init__` (`config.py`) does `int(c["rgb"][0])` for every
  color → `KeyError: 'rgb'`, so both tests using the fixture error before
  the test body runs.
- **Wrong method name** — tests call `detector.detect(frame)`; the real
  `ColorBlobsDetector` only defines `run()`, and the `ObjectDetector`
  protocol in `perception/base.py` requires `match(frame) -> list[ObjectHit]`.
- **Wrong frame shape** — `mock_frame` passes a PIL `Image` as `pixels`;
  real `Frame.pixels` is a row-major RGB `bytes` buffer.
- **Wrong config shape** — the HSV `h_min/h_max/…` schema matches neither
  `PerceptionConfig` (which stores `name`/`rgb`/`tolerance`) nor what the
  real detector reads (`config.get("object_detection", {}).get("color_blobs", {})`
  — and `PerceptionConfig` is a frozen object with no `.get()`).

## 🔴 3. Stray `perception/` directory at repo root

Two untracked files (`perception/pipeline.py`, `perception/objects.py`) sit
at the repository root — outside the `ai_game_agent` package (no
`__init__.py`; the real package is `src/ai_game_agent/perception/`). They
look like AI-generated drafts saved in the wrong directory:

- **`perception/objects.py` has a SyntaxError** — the leading `"""` on line 1
  is never closed before the next `"""`, so the imports are swallowed into a
  string literal:
  `SyntaxError: unterminated string literal (detected at line 12)`
  (verified with `python -m py_compile`).
- `perception/objects.py` references `BBox` without importing it and defines
  `ColorBlobsDetector` twice (a `TYPE_CHECKING` placeholder plus the real
  class).
- `perception/pipeline.py` uses `self.config.objects.enabled`; the real
  `PerceptionConfig` exposes `objects_enabled` (bool), not an `.objects`
  sub-object.
- Both files log via `print()` instead of the logging module, violating the
  repo's logging conventions.

**Recommendation:** delete the directory, or — if the code is intended —
merge it into `src/ai_game_agent/perception/` after fixing the issues above.

## 🟠 4. Committed `src/ai_game_agent/perception/objects.py` is broken at runtime

This is the module the new tests target, and it cannot work as written:

- `super().__init__(config)` — `ObjectDetector` is a `typing.Protocol`;
  `object.__init__()` takes no arguments → `TypeError` at construction.
- `DetectedObject` and `BBox` are referenced but never defined/imported →
  `NameError` at runtime.
- `self.config.get("object_detection", {})` — `PerceptionConfig` is a frozen
  wrapper with properties, no `.get()` → `AttributeError`.
- `ObjectHit(object_type=..., bbox=..., confidence=..., detection_source=...)`
  — the real `ObjectHit` is a dataclass with fields
  `(kind: str, bbox: BBox, confidence: float)` → `TypeError`.
- `frame.image` — real `Frame` exposes `pixels` (bytes) and `to_image()`;
  no `image` attribute.
- `print()` used for all logging.

Even if the tests were fixed to call `run()`, the detector would crash.
**Fix the implementation to the `ObjectDetector` protocol first**
(`match(frame) -> list[ObjectHit]`, `ObjectHit(kind=..., bbox=BBox(...),
confidence=...)`, no `print`), then write tests against it.

## 🟡 5. `docs/phase2-to-do.md` — mostly accurate, slightly stale

The document correctly describes the broken suite state at commit `0c03ec1`
and is a useful companion to the implementation plan. Minor discrepancy: it
reports `tests/test_objects.py` as "1 error, 1 failure", but the
working-tree version of that file (post-modification) fails all three tests
(fixture `KeyError` + `detect()` `AttributeError`).

## 🟡 6. Line-ending noise

Both modified test files emit `LF will be replaced by CRLF` warnings. The
housekeeping note in `docs/phase2-to-do.md` (set `core.autocrlf=input` or add
a `.gitattributes` forcing LF) applies.

---

## Recommended action plan

1. `git checkout -- tests/test_pipeline.py` — restore the passing suite.
2. Delete the stray root `perception/` directory.
3. Fix `src/ai_game_agent/perception/objects.py` to conform to the
   `ObjectDetector` protocol in `perception/base.py`.
4. Rewrite `tests/test_objects.py` against the real API (TDD: red → green).
5. Run `python -m pytest -q` and confirm the suite is green.
