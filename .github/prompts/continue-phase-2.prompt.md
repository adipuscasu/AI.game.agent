---
description: "Continue the Phase 2 (basic computer vision) implementation: pick the next outstanding task from docs/phase-2-implementation-plan.md, implement it TDD (one task only), then update that plan's status. Use when the user says to continue/resume Phase 2 or implement what's left of the phase-2 plan."
name: "Continue Phase 2"
argument-hint: "Optional: a specific step, e.g. \"step 5 (ui.py)\" or \"resolve the §13 asset decision\""
agent: "agent"
---

Continue implementing the Phase 2 plan in [phase-2-implementation-plan.md](../../docs/phase-2-implementation-plan.md). Work through the plan's build order (§12) **one task at a time**, and follow the engineering rules in [AGENTS.md](../../AGENTS.md) (especially TDD, error taxonomy, and the modular perception architecture).

If an argument was provided, work that specific item. Otherwise choose the **next** outstanding task using §14's status table (❌ = not started, ⚠️ = partial), in build-order priority (resolve any blocking §13 open question first).

For that single task, do exactly the following, in order:

1. **Scope it.** Restate in one sentence which task you're doing and why it's next. Do not start work on other tasks.
2. **Red.** Write the failing test(s) first, in the file named in §7.1 / §7.2 for this task (e.g. `tests/test_ui_zones.py`, `tests/test_objects.py`, `tests/test_ocr.py`, `tests/e2e/...`). Run them and confirm they fail for the right reason.
3. **Green.** Write the minimum code to make them pass, in the correct layer (perception vs config vs CLI). Respect:
   - interface-first (protocols in `perception/base.py`), dependency injection into `Perception`,
   - lazy heavy imports (`cv2`, `pytesseract` inside functions),
   - `extra="forbid"` config validation,
   - never fatal per-frame detector errors — record into `detector_errors`.
4. **Refactor.** Clean up while keeping tests green (remove the duplicated validation noted in §14 "Known cleanup" if you touch `template.py`).
5. **Verify.** Run `uv run pytest` and `uv run ruff check src tests`; both must be clean. If the task is the `analyze` CLI (step 8), also confirm the §14 checklist command(s) behave per the Phase 1 error contract (stdout empty, `error:` on stderr, exit 1).
6. **Update the plan doc.** In `docs/phase-2-implementation-plan.md`, reflect the new reality:
   - flip this task's row in the §14 status table to ✅ Done with its file paths,
   - remove it from "Next up" / "Remaining work, item by item" (re-number the rest),
   - move any now-satisfied §7.3 done-criteria / verification-checklist boxes to checked,
   - note anything still open or newly discovered.
7. **Report.** Summarize in a few lines: what you implemented, the test evidence (red→green), what §14 now shows as next, and — if this was the last outstanding item — confirm the phase is done against §7.3.

Then **stop** after this one task. Do not batch the next task unless the user asks.
