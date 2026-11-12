# Phase 7 Engineering Hardening — Companion Workstream

**Status:** proposed work packages; none of the work below is claimed as implemented.
**Relationship:** companion to `phase-7-implementation-plan.md` (authoritative milestone definition) and `phase-7-plan.md` (scenario execution plan). This document adds an engineering-hardening workstream; it does not redefine the first ASSISTED milestone, change its safety contract, or authorize unattended gameplay.
**Target base:** `develop`
**Prepared:** 2026-11-07

---

## 1. Purpose and boundaries

The Phase 7 gameplay objective remains one bounded, evidence-backed WoW scenario, with the first control-capable run in ASSISTED mode and explicit human approval for every AI-seam proposal. The work packages below address maintainability, diagnosability, platform coverage, evaluation-data quality, and resource safety around that objective.

Treat this as a **parallel, separately reviewable hardening track**, not a reason to delay observation-only work or silently expand the gameplay milestone. Each item should be implemented in its own PR unless repository evidence demonstrates that two changes are inseparable. Do not combine these changes with scenario logic, perception tuning, approval semantics, or live-game control in one broad PR.

The plan is evidence-led: inspect the current tree before implementation, preserve existing behavior unless a defect is demonstrated, and record actual commands/results. A planned test is not evidence that it passed.

### Non-goals

- No broad architectural rewrite or speculative abstraction.
- No behavior change hidden inside the CLI module split.
- No weakening of fail-closed approval, OBSERVE_ONLY, emergency-stop, or action validation.
- No live game input in CI; platform coverage must remain headless and mocked.
- No uploading screenshots or session evidence to external services by default.
- No claim that a cleanup or lifecycle refactor is complete until exceptional and cancellation paths are tested.

---

## 2. Work packages

### EH1 — Separate CLI parsing from application orchestration

**Suggested branch:** `refactor/cli-module-separation`

**Problem to validate:** `src/ai_game_agent/__main__.py` currently combines command parsing, subcommand wiring, configuration, collaborator construction, and run orchestration. Confirm the current file and call graph before choosing module boundaries.

**Scope**
- Move argument/parser construction into a focused CLI module.
- Move command-specific wiring into small subcommand modules where this reduces coupling.
- Keep application orchestration and dependency construction explicit; do not create a generic plugin/command framework.
- Preserve the existing console entry point and public CLI behavior.

**Compatibility requirements**
- Existing command names, flags, defaults, help text, exit codes, output, and error behavior remain unchanged unless a separately documented bug is found.
- Keep imports side-effect-free: importing CLI modules must not open devices, initialize a backend, connect to Ollama, or start the agent.
- Preserve dependency injection seams used by tests.

**Acceptance evidence**
- Add or retain parser tests for every subcommand and important flag/default.
- Add command-wiring tests using mocks; tests must not dispatch real input.
- Compare representative `--help` output and command behavior before/after.
- Run the full test suite, Ruff, and `git diff --check`.
- The PR description lists any intentional behavior changes separately; ideally there are none.

**Exit gate:** the entry point delegates to the new modules and the CLI contract is demonstrably unchanged.

### EH2 — Make agent-loop failures observable without compromising safe shutdown

**Suggested branch:** `fix/agent-error-observability`

**Problem to validate:** inspect exception handling in `src/ai_game_agent/agent/loop.py` and its collaborators. Identify where operational failures are classified, where unexpected programming errors may lose traceback/context, and whether cleanup failures can obscure the original failure.

**Scope**
- Distinguish expected operational failures (for example, capture/provider/backend failures already represented by project contracts) from unexpected exceptions.
- Preserve useful exception type, message, traceback, and causal context in logs/events or the returned report, using the project's existing observability conventions.
- Keep safe-pause/stop behavior and held-input release guarantees intact.
- Ensure cleanup failures are observable but do not replace or mask the primary exception; when both fail, retain both pieces of evidence.
- Avoid catching broad exceptions merely to continue the loop. Unexpected errors should not be silently converted into ordinary recoverable conditions.

**Acceptance evidence**
- Tests cover a representative expected operational failure and an unexpected programming error.
- Tests assert the original exception/traceback context remains diagnosable.
- Tests cover cleanup succeeding after a primary failure and cleanup itself failing; neither case may authorize later dispatch.
- Cancellation and F12/emergency-stop paths still release held inputs and do not dispatch subsequent actions.
- No sensitive credentials or unrelated screen contents are added to logs.

**Exit gate:** failures are diagnosable while every failure path remains fail-safe; the original cause is not lost when cleanup also fails.

### EH3 — Add Windows CI coverage alongside Linux

**Suggested branch:** `ci/windows-platform-coverage`

**Goal:** catch platform-specific import, packaging, path, console, and mocked-backend regressions on the project's actual Windows target without requiring a physical desktop or live game.

**Scope**
- Add a Windows runner job alongside the existing Linux job.
- Install the project and development dependencies using the documented `uv` workflow.
- Run import/packaging smoke checks, the headless test suite, Ruff, and diff/format checks as appropriate to the existing CI design.
- Use mocked capture/input backends. Do not launch WoW, touch real input devices, require an interactive desktop, or contact a live Ollama service.
- Mark genuinely interactive-console tests with an explicit, documented platform/runner condition rather than treating an environment limitation as a product pass.

**Acceptance evidence**
- Both Linux and Windows jobs execute on the same commit.
- Windows validates package installation/import and the supported CLI entry point.
- The CI summary reports skips distinctly from passes and explains any console-dependent skips.
- No secrets or user-specific local paths are required.
- Existing Linux coverage is not weakened to make the matrix green.

**Exit gate:** the Windows job is green on a clean hosted runner using only headless/mocked dependencies, and the Linux job remains green.

### EH4 — Version and structure the perception evaluation dataset

**Suggested branch:** `test/perception-evaluation-dataset`

**Goal:** make the labelled frame set reproducible, reviewable, and resistant to accidental training/held-out contamination.

**Scope**
- Define a small versioned dataset manifest format before adding tooling.
- Each sample should have a stable sample ID, relative image path or approved external/local reference, label(s), annotation version, split (`train`, `validation`, or `held_out`), and capture metadata.
- Capture metadata should include at least game/client version when known, screen resolution, window mode, UI scale, language, monitor/region, capture scale, and relevant UI/layout configuration.
- Record detector/model/template version and threshold/config version in evaluation output so metrics can be traced to the exact inputs.
- Keep raw screenshots local by default; document privacy review/redaction and repository-size rules before committing images.
- Prevent held-out samples from being used for threshold tuning. A split change must be explicit and versioned.
- Include negative, positive, borderline, and degraded/ambiguous cases, plus explicit detector-failure/not-run cases where representable.

**Acceptance evidence**
- A schema/manifest validator rejects missing required metadata, invalid labels/splits, duplicate IDs, and missing sample references.
- Tests prove the evaluator reports train/tuning metrics separately from held-out metrics and does not silently tune on held-out data.
- Evaluation output includes counts and confusion-matrix values, not percentages alone; unknown/detector-failure counts remain visible.
- A small synthetic fixture dataset validates the format without requiring game screenshots.
- The initial real dataset meets the Phase 7 baseline target of at least 50 manually labelled representative frames, with a held-out subset, before claiming calibrated performance.

**Exit gate:** another developer can identify the exact dataset version, capture conditions, labels, split, and detector configuration behind every reported metric.

### EH5 — Make recording and long-lived resource lifecycles explicit

**Suggested branch:** `fix/resource-lifecycle-management`

**Problem to validate:** trace the complete lifecycle of frame recording, event recording, capture/backend handles, and other long-lived resources from construction through normal return, exception, cancellation, and partial initialization. Do not assume a leak or failure from a local code fragment; map ownership and all exit paths first.

**Scope**
- Make ownership and closure responsibilities explicit for recorders, files, capture resources, and backend resources.
- Prefer context-manager/lifecycle APIs where they fit existing conventions; avoid introducing a framework for short-lived objects.
- Ensure buffered records are flushed/closed on normal completion and best-effort cleanup occurs on exceptions and cancellation.
- Preserve the primary failure if flush/close also fails; surface the cleanup failure without hiding the original cause.
- Define behavior for partial initialization (for example, one resource opens and a later constructor step fails).
- Do not make recording failure silently look like a complete, valid evidence session.

**Acceptance evidence**
- Tests cover normal shutdown, primary exception, cancellation/emergency stop, partial initialization, flush failure, and close failure.
- Tests assert resources are closed exactly once (or safely tolerate repeated close if that is the documented contract).
- A failed or incomplete recording is visibly distinguishable from a successfully finalized recording.
- Lifecycle changes do not alter action dispatch, approval, or event ordering.
- Full tests, Ruff, and `git diff --check` pass.

**Exit gate:** every owned resource has a documented owner and deterministic terminal path; incomplete evidence cannot be mistaken for a complete session.

---

## 3. Recommended implementation order

Keep each item in a separate PR and re-check dependencies against the live tree before starting:

1. **EH1 — CLI module separation.** Establishes clearer seams for later CLI and lifecycle work; behavior-preserving only.
2. **EH2 — Agent error observability.** Improves the ability to diagnose subsequent integration and evaluation failures.
3. **EH3 — Windows CI.** Add early enough to catch platform issues in the remaining work; it can proceed in parallel with EH1/EH2.
4. **EH4 — Versioned evaluation dataset.** Define the manifest and validator before collecting/tuning the full scenario dataset.
5. **EH5 — Resource lifecycle.** Trace ownership first; implement only the concrete lifecycle gaps found. This may be scheduled earlier if that trace reveals a current evidence-loss or resource-safety defect.

The order is a recommendation, not a dependency claim. A confirmed safety or evidence-loss defect takes priority and should be isolated in its own narrowly scoped fix.

## 4. Relationship to the Phase 7 gameplay work

- EH1–EH5 do not change the selected WoW scenario, its action allow-list, the AI-seam provenance rule, or the human approval boundary.
- The Phase 7 observer baseline and labelled frame collection can proceed with existing mechanisms while EH4 defines the durable dataset format; once the format is accepted, migrate/record the baseline evidence into it without changing labels to improve metrics.
- EH2 and EH5 strengthen failure reporting and evidence preservation, but neither permits continuing after an unknown/unsafe result.
- EH3 is CI-only and must not turn live desktop/game testing into an automated workflow.
- The first ASSISTED milestone remains governed by the acceptance criteria in the two existing Phase 7 plans. This companion workstream is tracked separately; completion of EH1–EH5 is not a substitute for a real, evidence-backed scenario run, and an unfinished non-blocking hardening item must not be reported as completed.

## 5. Review and reporting rules

For every hardening PR:

1. State the exact base/head commit and files inspected.
2. Separate confirmed defects from optional design improvements.
3. Show focused regression tests before reporting the full suite.
4. Report exact commands, pass/fail/skip counts, and platform/runner.
5. Do not claim a test ran if it was only added or inspected.
6. Do not push a final implementation commit on behalf of the repository owner; leave implementation branches for owner review and push/merge decisions.
7. Keep the PR scoped to one EH work package and list any behavior changes explicitly.

**Completion rule:** mark a work package complete only after its code, tests, and CI evidence are present on the reviewed branch. A plan entry, local agent report, or green unrelated workflow is not sufficient evidence.
