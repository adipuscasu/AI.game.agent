---
description: "Close out a phase implementation plan: verify every outstanding item is done, run the full verification checklist (pytest, ruff, CLI contract), confirm the Definition of Done, then mark the phase complete in the plan doc. Use when the user says finish/close/complete the phase, or after /continue-phase-2 has worked through all its tasks."
name: "Finish Phase"
argument-hint: "Optional: a plan path (default docs/phase-2-implementation-plan.md) or phase name"
agent: "agent"
---

Close out the phase described in [phase-2-implementation-plan.md](../../docs/phase-2-implementation-plan.md).
(If an argument names a different plan, target that file instead.)

1. **Audit the status.** Read the plan's status table (e.g. §14) and "Remaining work" list. Confirm every build-order step is ✅ Done. If any is still ❌/⚠️, **stop** and tell the user exactly which — do not fake completion.
2. **Run the verification checklist.** Execute every command in the plan's Definition-of-Done / verification-checklist section (e.g. §7.3 / §14):
   - `uv run pytest` (full suite, headless)
   - `uv run ruff check src tests`
   - each `analyze` / CLI contract command listed, asserting the exit code plus stdout/stderr shape per the Phase 1 error contract (stdout empty, `error:` on stderr, exit 1).
   Report the **actual output** of each line, not a paraphrase.
3. **Confirm the Definition of Done.** Walk each DoD bullet; state which are satisfied by the evidence above and which are not.
4. **Update the plan doc.** Only if steps 2–3 are fully green:
   - set the header **Status** line to "Complete" with today's date and a one-line summary,
   - check the remaining unchecked verification-checklist boxes,
   - collapse "Remaining work, item by item" into a short "Resolved" note,
   - add a brief "Phase complete" entry recording the pytest/ruff/CLI evidence.
   If anything failed, leave the status as In progress, list the exact failing commands, and say what is blocking.
5. **Report.** A short final summary: phase state, per-checklist green/red evidence, and the next recommended action (e.g. "start Phase 3 — input automation").

Do **not** implement new features in this step — only verify, finalize, and document.
