# EH5 — Independent Verification and Completion Audit (WORK PLAN)

> Resumable work plan. Status markers: `[ ]` not started · `[~]` in progress · `[x]` done.
> Last updated: 2026-11-11 (resume session, step 2 complete, F1–F6 audit underway).

## Constraints

- Work on existing branch `fix/resource-lifecycle-management`. Do **not** create another branch.
- Do **not** amend, reset, squash, push, or merge anything during this task.
- Do not rely on previous agent reports or commit messages as evidence.
- Do not modify skip conditions or tests simply to restore a previous test count.
- **STOP before amending, pushing, or opening a PR. Wait for explicit user approval.**

## Step 1 — Establish actual repository state

Record: `git branch --show-current`, `git status --short`, `git log --oneline --decorate -5`,
`git rev-list --left-right --count origin/develop...HEAD`, `git diff origin/develop...HEAD --stat`.
Inspect the complete diff of both commits.

- [x] Run git state commands
- [x] Inspect complete diff of `155de14` (EH5 implementation) — cli.py (`_act`, `_run`), windows.py (open rollback), 4 new test files sections
- [x] Inspect complete diff of `34e8bcc` (F3 hardening) — windows.py strict rollback (note + retain + `raise primary from release_exc`), close() retry-safe rework

**Findings (step 1):**
- Branch: `fix/resource-lifecycle-management`
- `git status --short`: empty (clean)
- `git log --oneline --decorate -5`:
  ```
  34e8bcc (HEAD -> fix/resource-lifecycle-management, origin/develop, origin/HEAD, develop) work in progress
  155de14 EH5: complete resource-lifecycle cleanup for F1-F6
  e8fd485 (origin/fix/resource-lifecycle-management) work in progress
  c847e92 Merge pull request #11 from adipuscasu/fix/agent-error-observability
  a2ef1e5 refactor(tests): split test_ai.py into category modules
  ```
- `git rev-list --left-right --count origin/develop...HEAD` → `0 0` (HEAD == origin/develop)
- `git diff origin/develop...HEAD --stat` → empty (no diff vs origin/develop)
- NOTE: the EH5 commits are ALREADY at develop / origin/develop's tip.
  `origin/fix/resource-lifecycle-management` lags HEAD by 2 commits (at e8fd485).
  Local `develop` also equals HEAD (34e8bcc).

## Step 2 — Investigate test-count discrepancy (897 passed/2 skipped vs 895 passed/4 skipped)

- [x] Run `uv run pytest -q -rs`, record exact skip reasons
- [x] Compare vs prior "OpenClaw" report (897 passed, 2 skipped — presumably pynput-absent env)
- [x] Classify each extra skip: legitimate env-dependent vs regression
- [x] Document conclusion

**Hypothesis to verify:** the 4 local skips are
`test_actions_bare.py:130` (cross-OS venv rule — Windows-box specific),
`test_actions_base.py:86` (pynput present), and the two CONIN$ skips
(`test_assisted_approval.py:2423,2439` — live Windows console). In a pynput-absent
environment the first two would run → 897 passed + 2 skipped = same 899 total.
897+2 = 899 = 895+4 ✓ (math consistent; verify conditions).

**CONCLUSION (verified 2026-11-11): `uv run pytest -q -rs` → 895 passed, 4 skipped.**
All 4 skips are legitimate, environment-conditional `skipif`s (inspected in source):
1. `test_actions_bare.py:130` — `skipif(sys.platform == "win32")`: pynput IS
   installed in this box's venv, so the bare-venv contract is inverted here;
   runs on Linux/CI.
2. `test_actions_base.py:86` — `skip` when `importlib.util.find_spec("pynput")`
   is not None: pynput present on this box.
3+4. `test_assisted_approval.py:2423,2439` — `skipif(sys.platform != "win32"
   or not sys.stdin.isatty())`: need a live CONIN$ console; absent under pytest.

In a pynput-absent environment (the OpenClaw report's env) items 1–2 RUN, giving
897 passed + 2 skipped = 899 total = 895 + 4. ✓ DISCREPANCY EXPLAINED — no
missing/regressed tests. Nothing to fix.

## Step 3 — Verify F1–F6 individually

### F1 — Timestamped filename `run-\d{8}-\d{6}\.jsonl`
- [x] Inspect the filename test, verify complete regex pattern — `test_run_record_filename_matches_timestamped_convention` (test_run_cli.py:1070): clock frozen via `time.strftime` monkeypatch, asserts `name == "run-20261111-120000.jsonl"` AND `re.fullmatch(r"run-\d{8}-\d{6}\.jsonl", name)` ✓
- [x] Run targeted test explicitly, record result — PASSED (in 895-passed full run)

### F2 — ExitStack resource ownership (cli.py)
- [x] Partial construction failures release previously acquired resources — `test_run_record_closes_when_agentloop_ctor_raises` (test_run_cli.py:663): `AgentLoop.__init__` raises → record stream spy `close_calls >= 1` and `closed is True` ✓
- [x] Ownership transferred correctly — `stack.enter_context(rec_path.open(...))` (cli.py:1107) inside `_run`; ExitStack closes it in the finally (cli.py:1172 `stack.close()`) ✓
- [x] Cleanup happens exactly once — single `stack.close()` in the finally; tested in F5 scenarios 1–5 ✓
- [x] Existing construction behavior remains compatible — all 895 suite tests pass with the constructor inside the try block ✓

### F3 — Windows backend cleanup (open/close/release_all)
- [x] Original open exceptions remain primary — `test_open_rollback_failure_preserves_primary_and_attaches_note` (test_windows_backend.py:498)
- [x] Rollback failures attached as notes — same test (asserts `__notes__`)
- [x] Failed listener stops remain retryable — `test_open_rollback_failure_retains_listener_for_close_retry` (:521) + `test_close_retries_failed_listener_release_and_no_double_release` (:545)
- [x] Successful listener stops are not repeated — :545 (third close, `stop_attempts == 2`)
- [x] Disarm, listener stop, controller release isolated — `test_close_releases_held_input_even_if_listener_stop_fails` (:566) + NEW `test_release_all_survives_controller_up_failure` + NEW `test_close_attempts_all_cleanup_when_stop_and_releases_fail`
- [x] Subsequent cleanup can complete — :521, :545 (close() retry lifecycle)
- [x] Coverage gap audit: disarm-failure = **not possible** (`_HotkeyDecider.disarm` is a pure attribute write, no I/O — documented); controller-release-failure = **gap, test added**; disarm+stop / stop+release / all-three = **gaps, composite test added**
- [x] Add missing tests RED → GREEN — 2 new tests (release-failure isolation + worst-case composite); GREEN first run (behavior already correct; implementation + test evidence documented per step 4)

### F4 — Flush failures
- [x] Flush failure without existing primary exception — NEW `test_f4_flush_failure_propagates_as_error` (+ CLI-contract test `test_f4_flush_failure_cli_contract`); GREEN first run (the recorder's flush failure propagates from `loop.run()` → `primary_exc` → raised after cleanup; implementation at cli.py:1132-1134 + 1170-1200)
- [x] Execution failure followed by flush failure (original stays primary) — NEW `test_f4_flush_failure_stays_primary_when_cleanup_fails`; GREEN first run (flush error stays primary by identity, backend-close failure noted via `add_note`)

### F5 — KeyboardInterrupt and cleanup in `_run`
- [x] Add regression tests: KeyboardInterrupt propagates; backend cleanup attempted;
      recording streams closed; exception identity + traceback preserved;
      cleanup failures observable; no double release — NEW `test_f5_ki_propagates_and_cleanup_completes` (KI by identity + traceback; backend close ×1, stream close ×1) and `test_f5_ki_plus_cleanup_failure_ki_stays_primary` (KI + both cleanup failures: KI stays primary, failures noted); GREEN first run
- [x] (existing) F5 scenarios 1–5 cleanup-failure semantics — test_run_cli.py:844-964

### F6 — `_act` exception precedence (PRIORITY)
- [x] Inspect entire `_act` implementation — cli.py:568-633: open() in protected region with `add_note` + re-raise by identity; execution in try/except/finally; close() exactly once; cleanup failures noted on primary / raised when none
- [x] Scenario 1: open() fails and close() fails — `test_act_s1_open_and_close_fail_open_stays_primary` + CLI-contract test
- [x] Scenario 2: execution fails and close() fails — `test_act_s2_execution_and_close_fail_execution_stays_primary` (identity + note)
- [x] Scenario 3: execution succeeds and close() fails — `test_act_s3_close_fails_after_success_close_propagates` + CLI-contract test
- [x] Scenario 4: execution and cleanup both succeed — `test_act_s4_success_closes_exactly_once_and_reports` (close ×1, exit 0, ActionLog JSON)
- [x] Original exceptions stay primary; identity + traceback preserved — S1/S2/KI tests assert `is` identity + `__traceback__`
- [x] Cleanup failures noted when primary exists; propagate when no primary — S1/S2 (notes) + S3 (propagate)
- [x] CLI error reporting + exit code 1 unchanged; KeyboardInterrupt never swallowed — `test_act_ki_during_open_never_swallowed` (KI by identity, close still attempted, failure noted)
- [x] **RED→GREEN evidence (2026-11-11):** 7 F6 tests GREEN on new code; with cli.py stashed to the committed (old) code: **4 failed / 3 passed** (S1×2, S2, KI fail — exactly the masking bugs; S3/S4 already correct). RED evidence recorded.

## Step 4 — Development discipline (for every genuine gap)

1. Write regression test → 2. Run → RED evidence → 3. Smallest fix →
4. GREEN evidence → 5. Run relevant existing tests → 6. Review diff.
Already-correct behavior: document implementation + test evidence instead.

## Step 5 — Final verification

- [x] `uv run pytest -q` → **909 passed, 4 skipped** (4 skips = the documented
      env-conditional `skipif`s: 2 pynput-present, 2 live-CONIN$; no regressions)
- [x] `uv run ruff check .` → **All checks passed** (3 RED-phase dead artifacts
      fixed: unused `captured` dict, unused `env` assignment, unused `WindowsBackend` import)
- [x] `uv run ruff format --check` on changed files → clean (cli.py + both test files reformatted)
- [x] `git diff --check` → clean
- [x] Branch diff review: ownership (ExitStack single-owner), exception precedence
      (F6 primary-identity + notes), retry safety (F3 listener retain/retry),
      double-release prevention (close ×1), no new regressions (909 green)

## Step 6 — Reporting and publication

- [ ] Actual branch + HEAD; exact commit history; exact test results + skip reasons
- [ ] F1–F6 individual status; RED→GREEN evidence; quality-gate results
- [ ] Complete changed-file list; outstanding issues
- **STOP. Wait for explicit user approval before publishing.**

## Progress log

- 2026-11-11: Plan created. Step 1 git-state commands done (see findings above).
- 2026-11-11: Step 0 (git investigation, per user direction) — see below.
- 2026-11-11 (resume): F6 `_act` rework + 7 scenario tests verified GREEN (7/7); RED
  evidence captured by stashing the new cli.py → 4 failed / 3 passed on the old code
  (S1×2, S2, KI = the masking bugs; S3/S4 pre-existing correct behavior).
- 2026-11-11 (resume): F3 audit complete — 8 existing tests cover the strict spec;
  2 coverage gaps found and tests added (`test_release_all_survives_controller_up_failure`,
  `test_close_attempts_all_cleanup_when_stop_and_releases_fail`); GREEN first run
  (behavior already correct). Disarm-failure documented as not possible (pure attribute
  write). test_windows_backend.py: 36 passed.
- 2026-11-11 (resume): F4 + F5-KI tests added (`test_f4_flush_failure_*` ×3,
  `test_f5_ki_*` ×2); GREEN first run (implementation already correct). test_run_cli.py: 33 passed.
- (next: step 5 full verification gates, then step 6 report — STOP before publication)

## Step 0 — Git history explanation (investigated 2026-11-11)

**Facts (verified):**
- `git log --oneline --graph --all`: `34e8bcc (HEAD, origin/develop, origin/HEAD, develop)` →
  `155de14` → `e8fd485 (origin/fix/resource-lifecycle-management)` → `c847e92` (merge PR #11) → …
  Linear history; **no merge commit for EH5** — the commits sit directly in develop's line.
- `git branch -a --contains 155de14` / `--contains 34e8bcc`: `develop`,
  `fix/resource-lifecycle-management`, `origin/develop`. ⇒ **Both EH5 commits ARE part of the
  published develop history.** The original assumption (EH5 isolated on the feature branch,
  unpublished) is incorrect.
- `gh pr list --state all`: no PR references `fix/resource-lifecycle-management` or the EH5
  commits (PRs: #5–#8, #10, #11 merged; #12 draft docs). ⇒ The commits were **pushed directly to
  `develop`**, not merged via PR.
- Local reflog: `develop@{0}: pull --ff … Fast-forward → 34e8bcc` (after clone at 338ff14 and
  earlier pulls); **no local "commit: EH5…" entry exists in this clone's reflog** — the two
  commits were authored/pushed from a different clone, then pulled here.
  `HEAD@{0}: merge develop: Fast-forward` shows this local fix-branch was ff-advanced to
  34e8bcc after checkout at `e8fd485` (the remote-tracking position).
- Commit metadata: `155de14` committed 2026-11-11 18:34:35 UTC (Adrian Puscasu);
  `34e8bcc` committed 2026-11-11 21:30:13 +0200 (same author).
- `origin/fix/resource-lifecycle-management` = `e8fd485` (pre-EH5); local branch = `34e8bcc`.

**Conclusion:** EH5 is published (on `origin/develop`). Nothing here should be re-pushed,
squashed, or "undone" without explicit user direction — that would be history rewriting on a
shared branch. Remaining work (F3/F4/F5/F6 audit + gap fixes) lands as NEW work on the local
branch only; publishing is a separate, user-authorized decision. Workplan file is untracked and
preserved.
