# Phase 7 Execution Plan — One Bounded WoW Scenario, ASSISTED First

**Status:** ready for implementation — *subordinate execution plan* to the
authoritative milestone definition in `phase-7-implementation-plan.md` (the
first control-capable milestone is ASSISTED mode with explicit human
approval)
**Branch:** `feature/phase-7-operational-autonomy`
**Base:** `develop` at `d4bc1ef192aed9c1849641541df190feb68d0e27`
**Date:** 2026-11-06
**Prerequisites:** Phases 1–6 and PR #7 merged

> **Relationship to the baseline.** `phase-7-implementation-plan.md` is the
> **authoritative** definition of the Phase 7 ASSISTED-first milestone. This file
> is its **subordinate execution plan**: it refines that document against the
> actual code and does not redefine the milestone. The first control-capable milestone is **ASSISTED
> mode with explicit human approval** — the same milestone the baseline
> defines. **Unattended crystal collection is a later, separate goal** (see
> §7.4) with its own prerequisites; it does not supersede the ASSISTED-first
> milestone. Where this file and the baseline disagree on the milestone, the
> baseline (ASSISTED-first) wins.

---

## 1. Goal (restated, testable)

Drive one small, bounded, repeatable World of Warcraft scenario from
observation → decision → action → verification. **The first control-capable
milestone is ASSISTED mode: the AI-proposed action the agent produces requires
an explicit human approval decision before it is dispatched** (the shipped
approval gate in `agent/loop.py` is scoped to AI-seam actions — see §5; the
deterministic path is proven separately to dispatch nothing under
`OBSERVE_ONLY`). It uses only
input primitives that already exist in the codebase (the loot action is a
mouse click on the marker plus a pickup key-press — both already
implemented; §2.1). The agent must:

1. Observe the game screen and extract the small set of scenario facts it needs.
2. Decide on the single action the scenario allows (deterministic policy
   first; local AI only as a fallback).
3. Execute that action through the existing executor — the scenario is the
   already-implemented loot sequence (click the marker + pickup key), so
   **no new input-control feature is added** (the pynput backend already
   implements it; §2.1).
4. Re-observe and classify the outcome as success, failure, or unknown using
   fresh visual evidence.
5. Stop safely when evidence is missing, stale, contradictory, or out of
   scenario.

"Done" for Phase 7 is a **reproducible, evidence-backed run** — not a green CLI
exit.

---

## 2. What the code already gives us (verified, with evidence)

These were read directly from the tree, not assumed. They bound the work and
prevent building things that already exist.

### 2.1 The input contract already supports the scenario (and more)

- `src/ai_game_agent/actions/action.py` defines a full input contract:
  `ActionKind.KEY_PRESS`, `KEY_HOLD`, `MOUSE_MOVE`, `MOUSE_CLICK`,
  `MOUSE_DOWN`, `MOUSE_UP`, `MOUSE_SCROLL`, and `DELAY`, with per-kind
  required/forbidden field validation.
- `src/ai_game_agent/actions/windows.py` (Phase 3, pynput backend) **fully
  implements** real mouse + keyboard I/O (`mouse_move`/`mouse_click`/
  `mouse_down`/`mouse_up`/`mouse_scroll`), and `mock.py` implements them for
  tests. So mouse control **already exists** — it is not missing.
- `src/ai_game_agent/games/wow/policy.py` already builds the full loot action
  sequence: `_looting_actions` emits
  **`MOUSE_CLICK` (center of the detected loot bounding box) → `DELAY` →
  `KEY_PRESS` (the loot pickup key)** (see `_click`, `_looting_actions`, and
  the `_key`/`_delay` builders). So the scenario the goal describes — pick up
  a glowing loot marker — is **already implemented**, including its mouse
  click.
- Consequence: **no new mouse-control feature is needed** for this scenario
  (the click is already there and the executor already runs it), and no
  executor work is required. The scenario is *not* keyboard-only — it is a
  click + key-press, both of which are fully present in the existing input
  contract and the pynput backend.

### 2.2 The two templates are the same in-game object

- `shots/wow_orb_template.png` (183×221) is a **byte-identical crop** of
  `shots/wow_crystal_template.png` (1061×1347): a `TM_CCOEFF_NORMED` match of
  orb-in-crystal scores **1.0000 at (242, 385) with 0.0 mean absolute pixel
  difference**.
- Implication for scenario design: the "loot" signal (a glowing collectible)
  can be driven by **one** calibrated template. Treating orb and crystal as two
  independent detectors is redundant and risks double-counting. Pick one
  canonical name (recommend keeping `wow_crystal` as the full, well-framed
  template and retiring/dedicating `wow_orb` to a sub-region, or use only
  `wow_crystal`).

### 2.3 The approval gate is the safety boundary of the ASSISTED milestone

- `src/ai_game_agent/agent/loop.py::_gate_ai_actions` is **fail-closed**: in
  `ASSISTED` mode an AI-proposed action dispatches **only** on an affirmative,
  valid approval decision. Missing approver, rejection, timeout, malformed
  decision, cancellation, or an approval exception all reject and pause
  (`no-approver-configured` is the recorded reason for the no-approver case).
- `src/ai_game_agent/actions/approver.py` provides `AutoApprover` (always
  approve — docstring: "Use in tests, CI, and explicit 'trust the AI' runs"),
  `TerminalApprover` (the **default human channel**: blocks on stdin for a y/N
  answer; non-interactive stdin, timeout, and interrupt are all *rejections*),
  and `QueueApprover` (a transport-agnostic remote/web approver, Phase 6). The
  CLI wires `TerminalApprover` for an ASSISTED run by default; `--approve-all`
  switches it to `AutoApprover` and `--approve-timeout` bounds the approval wait
  (`__main__.py`, ASSISTED-mode wiring). For the ASSISTED milestone the approver
  is therefore a human via `TerminalApprover`; `AutoApprover` is a *test
  double* by its docstring intent and is only reachable explicitly (`--approve-all`
  or direct injection), never by default config.
- Consequence: the ASSISTED milestone needs **no gate change** — the existing
  fail-closed gate plus a human approver is sufficient. The *later unattended
  goal* (§7.4) will additionally require exposing an explicit auto-approve mode
  via configuration (a new `approval: auto` setting that wires `AutoApprover`,
  or an equivalent explicit `AutoApprover`-backed `ASSISTED`-mode bypass). That
  is an *additive, opt-in* relaxation — the default remains fail-closed — and it
  is out of scope for the first milestone.

### 2.4 Documentation is already corrected (do not re-litigate)

- `phase-6-implementation-plan.md` §3.1/§4/§6 now state the shipped fail-closed
  contract (missing approver rejects and pauses; no "fallback dispatch"). This
  was corrected in PR #7 / commit `af3891b` and the historical notes in that
  plan record the earlier draft's fail-open wording as superseded.
- `README.md` now separates **implemented** capabilities from **roadmap**
  items (persistent memory and general planning are listed under roadmap, not
  under implemented features), and the `Architecture` diagram node is labelled
  `Decision (Rules + AI)` (not `Decision & Planning`) with matching prose.
- The **interactive CLI approval console is already implemented** in the shipped
  code: `TerminalApprover` is the default approver for `ASSISTED` mode and the
  `--approve-all` / `--approve-timeout` CLI flags switch to `AutoApprover` / a
  timeout-bounded variant. `AutoApprover`'s docstring marks it as a
  test/CI-friendly always-approve approver; the docs must not present it as
  the human approval boundary.
- Therefore §2.2 of the baseline plan (P0 documentation correction) is **done**;
  it is listed here only to record that it is complete. The residual README
  refinements that the baseline §7.0 checklist listed (the `Architecture`
  diagram node label and the `Architecture` section "planning" wording) have
  already been applied on this branch and are recorded under `### 7.0` below
  as already-complete.

---

## 3. Scenario (selected, bounded, already-implemented action)

**Scenario: "Loot the glowing crystal when the agent is in range."**

Chosen because it satisfies every constraint: one clear initial condition, one
observable fact, one bounded action (a click on the loot marker + the pickup
key), one observable completion condition, and it uses only primitives that
already exist and are already wired in the policy.

- **Initial condition:** the crystal/loot template is visible in the frame
  (detected at or above the calibrated confidence threshold) and the agent is
  `IDLE`/`READY` in the state machine.
- **Facts needed to recognize it:** a positive detection on the `ZONE_LOOT_GLOW`
  UI zone (or the `wow_crystal` template if the zone is not yet wired) at or
  above the calibrated confidence threshold (default 0.8), present in two
  consecutive frames to reject a transient. The signal name used must match
  what the policy actually reads (the policy uses `SIGNAL_LOOT_PRESENT`,
  derived from `ZONE_LOOT_GLOW`), so calibrate that signal specifically.
- **The one action the agent may propose:** the policy's existing loot sequence
  — `MOUSE_CLICK` on the center of the detected loot bounding box
  (`ZONE_LOOT_GLOW`), then `DELAY`, then `KEY_PRESS` of the configured pickup
  key (default `q`). This action is already implemented as a deterministic
  policy (`DeterministicPolicy._looting_actions`), which is what makes "no new
  input feature needed" provable. **Provenance:** under the shipped gate only
  the AI-seam path is gated, so the first control-capable run must be an
  AI-seam run (the provider proposes this same action and a human approves it)
  — see §5. The deterministic path is exercised separately to prove
  `OBSERVE_ONLY` dispatches zero input.
- **Expected visual result:** after the key press, the crystal is no longer
  detected (looted) — or, if the scenario is "open an interaction," the
  loot-glow state transitions. Define **one** of these as the completion
  condition before collecting evidence.
- **Abort conditions (all safe-pause, record evidence, zero further actions):**
  - no crystal detected after the action within the bounded observation window
    → classify **unknown** and pause;
  - a *different* state appears (target hostile, health low, zone change) →
    pause;
  - stale evidence (frame age beyond threshold) → reject and pause;
  - detector error (crystal detector did not run) → **unknown**, not "absent";
  - emergency stop / `OBSERVE_ONLY` boundary → no input, ever.
- **Why no new input feature is needed:** the loot action (click the marker +
  pickup key) is already built by the policy and the click is already
  implemented by the pynput executor — there is nothing to add. (Contrast:
  pathing/navigation and arbitrary mouse work *would* be new and are out of
  scope.)

> **Deliberately not chosen:** kill a training dummy (needs target-acquire +
> a combat rotation and a hostile-state transition — larger surface),
> pathing/navigation (out of scope), or any scenario needing a capability the
> executor/policy do not already have. The loot scenario is kept because its
> full action (click + key) is *already implemented*, which is what makes
> "no new input feature needed" provable.

### 3.1 Test environment & game-policy gate (must close before any input)

Per the baseline §3.3 and `AGENTS.md`: automated interaction with online games
may violate terms. Before the first control-capable run, the operator must
confirm the **selected environment and activity permit this automation**.
Acceptable evidence: a documented private test environment (e.g. a dedicated
test client, a throwaway character on a private/authorized setup, or a local
test harness/mock client) where the activity is permitted. If it cannot be
established, **Phase 7 stays observation-only** and the unattended goal (§7.4)
is deferred — do not weaken this gate.

This is a **one-time human gate** (the operator confirms the environment is
appropriate). It is distinct from, and in addition to, the per-action human
approval the ASSISTED milestone requires: closing the one-time gate does not
remove the requirement that each proposed action be approved at runtime.

---

## 4. Work packages (ordered, each independently testable)

Each package ends with a checkable exit gate. Run `uv run --extra dev pytest -q`
and `uv run ruff check src tests` after each.

### 7.0 — Close documentation and confirm the environment (no runtime change)

- [x] **Already done on this branch:** the fail-closed corrections in
      `docs/phase-6-implementation-plan.md` §3.1/§4/§6 (PR #7 / `af3891b`),
      the `README.md` implemented-vs-roadmap separation, the `Architecture`
      diagram node relabel (`Decision & Planning (Rules + AI)` →
      `Decision (Rules + AI)`), and the matching `Architecture` section prose
      ("planning" → "decision support"). These are part of this PR's diff.
- [ ] Close the §3.1 game-policy/environment gate with written evidence; record
      it in the scenario doc.
- **Exit gate:** a reviewer can point to (a) the merged fail-closed docs and
  (b) the written environment-authorization evidence. No code changed yet.

### 7.1 — Scenario state & deterministic policy wiring (no input dispatch)

- [ ] Add/confirm a single calibrated `wow_crystal` template signal in the
      scenario (retire or repurpose `wow_orb` per §2.2).
- [ ] Ensure `DeterministicPolicy` proposes the loot action sequence
      (click on the loot marker + pickup key; §3) when the loot fact is true
      and the agent is ready, and proposes nothing otherwise.
- [ ] Add deterministic fixture tests: loot-present → the loot action sequence;
      loot-absent → no action; detector-error → no action + unknown.
- **Exit gate:** with `OBSERVE_ONLY`, the loop logs the proposed action but
  dispatches **zero** input; tests pass headless.

### 7.2 — Opt-in auto-approve mode (later-goal prerequisite — not part of the first ASSISTED milestone)

- [ ] Add an explicit config knob (e.g. `approval: auto`) that wires the
      existing `AutoApprover` so an **AI-seam** action is auto-approved when
      the operator opts in. **Default stays `assisted`/fail-closed.**
- [ ] State the provenance explicitly and correctly: the shipped gate in
      `agent/loop.py` is keyed off the **AI seam** (state is `AI_ANALYSIS` and
      the provider decision produced the action), **not** off the
      `Action.source` field. For a deterministic policy action the gate does not
      apply under the current contract — do not paper over that by relabelling
      the deterministic action as `source == "ai"` to "make it pass through the
      gate"; that would misrepresent the action's provenance and, under the
      shipped gate, would not change the gating decision (see §5).
- [ ] Tests: `approval: auto` → an AI-seam action dispatches with no approver
      present; `approval: assisted` (default) with no approver → rejects,
      pauses, `no-approver-configured`; `assisted` + explicit rejection →
      rejects. All three tests must use a genuinely AI-seam action (one
      produced by the provider path), not a deterministic policy action
      relabelled as AI.
- **Exit gate:** the same scenario run dispatches under `approval: auto` with
  **zero** human input, and the fail-closed default is unchanged and tested.

### 7.3 — Real-screen observation baseline (Windows)

- [ ] Run the existing capture + perception pipeline against the real game
      screen in `OBSERVE_ONLY`, with a documented monitor/region/scale config.
- [ ] Collect ≥ 50 representative frames (positive / negative / borderline /
      degraded) + a small held-out set; label them.
- [ ] Record detector errors and distinguish "not detected" from "detector
      failed" from "detector did not run".
- [ ] Tune the `wow_crystal` threshold **only on the training split**; report
      precision/recall on the held-out split.
- **Exit gate:** repeated observation-only runs produce valid, timestamped,
  replayable records; the crystal is detected on the held-out positives with an
  acceptable precision/recall (record the numbers).

### 7.4 — Later goal: unattended end-to-end slice (not the first milestone)

**Status: follow-up goal — not part of the first ASSISTED milestone.** It is
recorded here so its prerequisites and exit condition are explicit. It may be
undertaken only after the ASSISTED milestone (7.1 + 7.3 + one approved bounded
decision) is accepted **and** each prerequisite below is met:

- **Prerequisite 1 — ASSISTED milestone accepted:** one bounded action is
  proposed, validated, and explicitly approved by a human in ASSISTED mode, with
  the fail-closed default (`assisted`) unchanged and still proven by tests
  (no-approver → reject + pause).
- **Prerequisite 2 — environment authorised:** the §3.1 environment / game-policy
  gate is closed with written authorization evidence, and the operator
  explicitly elects unattended operation.
- **Prerequisite 3 — opt-in auto-approve exposed:** an explicit auto-approve mode
  is available via configuration (e.g. `approval: auto` wiring the existing
  `AutoApprover`), the default remains fail-closed, and the action exercised is
  a **genuinely AI-seam** action (produced by the provider path in `AI_ANALYSIS`)
  so it demonstrably passes through the shipped gate (see §5). A deterministic
  policy action must **not** be relabelled as AI to "reach" the gate.

Only once all three prerequisites hold:
- [ ] Wire 7.1 + 7.2 on the real environment: `approval: auto`, the scenario
      action is produced by the **AI seam** (so the shipped gate in
      `agent/loop.py` is consulted), `OBSERVE_ONLY` is *not* set.
- [ ] Run the scenario with **no human at the keyboard**: the agent observes,
      the AI provider proposes the key press (AI seam), auto-approve dispatches it, the loop
      re-observes and classifies the outcome.
- [ ] Record the full evidence trail: commands, config, resolution/UI scale,
      model tag (if AI used), trial count, timings, frame captures, serialized
      observations, and loop/approval events.
- [ ] On the first **unknown/failure**, the loop must safe-pause with evidence
      and dispatch no further input.
- **Exit gate (later goal):** at least one fully unattended run succeeds
  (crystal looted / interaction completed) **and** at least one run that aborts
  on missing evidence safe-pauses cleanly — both with complete evidence. The
  operator intervened **zero times** at runtime.

### 7.5 — Evaluation report & close acceptance criteria

- [ ] Write the manual/automated validation report: trial counts, success rate,
      false positive/negative counts, detector-failure counts, latency, and the
      one-abort-on-unknown demonstration.
- [ ] Close **only** the acceptance criteria the evidence supports; leave the
      rest explicitly open.
- **Exit gate:** a reviewer can map each claimed capability to a recorded
      evidence artifact.

---

## 5. The one design decision to lock down (which path exercises the gate)

The shipped gate in `agent/loop.py` is keyed off the **AI seam**, not off the
`Action.source` field: it is consulted when the action is produced by the AI
decision path (state is `AI_ANALYSIS` and the provider returns a decision), and
it is not consulted for actions produced by the deterministic policy path.
The `Action.source` field is a **provenance tag** (the AI schema enforces
`source == "ai"` on AI-seam actions, and the deterministic policy sets a
non-AI source), not the gate's trigger. This means:

- The deterministic policy path does **not** pass through the `ASSISTED`
  approval gate under the current contract.
- Relabelling a deterministic action as `source == "ai"` to "make it pass
  through the gate" is **incorrect on both counts**: it misrepresents the
  action's provenance, and it does not change the gating decision (the gate
  still keys off the AI seam, not the `source` field).
- The only way to exercise the gate for the loot scenario is for the action to
  be produced by the AI seam (i.e. the loop enters `AI_ANALYSIS` and the
  provider returns the loot action), or to make a **separate, explicit safety
decision** to extend the gate to cover deterministic actions (baseline §7.3
  item 6) — the latter is a distinct change with its own tests, not an
  in-scope refinement of the first milestone.

**Decision for Phase 7 (ASSISTED milestone):** the first control-capable run
must be an **AI-seam** run — the loop enters `AI_ANALYSIS`, the provider
returns the loot action, and the gate resolves to a human approval decision.
This keeps the provenance accurate (the action really is AI-proposed),
exercises the shipped gate, and preserves the fail-closed default
(`assisted`) meaningfully and tested (no-approver → reject + pause). The later
unattended goal (§7.4), if undertaken, resolves the same gate to an opt-in
auto-approve for AI-seam actions.

Document this choice in the scenario doc and in the PR description. If the
operator instead wants the loot scenario to stay purely deterministic
(non-AI), state explicitly that the `ASSISTED` approval boundary does not
apply to that path and prove the boundary via the AI-seam regression tests;
treating the deterministic path as if it were gated is a separate safety
decision (baseline §7.3 item 6), not an in-scope assumption.

---

## 6. Explicit non-goals (unchanged from baseline, reaffirmed)

- No *new* mouse-control feature (the executor already implements mouse via pynput; §2.1) — and the selected scenario does not require one, so none is exercised in Phase 7.
- No unattended *long-running* sessions (this is one bounded scenario, repeated
  for evidence, not an open-ended run).
- No general questing/combat/routing; no web dashboard; no remote approver;
  no cloud model; no anti-cheat bypass, process-memory read, or code injection.
- No persistent memory / world model / RL / multi-agent.

---

## 7. Applied during this documentation PR (no further owner decision needed)

- README `Architecture` diagram node: relabel `Decision & Planning (Rules +
  AI)` → `Decision (Rules + AI)` — **applied in this PR's diff**.
- README `Architecture` section: "general planning" wording → "decision
  support" to match the shipped capability — **applied in this PR's diff**.
Both are cosmetic/prose and are already part of the `README.md` change in this
PR; no further owner decision is required to land them.

---

## 8. Open questions / unresolved scope (report only what is not yet closed)

1. **Completion condition for the loot scenario** — pick *one*: (a) crystal
   no longer detected after the key press, or (b) a loot-glow state transition.
   Must be fixed **before** 7.3 evidence collection so labels are defined up
   front. *(Decision needed from the operator; §3 states the requirement but
   does not force a choice.)*
2. **Exact game key bind** for the loot/interaction action in the chosen
   environment (default `q` from config; confirm the bind in the test client).
3. **Test-environment authorization evidence** for §3.1 — not yet provided;
   this is the hard gate before 7.2/7.4 dispatch any input.
4. Whether to keep `wow_orb` as a distinct template or retire it (recommend
   retire, per §2.2) — minor, resolve during 7.1.
5. *(Resolved in this PR: the README refinements listed in §7 are
   applied in this PR's diff; no open owner decision remains for them.)*

---

## 9. Definition of done (Phase 7 — first ASSISTED milestone)

- The selected loot scenario and test environment are documented, with the
  completion condition fixed and the environment-authorization evidence on
  record.
- A single calibrated `wow_crystal` signal is tuned on labelled, held-out
  frames with reported precision/recall.
- The one bounded loot action (click the marker + pickup key) is proposed by
  the AI seam for the control-capable run, and nothing else is proposed; the
  deterministic policy is proven to produce the same action and to dispatch
  nothing under `OBSERVE_ONLY`.
- A single bounded action is proposed, validated, and **explicitly approved by
  a human in ASSISTED mode** before it reaches the executor; the `assisted`
  fail-closed default is unchanged and still proven by tests (no-approver →
  reject + pause).
- At least one **ASSISTED** run succeeds (the action is approved, the crystal is
  looted / the interaction completes) and one run that hits missing evidence
  safe-pauses cleanly, both with complete evidence; every action was approved by
  a human at runtime.
- *(Later unattended goal, §7.4 — separate DoD, not part of this milestone)*
  once its three prerequisites are met, at least one fully unattended run
  succeeds and one run that hits missing evidence safe-pauses cleanly, with the
  operator intervening zero times at runtime.
- Automated regression checks pass headless; no WoW client, desktop, Ollama,
  or network required for CI.
- The validation report maps every claimed capability to a recorded evidence
  artifact, and only evidence-backed acceptance criteria are closed.

**Phase 7 does not authorize a claim of general autonomous gameplay.** The
first milestone proves one controlled operational slice — the agent proposes a
single bounded action that is explicitly approved by a human in ASSISTED mode —
and provides the evidence to decide what the next phase should tackle.
Unattended crystal collection is a later, separately-gated goal (§7.4).
