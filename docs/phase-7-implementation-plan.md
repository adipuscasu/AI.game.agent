# Phase 7: Operational Autonomy and Gameplay Validation

**Status:** proposed — planning baseline only; no Phase 7 implementation is included in this commit  
**Branch:** `feature/phase-7-operational-autonomy`  
**Base:** `develop` at `d4bc1ef192aed9c1849641541df190feb68d0e27`  
**Date:** 2026-11-06  
**Prerequisites:** Phases 1–6 and PR #7 merged

---

## 1. Purpose

Phases 1–6 established the main software foundations: screen capture, a structured perception pipeline, a deterministic state machine and WoW policy, action execution, local AI consultation, and fail-closed human approval. PR #7 completed an important part of the visual-grounding and approval-safety work.

The next risk is no longer primarily architectural. It is whether these components work together reliably against a real, changing game screen.

**Phase 7 is an evidence-driven validation phase.** Its objective is to demonstrate one small, bounded, repeatable World of Warcraft scenario from observation through a proposed action and verified outcome. It must produce reproducible evidence, not just a successful CLI exit or a syntactically valid AI decision.

The intended progression is:

```text
Real game screen
      |
      v
Capture and record
      |
      v
Perception + calibrated WoW state
      |
      v
Deterministic policy / local VLM proposal
      |
      v
Action validation and safety guard
      |
      v
Human approval (ASSISTED mode)
      |
      v
One bounded action
      |
      v
Fresh observation
      |
      v
Outcome verification
      |
      +---- success ----> continue / finish the scenario
      |
      +---- failure or unknown ----> safe pause and record evidence
```

The first control-capable milestone remains **ASSISTED**, not unattended AUTONOMOUS operation. Observation-only work comes first.

### 1.1 Primary outcome

At the end of Phase 7, the repository should contain a tested implementation and a manual validation record showing that the agent can:

1. Observe a real, explicitly selected game scenario.
2. Extract the small set of state facts required by that scenario.
3. Propose one valid, bounded action.
4. Require an affirmative human approval before an AI-proposed action is dispatched.
5. Observe the environment again after the action.
6. Classify the result as success, failure, or unknown using new visual evidence.
7. Stop safely when evidence is missing, contradictory, stale, or outside the scenario.

The scenario itself must be chosen and documented before implementation starts. Keep it deliberately narrow; do not make “play WoW” the acceptance criterion.

---

## 2. Current implementation baseline

The plan builds on existing contracts. It should extend them only where the scenario exposes a demonstrated gap.

| Existing component | Current responsibility | Phase 7 use |
| --- | --- | --- |
| `capture` / `Frame` | Real or mock frame acquisition; region and scale configuration | Capture a repeatable game view |
| `Perception` / `Observation` | Immutable, structured template, UI-zone, OCR and object results; detector errors are retained | Produce evidence for the selected scenario |
| `games/wow/DeterministicPolicy` | WoW-specific signals and proposed actions without direct input I/O | Keep scenario rules game-specific and deterministic where possible |
| `GameStateMachine` | State transitions, watchdogs and bounded recovery | Preserve explicit state and failure handling |
| `AgentLoop` | Capture → observe → decide → gate → dispatch; owns the event sequence | Remain the orchestration boundary |
| AI provider seam | Local provider receives a prompt and captured PNG bytes; returns a structured decision | Use visual reasoning only when deterministic evidence is insufficient |
| `ASSISTED` approval | AI actions require an affirmative approval; rejection or invalid approval fails closed | Human authorization for every AI-proposed action |
| Event recorder / CLI recording | Structured loop events can be written to JSONL; frames can be recorded separately | Preserve an auditable, replayable session |

Relevant implementation locations include:

- `src/ai_game_agent/agent/loop.py`
- `src/ai_game_agent/perception/observation.py`
- `src/ai_game_agent/perception/pipeline.py`
- `src/ai_game_agent/games/wow/policy.py`
- `src/ai_game_agent/actions/approver.py`
- `src/ai_game_agent/state/events.py`
- `src/ai_game_agent/__main__.py`
- `config/default.yaml`

### 2.1 Existing contracts to preserve

- The generic agent loop must not gain WoW-specific zone names, key bindings, or scenario rules.
- The WoW adapter/policy owns game-specific interpretation and action proposals.
- The policy proposes `Action` objects; only the executor dispatches input.
- `OBSERVE_ONLY` must remain a hard no-input boundary.
- In `ASSISTED`, an AI-proposed action must never dispatch without an affirmative, valid approval decision. Missing approver, rejection, timeout, malformed decision, cancellation, or an approval exception must not authorize execution.
- The current gate applies to actions marked `source == "ai"`; deterministic/non-AI actions are not automatically covered by that approval gate. The Phase 7 control scenario must make this distinction explicit. Do not assume that selecting `ASSISTED` means every action source is human-approved.
- Perception detector failures are recorded in `Observation.detector_errors`. A failed detector means **unknown**, not “the feature is absent.”
- The captured frame used for an AI decision must be the frame associated with that observation; do not silently fall back to a text-only decision if image encoding fails.
- Cleanup, held-input release, emergency stop, bounded recovery, and event sequencing remain mandatory.
- Local Ollama inference remains the default target. Phase 7 must not introduce a cloud dependency.

### 2.2 Documentation readiness correction

Before implementation work begins, reconcile the existing Phase 6 plan with the shipped fail-closed behavior. In `docs/phase-6-implementation-plan.md`, sections 3.1, 4, and 6 still describe the old behavior in which ASSISTED mode without an injected approver falls back to dispatch. That is inconsistent with the current implementation and must be corrected to state that a missing approver rejects the AI proposal and safely pauses before dispatch.

Also review the README feature list. Clearly separate implemented capabilities from roadmap items; do not present persistent memory, general planning, or navigation as shipped functionality unless the code actually supports those claims.

This is a documentation/safety-contract correction, not permission to weaken the implementation for backward compatibility.

---

## 3. Scope and non-goals

### 3.1 In scope

- One explicitly named, low-risk WoW scenario with a clear initial condition and observable completion condition.
- Real-screen capture using the existing capture abstractions and a documented monitor/region/scale configuration.
- A small, calibrated set of scenario-specific visual signals.
- A typed, serializable WoW state snapshot if the existing `Observation` is not sufficient for expressing the scenario's domain facts.
- Deterministic state interpretation before asking the VLM.
- A single-action decision path in ASSISTED mode.
- Fresh post-action observation and explicit outcome verification.
- Safe pause on failed or ambiguous verification.
- Session evidence: frame captures, structured observations, loop/action/approval events, timestamps, configuration, and a concise manual run report.
- Unit tests, deterministic replay tests, and a manually executed real-screen validation.

### 3.2 Explicitly out of scope

Do not use Phase 7 to build:

- Unattended autonomous gameplay or long-running unattended sessions.
- General questing, combat rotations, farming, route planning, or broad navigation.
- A generic planner, persistent world model, long-term memory, reinforcement learning, or multi-agent orchestration.
- A web dashboard, remote approval server, or a web adapter for `QueueApprover`.
- A cross-game adapter framework beyond preserving the existing game/core boundary.
- Cloud model providers or remote telemetry.
- A large perception rewrite or speculative image-processing optimization before end-to-end measurements exist.

These may be future projects, but they are not prerequisites for proving the first operational slice.

### 3.3 Game-policy and safety gate

The README already warns that automated interaction with online games may violate the game's terms. Before any control-capable live-game test, the operator must confirm that the selected environment and activity permit this type of automation. If that cannot be established, Phase 7 remains observation-only.

Use ordinary screen capture and the project's normal input abstraction. Do not add process-memory reading, code injection, anti-cheat bypasses, stealth behavior, or detection-evasion mechanisms. Do not treat a private test environment as permission to bypass a game's technical or contractual controls.

---

## 4. Work packages

Implement these as small, reviewable steps. Keep each package independently testable.

### 7.0 — Establish the scenario and repair the documentation

**Goal:** remove ambiguity before changing runtime behavior.

Tasks:

1. Correct the Phase 6 documentation contradictions identified in §2.2.
2. Update README wording so implemented features and future aspirations are clearly distinguished.
3. Select one scenario and write down:
   - initial screen/game conditions;
   - the observable facts needed to recognize that condition;
   - the one action the agent may propose;
   - the expected visual result;
   - timeout and abort conditions;
   - why the test environment is appropriate for input automation.
4. Fix the test setup variables for the first dataset: game resolution, window mode, UI scale, language, capture monitor/region, capture scale, and relevant UI layout.
5. Define the success/failure/unknown labels before collecting evaluation evidence.

**Exit gate:** a human can follow the scenario description and independently decide whether the initial condition and expected result are present.

### 7.1 — Real-screen observer baseline

**Goal:** make the real input to the agent reproducible before enabling actions.

Tasks:

1. Run the existing capture and perception pipeline against the selected game screen in observation-only mode.
2. Determine whether the existing monitor and region configuration captures the intended game area consistently. Add a narrowly scoped capture capability only if a real limitation is demonstrated; do not redesign capture pre-emptively.
3. Record frames and the corresponding serialized `Observation` values. Preserve timestamps and enough metadata to associate each frame, observation, and event.
4. Collect an initial manually labelled set of at least **50 representative frames**, including positive, negative, borderline, and visually degraded cases.
5. Keep a small held-out set that is not used while tuning thresholds.
6. Record detector errors and distinguish “not detected” from “detector did not run” or “detector failed.”

**Deliverables:**

- documented capture configuration;
- local, reproducible scenario frame set and labels (subject to repository size/privacy constraints);
- a short baseline report with observed false positives, false negatives, and detector failures.

**Exit gate:** repeated observation-only runs produce valid, timestamped records, and the scenario-critical evidence can be inspected without running the model or dispatching input.

### 7.2 — Calibrated scenario state

**Goal:** turn perception evidence into a small set of trustworthy game facts.

Tasks:

1. Define only the state fields required by the selected scenario. Prefer a small frozen, serializable WoW-domain snapshot over adding game-specific fields to the generic `Observation` type.
2. For each field, define its source, coordinate space, valid range, freshness, confidence/threshold policy, and unknown representation.
3. Reuse configured UI zones, templates, and OCR where they provide reliable evidence. Add a detector only for a documented gap.
4. Calibrate against the labelled frame set and evaluate on held-out examples.
5. Add hysteresis/debounce only where repeated-frame evidence shows it is needed.
6. Keep uncertain and failed readings unknown. Never convert an OCR failure, missing detector, out-of-bounds region, or stale result into a healthy/absent state.
7. Wire runtime-only signals such as a health fraction only after the reading is calibrated and its `None`/unknown behavior is explicitly tested.

**Exit gate:** the scenario-critical state can be derived deterministically from a frame/observation fixture, including explicit unknown cases, without requiring an LLM call.

### 7.3 — One bounded decision in ASSISTED mode

**Goal:** connect perception, policy, local AI and approval for the single selected scenario.

Tasks:

1. Keep deterministic policy as the first decision layer. Use the VLM only for the specific interpretation that cannot be handled reliably by configured deterministic evidence.
2. Ensure the prompt includes the current scenario state and the screenshot from the same capture.
3. Keep the model output structured and validate it through the existing decision/action schemas and safety boundary. Reject unsupported action kinds, malformed arguments, stale context, or proposals outside the scenario allow-list.
4. Start with **one action per decision**. Do not introduce multi-step plans or arbitrary action batches.
5. Use `ASSISTED` mode with the terminal human approver for manual trials. The action under test must be marked `source == "ai"` so it actually passes through the existing approval gate. Do not use `AutoApprover` for a real-game control trial.
6. If the selected scenario can only produce a deterministic/non-AI action, stop and decide explicitly whether Phase 7 should extend the approval contract to cover it. Treat that as a separate safety change with its own tests; do not quietly assume the current gate protects it.
7. Prove with regression tests that no input is dispatched on missing approval, rejection, timeout, cancellation, invalid output, provider failure, or safety-guard rejection.
8. Keep F12/emergency-stop and release-all behavior active for every control-capable run.

**Exit gate:** a valid scenario decision can reach the approval prompt, and only an affirmative approval permits the one bounded action to reach the executor.

### 7.4 — Post-action outcome verification

**Goal:** verify that the environment changed as expected instead of assuming that a successful input call means the objective succeeded.

Tasks:

1. Define the expected postcondition for the scenario before dispatching the action.
2. After dispatch, acquire a fresh frame and derive a fresh scenario state. Do not reuse the pre-action observation as evidence of success.
3. Compare the new evidence with the expected postcondition and classify the result as:
   - `success` — the required visual evidence is present;
   - `failure` — reliable evidence shows the expected result did not occur or an adverse condition occurred;
   - `unknown` — evidence is missing, stale, contradictory, or below the confidence threshold.
4. Correlate the proposal, approval, dispatch, subsequent observation, and verification result in the session record. Prefer extending existing event details/contracts over adding a new event type without a demonstrated need.
5. A `failure` or `unknown` result must not trigger an unbounded retry. Use a bounded recovery path or safely pause and preserve the evidence.
6. Test delayed visual updates, no-op input, partial result, detector failure, and contradictory signals.

**Exit gate:** the agent can distinguish “input was sent” from “the scenario objective was visibly achieved,” and ambiguous outcomes stop safely.

### 7.5 — End-to-end evaluation and operational hardening

**Goal:** demonstrate the complete scenario repeatedly and preserve enough evidence to reproduce failures.

Tasks:

1. Run replay tests over the labelled fixtures without a game, desktop, or Ollama dependency.
2. Run manual integration trials on the approved test environment in observation-only mode first, then in ASSISTED mode.
3. Record capture/perception latency, VLM latency, approval wait, action duration, time-to-verification, and total scenario duration. Report median and p95 where the sample size supports it.
4. Record each trial's initial condition, proposed action, approval, dispatch result, verified outcome, and failure/recovery notes.
5. Test F12 during idle, approval wait, and action execution; verify held inputs are released and no later action is dispatched.
6. Document setup, the exact manual commands, expected output, known limitations, and how to inspect/replay a failed session.
7. Make no performance optimization based on intuition. Use the recorded timings and payload sizes to decide whether image resizing, cropping, or other changes are justified.

**Exit gate:** all acceptance criteria in §6 are met and a reviewer can inspect the evidence for each trial.

---

## 5. Proposed implementation boundaries

These are design constraints, not a mandate to introduce every named abstraction. Confirm the smallest suitable API during the relevant work package.

### 5.1 Keep game-specific state out of the generic loop

The generic `AgentLoop` should continue to orchestrate collaborators without knowing WoW UI labels or scenario semantics. Prefer scenario interpretation and configuration in `games/wow` and the perception/config layers.

A WoW snapshot may be warranted if `Observation` plus policy signals cannot express the scenario cleanly. If introduced, make it immutable, serializable, explicit about unknown values, and derived from `Observation` rather than from hidden game state.

### 5.2 Keep AI as a proposer, not an authority

The VLM may interpret visual context and propose a schema-valid action. It must not bypass:

- action schema validation;
- operation-mode restrictions;
- `SafetyGuard` and executor limits;
- the ASSISTED approval gate;
- emergency stop and cleanup;
- post-action verification.

Do not allow free-form model text to become an OS input command.

### 5.3 Verification is a separate fact from dispatch

An executor result can establish whether an input operation was accepted by the backend. It cannot establish that the game performed the intended operation. Keep these facts distinct in types, events, reports, and tests.

A missing post-action observation must be `unknown`, not success. A retry must require an explicit, bounded policy and fresh evidence.

### 5.4 Preserve evidence locally

Keep frames, observations, and event records local by default. Do not add automatic uploads or telemetry. Include enough metadata to reproduce the run, but avoid recording credentials, account secrets, or unrelated personal information visible in the game/desktop.

---

## 6. Acceptance criteria

The following are initial targets for this deliberately small scenario. Establish the baseline in 7.1; any proposed threshold change must be justified in the evaluation report before it is used to claim completion.

| Area | Acceptance target |
| --- | --- |
| Labelled visual evaluation | At least 50 labelled frames, with a held-out subset |
| Scenario-critical state | At least 95% precision and 90% recall on the held-out set; report counts and confusion matrix, not just percentages |
| Unknown handling | 100% of injected detector-failure/stale/ambiguous cases remain non-actionable |
| Approval safety | 0 dispatches without a valid affirmative approval |
| Action scope | Only the single allow-listed scenario action can be proposed and dispatched |
| Outcome verification | At least 18 of 20 controlled attempts receive a correctly evidenced terminal classification; report success, failure, and unknown separately |
| Failure containment | No blind/unbounded retries; every injected verification failure ends in bounded recovery or safe pause |
| Emergency stop | F12 prevents subsequent dispatch and releases held input in each tested stop window |
| Reproducibility | Every manual trial has a linked frame/observation/event record and a concise result |
| Regression gates | Full tests, Ruff, and diff checks pass; no real game or Ollama is required by CI |

Do not hide unknown outcomes inside the success rate. If 20 trials cannot be performed safely or legitimately, report the smaller sample and leave the acceptance gate open rather than manufacturing a pass.

### 6.1 Automated checks

Run from the repository root:

```powershell
uv run --extra dev pytest -q
uv run ruff check src tests
git diff --check
```

Also run focused tests for each changed contract before the full suite. Keep CI deterministic and headless: no WoW client, physical desktop, Ollama server, or external network service should be required.

### 6.2 Manual checks

The manual test must use a documented Windows setup and the selected, approved scenario. Start with observation-only capture and inspect the evidence before enabling input. Then use ASSISTED mode and an explicit human approval for each AI proposal.

Capture and retain the exact command, configuration, app/game version where relevant, screen resolution/UI scale, model tag, trial count, timings, and any failures. Do not describe a mocked test or a prompt/response test as a real gameplay result.

---

## 7. Suggested PR sequence

Prefer small PRs with a narrow review surface:

1. **P0 — Documentation truth and Phase 6 safety contract.** Correct the stale fail-open descriptions; reconcile README feature claims.
2. **P1 — Observer baseline and scenario fixtures.** Add only the capture/evidence support demonstrated to be missing; define and label the first scenario.
3. **P2 — Calibrated WoW scenario state.** Add typed state and deterministic fixture tests, if needed.
4. **P3 — One-action ASSISTED vertical slice.** Connect the scenario decision to validated, approved execution.
5. **P4 — Outcome verification.** Add post-action observation, result classification, correlation, and bounded failure handling.
6. **P5 — Evaluation and operational documentation.** Run the manual trials, publish measurements, and close only the acceptance criteria supported by evidence.

Do not combine all six steps into one implementation PR. Each PR should state which acceptance criteria it advances and include test evidence.

---

## 8. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| UI layout or resolution changes invalidate coordinates | Fix and record display/UI settings for the first scenario; treat out-of-bounds or changed layout as unknown |
| OCR/template false positives create unsafe actions | Use held-out labelled frames, conservative thresholds, and explicit unknown states |
| VLM proposes plausible but incorrect actions | Narrow action allow-list, schema validation, ASSISTED approval, and post-action verification |
| Input dispatch succeeds but game ignores it | Verify the resulting game state from a fresh frame |
| Delayed animations make verification premature | Define a bounded observation window and classify unresolved results as unknown |
| Perception or model latency makes the loop stale | Record timestamps and age of evidence; reject stale proposals |
| Game policy or environment does not permit automation | Gate control trials on an appropriate environment; remain observation-only otherwise |
| Scope expands into general autonomy | Keep one scenario and one action as the Phase 7 boundary |

---

## 9. Definition of done

Phase 7 is complete only when:

- the Phase 6 documentation reflects the actual fail-closed approval contract;
- README claims distinguish implemented functionality from future roadmap;
- the selected scenario and test environment are documented;
- the observer baseline and labelled evaluation data are reproducible;
- scenario-critical state is calibrated and unknown-safe;
- one bounded action can be proposed, validated, and explicitly approved in ASSISTED mode;
- the environment is observed again and the result is verified from fresh evidence;
- failure, ambiguity, timeout, cancellation, and emergency stop all remain safe;
- automated regression checks pass;
- the manual evaluation report includes actual trial counts and measurements.

**Phase 7 does not authorize a claim of general autonomous gameplay.** It proves one controlled operational slice and provides evidence to decide what the next phase should tackle.
