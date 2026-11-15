# EH4 Step 2 — Reproducible Perception Evaluator: Workplan

Branch: `feature/eh4-reproducible-evaluator` (base `develop` @ `198b54a`, PR #14 merged).
Companion to Step 1: the versioned `DatasetManifest` / `Observation` label contract.

**Goal:** a deterministic, game-agnostic, independently testable evaluation engine
that scores a set of predictions against a validated manifest — per-class,
per-category, per-split — with optional injected state-machine evaluation, and
serializes to a byte-reproducible JSON report. A *measurement* tool, not an
optimizer.

## 1. Architecture / API

New modules under `src/ai_game_agent/evaluation/` (package docstring already
promises "reproducible scoring"):

| Module | Responsibility |
| --- | --- |
| `scoring.py` | IoU, one-to-one matching, TP/FP/FN metrics, `ScoringConfig`. Stdlib only, no perception imports beyond `Observation`/`BBox` types. |
| `state_eval.py` | `StateEvaluation` result type, `StateEvaluator` protocol (injectable adapter), expectation-vs-actual classification (TP/TN/FP/FN/UNKNOWN per signal, correct/incorrect/unknown per state, exact reason comparison). |
| `dataset.py` | `evaluate_dataset(manifest, predictions, config, state_evaluator)` — the public API. Validates inputs, keeps splits separate, classifies samples, never mutates inputs. |
| `report.py` | `EvaluationReport` dataclasses + `to_dict()` / `to_json()` (deterministic). |
| `__init__.py` | Exports + `EVALUATOR_VERSION` constant. |

Dependencies: **stdlib only**. No numpy, no OpenCV, no WoW modules, no LLM.
`GameStateMachine` (`state/machine.py`) is *not* imported by the generic code —
adapters implement the `StateEvaluator` protocol. The machine's contract
(`decide(observation, now) -> AgentState`, policy `transition_signals(observation)`,
`None` signal = unknown and never `False`) shapes the adapter interface only.

Public entry point:

```python
from ai_game_agent.evaluation import (
    EVALUATOR_VERSION,
    ScoringConfig,
    evaluate_dataset,
)

report = evaluate_dataset(
    manifest,  # validated DatasetManifest
    predictions,  # Mapping[str, Observation], keyed by sample id
    config=ScoringConfig(),  # iou_threshold default 0.5 (keyword-only)
    state_evaluator=None,  # optional StateEvaluator adapter
)
report.to_json()  # byte-identical for identical inputs
```

## 2. Matching semantics (scoring.py)

Four categories, each with its own class identity:

| Category | Class key | Extra match condition |
| --- | --- | --- |
| templates | `TemplateHit.template` | — |
| ui_zones | `(UiZoneHit.zone, UiZoneHit.check)` | — |
| text_regions | `TextRegion.text` normalized | normalized texts must be equal |
| objects | `ObjectHit.kind` | — |

**Text normalization** (explicit, deterministic, no fuzzy matching):
`str.casefold()` → collapse every run of whitespace (`\t \r\n`) to a single
space → strip. `normalize_text("  Hello   World  ") == "hello world"`.

**IoU**: standard axis-aligned intersection-over-union on integer `BBox`.
Edge rule: if both boxes have zero area (union == 0), IoU is `1.0` when the
two boxes are identical, `0.0` otherwise — documented and tested.

**Assignment**: one-to-one, optimal, lexicographic objective —
*(1) maximize the number of valid matches (IoU ≥ threshold),
(2) then maximize total matched IoU.* Implemented as maximum-weight bipartite
matching (Hungarian algorithm, O(n³), stdlib), with weight
`BIG + iou` for valid pairs and `-BIG` for invalid ones (`BIG = n + 1`
exceeds any achievable IoU sum). Ties resolve to the lowest index — deterministic.
This is deliberate: greedy-by-IoU is *not* optimal for the count-first objective
(covered by a test).

Expected with no valid predicted partner → FN. Predicted with no valid
expected partner → FP. Class-mismatched hits never match.

## 3. Metric definitions

For each class, category, split, and overall (per split):

- `tp` / `fp` / `fn` counts; `expected_total`, `predicted_total`.
- `precision = tp / (tp + fp)` — `None` when the denominator is 0 (never a
  manufactured 1.0). `recall = tp / (tp + fn)` — `None` when 0.
- `f1 = 2pr / (p + r)` — `None` if either side is `None`, else the harmonic mean.
- **ui_zones only**: matched pairs additionally report `value_error =
  abs(value_expected − value_predicted)` (measured-evidence error).
  `UiZoneHit.value` is evidence, not confidence — localization accuracy (IoU)
  and measurement accuracy (value error) are separate report fields.
- No silent confidence filtering. (A future explicit
  `ScoringConfig.confidence_floor` may be added; not in this step.)

## 4. Incomplete-evidence policy (conservative, explicit)

- `Observation.detector_errors` (free text, non-empty) means the observation
  did not fully run. Such a sample is **never** scored as clean — missing
  detections in it are *missing evidence*, not negative detections.
- Sample statuses, reported in four separate buckets:
  - `scored` — prediction present, no detector errors on either side.
  - `incomplete` — prediction present but `detector_errors` non-empty →
    excluded from metrics, listed with the error strings.
  - `skipped` — the *label* observation carries `detector_errors` →
    ground truth itself incomplete → excluded from metrics, listed.
  - `missing` — no prediction at all for the sample id → reported as missing
    evidence, excluded from metrics.
- Detector error strings are copied into the report verbatim (no parsing).

## 5. Split isolation

`calibration` and `held_out` are scored independently with the *same* config
and reported in separate sections. No combined/overall-across-splits numbers.
No threshold tuning, no feedback loop: the API has no mutation path back into
config or dataset. Sample order is manifest order (deterministic).

## 6. State-evaluation interface (state_eval.py)

Injectable protocol — no WoW imports in generic code:

```python
class StateEvaluation:
    signals: Mapping[str, bool | None]  # missing key == unknown
    state: str | None  # None == adapter could not determine
    reason: str | None


class StateEvaluator(Protocol):
    def evaluate(self, observation: Observation) -> StateEvaluation: ...
```

Per expected signal (`StateExpectation.signals` is `str -> bool`):

| expected | actual | verdict |
| --- | --- | --- |
| True | True | TP |
| False | False | TN |
| True | False | FN |
| False | True | FP |
| True/False | `None` or key absent | **UNKNOWN — never false** |

State: `correct` / `incorrect` (with `expected` vs `actual` confusion pair) /
`unknown` (adapter returned `None`). Reason is compared **only** when the
expectation supplies one; comparison is exact string equality, reported as
`reason_match: bool | None`.

Samples without a state expectation remain valid observation-only samples
(`state: not_evaluated` in the report). **No adapter ⇒ state evaluation is
explicitly `not_performed` in the report — never a silent success.**
Adapters may be stateful: the evaluator feeds them in manifest order and does
not reset them; the caller supplies a fresh adapter per run (documented).
**One adapter spans both splits** — calibration samples first, then held-out,
with no reset between them — so a stateful machine evaluates the held-out
samples with its post-calibration history. That is the intended semantics
for an evolving state machine; it does mean the held-out state section is
not independent of calibration's. Per-split adapters (if independence is
ever required) are a *proposed* API change, not made in this step
(regression test: `test_stateful_adapter_is_shared_across_splits_in_manifest_order`).

## 7. Dataset API (dataset.py)

- `evaluate_dataset(manifest, predictions, config, state_evaluator=None)`.
- `predictions` keyed by sample id; **foreign ids are rejected** (`ValueError`
  listing them) — a prediction that belongs to no sample is an input contract
  violation, not a reportable observation.
- `ScoringConfig.iou_threshold` validated: real number (not bool), finite,
  `0.0 <= t <= 1.0`; default `0.5`.
- Never mutates the manifest, samples, or observations (inputs are frozen
  dataclasses anyway; the API must not construct mutated copies as if they
  were the originals).
- Deterministic: same inputs ⇒ same report (sample order = manifest order;
  per-class maps sorted).

## 8. Report schema (report.py)

```jsonc
{
  "schema": "evaluation-report",
  "evaluator_version": 2,
  "manifest_schema_version": 1,
  "dataset": "...", "dataset_version": "...",
  "scoring": {"iou_threshold": 0.5},
  "counts": {"scored": 0, "incomplete": 0, "skipped": 0, "missing": 0, "total": 0},
  "splits": {
    "calibration": {
      "overall": {"tp":0,"fp":0,"fn":0,"expected_total":0,"predicted_total":0,
                  "precision":null,"recall":null,"f1":null},
      "categories": { "templates": {...same shape + "classes": {...}} , ... },
      "state": { "performed": true|false|..., "samples_evaluated": 0,
                 "signals": {"TP":0,"TN":0,"FP":0,"FN":0,"UNKNOWN":0},
                 "states": {"correct":0,"incorrect":0,"unknown":0},
                 "reason_matches": 0, "reason_mismatches": 0 }
    },
    "held_out": { ... }
  },
  "samples": [
    { "id": "...", "split": "...", "status": "scored|incomplete|missing|skipped",
      "detector_errors": [...],
      "categories": { "templates": {"tp":..,"fp":..,"fn":..,
                       "matches":[{"iou":..,"expected":"..","predicted":".."}],
                       "false_positives":[...],"false_negatives":[...]} , ... },
      "state": { ...per-sample signal/state/reason result... } }
  ]
}
```

Serialization rules: `to_dict()` builds fixed-order dicts; `to_json()` is
`json.dumps(deterministic separators, ensure_ascii=True)` — **identical inputs
⇒ byte-identical JSON**. No timestamps, no environment data, no non-finite
numbers (all floats validated finite at construction), stable ordering
(per-class maps sorted by key; samples in manifest order).

## 9. TDD sequence (strict RED → GREEN → REFACTOR)

1. `scoring.py`: IoU (overlap/disjoint/zero-area/identical) → config validation
   → per-category class identity (cross-class never matches) → optimal
   assignment (incl. the greedy-suboptimal case) → deterministic tie-breaking
   → TP/FP/FN & P/R/F1 (zero-denominator → `None`) → ui_zone `value_error` →
   text normalization + text-mismatch handling.
2. `state_eval.py`: signal verdict matrix incl. UNKNOWN → state
   correct/incorrect/unknown + confusion → reason only-when-expected →
   protocol shape.
3. `dataset.py`: sample status classification (scored/incomplete/missing/
   skipped) → split isolation → foreign-id rejection → no adapter ⇒
   `not_performed` → no-mutation → deterministic sample order.
4. `report.py`: full schema shape → byte-identical `to_json()` → no non-finite
   → stable ordering → counts.
5. `__init__.py` exports + `EVALUATOR_VERSION`; runnable synthetic example
   (docs §10); workplan status update.

## 10. Test matrix

| Area | Cases |
| --- | --- |
| IoU | identical, partial, disjoint, zero-area pair, identical zero-area |
| Matching | one-to-one, threshold boundary, cross-class isolation, two expecteds sharing one predicted (count wins), greedy-failure optimality, deterministic tie-break, text-normalized equality, text-mismatch → FP+FN |
| Metrics | pure TP / pure FP / pure FN, mixed, zero-denominator P and R → `None`, F1 → `None`, per-class vs per-category vs overall consistency |
| Incomplete | prediction with detector_errors → incomplete bucket, not scored; label with errors → skipped; missing prediction → missing bucket, explicitly reported |
| State | full verdict matrix, unknown-never-false, state confusion, reason present/absent, no adapter ⇒ `not_performed`, stateful adapter fed in manifest order |
| Split isolation | same sample content in both splits reported separately; no combined numbers |
| Report | byte-identical JSON across two runs, sorted class maps, `None` → `null`, counts add up, foreign ids raise |

## 11. Exclusions

No CLI (the package API is the interface; a CLI command is a candidate
follow-up, not justified yet). No gameplay/loop changes, nothing in
`agent/loop.py`. No WoW imports in generic modules. No new dependencies.
No threshold auto-tuning. No state-machine runtime changes. No speculative
abstractions (no plugin registry, no config-file loader — plain dataclasses).

## 12. Status (updated as work proceeds)

| Step | Status | Evidence |
| --- | --- | --- |
| Step 0 workplan | done | this file |
| Step 1 scoring | done | `scoring.py` + `tests/test_eval_scoring.py` (34 tests green: IoU edge cases, optimal vs greedy assignment, tie-breaks, zero-denominator `None`, `value_error`, text normalization) |
| Step 2 state_eval | done | `state_eval.py` + `tests/test_eval_state.py` (27 tests green: full verdict matrix, unknown-never-false, reason only-when-expected, protocol shape) |
| Step 3 dataset API | done | `dataset.py` + `tests/test_eval_dataset.py` (20 tests green: status precedence missing > skipped > incomplete > scored, foreign-id rejection, no-adapter `not_performed`, stateful-adapter manifest order, non-mutation, deterministic JSON) |
| Step 4 report | done | `report.py` + `tests/test_eval_report.py` (27 tests green: constructible frozen dataclasses, `Counts` invariants, sorted class maps, not-performed ⇒ zero samples, byte-identical `to_json`, no non-finite tokens, ASCII-safe) |
| Step 5 TDD + example + docs | done | `__init__.py` exports (34 names) + `EVALUATOR_VERSION`; runnable `docs/eh4-evaluator-example.py` (all four status buckets, UNKNOWN signal, `None` rates on held_out, byte-identical re-run) |

**2026-11-15 fixes:** (a) `report.py` `__post_init__` reassigned frozen fields
(`FrozenInstanceError` on every construction) — now validate-only; (b)
`test_state_eval_only_for_scored_samples` used the default `source="test"`
where the recording adapter's documented stand-in contract expects the sample
id — test setup corrected, implementation unchanged.

**2026-11-15 (second pass — the lint gates were red at the "done" claim):**
`ruff check src tests` reported 44 errors and `ruff format --check` 7
unformatted files, *all* in the new evaluator files (pre-existing code clean).
Fixed: auto-fixes (I001 import order ×5, UP035 `collections.abc` imports,
UP017 `datetime.UTC` ×2, UP037 unquoted annotations ×2, F541, F632) +
`ruff format` on 8 files; manual: B008 (`ScoringConfig()` default →
module-level `_DEFAULT_CONFIG` singleton in `dataset.py`), B905
(`zip(..., strict=True)` ×3 — all same-length pairs by construction), B017
(`pytest.raises(Exception)` → `dataclasses.FrozenInstanceError` ×2 —
exception narrowed, not weakened), one E501 f-string split.

**Verification gates (re-run 2026-11-15, after the fixes):**
`uv run --extra dev python -m pytest -q` → **1099 passed, 17 skipped**
(skips are pre-existing environment conditions: POSIX-only fd test,
live-console CONIN$ ×2, symlink privilege, …); `ruff check src tests` →
**0 errors**; `ruff format --check` on all 11 touched files + example →
**all formatted**; `git diff --check` → clean;
`docs/eh4-evaluator-example.py` → **runs end-to-end** (all four status
buckets, UNKNOWN signal, `None` rates on held_out, byte-identical `to_json`
across two runs).

**Not yet:** commit/push (requires explicit authorization per §1). Final
§14 report delivered in-session 2026-11-15 with this evidence.

**PR #15 hardening (2026-11-15, post-review fixes — uncommitted):**

- **F1** value errors were aligned to the class-iteration order, not to the
  match order — `_sample_category` pairs `value_errors[i]` with the
  *sorted* match `i`, so errors could attach to the wrong zone when a
  class's key order differs from its expected-index order. Fixed:
  `scoring._score_class_partitioned` now keys value errors by the global
  `(expected_index, predicted_index)` pair and derives the tuple from the
  sorted match list. Regression tests:
  `test_score_ui_zones_value_errors_align_with_matches_across_classes`,
  `test_sample_report_value_error_belongs_to_the_matched_zone`.
- **F2** `class_key_string` joined parts with `:`, so
  `("health:bar", "brightness")` collided with
  `("health", "bar:brightness")` — one report class bucket each. Fixed:
  the encoding is now the canonical-JSON list of the stringified parts
  (injective; part boundaries survive decoding). `EVALUATOR_VERSION`
  bumped 1 → 2 — v1 keys are not valid v2 keys, so readers must handle it.
  Regression tests: `test_class_key_string_is_injective_across_part_boundaries`,
  `test_class_key_string_round_trips_part_boundaries`,
  `test_evaluator_version_pinned_after_class_key_encoding_change`.
- **F3** the frozen report dataclasses still held the caller's *mutable*
  dicts (`CategoryMetrics.classes`, `SplitReport.categories`,
  `EvaluationReport.splits`): a later mutation of the passed-in dict
  changed what the report held and serialized. Fixed: each mapping field is
  defensively copied in `__post_init__` and exposed as a
  `MappingProxyType` view (constructor-time normalization via
  `object.__setattr__` — the only way to set a field on a frozen
  dataclass). Regression tests: `test_category_metrics_does_not_alias_caller_dict`,
  `test_category_metrics_classes_view_rejects_mutation`,
  `test_split_report_does_not_alias_caller_categories`,
  `test_evaluation_report_does_not_alias_caller_splits`.
- **Stateful split semantics (pinned):** one adapter instance spans both
  splits, fed in manifest order with no reset — a stateful machine
  evaluates held-out samples with its post-calibration history (intended;
  held-out state results are not independent). Documented in §6,
  `dataset.py`, `state_eval.py`. Regression test:
  `test_stateful_adapter_is_shared_across_splits_in_manifest_order`.

**Not yet:** commit/push of the hardening (requires explicit authorization
per §1).
