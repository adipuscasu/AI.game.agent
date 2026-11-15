"""EH4 step 2: the dataset evaluation API — scoring + state over one manifest.

The contract (workplan §4–§7):

* A sample lands in exactly one bucket:
  - ``missing`` — no prediction for the sample id (missing *evidence*, never
    an empty observation).
  - ``skipped`` — the labelled observation carries detector errors; the pair
    cannot be trusted at all.
  - ``incomplete`` — the prediction carries detector errors; excluded from
    metrics, never counted as a clean negative.
  - ``scored`` — neither side reported errors; the only bucket that enters
    the metrics.
  Precedence when several apply: missing > skipped > incomplete.
* Split isolation: metrics are computed per split and reported separately;
  no combined section exists. This is a measurement tool, not an optimizer —
  held_out is never fed back into tuning here.
* State evaluation: only ``scored`` samples with a state expectation are fed
  to the injected :class:`StateEvaluator`, in manifest order, with no reset
  between samples (adapters may be stateful — callers supply a fresh one per
  run). No adapter ⇒ state sections are explicitly ``performed=False``,
  never a silent zero section.
* Foreign prediction ids are rejected up front, all of them named.
* The manifest and its observations are never mutated; sample order in the
  report is manifest order; identical inputs produce identical reports.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from ai_game_agent.evaluation.manifest import DatasetManifest, ManifestError, validate_manifest
from ai_game_agent.evaluation.report import (
    CategoryMetrics,
    Counts,
    EvaluationReport,
    MatchDetail,
    SampleCategory,
    SampleReport,
    SplitReport,
    StateSection,
)
from ai_game_agent.evaluation.scoring import (
    CATEGORIES,
    CATEGORY_KEY_FUNCTIONS,
    SCORE_FUNCTIONS,
    CategoryScore,
    ScoringConfig,
    class_key_string,
    compute_metrics,
)
from ai_game_agent.evaluation.state_eval import (
    StateEvaluator,
    StateVerdict,
    aggregate_state,
    classify_state,
)
from ai_game_agent.perception.observation import Observation

__all__ = ["evaluate_dataset"]

_SPLITS: tuple[str, ...] = ("calibration", "held_out")
_DEFAULT_CONFIG = ScoringConfig()


def _class_string(category: str, hit: object) -> str:
    return class_key_string(category, CATEGORY_KEY_FUNCTIONS[category](hit))


def _sample_category(
    category: str, score: CategoryScore, expected: tuple, predicted: tuple
) -> SampleCategory:
    details = []
    for i, match in enumerate(score.matches):
        key = _class_string(category, expected[match.expected_index])
        detail = MatchDetail(iou=match.iou, expected=key, predicted=key)
        if score.value_errors is not None:
            detail = MatchDetail(
                iou=match.iou, expected=key, predicted=key, value_error=score.value_errors[i]
            )
        details.append(detail)
    return SampleCategory(
        category=category,
        tp=score.tp,
        fp=len(score.false_positives),
        fn=len(score.false_negatives),
        matches=tuple(details),
        false_positives=score.false_positives,
        false_negatives=score.false_negatives,
    )


def _classify_status(
    label: Observation, prediction: Observation | None
) -> tuple[str, tuple[str, ...]]:
    """The per-sample status and the detector errors the row must carry."""
    if prediction is None:
        return "missing", label.detector_errors
    if label.detector_errors:
        # Both sides' evidence is reported; prediction errors come first.
        return "skipped", (*prediction.detector_errors, *label.detector_errors)
    if prediction.detector_errors:
        return "incomplete", prediction.detector_errors
    return "scored", ()


def _score_categories(
    label: Observation, prediction: Observation, config: ScoringConfig
) -> list[CategoryScore]:
    return [
        SCORE_FUNCTIONS[category](getattr(label, category), getattr(prediction, category), config)
        for category in CATEGORIES
    ]


def _scored_sample_ids(
    manifest: DatasetManifest, predictions: Mapping[str, Observation]
) -> set[str]:
    """The ids that pass the incomplete/skipped rules (a prediction exists and
    neither side reports detector errors)."""
    scored = set()
    for sample in manifest.samples:
        prediction = predictions.get(sample.id)
        if prediction is None or sample.detections.detector_errors or prediction.detector_errors:
            continue
        scored.add(sample.id)
    return scored


def _aggregate_split(
    manifest: DatasetManifest,
    predictions: Mapping[str, Observation],
    scored_ids: set[str],
    config: ScoringConfig,
    split_name: str,
) -> tuple[object, dict[str, CategoryMetrics]]:
    """Per-split overall metrics and per-category/per-class metrics.

    Only ``scored`` samples of this split contribute. The class buckets are
    ``[tp, fp, fn, expected_total, predicted_total]`` keyed by the report's
    class-identity string, so the per-class partition is exactly the
    matcher's."""
    class_map = {category: {} for category in CATEGORIES}
    totals = [0, 0, 0, 0, 0]  # tp, fp, fn, expected_total, predicted_total

    for sample in manifest.samples:
        if sample.split != split_name or sample.id not in scored_ids:
            continue
        label = sample.detections
        prediction = predictions[sample.id]
        for category in CATEGORIES:
            expected = getattr(label, category)
            predicted = getattr(prediction, category)
            score = SCORE_FUNCTIONS[category](expected, predicted, config)
            exp_class = [_class_string(category, hit) for hit in expected]
            pred_class = [_class_string(category, hit) for hit in predicted]
            buckets = class_map[category]
            for name in exp_class:
                buckets.setdefault(name, [0, 0, 0, 0, 0])[3] += 1
            for name in pred_class:
                buckets.setdefault(name, [0, 0, 0, 0, 0])[4] += 1
            for match in score.matches:
                buckets[exp_class[match.expected_index]][0] += 1
            for index in score.false_positives:
                buckets[pred_class[index]][1] += 1
            for index in score.false_negatives:
                buckets[exp_class[index]][2] += 1
            totals[0] += score.tp
            totals[1] += len(score.false_positives)
            totals[2] += len(score.false_negatives)
            totals[3] += len(expected)
            totals[4] += len(predicted)

    categories: dict[str, CategoryMetrics] = {}
    for category in CATEGORIES:
        buckets = class_map[category]
        categories[category] = CategoryMetrics(
            category=category,
            overall=compute_metrics(
                tp=sum(b[0] for b in buckets.values()),
                fp=sum(b[1] for b in buckets.values()),
                fn=sum(b[2] for b in buckets.values()),
                expected_total=sum(b[3] for b in buckets.values()),
                predicted_total=sum(b[4] for b in buckets.values()),
            ),
            classes={
                name: compute_metrics(
                    tp=b[0], fp=b[1], fn=b[2], expected_total=b[3], predicted_total=b[4]
                )
                for name, b in buckets.items()
            },
        )
    overall = compute_metrics(
        tp=totals[0],
        fp=totals[1],
        fn=totals[2],
        expected_total=totals[3],
        predicted_total=totals[4],
    )
    return overall, categories


def evaluate_dataset(
    manifest: DatasetManifest,
    predictions: Mapping[str, Observation],
    *,
    config: ScoringConfig = _DEFAULT_CONFIG,
    state_evaluator: StateEvaluator | None = None,
) -> EvaluationReport:
    """Evaluate a validated manifest against predictions, split by split.

    ``predictions`` maps sample id → predicted :class:`Observation`; ids not
    present in the manifest are rejected with all of them named.
    ``state_evaluator`` is optional — without it, the report's state sections
    are explicitly ``performed=False``.
    """
    if not isinstance(manifest, DatasetManifest):
        raise TypeError(f"manifest must be a DatasetManifest, got {type(manifest).__name__}")
    problems = validate_manifest(manifest)
    if problems:
        raise ManifestError("\n".join(problems))
    if not isinstance(predictions, Mapping):
        raise TypeError(
            f"predictions must be a mapping of sample id -> Observation, "
            f"got {type(predictions).__name__}"
        )
    for sample_id, prediction in predictions.items():
        if not isinstance(prediction, Observation):
            raise TypeError(
                f"predictions[{sample_id!r}] must be an Observation, "
                f"got {type(prediction).__name__}"
            )

    known = {sample.id for sample in manifest.samples}
    foreign = sorted(set(predictions) - known)
    if foreign:
        raise ValueError(f"unknown prediction ids: {foreign}")

    scored_ids = _scored_sample_ids(manifest, predictions)
    status_tally: Counter[str] = Counter()
    samples: list[SampleReport] = []
    split_verdicts: dict[str, list[StateVerdict]] = {split: [] for split in _SPLITS}

    # One pass in manifest order: statuses, sample rows, and — when enabled —
    # the state feed. The adapter sees samples in manifest order and is never
    # reset between samples.
    for sample in manifest.samples:
        label = sample.detections
        prediction = predictions.get(sample.id)
        status, errors = _classify_status(label, prediction)
        status_tally[status] += 1

        categories: tuple[SampleCategory, ...] = ()
        verdict: StateVerdict | None = None
        if status == "scored":
            scores = _score_categories(label, prediction, config)  # type: ignore[arg-type]
            categories = tuple(
                _sample_category(
                    category, score, getattr(label, category), getattr(prediction, category)
                )  # type: ignore[arg-type]
                for category, score in zip(CATEGORIES, scores, strict=True)
            )
            if state_evaluator is not None and sample.state is not None:
                verdict = classify_state(sample.state, state_evaluator.evaluate(prediction))  # type: ignore[arg-type]
                split_verdicts[sample.split].append(verdict)

        samples.append(
            SampleReport(
                id=sample.id,
                split=sample.split,
                status=status,
                detector_errors=errors,
                categories=categories,
                state=verdict,
            )
        )

    splits: dict[str, SplitReport] = {}
    for split_name in _SPLITS:
        overall, categories_metrics = _aggregate_split(
            manifest, predictions, scored_ids, config, split_name
        )
        if state_evaluator is None:
            state_section = StateSection(performed=False, aggregate=aggregate_state(()))
        else:
            state_section = StateSection(
                performed=True, aggregate=aggregate_state(split_verdicts[split_name])
            )
        splits[split_name] = SplitReport(
            split=split_name, overall=overall, categories=categories_metrics, state=state_section
        )

    counts = Counts(
        scored=status_tally.get("scored", 0),
        incomplete=status_tally.get("incomplete", 0),
        skipped=status_tally.get("skipped", 0),
        missing=status_tally.get("missing", 0),
        total=len(manifest.samples),
    )
    return EvaluationReport(
        dataset=manifest.dataset,
        dataset_version=manifest.dataset_version,
        manifest_schema_version=manifest.schema_version,
        iou_threshold=config.iou_threshold,
        counts=counts,
        splits=splits,
        samples=tuple(samples),
    )
