"""EH4 step 2: dataset evaluation API — statuses, splits, state, determinism.

Covers the §4–§7 contract: scored/incomplete/skipped/missing buckets,
split isolation, foreign-id rejection, no-adapter ``not_performed``,
state-evaluation gating, stateful-adapter manifest order, and non-mutation.
"""

from __future__ import annotations

import datetime
import json

import pytest

from ai_game_agent.evaluation.dataset import evaluate_dataset
from ai_game_agent.evaluation.manifest import (
    CaptureMetadata,
    DatasetManifest,
    ManifestError,
    Sample,
    StateExpectation,
)
from ai_game_agent.evaluation.report import EvaluationReport
from ai_game_agent.evaluation.state_eval import StateEvaluation
from ai_game_agent.perception.observation import BBox, Observation, TemplateHit, UiZoneHit

_DT = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def _obs(
    templates=(),
    ui_zones=(),
    text_regions=(),
    objects=(),
    errors=(),
    source: str = "test",
) -> Observation:
    return Observation(
        frame_width=100,
        frame_height=100,
        captured_at=_DT,
        source=source,
        templates=tuple(templates),
        ui_zones=tuple(ui_zones),
        text_regions=tuple(text_regions),
        objects=tuple(objects),
        detector_errors=tuple(errors),
    )


def _template(name: str, x: int = 0) -> TemplateHit:
    return TemplateHit(template=name, bbox=BBox(x=x, y=0, width=10, height=10), confidence=0.9)


def _sample(
    sid: str,
    label: Observation,
    split: str = "calibration",
    state: StateExpectation | None = None,
) -> Sample:
    return Sample(id=sid, image=f"{sid}.png", split=split, detections=label, state=state)


def _manifest(samples) -> DatasetManifest:
    return DatasetManifest(
        dataset="ds",
        dataset_version="1.0",
        created="2026-01-01",
        capture=CaptureMetadata(100, 100, "windowed", 1.0),
        samples=tuple(samples),
    )


# ---- Sample statuses ---------------------------------------------------------


def test_scored_sample_is_scored() -> None:
    m = _manifest([_sample("a", _obs(templates=[_template("t")]))])
    report = evaluate_dataset(m, {"a": _obs(templates=[_template("t")])})
    assert report.samples[0].status == "scored"
    assert report.counts.scored == 1
    assert report.counts.total == 1


def test_missing_prediction_is_missing_bucket() -> None:
    m = _manifest([_sample("a", _obs(templates=[_template("t")]))])
    report = evaluate_dataset(m, {})
    assert report.samples[0].status == "missing"
    assert report.counts.missing == 1


def test_prediction_with_detector_errors_is_incomplete() -> None:
    m = _manifest([_sample("a", _obs(templates=[_template("t")]))])
    report = evaluate_dataset(m, {"a": _obs(templates=[_template("t")], errors=["ocr crashed"])})
    assert report.samples[0].status == "incomplete"
    assert report.counts.incomplete == 1
    assert report.samples[0].detector_errors == ("ocr crashed",)


def test_label_with_detector_errors_is_skipped() -> None:
    m = _manifest([_sample("a", _obs(templates=[_template("t")], errors=["label bad"]))])
    report = evaluate_dataset(m, {"a": _obs(templates=[_template("t")])})
    assert report.samples[0].status == "skipped"
    assert report.counts.skipped == 1
    assert report.samples[0].detector_errors == ("label bad",)


def test_missing_takes_precedence_over_label_errors() -> None:
    m = _manifest([_sample("a", _obs(errors=["label bad"]))])
    report = evaluate_dataset(m, {})
    assert report.samples[0].status == "missing"


def test_skipped_takes_precedence_over_incomplete() -> None:
    m = _manifest([_sample("a", _obs(errors=["label bad"]))])
    report = evaluate_dataset(m, {"a": _obs(errors=["pred bad"])})
    assert report.samples[0].status == "skipped"
    assert report.samples[0].detector_errors == ("pred bad", "label bad")


def test_excluded_samples_do_not_enter_metrics() -> None:
    # One clean scored TP plus one missing sample that would otherwise be FN:
    # recall must be 1.0 over scored evidence, not 0.5.
    m = _manifest(
        [
            _sample("a", _obs(templates=[_template("t")])),
            _sample("b", _obs(templates=[_template("t")])),
        ]
    )
    report = evaluate_dataset(m, {"a": _obs(templates=[_template("t")])})
    overall = report.splits["calibration"].overall
    assert overall.tp == 1 and overall.fn == 0 and overall.fp == 0
    assert overall.recall == 1.0


# ---- Split isolation ---------------------------------------------------------


def test_splits_reported_separately() -> None:
    m = _manifest(
        [
            _sample("a", _obs(templates=[_template("t")]), split="calibration"),
            _sample("b", _obs(templates=[_template("t")]), split="held_out"),
        ]
    )
    # calibration: TP. held_out: no prediction (missing evidence).
    report = evaluate_dataset(m, {"a": _obs(templates=[_template("t")])})
    cal = report.splits["calibration"]
    ho = report.splits["held_out"]
    assert cal.overall.tp == 1 and cal.overall.fn == 0
    # The only held_out sample is *missing*: per the incomplete-evidence
    # policy (workplan §4) it is excluded from metrics, never counted as a
    # negative — the split shows no scored evidence at all.
    assert ho.overall.tp == 0 and ho.overall.fp == 0 and ho.overall.fn == 0
    assert cal.overall.recall == 1.0
    # held_out has no scored evidence: every rate is None — never a
    # manufactured 0.0 (zero denominator).
    assert ho.overall.precision is None
    assert ho.overall.recall is None
    # No combined-across-splits section exists.
    assert set(report.to_dict()) == {
        "schema",
        "evaluator_version",
        "manifest_schema_version",
        "dataset",
        "dataset_version",
        "scoring",
        "counts",
        "splits",
        "samples",
    }


def test_sample_order_is_manifest_order() -> None:
    m = _manifest(
        [
            _sample("zz", _obs()),
            _sample("aa", _obs(), split="held_out"),
            _sample("mm", _obs()),
        ]
    )
    report = evaluate_dataset(m, {})
    assert [s.id for s in report.samples] == ["zz", "aa", "mm"]


# ---- Input validation --------------------------------------------------------


def test_foreign_prediction_ids_rejected() -> None:
    m = _manifest([_sample("a", _obs())])
    try:
        evaluate_dataset(m, {"a": _obs(), "ghost": _obs()})
        raise AssertionError("expected ValueError for foreign prediction id")
    except ValueError as exc:
        assert "ghost" in str(exc)


def test_multiple_foreign_ids_all_listed() -> None:
    m = _manifest([_sample("a", _obs())])
    try:
        evaluate_dataset(m, {"b": _obs(), "c": _obs()})
        raise AssertionError("expected ValueError for foreign prediction ids")
    except ValueError as exc:
        assert "b" in str(exc) and "c" in str(exc)


def test_predictions_must_be_a_mapping() -> None:
    m = _manifest([_sample("a", _obs())])
    try:
        evaluate_dataset(m, ["not", "a", "mapping"])  # type: ignore[arg-type]
        raise AssertionError("expected TypeError for non-mapping predictions")
    except TypeError:
        pass


def test_prediction_values_must_be_observations() -> None:
    m = _manifest([_sample("a", _obs())])
    try:
        evaluate_dataset(m, {"a": "not an observation"})  # type: ignore[dict-item]
        raise AssertionError("expected TypeError for non-Observation prediction")
    except TypeError:
        pass


def test_invalid_manifest_rejected() -> None:
    # Duplicate sample ids fail structural validation.
    m = _manifest([_sample("a", _obs()), _sample("a", _obs())])
    try:
        evaluate_dataset(m, {})
        raise AssertionError("expected ManifestError for invalid manifest")
    except ManifestError:
        pass


def test_returns_evaluation_report() -> None:
    m = _manifest([_sample("a", _obs())])
    report = evaluate_dataset(m, {"a": _obs()})
    assert isinstance(report, EvaluationReport)


# ---- State evaluation --------------------------------------------------------


def _expectation() -> StateExpectation:
    return StateExpectation(signals={"target": True}, state="COMBAT")


class _RecordingMachine:
    """Stateful adapter: emits a fixed sequence, records feed order by
    the prediction's ``source`` field (a stand-in for the sample id)."""

    def __init__(self, outputs) -> None:
        self._outputs = list(outputs)
        self.feed_order: list[str] = []

    def evaluate(self, observation) -> StateEvaluation:
        self.feed_order.append(observation.source)
        return self._outputs.pop(0)


def test_state_evaluated_with_adapter_and_expectation() -> None:
    m = _manifest([_sample("a", _obs(), state=_expectation())])
    adapter = _RecordingMachine(
        [StateEvaluation(signals={"target": True}, state="COMBAT", reason="hit")]
    )
    report = evaluate_dataset(m, {"a": _obs()}, state_evaluator=adapter)
    section = report.splits["calibration"].state
    assert section.performed is True
    assert section.aggregate.samples_evaluated == 1
    assert section.aggregate.signal_tp == 1
    assert section.aggregate.state_correct == 1
    sample = report.samples[0]
    assert sample.state is not None and sample.state.state == "correct"


def test_no_adapter_is_explicitly_not_performed() -> None:
    m = _manifest([_sample("a", _obs(), state=_expectation())])
    report = evaluate_dataset(m, {"a": _obs()})
    section = report.splits["calibration"].state
    assert section.performed is False
    assert section.aggregate.samples_evaluated == 0
    # The sample row is explicit about it, not silently empty.
    row = report.samples[0].to_dict()
    assert report.samples[0].state is None
    assert row["state"] == {"evaluated": False, "state": "not_evaluated"}


def test_adapter_without_expectation_yields_zero_aggregate() -> None:
    m = _manifest([_sample("a", _obs())])
    adapter = _RecordingMachine([StateEvaluation(signals={"target": True}, state="COMBAT")])
    report = evaluate_dataset(m, {"a": _obs()}, state_evaluator=adapter)
    section = report.splits["calibration"].state
    assert section.performed is True
    assert section.aggregate.samples_evaluated == 0
    assert report.samples[0].state is None


def test_unknown_signal_is_its_own_bucket() -> None:
    m = _manifest([_sample("a", _obs(), state=_expectation())])
    adapter = _RecordingMachine([StateEvaluation(signals={"target": None}, state="COMBAT")])
    report = evaluate_dataset(m, {"a": _obs()}, state_evaluator=adapter)
    aggregate = report.splits["calibration"].state.aggregate
    assert aggregate.signal_unknown == 1
    assert aggregate.signal_fn == 0  # unknown never folds into FN


def test_state_eval_only_for_scored_samples() -> None:
    # Missing prediction: nothing to feed the adapter; no state evaluation.
    m = _manifest(
        [
            _sample("a", _obs(), state=_expectation()),
            _sample("b", _obs(), state=_expectation(), split="held_out"),
        ]
    )
    adapter = _RecordingMachine([StateEvaluation(signals={"target": True}, state="COMBAT")])
    # _RecordingMachine records the prediction's ``source`` as a stand-in for
    # the sample id, so the prediction for sample "a" carries source="a".
    report = evaluate_dataset(m, {"a": _obs(source="a")}, state_evaluator=adapter)
    assert report.splits["calibration"].state.aggregate.samples_evaluated == 1
    assert report.splits["held_out"].state.aggregate.samples_evaluated == 0
    assert adapter.feed_order == ["a"]


def test_stateful_adapter_receives_samples_in_manifest_order() -> None:
    obs_a = _obs(source="first")
    obs_b = _obs(source="second")
    m = _manifest(
        [
            _sample("a", _obs(), state=_expectation()),
            _sample("b", _obs(), state=_expectation()),
        ]
    )
    outputs = [
        StateEvaluation(signals={"target": True}, state="COMBAT"),
        StateEvaluation(signals={"target": True}, state="COMBAT"),
    ]
    adapter = _RecordingMachine(outputs)
    evaluate_dataset(m, {"a": obs_a, "b": obs_b}, state_evaluator=adapter)
    assert adapter.feed_order == ["first", "second"]


def test_stateful_adapter_is_shared_across_splits_in_manifest_order() -> None:
    # Pinned semantics (workplan §6): ONE adapter instance spans both splits,
    # fed in manifest order with no reset between them. A machine whose
    # output depends on its own history therefore sees the calibration
    # samples before the held-out ones — the held_out verdict is evaluated
    # with the machine's post-calibration state, not a fresh one.

    class _CountingMachine:
        """Emits state ``"S1"`` on its first call, ``"S2"`` afterwards."""

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, observation) -> StateEvaluation:
            self.calls += 1
            return StateEvaluation(
                signals={"target": True}, state="S1" if self.calls == 1 else "S2"
            )

    m = _manifest(
        [
            _sample("a", _obs(), state=_expectation(), split="calibration"),
            _sample("b", _obs(), state=_expectation(), split="held_out"),
        ]
    )
    adapter = _CountingMachine()
    report = evaluate_dataset(
        m, {"a": _obs(source="a"), "b": _obs(source="b")}, state_evaluator=adapter
    )
    assert adapter.calls == 2  # one shared instance, no reset between splits
    # The calibration sample consumed the first call; the held-out sample
    # saw the machine's history, so it got the second output.
    assert report.samples[0].state is not None
    assert report.samples[1].state is not None
    assert report.samples[0].state.actual_state == "S1"
    assert report.samples[1].state.actual_state == "S2"
    # Both disagree with the expected state (COMBAT) — what the test pins is
    # *which* call each split received, i.e. the shared-history ordering.
    assert report.samples[0].state.state == "incorrect"
    assert report.samples[1].state.state == "incorrect"


def test_stateful_adapter_follows_interleaved_manifest_order() -> None:
    # Pinned semantics: the feed is manifest order, NOT calibration-first —
    # the validator does not require calibration to precede held-out, so a
    # held-out sample may consume the machine's earliest calls. The shared
    # history therefore crosses both sections.

    class _CountingMachine:
        """Records each call's position and emits a state named after it."""

        def __init__(self) -> None:
            self.order: list[tuple[str, int]] = []

        def evaluate(self, observation) -> StateEvaluation:
            self.order.append((observation.source, len(self.order) + 1))
            return StateEvaluation(signals={"target": True}, state=f"S{len(self.order)}")

    m = _manifest(
        [
            _sample("b", _obs(), state=_expectation(), split="held_out"),  # held-out first
            _sample("a", _obs(), state=_expectation(), split="calibration"),
            _sample("c", _obs(), state=_expectation(), split="held_out"),
        ]
    )
    adapter = _CountingMachine()
    report = evaluate_dataset(
        m,
        {"a": _obs(source="a"), "b": _obs(source="b"), "c": _obs(source="c")},
        state_evaluator=adapter,
    )
    # Manifest order, not calibration-first: b, a, c.
    assert adapter.order == [("b", 1), ("a", 2), ("c", 3)]
    by_id = {s.id: s for s in report.samples}
    assert by_id["b"].state is not None and by_id["b"].state.actual_state == "S1"
    assert by_id["a"].state is not None and by_id["a"].state.actual_state == "S2"
    assert by_id["c"].state is not None and by_id["c"].state.actual_state == "S3"
    # One shared adapter, so the sections mix history: held-out holds calls
    # 1 and 3, calibration call 2.
    assert report.splits["held_out"].state.aggregate.samples_evaluated == 2
    assert report.splits["calibration"].state.aggregate.samples_evaluated == 1


def test_two_distinct_ui_zone_classes_survive_into_report() -> None:
    # F2 regression: the *actual* old-encoding collision pair —
    # ("health:bar", "brightness") and ("health", "bar:brightness") both
    # flattened to "health:bar:brightness" under the colon-joined encoding —
    # is scored and reported independently: one matched, one missed, each
    # under its own injective class key in both the object and the
    # serialized JSON.
    from ai_game_agent.evaluation.scoring import class_key_string

    key_a = ("health:bar", "brightness")
    key_b = ("health", "bar:brightness")
    # Guard: under the retired colon-join these two identities collided, so
    # this pair exercises the part-boundary fix — not just any two keys.
    assert ":".join(key_a) == ":".join(key_b)

    def _zone(zone: str, check: str, value: float = 0.5) -> UiZoneHit:
        return UiZoneHit(zone=zone, check=check, bbox=BBox(0, 0, 10, 10), value=value)

    label = _obs(ui_zones=[_zone(*key_a), _zone(*key_b)])
    prediction = _obs(ui_zones=[_zone(*key_a)])  # class A matched, class B missed
    m = _manifest([_sample("a", label)])
    report = evaluate_dataset(m, {"a": prediction})
    assert report.samples[0].status == "scored"

    a_key = class_key_string("ui_zones", key_a)
    b_key = class_key_string("ui_zones", key_b)
    assert a_key != b_key

    classes = report.splits["calibration"].categories["ui_zones"].classes
    assert set(classes) == {a_key, b_key}
    assert classes[a_key].tp == 1 and classes[a_key].fp == 0 and classes[a_key].fn == 0
    assert classes[b_key].tp == 0 and classes[b_key].fp == 0 and classes[b_key].fn == 1

    # The serialized form keeps both classes under distinct keys, with the
    # per-class partition intact.
    data = json.loads(report.to_json())
    json_classes = data["splits"]["calibration"]["categories"]["ui_zones"]["classes"]
    assert set(json_classes) == {a_key, b_key}
    assert json_classes[a_key]["tp"] == 1 and json_classes[a_key]["fn"] == 0
    assert json_classes[b_key]["tp"] == 0 and json_classes[b_key]["fn"] == 1


def test_sample_report_value_error_belongs_to_the_matched_zone() -> None:
    # F1 regression at the report layer: the value_error on each match must
    # be the error of *that* matched zone pair. Two zones whose key order
    # ("a" < "z") is the reverse of their expected-index order (z first).
    def _zone(zone: str, value: float) -> UiZoneHit:
        return UiZoneHit(zone=zone, check="bar", bbox=BBox(0, 0, 10, 10), value=value)

    label = _obs(ui_zones=[_zone("z", 0.5), _zone("a", 0.2)])
    prediction = _obs(ui_zones=[_zone("z", 0.9), _zone("a", 0.1)])
    m = _manifest([_sample("a", label)])
    report = evaluate_dataset(m, {"a": prediction})
    row = report.samples[0]
    assert row.status == "scored"
    zones = next(c for c in row.categories if c.category == "ui_zones")
    assert [m.value_error for m in zones.matches] == [pytest.approx(0.4), pytest.approx(0.1)]


# ---- Determinism and non-mutation ---------------------------------------------


def test_deterministic_json_across_runs() -> None:
    m = _manifest(
        [
            _sample("a", _obs(templates=[_template("t")])),
            _sample("b", _obs(templates=[_template("t")]), split="held_out"),
        ]
    )
    predictions = {
        "a": _obs(templates=[_template("t", x=1)]),
        "b": _obs(templates=[_template("other")]),
    }
    first = evaluate_dataset(m, predictions).to_json()
    second = evaluate_dataset(m, predictions).to_json()
    assert first == second


def test_manifest_and_observations_not_mutated() -> None:
    label = _obs(templates=[_template("t")])
    prediction = _obs(templates=[_template("t", x=1)])
    m = _manifest([_sample("a", label)])
    label_before = label
    pred_before = prediction
    sample_before = m.samples[0]
    evaluate_dataset(m, {"a": prediction})
    assert m.samples[0] is sample_before
    assert m.samples[0].detections == label_before
    assert prediction == pred_before
