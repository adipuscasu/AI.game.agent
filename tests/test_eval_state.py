"""EH4 step 2: state-evaluation contract — expectation vs. actual classification.

Covers the §6 contract:

* Per-signal verdict matrix (TP / TN / FP / FN / **UNKNOWN — never false**).
* State outcome ``correct`` / ``incorrect`` (with the confusion pair) /
  ``unknown`` (adapter could not determine).
* Reason compared *only* when the expectation supplies one, exact equality.
* The injectable ``StateEvaluator`` protocol (structural — no WoW imports).
* Deterministic, order-preserving, and pure aggregation of per-sample verdicts.
"""

from __future__ import annotations

import dataclasses
import datetime

import pytest

from ai_game_agent.evaluation.manifest import StateExpectation
from ai_game_agent.evaluation.state_eval import (
    SignalVerdict,
    StateAggregate,
    StateEvaluation,
    StateEvaluator,
    StateVerdict,
    aggregate_state,
    classify_state,
)
from ai_game_agent.perception.observation import Observation


def _obs() -> Observation:
    return Observation(
        frame_width=100,
        frame_height=100,
        captured_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        source="test",
        templates=(),
        ui_zones=(),
        text_regions=(),
        objects=(),
        detector_errors=(),
    )


def _expectation(
    signals: dict[str, bool], state: str = "IDLE", reason: str | None = None
) -> StateExpectation:
    return StateExpectation(signals=signals, state=state, reason=reason)


# ---- StateEvaluation (the adapter's actual output) ---------------------------


def test_state_evaluation_defaults() -> None:
    ev = StateEvaluation(signals={})
    assert ev.state is None
    assert ev.reason is None
    assert dict(ev.signals) == {}


def test_state_evaluation_is_frozen() -> None:
    ev = StateEvaluation(signals={"a": True}, state="IDLE")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.state = "DEAD"  # type: ignore[misc]


def test_state_evaluation_rejects_non_mapping_signals() -> None:
    with pytest.raises(ValueError):
        StateEvaluation(signals=[("a", True)])  # type: ignore[arg-type]


def test_state_evaluation_rejects_non_bool_signal_values() -> None:
    with pytest.raises(ValueError):
        StateEvaluation(signals={"a": "yes"})  # type: ignore[dict-item]


# ---- Per-signal verdict matrix ------------------------------------------------


def _classify(signals: dict[str, bool], actual_signals: dict[str, bool | None]) -> StateVerdict:
    return classify_state(
        _expectation(signals),
        StateEvaluation(signals=actual_signals),
    )


def test_signal_true_true_is_tp() -> None:
    v = _classify({"target": True}, {"target": True})
    assert v.signals[0].verdict == "TP"


def test_signal_false_false_is_tn() -> None:
    v = _classify({"target": False}, {"target": False})
    assert v.signals[0].verdict == "TN"


def test_signal_true_false_is_fn() -> None:
    v = _classify({"target": True}, {"target": False})
    assert v.signals[0].verdict == "FN"


def test_signal_false_true_is_fp() -> None:
    v = _classify({"target": False}, {"target": True})
    assert v.signals[0].verdict == "FP"


def test_signal_actual_none_is_unknown() -> None:
    # An explicit ``None`` is *unknown*, and must be counted as unknown, not FN.
    v = _classify({"target": True}, {"target": None})
    assert v.signals[0].verdict == "UNKNOWN"
    assert v.signals[0].actual is None


def test_signal_missing_key_is_unknown() -> None:
    # A signal the adapter did not report at all is also unknown, never false.
    v = _classify({"target": True}, {})
    assert v.signals[0].verdict == "UNKNOWN"


def test_signal_negative_expected_missing_actual_is_unknown() -> None:
    v = _classify({"loot": False}, {})
    assert v.signals[0].verdict == "UNKNOWN"


def test_multiple_signals_classified_independently() -> None:
    v = _classify(
        {"target": True, "loot": False, "dead": True},
        {"target": True, "loot": True, "dead": None},
    )
    assert [s.verdict for s in v.signals] == ["TP", "FP", "UNKNOWN"]
    assert [s.name for s in v.signals] == ["target", "loot", "dead"]


def test_signal_verdict_records_expected_and_actual() -> None:
    v = _classify({"target": True}, {"target": False})
    s = v.signals[0]
    assert s.name == "target"
    assert s.expected is True
    assert s.actual is False
    assert s.verdict == "FN"


def test_extra_actual_signals_are_not_scored() -> None:
    # We score only what the expectation declares; extra actuals are ignored.
    v = _classify({"target": True}, {"target": True, "surprise": True})
    assert len(v.signals) == 1
    assert v.signals[0].verdict == "TP"


# ---- State outcome -----------------------------------------------------------


def test_state_correct_when_equal() -> None:
    v = classify_state(
        _expectation({"a": True}, state="TARGETING"),
        StateEvaluation(signals={"a": True}, state="TARGETING"),
    )
    assert v.state == "correct"
    assert v.expected_state == "TARGETING"
    assert v.actual_state == "TARGETING"


def test_state_incorrect_records_confusion_pair() -> None:
    v = classify_state(
        _expectation({"a": True}, state="TARGETING"),
        StateEvaluation(signals={"a": True}, state="IDLE"),
    )
    assert v.state == "incorrect"
    assert v.expected_state == "TARGETING"
    assert v.actual_state == "IDLE"


def test_state_unknown_when_adapter_returns_none() -> None:
    v = classify_state(
        _expectation({"a": True}, state="TARGETING"),
        StateEvaluation(signals={"a": True}),  # state omitted → None
    )
    assert v.state == "unknown"
    assert v.expected_state == "TARGETING"
    assert v.actual_state is None


def test_state_case_sensitive() -> None:
    # State names are stable identifiers, not free text — casing must matter.
    v = classify_state(
        _expectation({"a": True}, state="targeting"),
        StateEvaluation(signals={"a": True}, state="TARGETING"),
    )
    assert v.state == "incorrect"


# ---- Reason comparison -------------------------------------------------------


def test_reason_not_compared_when_unexpected() -> None:
    v = classify_state(
        _expectation({"a": True}, state="IDLE", reason=None),
        StateEvaluation(signals={"a": True}, state="IDLE", reason="whatever"),
    )
    assert v.reason_match is None


def test_reason_match_when_exact() -> None:
    v = classify_state(
        _expectation({"a": True}, state="IDLE", reason="target-detected"),
        StateEvaluation(signals={"a": True}, state="IDLE", reason="target-detected"),
    )
    assert v.reason_match is True


def test_reason_mismatch_when_differs() -> None:
    v = classify_state(
        _expectation({"a": True}, state="IDLE", reason="target-detected"),
        StateEvaluation(signals={"a": True}, state="IDLE", reason="loot-detected"),
    )
    assert v.reason_match is False


def test_reason_mismatch_when_actual_absent() -> None:
    v = classify_state(
        _expectation({"a": True}, state="IDLE", reason="target-detected"),
        StateEvaluation(signals={"a": True}, state="IDLE"),  # reason None
    )
    assert v.reason_match is False


def test_reason_comparison_is_exact_not_casefolded() -> None:
    v = classify_state(
        _expectation({"a": True}, state="IDLE", reason="target-detected"),
        StateEvaluation(signals={"a": True}, state="IDLE", reason="TARGET-DETECTED"),
    )
    assert v.reason_match is False


def test_reason_independent_of_state_outcome() -> None:
    # A wrong state can still have a matching reason, and vice versa.
    v = classify_state(
        _expectation({"a": True}, state="IDLE", reason="r"),
        StateEvaluation(signals={"a": True}, state="DEAD", reason="r"),
    )
    assert v.state == "incorrect"
    assert v.reason_match is True


# ---- Protocol shape ----------------------------------------------------------


def test_state_evaluator_protocol_is_structural() -> None:
    class _Adapter:
        def evaluate(self, observation: Observation) -> StateEvaluation:
            return StateEvaluation(signals={"target": True}, state="IDLE")

    adapter: StateEvaluator = _Adapter()  # type: ignore[assignment]
    result = adapter.evaluate(_obs())
    assert result.state == "IDLE"
    assert dict(result.signals) == {"target": True}


# ---- Aggregation -------------------------------------------------------------


def _verdict(
    verdicts: list[str],
    state: str = "correct",
    reason_match: bool | None = True,
) -> StateVerdict:
    signals = tuple(
        SignalVerdict(name=f"s{i}", expected=True, actual=True, verdict=v)
        for i, v in enumerate(verdicts)
    )
    return StateVerdict(
        signals=signals,
        state=state,
        expected_state="IDLE",
        actual_state="IDLE" if state != "unknown" else None,
        reason_match=reason_match,
    )


def test_aggregate_empty() -> None:
    a = aggregate_state([])
    assert a == StateAggregate(
        samples_evaluated=0,
        signal_tp=0,
        signal_tn=0,
        signal_fp=0,
        signal_fn=0,
        signal_unknown=0,
        state_correct=0,
        state_incorrect=0,
        state_unknown=0,
        reason_matches=0,
        reason_mismatches=0,
    )


def test_aggregate_counts_signals_states_and_reasons() -> None:
    a = aggregate_state(
        [
            _verdict(["TP", "FN"], state="correct", reason_match=True),
            _verdict(["FP"], state="incorrect", reason_match=False),
            _verdict(["UNKNOWN"], state="unknown", reason_match=None),
        ]
    )
    assert a.samples_evaluated == 3
    assert (a.signal_tp, a.signal_tn, a.signal_fp, a.signal_fn, a.signal_unknown) == (
        1,
        0,
        1,
        1,
        1,
    )
    assert (a.state_correct, a.state_incorrect, a.state_unknown) == (1, 1, 1)
    assert a.reason_matches == 1
    assert a.reason_mismatches == 1


def test_aggregate_deterministic() -> None:
    vs = [_verdict(["TP", "TN"]), _verdict(["FP"]), _verdict(["FN"])]
    assert aggregate_state(vs) == aggregate_state(list(vs))
