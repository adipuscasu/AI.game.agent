"""EH4 step 2: the deterministic evaluation report (workplan §8).

Contract under test:

* frozen value objects — construction validates, ``__post_init__`` must not
  attempt to reassign frozen fields (a report must be *buildable*);
* zero denominators serialize as ``null`` (``None``), never a manufactured
  rate;
* class maps emit **sorted by key** regardless of insertion order;
* non-finite floats are rejected at construction;
* bucket invariants: ``Counts.total == scored + incomplete + skipped +
  missing`` and ``counts`` must agree with the per-sample status rows;
* a scored sample carries all four categories; an excluded sample carries
  none;
* a not-performed state section must have ``samples_evaluated == 0``;
* ``to_json()`` is byte-identical for identical inputs (fixed key order,
  fixed separators, ``ensure_ascii=True``), parses back, and contains no
  ``NaN``/``Infinity`` tokens.
"""

from __future__ import annotations

import json

import pytest

from ai_game_agent.evaluation.manifest import SPLITS
from ai_game_agent.evaluation.report import (
    CATEGORIES,  # re-exported from scoring via report
    EVALUATOR_VERSION,
    REPORT_SCHEMA,
    CategoryMetrics,
    Counts,
    EvaluationReport,
    MatchDetail,
    SampleCategory,
    SampleReport,
    SplitReport,
    StateSection,
)
from ai_game_agent.evaluation.scoring import compute_metrics
from ai_game_agent.evaluation.state_eval import (
    StateAggregate,
    StateVerdict,
)


def _metrics(tp: int = 0, fp: int = 0, fn: int = 0) -> object:
    return compute_metrics(
        tp=tp,
        fp=fp,
        fn=fn,
        expected_total=tp + fn,
        predicted_total=tp + fp,
    )


def _zero() -> object:
    """All-zero metrics — every rate is ``None`` (zero denominator)."""
    return _metrics(0, 0, 0)


def _state_section(performed: bool = False) -> StateSection:
    aggregate = StateAggregate(
        samples_evaluated=1 if performed else 0,
        signal_tp=1 if performed else 0,
        signal_tn=0,
        signal_fp=0,
        signal_fn=0,
        signal_unknown=0,
        state_correct=1 if performed else 0,
        state_incorrect=0,
        state_unknown=0,
        reason_matches=0,
        reason_mismatches=0,
    )
    return StateSection(performed=performed, aggregate=aggregate)


def _sample_report(status: str = "scored") -> SampleReport:
    if status == "scored":
        categories = tuple(
            SampleCategory(
                category=c,
                tp=0,
                fp=0,
                fn=0,
                matches=(),
                false_positives=(),
                false_negatives=(),
            )
            for c in CATEGORIES
        )
    else:
        categories = ()
    return SampleReport(
        id="s1",
        split="calibration",
        status=status,
        detector_errors=(),
        categories=categories,
    )


def _split_report(split: str = "calibration") -> SplitReport:
    return SplitReport(
        split=split,
        overall=_zero(),
        categories={
            c: CategoryMetrics(category=c, overall=_zero(), classes={}) for c in CATEGORIES
        },
        state=_state_section(),
    )


def _report(counts: Counts, samples: tuple[SampleReport, ...]) -> EvaluationReport:
    return EvaluationReport(
        dataset="ds",
        dataset_version="1.0",
        manifest_schema_version=1,
        iou_threshold=0.5,
        counts=counts,
        splits={split: _split_report(split) for split in SPLITS},
        samples=samples,
    )


# ---- Construction: frozen dataclasses must be buildable ----------------------


def test_match_detail_is_constructible() -> None:
    d = MatchDetail(iou=1.0, expected="t", predicted="t")
    assert d.iou == 1.0
    assert d.expected == "t"
    assert d.value_error is None


def test_match_detail_with_value_error_is_constructible() -> None:
    d = MatchDetail(iou=0.8, expected="zone", predicted="zone", value_error=0.25)
    assert d.value_error == 0.25


def test_match_detail_rejects_out_of_range_iou() -> None:
    with pytest.raises(ValueError):
        MatchDetail(iou=1.5, expected="t", predicted="t")


def test_match_detail_rejects_non_finite_iou() -> None:
    with pytest.raises(ValueError):
        MatchDetail(iou=float("nan"), expected="t", predicted="t")
    with pytest.raises(ValueError):
        MatchDetail(iou=float("inf"), expected="t", predicted="t")


def test_match_detail_rejects_negative_value_error() -> None:
    with pytest.raises(ValueError):
        MatchDetail(iou=1.0, expected="t", predicted="t", value_error=-0.1)


def test_counts_is_constructible() -> None:
    c = Counts(scored=1, incomplete=0, skipped=0, missing=0, total=1)
    assert c.scored == 1
    assert c.total == 1


def test_counts_rejects_total_mismatch() -> None:
    with pytest.raises(ValueError, match="total"):
        Counts(scored=1, incomplete=0, skipped=0, missing=0, total=5)


def test_counts_rejects_negative() -> None:
    with pytest.raises(ValueError):
        Counts(scored=-1, incomplete=0, skipped=0, missing=0, total=0)


def test_counts_rejects_bool() -> None:
    with pytest.raises(ValueError):
        Counts(scored=True, incomplete=0, skipped=0, missing=0, total=1)


def test_sample_category_enforces_tp_equals_matches() -> None:
    with pytest.raises(ValueError, match="tp"):
        SampleCategory(
            category="templates",
            tp=1,
            fp=0,
            fn=0,
            matches=(),
            false_positives=(),
            false_negatives=(),
        )


def test_sample_category_rejects_unknown_category() -> None:
    with pytest.raises(ValueError):
        SampleCategory(
            category="bogus", tp=0, fp=0, fn=0, matches=(), false_positives=(), false_negatives=()
        )


def test_sample_report_scored_requires_all_categories() -> None:
    with pytest.raises(ValueError):
        SampleReport(
            id="s1",
            split="calibration",
            status="scored",
            detector_errors=(),
            categories=(),  # scored but no categories -> construction bug
        )


def test_sample_report_excluded_must_not_carry_categories() -> None:
    with pytest.raises(ValueError):
        SampleReport(
            id="s1",
            split="calibration",
            status="missing",
            detector_errors=(),
            categories=(
                SampleCategory(
                    category="templates",
                    tp=0,
                    fp=0,
                    fn=0,
                    matches=(),
                    false_positives=(),
                    false_negatives=(),
                ),
            ),
        )


def test_state_section_not_performed_must_have_zero_samples() -> None:
    aggregate = StateAggregate(
        samples_evaluated=3,
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
    with pytest.raises(ValueError, match="samples_evaluated"):
        StateSection(performed=False, aggregate=aggregate)


def test_evaluation_report_counts_must_match_sample_rows() -> None:
    with pytest.raises(ValueError, match="counts"):
        _report(Counts(scored=1, incomplete=0, skipped=0, missing=0, total=1), ())


def test_evaluation_report_rejects_bad_threshold() -> None:
    for bad in (1.5, -0.1, float("nan"), True):
        with pytest.raises(ValueError):
            EvaluationReport(
                dataset="ds",
                dataset_version="1.0",
                manifest_schema_version=1,
                iou_threshold=bad,
                counts=Counts(0, 0, 0, 0, 0),
                splits={split: _split_report(split) for split in SPLITS},
                samples=(),
            )


def test_evaluation_report_rejects_nonpositive_schema_version() -> None:
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            EvaluationReport(
                dataset="ds",
                dataset_version="1.0",
                manifest_schema_version=bad,
                iou_threshold=0.5,
                counts=Counts(0, 0, 0, 0, 0),
                splits={split: _split_report(split) for split in SPLITS},
                samples=(),
            )


# ---- Mapping fields: decoupled from the caller's dict (F3) ------------------
#
# The report is a value object: once constructed, later mutation of the dict
# the caller passed in must not change what the report holds or serializes,
# and the exposed views must reject writes.


def test_category_metrics_does_not_alias_caller_dict() -> None:
    classes = {"a": _metrics(1, 0, 0)}
    cm = CategoryMetrics(category="templates", overall=_zero(), classes=classes)
    classes.clear()  # external mutation after construction
    classes["x"] = _metrics(9, 9, 9)
    assert set(cm.classes) == {"a"}
    assert cm.classes["a"] == _metrics(1, 0, 0)
    assert cm.to_dict()["classes"]["a"]["tp"] == 1


def test_category_metrics_classes_view_rejects_mutation() -> None:
    cm = CategoryMetrics(category="templates", overall=_zero(), classes={"a": _zero()})
    with pytest.raises(TypeError):
        cm.classes["b"] = _zero()
    with pytest.raises(TypeError):
        del cm.classes["a"]


def test_split_report_does_not_alias_caller_categories() -> None:
    categories = {c: CategoryMetrics(category=c, overall=_zero(), classes={}) for c in CATEGORIES}
    sr = SplitReport(
        split="calibration", overall=_zero(), categories=categories, state=_state_section()
    )
    categories.clear()  # external mutation after construction
    categories["bogus"] = None
    assert set(sr.categories) == set(CATEGORIES)
    with pytest.raises(TypeError):
        sr.categories["bogus"] = None


def test_evaluation_report_does_not_alias_caller_splits() -> None:
    splits = {split: _split_report(split) for split in SPLITS}
    rep = EvaluationReport(
        dataset="ds",
        dataset_version="1.0",
        manifest_schema_version=1,
        iou_threshold=0.5,
        counts=Counts(0, 0, 0, 0, 0),
        splits=splits,
        samples=(),
    )
    splits["calibration"] = _split_report("held_out")  # swap after construction
    splits["held_out"] = None
    assert set(rep.splits) == set(SPLITS)
    assert rep.splits["calibration"].split == "calibration"
    with pytest.raises(TypeError):
        rep.splits["calibration"] = None


# ---- to_dict / to_json shape -------------------------------------------------


def test_to_dict_has_fixed_key_order() -> None:
    report = _report(Counts(1, 0, 0, 0, 1), (_sample_report("scored"),))
    d = report.to_dict()
    assert list(d) == [
        "schema",
        "evaluator_version",
        "manifest_schema_version",
        "dataset",
        "dataset_version",
        "scoring",
        "counts",
        "splits",
        "samples",
    ]
    assert d["schema"] == REPORT_SCHEMA
    assert d["evaluator_version"] == EVALUATOR_VERSION
    assert d["scoring"] == {"iou_threshold": 0.5}


def test_evaluator_version_pinned_after_class_key_encoding_change() -> None:
    # F2: the per-class key encoding changed from `:`-join to an injective
    # canonical form — a report-shape change readers must handle, so the
    # version is pinned here (v1 keys are not valid v2 keys).
    assert EVALUATOR_VERSION == 2


def test_to_dict_emits_both_splits() -> None:
    report = _report(Counts(0, 0, 0, 0, 0), ())
    d = report.to_dict()
    assert set(d["splits"]) == set(SPLITS)
    for split in SPLITS:
        assert d["splits"][split]["overall"]["precision"] is None
        assert d["splits"][split]["overall"]["recall"] is None
        assert d["splits"][split]["overall"]["f1"] is None


def test_class_maps_emit_sorted_by_key() -> None:
    cm = CategoryMetrics(
        category="templates",
        overall=_zero(),
        classes={"zeta": _zero(), "alpha": _zero(), "mid": _zero()},
    )
    d = cm.to_dict()
    assert list(d["classes"]) == ["alpha", "mid", "zeta"]


def test_state_section_dict_shape() -> None:
    d = _state_section(performed=True).to_dict()
    assert d["performed"] is True
    assert d["samples_evaluated"] == 1
    assert d["signals"] == {"TP": 1, "TN": 0, "FP": 0, "FN": 0, "UNKNOWN": 0}
    assert d["states"] == {"correct": 1, "incorrect": 0, "unknown": 0}
    assert d["reason_matches"] == 0
    assert d["reason_mismatches"] == 0


def test_state_sample_none_serializes_as_not_evaluated() -> None:
    d = _sample_report("scored").to_dict()
    assert d["state"] == {"evaluated": False, "state": "not_evaluated"}


def test_state_sample_verdict_serializes() -> None:
    from ai_game_agent.evaluation.state_eval import SignalVerdict

    verdict = StateVerdict(
        signals=(SignalVerdict(name="target", expected=True, actual=True, verdict="TP"),),
        state="correct",
        expected_state="COMBAT",
        actual_state="COMBAT",
        reason_match=None,
    )
    s = SampleReport(
        id="s1",
        split="calibration",
        status="scored",
        detector_errors=(),
        categories=tuple(
            SampleCategory(
                category=c, tp=0, fp=0, fn=0, matches=(), false_positives=(), false_negatives=()
            )
            for c in CATEGORIES
        ),
        state=verdict,
    )
    d = s.to_dict()
    assert d["state"]["evaluated"] is True
    assert d["state"]["state"] == "correct"
    assert d["state"]["signals"][0]["verdict"] == "TP"


# ---- Determinism / JSON ------------------------------------------------------


def _full_report() -> EvaluationReport:
    return _report(
        Counts(2, 1, 1, 1, 5),
        (
            _sample_report("scored"),
            _sample_report("scored"),
            _sample_report("incomplete"),
            _sample_report("skipped"),
            _sample_report("missing"),
        ),
    )


def test_to_json_is_byte_identical_across_runs() -> None:
    a = _full_report().to_json()
    b = _full_report().to_json()
    assert a == b


def test_to_json_round_trips_and_has_no_non_finite_tokens() -> None:
    text = _full_report().to_json()
    assert "NaN" not in text
    assert "Infinity" not in text
    d = json.loads(text)
    assert d["counts"]["total"] == 5
    assert d["counts"]["scored"] == 2
    # None -> null in the JSON text, not the string "None".
    assert '"precision":null' in text


def test_to_json_is_ascii_only() -> None:
    text = _full_report().to_json()
    assert all(ord(ch) < 128 for ch in text)


def test_to_json_escapes_non_ascii_input() -> None:
    report = _report(Counts(0, 0, 0, 0, 0), ())
    # Rebuild with a non-ASCII dataset name to exercise ensure_ascii.
    report = EvaluationReport(
        dataset="désètnâme",
        dataset_version="1.0",
        manifest_schema_version=1,
        iou_threshold=0.5,
        counts=Counts(0, 0, 0, 0, 0),
        splits={split: _split_report(split) for split in SPLITS},
        samples=(),
    )
    text = report.to_json()
    assert all(ord(ch) < 128 for ch in text)
    assert json.loads(text)["dataset"] == "désètnâme"
