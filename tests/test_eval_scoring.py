"""EH4 step 2: observation scoring — IoU, matching, and metrics.

Covers the §2–§3 contract: class-isolated one-to-one matching with a
count-then-IoU optimal assignment, deterministic tie-breaking, zero-denominator
metrics as ``None``, ui-zone value errors, and explicit text normalization.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from ai_game_agent.evaluation.scoring import (
    ScoringConfig,
    assign_matches,
    bbox_iou,
    class_key_string,
    compute_metrics,
    merge_metrics,
    normalize_text,
    score_objects,
    score_templates,
    score_text_regions,
    score_ui_zones,
)
from ai_game_agent.perception.observation import BBox, ObjectHit, TemplateHit, TextRegion, UiZoneHit

# ---- IoU ---------------------------------------------------------------------


def test_iou_identical_boxes() -> None:
    box = BBox(x=10, y=20, width=100, height=50)
    assert bbox_iou(box, box) == 1.0


def test_iou_partial_overlap() -> None:
    a = BBox(x=0, y=0, width=10, height=10)
    b = BBox(x=5, y=5, width=10, height=10)
    # intersection 5x5=25, union 100+100-25=175
    assert bbox_iou(a, b) == pytest.approx(25 / 175)


def test_iou_containment() -> None:
    outer = BBox(x=0, y=0, width=20, height=20)
    inner = BBox(x=5, y=5, width=5, height=5)
    assert bbox_iou(inner, outer) == pytest.approx(25 / 400)


def test_iou_disjoint_boxes() -> None:
    a = BBox(x=0, y=0, width=10, height=10)
    b = BBox(x=20, y=20, width=10, height=10)
    assert bbox_iou(a, b) == 0.0


def test_iou_edge_touching_is_disjoint() -> None:
    # Sharing an edge means zero overlap area.
    a = BBox(x=0, y=0, width=10, height=10)
    b = BBox(x=10, y=0, width=10, height=10)
    assert bbox_iou(a, b) == 0.0


def test_iou_zero_area_identical_boxes() -> None:
    a = BBox(x=5, y=5, width=0, height=0)
    assert bbox_iou(a, a) == 1.0


def test_iou_zero_area_distinct_boxes() -> None:
    # Both zero area, not identical: no evidence of a match.
    a = BBox(x=5, y=5, width=0, height=0)
    b = BBox(x=6, y=6, width=0, height=0)
    assert bbox_iou(a, b) == 0.0


# ---- ScoringConfig -----------------------------------------------------------


def test_config_default_threshold() -> None:
    assert ScoringConfig().iou_threshold == 0.5


@pytest.mark.parametrize("threshold", [0.0, 0.3, 1.0])
def test_config_accepts_valid_thresholds(threshold: float) -> None:
    assert ScoringConfig(iou_threshold=threshold).iou_threshold == threshold


@pytest.mark.parametrize(
    "bad",
    [True, False, "0.5", float("nan"), float("inf"), float("-inf"), -0.1, 1.1, None],
)
def test_config_rejects_invalid_thresholds(bad: object) -> None:
    with pytest.raises(ValueError):
        ScoringConfig(iou_threshold=bad)


def test_config_is_frozen() -> None:
    config = ScoringConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.iou_threshold = 0.1  # type: ignore[misc]


# ---- assign_matches (the core assignment) ------------------------------------


def test_assign_exact_match_matrix() -> None:
    config = ScoringConfig(iou_threshold=0.5)
    score = assign_matches([[1.0, 0.0], [0.0, 1.0]], config)
    assert [(m.expected_index, m.predicted_index) for m in score.matches] == [(0, 0), (1, 1)]
    assert score.false_positives == ()
    assert score.false_negatives == ()


def test_assign_count_wins_over_total_iou() -> None:
    # Greedy-by-IoU would take (0,0)=0.9 and leave expected 1 unmatched (1 match).
    # Count-first optimality must choose (0,1)=0.85 + (1,0)=0.85 (2 matches).
    config = ScoringConfig(iou_threshold=0.5)
    score = assign_matches([[0.9, 0.85], [0.85, 0.4]], config)
    assert len(score.matches) == 2
    assert {(m.expected_index, m.predicted_index) for m in score.matches} == {(0, 1), (1, 0)}


def test_assign_below_threshold_never_matches() -> None:
    config = ScoringConfig(iou_threshold=0.5)
    score = assign_matches([[0.49, 0.49], [0.49, 0.49]], config)
    assert score.matches == ()
    assert score.false_negatives == (0, 1)
    assert score.false_positives == (0, 1)


def test_assign_threshold_boundary_inclusive() -> None:
    config = ScoringConfig(iou_threshold=0.5)
    score = assign_matches([[0.5]], config)
    assert [(m.expected_index, m.predicted_index) for m in score.matches] == [(0, 0)]


def test_assign_one_to_one_only() -> None:
    config = ScoringConfig(iou_threshold=0.5)
    score = assign_matches([[0.9, 0.8]], config)
    assert len(score.matches) == 1
    assert score.false_positives == (1 - score.matches[0].predicted_index,)
    assert score.false_negatives == ()


def test_assign_rectangular_more_predicted() -> None:
    config = ScoringConfig(iou_threshold=0.5)
    score = assign_matches([[0.7, 0.6, 0.5]], config)
    assert len(score.matches) == 1
    assert sorted(score.false_positives) == [1, 2]


def test_assign_rectangular_more_expected() -> None:
    config = ScoringConfig(iou_threshold=0.5)
    score = assign_matches([[0.8], [0.8]], config)
    assert len(score.matches) == 1
    assert score.false_negatives == (1,)
    assert score.false_positives == ()


def test_assign_deterministic_tie_breaking() -> None:
    config = ScoringConfig(iou_threshold=0.5)
    a = assign_matches([[0.6, 0.6], [0.6, 0.6]], config)
    b = assign_matches([[0.6, 0.6], [0.6, 0.6]], config)
    assert [(m.expected_index, m.predicted_index) for m in a.matches] == [
        (m.expected_index, m.predicted_index) for m in b.matches
    ]
    # Deterministic, not random: identical input gives an identical pairing.
    assert {(m.expected_index, m.predicted_index) for m in a.matches} == {(0, 0), (1, 1)}


def test_assign_deterministic_asymmetric_tie() -> None:
    # Both assignments have the same count (2); totals: (0,1)+(1,0)=1.2 beats
    # (0,0)+(1,1)=0.9 — and the pairing is deterministic.
    config = ScoringConfig(iou_threshold=0.5)
    score = assign_matches([[0.6, 0.6], [0.6, 0.3]], config)
    assert {(m.expected_index, m.predicted_index) for m in score.matches} == {(0, 1), (1, 0)}


def test_assign_empty_sides() -> None:
    config = ScoringConfig()
    assert assign_matches([], config).matches == ()
    assert assign_matches([], config).false_negatives == ()


def test_assign_preserves_index_order_in_results() -> None:
    config = ScoringConfig(iou_threshold=0.5)
    score = assign_matches([[0.9, 0.7], [0.7, 0.9]], config)
    pairs = [(m.expected_index, m.predicted_index) for m in score.matches]
    assert pairs == sorted(pairs)
    assert score.false_positives == tuple(sorted(score.false_positives))
    assert score.false_negatives == tuple(sorted(score.false_negatives))


# ---- Text normalization ------------------------------------------------------


def test_normalize_text_casefold_and_whitespace() -> None:
    assert normalize_text("  Hello   World ") == "hello world"
    assert normalize_text("HEALTH\t120") == "health 120"
    assert normalize_text("a\r\nb") == "a b"


def test_normalize_text_empty() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_class_key_string_is_injective_across_part_boundaries() -> None:
    # F2 regression: a part containing the old join separator must not
    # collide with a different part split — the encoding is injective.
    a = class_key_string("ui_zones", ("health:bar", "brightness"))
    b = class_key_string("ui_zones", ("health", "bar:brightness"))
    assert a != b


def test_class_key_string_round_trips_part_boundaries() -> None:
    # The encoding keeps part boundaries unambiguous: decoding the string
    # recovers the exact parts.
    parts = ("zone:a", "b:c")
    encoded = class_key_string("ui_zones", parts)
    assert json.loads(encoded) == list(parts)


# ---- Per-category scoring ----------------------------------------------------


def _template(name: str, x: int) -> TemplateHit:
    return TemplateHit(template=name, bbox=BBox(x=x, y=0, width=10, height=10), confidence=0.9)


def test_score_templates_match_and_isolate_by_name() -> None:
    expected = (_template("target", 0), _template("loot", 100))
    predicted = (_template("target", 2),)
    score = score_templates(expected, predicted, ScoringConfig())
    assert score.tp == 1
    assert score.false_negatives == (1,)  # "loot" has no predicted partner
    assert score.false_positives == ()


def test_score_templates_no_confidence_filtering() -> None:
    low = TemplateHit(template="target", bbox=BBox(0, 0, 10, 10), confidence=0.0)
    high = TemplateHit(template="target", bbox=BBox(0, 0, 10, 10), confidence=0.99)
    score = score_templates((high,), (low,), ScoringConfig())
    # Scoring must not silently filter by confidence; the match stands.
    assert score.tp == 1


def _zone(zone: str, check: str, value: float) -> UiZoneHit:
    return UiZoneHit(zone=zone, check=check, bbox=BBox(0, 0, 10, 10), value=value)


def test_score_ui_zones_identity_is_zone_and_check() -> None:
    expected = (_zone("health_bar", "brightness", 0.8), _zone("health_bar", "presence", 1.0))
    predicted = (_zone("health_bar", "presence", 0.95),)
    score = score_ui_zones(expected, predicted, ScoringConfig())
    assert score.tp == 1
    # Same zone, different check: never a match.
    assert score.false_negatives == (0,)
    assert score.false_positives == ()


def test_score_ui_zones_reports_value_error() -> None:
    expected = (_zone("health_bar", "brightness", 0.8),)
    predicted = (_zone("health_bar", "brightness", 0.55),)
    score = score_ui_zones(expected, predicted, ScoringConfig())
    assert score.tp == 1
    assert score.value_errors == (pytest.approx(0.25),)


def test_score_ui_zones_value_error_is_absolute() -> None:
    expected = (_zone("health_bar", "brightness", 0.2),)
    predicted = (_zone("health_bar", "brightness", 0.9),)
    score = score_ui_zones(expected, predicted, ScoringConfig())
    assert score.value_errors == (pytest.approx(0.7),)


def test_score_ui_zones_value_errors_align_with_matches_across_classes() -> None:
    # value_errors must stay aligned with `matches` (sorted by expected
    # index) even when the internal class-iteration order differs from the
    # expected-index order: zone "a" sorts before "z" by key, but expected
    # index 0 is "z". One misaligned entry per pair makes this detectable.
    expected = (_zone("z", "bar", 0.5), _zone("a", "bar", 0.2))
    predicted = (_zone("z", "bar", 0.9), _zone("a", "bar", 0.1))
    score = score_ui_zones(expected, predicted, ScoringConfig())
    assert [(m.expected_index, m.predicted_index) for m in score.matches] == [(0, 0), (1, 1)]
    # match (0,0) is the "z" pair: |0.5 - 0.9| = 0.4; match (1,1) is "a": 0.1.
    assert score.value_errors == (pytest.approx(0.4), pytest.approx(0.1))


def test_score_text_regions_match_on_normalized_text() -> None:
    def text(t: str) -> TextRegion:
        return TextRegion(text=t, bbox=BBox(0, 0, 20, 10), confidence=0.9)

    score = score_text_regions((text("  Hello   World "),), (text("hello world"),), ScoringConfig())
    assert score.tp == 1


def test_score_text_regions_mismatched_text_is_fp_and_fn() -> None:
    def text(t: str) -> TextRegion:
        return TextRegion(text=t, bbox=BBox(0, 0, 20, 10), confidence=0.9)

    # Identical box, different text: no fuzzy matching — a miss on both sides.
    score = score_text_regions((text("hello world"),), (text("help world"),), ScoringConfig())
    assert score.tp == 0
    assert score.false_positives == (0,)
    assert score.false_negatives == (0,)


def test_score_text_regions_never_fuzzy() -> None:
    def text(t: str) -> TextRegion:
        return TextRegion(text=t, bbox=BBox(0, 0, 20, 10), confidence=0.9)

    # Off by one character is a different text after explicit normalization.
    score = score_text_regions((text("120"),), (text("121"),), ScoringConfig())
    assert score.tp == 0


def _object(kind: str, x: int) -> ObjectHit:
    return ObjectHit(kind=kind, bbox=BBox(x=x, y=0, width=10, height=10), confidence=0.9)


def test_score_objects_identity_is_kind() -> None:
    score = score_objects((_object("enemy", 0),), (_object("ally", 0),), ScoringConfig())
    assert score.tp == 0
    assert score.false_positives == (0,)
    assert score.false_negatives == (0,)


def test_score_empty_categories() -> None:
    score = score_templates((), (), ScoringConfig())
    assert score.tp == 0
    assert score.matches == ()
    assert score.false_positives == ()
    assert score.false_negatives == ()


def test_score_results_are_deterministic() -> None:
    expected = (_template("a", 0), _template("a", 5), _template("b", 0))
    predicted = (_template("a", 2), _template("a", 7), _template("b", 1), _template("c", 0))
    config = ScoringConfig()
    a = score_templates(expected, predicted, config)
    b = score_templates(expected, predicted, config)
    assert a == b


# ---- Metrics ------------------------------------------------------------------


def test_metrics_pure_tp() -> None:
    m = compute_metrics(tp=4, fp=0, fn=0, expected_total=4, predicted_total=4)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0


def test_metrics_zero_denominator_precision_is_none() -> None:
    m = compute_metrics(tp=0, fp=0, fn=3, expected_total=3, predicted_total=0)
    assert m.precision is None
    assert m.recall == 0.0
    # F1 is undefined when precision is undefined — never a manufactured rate.
    assert m.f1 is None


def test_metrics_zero_denominator_recall_is_none() -> None:
    m = compute_metrics(tp=0, fp=3, fn=0, expected_total=0, predicted_total=3)
    assert m.recall is None
    assert m.precision == 0.0
    assert m.f1 is None


def test_metrics_both_zero_denominators_all_none() -> None:
    m = compute_metrics(tp=0, fp=0, fn=0, expected_total=0, predicted_total=0)
    assert m.precision is None and m.recall is None and m.f1 is None


def test_metrics_never_manufactures_one() -> None:
    # Empty expected with no predictions is *no evidence*, not perfect recall.
    m = compute_metrics(tp=0, fp=0, fn=0, expected_total=0, predicted_total=0)
    assert m.recall != 1.0
    assert m.precision != 1.0


def test_metrics_f1_harmonic_mean() -> None:
    # tp=1, fp=1, fn=1 -> P=0.5, R=0.5, F1=0.5
    m = compute_metrics(tp=1, fp=1, fn=1, expected_total=2, predicted_total=2)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(0.5)
    # tp=2, fp=1, fn=2 -> P=2/3, R=0.5, F1=2*(2/3*0.5)/(2/3+0.5)=4/7
    m2 = compute_metrics(tp=2, fp=1, fn=2, expected_total=4, predicted_total=3)
    assert m2.f1 == pytest.approx(4 / 7)


def test_metrics_rejects_non_finite_and_negative() -> None:
    with pytest.raises(ValueError):
        compute_metrics(tp=float("inf"), fp=0, fn=0, expected_total=0, predicted_total=0)
    with pytest.raises(ValueError):
        compute_metrics(tp=-1, fp=0, fn=0, expected_total=0, predicted_total=0)


def test_merge_metrics_combines_and_recomputes() -> None:
    a = compute_metrics(tp=1, fp=1, fn=0, expected_total=1, predicted_total=2)
    b = compute_metrics(tp=0, fp=0, fn=1, expected_total=1, predicted_total=0)
    merged = merge_metrics(a, b)
    assert (merged.tp, merged.fp, merged.fn) == (1, 1, 1)
    assert (merged.expected_total, merged.predicted_total) == (2, 2)
    assert merged.precision == pytest.approx(0.5)
    assert merged.recall == pytest.approx(0.5)


def test_merge_metrics_empty() -> None:
    m = merge_metrics()
    assert m.tp == 0 and m.precision is None and m.recall is None


def test_metrics_finite_outputs() -> None:
    m = compute_metrics(tp=10, fp=3, fn=1, expected_total=11, predicted_total=13)
    for value in (m.precision, m.recall, m.f1):
        assert value is None or math.isfinite(value)
