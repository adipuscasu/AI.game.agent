"""EH4 step 2: observation scoring — IoU matching and detection metrics.

The scoring contract (workplan §2–§3):

* Four categories with distinct class identities: templates (name),
  ui_zones (``(zone, check)``), text_regions (normalized text), objects (kind).
  Class-mismatched hits never match, whatever their boxes say.
* One-to-one matching per class, chosen to **maximize the number of valid
  matches (IoU >= threshold), then total matched IoU** — a lexicographic
  maximum-weight bipartite matching, not greedy-by-IoU (greedy is provably
  suboptimal for the count-first objective; see the test matrix).
* Text normalization is explicit and non-fuzzy: ``casefold`` + collapse
  whitespace runs + strip. One different character is a miss.
* Zero denominators produce ``None`` metrics, never a manufactured ``1.0``.
* ``UiZoneHit.value`` is measured evidence, not confidence: matched zone
  pairs additionally report an absolute ``value_error``.

Stdlib only; the module imports the perception *contract* types (the frozen
``Observation`` dataclasses) and nothing else from the game stack.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_game_agent.perception.observation import (
    BBox,
    ObjectHit,
    Observation,
    TemplateHit,
    TextRegion,
    UiZoneHit,
)

__all__ = [
    "CATEGORIES",
    "CATEGORY_KEY_FUNCTIONS",
    "CategoryScore",
    "Match",
    "Metrics",
    "SCORE_FUNCTIONS",
    "ScoringConfig",
    "assign_matches",
    "bbox_iou",
    "category_hits",
    "class_key",
    "class_key_string",
    "compute_metrics",
    "merge_metrics",
    "normalize_text",
    "score_objects",
    "score_templates",
    "score_text_regions",
    "score_ui_zones",
]


@dataclass(frozen=True)
class ScoringConfig:
    """Explicit scoring parameters. No hidden defaults beyond this.

    ``iou_threshold`` — minimum intersection-over-union for a valid match.
    No confidence filtering is performed anywhere in scoring; if one is ever
    added it must be a documented, tested field here, not a silent cut.
    """

    iou_threshold: float = 0.5

    def __post_init__(self) -> None:
        value = self.iou_threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"iou_threshold must be a real number, got {value!r} ({type(value).__name__})"
            )
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError(
                f"iou_threshold must be a finite number, got {self.iou_threshold!r}"
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {self.iou_threshold!r}")


@dataclass(frozen=True)
class Match:
    """One valid one-to-one pairing within a class."""

    expected_index: int
    predicted_index: int
    iou: float


@dataclass(frozen=True)
class CategoryScore:
    """The scoring outcome for one category of one sample.

    ``matches`` and the index lists are sorted by their leading index, so the
    same inputs always produce an identical result. ``value_errors`` is set
    only for ui_zones (absolute ``|value_expected - value_predicted|`` per
    match, aligned with ``matches``) — ui-zone evidence error, reported
    separately from localization (IoU) accuracy.
    """

    matches: tuple[Match, ...]
    false_positives: tuple[int, ...]
    false_negatives: tuple[int, ...]
    value_errors: tuple[float, ...] | None = None

    @property
    def tp(self) -> int:
        return len(self.matches)


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union for axis-aligned integer boxes in frame space.

    Edge rule: two zero-area boxes match perfectly only when identical —
    with no area there is no overlap fraction to measure.
    """
    ix = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    iy = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    inter = ix * iy
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter
    if union == 0:
        return 1.0 if a == b else 0.0
    return inter / union


def normalize_text(text: str) -> str:
    """The *only* text equivalence used in scoring — explicit, non-fuzzy.

    ``casefold``, collapse every whitespace run to a single space, strip.
    Different visible text (one character apart included) never matches.
    """
    return " ".join(text.casefold().split())


# ---- Class identities --------------------------------------------------------


def _template_key(hit: Any) -> tuple:
    return (hit.template,)


def _ui_zone_key(hit: Any) -> tuple:
    return (hit.zone, hit.check)


def _text_region_key(hit: Any) -> tuple:
    return (normalize_text(hit.text),)


def _object_key(hit: Any) -> tuple:
    return (hit.kind,)


#: The four detection categories, in report order.
CATEGORIES: tuple[str, ...] = ("templates", "ui_zones", "text_regions", "objects")

#: The class-identity key for each category — the *exact* partition the
#: matcher uses, so per-class reporting groups identically to scoring.
CATEGORY_KEY_FUNCTIONS: Mapping[str, Callable[[Any], tuple]] = {
    "templates": _template_key,
    "ui_zones": _ui_zone_key,
    "text_regions": _text_region_key,
    "objects": _object_key,
}

#: The hit collection of each category on an :class:`Observation`.
_CATEGORY_COLLECTIONS: Mapping[str, Callable[[Observation], tuple]] = {
    "templates": lambda obs: obs.templates,
    "ui_zones": lambda obs: obs.ui_zones,
    "text_regions": lambda obs: obs.text_regions,
    "objects": lambda obs: obs.objects,
}


def class_key(category: str, hit: Any) -> tuple:
    """The class-identity key of ``hit`` within ``category`` (matching
    semantics — see :data:`CATEGORY_KEY_FUNCTIONS`)."""
    try:
        return CATEGORY_KEY_FUNCTIONS[category](hit)
    except KeyError:
        raise ValueError(
            f"unknown category {category!r}; expected one of {list(CATEGORIES)}"
        ) from None


def class_key_string(category: str, key: tuple) -> str:
    """The report's per-class map key: one deterministic string for a
    class-identity tuple (each part stringified, joined with ``:``)."""
    parts = tuple(str(part) for part in key)
    if not parts:
        raise ValueError("class key must not be empty")
    return ":".join(parts)


def category_hits(category: str, observation: Observation) -> tuple:
    """The ``category`` hit collection of an observation."""
    try:
        return _CATEGORY_COLLECTIONS[category](observation)
    except KeyError:
        raise ValueError(
            f"unknown category {category!r}; expected one of {list(CATEGORIES)}"
        ) from None


# ---- Core assignment ---------------------------------------------------------


def _hungarian(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Min-cost one-to-one assignment for a rectangular ``n x m`` cost
    matrix with ``n <= m``; returns ``(row, col)`` pairs (every row used).

    Deterministic: with equal candidate costs the lowest column index wins
    (strict ``<`` comparisons over a fixed iteration order).
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    inf = math.inf
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    assignment: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        way = [0] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment.append((p[j] - 1, j - 1))
    return assignment


def assign_matches(iou_matrix: Sequence[Sequence[float]], config: ScoringConfig) -> CategoryScore:
    """Assign expected (rows) to predicted (columns) hits for one class.

    Objective, lexicographic: maximize the number of pairs with
    ``iou >= config.iou_threshold``, then the sum of matched IoUs. Valid
    pairs carry weight ``BIG + iou``; invalid ones ``-BIG`` with
    ``BIG >`` the maximum possible IoU sum, so weight order *is* the
    lexicographic order. Results are index-sorted and deterministic.
    """
    expected_count = len(iou_matrix)
    predicted_count = len(iou_matrix[0]) if expected_count else 0
    if expected_count == 0 or predicted_count == 0:
        return CategoryScore(
            matches=(),
            false_positives=tuple(range(predicted_count)),
            false_negatives=tuple(range(expected_count)),
        )

    big = max(expected_count, predicted_count) + 1
    valid = [[i >= config.iou_threshold for i in row] for row in iou_matrix]
    weight = [
        [(big + iou) if ok else -big for iou, ok in zip(row, ok_row, strict=True)]
        for row, ok_row in zip(iou_matrix, valid, strict=True)
    ]

    if expected_count <= predicted_count:
        cost = [[-w for w in row] for row in weight]
        pairs = _hungarian(cost)
        row_of, col_of = 0, 1
    else:
        cost = [[-row[j] for row in weight] for j in range(predicted_count)]
        pairs = _hungarian(cost)
        row_of, col_of = 1, 0

    matches = [(pair[row_of], pair[col_of]) for pair in pairs if valid[pair[row_of]][pair[col_of]]]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return CategoryScore(
        matches=tuple(
            Match(expected_index=r, predicted_index=c, iou=float(iou_matrix[r][c]))
            for r, c in sorted(matches)
        ),
        false_positives=tuple(c for c in range(predicted_count) if c not in matched_cols),
        false_negatives=tuple(r for r in range(expected_count) if r not in matched_rows),
    )


# ---- Per-category scoring ----------------------------------------------------


def _score_class_partitioned(
    expected: Sequence[Any],
    predicted: Sequence[Any],
    config: ScoringConfig,
    key: Callable[[Any], tuple],
    *,
    value_errors: bool = False,
) -> CategoryScore:
    """Score one category: partition both sides by class key, run the
    optimal assignment inside each class (classes never compete)."""
    exp_by_key: dict[tuple, list[int]] = {}
    for i, hit in enumerate(expected):
        exp_by_key.setdefault(key(hit), []).append(i)
    pred_by_key: dict[tuple, list[int]] = {}
    for i, hit in enumerate(predicted):
        pred_by_key.setdefault(key(hit), []).append(i)

    all_matches: list[Match] = []
    fps: list[int] = []
    fns: list[int] = []
    verrors: list[float] = []
    for class_key in sorted(set(exp_by_key) | set(pred_by_key), key=repr):
        exp_idx = exp_by_key.get(class_key, [])
        pred_idx = pred_by_key.get(class_key, [])
        if not exp_idx or not pred_idx:
            # No possible match on one side: every hit there is a miss.
            # (Handled directly — an empty IoU matrix carries no dimension.)
            fns.extend(exp_idx)
            fps.extend(pred_idx)
            continue
        matrix = [
            [bbox_iou(expected[e].bbox, predicted[p].bbox) for p in pred_idx] for e in exp_idx
        ]
        score = assign_matches(matrix, config)
        all_matches.extend(
            Match(
                expected_index=exp_idx[m.expected_index],
                predicted_index=pred_idx[m.predicted_index],
                iou=m.iou,
            )
            for m in score.matches
        )
        fps.extend(pred_idx[i] for i in score.false_positives)
        fns.extend(exp_idx[i] for i in score.false_negatives)
        if value_errors:
            verrors.extend(
                abs(
                    expected[exp_idx[m.expected_index]].value
                    - predicted[pred_idx[m.predicted_index]].value
                )
                for m in score.matches
            )

    all_matches.sort(key=lambda m: (m.expected_index, m.predicted_index))
    return CategoryScore(
        matches=tuple(all_matches),
        false_positives=tuple(sorted(fps)),
        false_negatives=tuple(sorted(fns)),
        value_errors=tuple(verrors) if value_errors else None,
    )


def score_templates(
    expected: tuple[TemplateHit, ...], predicted: tuple[TemplateHit, ...], config: ScoringConfig
) -> CategoryScore:
    """Templates match on template name + box IoU >= threshold."""
    return _score_class_partitioned(expected, predicted, config, _template_key)


def score_ui_zones(
    expected: tuple[UiZoneHit, ...], predicted: tuple[UiZoneHit, ...], config: ScoringConfig
) -> CategoryScore:
    """UI zones match on ``(zone, check)`` + box IoU; matched pairs also
    report the absolute evidence-value error (separate from IoU)."""
    return _score_class_partitioned(expected, predicted, config, _ui_zone_key, value_errors=True)


def score_text_regions(
    expected: tuple[TextRegion, ...], predicted: tuple[TextRegion, ...], config: ScoringConfig
) -> CategoryScore:
    """Text regions match only when the *normalized texts are equal* and the
    boxes overlap enough — no fuzzy matching, ever."""
    return _score_class_partitioned(expected, predicted, config, _text_region_key)


def score_objects(
    expected: tuple[ObjectHit, ...], predicted: tuple[ObjectHit, ...], config: ScoringConfig
) -> CategoryScore:
    """Objects match on kind + box IoU >= threshold."""
    return _score_class_partitioned(expected, predicted, config, _object_key)


#: The per-category scoring entry points, in report order.
SCORE_FUNCTIONS: Mapping[str, Callable[..., CategoryScore]] = {
    "templates": score_templates,
    "ui_zones": score_ui_zones,
    "text_regions": score_text_regions,
    "objects": score_objects,
}


# ---- Metrics -----------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """Counts plus derived rates. Rates are ``None`` on a zero denominator —
    "no evidence" is never reported as a manufactured perfect score."""

    tp: int
    fp: int
    fn: int
    expected_total: int
    predicted_total: int
    precision: float | None
    recall: float | None
    f1: float | None


def _int_field(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r} ({type(value).__name__})")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        raise ValueError(f"non-finite rate from {numerator}/{denominator}")
    return value


def compute_metrics(
    *, tp: int, fp: int, fn: int, expected_total: int, predicted_total: int
) -> Metrics:
    counts = {
        "tp": _int_field("tp", tp),
        "fp": _int_field("fp", fp),
        "fn": _int_field("fn", fn),
        "expected_total": _int_field("expected_total", expected_total),
        "predicted_total": _int_field("predicted_total", predicted_total),
    }
    precision = _rate(counts["tp"], counts["tp"] + counts["fp"])
    recall = _rate(counts["tp"], counts["tp"] + counts["fn"])
    if precision is None or recall is None:
        f1 = None
    else:
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
            if not math.isfinite(f1):
                raise ValueError(f"non-finite f1 from precision={precision}, recall={recall}")
    return Metrics(
        tp=counts["tp"],
        fp=counts["fp"],
        fn=counts["fn"],
        expected_total=counts["expected_total"],
        predicted_total=counts["predicted_total"],
        precision=precision,
        recall=recall,
        f1=f1,
    )


def merge_metrics(*metrics: Metrics) -> Metrics:
    """Combine metrics by summing the counts and *recomputing* the rates —
    averaging rates would weight unequal denominators wrongly."""
    return compute_metrics(
        tp=sum(m.tp for m in metrics),
        fp=sum(m.fp for m in metrics),
        fn=sum(m.fn for m in metrics),
        expected_total=sum(m.expected_total for m in metrics),
        predicted_total=sum(m.predicted_total for m in metrics),
    )
