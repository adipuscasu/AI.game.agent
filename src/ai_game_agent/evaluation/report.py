"""EH4 step 2: the deterministic evaluation report (workplan §8).

Contract:

* The report is a frozen, JSON-serializable value object. ``to_dict()`` has
  a **fixed key order** and ``to_json()`` uses fixed separators +
  ``ensure_ascii=True``, so identical inputs produce byte-identical output.
* Every ``Metrics`` section comes from the scoring contract — a zero
  denominator is ``None``, never a manufactured rate.
* Class maps are emitted **sorted by class key**; samples keep manifest
  order (the caller's input order, never re-sorted).
* Non-finite floats are rejected at construction — a report must never
  serialize ``NaN``/``Infinity`` into "valid" JSON.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ai_game_agent.evaluation.manifest import SPLITS
from ai_game_agent.evaluation.scoring import CATEGORIES, Metrics
from ai_game_agent.evaluation.state_eval import StateAggregate, StateVerdict

__all__ = [
    "CategoryMetrics",
    "Counts",
    "EVALUATOR_VERSION",
    "MatchDetail",
    "REPORT_SCHEMA",
    "SAMPLE_STATUSES",
    "SampleCategory",
    "SampleReport",
    "SplitReport",
    "StateSection",
    "EvaluationReport",
]

#: Bumped whenever the report *shape* changes in a way readers must handle.
EVALUATOR_VERSION = 1

REPORT_SCHEMA = "evaluation-report"

#: Per-sample outcome buckets — scored samples enter the metrics; the rest
#: are reported, never silently dropped (workplan §4).
SAMPLE_STATUSES: tuple[str, ...] = ("scored", "incomplete", "skipped", "missing")

_SIGNAL_ORDER: tuple[str, ...] = ("TP", "TN", "FP", "FN", "UNKNOWN")
_STATE_ORDER: tuple[str, ...] = ("correct", "incorrect", "unknown")


def _int_field(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r} ({type(value).__name__})")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r} ({type(value).__name__})")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class Counts:
    """Sample status buckets. ``total`` is the invariant sum of the other
    four — a report where they disagree is a construction bug, not data."""

    scored: int
    incomplete: int
    skipped: int
    missing: int
    total: int

    def __post_init__(self) -> None:
        _int_field("scored", self.scored)
        _int_field("incomplete", self.incomplete)
        _int_field("skipped", self.skipped)
        _int_field("missing", self.missing)
        _int_field("total", self.total)
        expected = self.scored + self.incomplete + self.skipped + self.missing
        if self.total != expected:
            raise ValueError(
                f"total must equal the sum of the status buckets, got {self.total} != {expected}"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "scored": self.scored,
            "incomplete": self.incomplete,
            "skipped": self.skipped,
            "missing": self.missing,
            "total": self.total,
        }


@dataclass(frozen=True)
class MatchDetail:
    """One matched pair in a sample's report entry.

    ``expected``/``predicted`` are the class-identity strings (they are
    equal within a match — matching is per class). ``value_error`` is set
    only for ui_zones (absolute evidence-value error, not IoU).
    """

    iou: float
    expected: str
    predicted: str
    value_error: float | None = None

    def __post_init__(self) -> None:
        iou = _finite_float("iou", self.iou)
        if not 0.0 <= iou <= 1.0:
            raise ValueError(f"iou must be in [0.0, 1.0], got {iou}")
        _nonempty_str("expected", self.expected)
        _nonempty_str("predicted", self.predicted)
        if self.value_error is not None:
            value = _finite_float("value_error", self.value_error)
            if value < 0.0:
                raise ValueError(f"value_error must be >= 0, got {value}")

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "iou": self.iou,
            "expected": self.expected,
            "predicted": self.predicted,
        }
        if self.value_error is not None:
            d["value_error"] = self.value_error
        return d


@dataclass(frozen=True)
class SampleCategory:
    """One sample's per-category outcome. Counts are derived from the hit
    lists: ``tp == len(matches)``, ``fp == len(false_positives)`` and
    ``fn == len(false_negatives)`` — enforced, so a report cannot disagree
    with its own detail."""

    category: str
    tp: int
    fp: int
    fn: int
    matches: tuple[MatchDetail, ...]
    false_positives: tuple[int, ...]
    false_negatives: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {list(CATEGORIES)}, got {self.category!r}")
        _int_field("tp", self.tp)
        _int_field("fp", self.fp)
        _int_field("fn", self.fn)
        if self.tp != len(self.matches):
            raise ValueError(f"tp must equal len(matches), got {self.tp} != {len(self.matches)}")
        if self.fp != len(self.false_positives):
            raise ValueError(
                f"fp must equal len(false_positives), got {self.fp} != {len(self.false_positives)}"
            )
        if self.fn != len(self.false_negatives):
            raise ValueError(
                f"fn must equal len(false_negatives), got {self.fn} != {len(self.false_negatives)}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "matches": [m.to_dict() for m in self.matches],
            "false_positives": list(self.false_positives),
            "false_negatives": list(self.false_negatives),
        }


@dataclass(frozen=True)
class SampleReport:
    """One sample's row in the report (manifest order is preserved).

    ``categories`` holds all four category outcomes for ``scored`` samples
    and is empty for the excluded buckets (missing / incomplete / skipped
    are reported *as* their status, not as zero scores). ``state`` is
    ``None`` when the sample has no state expectation or no adapter ran.
    """

    id: str
    split: str
    status: str
    detector_errors: tuple[str, ...]
    categories: tuple[SampleCategory, ...]
    state: StateVerdict | None = None

    def __post_init__(self) -> None:
        _nonempty_str("id", self.id)
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {list(SPLITS)}, got {self.split!r}")
        if self.status not in SAMPLE_STATUSES:
            raise ValueError(f"status must be one of {list(SAMPLE_STATUSES)}, got {self.status!r}")
        if not isinstance(self.detector_errors, tuple) or not all(
            isinstance(e, str) for e in self.detector_errors
        ):
            raise ValueError("detector_errors must be a tuple of strings")
        if not isinstance(self.categories, tuple):
            raise ValueError("categories must be a tuple of SampleCategory")
        if self.status == "scored":
            if tuple(c.category for c in self.categories) != CATEGORIES:
                raise ValueError(
                    f"a scored sample needs all {list(CATEGORIES)} categories, "
                    f"got {[c.category for c in self.categories]}"
                )
        elif self.categories:
            raise ValueError(
                f"an excluded sample (status={self.status}) must not carry category scores"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "split": self.split,
            "status": self.status,
            "detector_errors": list(self.detector_errors),
            "categories": (
                {c.category: c.to_dict() for c in self.categories} if self.categories else None
            ),
            "state": _state_sample_dict(self.state),
        }


def _state_sample_dict(state: StateVerdict | None) -> dict[str, object] | None:
    if state is None:
        return {"evaluated": False, "state": "not_evaluated"}
    return {
        "evaluated": True,
        "state": state.state,
        "expected_state": state.expected_state,
        "actual_state": state.actual_state,
        "signals": [
            {
                "name": s.name,
                "expected": s.expected,
                "actual": s.actual,
                "verdict": s.verdict,
            }
            for s in state.signals
        ],
        "reason_match": state.reason_match,
    }


@dataclass(frozen=True)
class CategoryMetrics:
    """One category's split-level metrics: overall plus per class.

    ``classes`` is keyed by the class-identity string; emission is sorted
    by key (JSON map order must be deterministic)."""

    category: str
    overall: Metrics
    classes: Mapping[str, Metrics]

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {list(CATEGORIES)}, got {self.category!r}")
        if not isinstance(self.overall, Metrics):
            raise ValueError(f"overall must be a Metrics, got {type(self.overall).__name__}")
        if not isinstance(self.classes, Mapping):
            raise ValueError("classes must be a mapping of class key -> Metrics")
        for key, value in self.classes.items():
            _nonempty_str("classes key", key)
            if not isinstance(value, Metrics):
                raise ValueError(f"classes[{key!r}] must be a Metrics, got {type(value).__name__}")

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": _metrics_dict(self.overall),
            "classes": {key: _metrics_dict(self.classes[key]) for key in sorted(self.classes)},
        }


def _metrics_dict(m: Metrics) -> dict[str, object]:
    return {
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "expected_total": m.expected_total,
        "predicted_total": m.predicted_total,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
    }


@dataclass(frozen=True)
class StateSection:
    """One split's state-evaluation section.

    ``performed`` is ``False`` when no adapter was supplied — an explicit
    "not performed", never a silent zero section (workplan §6). When not
    performed the aggregate must be all zeros (nothing was evaluated)."""

    performed: bool
    aggregate: StateAggregate

    def __post_init__(self) -> None:
        if not isinstance(self.performed, bool):
            raise ValueError(f"performed must be a boolean, got {self.performed!r}")
        if not isinstance(self.aggregate, StateAggregate):
            raise ValueError(
                f"aggregate must be a StateAggregate, got {type(self.aggregate).__name__}"
            )
        if not self.performed and self.aggregate.samples_evaluated != 0:
            raise ValueError(
                "a not-performed state section must have samples_evaluated == 0, "
                f"got {self.aggregate.samples_evaluated}"
            )

    def to_dict(self) -> dict[str, object]:
        a = self.aggregate
        return {
            "performed": self.performed,
            "samples_evaluated": a.samples_evaluated,
            "signals": {name: getattr(a, f"signal_{name.lower()}") for name in _SIGNAL_ORDER},
            "states": {name: getattr(a, f"state_{name}") for name in _STATE_ORDER},
            "reason_matches": a.reason_matches,
            "reason_mismatches": a.reason_mismatches,
        }


@dataclass(frozen=True)
class SplitReport:
    """One split's section — splits are reported separately and never
    combined (workplan §5: held_out must stay honest)."""

    split: str
    overall: Metrics
    categories: Mapping[str, CategoryMetrics]
    state: StateSection

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {list(SPLITS)}, got {self.split!r}")
        if not isinstance(self.overall, Metrics):
            raise ValueError(f"overall must be a Metrics, got {type(self.overall).__name__}")
        if not isinstance(self.categories, Mapping):
            raise ValueError("categories must be a mapping of category -> CategoryMetrics")
        if tuple(self.categories) != CATEGORIES:
            raise ValueError(
                f"categories must cover exactly {list(CATEGORIES)}, got {list(self.categories)}"
            )
        if not isinstance(self.state, StateSection):
            raise ValueError(f"state must be a StateSection, got {type(self.state).__name__}")

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": _metrics_dict(self.overall),
            "categories": {c: self.categories[c].to_dict() for c in CATEGORIES},
            "state": self.state.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationReport:
    """The full, deterministic evaluation outcome."""

    dataset: str
    dataset_version: str
    manifest_schema_version: int
    iou_threshold: float
    counts: Counts
    splits: Mapping[str, SplitReport]
    samples: tuple[SampleReport, ...]

    def __post_init__(self) -> None:
        _nonempty_str("dataset", self.dataset)
        _nonempty_str("dataset_version", self.dataset_version)
        if (
            isinstance(self.manifest_schema_version, bool)
            or not isinstance(self.manifest_schema_version, int)
            or self.manifest_schema_version <= 0
        ):
            raise ValueError(
                "manifest_schema_version must be a positive integer, "
                f"got {self.manifest_schema_version!r}"
            )
        threshold = _finite_float("iou_threshold", self.iou_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {threshold}")
        if not isinstance(self.counts, Counts):
            raise ValueError(f"counts must be a Counts, got {type(self.counts).__name__}")
        if not isinstance(self.splits, Mapping) or set(self.splits) != set(SPLITS):
            raise ValueError(f"splits must cover exactly {list(SPLITS)}, got {list(self.splits)}")
        for key, value in self.splits.items():
            if not isinstance(value, SplitReport):
                raise ValueError(
                    f"splits[{key!r}] must be a SplitReport, got {type(value).__name__}"
                )
            if value.split != key:
                raise ValueError(f"splits[{key!r}].split must be {key!r}, got {value.split!r}")
        if not isinstance(self.samples, tuple):
            raise ValueError(f"samples must be a tuple, got {type(self.samples).__name__}")
        for sample in self.samples:
            if not isinstance(sample, SampleReport):
                raise ValueError(
                    f"samples entries must be SampleReport, got {type(sample).__name__}"
                )
        # Bucket totals must agree with the sample rows — the two views are
        # the same data and a disagreement is a construction bug.
        tally = Counter(s.status for s in self.samples)
        if (
            tally.get("scored", 0) != self.counts.scored
            or tally.get("incomplete", 0) != self.counts.incomplete
            or tally.get("skipped", 0) != self.counts.skipped
            or tally.get("missing", 0) != self.counts.missing
        ):
            raise ValueError("counts must match the per-sample statuses")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "evaluator_version": EVALUATOR_VERSION,
            "manifest_schema_version": self.manifest_schema_version,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "scoring": {"iou_threshold": self.iou_threshold},
            "counts": self.counts.to_dict(),
            "splits": {split: self.splits[split].to_dict() for split in SPLITS},
            "samples": [s.to_dict() for s in self.samples],
        }

    def to_json(self) -> str:
        """Byte-deterministic serialization: fixed key order (the dataclass
        construction order), fixed separators, ASCII-safe escapes."""
        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(",", ":"))
