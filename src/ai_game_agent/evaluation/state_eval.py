"""EH4 step 2: generic state-machine evaluation — expectation vs. actual.

The scoring contract (workplan §6):

* A state machine is *injected* through the :class:`StateEvaluator` protocol —
  the generic code never imports a game state machine. The protocol takes the
  predicted :class:`Observation` and returns a :class:`StateEvaluation` (the
  machine's actual signals/state/reason).
* Per expected signal, the verdict is one of ``TP``/``TN``/``FP``/``FN`` — or
  ``UNKNOWN`` when the adapter produced no signal (value ``None`` or key
  absent). **Unknown is never folded into a false negative**: a missing signal
  is missing *evidence*, which is reported in its own bucket.
* The state outcome is ``correct`` / ``incorrect`` (the expected/actual pair
  is kept as the confusion info) / ``unknown`` (the adapter could not
  determine a state).
* The transition reason is compared **only when the expectation supplies one**;
  comparison is exact string equality (reasons are stable identifiers, not
  free text).

Pure and stdlib-only: this module only imports the perception *contract* type
(:class:`Observation`) and the label type (:class:`StateExpectation`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ai_game_agent.evaluation.manifest import StateExpectation
from ai_game_agent.perception.observation import Observation

__all__ = [
    "StateAggregate",
    "StateEvaluation",
    "StateEvaluator",
    "StateVerdict",
    "SignalVerdict",
    "aggregate_state",
    "classify_state",
]

_SIGNAL_VERDICTS = ("TP", "TN", "FP", "FN", "UNKNOWN")
_STATE_OUTCOMES = ("correct", "incorrect", "unknown")


@dataclass(frozen=True)
class StateEvaluation:
    """The adapter's *actual* output for one observation.

    ``signals`` maps signal name → ``bool`` value; ``None`` (or a missing key)
    means the adapter had no evidence for that signal — that is *unknown*, and
    it is never interpreted as ``False``. ``state`` is the machine's state
    name, or ``None`` when the adapter could not determine one. ``reason`` is
    the adapter's transition reason, or ``None``.
    """

    signals: Mapping[str, bool | None]
    state: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.signals, Mapping):
            raise ValueError(
                f"StateEvaluation.signals must be a mapping, got {type(self.signals).__name__}"
            )
        for key, value in self.signals.items():
            if not isinstance(key, str):
                raise ValueError(f"StateEvaluation.signals keys must be strings, got {key!r}")
            if value is not None and not isinstance(value, bool):
                raise ValueError(
                    f"StateEvaluation.signals[{key!r}] must be a boolean or None, got {value!r}"
                )
        if self.state is not None and not isinstance(self.state, str):
            raise ValueError(f"StateEvaluation.state must be a string or None, got {self.state!r}")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError(
                f"StateEvaluation.reason must be a string or None, got {self.reason!r}"
            )


class StateEvaluator(Protocol):
    """The injectable state-machine seam (structural — game adapters
    implement it; the generic evaluator never imports a specific machine).

    Adapters may be stateful: the dataset evaluator feeds them in manifest
    order and does not reset them between samples — callers supply a fresh
    adapter per run.

    **Splits share one adapter.** The same instance is fed across both
    splits (calibration samples first, then held-out, in manifest order) and
    is never reset between them, so a machine with internal history evaluates
    the held-out samples *with its post-calibration state*. That is the
    intended semantics for an evolving state machine; it does mean the
    held-out state section is not independent of the calibration one. If
    per-split independence is required, the API will need per-split adapters
    (a future change, not made here).
    """

    def evaluate(self, observation: Observation) -> StateEvaluation: ...


@dataclass(frozen=True)
class SignalVerdict:
    """The classification of one expected signal against the actual one."""

    name: str
    expected: bool
    actual: bool | None
    verdict: str  # one of TP | TN | FP | FN | UNKNOWN


@dataclass(frozen=True)
class StateVerdict:
    """The per-sample state outcome.

    ``state`` is ``"correct"`` / ``"incorrect"`` / ``"unknown"``; the
    expected and actual state names are kept (the confusion pair when
    incorrect, and the expected name alone when unknown). ``reason_match`` is
    ``None`` when the expectation declared no reason to compare, else whether
    the adapter's reason equals it exactly.
    """

    signals: tuple[SignalVerdict, ...]
    state: str  # one of correct | incorrect | unknown
    expected_state: str
    actual_state: str | None
    reason_match: bool | None


def _signal_verdict(name: str, expected: bool, actual: bool | None) -> str:
    if actual is None:
        # Missing evidence — its own bucket, never collapsed into FN/TN.
        return "UNKNOWN"
    if expected and actual:
        return "TP"
    if not expected and not actual:
        return "TN"
    if expected and not actual:
        return "FN"
    return "FP"


def classify_state(expected: StateExpectation, actual: StateEvaluation) -> StateVerdict:
    """Classify one sample's expected state output against the adapter's.

    Only the signals the expectation declares are scored; extra actual
    signals carry no expectation and are ignored. Verdicts keep the
    expectation's signal order (deterministic).
    """
    signals = tuple(
        SignalVerdict(
            name=name,
            expected=value,
            actual=actual.signals.get(name),
            verdict=_signal_verdict(name, value, actual.signals.get(name)),
        )
        for name, value in expected.signals.items()
    )

    if actual.state is None:
        state = "unknown"
    elif actual.state == expected.state:
        state = "correct"
    else:
        state = "incorrect"

    if expected.reason is None:
        reason_match = None
    else:
        reason_match = actual.reason == expected.reason

    return StateVerdict(
        signals=signals,
        state=state,
        expected_state=expected.state,
        actual_state=actual.state,
        reason_match=reason_match,
    )


@dataclass(frozen=True)
class StateAggregate:
    """Counts over a set of per-sample state verdicts.

    Signal and state tallies are separate buckets (they are different
    measurements on the same sample). ``reason_mismatches`` counts only
    verdicts where a reason *was* expected; verdicts with no expected reason
    contribute to neither reason bucket.
    """

    samples_evaluated: int
    signal_tp: int
    signal_tn: int
    signal_fp: int
    signal_fn: int
    signal_unknown: int
    state_correct: int
    state_incorrect: int
    state_unknown: int
    reason_matches: int
    reason_mismatches: int


def aggregate_state(verdicts: Sequence[StateVerdict]) -> StateAggregate:
    """Sum the per-sample verdicts into the report's state section."""
    counts = {v: 0 for v in _SIGNAL_VERDICTS}
    states = {s: 0 for s in _STATE_OUTCOMES}
    reason_matches = 0
    reason_mismatches = 0
    for verdict in verdicts:
        for signal in verdict.signals:
            counts[signal.verdict] += 1
        states[verdict.state] += 1
        if verdict.reason_match is True:
            reason_matches += 1
        elif verdict.reason_match is False:
            reason_mismatches += 1
    return StateAggregate(
        samples_evaluated=len(verdicts),
        signal_tp=counts["TP"],
        signal_tn=counts["TN"],
        signal_fp=counts["FP"],
        signal_fn=counts["FN"],
        signal_unknown=counts["UNKNOWN"],
        state_correct=states["correct"],
        state_incorrect=states["incorrect"],
        state_unknown=states["unknown"],
        reason_matches=reason_matches,
        reason_mismatches=reason_mismatches,
    )
