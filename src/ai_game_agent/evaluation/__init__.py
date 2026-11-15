"""EH4: versioned perception-evaluation dataset and reproducible scoring.

This package is deliberately game-agnostic: it knows the ``Observation``
perception contract (what a detection *is*) and a generic
``signals -> (state, reason)`` decision seam, but no specific game. Game
adapters plug in their state machine at evaluation time.

Public API (also importable directly from the submodules):

* ``DatasetManifest`` / ``load_manifest`` / ``validate_manifest`` — the
  versioned label contract (step 1).
* ``ScoringConfig`` — matching config (IoU threshold).
* ``evaluate_dataset(manifest, predictions, config, state_evaluator)`` —
  the dataset evaluation entry point (step 2).
* ``StateEvaluator`` — the injectable adapter protocol for state
  evaluation.
* ``EvaluationReport`` — the deterministic report; ``to_json()`` is
  byte-identical for identical inputs.
* ``EVALUATOR_VERSION`` / ``MANIFEST_SCHEMA_VERSION`` — schema versions.
"""

from ai_game_agent.evaluation.dataset import evaluate_dataset
from ai_game_agent.evaluation.manifest import (
    MANIFEST_SCHEMA_VERSION,
    SPLITS,
    CaptureMetadata,
    DatasetManifest,
    ManifestError,
    Sample,
    StateExpectation,
    load_manifest,
    validate_manifest,
)
from ai_game_agent.evaluation.report import (
    EVALUATOR_VERSION,
    REPORT_SCHEMA,
    SAMPLE_STATUSES,
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
    ScoringConfig,
    bbox_iou,
    compute_metrics,
    merge_metrics,
    normalize_text,
)
from ai_game_agent.evaluation.state_eval import (
    SignalVerdict,
    StateAggregate,
    StateEvaluation,
    StateEvaluator,
    StateVerdict,
    aggregate_state,
    classify_state,
)

__all__ = [
    # manifest (step 1)
    "CaptureMetadata",
    "DatasetManifest",
    "ManifestError",
    "MANIFEST_SCHEMA_VERSION",
    "Sample",
    "SPLITS",
    "StateExpectation",
    "load_manifest",
    "validate_manifest",
    # scoring
    "CATEGORIES",
    "ScoringConfig",
    "bbox_iou",
    "compute_metrics",
    "merge_metrics",
    "normalize_text",
    # state evaluation
    "SignalVerdict",
    "StateAggregate",
    "StateEvaluation",
    "StateEvaluator",
    "StateVerdict",
    "aggregate_state",
    "classify_state",
    # dataset API + report
    "CategoryMetrics",
    "Counts",
    "EVALUATOR_VERSION",
    "EvaluationReport",
    "MatchDetail",
    "REPORT_SCHEMA",
    "SAMPLE_STATUSES",
    "SampleCategory",
    "SampleReport",
    "SplitReport",
    "StateSection",
    "evaluate_dataset",
]
