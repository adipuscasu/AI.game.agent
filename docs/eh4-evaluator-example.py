"""EH4 step 2 — runnable synthetic example for the evaluation package.

Run with:

    uv run python docs/eh4-evaluator-example.py

What it shows:

* an in-memory ``DatasetManifest`` (the file-based ``load_manifest`` adds
  image-existence checks; the synthetic manifest below exercises the same
  validated types);
* predictions keyed by sample id covering all four sample statuses:
  ``scored`` (x2) / ``incomplete`` / ``skipped`` / ``missing``;
* ``ScoringConfig`` with an explicit IoU threshold;
* an injected ``StateEvaluator`` adapter (a fake state machine — the
  package never imports game code);
* split isolation (``calibration`` and ``held_out`` reported separately);
* ``None`` state signals (unknown, never folded into false) and
  detector errors (excluded samples reported, never silently scored);
* byte-identical ``to_json()`` across two runs.
"""

from __future__ import annotations

import datetime

from ai_game_agent.evaluation import (
    EVALUATOR_VERSION,
    CaptureMetadata,
    DatasetManifest,
    Sample,
    ScoringConfig,
    StateEvaluation,
    StateExpectation,
    evaluate_dataset,
    validate_manifest,
)
from ai_game_agent.perception.observation import BBox, Observation, TemplateHit

_DT = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def obs(
    templates: tuple[TemplateHit, ...] = (),
    errors: tuple[str, ...] = (),
) -> Observation:
    return Observation(
        frame_width=1280,
        frame_height=720,
        captured_at=_DT,
        source="synthetic",
        templates=templates,
        ui_zones=(),
        text_regions=(),
        objects=(),
        detector_errors=errors,
    )


def hit(name: str, x: int) -> TemplateHit:
    return TemplateHit(template=name, bbox=BBox(x=x, y=10, width=40, height=20), confidence=0.9)


# ---- Synthetic dataset -------------------------------------------------------
# Five samples across both splits, exercising all four status buckets plus
# two scored samples (one a clean TP + reason match, one an UNKNOWN signal).

manifest = DatasetManifest(
    dataset="synthetic-demo",
    dataset_version="0.1",
    created="2026-01-01",
    capture=CaptureMetadata(1280, 720, "windowed", 1.0),
    samples=(
        # scored: perfect template match; expectation met including reason.
        Sample(
            id="c1",
            image="c1.png",
            split="calibration",
            detections=obs(templates=(hit("boss", 100),)),
            state=StateExpectation(
                signals={"boss_visible": True}, state="COMBAT", reason="boss_detected"
            ),
        ),
        # scored: template matches, but the adapter reports NO minion signal
        # -> minion_visible is UNKNOWN (missing evidence), never False.
        Sample(
            id="c2",
            image="c2.png",
            split="calibration",
            detections=obs(templates=(hit("minion", 40),)),
            state=StateExpectation(signals={"minion_visible": True}, state="COMBAT"),
        ),
        # incomplete: the *prediction* carried detector errors — missing
        # detections there are missing evidence, not negative detections.
        Sample(
            id="h1",
            image="h1.png",
            split="held_out",
            detections=obs(templates=(hit("minion", 40),)),
        ),
        # skipped: the *label* itself is incomplete — ground truth cannot be
        # trusted for this frame.
        Sample(
            id="h2",
            image="h2.png",
            split="held_out",
            detections=obs(templates=(hit("boss", 100),), errors=("labeler: frame was corrupted",)),
        ),
        # missing: no prediction was produced for this sample at all.
        Sample(
            id="h3",
            image="h3.png",
            split="held_out",
            detections=obs(templates=(hit("boss", 100),)),
            state=StateExpectation(signals={"boss_visible": True}, state="COMBAT"),
        ),
    ),
)
problems = validate_manifest(manifest)
assert not problems, problems

# ---- Predictions -------------------------------------------------------------
# h1's prediction crashed partway through detection: it reported the minion
# AND a detector error -> the sample is incomplete, not scored. h2's label is
# corrupted -> skipped (the label wins: ground truth is untrustworthy even
# though a prediction exists). h3 has no prediction at all -> missing.

predictions: dict[str, Observation] = {
    "c1": obs(templates=(hit("boss", 100),)),
    "c2": obs(templates=(hit("minion", 40),)),
    "h1": obs(templates=(hit("minion", 40),), errors=("ocr: timed out",)),
    "h2": obs(templates=(hit("boss", 100),)),
}

# ---- Adapter: a fake state machine -------------------------------------------
# The real adapter would call the game's GameStateMachine; here a fixed map
# stands in, keyed by which template the observation carries. Only *scored*
# samples are fed (c1, c2 — h1 is incomplete, h2/h3 excluded), in manifest
# order. c1 gets a full TP + matching reason; c2's minion signal is absent
# -> UNKNOWN (the package never folds a missing signal into False).


class _FakeMachine:
    def __init__(self) -> None:
        self._by_template = {
            "boss": StateEvaluation(
                signals={"boss_visible": True}, state="COMBAT", reason="boss_detected"
            ),
            "minion": StateEvaluation(signals={}, state="COMBAT", reason=None),
        }
        self.fed: list[str] = []

    def evaluate(self, observation: Observation) -> StateEvaluation:
        name = observation.templates[0].template if observation.templates else ""
        self.fed.append(name)
        return self._by_template.get(name, StateEvaluation(signals={}, state=None, reason=None))


adapter = _FakeMachine()
report = evaluate_dataset(
    manifest, predictions, config=ScoringConfig(iou_threshold=0.5), state_evaluator=adapter
)

# ---- Read the report ---------------------------------------------------------
print(f"evaluator v{EVALUATOR_VERSION} - dataset {report.dataset} {report.dataset_version}")
print(f"counts: {report.counts.to_dict()}")
for split in ("calibration", "held_out"):
    section = report.splits[split]
    m = section.overall
    print(
        f"{split}: tp={m.tp} fp={m.fp} fn={m.fn} "
        f"precision={m.precision} recall={m.recall} f1={m.f1} "
        f"state.performed={section.state.performed} samples_evaluated={section.state.aggregate.samples_evaluated}"
    )
    print(
        f"  signals={section.state.to_dict()['signals']} "
        f"states={section.state.to_dict()['states']} "
        f"reason_matches={section.state.aggregate.reason_matches}"
    )
for sample in report.samples:
    print(
        f"sample {sample.id}: status={sample.status} detector_errors={list(sample.detector_errors)}"
    )

# ---- Determinism -------------------------------------------------------------
first = report.to_json()
again = evaluate_dataset(
    manifest, predictions, config=ScoringConfig(iou_threshold=0.5), state_evaluator=_FakeMachine()
).to_json()
assert first == again, "identical inputs must produce byte-identical JSON"
print(f"\nto_json() byte-identical across two runs ({len(first)} bytes)")
print(first[:200] + " ...")
