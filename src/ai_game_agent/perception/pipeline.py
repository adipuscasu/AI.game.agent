"""The perception pipeline (Phase 2, step 3).

``Perception`` composes independently injectable detectors into the stable
``Observation`` contract. Design rules (plan §5.5):

* Runs detectors in a fixed order: templates → ui zones → ocr → objects.
* Each detector (or per-item unit: one template name, one UI zone, one OCR
  region) is wrapped in ``try/except Exception``; failures are recorded in
  ``Observation.detector_errors`` as a short, greppable string and the
  pipeline continues. AGENTS.md: ``PERCEPTION_ERROR`` is logged, not raised
  through the pipeline.
* Any detector may be ``None`` (disabled) — the pipeline simply skips it.
* ``PerceptionConfig.enabled == False`` short-circuits to an empty
  observation without consulting any detector.
* ``observe()`` is pure given its inputs: same frame + config ⇒ equal
  observation (the timestamp comes from the frame, never the wall clock).
"""

from __future__ import annotations

from ai_game_agent.capture import Frame
from ai_game_agent.config import PerceptionConfig
from ai_game_agent.perception.base import (
    ObjectDetector,
    OcrEngine,
    TemplateMatcher,
    UiRegionDetector,
)
from ai_game_agent.perception.observation import (
    BBox,
    ObjectHit,
    Observation,
    TemplateHit,
    TextRegion,
    UiZoneHit,
)

__all__ = ["Perception"]


class Perception:
    """Composes detectors into an :class:`Observation`.

    Detectors are optional keyword dependencies; pass ``None`` (or omit) to
    disable a subsystem. Real OpenCV/OCR implementations are injected by the
    caller — this module never imports them.
    """

    __slots__ = (
        "_config",
        "_template_matcher",
        "_ui_detector",
        "_ocr_engine",
        "_object_detector",
    )

    def __init__(
        self,
        config: PerceptionConfig,
        *,
        template_matcher: TemplateMatcher | None = None,
        ui_detector: UiRegionDetector | None = None,
        ocr_engine: OcrEngine | None = None,
        object_detector: ObjectDetector | None = None,
    ) -> None:
        self._config = config
        self._template_matcher = template_matcher
        self._ui_detector = ui_detector
        self._ocr_engine = ocr_engine
        self._object_detector = object_detector

    @property
    def config(self) -> PerceptionConfig:
        return self._config

    def observe(self, frame: Frame) -> Observation:
        """Run all enabled detectors against ``frame`` and build an observation.

        Never raises for detector failures: they are recorded in
        ``Observation.detector_errors`` and the rest of the observation stands.
        """
        if not self._config.enabled:
            return Observation(
                frame_width=frame.width,
                frame_height=frame.height,
                captured_at=frame.captured_at,
                source=frame.source,
                templates=(),
                ui_zones=(),
                text_regions=(),
                objects=(),
                detector_errors=(),
            )

        template_hits: list[TemplateHit] = []
        ui_zone_hits: list[UiZoneHit] = []
        text_regions: list[TextRegion] = []
        object_hits: list[ObjectHit] = []
        errors: list[str] = []

        self._run_templates(frame, template_hits, errors)
        self._run_ui_zones(frame, ui_zone_hits, errors)
        self._run_ocr(frame, text_regions, errors)
        self._run_objects(frame, object_hits, errors)

        return Observation(
            frame_width=frame.width,
            frame_height=frame.height,
            captured_at=frame.captured_at,
            source=frame.source,
            templates=tuple(template_hits),
            ui_zones=tuple(ui_zone_hits),
            text_regions=tuple(text_regions),
            objects=tuple(object_hits),
            detector_errors=tuple(errors),
        )

    # -- subsystem runners ---------------------------------------------------

    def _run_templates(
        self, frame: Frame, hits: list[TemplateHit], errors: list[str]
    ) -> None:
        matcher = self._template_matcher
        if matcher is None:
            return
        for name in self._config.templates:
            try:
                hit = matcher.match(frame, name)
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal
                errors.append(f"templates:{name}: {exc}")
                continue
            if hit is not None:
                hits.append(hit)

    def _run_ui_zones(
        self, frame: Frame, hits: list[UiZoneHit], errors: list[str]
    ) -> None:
        detector = self._ui_detector
        if detector is None:
            return
        for zone in self._config.ui_zones:
            try:
                hit = detector.detect(frame, zone)
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal
                errors.append(f"ui:{zone.name}: {exc}")
                continue
            if hit is not None:
                hits.append(hit)

    def _run_ocr(
        self, frame: Frame, regions: list[TextRegion], errors: list[str]
    ) -> None:
        engine = self._ocr_engine
        if engine is None or not self._config.ocr_enabled:
            return
        for region in self._config.ocr_regions:
            bbox = BBox(
                x=int(region["x"]),
                y=int(region["y"]),
                width=int(region["width"]),
                height=int(region["height"]),
            )
            name = str(region.get("name") or "unnamed")
            try:
                text = engine.read(frame, bbox)
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal
                errors.append(f"ocr:{name}: {exc}")
                continue
            if text is not None:
                regions.append(text)

    def _run_objects(
        self, frame: Frame, hits: list[ObjectHit], errors: list[str]
    ) -> None:
        detector = self._object_detector
        if detector is None or not self._config.objects_enabled:
            return
        try:
            hits.extend(detector.match(frame))
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            errors.append(f"objects: {exc}")
