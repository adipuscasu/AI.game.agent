"""Detector protocol interfaces for the perception pipeline (Phase 2, step 3).

This module defines the *interfaces* the ``Perception`` pipeline programs
against. It deliberately contains no OpenCV, no OCR engine, and no other
heavy dependency — only the project's own ``Frame`` and ``Observation``
contract types — so it imports cleanly in a bare venv.

Real implementations (``template.py``, ``ui.py``, ``ocr.py``, ``objects.py``)
live in separate modules and are injected into the pipeline; tests inject
fakes that structurally conform to these protocols.

The protocols use ``typing.Protocol`` so conformance is structural (no base
class required), which keeps fakes and third-party detectors decoupled from
this package.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_game_agent.capture import Frame
from ai_game_agent.config import UiZone
from ai_game_agent.perception.observation import (
    BBox,
    ObjectHit,
    TemplateHit,
    TextRegion,
    UiZoneHit,
)

__all__ = [
    "ObjectDetector",
    "OcrEngine",
    "PerceptionError",
    "TemplateMatcher",
    "UiRegionDetector",
]


class PerceptionError(Exception):
    """Raised by the perception layer for configuration/setup failures.

    Per AGENTS.md this is the ``PERCEPTION_ERROR`` class. It is *not* used
    for per-frame detector failures (those are recorded in
    ``Observation.detector_errors``); it signals setup problems that should
    fail fast, e.g. a template file missing at matcher construction.
    """


@runtime_checkable
class TemplateMatcher(Protocol):
    """Matches a named template image against a frame.

    ``template`` is the *logical name* from configuration (e.g.
    ``"target_frame"``), not a file path. Implementations load template files
    themselves and decide whether to honor the configured threshold.
    """

    def match(self, frame: Frame, template: str) -> TemplateHit | None:
        """Return a hit if the template is found, else ``None``."""
        ...


@runtime_checkable
class UiRegionDetector(Protocol):
    """Checks one configured UI zone against a frame.

    The pipeline calls this once per zone in ``PerceptionConfig.ui_zones``.
    Implementations return a :class:`UiZoneHit` (a semantic UI state with the
    zone's name, check, bbox, and measured value) when the check fires, or
    ``None`` when it legitimately does not.

    Contract: per-frame operational failures (bad frame, missing optional
    dependency, out-of-bounds region, ...) must *raise* (``PerceptionError``
    is conventional) so the pipeline can record them in
    ``Observation.detector_errors``. Only "the check did not fire" returns
    ``None`` — the pipeline must be able to tell "not present" apart from
    "detector failed".
    """

    def detect(self, frame: Frame, zone: UiZone) -> UiZoneHit | None:
        """Return a hit if the zone's check fired, else ``None``."""
        ...


@runtime_checkable
class OcrEngine(Protocol):
    """Reads text from a single frame region."""

    def read(self, frame: Frame, region: BBox) -> TextRegion | None:
        """Return the recognized text for ``region``, or ``None`` if none."""
        ...


@runtime_checkable
class ObjectDetector(Protocol):
    """Detects objects (e.g. color blobs) across a frame."""

    def match(self, frame: Frame) -> list[ObjectHit]:
        """Return all object hits found in ``frame`` (empty list if none)."""
        ...
