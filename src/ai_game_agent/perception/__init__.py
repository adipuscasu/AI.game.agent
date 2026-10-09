"""The perception layer (Phase 2).

Public re-exports so callers can do::

    from ai_game_agent.perception import Perception, Observation, PerceptionError

Heavy dependencies (OpenCV, OCR engines) are imported lazily by the
implementation modules that need them, so importing this package in a bare
venv is always safe.
"""

from ai_game_agent.perception.base import (
    ObjectDetector,
    OcrEngine,
    PerceptionError,
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
from ai_game_agent.perception.pipeline import Perception

__all__ = [
    "BBox",
    "ObjectDetector",
    "ObjectHit",
    "Observation",
    "OcrEngine",
    "Perception",
    "PerceptionError",
    "TemplateHit",
    "TemplateMatcher",
    "TextRegion",
    "UiRegionDetector",
    "UiZoneHit",
]
