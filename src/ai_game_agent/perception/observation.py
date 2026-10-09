"""The Phase 2 perception contract: the stable ``Observation`` output type.

``Observation`` is the single output of the perception layer. Everything
downstream — the Phase 4 state machine, the Phase 5 VLM context — consumes it,
so it is deliberately simple, immutable, and JSON-serializable.

Design rules (see ``docs/phase-2-implementation-plan.md`` §4):

* Frozen dataclasses, tuple collections — mirrors Phase 1's ``Frame``:
  hashable, no accidental mutation.
* ``detector_errors`` — a per-detector failure is recorded as a string and
  never fails the pipeline (AGENTS.md ``PERCEPTION_ERROR`` is logged, not
  raised through the pipeline).
* ``to_dict()`` / ``to_json()`` — for logging, replay, and later AI context;
  ``to_dict()`` is always JSON-serializable (ISO timestamps, plain values).
* Coordinates are in *frame space* (after Phase 1 region/scale), not screen
  space — keeps the contract independent of capture configuration.
* ``schema_version`` — a cheap guard so later consumers can detect shape drift.

This module must stay dependency-free (no numpy / OpenCV) so the core package
remains importable with zero extras installed.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass

__all__ = [
    "BBox",
    "Observation",
    "ObjectHit",
    "TemplateHit",
    "TextRegion",
    "UiZoneHit",
]

# Bump if the serialized shape below changes in a way consumers must handle.
# v2: UI-zone detections moved out of ``templates`` into their own ``ui_zones``
# collection (see ``UiZoneHit``) — template matches and semantic UI state are
# now distinct concepts instead of one conflated list.
SCHEMA_VERSION = 2


def _validate_bounding_box(*, x: int, y: int, width: int, height: int) -> None:
    if width < 0:
        raise ValueError(f"width must be >= 0, got {width}")
    if height < 0:
        raise ValueError(f"height must be >= 0, got {height}")


def _validate_confidence(confidence: float) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}")


@dataclass(frozen=True)
class BBox:
    """An axis-aligned bounding box in frame space (integer pixels).

    ``x``/``y`` is the top-left corner; ``width``/``height`` are the extents.
    A zero-sized box is valid (e.g. a detected point); negative extents are not.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        _validate_bounding_box(x=self.x, y=self.y, width=self.width, height=self.height)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class TemplateHit:
    """A template (or UI-zone) match, with its position and confidence."""

    template: str
    bbox: BBox
    confidence: float

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "template": self.template,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TextRegion:
    """OCR output for a single configured text region."""

    text: str
    bbox: BBox
    confidence: float

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ObjectHit:
    """A detected object (e.g. a color blob), with its position and confidence."""

    kind: str
    bbox: BBox
    confidence: float

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class UiZoneHit:
    """A configured UI zone whose check fired (a *semantic UI state*).

    This is deliberately distinct from :class:`TemplateHit`: a template match
    says "this image was found here"; a UI-zone hit says "this part of the game
    UI is in this state" (e.g. the health bar is bright, an action bar is
    present). Downstream state machines treat those differently, so they are
    separate collections on :class:`Observation` rather than one conflated list.

    ``value`` is the measured quantity that triggered the check, normalized to
    ``[0.0, 1.0]`` where applicable (e.g. ``brightness`` → average luminance
    / 255; ``color_present`` → matched pixel fraction; ``presence`` → ``1.0``).
    It is the observable evidence, not a classifier confidence score.
    """

    zone: str
    check: str
    bbox: BBox
    value: float

    def __post_init__(self) -> None:
        _validate_confidence(self.value)  # value is constrained to [0.0, 1.0]

    def to_dict(self) -> dict[str, object]:
        return {
            "zone": self.zone,
            "check": self.check,
            "bbox": self.bbox.to_dict(),
            "value": self.value,
        }


@dataclass(frozen=True)
class Observation:
    """The single stable output of the perception layer.

    All hit collections are tuples (immutable, hashable). ``detector_errors``
    carries per-detector failure strings and never indicates a pipeline-level
    failure — the observation is still valid with a populated ``detector_errors``.
    """

    frame_width: int
    frame_height: int
    captured_at: datetime.datetime
    source: str
    templates: tuple[TemplateHit, ...]
    ui_zones: tuple[UiZoneHit, ...]
    text_regions: tuple[TextRegion, ...]
    objects: tuple[ObjectHit, ...]
    detector_errors: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "captured_at": self.captured_at.isoformat(),
            "source": self.source,
            "templates": [t.to_dict() for t in self.templates],
            "ui_zones": [u.to_dict() for u in self.ui_zones],
            "text_regions": [t.to_dict() for t in self.text_regions],
            "objects": [o.to_dict() for o in self.objects],
            "detector_errors": list(self.detector_errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
