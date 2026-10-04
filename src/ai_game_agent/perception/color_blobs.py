"""Color-blob object detection (Phase 2, step 6).

``ColorBlobsDetector`` implements the :class:`ai_game_agent.perception.base.
ObjectDetector` protocol (structural — no base class needed)::

    match(frame: Frame) -> list[ObjectHit]

It finds blobs of a configured RGB color (within a per-channel tolerance)
using OpenCV thresholding + contour analysis. Design rules:

* ``cv2`` / ``numpy`` are imported at module top — they are required extras,
  and this module is imported only when the subsystem is selected
  (``PerceptionConfig.objects_enabled`` and the color list are non-empty),
  so a bare venv without the extras never loads it.
* A single configured color may produce multiple :class:`ObjectHit`
  entries (one per detected blob) sharing the same ``kind``.
* Per-blob failures are not raised; the pipeline records any exception in
  ``Observation.detector_errors`` and continues.
* Setup failures (missing dependency) raise :class:`PerceptionError` so the
  CLI can fail fast with a clean ``error:`` line.
"""

from __future__ import annotations

from typing import Any

from ai_game_agent.capture import Frame
from ai_game_agent.perception.base import PerceptionError
from ai_game_agent.perception.observation import BBox, ObjectHit

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without extras
    raise PerceptionError(
        "color-blob detection requires OpenCV and NumPy; "
        "install the 'vision' extra (pip install ai-game-agent[vision])"
    ) from exc

__all__ = ["ColorBlobsDetector", "MIN_BLOB_AREA"]

# Contours smaller than this (in pixels^2) are treated as noise and dropped.
MIN_BLOB_AREA = 16

# Confidence ceiling: a blob that covers this fraction of the frame gets 1.0.
_MAX_AREA_FRACTION = 256


class ColorBlobsDetector:
    """Detects colored blobs (e.g. loot glows, hit markers) in a frame.

    Each configured color is an RGB triple with a per-channel tolerance;
    the detector threshold-masks the frame in BGR space and reports every
    contour whose area clears ``MIN_BLOB_AREA`` as an :class:`ObjectHit`.

    Parameters
    ----------
    colors:
        Sequence of color specs, each a dict with keys ``name`` (str),
        ``rgb`` (3-element list/tuple of ints 0-255), and optional
        ``tolerance`` (int, default 40). A ``None`` / empty sequence is
        accepted and simply yields no hits.
    min_area:
        Minimum contour area (px^2) for a blob to count (default 16).
    """

    def __init__(
        self,
        colors: Any,
        *,
        min_area: int = MIN_BLOB_AREA,
    ) -> None:
        if min_area < 1:
            raise ValueError("min_area must be >= 1")
        self._min_area = int(min_area)
        self._colors = _normalize_colors(colors)

    # -- protocol ------------------------------------------------------------
    def match(self, frame: Frame) -> list[ObjectHit]:
        """Return one :class:`ObjectHit` per detected blob (empty list if none)."""
        img = _frame_to_bgr(frame)
        hits: list[ObjectHit] = list()
        for spec in self._colors:
            hits.extend(_match_color(img, spec, self._min_area))
        return hits

    # -- introspection -------------------------------------------------------
    @property
    def colors(self) -> tuple[dict[str, object], ...]:
        """The configured color specs (as plain dicts)."""
        return tuple(dict(c) for c in self._colors)


# -- helpers -------------------------------------------------------------

def _normalize_colors(colors: Any) -> list[dict[str, object]]:
    """Accept a dict, a list of dicts, or an empty value; return a list of
    color-spec dicts with ``name`` / ``rgb`` / ``tolerance`` keys."""
    if colors is None:
        return []
    if isinstance(colors, dict):
        # Accept a mapping name -> spec, or name -> [r, g, b].
        items: list[dict[str, object]] = []
        for name, spec in colors.items():
            items.append(_one_spec(name, spec))
        return items
    if isinstance(colors, (list, tuple)):
        return [_one_spec(c.get("name", f"color_{i}"), c) for i, c in enumerate(colors)]
    raise ValueError(f"unsupported color spec: {colors!r}")


def _one_spec(name: object, spec: Any) -> dict[str, object]:
    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        spec = {"rgb": list(spec)}
    if not isinstance(spec, dict):
        raise ValueError(f"color spec for {name!r} must be a dict or [r, g, b] list")
    rgb = spec.get("rgb")
    if rgb is None:
        raise ValueError(f"color spec for {name!r} missing 'rgb'")
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        raise ValueError(f"color spec for {name!r} has invalid 'rgb'")
    tol = int(spec.get("tolerance", 40))
    if tol < 0:
        raise ValueError(f"color spec for {name!r} has negative tolerance")
    return {
        "name": str(name),
        "rgb": [int(v) for v in rgb],
        "tolerance": tol,
    }


def _frame_to_bgr(frame: Frame) -> np.ndarray:
    """Convert a :class:`Frame` (row-major RGB bytes) to a BGR ``uint8`` array."""
    arr = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    # RGB -> BGR (single conversion site; base types stay RGB).
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _match_color(img: np.ndarray, spec: dict[str, object], min_area: int) -> list[ObjectHit]:
    """Find all blobs of ``spec``'s color in ``img`` (BGR uint8)."""
    rgb = spec["rgb"]  # [r, g, b] ints 0-255
    tol = int(spec["tolerance"])
    bgr = [int(rgb[2]), int(rgb[1]), int(rgb[0])]  # RGB -> BGR
    lower = np.array([max(0, c - tol) for c in bgr], dtype=np.uint8)
    upper = np.array([min(255, c + tol) for c in bgr], dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hits: list[ObjectHit] = []
    frame_area = int(img.shape[0] * img.shape[1]) or 1
    name = str(spec["name"])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        confidence = min(1.0, area / max(1.0, frame_area / _MAX_AREA_FRACTION))
        hits.append(
            ObjectHit(
                kind=name,
                bbox=BBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                confidence=round(confidence, 4),
            )
        )
    return hits
