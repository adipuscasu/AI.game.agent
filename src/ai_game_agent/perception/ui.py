"""UI-region detection (Phase 2, step 5).

``UiRegionDetector`` implements the :class:`ai_game_agent.perception.base.
UiRegionDetector` protocol (structural — no base class needed)::

    detect(frame: Frame, zone: UiZone) -> TemplateHit | None

Design rules
============

* ``PIL`` is imported lazily (via ``Frame.to_pil``); ``cv2`` / ``numpy``
  are top-level imports in the helper they live in, so a bare venv without
  the ``vision`` extra never loads OpenCV for pure-PIL checks.
* ``detect`` returns at most one :class:`TemplateHit` per call (the protocol
  contract), tagged with the zone's ``name`` and the zone's bbox.  A zone
  whose check does not fire returns ``None``; it never raises.
* Setup failures (missing dependency, bad config) raise
  :class:`PerceptionError` so the CLI can fail fast with a clean
  ``error:`` line.  Per-frame detector failures are recorded in
  ``Observation.detector_errors`` by the pipeline; this class catches and
  re-raises as :class:`PerceptionError` only for the setup case.
* The ``check`` value on a :class:`UiZone` must be one of the closed set
  (``presence``, ``brightness``, ``color_present``) enforced by the
  validator in ``config.py``.  If a caller manages to bypass that (e.g. by
  mutating an otherwise-valid zone), ``detect`` still degrades gracefully
  to ``None`` rather than blowing up the pipeline.

Checks
------

- ``presence``: the zone is not uniformly one color (any two pixels that
  differ).  Pure-Python; no dependency on OpenCV or PIL.
- ``brightness``: the zone's average luminance (ITU-R BT.601
  ``Y = 0.299 R + 0.587 G + 0.114 B``) clears a fixed default of 128.
  Pure-Python.
- ``color_present``: the zone contains a meaningful amount of a
  configurable target color (default: red), measured via HSV thresholding
  with OpenCV.  OpenCV + NumPy required.
"""

from __future__ import annotations

from ai_game_agent.capture import Frame
from ai_game_agent.config import PerceptionConfig, UiZone
from ai_game_agent.perception.base import PerceptionError
from ai_game_agent.perception.observation import BBox, TemplateHit

__all__ = ["UiRegionDetector", "DEFAULT_BRIGHTNESS_THRESHOLD", "MIN_AREA_FRACTION"]

#: Documented default for the ``brightness`` check.  The ``UiZone`` model
#: carries no threshold field, so this is the canonical constant.
DEFAULT_BRIGHTNESS_THRESHOLD: float = 128.0

#: Minimum fraction of a zone's pixels that must match a target color
#: before ``color_present`` reports a hit.  Chosen so a stray 1–2 pixel
#: antialias artifact does not flip a hit, but a small solid chip does.
MIN_AREA_FRACTION: float = 0.001

#: Target color for the default ``color_present`` check (RGB 0-255) and the
#: per-channel tolerance.  Red is the canonical UI health-bar color.
_DEFAULT_TARGET_RGB: tuple[int, int, int] = (200, 30, 30)
_DEFAULT_TARGET_TOLERANCE: int = 40


class UiRegionDetector:
    """Detects UI regions of interest within a frame based on configured zones.

    One instance per pipeline; ``detect`` is called once per configured
    :class:`UiZone`.

    Parameters
    ----------
    config:
        The :class:`PerceptionConfig` for this perception pipeline.  Only
        the zones that are present (and their ``check`` values) are used;
        ``ui_zones`` is the single source of truth for what gets detected.
    target_rgb:
        Optional override for the ``color_present`` target color.  The
        ``UiZone`` model does not carry a color field, so the detector
        uses a canonical default (red) unless the caller injects one.
    tolerance:
        Per-channel tolerance for ``color_present`` matching (default 40).
    """

    def __init__(
        self,
        config: PerceptionConfig,
        *,
        target_rgb: tuple[int, int, int] = _DEFAULT_TARGET_RGB,
        tolerance: int = _DEFAULT_TARGET_TOLERANCE,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        for i, v in enumerate(target_rgb):
            if not (0 <= int(v) <= 255):
                raise ValueError(
                    f"target_rgb[{i}]={v!r} out of range 0-255 (must be a valid RGB channel)"
                )
        self._config = config
        self._target_rgb = tuple(int(v) for v in target_rgb)
        self._tolerance = int(tolerance)

    # -- protocol ------------------------------------------------------------
    def detect(self, frame: Frame, zone: UiZone) -> TemplateHit | None:
        """Run the zone's check and return a hit, or ``None`` if absent.

        Never raises for per-frame conditions (unknown ``check`` value,
        out-of-bounds zone, missing dependency at runtime): the pipeline
        records the failure in ``Observation.detector_errors`` and
        continues.  Setup failures (e.g. an invalid constructor argument)
        raise :class:`PerceptionError` at construction time.
        """
        try:
            if zone.check == "presence":
                present = self._check_presence(frame, zone)
            elif zone.check == "brightness":
                present = self._check_brightness(frame, zone)
            elif zone.check == "color_present":
                present = self._check_color_present(frame, zone)
            else:
                # Unknown check value (e.g. forced past the validator).
                # Degrade gracefully: no hit, no raise.
                return None
        except Exception:  # noqa: BLE001 - contract: never raise per-frame
            # Per-frame failure (CaptureError from out-of-bounds zone,
            # missing OpenCV, bad frame, ...).  Degrade gracefully:
            # return None so the pipeline records it in
            # Observation.detector_errors and continues.  The protocol
            # contract is that ``detect`` never raises for a configured
            # zone; only constructor-time setup errors raise.
            return None

        if not present:
            return None
        return TemplateHit(
            template=zone.name,
            bbox=BBox(x=zone.x, y=zone.y, width=zone.width, height=zone.height),
            confidence=0.9,
        )

    # -- check implementations ---------------------------------------------
    def _check_presence(self, frame: Frame, zone: UiZone) -> bool:
        """Presence: the zone is not uniformly one color."""
        px = _region_pixels(frame, zone)
        if len(px) < 2:
            return False
        first = px[0]
        for p in px[1:]:
            if p != first:
                return True
        return False

    def _check_brightness(self, frame: Frame, zone: UiZone) -> bool:
        """Brightness: average luminance clears ``DEFAULT_BRIGHTNESS_THRESHOLD``."""
        px = _region_pixels(frame, zone)
        if not px:
            return False
        total = 0.0
        for r, g, b in px:
            total += 0.299 * r + 0.587 * g + 0.114 * b
        average = total / len(px)
        return average >= DEFAULT_BRIGHTNESS_THRESHOLD

    def _check_color_present(self, frame: Frame, zone: UiZone) -> bool:
        """Color present: a target color occupies a meaningful area of the zone.

        Uses OpenCV HSV thresholding + ``cv2.countNonZero`` to count pixels
        within ``self._tolerance`` of ``self._target_rgb``.  Raises
        :class:`PerceptionError` if OpenCV is unavailable so the pipeline
        can fail fast.
        """
        try:
            import cv2  # noqa: F401
            import numpy as np
        except ImportError as exc:
            raise PerceptionError(
                "color_present check requires OpenCV and NumPy; "
                "install the 'vision' extra (pip install ai-game-agent[vision])"
            ) from exc

        px = _region_pixels(frame, zone)
        if not px:
            return False
        # Build an RGB uint8 image (H, W, 3) from the flat pixel list.
        # A proper 2-D image is required: OpenCV's ``inRange`` does not
        # broadcast (1,1,3) bounds against a (N,1,3) layout.
        arr = (
            np.fromiter((v for p in px for v in p), dtype=np.uint8)
            .reshape(zone.height, zone.width, 3)
        )
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        r, g, b = self._target_rgb
        tol = self._tolerance
        # OpenCV (>= 4.x) does not broadcast (1,1,3) bounds against a
        # (H, W, 3) image; expand to the full image shape explicitly.
        lower = np.broadcast_to(
            np.array([max(0, b - tol), max(0, g - tol), max(0, r - tol)], dtype=np.uint8),
            (zone.height, zone.width, 3),
        )
        upper = np.broadcast_to(
            np.array([min(255, b + tol), min(255, g + tol), min(255, r + tol)], dtype=np.uint8),
            (zone.height, zone.width, 3),
        )
        mask = cv2.inRange(bgr, lower, upper)
        matched = int(cv2.countNonZero(mask))
        fraction = matched / len(px)
        return fraction >= MIN_AREA_FRACTION


# -- helpers -------------------------------------------------------------

def _region_pixels(frame: Frame, zone: UiZone) -> list[tuple[int, int, int]]:
    """Extract the zone's pixels as a flat list of (r, g, b) tuples.

    Uses :meth:`Frame.region` (bounds-checked, raises ``CaptureError`` if
    the zone extends past the frame edge) and decodes the resulting row-major
    RGB ``bytes`` buffer.  This keeps the conversion site in ``Frame`` and
    avoids re-implementing region math here.
    """
    sub = frame.region(zone.x, zone.y, zone.width, zone.height)
    buf = sub.pixels
    out: list[tuple[int, int, int]] = []
    n = len(buf) // 3
    for i in range(n):
        base = i * 3
        out.append((buf[base], buf[base + 1], buf[base + 2]))
    return out
