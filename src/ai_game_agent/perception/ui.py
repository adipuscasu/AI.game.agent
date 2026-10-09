"""UI-region detection (Phase 2, step 5).

``UiRegionDetector`` implements the :class:`ai_game_agent.perception.base.
UiRegionDetector` protocol (structural — no base class needed)::

    detect(frame: Frame, zone: UiZone) -> UiZoneHit | None

Design rules
============

* ``PIL`` is imported lazily (via ``Frame.to_pil``); ``cv2`` / ``numpy``
  are imported inside the check that needs them, so a bare venv without
  the ``vision`` extra never loads OpenCV for pure-Python checks.
* ``detect`` returns at most one :class:`UiZoneHit` per call, tagged with
  the zone's ``name``, the check's identity, the zone's bbox, and a
  normalized measured ``value``.  A zone whose check legitimately does not
  fire returns ``None``.
* **Error contract (pipeline §5.5):** per-frame operational failures —
  out-of-bounds zone, malformed pixel buffer, missing optional dependency —
  raise :class:`PerceptionError`.  The pipeline owns the error boundary:
  it catches the exception, records ``"ui:<zone>: <reason>"`` in
  ``Observation.detector_errors``, and continues with the next zone.
  Returning ``None`` on failure would make "the check did not fire"
  indistinguishable from "the detector failed" — a distinction downstream
  state machines need ("health bar not visible" vs. "perception failed,
  unknown state").
* Setup failures (invalid constructor arguments) raise ``ValueError`` /
  :class:`PerceptionError` at construction time so the CLI can fail fast
  with a clean ``error:`` line.
* The ``check`` value on a :class:`UiZone` must be one of the closed set
  (``presence``, ``brightness``, ``color_present``) enforced by the
  validator in ``config.py``.  If a caller bypasses that with a bogus value,
  ``detect`` raises :class:`PerceptionError` — that is a configuration
  failure, not a "not detected" result.

Checks
------

- ``presence``: the zone is not uniformly one color (any two pixels that
  differ).  Pure-Python; no dependency on OpenCV or PIL.
- ``brightness``: the zone's average luminance (ITU-R BT.601
  ``Y = 0.299 R + 0.587 G + 0.114 B``) clears the zone's threshold
  (default 128).  Pure-Python.
- ``color_present``: the zone contains a meaningful amount of the target
  color (default: red), measured via RGB thresholding with OpenCV.
  OpenCV + NumPy required.
"""

from __future__ import annotations

from ai_game_agent.capture import Frame
from ai_game_agent.config import PerceptionConfig, UiZone
from ai_game_agent.perception.base import PerceptionError
from ai_game_agent.perception.observation import BBox, UiZoneHit

__all__ = ["UiRegionDetector", "DEFAULT_BRIGHTNESS_THRESHOLD", "MIN_AREA_FRACTION"]

#: Documented default for the ``brightness`` check, used when a zone does
#: not specify one.  Luminance scale is 0–255 (BT.601).
DEFAULT_BRIGHTNESS_THRESHOLD: float = 128.0

#: Minimum fraction of a zone's pixels that must match the target color
#: before ``color_present`` reports a hit.  Chosen so a stray 1–2 pixel
#: antialias artifact does not flip a hit, but a small solid chip does.
MIN_AREA_FRACTION: float = 0.001

#: Target color for the ``color_present`` check (RGB 0-255) and the
#: per-channel tolerance, used when neither the config nor the constructor
#: overrides them.  Red is the canonical UI health-bar color.
DEFAULT_TARGET_RGB: tuple[int, int, int] = (200, 30, 30)
DEFAULT_TARGET_TOLERANCE: int = 40


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
        Override for the ``color_present`` target color.  Per-zone
        ``target_rgb`` values (set in config) take precedence over this.
    tolerance:
        Per-channel tolerance for ``color_present`` matching (default 40).
        Per-zone ``tolerance`` values take precedence over this.
    """

    def __init__(
        self,
        config: PerceptionConfig,
        *,
        target_rgb: tuple[int, int, int] = DEFAULT_TARGET_RGB,
        tolerance: int = DEFAULT_TARGET_TOLERANCE,
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
    def detect(self, frame: Frame, zone: UiZone) -> UiZoneHit | None:
        """Run the zone's check; return a :class:`UiZoneHit`, or ``None`` if absent.

        Returns ``None`` *only* for legitimate "not detected" results.  Any
        operational failure (out-of-bounds zone, malformed frame, missing
        optional dependency, unknown check value) raises
        :class:`PerceptionError`; the pipeline records it in
        ``Observation.detector_errors`` and continues with the next zone.
        """
        if zone.check == "presence":
            value = self._check_presence(frame, zone)
        elif zone.check == "brightness":
            value = self._check_brightness(frame, zone)
        elif zone.check == "color_present":
            value = self._check_color_present(frame, zone)
        else:
            # Unknown check value (e.g. forced past the validator).  This is
            # a configuration failure, not a "not detected" result: the
            # pipeline must be able to record it in detector_errors.
            raise PerceptionError(
                f"ui:{zone.name}: unknown check {zone.check!r} "
                "(expected one of: presence, brightness, color_present)"
            )

        if value is None:
            return None
        return UiZoneHit(
            zone=zone.name,
            check=zone.check,
            bbox=BBox(x=zone.x, y=zone.y, width=zone.width, height=zone.height),
            value=value,
        )

    # -- check implementations -----------------------------------------------
    # Each returns the normalized measured value (0.0–1.0) when the check
    # fired, or ``None`` when it legitimately did not.  Operational failures
    # raise (CaptureError / ImportError / ValueError) and are *not* caught
    # here: the pipeline is the error boundary.

    def _check_presence(self, frame: Frame, zone: UiZone) -> float | None:
        """Presence: the zone is not uniformly one color.  Value: 1.0."""
        px = _region_pixels(frame, zone)
        if len(px) < 2:
            return None
        first = px[0]
        for p in px[1:]:
            if p != first:
                return 1.0
        return None

    def _check_brightness(self, frame: Frame, zone: UiZone) -> float | None:
        """Brightness: average luminance clears the zone's threshold.

        Value: average luminance / 255 (how bright the zone is, in [0, 1]).
        Threshold: zone.threshold when set, else ``DEFAULT_BRIGHTNESS_THRESHOLD``.
        """
        px = _region_pixels(frame, zone)
        if not px:
            return None
        total = 0.0
        for r, g, b in px:
            total += 0.299 * r + 0.587 * g + 0.114 * b
        average = total / len(px)
        _th = zone.threshold if zone.threshold is not None else DEFAULT_BRIGHTNESS_THRESHOLD
        threshold = float(_th)
        if average < threshold:
            return None
        return round(min(1.0, average / 255.0), 4)

    def _check_color_present(self, frame: Frame, zone: UiZone) -> float | None:
        """Color present: the target color occupies a meaningful area of the zone.

        Value: the fraction of zone pixels matched (the hit's measured
        evidence, in [0, 1]).  Target color / tolerance come from the zone
        when set, else the detector's constructor defaults.  Raises
        :class:`PerceptionError` if OpenCV is unavailable — a missing
        dependency is a failure the pipeline must record, not a "no hit".
        """
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise PerceptionError(
                "color_present check requires OpenCV and NumPy; "
                "install the 'vision' extra (pip install ai-game-agent[vision])"
            ) from exc

        px = _region_pixels(frame, zone)
        if not px:
            return None

        target = tuple(zone.target_rgb) if zone.target_rgb else self._target_rgb  # type: ignore[assignment]
        tol = int(zone.tolerance if zone.tolerance is not None else self._tolerance)
        if tol < 0:
            raise PerceptionError(f"ui:{zone.name}: tolerance must be >= 0, got {tol}")

        # Build an RGB uint8 image (H, W, 3) from the flat pixel list.
        # A proper 2-D image is required: OpenCV's ``inRange`` does not
        # broadcast (1,1,3) bounds against a (N,1,3) layout.
        arr = (
            np.fromiter((v for p in px for v in p), dtype=np.uint8)
            .reshape(zone.height, zone.width, 3)
        )
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        r, g, b = target
        # OpenCV (>= 4.x) does not broadcast (1,1,3) bounds against an
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
        if fraction < MIN_AREA_FRACTION:
            return None
        return round(min(1.0, fraction), 4)


# -- helpers -------------------------------------------------------------

def _region_pixels(frame: Frame, zone: UiZone) -> list[tuple[int, int, int]]:
    """Extract the zone's pixels as a flat list of (r, g, b) tuples.

    Uses :meth:`Frame.region` (bounds-checked, raises ``CaptureError`` if
    the zone extends past the frame edge) and decodes the resulting row-major
    RGB ``bytes`` buffer.  This keeps the conversion site in ``Frame`` and
    avoids re-implementing region math here.  A CaptureError here is a
    per-frame failure that must propagate to the pipeline, so it is
    deliberately not caught.
    """
    sub = frame.region(zone.x, zone.y, zone.width, zone.height)
    buf = sub.pixels
    out: list[tuple[int, int, int]] = []
    n = len(buf) // 3
    for i in range(n):
        base = i * 3
        out.append((buf[base], buf[base + 1], buf[base + 2]))
    return out
