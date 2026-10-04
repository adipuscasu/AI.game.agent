"""Step 5 tests: ``UiRegionDetector`` (perception/ui.py).

These tests drive the **real** ``UiRegionDetector`` class against the
``UiRegionDetector`` protocol contract in ``perception/base.py``::

    detect(frame: Frame, zone: UiZone) -> UiZoneHit | None

and the real ``UiZone`` config model in ``ai_game_agent.config``.

Real contract (verified against ``config.py`` / ``perception/observation.py``):

- ``UiZone(name, x, y, width, height, check, threshold=None, target_rgb=None,
  tolerance=None)`` -- positional + optional per-zone overrides, immutable,
  with ``check`` restricted to the closed set
  ``{presence, brightness, color_present}``.
- ``UiZoneHit(zone, check, bbox, value)`` and ``BBox(x, y, width, height)``.
- ``Frame`` (from ``ai_game_agent.capture``) has ``width``, ``height`` and
  ``pixels`` as a row-major RGB ``bytes`` buffer.

Design notes
============

- ``presence`` and ``brightness`` are pure-Python (no OpenCV or PIL needed
  beyond ``Frame.pixels``), so these tests run headless with no template
  assets.
- ``color_present`` uses OpenCV RGB thresholding; those paths are exercised
  by the OpenCV-dependent tests and skipped cleanly when ``cv2`` is absent.
- Per-frame operational failures (unknown ``check`` value, out-of-bounds
  zone, missing OpenCV) **raise** :class:`PerceptionError` so the pipeline
  can record them in ``Observation.detector_errors``.  Only a legitimate
  "check did not fire" result returns ``None``.
- ``UiZone`` now carries optional per-zone ``threshold`` / ``target_rgb`` /
  ``tolerance`` overrides; when unset the detector uses its documented
  defaults (brightness 128, red 200/30/30, tolerance 40, min area 0.001).
"""

from __future__ import annotations

import datetime as _dt
import io
import struct
from types import SimpleNamespace

import pytest

from ai_game_agent.__main__ import _build_perception
from ai_game_agent.capture import Frame
from ai_game_agent.config import PerceptionConfig, UiZone
from ai_game_agent.perception.base import UiRegionDetector
from ai_game_agent.perception.observation import UiZoneHit
from ai_game_agent.perception.pipeline import Perception
from ai_game_agent.perception.ui import UiRegionDetector as RealUiRegionDetector

# Confirm we are actually testing the real implementation, not a local
# re-implementation. (The Step 5 draft of this test file defined its own
# ``UiRegionDetector`` class; that is the bug this rewrite fixes.)
assert RealUiRegionDetector is not None


def _frame(width: int, height: int, pixel_at) -> Frame:
    """Build a real :class:`Frame` from a per-pixel callable.

    ``pixel_at(x, y)`` returns an ``(r, g, b)`` tuple (int 0-255).
    ``pixels`` is the row-major RGB ``bytes`` layout ``Frame`` expects.
    """
    buf = io.BytesIO()
    for y in range(height):
        for x in range(width):
            r, g, b = pixel_at(x, y)
            buf.write(struct.pack("BBB", r, g, b))
    return Frame(
        width, height, buf.getvalue(), _dt.datetime.now(_dt.UTC), "test"
    )


def _solid(width: int, height: int, rgb) -> Frame:
    r, g, b = rgb
    return _frame(width, height, lambda x, y: (r, g, b))


def _checkerboard(width: int, height: int, light, dark) -> Frame:
    def px(x, y):
        return light if (x + y) % 8 == 0 else dark

    return _frame(width, height, px)


def _config(*zones: UiZone) -> PerceptionConfig:
    """Build a real ``PerceptionConfig`` with the given zones enabled."""
    return PerceptionConfig(
        enabled=True,
        template_threshold=0.8,
        templates={},
        ui_zones=tuple(zones),
        ocr_enabled=False,
        ocr_regions=(),
        objects_enabled=False,
        object_colors=(),
    )


def _detector(config: PerceptionConfig) -> UiRegionDetector:
    """Construct the real detector from a config (the documented ctor)."""
    return RealUiRegionDetector(config)


def _zone(name: str, x: int, y: int, w: int, h: int, check: str) -> UiZone:
    return UiZone(name, x, y, w, h, check)


# --- constructor / protocol conformance ------------------------------------

class TestConstruction:
    def test_detect_satisfies_protocol(self):
        """The real class must satisfy the protocol's method contract."""
        cfg = _config(_zone("menu", 0, 0, 100, 100, "presence"))
        det = _detector(cfg)
        assert hasattr(det, "detect")
        assert callable(det.detect)

    def test_detect_returns_UiZoneHit_or_None(self):
        """Signature: detect(frame, zone) -> UiZoneHit | None."""
        cfg = _config(_zone("menu", 0, 0, 64, 36, "presence"))
        det = _detector(cfg)
        frame = _solid(128, 72, (0, 0, 0))
        result = det.detect(frame, cfg.ui_zones[0])
        assert result is None or isinstance(result, UiZoneHit)


# --- presence check ---------------------------------------------------------

class TestPresence:
    def test_non_uniform_region_reports_presence(self):
        """A region that is not uniformly one color is 'present'."""
        zone = _zone("hp_bar", 0, 0, 64, 36, "presence")
        det = _detector(_config(zone))
        frame = _checkerboard(128, 72, (255, 255, 255), (10, 10, 10))
        hit = det.detect(frame, zone)
        assert hit is not None, "a non-uniform region should be detected as present"
        assert hit.zone == "hp_bar"
        assert hit.value > 0.0
        assert hit.bbox is not None

    def test_uniform_region_reports_absent(self):
        """A region that is a single solid color is 'absent'."""
        zone = _zone("ghost", 0, 0, 64, 36, "presence")
        det = _detector(_config(zone))
        frame = _solid(128, 72, (50, 60, 70))
        assert det.detect(frame, zone) is None

    def test_hit_bbox_matches_zone_region(self):
        zone = _zone("menu", 10, 20, 90, 52, "presence")
        det = _detector(_config(zone))
        frame = _solid(128, 72, (0, 0, 0))
        # Introduce a single non-uniform pixel inside the zone.
        frame = _frame(128, 72, lambda x, y: (255, 0, 0) if (x, y) == (15, 25) else (0, 0, 0))
        hit = det.detect(frame, zone)
        assert hit is not None
        assert (hit.bbox.x, hit.bbox.y, hit.bbox.width, hit.bbox.height) == (
            10, 20, 90, 52
        )


# --- brightness check -------------------------------------------------------

class TestBrightness:
    def test_bright_region_reports_hit(self):
        zone = _zone("bright_btn", 0, 0, 32, 32, "brightness")
        det = _detector(_config(zone))
        frame = _solid(128, 72, (220, 220, 220))
        hit = det.detect(frame, zone)
        assert hit is not None
        assert hit.zone == "bright_btn"

    def test_dark_region_reports_none(self):
        zone = _zone("dark_bg", 0, 0, 32, 32, "brightness")
        det = _detector(_config(zone))
        frame = _solid(128, 72, (20, 20, 20))
        assert det.detect(frame, zone) is None

    def test_threshold_defaults_to_128(self):
        """``UiZone`` carries no threshold, so the detector must use 128."""
        zone = _zone("mid", 0, 0, 32, 32, "brightness")
        det = _detector(_config(zone))
        just_below = _solid(128, 72, (127, 127, 127))
        assert det.detect(just_below, zone) is None
        just_above = _solid(128, 72, (129, 129, 129))
        assert det.detect(just_above, zone) is not None


# --- error isolation --------------------------------------------------------

class TestErrorIsolation:
    def test_unknown_check_raises(self):
        """A ``check`` outside the closed set must raise, not return ``None``.

        ``UiZone.__init__`` enforces the closed set, so we build a valid
        zone, then force an invalid ``check`` past the validator with
        ``object.__setattr__``.  The detector must raise
        :class:`PerceptionError` so the pipeline records it in
        ``detector_errors``.
        """
        zone = _zone("weird", 0, 0, 32, 32, "presence")
        object.__setattr__(zone, "check", "not_a_real_check")
        det = _detector(_config(_zone("other", 0, 0, 32, 32, "presence")))
        frame = _solid(128, 72, (0, 0, 0))
        from ai_game_agent.perception.base import PerceptionError
        with pytest.raises(PerceptionError, match="unknown check"):
            det.detect(frame, zone)

    def test_detect_never_raises_on_empty_frame(self):
        det = _detector(_config(_zone("x", 0, 0, 1, 1, "presence")))
        frame = _frame(128, 72, lambda x, y: (0, 0, 0))
        result = det.detect(frame, _zone("x", 0, 0, 1, 1, "presence"))
        assert result is None or isinstance(result, UiZoneHit)


# --- color_present check (OpenCV required) ---------------------------------

cv2 = pytest.importorskip("cv2", reason="color_present tests require OpenCV")


class TestColorPresent:
    def test_red_region_reports_hit(self):
        zone = _zone("red_health", 0, 0, 64, 36, "color_present")
        det = _detector(_config(zone))
        frame = _solid(128, 72, (200, 30, 30))  # clearly red
        hit = det.detect(frame, zone)
        assert hit is not None
        assert hit.zone == "red_health"

    def test_blue_region_is_not_red(self):
        zone = _zone("red_health", 0, 0, 64, 36, "color_present")
        det = _detector(_config(zone))
        frame = _solid(128, 72, (30, 30, 200))  # blue, not red
        assert det.detect(frame, zone) is None

    def test_detect_never_raises_on_color_present(self):
        zone = _zone("c", 0, 0, 8, 8, "color_present")
        det = _detector(_config(zone))
        frame = _solid(128, 72, (10, 10, 10))
        result = det.detect(frame, zone)
        assert result is None or isinstance(result, UiZoneHit)


# --- wiring: _build_perception must actually wire the UI detector ----------
#
# The Step 5 bug: ``_build_perception`` reads ``cfg.perception.ui_zones``
# (and can even toggle them off via ``--no-templates``) but then hardcodes
# ``ui_detector=None`` in the ``Perception`` it returns.  So a configured
# zone never runs -- the detector is dead code.  These tests pin the wiring.


def _args(**overrides):
    """A minimal ``args`` namespace matching what ``_build_perception`` reads."""
    base = {"no_templates": False, "no_ocr": False, "no_objects": False}
    base.update(overrides)
    return SimpleNamespace(**base)


def _cfg_with_zones(*zones):
    perception_cfg = PerceptionConfig(
        enabled=True,
        template_threshold=0.8,
        templates={},
        ui_zones=tuple(zones),
        ocr_enabled=False,
        ocr_regions=(),
        objects_enabled=False,
        object_colors=(),
    )
    return SimpleNamespace(perception=perception_cfg)


class TestBuildPerceptionWiring:
    def test_wires_ui_detector_when_zones_configured(self):
        """Configured ``ui_zones`` must yield a non-None ``ui_detector``."""
        zone = UiZone("action_bar", 0, 0, 32, 32, "presence")
        perception = _build_perception(_args(), _cfg_with_zones(zone))
        assert isinstance(perception, Perception)
        assert perception._ui_detector is not None, (
            "_build_perception must construct a UiRegionDetector when "
            "ui_zones are configured; it is currently hardcoded to None"
        )

    def test_no_templates_flag_disables_ui_detector(self):
        """``--no-templates`` must clear the UI detector even with zones set."""
        zone = UiZone("action_bar", 0, 0, 32, 32, "presence")
        perception = _build_perception(
            _args(no_templates=True), _cfg_with_zones(zone)
        )
        assert perception._ui_detector is None

    def test_no_zones_yields_no_ui_detector(self):
        """With no configured zones there is nothing to detect."""
        perception = _build_perception(_args(), _cfg_with_zones())
        assert perception._ui_detector is None

    def test_wired_detector_produces_hit_on_observe(self):
        """End-to-end: a configured presence zone that fires must appear in
        ``observation.templates``.  This is the observable symptom of the bug."""
        zone = UiZone("action_bar", 0, 0, 32, 32, "presence")
        perception = _build_perception(_args(), _cfg_with_zones(zone))
        # A non-uniform frame (checkerboard) so the ``presence`` check fires.
        frame = _checkerboard(128, 72, (200, 200, 200), (20, 20, 20))
        observation = perception.observe(frame)
        hit_zones = [u.zone for u in observation.ui_zones]
        assert "action_bar" in hit_zones, (
            f"configured presence zone should produce a hit; "
            f"got ui_zones={hit_zones}, errors={observation.detector_errors}"
        )
