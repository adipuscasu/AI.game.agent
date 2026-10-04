"""Step 6 tests: ``ColorBlobsDetector`` (perception/color_blobs.py).

These tests drive the **real** ``ColorBlobsDetector`` class against the
``ObjectDetector`` protocol contract in ``perception/base.py``::

    match(frame: Frame) -> list[ObjectHit]

and the real config / observation models.  The detector threshold-masks a
frame for a configured RGB color (within a per-channel tolerance) and reports
every contour whose area clears ``MIN_BLOB_AREA`` as an ``ObjectHit``.

Because ``color_blobs`` imports OpenCV + NumPy at module top (they are the
required ``vision`` extras), the whole file is skipped when those are absent --
matching the design that this module is only loaded when the objects subsystem
is actually selected.

Coverage notes
==============

- ``match`` positive / negative paths (blob found, no matching color, empty).
- Multiple blobs of one kind, and one blob each of two kinds.
- Tolerance boundary (in-range hits, out-of-range misses).
- ``min_area`` filtering (small blob dropped, ``min_area=1`` includes it).
- Confidence is clamped to ``[0.0, 1.0]`` and saturates at ``1.0`` for a
  large blob on a small frame.
- Constructor validation (``min_area``, spec shape, tolerance) and the
  ``colors`` introspection property.
- A wiring guard: ``_build_perception`` must construct a ``ColorBlobsDetector``
  when object colors are configured (mirrors the Step 5 ``ui_detector`` guard).
"""

from __future__ import annotations

import datetime as _dt
import io
import struct
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2", reason="color_blobs requires OpenCV (vision extra)")
np = pytest.importorskip("numpy", reason="color_blobs requires NumPy (vision extra)")

from ai_game_agent.capture import Frame
from ai_game_agent.config import PerceptionConfig
from ai_game_agent.perception.color_blobs import ColorBlobsDetector
from ai_game_agent.perception.observation import BBox  # noqa: F401  (document contract)


# --- frame helpers -----------------------------------------------------------

def _frame(width: int, height: int, pixel_at) -> Frame:
    """Build a real :class:`Frame` from a per-pixel callable.

    ``pixel_at(x, y)`` returns an ``(r, g, b)`` tuple (int 0-255).  ``pixels``
    is the row-major RGB ``bytes`` layout ``Frame`` expects.
    """
    buf = io.BytesIO()
    for y in range(height):
        for x in range(width):
            r, g, b = pixel_at(x, y)
            buf.write(struct.pack("BBB", r, g, b))
    return Frame(
        width, height, buf.getvalue(), _dt.datetime.now(_dt.timezone.utc), "test"
    )


def _solid(width: int, height: int, rgb) -> Frame:
    r, g, b = rgb
    return _frame(width, height, lambda x, y: (r, g, b))


def _planted(width, height, bg, blob_rgb, x0, y0, size) -> Frame:
    """A solid ``bg`` frame with a ``size``x``size`` square of ``blob_rgb``
    whose top-left corner is at ``(x0, y0)``."""

    def px(x, y):
        if x0 <= x < x0 + size and y0 <= y < y0 + size:
            return blob_rgb
        return bg

    return _frame(width, height, px)


def _det(colors, *, min_area=None):
    if min_area is None:
        return ColorBlobsDetector(colors)
    return ColorBlobsDetector(colors, min_area=min_area)


BG = (0, 0, 0)
RED = (200, 30, 30)


# --- match() -----------------------------------------------------------------

class TestMatch:
    def test_finds_planted_blob(self):
        frame = _planted(32, 32, BG, RED, 4, 4, 10)
        det = _det({"loot": {"rgb": [200, 30, 30], "tolerance": 40}})
        hits = det.match(frame)
        assert len(hits) == 1
        hit = hits[0]
        assert hit.kind == "loot"
        b = hit.bbox
        assert isinstance(b, BBox)
        # bbox must cover the planted square (top-left at/inside, right/bottom
        # at/after) -- tolerant to a one-pixel cv2 rounding difference.
        assert b.x <= 4 and b.y <= 4
        assert b.x + b.width >= 14
        assert b.y + b.height >= 14
        assert 0.0 < hit.confidence <= 1.0

    def test_no_matching_color_returns_empty(self):
        frame = _solid(32, 32, (30, 30, 200))  # blue, not the configured red
        det = _det({"loot": {"rgb": [200, 30, 30], "tolerance": 20}})
        assert det.match(frame) == []

    def test_black_background_only_no_hit(self):
        frame = _solid(32, 32, BG)
        det = _det({"loot": {"rgb": [200, 30, 30], "tolerance": 40}})
        assert det.match(frame) == []

    def test_multiple_blobs_same_kind(self):
        def px(x, y):
            if 2 <= x < 12 and 2 <= y < 12:
                return RED
            if 20 <= x < 30 and 20 <= y < 30:
                return RED
            return BG

        frame = _frame(32, 32, px)
        det = _det({"loot": {"rgb": [200, 30, 30], "tolerance": 40}})
        hits = det.match(frame)
        assert len(hits) == 2
        assert all(h.kind == "loot" for h in hits)
        (a, c) = sorted((h.bbox for h in hits), key=lambda bb: (bb.x, bb.y))
        assert (a.x, a.y) != (c.x, c.y)

    def test_two_colors_two_kinds(self):
        def px(x, y):
            if 2 <= x < 12 and 2 <= y < 12:
                return (200, 30, 30)  # red
            if 20 <= x < 30 and 20 <= y < 30:
                return (30, 200, 30)  # green
            return BG

        frame = _frame(32, 32, px)
        det = _det([
            {"name": "red", "rgb": [200, 30, 30], "tolerance": 20},
            {"name": "green", "rgb": [30, 200, 30], "tolerance": 20},
        ])
        hits = det.match(frame)
        assert sorted(h.kind for h in hits) == ["green", "red"]

    def test_within_tolerance_hits(self):
        # blob is 20 off on the red channel, tolerance 20 -> in range
        frame = _planted(32, 32, BG, (180, 20, 20), 6, 6, 10)
        det = _det({"loot": {"rgb": [200, 30, 30], "tolerance": 20}})
        assert len(det.match(frame)) == 1

    def test_outside_tolerance_no_hit(self):
        # blob is 40 off on the red channel, tolerance 20 -> out of range
        frame = _planted(32, 32, BG, (240, 70, 70), 6, 6, 10)
        det = _det({"loot": {"rgb": [200, 30, 30], "tolerance": 20}})
        assert det.match(frame) == []

    def test_confidence_clamped_to_one_for_large_blob(self):
        # a large blob on a small frame must saturate confidence at 1.0
        frame = _planted(32, 32, BG, RED, 0, 0, 28)
        det = _det({"loot": {"rgb": [200, 30, 30], "tolerance": 40}})
        hits = det.match(frame)
        assert len(hits) == 1
        assert hits[0].confidence == 1.0

    def test_min_area_filters_small_blobs(self):
        # a 3x3 blob (contourArea ~4) is below the default MIN_BLOB_AREA (16)
        small = _planted(32, 32, BG, RED, 4, 4, 3)
        assert _det({"loot": {"rgb": [200, 30, 30], "tolerance": 40}}).match(small) == []
        # a large min_area excludes even a 10x10 blob
        big = _planted(32, 32, BG, RED, 4, 4, 10)
        assert _det({"loot": {"rgb": [200, 30, 30], "tolerance": 40}},
                    min_area=100000).match(big) == []
        # and a low min_area includes the small blob
        assert len(_det({"loot": {"rgb": [200, 30, 30], "tolerance": 40}},
                        min_area=1).match(small)) == 1


# --- constructor / introspection --------------------------------------------

class TestConstructor:
    def test_min_area_must_be_positive(self):
        with pytest.raises(ValueError):
            ColorBlobsDetector({"c": {"rgb": [1, 2, 3]}}, min_area=0)

    def test_none_colors_yields_no_hits(self):
        det = ColorBlobsDetector(None)
        assert det.match(_solid(16, 16, RED)) == []
        assert det.colors == ()

    def test_empty_list_yields_no_hits(self):
        det = ColorBlobsDetector([])
        assert det.match(_solid(16, 16, RED)) == []
        assert det.colors == ()

    def test_missing_rgb_raises(self):
        with pytest.raises(ValueError):
            ColorBlobsDetector({"c": {"tolerance": 10}})

    def test_invalid_rgb_raises(self):
        with pytest.raises(ValueError):
            ColorBlobsDetector({"c": {"rgb": [1, 2]}})

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            ColorBlobsDetector({"c": {"rgb": [1, 2, 3], "tolerance": -1}})

    def test_unsupported_spec_raises(self):
        with pytest.raises(ValueError):
            ColorBlobsDetector(42)

    def test_rgb_triple_shortcut(self):
        # name -> [r, g, b] (no 'rgb' key) is accepted
        det = ColorBlobsDetector({"loot": [200, 30, 30]})
        (spec,) = det.colors
        assert spec["name"] == "loot"
        assert spec["rgb"] == [200, 30, 30]
        assert spec["tolerance"] == 40  # default

    def test_colors_property_reflects_specs(self):
        det = ColorBlobsDetector([
            {"name": "a", "rgb": [1, 2, 3], "tolerance": 5},
            {"name": "b", "rgb": [4, 5, 6]},
        ])
        assert det.colors == (
            {"name": "a", "rgb": [1, 2, 3], "tolerance": 5},
            {"name": "b", "rgb": [4, 5, 6], "tolerance": 40},
        )


# --- wiring guard ------------------------------------------------------------

def _perception_cfg(objects_enabled: bool, object_colors: tuple) -> PerceptionConfig:
    return PerceptionConfig(
        enabled=True,
        template_threshold=0.8,
        templates={},
        ui_zones=(),
        ocr_enabled=False,
        ocr_regions=(),
        objects_enabled=objects_enabled,
        object_colors=object_colors,
    )


def _args(**overrides):
    base = {"no_templates": False, "no_ocr": False, "no_objects": False}
    base.update(overrides)
    return SimpleNamespace(**base)


class TestBuildPerceptionWiring:
    """Guard: ``_build_perception`` must wire a ``ColorBlobsDetector`` when
    object colors are configured (mirrors the Step 5 ``ui_detector`` guard)."""

    def test_wires_object_detector_when_colors_configured(self):
        from ai_game_agent.__main__ import _build_perception
        from ai_game_agent.perception.pipeline import Perception

        perception_cfg = _perception_cfg(
            True, ({"name": "loot", "rgb": [200, 30, 30], "tolerance": 40},)
        )
        perception = _build_perception(_args(), SimpleNamespace(perception=perception_cfg))
        assert isinstance(perception, Perception)
        assert perception._object_detector is not None

    def test_no_objects_flag_disables_detector(self):
        from ai_game_agent.__main__ import _build_perception

        perception_cfg = _perception_cfg(
            True, ({"name": "loot", "rgb": [200, 30, 30], "tolerance": 40},)
        )
        perception = _build_perception(
            _args(no_objects=True), SimpleNamespace(perception=perception_cfg)
        )
        assert perception._object_detector is None

    def test_no_colors_yields_no_detector(self):
        from ai_game_agent.__main__ import _build_perception

        perception_cfg = _perception_cfg(False, ())
        perception = _build_perception(_args(), SimpleNamespace(perception=perception_cfg))
        assert perception._object_detector is None
