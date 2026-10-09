"""Red-phase tests for the Phase 2 OCR engine (step 7).

These tests define the contract for ``ai_game_agent.perception.ocr``:

* ``TesseractEngine`` — the real implementation of the ``OcrEngine``
  protocol (``base.py``). It wraps a pluggable Tesseract *backend* so tests
  never need a real Tesseract binary; production wires the backend through
  ``pytesseract`` (the ``ocr`` extra).
* Region cropping — the engine must crop the frame to the requested
  ``BBox`` before handing pixels to the backend, and must clamp boxes that
  exceed frame bounds.
* Text parsing — backend raw output is normalized into ``TextRegion``
  (whitespace-stripped, confidence clamped, bbox in frame space).
* Failure policy — a backend that raises must propagate (the ``Perception``
  pipeline is responsible for catching per-region errors into
  ``Observation.detector_errors``); the engine itself must not swallow
  errors.

The tests import the real module and never touch OpenCV/Tesseract, so they
run in a bare venv.
"""

from __future__ import annotations

import datetime
from typing import Protocol

import pytest

from ai_game_agent.capture import Frame
from ai_game_agent.perception import (
    BBox,
    OcrEngine,
    Perception,
    TextRegion,
)

# The real module under test. Importing it must not require any extras.
from ai_game_agent.perception.ocr import TesseractEngine

TS = datetime.datetime(2026, 10, 3, 12, 0, 0, tzinfo=datetime.UTC)


# ---------------------------------------------------------------------------
# Test backend — a fake Tesseract that records the pixels it is given.
# ---------------------------------------------------------------------------


class FakeTesseractBackend(Protocol):
    def image_to_data(
        self, image: object, *, config: str | None = None
    ) -> str: ...


class RecordingBackend:
    """Returns a canned ``image_to_data`` TSV and records the image given."""

    def __init__(self, data: str = "") -> None:
        self.data = data
        self.last_image: object | None = None
        self.last_config: str | None = None

    def image_to_data(self, image: object, *, config: str | None = None) -> str:
        self.last_image = image
        self.last_config = config
        return self.data


class ExplodingBackend:
    """Always raises, to verify the engine does not swallow errors."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("tesseract exploded")

    def image_to_data(self, image: object, *, config: str | None = None) -> str:
        raise self.exc


def _frame(width: int = 100, height: int = 60) -> Frame:
    # A 100x60 RGB frame: each pixel is (0, 0, 0) so bytes are deterministic.
    return Frame(
        width=width,
        height=height,
        pixels=bytes(width * height * 3),
        captured_at=TS,
        source="mock",
    )


def _bbox(x: int = 10, y: int = 20, width: int = 40, height: int = 20) -> BBox:
    return BBox(x=x, y=y, width=width, height=height)


def _tesseract_data(text: str, conf: str = "95.5") -> str:
    """Build a minimal Tesseract TSV (level 5 = word) matching the header."""
    header = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
    )
    word = f"5\t1\t1\t1\t1\t1\t10\t10\t100\t20\t{conf}\t{text}\n"
    return header + word


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestConformance:
    def test_engine_conforms_to_ocr_engine_protocol(self) -> None:
        engine = TesseractEngine(RecordingBackend())
        assert isinstance(engine, OcrEngine)

    def test_engine_exposes_read_method(self) -> None:
        engine = TesseractEngine(RecordingBackend())
        assert callable(getattr(engine, "read", None))


# ---------------------------------------------------------------------------
# Basic read path
# ---------------------------------------------------------------------------


class TestReadBasic:
    def test_read_returns_text_region_for_recognized_text(self) -> None:
        backend = RecordingBackend(data=_tesseract_data("GOLD"))
        engine = TesseractEngine(backend)
        region = engine.read(_frame(), _bbox())
        assert region is not None
        assert region.text == "GOLD"

    def test_read_bbox_is_in_frame_space_not_crop_space(self) -> None:
        """The returned bbox must be the *requested* region (frame space)."""
        backend = RecordingBackend(data=_tesseract_data("HP"))
        engine = TesseractEngine(backend)
        requested = BBox(x=30, y=40, width=16, height=8)
        region = engine.read(_frame(width=200, height=100), requested)
        assert region is not None
        assert region.bbox == requested

    def test_read_confidence_clamped_to_unit_interval(self) -> None:
        backend = RecordingBackend(data=_tesseract_data("x", conf="101.0"))
        engine = TesseractEngine(backend)
        region = engine.read(_frame(), _bbox())
        assert region is not None
        assert 0.0 <= region.confidence <= 1.0

    def test_read_low_confidence_normalized(self) -> None:
        """Tesseract conf is a 0-100 float; engine must divide by 100."""
        backend = RecordingBackend(data=_tesseract_data("x", conf="42.0"))
        engine = TesseractEngine(backend)
        region = engine.read(_frame(), _bbox())
        assert region is not None
        assert region.confidence == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Empty / no-text results
# ---------------------------------------------------------------------------


class TestReadEmpty:
    def test_read_returns_none_when_no_words(self) -> None:
        backend = RecordingBackend(data=_tesseract_data("") + "")
        # Header-only TSV (no word rows) means no recognized text.
        backend.data = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
            "left\ttop\twidth\theight\tconf\ttext\n"
        )
        engine = TesseractEngine(backend)
        assert engine.read(_frame(), _bbox()) is None

    def test_read_returns_none_when_text_is_whitespace_only(self) -> None:
        backend = RecordingBackend(data=_tesseract_data("   "))
        engine = TesseractEngine(backend)
        assert engine.read(_frame(), _bbox()) is None


# ---------------------------------------------------------------------------
# Region cropping
# ---------------------------------------------------------------------------


class TestCropping:
    def test_backend_receives_cropped_pixels(self) -> None:
        backend = RecordingBackend(data=_tesseract_data("x"))
        engine = TesseractEngine(backend)
        frame = _frame(width=100, height=60)
        engine.read(frame, BBox(x=10, y=20, width=30, height=15))
        img = backend.last_image
        assert img is not None
        # The crop must be 30x15 — not the full 100x60.
        assert hasattr(img, "size")
        w, h = img.size
        assert (w, h) == (30, 15)

    def test_crop_clamps_to_frame_bounds(self) -> None:
        """A bbox that overflows the frame must be clamped, not raise."""
        backend = RecordingBackend(data=_tesseract_data("x"))
        engine = TesseractEngine(backend)
        frame = _frame(width=20, height=10)
        # Request a box that starts in-frame but overflows the right/bottom.
        region = engine.read(frame, BBox(x=15, y=8, width=50, height=50))
        # Should not raise; whatever it returns is fine as long as no error.
        # (Empty text → None, or a TextRegion with clamped extent.)
        assert region is None or isinstance(region, TextRegion)

    def test_crop_zero_area_after_clamp_returns_none(self) -> None:
        """A bbox entirely outside the frame has zero area → no read."""
        backend = RecordingBackend(data=_tesseract_data("x"))
        engine = TesseractEngine(backend)
        frame = _frame(width=20, height=10)
        result = engine.read(frame, BBox(x=100, y=100, width=10, height=10))
        assert result is None


# ---------------------------------------------------------------------------
# Error propagation (pipeline is responsible for catching)
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    def test_backend_exception_propagates(self) -> None:
        engine = TesseractEngine(ExplodingBackend(RuntimeError("no tesseract")))
        with pytest.raises(RuntimeError, match="no tesseract"):
            engine.read(_frame(), _bbox())

    def test_pipeline_records_ocr_error_not_fatal(self) -> None:
        """A failing engine must land in ``detector_errors``, not crash."""
        from ai_game_agent.config import PerceptionConfig

        engine = TesseractEngine(ExplodingBackend(RuntimeError("boom")))
        config = PerceptionConfig(
            ocr_enabled=True,
            ocr_regions=(
                {"x": 0, "y": 0, "width": 10, "height": 5, "name": "hp"},
            ),
        )
        pipeline = Perception(config=config, ocr_engine=engine)
        obs = pipeline.observe(_frame())
        assert obs.text_regions == ()
        assert any("hp" in e for e in obs.detector_errors)


# ---------------------------------------------------------------------------
# Multiple words → single joined region
# ---------------------------------------------------------------------------


class TestMultiWord:
    def test_multiple_words_joined_into_single_text(self) -> None:
        header = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
            "left\ttop\twidth\theight\tconf\ttext\n"
        )
        words = (
            "5\t1\t1\t1\t1\t1\t5\t5\t20\t10\t90.0\tHP\n"
            "5\t1\t1\t1\t1\t2\t30\t5\t20\t10\t92.0\t100\n"
        )
        backend = RecordingBackend(data=header + words)
        engine = TesseractEngine(backend)
        region = engine.read(_frame(), _bbox())
        assert region is not None
        assert "HP" in region.text
        assert "100" in region.text
        # Both words present, order preserved.
        assert region.text.index("HP") < region.text.index("100")
