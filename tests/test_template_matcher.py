"""Tests for the OpenCV ``CvTemplateMatcher`` (Phase 2, step 4).

Uses small synthetic PNG templates drawn with Pillow (dev extra) — no real
game assets (plan §7.1). Behaviors under test:

* planted template found at the exact position with high confidence;
* threshold rejects a frame without the template;
* ``confidence`` stays in ``[0.0, 1.0]``;
* missing or corrupt template file → ``PerceptionError`` at construction
  (fail fast, plan §5.1 / §10);
* unknown template name at match time → ``PerceptionError``;
* template larger than the frame → ``PerceptionError``;
* protocol conformance (``TemplateMatcher``) and determinism.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ai_game_agent.capture import Frame
from ai_game_agent.perception.base import PerceptionError, TemplateMatcher
from ai_game_agent.perception.observation import BBox
from ai_game_agent.perception.template import CvTemplateMatcher

TS = datetime.datetime(2026, 9, 27, 12, 0, 0, tzinfo=datetime.UTC)

FRAME_SIZE = (256, 144)  # width, height (plan §7.1)
TEMPLATE_SIZE = (32, 24)
PLANT_AT = (10, 12)  # top-left where the template is pasted into the frame


# ---------------------------------------------------------------------------
# Synthetic assets
# ---------------------------------------------------------------------------


def _template_pil() -> Image.Image:
    """A high-contrast, deterministic template pattern."""
    img = Image.new("RGB", TEMPLATE_SIZE, (40, 40, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [4, 4, TEMPLATE_SIZE[0] - 5, TEMPLATE_SIZE[1] - 5], outline=(255, 255, 0), width=3
    )
    draw.rectangle([10, 8, 22, 16], fill=(255, 80, 80))
    return img


def _save_template(tmp_path: Path) -> Path:
    path = tmp_path / "target_frame.png"
    _template_pil().save(path, "PNG")
    return path


def _flat_frame(width: int = FRAME_SIZE[0], height: int = FRAME_SIZE[1]) -> Frame:
    return Frame(
        width=width,
        height=height,
        pixels=bytes([128] * (width * height * 3)),
        captured_at=TS,
        source="mock",
    )


def _planted_frame() -> Frame:
    img = Image.new("RGB", FRAME_SIZE, (128, 128, 128))
    img.paste(_template_pil(), PLANT_AT)
    return Frame(
        width=img.width,
        height=img.height,
        pixels=img.tobytes(),
        captured_at=TS,
        source="mock",
    )


# ---------------------------------------------------------------------------
# Matching behavior
# ---------------------------------------------------------------------------


def test_finds_planted_template_at_planted_position(tmp_path: Path) -> None:
    matcher = CvTemplateMatcher({"target_frame": str(_save_template(tmp_path))}, threshold=0.8)
    hit = matcher.match(_planted_frame(), "target_frame")

    assert hit is not None
    assert hit.template == "target_frame"
    assert hit.bbox == BBox(
        x=PLANT_AT[0], y=PLANT_AT[1], width=TEMPLATE_SIZE[0], height=TEMPLATE_SIZE[1]
    )
    assert 0.9 <= hit.confidence <= 1.0


def test_confidence_is_within_unit_interval(tmp_path: Path) -> None:
    matcher = CvTemplateMatcher({"t": str(_save_template(tmp_path))}, threshold=0.0)
    hit = matcher.match(_planted_frame(), "t")
    assert hit is not None
    assert 0.0 <= hit.confidence <= 1.0


def test_rejects_frame_without_the_template(tmp_path: Path) -> None:
    matcher = CvTemplateMatcher({"target_frame": str(_save_template(tmp_path))}, threshold=0.8)
    assert matcher.match(_flat_frame(), "target_frame") is None


def test_threshold_filters_low_scores(tmp_path: Path) -> None:
    path = str(_save_template(tmp_path))
    # A flat frame scores ~0.0: default threshold rejects it...
    assert CvTemplateMatcher({"t": path}, threshold=0.8).match(_flat_frame(), "t") is None
    # ...but a zero threshold admits it, with a low recorded confidence.
    hit = CvTemplateMatcher({"t": path}, threshold=0.0).match(_flat_frame(), "t")
    assert hit is not None
    assert hit.confidence < 0.8


def test_deterministic_repeated_matches(tmp_path: Path) -> None:
    matcher = CvTemplateMatcher({"t": str(_save_template(tmp_path))}, threshold=0.8)
    assert matcher.match(_planted_frame(), "t") == matcher.match(_planted_frame(), "t")


def test_template_larger_than_max_dimensions_raises(tmp_path: Path) -> None:
    big = tmp_path / "big.png"
    Image.new("RGB", (400, 400), (10, 200, 10)).save(big, "PNG")
    with pytest.raises(PerceptionError, match="larger than the maximum"):
        CvTemplateMatcher({"big": str(big)}, threshold=0.8, max_dimensions=256)


def test_template_larger_than_frame_raises(tmp_path: Path) -> None:
    big = tmp_path / "big.png"
    Image.new("RGB", (400, 400), (10, 200, 10)).save(big, "PNG")
    matcher = CvTemplateMatcher({"big": str(big)}, threshold=0.8)
    with pytest.raises(PerceptionError, match="larger than the"):
        matcher.match(_flat_frame(), "big")


def test_protocol_conformance() -> None:
    matcher = CvTemplateMatcher({}, threshold=0.8)
    assert isinstance(matcher, TemplateMatcher)


# ---------------------------------------------------------------------------
# Fail-fast construction and setup errors (plan §5.1, §10)
# ---------------------------------------------------------------------------


def test_missing_template_file_raises_at_construction(tmp_path: Path) -> None:
    with pytest.raises(PerceptionError, match="not found"):
        CvTemplateMatcher({"ghost": str(tmp_path / "missing.png")}, threshold=0.8)


def test_corrupt_template_file_raises_at_construction(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.png"
    bad.write_text("this is definitely not a PNG", encoding="utf-8")
    with pytest.raises(PerceptionError, match="decode"):
        CvTemplateMatcher({"corrupt": str(bad)}, threshold=0.8)


def test_invalid_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="threshold"):
        CvTemplateMatcher({}, threshold=1.5)


def test_match_unknown_template_name_raises() -> None:
    matcher = CvTemplateMatcher({}, threshold=0.8)
    with pytest.raises(PerceptionError, match="unknown template"):
        matcher.match(_flat_frame(), "nope")
