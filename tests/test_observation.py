"""Tests for the Phase 2 ``Observation`` contract.

The ``Observation`` dataclass is the single stable output of the perception
layer. It must be immutable, hashable, JSON-serializable, and carry the
``schema_version`` guard agreed in the plan's open-questions section.
"""

from __future__ import annotations

import datetime
import json

import pytest

from ai_game_agent.perception.observation import (
    BBox,
    ObjectHit,
    Observation,
    TemplateHit,
    TextRegion,
)


def _ts() -> datetime.datetime:
    return datetime.datetime(2026, 9, 27, 12, 0, 0, tzinfo=datetime.UTC)


def _bbox() -> BBox:
    return BBox(x=10, y=20, width=32, height=16)


def _full() -> Observation:
    return Observation(
        frame_width=128,
        frame_height=72,
        captured_at=_ts(),
        source="mock",
        templates=(TemplateHit(template="target_frame", bbox=_bbox(), confidence=0.94),),
        text_regions=(TextRegion(text="12 / 12", bbox=_bbox(), confidence=0.88),),
        objects=(ObjectHit(kind="loot_glow", bbox=_bbox(), confidence=0.7),),
        detector_errors=("ocr: tesseract not installed",),
    )


class TestBBox:
    def test_fields(self) -> None:
        assert _bbox() == BBox(x=10, y=20, width=32, height=16)

    def test_frozen(self) -> None:
        box = _bbox()
        with pytest.raises(AttributeError):
            box.x = 5  # type: ignore[misc]

    def test_negative_dimension_rejected(self) -> None:
        with pytest.raises(ValueError):
            BBox(x=0, y=0, width=-1, height=4)
        with pytest.raises(ValueError):
            BBox(x=0, y=0, width=4, height=-1)


class TestObservationShape:
    def test_collections_are_tuples(self) -> None:
        obs = _full()
        assert isinstance(obs.templates, tuple)
        assert isinstance(obs.text_regions, tuple)
        assert isinstance(obs.objects, tuple)
        assert isinstance(obs.detector_errors, tuple)

    def test_frozen(self) -> None:
        obs = _full()
        with pytest.raises(AttributeError):
            obs.source = "other"  # type: ignore[misc]

    def test_hashable(self) -> None:
        # Frozen dataclass with tuple fields must be hashable.
        assert hash(_full()) == hash(_full())

    def test_schema_version_defaults_to_one(self) -> None:
        obs = Observation(
            frame_width=2,
            frame_height=2,
            captured_at=_ts(),
            source="mock",
            templates=(),
            text_regions=(),
            objects=(),
            detector_errors=(),
        )
        assert obs.schema_version == 1


class TestEmptyObservation:
    def test_all_empty_is_valid(self) -> None:
        obs = Observation(
            frame_width=128,
            frame_height=72,
            captured_at=_ts(),
            source="mock",
            templates=(),
            text_regions=(),
            objects=(),
            detector_errors=(),
        )
        d = obs.to_dict()
        assert d["templates"] == []
        assert d["text_regions"] == []
        assert d["objects"] == []
        assert d["detector_errors"] == []


class TestConfidenceBounds:
    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            TemplateHit(template="t", bbox=_bbox(), confidence=1.5)
        with pytest.raises(ValueError):
            TextRegion(text="x", bbox=_bbox(), confidence=-0.1)
        with pytest.raises(ValueError):
            ObjectHit(kind="k", bbox=_bbox(), confidence=2.0)


class TestSerialization:
    def test_to_dict_is_json_serializable(self) -> None:
        d = _full().to_dict()
        # Must round-trip through json without lossy objects.
        assert json.loads(json.dumps(d)) == d

    def test_dict_keys(self) -> None:
        d = _full().to_dict()
        assert set(d) == {
            "schema_version",
            "frame_width",
            "frame_height",
            "captured_at",
            "source",
            "templates",
            "text_regions",
            "objects",
            "detector_errors",
        }

    def test_timestamp_is_iso_string(self) -> None:
        d = _full().to_dict()
        assert d["captured_at"] == _ts().isoformat()

    def test_nested_shapes(self) -> None:
        d = _full().to_dict()
        t = d["templates"][0]
        assert t["template"] == "target_frame"
        assert t["bbox"] == {"x": 10, "y": 20, "width": 32, "height": 16}
        assert t["confidence"] == 0.94

    def test_to_json_round_trips(self) -> None:
        payload = json.loads(_full().to_json())
        assert payload["source"] == "mock"
        assert payload["schema_version"] == 1
        assert payload["detector_errors"] == ["ocr: tesseract not installed"]

    def test_empty_to_json_round_trips(self) -> None:
        empty = Observation(
            frame_width=1,
            frame_height=1,
            captured_at=_ts(),
            source="mock",
            templates=(),
            text_regions=(),
            objects=(),
            detector_errors=(),
        )
        payload = json.loads(empty.to_json())
        assert payload["templates"] == []
