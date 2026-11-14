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
    UiZoneHit,
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
        ui_zones=(UiZoneHit(zone="hp_bar", check="brightness", bbox=_bbox(), value=0.85),),
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

    def test_schema_version_is_two(self) -> None:
        obs = Observation(
            frame_width=2,
            frame_height=2,
            captured_at=_ts(),
            source="mock",
            templates=(),
            ui_zones=(),
            text_regions=(),
            objects=(),
            detector_errors=(),
        )
        assert obs.schema_version == 2


class TestEmptyObservation:
    def test_all_empty_is_valid(self) -> None:
        obs = Observation(
            frame_width=128,
            frame_height=72,
            captured_at=_ts(),
            source="mock",
            templates=(),
            ui_zones=(),
            text_regions=(),
            objects=(),
            detector_errors=(),
        )
        d = obs.to_dict()
        assert d["templates"] == []
        assert d["ui_zones"] == []
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
            "ui_zones",
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
        assert payload["schema_version"] == 2
        assert payload["detector_errors"] == ["ocr: tesseract not installed"]

    def test_empty_to_json_round_trips(self) -> None:
        empty = Observation(
            frame_width=1,
            frame_height=1,
            captured_at=_ts(),
            source="mock",
            templates=(),
            ui_zones=(),
            text_regions=(),
            objects=(),
            detector_errors=(),
        )
        payload = json.loads(empty.to_json())
        assert payload["templates"] == []
        assert payload["ui_zones"] == []


class TestFromDict:
    """``from_dict`` is the inverse of ``to_dict`` — the label-deserialization
    path used by the EH4 dataset manifest. It must be strict: the manifest
    is data authored outside this codebase, so unknown keys and missing keys
    are shape drift and must be rejected, not silently absorbed."""

    def test_round_trips_full(self) -> None:
        assert Observation.from_dict(_full().to_dict()) == _full()

    def test_round_trips_empty(self) -> None:
        empty = Observation(
            frame_width=1,
            frame_height=1,
            captured_at=_ts(),
            source="synthetic",
            templates=(),
            ui_zones=(),
            text_regions=(),
            objects=(),
            detector_errors=(),
        )
        assert Observation.from_dict(empty.to_dict()) == empty

    def test_detector_errors_and_schema_version_have_defaults(self) -> None:
        d = _full().to_dict()
        del d["detector_errors"]
        del d["schema_version"]
        obs = Observation.from_dict(d)
        assert obs.detector_errors == ()
        assert obs.schema_version == _full().schema_version

    def test_unknown_top_level_key_rejected(self) -> None:
        d = _full().to_dict()
        d["bogus"] = 1
        with pytest.raises(ValueError, match="unknown"):
            Observation.from_dict(d)

    def test_unknown_nested_key_rejected(self) -> None:
        d = _full().to_dict()
        d["templates"][0]["bogus"] = 1
        with pytest.raises(ValueError, match="unknown"):
            Observation.from_dict(d)

    def test_missing_required_key_rejected(self) -> None:
        d = _full().to_dict()
        del d["source"]
        with pytest.raises(ValueError, match="missing"):
            Observation.from_dict(d)

    def test_out_of_range_confidence_rejected(self) -> None:
        d = _full().to_dict()
        d["templates"][0]["confidence"] = 1.5
        with pytest.raises(ValueError):
            Observation.from_dict(d)

    def test_negative_bbox_extents_rejected(self) -> None:
        d = _full().to_dict()
        d["objects"][0]["bbox"]["width"] = -5
        with pytest.raises(ValueError):
            Observation.from_dict(d)

    def test_invalid_timestamp_rejected(self) -> None:
        d = _full().to_dict()
        d["captured_at"] = "not-a-timestamp"
        with pytest.raises(ValueError, match="captured_at"):
            Observation.from_dict(d)

    # -- schema_version guard (F1) -----------------------------------------

    def test_explicit_current_schema_version_accepted(self) -> None:
        d = _full().to_dict()
        d["schema_version"] = 2
        assert Observation.from_dict(d) == _full()

    @pytest.mark.parametrize("bad", [1, 999, None, True, "2", 2.0])
    def test_invalid_schema_version_rejected(self, bad: object) -> None:
        d = _full().to_dict()
        d["schema_version"] = bad
        with pytest.raises(ValueError, match="schema_version"):
            Observation.from_dict(d)

    # -- detector_errors validation (F2) -----------------------------------

    def test_detector_errors_list_of_strings_accepted(self) -> None:
        d = _full().to_dict()
        d["detector_errors"] = ["ocr: tesseract not installed", "ui:zone_a: timeout"]
        obs = Observation.from_dict(d)
        assert obs.detector_errors == ("ocr: tesseract not installed", "ui:zone_a: timeout")

    @pytest.mark.parametrize("bad", [None, "disk error", {"error": "OCR failed"}, 5])
    def test_detector_errors_wrong_container_rejected(self, bad: object) -> None:
        d = _full().to_dict()
        d["detector_errors"] = bad
        with pytest.raises(ValueError, match="detector_errors"):
            Observation.from_dict(d)

    def test_huge_int_confidence_raises_value_error(self) -> None:
        # ``float(10**400)`` raises OverflowError — like every other
        # malformed numeric it must normalize to ValueError, never escape.
        d = _full().to_dict()
        d["templates"][0]["confidence"] = 10**400
        with pytest.raises(ValueError, match="confidence"):
            Observation.from_dict(d)

    @pytest.mark.parametrize("bad_entry", [5, None, ["nested"], {"error": "OCR failed"}])
    def test_detector_errors_non_string_entry_rejected(self, bad_entry: object) -> None:
        d = _full().to_dict()
        d["detector_errors"] = ["ok", bad_entry]
        with pytest.raises(ValueError, match="detector_errors"):
            Observation.from_dict(d)

    # --- strict numeric coercion (EH4 audit B1/B2) -------------------------
    # ``from_dict`` is the deserializer for *authored* JSON: it must raise
    # ValueError for every non-numeric value (never leak TypeError) and must
    # not accept JSON types the serializer never emits — bool is a distinct
    # JSON type, not a number, and neither is a string.

    def test_none_frame_dimensions_rejected(self) -> None:
        d = _full().to_dict()
        d["frame_width"] = None
        with pytest.raises(ValueError, match="frame_width"):
            Observation.from_dict(d)
        d = _full().to_dict()
        d["frame_height"] = None
        with pytest.raises(ValueError, match="frame_height"):
            Observation.from_dict(d)

    def test_none_confidence_and_value_rejected(self) -> None:
        d = _full().to_dict()
        d["templates"][0]["confidence"] = None
        with pytest.raises(ValueError, match="confidence"):
            Observation.from_dict(d)
        d = _full().to_dict()
        d["ui_zones"][0]["value"] = None
        with pytest.raises(ValueError, match="value"):
            Observation.from_dict(d)

    def test_none_bbox_extents_rejected(self) -> None:
        d = _full().to_dict()
        d["objects"][0]["bbox"]["x"] = None
        with pytest.raises(ValueError, match="bbox"):
            Observation.from_dict(d)

    def test_bool_frame_dimensions_rejected(self) -> None:
        d = _full().to_dict()
        d["frame_width"] = True
        with pytest.raises(ValueError, match="frame_width"):
            Observation.from_dict(d)

    def test_bool_bbox_extents_rejected(self) -> None:
        d = _full().to_dict()
        d["objects"][0]["bbox"]["width"] = True
        with pytest.raises(ValueError, match="bbox"):
            Observation.from_dict(d)

    def test_bool_confidence_rejected(self) -> None:
        d = _full().to_dict()
        d["templates"][0]["confidence"] = True
        with pytest.raises(ValueError, match="confidence"):
            Observation.from_dict(d)

    def test_string_numeric_rejected(self) -> None:
        # JSON strings are a different type; ``int("128")`` must not coerce.
        d = _full().to_dict()
        d["frame_width"] = "128"
        with pytest.raises(ValueError, match="frame_width"):
            Observation.from_dict(d)
        d = _full().to_dict()
        d["templates"][0]["confidence"] = "0.94"
        with pytest.raises(ValueError, match="confidence"):
            Observation.from_dict(d)
