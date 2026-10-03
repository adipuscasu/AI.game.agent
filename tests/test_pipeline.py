"""Tests for the Phase 2 ``Perception`` pipeline (step 3).

The pipeline composes injected detector implementations (protocols from
``perception/base.py``) into the stable ``Observation`` contract.  No OpenCV
or other extras are required: all detectors here are fakes that structurally
conform to the protocols.

Behaviors under test (plan §7.1):

* all-on composition yields a full observation;
* one detector raising → recorded in ``detector_errors``, others still present;
* disabled detectors are skipped;
* determinism: two calls, same frame ⇒ equal observations.
"""

from __future__ import annotations

import datetime
import json
from typing import Protocol

import pytest

from ai_game_agent.capture import Frame
from ai_game_agent.config import PerceptionConfig, UiZone
from ai_game_agent.perception import (
    BBox,
    ObjectDetector,
    Observation,
    OcrEngine,
    Perception,
    PerceptionError,
    TemplateHit,
    TemplateMatcher,
    TextRegion,
    UiRegionDetector,
)
from ai_game_agent.perception.observation import ObjectHit

TS = datetime.datetime(2026, 9, 27, 12, 0, 0, tzinfo=datetime.UTC)


def _frame(width: int = 128, height: int = 72) -> Frame:
    return Frame(
        width=width,
        height=height,
        pixels=bytes(width * height * 3),
        captured_at=TS,
        source="mock",
    )


def _bbox(x: int = 10, y: int = 20, width: int = 32, height: int = 16) -> BBox:
    return BBox(x=x, y=y, width=width, height=height)


# ---------------------------------------------------------------------------
# Fakes — structurally conform to the base.py protocols.
# ---------------------------------------------------------------------------


class FakeTemplateMatcher:
    """Returns canned hits per template name; optionally raises for some."""

    def __init__(
        self,
        hits: dict[str, TemplateHit] | None = None,
        fail_on: tuple[str, ...] = (),
    ) -> None:
        self._hits = dict(hits or {})
        self._fail_on = set(fail_on)
        self.calls: list[tuple[Frame, str]] = []

    def match(self, frame: Frame, template: str) -> TemplateHit | None:
        self.calls.append((frame, template))
        if template in self._fail_on:
            raise RuntimeError(f"boom for {template}")
        return self._hits.get(template)


class FakeUiDetector:
    """Returns canned hits per zone name; optionally raises for some zones."""

    def __init__(
        self,
        hits: dict[str, TemplateHit] | None = None,
        fail_zones: tuple[str, ...] = (),
    ) -> None:
        self._hits = dict(hits or {})
        self._fail_zones = set(fail_zones)
        self.calls: list[tuple[Frame, UiZone]] = []

    def detect(self, frame: Frame, zone: UiZone) -> TemplateHit | None:
        self.calls.append((frame, zone))
        if zone.name in self._fail_zones:
            raise RuntimeError(f"boom for {zone.name}")
        return self._hits.get(zone.name)


class FakeOcrEngine:
    """Returns canned text per region BBox; optionally raises for some."""

    def __init__(
        self,
        hits: dict[BBox, TextRegion] | None = None,
        fail_regions: tuple[BBox, ...] = (),
    ) -> None:
        self._hits = dict(hits or {})
        self._fail_regions = set(fail_regions)
        self.calls: list[tuple[Frame, BBox]] = []

    def read(self, frame: Frame, region: BBox) -> TextRegion | None:
        self.calls.append((frame, region))
        if region in self._fail_regions:
            raise RuntimeError("ocr exploded")
        return self._hits.get(region)


class FakeObjectDetector:
    """Returns a fixed list of object hits; optionally raises."""

    def __init__(self, objects: tuple[ObjectHit, ...] = (), raise_error: bool = False) -> None:
        self._objects = list(objects)
        self._raise_error = raise_error
        self.calls: list[Frame] = []

    def match(self, frame: Frame) -> list[ObjectHit]:
        self.calls.append(frame)
        if self._raise_error:
            raise RuntimeError("detector on fire")
        return list(self._objects)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _config(**overrides: object) -> PerceptionConfig:
    base: dict[str, object] = {
        "enabled": True,
        "templates": {"target_frame": "assets/templates/target_frame.png"},
        "ui_zones": (UiZone("action_bar", 0, 60, 32, 12, "brightness"),),
        "ocr_enabled": True,
        "ocr_regions": (
            {"x": 10, "y": 20, "width": 32, "height": 16, "name": "hp_text"},
        ),
        "objects_enabled": True,
        "object_colors": ({"name": "loot_glow", "rgb": (255, 215, 0), "tolerance": 40},),
    }
    base.update(overrides)
    return PerceptionConfig(**base)


def _full_fakes() -> dict[str, object]:
    return {
        "template_matcher": FakeTemplateMatcher(
            hits={"target_frame": TemplateHit("target_frame", _bbox(0, 0, 24, 24), 0.94)}
        ),
        "ui_detector": FakeUiDetector(
            hits={"action_bar": TemplateHit("ui:action_bar", _bbox(0, 60, 32, 12), 0.9)}
        ),
        "ocr_engine": FakeOcrEngine(
            hits={_bbox(): TextRegion("12 / 12", _bbox(), 0.88)}
        ),
        "object_detector": FakeObjectDetector(
            objects=(ObjectHit("loot_glow", _bbox(50, 40, 8, 8), 0.7),)
        ),
    }


# ---------------------------------------------------------------------------
# Construction / interfaces
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_constructs_with_config_only(self) -> None:
        p = Perception(
            _config(templates=None, ui_zones=(), ocr_enabled=False, objects_enabled=False)
        )
        assert p is not None

    def test_perception_error_is_exception(self) -> None:
        assert issubclass(PerceptionError, Exception)

    @pytest.mark.parametrize(
        ("protocol", "fake"),
        [
            (TemplateMatcher, FakeTemplateMatcher()),
            (UiRegionDetector, FakeUiDetector()),
            (OcrEngine, FakeOcrEngine()),
            (ObjectDetector, FakeObjectDetector()),
        ],
    )
    def test_fakes_satisfy_protocols(self, protocol: type[Protocol], fake: object) -> None:
        assert isinstance(fake, protocol)


# ---------------------------------------------------------------------------
# observe() composition
# ---------------------------------------------------------------------------


class TestObserveComposition:
    def test_all_on_yields_full_observation(self) -> None:
        p = Perception(_config(), **_full_fakes())
        frame = _frame()

        obs = p.observe(frame)

        assert isinstance(obs, Observation)
        assert obs.frame_width == 128
        assert obs.frame_height == 72
        assert obs.captured_at == TS
        assert obs.source == "mock"
        assert obs.templates == (
            TemplateHit("target_frame", _bbox(0, 0, 24, 24), 0.94),
            TemplateHit("ui:action_bar", _bbox(0, 60, 32, 12), 0.9),
        )
        assert obs.text_regions == (TextRegion("12 / 12", _bbox(), 0.88),)
        assert obs.objects == (ObjectHit("loot_glow", _bbox(50, 40, 8, 8), 0.7),)
        assert obs.detector_errors == ()
        # JSON-serializable end-to-end.
        assert json.loads(obs.to_json())["schema_version"] == obs.schema_version

    def test_none_hits_are_filtered_out(self) -> None:
        fakes = {
            "template_matcher": FakeTemplateMatcher(),
            "ui_detector": FakeUiDetector(),
            "ocr_engine": FakeOcrEngine(),
            "object_detector": FakeObjectDetector(),
        }
        p = Perception(_config(), **fakes)  # all fakes return None / empty

        obs = p.observe(_frame())

        assert obs.templates == ()
        assert obs.text_regions == ()
        assert obs.objects == ()
        assert obs.detector_errors == ()

    def test_ui_zone_hit_is_pass_through(self) -> None:
        zone_hit = TemplateHit("ui:action_bar", _bbox(0, 60, 32, 12), 0.9)
        p = Perception(
            _config(templates=None, ocr_enabled=False, objects_enabled=False),
            ui_detector=FakeUiDetector(hits={"action_bar": zone_hit}),
        )

        obs = p.observe(_frame())

        assert obs.templates == (zone_hit,)

    def test_ocr_receives_configured_region_bbox(self) -> None:
        engine = FakeOcrEngine(hits={_bbox(): TextRegion("hp", _bbox(), 0.5)})
        p = Perception(
            _config(templates=None, ui_zones=(), objects_enabled=False),
            ocr_engine=engine,
        )

        p.observe(_frame())

        assert [(f.width, f.height, r) for f, r in engine.calls] == [
            (128, 72, _bbox(10, 20, 32, 16))
        ]

    def test_determinism_same_frame_equal_observations(self) -> None:
        p = Perception(_config(), **_full_fakes())
        frame = _frame()

        assert p.observe(frame) == p.observe(frame)


class TestDisabledDetectors:
    def test_missing_detector_is_skipped(self) -> None:
        fakes = _full_fakes()
        del fakes["object_detector"]
        p = Perception(_config(), **fakes)

        obs = p.observe(_frame())

        assert obs.objects == ()
        assert obs.templates != ()
        assert obs.detector_errors == ()

    def test_ocr_disabled_in_config_engine_not_called(self) -> None:
        engine = FakeOcrEngine(hits={_bbox(): TextRegion("hp", _bbox(), 0.5)})
        p = Perception(
            _config(ocr_enabled=False),
            ocr_engine=engine,
        )

        obs = p.observe(_frame())

        assert obs.text_regions == ()
        assert engine.calls == []

    def test_objects_disabled_in_config_detector_not_called(self) -> None:
        detector = FakeObjectDetector(objects=(ObjectHit("x", _bbox(), 0.5),))
        p = Perception(_config(objects_enabled=False), object_detector=detector)

        obs = p.observe(_frame())

        assert obs.objects == ()
        assert detector.calls == []

    def test_config_disabled_returns_empty_observation(self) -> None:
        fakes = _full_fakes()
        p = Perception(_config(enabled=False), **fakes)

        obs = p.observe(_frame())

        assert obs.templates == ()
        assert obs.text_regions == ()
        assert obs.objects == ()
        assert obs.detector_errors == ()
        # Detectors must never be consulted when perception is disabled.
        for fake in fakes.values():
            assert fake.calls == []


class TestDetectorErrors:
    def test_one_detector_failing_others_still_present(self) -> None:
        fakes = _full_fakes()
        fakes["object_detector"] = FakeObjectDetector(raise_error=True)
        p = Perception(_config(), **fakes)

        obs = p.observe(_frame())

        assert obs.objects == ()
        assert obs.templates != ()
        assert obs.text_regions != ()
        assert len(obs.detector_errors) == 1
        assert obs.detector_errors[0].startswith("objects:")
        assert "detector on fire" in obs.detector_errors[0]

    def test_single_failing_template_others_still_match(self) -> None:
        matcher = FakeTemplateMatcher(
            hits={"ok": TemplateHit("ok", _bbox(), 0.9)},
            fail_on=("bad",),
        )
        p = Perception(
            _config(
                templates={"bad": "assets/templates/bad.png", "ok": "assets/templates/ok.png"},
                ui_zones=(),
                ocr_enabled=False,
                objects_enabled=False,
            ),
            template_matcher=matcher,
        )

        obs = p.observe(_frame())

        assert obs.templates == (TemplateHit("ok", _bbox(), 0.9),)
        assert len(obs.detector_errors) == 1
        assert obs.detector_errors[0].startswith("templates:bad")

    def test_single_failing_ocr_region_others_still_read(self) -> None:
        bad = _bbox(x=0, y=0, width=10, height=10)
        good = _bbox()
        engine = FakeOcrEngine(
            hits={good: TextRegion("hp", good, 0.9)},
            fail_regions=(bad,),
        )
        p = Perception(
            _config(
                templates=None,
                ui_zones=(),
                ocr_regions=(
                    {"x": 0, "y": 0, "width": 10, "height": 10, "name": "bad_zone"},
                    {"x": 10, "y": 20, "width": 32, "height": 16, "name": "hp_text"},
                ),
                objects_enabled=False,
            ),
            ocr_engine=engine,
        )

        obs = p.observe(_frame())

        assert obs.text_regions == (TextRegion("hp", good, 0.9),)
        assert len(obs.detector_errors) == 1
        assert obs.detector_errors[0].startswith("ocr:")

    def test_single_failing_ui_zone_others_still_checked(self) -> None:
        zone_a = UiZone("zone_a", 0, 0, 10, 10, "brightness")
        zone_b = UiZone("zone_b", 0, 0, 10, 10, "brightness")
        hit_b = TemplateHit("ui:zone_b", _bbox(), 0.8)
        detector = FakeUiDetector(hits={"zone_b": hit_b}, fail_zones=("zone_a",))
        p = Perception(
            _config(
                templates=None,
                ui_zones=(zone_a, zone_b),
                ocr_enabled=False,
                objects_enabled=False,
            ),
            ui_detector=detector,
        )

        obs = p.observe(_frame())

        assert obs.templates == (hit_b,)
        assert len(obs.detector_errors) == 1
        assert obs.detector_errors[0].startswith("ui:zone_a")
