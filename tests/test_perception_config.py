"""Tests for the Phase 2 ``PerceptionConfig`` in ``config.py``."""

from __future__ import annotations

import pytest

from ai_game_agent.config import (
    ConfigError,
    PerceptionConfig,
    UiZone,
    load_config,
)


def test_default_perception_runs_nothing_by_default():
    # ``enabled`` defaults to true (the CLI ``--no-X`` flags are opt-outs),
    # but with no templates, no zones, and ocr/objects off, the net effect of
    # the shipped default is an empty observation — Phase 1 behavior preserved.
    cfg = load_config()
    p = cfg.perception
    assert p.enabled is True
    assert p.template_threshold == 0.8
    assert p.templates == {}
    assert p.ui_zones == ()
    assert p.ocr_enabled is False
    assert p.ocr_regions == []
    assert p.objects_enabled is False
    assert p.object_colors == []


def test_master_enabled_flag_can_turn_off_perception(tmp_path):
    path = tmp_path / "off.yaml"
    path.write_text("perception:\n  enabled: false\n", encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg.perception.enabled is False


def test_perception_block_is_immutable():
    cfg = load_config()
    p = cfg.perception
    with pytest.raises(AttributeError):
        p.enabled = True  # type: ignore[misc]


def test_load_perception_full_block(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        """
perception:
  enabled: true
  template_threshold: 0.75
  templates:
    target_frame: assets/templates/target_frame.png
    loot_glow: assets/templates/loot_glow.png
  ui_zones:
    - name: action_bar
      x: 640
      y: 940
      width: 640
      height: 120
      check: brightness
    - name: target_frame_zone
      x: 400
      y: 100
      width: 200
      height: 200
      check: presence
  ocr:
    enabled: true
    regions:
      - {x: 10, y: 20, width: 100, height: 20, name: health_bar}
  objects:
    enabled: true
    colors:
      - name: loot_glow
        rgb: [255, 200, 0]
        tolerance: 40
""",
        encoding="utf-8",
    )
    cfg = load_config(str(path))
    p = cfg.perception
    assert p.enabled is True
    assert p.template_threshold == 0.75
    assert p.templates == {
        "target_frame": "assets/templates/target_frame.png",
        "loot_glow": "assets/templates/loot_glow.png",
    }
    zones = list(p.ui_zones)
    assert zones == [
        UiZone(name="action_bar", x=640, y=940, width=640, height=120, check="brightness"),
        UiZone(name="target_frame_zone", x=400, y=100, width=200, height=200, check="presence"),
    ]
    assert p.ocr_enabled is True
    assert p.ocr_regions == [
        {"x": 10, "y": 20, "width": 100, "height": 20, "name": "health_bar"}
    ]
    assert p.objects_enabled is True
    # The public view is the JSON/YAML shape (``rgb`` a list).
    assert p.object_colors == [
        {"name": "loot_glow", "rgb": [255, 200, 0], "tolerance": 40}
    ]


def test_load_perception_rejects_unknown_key(tmp_path):
    bad = tmp_path / "unknown.yaml"
    bad.write_text("perception:\n  enable: true\n", encoding="utf-8")
    try:
        load_config(str(bad))
    except ConfigError as exc:
        assert "enable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown perception key")


def test_load_perception_rejects_unknown_ui_zone_key(tmp_path):
    bad = tmp_path / "bad-zone.yaml"
    bad.write_text(
        "perception:\n  ui_zones:\n    - {name: z, x: 0, y: 0, width: 1,\n"
        "      height: 1, checks: brightness}\n",
        encoding="utf-8",
    )
    try:
        load_config(str(bad))
    except ConfigError as exc:
        assert "checks" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown ui_zones key")


def test_load_perception_rejects_unknown_check_value(tmp_path):
    bad = tmp_path / "bad-check.yaml"
    bad.write_text(
        "perception:\n  ui_zones:\n"
        "    - {name: z, x: 0, y: 0, width: 1, height: 1, check: loud}\n",
        encoding="utf-8",
    )
    try:
        load_config(str(bad))
    except ConfigError as exc:
        # Should name the offending value.
        assert "loud" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown check value")


def test_load_perception_rejects_invalid_threshold(tmp_path):
    bad = tmp_path / "bad-thr.yaml"
    bad.write_text("perception:\n  template_threshold: 1.5\n", encoding="utf-8")
    try:
        load_config(str(bad))
    except ConfigError as exc:
        assert "template_threshold" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for out-of-range template_threshold")


def test_ui_zone_is_immutable():
    z = UiZone(name="z", x=0, y=0, width=1, height=1, check="brightness")
    with pytest.raises(AttributeError):
        z.check = "presence"  # type: ignore[misc]


def test_ui_zone_requires_positive_dimensions():
    with pytest.raises(ValueError):
        UiZone(name="z", x=0, y=0, width=0, height=1, check="brightness")
    with pytest.raises(ValueError):
        UiZone(name="z", x=0, y=0, width=1, height=0, check="brightness")


def test_perception_config_public_view_is_frozen():
    p = PerceptionConfig(enabled=True)
    with pytest.raises(AttributeError):
        p.template_threshold = 0.5  # type: ignore[misc]
