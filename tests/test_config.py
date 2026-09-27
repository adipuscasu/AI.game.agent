from ai_game_agent.config import ConfigError, Region, load_config


def test_load_default_config_from_repo_default_yaml(tmp_path, monkeypatch):
    # load_config() with no path should resolve config/default.yaml relative
    # to the repository root (project root), not the CWD.
    monkeypatch.chdir(tmp_path)  # prove CWD-independence
    cfg = load_config()

    assert cfg.ai.provider == "ollama"
    assert cfg.ai.base_url == "http://localhost:11434"
    assert cfg.capture.backend == "mss"
    assert cfg.capture.fps == 30
    assert cfg.capture.region is None
    assert cfg.safety.emergency_stop_keys == ("F12",)
    assert cfg.logging.level == "INFO"


def test_load_config_from_explicit_path(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        """
ai:
  provider: ollama
  model: local-model
  base_url: http://localhost:9999
capture:
  backend: mock
  region: {x: 10, y: 20, width: 320, height: 240}
  fps: 15
safety:
  emergency_stop_keys: [F9, F10]
""",
        encoding="utf-8",
    )
    cfg = load_config(str(path))

    assert cfg.ai.model == "local-model"
    assert cfg.ai.base_url == "http://localhost:9999"
    assert cfg.capture.backend == "mock"
    assert cfg.capture.region == Region(x=10, y=20, width=320, height=240)
    assert cfg.capture.fps == 15
    assert cfg.safety.emergency_stop_keys == ("F9", "F10")
    # Untouched sections keep their defaults.
    assert cfg.capture.scale == 1.0
    assert cfg.logging.directory == "logs"


def test_load_config_missing_file_raises_config_error(tmp_path):
    missing = tmp_path / "nope.yaml"
    try:
        load_config(str(missing))
    except ConfigError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for missing file")


def test_load_config_invalid_yaml_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("capture: [unclosed", encoding="utf-8")
    try:
        load_config(str(bad))
    except ConfigError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid YAML")


def test_load_config_rejects_invalid_values(tmp_path):
    bad = tmp_path / "invalid.yaml"
    bad.write_text(
        """
capture:
  fps: not-a-number
  region: {x: -5, y: 0, width: 0, height: 100}
""",
        encoding="utf-8",
    )
    try:
        load_config(str(bad))
    except ConfigError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid values")


def test_capture_config_allows_fps_zero():
    # fps=0 is a documented "no pacing" value (CLI headless mode); it must be
    # representable in the config model rather than clamped to 1.
    from ai_game_agent.config import CaptureConfig

    cfg = CaptureConfig(backend="mock", fps=0)
    assert cfg.fps == 0




def test_capture_config_rejects_negative_fps():
    import pytest
    from pydantic import ValidationError

    from ai_game_agent.config import CaptureConfig

    with pytest.raises(ValidationError):
        CaptureConfig(backend="mock", fps=-1)


def test_region_validates_positive_dimensions():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Region(x=0, y=0, width=0, height=10)
