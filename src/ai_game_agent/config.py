"""Configuration loading and validation for ai-game-agent.

All runtime configuration lives outside source code (see ``config/``).
This module defines the typed schema (pydantic) and loads YAML files into it.

The public classes (``Config``, ``CaptureConfig``, ``Region``, ...) expose a
frozen, attribute-based API. Validation is delegated to pydantic models;
the public classes are immutable views over the validated models.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_FILE = _REPO_ROOT / "config" / "default.yaml"


class ConfigError(Exception):
    """Raised when configuration is missing, unreadable, or invalid."""


class _Region(BaseModel):
    """Axis-aligned capture region of interest in screen pixels."""

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class _CaptureConfig(BaseModel):
    """Screen capture settings (Phase 1)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    backend: str = "mss"
    region: _Region | None = None
    # fps=0 is a valid "no pacing" value used by the CLI for headless/CI runs.
    fps: int = Field(default=30, ge=0)
    scale: float = Field(default=1.0, gt=0)
    record_enabled: bool = False
    record_directory: str = "recordings"
    record_max_files: int = Field(default=1000, gt=0)

    @field_validator("region", mode="before")
    @classmethod
    def _coerce_region(cls, value: object) -> object:
        """Accept the public immutable :class:`Region` as well as dicts."""
        if value is None or isinstance(value, _Region):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "x") and hasattr(value, "width"):
            return {
                "x": value.x,
                "y": value.y,
                "width": value.width,
                "height": value.height,
            }
        return value

    @model_validator(mode="before")
    @classmethod
    def _flatten_record_block(cls, data: object) -> object:
        """Accept the nested ``record: {enabled, directory, max_files}`` form."""
        if isinstance(data, dict) and isinstance(data.get("record"), dict):
            record = dict(data["record"])
            data = dict(data)
            data.pop("record")
            data.setdefault("record_enabled", record.get("enabled", False))
            data.setdefault("record_directory", record.get("directory", "recordings"))
            data.setdefault("record_max_files", record.get("max_files", 1000))
        return data


class _AIConfig(BaseModel):
    """AI provider settings (Phase 5+). Local-first via Ollama."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    provider: str = "ollama"
    model: str = "qwen3.8:27b"
    base_url: str = "http://localhost:11434"


class _SafetyConfig(BaseModel):
    """Safety / emergency-stop settings (Phase 3+)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    emergency_stop_keys: tuple[str, ...] = ("F12",)
    max_action_duration_ms: int = Field(default=5000, gt=0)
    max_consecutive_actions: int = Field(default=50, gt=0)


class _LoggingConfig(BaseModel):
    """Logging settings."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    level: str = "INFO"
    directory: str = "logs"


class _Config(BaseModel):
    """Top-level configuration object for the agent."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    ai: _AIConfig = Field(default_factory=_AIConfig)
    capture: _CaptureConfig = Field(default_factory=_CaptureConfig)
    safety: _SafetyConfig = Field(default_factory=_SafetyConfig)
    logging: _LoggingConfig = Field(default_factory=_LoggingConfig)


class Region:
    """Axis-aligned capture region of interest in screen pixels.

    Validated on construction: ``width`` and ``height`` must be positive,
    ``x`` and ``y`` must be non-negative, and all values must be integers.
    """

    __slots__ = ("x", "y", "width", "height")

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        _Region(x=x, y=y, width=width, height=height)  # validates, then discards
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Region(x={self.x}, y={self.y}, width={self.width}, height={self.height})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Region):
            return NotImplemented
        return (self.x, self.y, self.width, self.height) == (
            other.x,
            other.y,
            other.width,
            other.height,
        )

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.width, self.height))


class _FrozenConfig:
    """Base for the immutable public configuration views."""

    __slots__ = ("_m",)
    _m: BaseModel

    def __init__(self, model: BaseModel) -> None:
        object.__setattr__(self, "_m", model)

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self._m!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._m == other._m

    def __hash__(self) -> int:
        return hash(self._m)


class AIConfig(_FrozenConfig):
    """AI provider settings (Phase 5+). Local-first via Ollama."""

    __slots__ = ()

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen3.8:27b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        super().__init__(_AIConfig(provider=provider, model=model, base_url=base_url))

    @property
    def provider(self) -> str:
        return self._m.provider

    @property
    def model(self) -> str:
        return self._m.model

    @property
    def base_url(self) -> str:
        return self._m.base_url


class CaptureConfig(_FrozenConfig):
    """Screen capture settings (Phase 1)."""

    __slots__ = ()

    def __init__(
        self,
        backend: str = "mss",
        region: Region | None = None,
        fps: int = 30,
        scale: float = 1.0,
        record_enabled: bool = False,
        record_directory: str = "recordings",
        record_max_files: int = 1000,
    ) -> None:
        super().__init__(
            _CaptureConfig(
                backend=backend,
                region=region,
                fps=fps,
                scale=scale,
                record_enabled=record_enabled,
                record_directory=record_directory,
                record_max_files=record_max_files,
            )
        )

    @property
    def backend(self) -> str:
        return self._m.backend

    @property
    def region(self) -> Region | None:
        r = self._m.region
        if r is None:
            return None
        return Region(r.x, r.y, r.width, r.height)

    @property
    def fps(self) -> int:
        return self._m.fps

    @property
    def scale(self) -> float:
        return self._m.scale

    @property
    def record_enabled(self) -> bool:
        return self._m.record_enabled

    @property
    def record_directory(self) -> str:
        return self._m.record_directory

    @property
    def record_max_files(self) -> int:
        return self._m.record_max_files


class SafetyConfig(_FrozenConfig):
    """Safety / emergency-stop settings (Phase 3+)."""

    __slots__ = ()

    def __init__(
        self,
        emergency_stop_keys: tuple[str, ...] = ("F12",),
        max_action_duration_ms: int = 5000,
        max_consecutive_actions: int = 50,
    ) -> None:
        super().__init__(
            _SafetyConfig(
                emergency_stop_keys=emergency_stop_keys,
                max_action_duration_ms=max_action_duration_ms,
                max_consecutive_actions=max_consecutive_actions,
            )
        )

    @property
    def emergency_stop_keys(self) -> tuple[str, ...]:
        return tuple(self._m.emergency_stop_keys)

    @property
    def max_action_duration_ms(self) -> int:
        return self._m.max_action_duration_ms

    @property
    def max_consecutive_actions(self) -> int:
        return self._m.max_consecutive_actions


class LoggingConfig(_FrozenConfig):
    """Logging settings."""

    __slots__ = ()

    def __init__(self, level: str = "INFO", directory: str = "logs") -> None:
        super().__init__(_LoggingConfig(level=level, directory=directory))

    @property
    def level(self) -> str:
        return self._m.level

    @property
    def directory(self) -> str:
        return self._m.directory


class Config:
    """Top-level configuration object for the agent."""

    __slots__ = ("ai", "capture", "safety", "logging")

    def __init__(
        self,
        ai: AIConfig | None = None,
        capture: CaptureConfig | None = None,
        safety: SafetyConfig | None = None,
        logging: LoggingConfig | None = None,
    ) -> None:
        self.ai = ai if ai is not None else AIConfig()
        self.capture = capture if capture is not None else CaptureConfig()
        self.safety = safety if safety is not None else SafetyConfig()
        self.logging = logging if logging is not None else LoggingConfig()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Config(ai={self.ai!r}, capture={self.capture!r}, "
            f"safety={self.safety!r}, logging={self.logging!r})"
        )


def _region_from_model(region: _Region | None) -> Region | None:
    """Translate a private pydantic region into the public immutable one."""
    if region is None:
        return None
    return Region(region.x, region.y, region.width, region.height)


def _config_from_model(model: _Config) -> Config:
    return Config(
        ai=AIConfig(model.ai.provider, model.ai.model, model.ai.base_url),
        capture=CaptureConfig(
            backend=model.capture.backend,
            region=_region_from_model(model.capture.region),
            fps=model.capture.fps,
            scale=model.capture.scale,
            record_enabled=model.capture.record_enabled,
            record_directory=model.capture.record_directory,
            record_max_files=model.capture.record_max_files,
        ),
        safety=SafetyConfig(
            tuple(model.safety.emergency_stop_keys),
            model.safety.max_action_duration_ms,
            model.safety.max_consecutive_actions,
        ),
        logging=LoggingConfig(model.logging.level, model.logging.directory),
    )


def load_config(path: str | None = None) -> Config:
    """Load configuration from a YAML file into a validated :class:`Config`.

    Args:
        path: Optional path to a YAML file. When omitted, the repository's
            ``config/default.yaml`` is used.

    Raises:
        ConfigError: If the file cannot be read or parsed, or contains
            values that do not match the configuration schema.
    """
    file_path = Path(path) if path is not None else _DEFAULT_CONFIG_FILE
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {file_path}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {file_path}: {exc}") from exc

    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {file_path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration root must be a mapping, got {type(raw).__name__}")

    try:
        model = _Config.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError and similar
        raise ConfigError(f"invalid configuration in {file_path}: {exc}") from exc

    return _config_from_model(model)
