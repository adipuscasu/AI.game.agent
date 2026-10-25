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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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

    # ``extra="forbid"`` (M3): a typo such as ``backends:`` must surface as a
    # ConfigError instead of silently falling back to the default backend.
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str = "mss"
    region: _Region | None = None
    # mss monitor selector (mss monitors[] index): 0 = whole virtual screen,
    # 1 = primary (default), n >= 2 = (n-1)th secondary. Ignored by backends
    # that do not expose monitors (mock).
    monitor: int = Field(default=1, ge=0)
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

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = "ollama"
    model: str = "qwen3.8:27b"
    base_url: str = "http://localhost:11434"
    timeout: float = Field(default=60.0, gt=0, le=600)


class _InputConfig(BaseModel):
    """Input control settings (Phase 3, step 2).

    ``backend`` selects the input backend (``mock`` | ``windows``); ``mode``
    is the default operating mode.

    Layer split (matches the tests):
    * the **config** layer fails fast on unknown *keys* (``extra="forbid"``,
      e.g. a ``backends:`` typo) and validates the ``mode`` against the closed
      ``OperationMode`` set — a bad mode is a schema error at load;
    * the **factory** (:func:`~ai_game_agent.actions.base.create_backend`) owns
      the ``backend`` *value*: it normalizes case/whitespace (``"  MOCK "`` ->
      mock) and raises :class:`ActionError` for an unknown name. So this model
      does NOT closed-set-validate ``backend`` — the constructor stays
      permissive so the factory can apply its selection contract.

    The canonical ``mode`` spellings are the ``OperationMode``
    (``actions/modes.py``, step 3) string values.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str = "mock"
    # The default operating mode. OBSERVE_ONLY is the safe default: the
    # shipped CLI emits no OS input unless explicitly promoted.
    mode: str = "observe_only"
    # The emergency-stop hotkey is defined in the safety block
    # (safety.emergency_stop_keys) and read through by the Windows adapter.

    @field_validator("mode")
    @classmethod
    def _mode_is_known(cls, v: str) -> str:
        allowed = ("observe_only", "assisted", "semi_autonomous", "autonomous")
        if v not in allowed:
            raise ValueError(f"invalid input.mode {v!r}; expected one of {list(allowed)}")
        return v


class _SafetyConfig(BaseModel):
    """Safety / emergency-stop settings (Phase 3+)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    emergency_stop_keys: tuple[str, ...] = ("F12",)
    max_action_duration_ms: int = Field(default=5000, gt=0)
    max_consecutive_actions: int = Field(default=50, gt=0)
    # Bounded retry: after this many backend errors the executor PAUSES
    # (never an infinite loop). Phase 3, step 4.
    max_backend_errors: int = Field(default=3, gt=0)


_VALID_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"})


class _LoggingConfig(BaseModel):
    """Logging settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: str = "INFO"
    directory: str = "logs"

    @field_validator("level")
    @classmethod
    def _check_level(cls, value: str) -> str:
        # L2: reject unknown levels at load time so a typo fails fast instead
        # of silently producing the wrong verbosity at runtime.
        if value.upper() not in _VALID_LOG_LEVELS:
            allowed = ", ".join(sorted(_VALID_LOG_LEVELS))
            raise ValueError(f"invalid logging level {value!r}; expected one of: {allowed}")
        return value.upper()


_VALID_UI_CHECKS = frozenset({"presence", "brightness", "color_present"})


class _UiZoneConfig(BaseModel):
    """A configured UI region to inspect (Phase 2).

    ``check`` is a small closed set of deterministic heuristics (presence,
    brightness, color_present). Unknown values must fail at load, not silently
    disable the zone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    check: str
    #: Optional per-zone overrides (all optional, defaulted to the detector
    #: canonical constants when unset).
    threshold: float | None = Field(default=None, ge=0.0, le=255.0)
    target_rgb: tuple[int, int, int] | None = None
    tolerance: int | None = Field(default=None, ge=0)

    @field_validator("check")
    @classmethod
    def _check_is_known(cls, value: str) -> str:
        if value not in _VALID_UI_CHECKS:
            allowed = ", ".join(sorted(_VALID_UI_CHECKS))
            raise ValueError(f"invalid ui zone check {value!r}; expected one of: {allowed}")
        return value

    @field_validator("target_rgb")
    @classmethod
    def _check_target_rgb(cls, value: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        if value is None:
            return None
        if any(not 0 <= c <= 255 for c in value):
            raise ValueError(f"target_rgb channels must be in [0, 255], got {value!r}")
        return value


class _OcrRegionConfig(BaseModel):
    """A configured text region to OCR (Phase 2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    name: str


class _OcrConfig(BaseModel):
    """OCR settings (Phase 2). Default off: needs the ``ocr`` extra + binary.

    There is intentionally no ``engine`` selector: the pipeline only supports
    the Tesseract backend (``perception/ocr.py``), so a config knob that could
    not change behavior would be dead surface.

    ``min_confidence`` is a noise floor: Tesseract emits low-confidence
    garbage on real frames (e.g. a single "i" at 0.34 from UI anti-aliasing);
    results strictly below the floor are dropped by the pipeline. The gate
    lives in the pipeline (not the engine) so any future engine benefits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    regions: tuple[_OcrRegionConfig, ...] = ()
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class _ObjectColorConfig(BaseModel):
    """A reference color for blob detection (Phase 2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    rgb: tuple[int, int, int]
    tolerance: int = Field(default=40, ge=0)

    @field_validator("rgb")
    @classmethod
    def _check_rgb_range(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(not 0 <= c <= 255 for c in value):
            raise ValueError(f"rgb channels must be in [0, 255], got {value!r}")
        return value


class _ObjectConfig(BaseModel):
    """Basic object-detection settings (Phase 2). Default off."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    colors: tuple[_ObjectColorConfig, ...] = ()


class _PerceptionConfig(BaseModel):
    """Perception settings (Phase 2).

    ``enabled`` defaults to true — the CLI ``--no-*`` flags are opt-outs — but
    with no templates, no zones, and ocr/objects off, the shipped default has
    the net effect of an empty observation (Phase 1 behavior preserved).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    template_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    templates: dict[str, str] = {}
    ui_zones: tuple[_UiZoneConfig, ...] = ()
    ocr: _OcrConfig = Field(default_factory=_OcrConfig)
    objects: _ObjectConfig = Field(default_factory=_ObjectConfig)


class _AgentConfig(BaseModel):
    """Policy parameters for the deterministic state machine (Phase 4, §4.5).

    Mirrors the exact constructor surface of
    :class:`~ai_game_agent.games.wow.policy.DeterministicPolicy` so that
    ``DeterministicPolicy.from_config`` is a pure pass-through. Defaults match
    the policy's built-in defaults, keeping an existing config file (no
    ``agent:`` block) backward compatible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="wow", min_length=1)
    combat_keys: tuple[str, ...] = Field(default_factory=lambda: ("q", "w"))
    combat_delay_ms: int = Field(default=120, gt=0)
    loot_pickup_key: str = Field(default="f", min_length=1)
    loot_delay_ms: int = Field(default=200, gt=0)
    recovery_key: str = Field(default="r", min_length=1)
    recovery_delay_ms: int = Field(default=300, gt=0)
    health_low: float = Field(default=0.30, gt=0, le=1)
    health_recovered: float = Field(default=0.50, gt=0, le=1)
    # Loop knobs (Phase 4 close-out): the state-machine watchdog timeout and
    # the bounded-recovery budget. Defaults match the
    # :class:`~ai_game_agent.state.machine.GameStateMachine` built-ins
    # (``state_timeout_s=5.0``, ``recovery_attempts=3``), so a config file
    # without these keys is backward compatible.
    state_timeout_s: float = Field(default=5.0, gt=0)
    recovery_attempts: int = Field(default=3, gt=0)
    # Phase 5 close-out: opt-in flag for the AI_ANALYSIS seam. Default False
    # (AI off) keeps every existing config + test unchanged; the policy only
    # emits SIGNAL_AI_ANALYSIS_REQUIRED when this is True.
    ai_enabled: bool = Field(default=False)

    @field_validator("state_timeout_s", "recovery_attempts", mode="before")
    @classmethod
    def _reject_bool(cls, value: object) -> object:
        # pydantic coerces ``True`` -> ``1`` / ``True`` -> ``1.0`` on bare
        # int/float fields, so a bool must be rejected *before* coercion
        # (the same discipline the delay fields use on the public facade).
        if isinstance(value, bool):
            raise ValueError(f"must be a number, not a bool (got {value!r})")
        return value

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> _AgentConfig:
        if not (self.health_low < self.health_recovered):
            raise ValueError(
                f"health_low must be < health_recovered, "
                f"got {self.health_low!r} !< {self.health_recovered!r}"
            )
        return self


class AgentConfig:
    """Immutable, frozen view over :class:`_AgentConfig`.

    Mirrors the public/private split used by every other config section
    (``CaptureConfig``/``_CaptureConfig``, etc.): callers read attributes,
    the model enforces validation, and ``from_model`` gives a single
    construction path so the frozen view can never drift from the validated
    model. Hashable and value-equal.
    """

    __slots__ = (
        "name",
        "combat_keys",
        "combat_delay_ms",
        "loot_pickup_key",
        "loot_delay_ms",
        "recovery_key",
        "recovery_delay_ms",
        "health_low",
        "health_recovered",
        "state_timeout_s",
        "recovery_attempts",
        "ai_enabled",
    )

    def __init__(
        self,
        name: str = "wow",
        combat_keys: tuple[str, ...] = ("q", "w"),
        combat_delay_ms: int = 120,
        loot_pickup_key: str = "f",
        loot_delay_ms: int = 200,
        recovery_key: str = "r",
        recovery_delay_ms: int = 300,
        health_low: float = 0.30,
        health_recovered: float = 0.50,
        state_timeout_s: float = 5.0,
        recovery_attempts: int = 3,
        ai_enabled: bool = False,
    ) -> None:
        # Validate against the same rules the model enforces (so a direct
        # AgentConfig(...) call is just as strict as a YAML round-trip).
        if not name or not isinstance(name, str):
            raise ValueError(f"policy name must be a non-empty string, got {name!r}")
        if not (isinstance(combat_keys, (list, tuple)) and combat_keys):
            raise ValueError("combat_keys must be a non-empty tuple/list")
        for key in combat_keys:
            if not (isinstance(key, str) and key):
                raise ValueError(f"combat key must be a non-empty string, got {key!r}")
        for key_name, key in (
            ("loot_pickup_key", loot_pickup_key),
            ("recovery_key", recovery_key),
        ):
            if not (isinstance(key, str) and key):
                raise ValueError(f"{key_name} must be a non-empty string, got {key!r}")
        for delay_name, delay in (
            ("combat_delay_ms", combat_delay_ms),
            ("loot_delay_ms", loot_delay_ms),
            ("recovery_delay_ms", recovery_delay_ms),
        ):
            if not (isinstance(delay, int) and not isinstance(delay, bool) and delay > 0):
                raise ValueError(f"{delay_name} must be a positive int, got {delay!r}")
        for t_name, t in (("health_low", health_low), ("health_recovered", health_recovered)):
            if not (0.0 < t <= 1.0):
                raise ValueError(f"{t_name} must be in (0, 1], got {t!r}")
        if not (health_low < health_recovered):
            raise ValueError(
                f"health_low must be < health_recovered, got {health_low!r} !< {health_recovered!r}"
            )
        if not (
            isinstance(state_timeout_s, (int, float))
            and not isinstance(state_timeout_s, bool)
            and state_timeout_s > 0
        ):
            raise ValueError(f"state_timeout_s must be a positive number, got {state_timeout_s!r}")
        if not (
            isinstance(recovery_attempts, int)
            and not isinstance(recovery_attempts, bool)
            and recovery_attempts > 0
        ):
            raise ValueError(f"recovery_attempts must be a positive int, got {recovery_attempts!r}")
        if not isinstance(ai_enabled, bool):
            raise ValueError(f"ai_enabled must be a bool, got {ai_enabled!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "combat_keys", tuple(combat_keys))
        object.__setattr__(self, "combat_delay_ms", combat_delay_ms)
        object.__setattr__(self, "loot_pickup_key", loot_pickup_key)
        object.__setattr__(self, "loot_delay_ms", loot_delay_ms)
        object.__setattr__(self, "recovery_key", recovery_key)
        object.__setattr__(self, "recovery_delay_ms", recovery_delay_ms)
        object.__setattr__(self, "health_low", health_low)
        object.__setattr__(self, "health_recovered", health_recovered)
        object.__setattr__(self, "state_timeout_s", state_timeout_s)
        object.__setattr__(self, "recovery_attempts", recovery_attempts)
        object.__setattr__(self, "ai_enabled", ai_enabled)

    @classmethod
    def from_model(cls, model: _AgentConfig) -> AgentConfig:
        return cls(
            name=model.name,
            combat_keys=model.combat_keys,
            combat_delay_ms=model.combat_delay_ms,
            loot_pickup_key=model.loot_pickup_key,
            loot_delay_ms=model.loot_delay_ms,
            recovery_key=model.recovery_key,
            recovery_delay_ms=model.recovery_delay_ms,
            health_low=model.health_low,
            health_recovered=model.health_recovered,
            state_timeout_s=model.state_timeout_s,
            recovery_attempts=model.recovery_attempts,
            ai_enabled=model.ai_enabled,
        )

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover
        raise AttributeError("AgentConfig is immutable")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"AgentConfig(name={self.name!r}, combat_keys={self.combat_keys!r}, "
            f"combat_delay_ms={self.combat_delay_ms!r}, "
            f"loot_pickup_key={self.loot_pickup_key!r}, loot_delay_ms={self.loot_delay_ms!r}, "
            f"recovery_key={self.recovery_key!r}, recovery_delay_ms={self.recovery_delay_ms!r}, "
            f"health_low={self.health_low!r}, health_recovered={self.health_recovered!r}, "
            f"state_timeout_s={self.state_timeout_s!r}, "
            f"recovery_attempts={self.recovery_attempts!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentConfig):
            return NotImplemented
        return (
            self.name == other.name
            and self.combat_keys == other.combat_keys
            and self.combat_delay_ms == other.combat_delay_ms
            and self.loot_pickup_key == other.loot_pickup_key
            and self.loot_delay_ms == other.loot_delay_ms
            and self.recovery_key == other.recovery_key
            and self.recovery_delay_ms == other.recovery_delay_ms
            and self.health_low == other.health_low
            and self.health_recovered == other.health_recovered
            and self.state_timeout_s == other.state_timeout_s
            and self.recovery_attempts == other.recovery_attempts
            and self.ai_enabled == other.ai_enabled
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.combat_keys,
                self.combat_delay_ms,
                self.loot_pickup_key,
                self.loot_delay_ms,
                self.recovery_key,
                self.recovery_delay_ms,
                self.health_low,
                self.health_recovered,
                self.state_timeout_s,
                self.recovery_attempts,
                self.ai_enabled,
            )
        )


class _Config(BaseModel):
    """Top-level configuration object for the agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ai: _AIConfig = Field(default_factory=_AIConfig)
    capture: _CaptureConfig = Field(default_factory=_CaptureConfig)
    input: _InputConfig = Field(default_factory=_InputConfig)
    safety: _SafetyConfig = Field(default_factory=_SafetyConfig)
    logging: _LoggingConfig = Field(default_factory=_LoggingConfig)
    perception: _PerceptionConfig = Field(default_factory=_PerceptionConfig)
    agent: _AgentConfig = Field(default_factory=_AgentConfig)


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


class UiZone:
    """A configured UI region to inspect (Phase 2).

    Immutable public view over the validated ``_UiZoneConfig``. ``check`` is
    one of the closed set (presence, brightness, color_present).
    """

    __slots__ = (
        "name", "x", "y", "width", "height", "check", "threshold", "target_rgb", "tolerance"
    )

    def __init__(
        self,
        name: str,
        x: int,
        y: int,
        width: int,
        height: int,
        check: str,
        threshold: float | None = None,
        target_rgb: tuple[int, int, int] | None = None,
        tolerance: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("ui zone width and height must be positive")
        if check not in _VALID_UI_CHECKS:
            allowed = ", ".join(sorted(_VALID_UI_CHECKS))
            raise ValueError(f"invalid ui zone check {check!r}; expected one of: {allowed}")
        if threshold is not None and not 0 <= threshold <= 255:
            # BT.601 luminance is [0, 255]; a value outside that range could
            # never fire (or always fire) the brightness check — fail fast
            # instead of silently degrading (mirrors the pydantic model).
            raise ValueError(
                f"ui zone threshold must be in [0, 255], got {threshold!r}"
            )
        # object.__setattr__ bypasses the immutability guard below (same
        # pattern as _FrozenConfig); post-construction writes raise AttributeError.
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "check", check)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "target_rgb", target_rgb)
        object.__setattr__(self, "tolerance", tolerance)

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover
        raise AttributeError("UiZone is immutable")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"UiZone(name={self.name!r}, x={self.x}, y={self.y}, "
            f"width={self.width}, height={self.height}, check={self.check!r}, "
            f"threshold={self.threshold!r}, target_rgb={self.target_rgb!r}, "
            f"tolerance={self.tolerance!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UiZone):
            return NotImplemented
        return (
            (self.name, self.x, self.y, self.width, self.height, self.check,
             self.threshold, self.target_rgb, self.tolerance)
            == (other.name, other.x, other.y, other.width, other.height, other.check,
                other.threshold, other.target_rgb, other.tolerance)
        )

    def __hash__(self) -> int:
        return hash((self.name, self.x, self.y, self.width, self.height, self.check,
                        self.threshold, self.target_rgb, self.tolerance))


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
        timeout: float = 60.0,
    ) -> None:
        try:
            m = _AIConfig(provider=provider, model=model, base_url=base_url, timeout=timeout)
        except ValidationError as exc:
            raise ConfigError(f"invalid AI configuration: {exc}") from exc
        super().__init__(m)

    @property
    def provider(self) -> str:
        return self._m.provider

    @property
    def model(self) -> str:
        return self._m.model

    @property
    def base_url(self) -> str:
        return self._m.base_url

    @property
    def timeout(self) -> float:
        return self._m.timeout


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
        monitor: int = 1,
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
                monitor=monitor,
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

    @property
    def monitor(self) -> int:
        return self._m.monitor


class SafetyConfig(_FrozenConfig):
    """Safety / emergency-stop settings (Phase 3+)."""

    __slots__ = ()

    def __init__(
        self,
        emergency_stop_keys: tuple[str, ...] = ("F12",),
        max_action_duration_ms: int = 5000,
        max_consecutive_actions: int = 50,
        max_backend_errors: int = 3,
    ) -> None:
        super().__init__(
            _SafetyConfig(
                emergency_stop_keys=emergency_stop_keys,
                max_action_duration_ms=max_action_duration_ms,
                max_consecutive_actions=max_consecutive_actions,
                max_backend_errors=max_backend_errors,
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

    @property
    def max_backend_errors(self) -> int:
        """How many consecutive backend I/O failures before a safe pause."""
        return self._m.max_backend_errors


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


class PerceptionConfig(_FrozenConfig):
    """Perception settings (Phase 2)."""

    __slots__ = ()

    def __init__(
        self,
        enabled: bool = True,
        template_threshold: float = 0.8,
        templates: dict[str, str] | None = None,
        ui_zones: tuple[UiZone, ...] = (),
        ocr_enabled: bool = False,
        ocr_regions: tuple[dict[str, object], ...] | list[dict[str, object]] = (),
        ocr_min_confidence: float = 0.5,
        objects_enabled: bool = False,
        object_colors: tuple[dict[str, object], ...] | list[dict[str, object]] = (),
    ) -> None:
        ui_zone_models = tuple(
            _UiZoneConfig(
                name=z.name, x=z.x, y=z.y, width=z.width, height=z.height, check=z.check,
                threshold=z.threshold, target_rgb=z.target_rgb, tolerance=z.tolerance,
            )
            for z in ui_zones
        )
        ocr_regions_models = tuple(
            _OcrRegionConfig(
                x=int(r["x"]), y=int(r["y"]), width=int(r["width"]),
                height=int(r["height"]), name=str(r.get("name", "")),
            )
            for r in (ocr_regions or ())
        )
        object_color_models = tuple(
            _ObjectColorConfig(
                name=str(c["name"]),
                rgb=(int(c["rgb"][0]), int(c["rgb"][1]), int(c["rgb"][2])),
                tolerance=int(c.get("tolerance", 40)),
            )
            for c in (object_colors or ())
        )
        super().__init__(
            _PerceptionConfig(
                enabled=enabled,
                template_threshold=template_threshold,
                templates=dict(templates or {}),
                ui_zones=ui_zone_models,
                ocr=_OcrConfig(
                    enabled=ocr_enabled,
                    regions=ocr_regions_models,
                    min_confidence=ocr_min_confidence,
                ),
                objects=_ObjectConfig(
                    enabled=objects_enabled,
                    colors=object_color_models,
                ),
            )
        )

    @property
    def enabled(self) -> bool:
        return self._m.enabled

    @property
    def template_threshold(self) -> float:
        return self._m.template_threshold

    @property
    def templates(self) -> dict[str, str]:
        return dict(self._m.templates)

    @property
    def ui_zones(self) -> tuple[UiZone, ...]:
        return tuple(
            UiZone(
                z.name, z.x, z.y, z.width, z.height, z.check,
                threshold=z.threshold, target_rgb=z.target_rgb, tolerance=z.tolerance,
            )
            for z in self._m.ui_zones
        )

    @property
    def ocr_enabled(self) -> bool:
        return self._m.ocr.enabled

    @property
    def ocr_min_confidence(self) -> float:
        """Noise floor: OCR results strictly below this are dropped."""
        return self._m.ocr.min_confidence

    @property
    def ocr_regions(self) -> list[dict[str, object]]:
        """Configured OCR regions as JSON-shaped dicts (lists, not tuples)."""
        return [r.model_dump(mode="json") for r in self._m.ocr.regions]

    @property
    def objects_enabled(self) -> bool:
        return self._m.objects.enabled

    @property
    def object_colors(self) -> list[dict[str, object]]:
        """Configured object-detection colors as JSON-shaped dicts."""
        return [c.model_dump(mode="json") for c in self._m.objects.colors]


class InputConfig(_FrozenConfig):
    """Input / action-execution settings (Phase 3+).

    ``backend`` selects the :class:`~ai_game_agent.actions.base.InputBackend`
    (``mock`` for headless/CI, ``windows`` for the real pynput backend); ``mode``
    is the initial :class:`~ai_game_agent.actions.modes.OperationMode`.
    """

    __slots__ = ()

    def __init__(self, backend: str = "mock", mode: str = "observe_only") -> None:
        super().__init__(_InputConfig(backend=backend, mode=mode))

    @property
    def backend(self) -> str:
        return self._m.backend

    @property
    def mode(self) -> str:
        return self._m.mode


class Config:
    """Top-level configuration object for the agent."""

    __slots__ = ("ai", "capture", "safety", "logging", "perception", "input", "agent")

    def __init__(
        self,
        ai: AIConfig | None = None,
        capture: CaptureConfig | None = None,
        safety: SafetyConfig | None = None,
        logging: LoggingConfig | None = None,
        perception: PerceptionConfig | None = None,
        input: InputConfig | None = None,
        agent: AgentConfig | None = None,
    ) -> None:
        self.ai = ai if ai is not None else AIConfig()
        self.capture = capture if capture is not None else CaptureConfig()
        self.safety = safety if safety is not None else SafetyConfig()
        self.logging = logging if logging is not None else LoggingConfig()
        self.perception = perception if perception is not None else PerceptionConfig()
        self.input = input if input is not None else InputConfig()
        self.agent = agent if agent is not None else AgentConfig()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Config(ai={self.ai!r}, capture={self.capture!r}, "
            f"safety={self.safety!r}, logging={self.logging!r}, "
            f"perception={self.perception!r}, input={self.input!r}, "
            f"agent={self.agent!r})"
        )


def _region_from_model(region: _Region | None) -> Region | None:
    """Translate a private pydantic region into the public immutable one."""
    if region is None:
        return None
    return Region(region.x, region.y, region.width, region.height)


def _config_from_model(model: _Config) -> Config:
    return Config(
        ai=AIConfig(
            model.ai.provider, model.ai.model, model.ai.base_url, model.ai.timeout
        ),
        capture=CaptureConfig(
            backend=model.capture.backend,
            region=_region_from_model(model.capture.region),
            fps=model.capture.fps,
            scale=model.capture.scale,
            record_enabled=model.capture.record_enabled,
            record_directory=model.capture.record_directory,
            record_max_files=model.capture.record_max_files,
            monitor=model.capture.monitor,
        ),
        safety=SafetyConfig(
            tuple(model.safety.emergency_stop_keys),
            model.safety.max_action_duration_ms,
            model.safety.max_consecutive_actions,
            model.safety.max_backend_errors,
        ),
        logging=LoggingConfig(model.logging.level, model.logging.directory),
        input=InputConfig(model.input.backend, model.input.mode),
        perception=PerceptionConfig(
            enabled=model.perception.enabled,
            template_threshold=model.perception.template_threshold,
            templates=dict(model.perception.templates),
            ui_zones=tuple(
                UiZone(z.name, z.x, z.y, z.width, z.height, z.check,
                       threshold=z.threshold, target_rgb=z.target_rgb,
                       tolerance=z.tolerance)
                for z in model.perception.ui_zones
            ),
            ocr_enabled=model.perception.ocr.enabled,
            ocr_regions=tuple(r.model_dump() for r in model.perception.ocr.regions),
            ocr_min_confidence=model.perception.ocr.min_confidence,
            objects_enabled=model.perception.objects.enabled,
            object_colors=tuple(c.model_dump() for c in model.perception.objects.colors),
        ),
        agent=AgentConfig.from_model(model.agent),
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
