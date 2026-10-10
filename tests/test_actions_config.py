"""Step 2: ``InputConfig`` / ``safety.max_backend_errors`` — config-first wiring.

Mirrors Phase 2's config tests (``test_perception_config.py``): the typed
models are frozen, ``extra="forbid"`` (a typo in the ``input:`` block is a
load error, never a silent default), the safe defaults hold (``mock`` +
``observe_only`` — the shipped CLI emits no OS input unless explicitly
promoted), and the safety bound ``max_backend_errors`` round-trips through
YAML. All assertions are headless (no pynput, no desktop).
"""

from __future__ import annotations

import pytest

from ai_game_agent.config import ConfigError, InputConfig, SafetyConfig, load_config

# --- defaults (safe: no OS input unless promoted) ----------------------------


def test_input_config_defaults_to_safe_backend_and_mode():
    cfg = InputConfig()
    assert cfg.backend == "mock"  # deterministic, CI/backend-free
    assert cfg.mode == "observe_only"  # OBSERVE_ONLY emits zero OS input


def test_config_carries_input_block_with_safe_defaults():
    from ai_game_agent.config import Config

    assert Config().input.backend == "mock"
    assert Config().input.mode == "observe_only"


def test_default_yaml_input_block_is_safe():
    cfg = load_config()  # resolves config/default.yaml
    assert cfg.input.backend == "mock"
    assert cfg.input.mode == "observe_only"


# --- immutability -----------------------------------------------------------


def test_input_config_is_immutable():
    cfg = InputConfig()
    with pytest.raises(AttributeError):
        cfg.backend = "windows"


def test_input_config_is_frozen_and_hashable():
    # The public views expose the frozen pydantic model: mutation via the
    # model is rejected too, and equality/hash reflect the values.
    a = InputConfig(backend="mock", mode="assisted")
    b = InputConfig(backend="mock", mode="assisted")
    assert a == b
    assert hash(a) == hash(b)


def test_safety_max_backend_errors_defaults_bounded():
    assert SafetyConfig().max_backend_errors == 3


# --- extra="forbid": a typo is a load error, never a silent default ----------


def test_load_config_rejects_unknown_input_key(tmp_path):
    bad = tmp_path / "bad-input.yaml"
    bad.write_text(
        "input:\n"
        "  backend: mock\n"
        "  backends: windows   # typo: unknown key\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="backends"):
        load_config(str(bad))


def test_load_config_rejects_invalid_input_mode(tmp_path):
    bad = tmp_path / "bad-mode.yaml"
    bad.write_text(
        "input:\n"
        "  backend: mock\n"
        "  mode: obseve_only   # typo: not an OperationMode value\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="mode"):
        load_config(str(bad))


# --- accepted values round-trip ---------------------------------------------


def test_load_config_accepts_all_operation_modes(tmp_path):
    for mode in ("observe_only", "assisted", "semi_autonomous", "autonomous"):
        good = tmp_path / f"mode-{mode}.yaml"
        good.write_text(f"input:\n  backend: mock\n  mode: {mode}\n", encoding="utf-8")
        cfg = load_config(str(good))
        assert cfg.input.mode == mode


def test_load_config_accepts_windows_backend_selection(tmp_path):
    # Selecting the windows *backend* is legal config (it just needs the
    # "input" extra to actually run); the guard against a missing library is
    # the factory's job, not the config layer's.
    good = tmp_path / "win.yaml"
    good.write_text("input:\n  backend: windows\n", encoding="utf-8")
    assert load_config(str(good)).input.backend == "windows"


def test_safety_max_backend_errors_round_trips(tmp_path):
    good = tmp_path / "safety.yaml"
    good.write_text("safety:\n  max_backend_errors: 5\n", encoding="utf-8")
    assert load_config(str(good)).safety.max_backend_errors == 5


def test_safety_max_backend_errors_rejects_zero(tmp_path):
    # Bounded retry must be a positive bound; 0 would mean "pause on the
    # first backend error", which is not a sane default — reject at load.
    bad = tmp_path / "safety-zero.yaml"
    bad.write_text("safety:\n  max_backend_errors: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(bad))
