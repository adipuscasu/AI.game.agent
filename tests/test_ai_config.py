"""AI config, provider validation, and factory tests.

Split from tests/test_ai.py (2026-11-08).

Verifies:
  - AIConfig.timeout default, custom, Config wiring, load_config default
  - OllamaProvider timeout validation (finite positive number)
  - create_client factory (success, unsupported provider, importability)

Pure (no network): providers are configured or rejected before any I/O.
"""

from __future__ import annotations

import pytest


class TestAIConfigTimeout:
    def test_default_timeout_is_60(self):
        from ai_game_agent.config import AIConfig

        assert AIConfig().timeout == 60.0

    def test_custom_timeout(self):
        from ai_game_agent.config import AIConfig

        assert AIConfig(timeout=12.5).timeout == 12.5

    def test_timeout_zero_rejected(self):
        from ai_game_agent.config import AIConfig, ConfigError

        with pytest.raises(ConfigError):
            AIConfig(timeout=0.0)

    def test_timeout_negative_rejected(self):
        from ai_game_agent.config import AIConfig, ConfigError

        with pytest.raises(ConfigError):
            AIConfig(timeout=-1.0)

    def test_config_wires_timeout_default(self):
        from ai_game_agent.config import Config

        assert Config().ai.timeout == 60.0

    def test_load_config_default_timeout(self):
        from ai_game_agent.config import load_config

        cfg = load_config("config/default.yaml")
        assert cfg.ai.timeout == 60.0


class TestOllamaProviderTimeoutValidation:
    """OllamaProvider is public — constructor must validate timeout itself.

    The config layer (``AIConfig.timeout``) rejects non-finite values via
    pydantic, but direct instantiation can bypass it. The constructor must
    accept only a finite, strictly positive timeout and raise ``AIError``
    (not let a bad value leak into the HTTP transport).
    """

    def _provider(self, timeout: float):
        from ai_game_agent.ai import OllamaProvider

        return OllamaProvider(
            model="m", base_url="http://localhost:11434", timeout=timeout
        )

    def test_nan_timeout_rejected(self):
        from ai_game_agent.ai import AIError

        with pytest.raises(AIError):
            self._provider(float("nan"))

    def test_positive_infinity_timeout_rejected(self):
        from ai_game_agent.ai import AIError

        with pytest.raises(AIError):
            self._provider(float("inf"))

    def test_negative_infinity_timeout_rejected(self):
        from ai_game_agent.ai import AIError

        with pytest.raises(AIError):
            self._provider(float("-inf"))

    def test_zero_timeout_rejected(self):
        from ai_game_agent.ai import AIError

        with pytest.raises(AIError):
            self._provider(0.0)

    def test_negative_finite_timeout_rejected(self):
        from ai_game_agent.ai import AIError

        with pytest.raises(AIError):
            self._provider(-1.0)

    def test_finite_positive_timeout_accepted(self):
        p = self._provider(12.5)
        assert p.timeout == 12.5


class TestCreateClient:
    def test_returns_ollama_provider(self):
        from ai_game_agent.ai import create_client
        from ai_game_agent.config import AIConfig

        client = create_client(AIConfig())
        assert client.name == "ollama"

    def test_unsupported_provider_raises_ai_error(self):
        from ai_game_agent.ai import AIError, create_client
        from ai_game_agent.config import AIConfig

        with pytest.raises(AIError) as exc_info:
            create_client(AIConfig(provider="bogus"))
        assert "unsupported" in str(exc_info.value).lower()
        assert "bogus" in str(exc_info.value)

    def test_ollama_case_insensitive(self):
        from ai_game_agent.ai import create_client
        from ai_game_agent.config import AIConfig

        # "Ollama" (title case) must still resolve.
        client = create_client(AIConfig(provider="Ollama"))
        assert client.name == "ollama"

    def test_ai_package_importable_in_bare_venv(self):
        """Importing ai_game_agent.ai must not drag in cv2/PyYAML/etc."""
        import importlib
        import sys

        # Remove any cached heavy deps to simulate a bare env import path.
        blocked = {"cv2", "yaml", "PIL", "pynput", "mss", "requests", "httpx"}
        saved = {}
        for mod in blocked:
            if mod in sys.modules:
                saved[mod] = sys.modules.pop(mod)
        try:
            # Force a fresh import of the ai package.
            for key in [k for k in sys.modules if k == "ai_game_agent.ai"
                        or k.startswith("ai_game_agent.ai.")]:
                del sys.modules[key]
            importlib.import_module("ai_game_agent.ai")
        finally:
            sys.modules.update(saved)
