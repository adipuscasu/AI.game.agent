"""AI provider abstractions (Phase 5).

Defines the provider boundary: the :class:`AIError` error type, the
:class:`AIProvider` protocol that every concrete provider implements, and the
:func:`create_client` factory. The factory is the single place where the
provider name is resolved to a concrete class, keeping the rest of the code
base provider-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import AIConfig


class AIError(Exception):
    """Raised when the AI provider fails, returns malformed output, or the
    requested provider is not supported.

    Subclasses:
        - :class:`AIProviderError` (planned, Phase 6): transport / HTTP
          failures (connection refused, timeout, non-2xx status).
    """


@runtime_checkable
class AIProvider(Protocol):
    """Common surface every concrete AI provider must implement.

    The provider is a *suggestion engine*: it takes structured input
    (a screenshot, a prompt, a game-state summary) and returns a raw text
    response. It never executes actions; it only proposes.
    """

    #: Human-readable provider name (e.g. ``"ollama"``).
    name: str

    def complete(
        self,
        prompt: str,
        *,
        image: bytes | None = None,
        image_mime: str = "image/png",
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Send a (possibly image-grounded) prompt to the model.

        Args:
            prompt: The user / system prompt text.
            image: Optional raw image bytes (PNG or JPEG).
            image_mime: MIME type of ``image`` (default ``image/png``).
            system: Optional separate system message.
            temperature: Sampling temperature.
            max_tokens: Optional token cap.

        Returns:
            The model's text response.

        Raises:
            AIError: On any transport or response-shape failure.
        """
        ...

    def decide(
        self,
        prompt: str,
        *,
        image: bytes | None = None,
        image_mime: str = "image/png",
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        format_schema: dict | None = None,
    ) -> dict:
        """Ask the model for a structured decision and parse it.

        Returns a :class:`dict` that must conform to :class:`AIDecision`
        (validated by the caller via :meth:`AIDecision.from_dict`).

        Args:
            format_schema: Optional JSON schema for structured output
                (provider-specific; Ollama uses this for ``format``).

        Returns:
            Parsed JSON object (dict).

        Raises:
            AIError: On transport failure or unparseable response.
        """
        ...


def create_client(config: AIConfig, **kwargs: object) -> AIProvider:
    """Instantiate an AI provider from :class:`AIConfig`.

    Supported providers (Phase 5): ``ollama``.
    Planned (Phase 6+): ``openai``, ``vllm``, ``llama.cpp``.

    Args:
        config: Validated AI configuration (provider, model, base_url, timeout).
        **kwargs: Provider-specific overrides.

    Returns:
        A concrete :class:`AIProvider` instance.

    Raises:
        AIError: If the provider name is not supported.
    """
    provider = (config.provider or "").strip().lower()

    if provider == "ollama":
        from .ollama import OllamaProvider  # deferred: keeps base importable

        return OllamaProvider(
            model=config.model,
            base_url=config.base_url,
            timeout=config.timeout,
            **kwargs,
        )

    raise AIError(
        f"unsupported AI provider {config.provider!r}; supported: 'ollama'"
    )
