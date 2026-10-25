"""AI provider integration for the agent (Phase 5).

This package lets the agent ask a local VLM (Ollama) for *suggestions* --
grounded observations and action decisions -- without giving the model any
direct power over the machine. The core contract:

- **Model proposes; application validates.** AI output is plain text/JSON.
  It is parsed into the existing canonical :class:`Action` (Phase-3 frozen
  dataclass) and re-validated against the Phase-3/4 contracts before it can
  ever be executed. A malformed or unsafe suggestion raises
  :class:`AIError` and is dropped.
- **Local-first / private.** Default provider is Ollama on
  ``http://localhost:11434``; no API keys, no cloud round-trips.
- **Stdlib transport only.** No ``requests``/``httpx`` -- HTTP goes through
  :mod:`urllib` so the package stays importable in a bare venv.

Public surface (exported by the package): :class:`AIError`,
:class:`AIDecision`, :class:`AIProvider`, :class:`OllamaProvider`,
:func:`create_client`.
"""

from __future__ import annotations

from .base import AIError, AIProvider, create_client
from .ollama import OllamaProvider
from .schema import AIDecision

__all__ = [
    "AIError",
    "AIProvider",
    "AIDecision",
    "OllamaProvider",
    "create_client",
]
