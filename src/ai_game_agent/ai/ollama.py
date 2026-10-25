"""Ollama provider (Phase 5).

Thin stdlib-urllib client for the Ollama HTTP API. No external dependencies.

Endpoints used:
    - ``POST /api/generate`` -- single-turn completion (default).
    - ``POST /api/chat``     -- chat-style (supports system + messages).

Both support:
    - ``images`` -- list of base64-encoded image bytes for VLM grounding.
    - ``format`` -- ``"json"`` or a JSON schema object for structured output.
    - ``stream: false`` -- required for a single-response contract.

The provider is deliberately simple: it turns a prompt (+ optional image)
into a text response, and for :meth:`decide` it enforces JSON output and
parses the result.
"""

from __future__ import annotations

import base64
import json
import math
import urllib.error
import urllib.request

from .base import AIError
from .schema import AIDecision

#: Default per-request timeout (seconds); overridable per config.
_DEFAULT_TIMEOUT = 60.0

#: MIME types the client will base64-encode and send to Ollama.
_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp"})


class OllamaProvider:
    """Ollama-backed :class:`AIProvider` implementation.

    Args:
        model: Ollama model name (e.g. ``"qwen3.8:27b"``).
        base_url: Ollama server root URL (e.g. ``"http://localhost:11434"``).
        timeout: Per-request socket timeout in seconds.
    """

    def __init__(
        self,
        model: str = "qwen3.8:27b",
        base_url: str = "http://localhost:11434",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not model:
            raise AIError("Ollama model name must be a non-empty string")
        if not base_url:
            raise AIError("Ollama base_url must be a non-empty string")
        if not math.isfinite(timeout) or timeout <= 0:
            raise AIError(
                f"Ollama timeout must be a finite positive number, got {timeout!r}"
            )

        self.model = model
        # Normalise: strip trailing slash, no trailing path.
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.name = "ollama"

    # ------------------------------------------------------------------ #
    # AIProvider protocol                                                 #
    # ------------------------------------------------------------------ #

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
        """Send a prompt (optionally with a grounded image) to Ollama.

        Raises:
            AIError: On transport failure, non-2xx response, or malformed
                response shape.
        """
        if image is not None and image_mime not in _IMAGE_MIMES:
            raise AIError(
                f"unsupported image_mime {image_mime!r}; expected one of "
                f"{sorted(_IMAGE_MIMES)}"
            )

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        if image is not None:
            payload["images"] = [self._encode_image(image, image_mime)]

        # If a system message is present, prepend it to the prompt (Ollama
        # /api/generate has no separate system field).
        if system:
            payload["prompt"] = f"{system}\n\n{prompt}"

        response = self._post("/api/generate", payload)
        text = response.get("response")
        if not isinstance(text, str):
            raise AIError(
                "Ollama response missing 'response' string field"
            )
        return text

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
    ) -> AIDecision:
        """Ask Ollama for a JSON decision object and validate it.

        Uses ``format: "json"`` (or a JSON schema) to force structured output,
        then parses and validates the payload via :meth:`AIDecision.from_dict`
        so a malformed or unknown action never reaches the caller.

        Raises:
            AIError: On transport failure, unparseable JSON, or a decision
                that fails :class:`AIDecision` validation.
        """
        if image is not None and image_mime not in _IMAGE_MIMES:
            raise AIError(
                f"unsupported image_mime {image_mime!r}; expected one of "
                f"{sorted(_IMAGE_MIMES)}"
            )

        # Force JSON mode.
        if format_schema is not None:
            fmt: object = format_schema
        else:
            fmt = "json"

        # Append an explicit instruction to emit a single JSON object.
        json_hint = (
            "\n\nRespond with a single JSON object only. "
            "No prose, no markdown fences, no commentary."
        )
        full_prompt = f"{prompt}{json_hint}"
        if system:
            full_prompt = f"{system}\n\n{full_prompt}"

        payload: dict = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "temperature": temperature,
            "format": fmt,
        }
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        if image is not None:
            payload["images"] = [self._encode_image(image, image_mime)]

        response = self._post("/api/generate", payload)
        text = response.get("response")
        if not isinstance(text, str):
            raise AIError("Ollama decision response missing 'response' string")

        data = self._parse_json(text)
        try:
            return AIDecision.from_dict(data)
        except AIError:
            raise
        except Exception as exc:  # defensive: from_dict should only raise AIError
            raise AIError(f"invalid AI decision: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Transport                                                           #
    # ------------------------------------------------------------------ #

    def _post(self, path: str, payload: dict) -> dict:
        """POST a JSON payload to ``base_url + path`` and parse the JSON response.

        Raises:
            AIError: On connection errors, timeouts, non-2xx status, or
                malformed response body.
        """
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            err_body = exc.read()[:512]
            raise AIError(
                f"Ollama HTTP {exc.code} from {path}: {exc.reason}; "
                f"body={err_body!r}"
            ) from exc
        except urllib.error.URLError as exc:
            raise AIError(
                f"Ollama connection failed ({path}): {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise AIError(
                f"Ollama request timed out after {self.timeout}s ({path})"
            ) from exc
        except OSError as exc:
            raise AIError(
                f"Ollama I/O error ({path}): {exc}"
            ) from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AIError(
                f"Ollama returned non-JSON response from {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise AIError(
                f"Ollama response from {path} must be a JSON object, "
                f"got {type(data).__name__}"
            )
        return data

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_image(image: bytes, mime: str) -> str:
        """Base64-encode raw image bytes for Ollama's ``images`` field."""
        if not isinstance(image, (bytes, bytearray)):
            raise AIError(
                f"image must be bytes, got {type(image).__name__}"
            )
        return base64.b64encode(bytes(image)).decode("ascii")

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse a JSON object from model output, tolerating common noise.

        Handles:
            - Leading/trailing whitespace.
            - Markdown code fences (`` ```json ... ``` ``).
            - A single top-level JSON object embedded in prose.

        Raises:
            AIError: If no valid JSON object can be extracted.
        """
        candidate = text.strip()

        # Strip markdown fences.
        if candidate.startswith("```"):
            # Find the first newline after the opening fence.
            first_nl = candidate.find("\n")
            if first_nl == -1:
                raise AIError("Markdown fence present but no content found")
            candidate = candidate[first_nl + 1:]
            if candidate.rstrip().endswith("```"):
                candidate = candidate.rstrip()[:-3].rstrip()

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            # Fallback: scan for the first ``{`` and last ``}``.
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise AIError(
                    f"Could not parse AI decision as JSON: {text[:200]!r}"
                ) from None
            try:
                data = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError as exc:
                raise AIError(
                    f"Could not parse AI decision as JSON: {exc}"
                ) from exc

        if not isinstance(data, dict):
            raise AIError(
                f"AI decision JSON must be an object, got {type(data).__name__}"
            )
        return data
