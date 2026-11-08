"""Ollama request-shape tests: what OllamaProvider sends.

Split from tests/test_ai.py (2026-11-08).

Verifies /api/generate and /api/chat request bodies, image
grounding (data-URL encoding), JSON-mode formatting, and that
decide() returns a validated AIDecision. Transport runs against the
scripted stdlib HTTP harness (tests/_ai_ollama_harness.py) — CI-safe,
no live Ollama required.
"""

from __future__ import annotations

import base64
import json
import unittest

import pytest
from _ai_ollama_harness import _png_bytes, _ScriptedOllamaServer


class TestDecideValidation:
    """decide() must return a *validated* AIDecision, not a raw dict.

    The package contract ("model proposes; application validates") means a
    malformed or unknown action (e.g. an invented ``"noop"`` kind) is
    rejected with AIError instead of being handed back for execution.
    """

    def test_decide_returns_validated_decision(self):
        from ai_game_agent.ai import AIDecision, OllamaProvider

        resp = {"response": json.dumps(
            {"action": {"kind": "mouse_click", "schema_version": 1}}
        )}
        server = _ScriptedOllamaServer(response=resp)
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            out = p.decide("choose")
            assert isinstance(out, AIDecision)
            assert out.action.kind.value == "mouse_click"
            assert out.action.source == "ai"
        finally:
            server.stop()

    def test_decide_rejects_unknown_action_kind(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        # "noop" is not an ActionKind: the model inventing kinds must be
        # rejected, not handed back for execution.
        resp = {"response": json.dumps(
            {"action": {"kind": "noop", "schema_version": 1}}
        )}
        server = _ScriptedOllamaServer(response=resp)
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError):
                p.decide("choose")
        finally:
            server.stop()

    def test_decide_rejects_missing_action(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        resp = {"response": json.dumps({"reasoning": "no action available"})}
        server = _ScriptedOllamaServer(response=resp)
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError):
                p.decide("choose")
        finally:
            server.stop()


class TestOllamaRequestShape:
    def test_generate_request_shape(self):
        from ai_game_agent.ai import OllamaProvider

        server = _ScriptedOllamaServer(response={"response": "hello"})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            out = p.complete("say hi", temperature=0.2)
            assert out == "hello"

            req = server.requests[-1]
            assert req["path"] == "/api/generate"
            assert req["payload"]["model"] == "m"
            assert req["payload"]["stream"] is False
            assert req["payload"]["temperature"] == 0.2
            assert "images" not in req["payload"]
            assert req["headers"].get("Content-Type") == "application/json"
        finally:
            server.stop()

    def test_generate_with_image(self):
        from ai_game_agent.ai import OllamaProvider

        server = _ScriptedOllamaServer(response={"response": "ok"})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            img = _png_bytes()
            out = p.complete("describe", image=img)
            assert out == "ok"

            req = server.requests[-1]
            assert len(req["payload"]["images"]) == 1
            decoded = base64.b64decode(req["payload"]["images"][0])
            assert decoded == img
        finally:
            server.stop()

    def test_generate_with_system_prepends(self):
        from ai_game_agent.ai import OllamaProvider

        server = _ScriptedOllamaServer(response={"response": "ok"})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            p.complete("do thing", system="You are an assistant.")
            req = server.requests[-1]
            assert req["payload"]["prompt"].startswith("You are an assistant.")
            assert "do thing" in req["payload"]["prompt"]
        finally:
            server.stop()

    def test_decide_forces_json_mode(self):
        from ai_game_agent.ai import OllamaProvider

        resp = {"response": json.dumps({"action": {"kind": "mouse_click",
                                                     "schema_version": 1}})}
        server = _ScriptedOllamaServer(response=resp)
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            out = p.decide("choose an action")
            assert out.action.kind.value == "mouse_click"

            req = server.requests[-1]
            assert req["payload"]["format"] == "json"
            assert req["payload"]["stream"] is False
        finally:
            server.stop()

    def test_decide_with_format_schema(self):
        from ai_game_agent.ai import OllamaProvider

        resp = {"response": json.dumps({"action": {"kind": "mouse_click",
                                                     "schema_version": 1}})}
        server = _ScriptedOllamaServer(response=resp)
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            schema = {"type": "object", "properties": {}}
            p.decide("choose", format_schema=schema)
            req = server.requests[-1]
            assert req["payload"]["format"] == schema
        finally:
            server.stop()

    def test_decide_max_tokens_sets_num_predict(self):
        from ai_game_agent.ai import OllamaProvider

        resp = {"response": json.dumps({"action": {"kind": "mouse_click",
                                                     "schema_version": 1}})}
        server = _ScriptedOllamaServer(response=resp)
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            p.decide("choose", max_tokens=256)
            req = server.requests[-1]
            assert req["payload"]["options"]["num_predict"] == 256
        finally:
            server.stop()

    def test_unsupported_image_mime_raises(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        server = _ScriptedOllamaServer()
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError):
                p.complete("x", image=b"png", image_mime="image/tiff")
        finally:
            server.stop()


class TestDecideImageGrounding:
    """F1 (readiness): the ``decide()`` seam — image grounding must work
    alongside the structured-output contract: the grounded image is base64-
    encoded into ``payload["images"]``, JSON mode is still forced, and a
    validated :class:`AIDecision` comes back."""

    @staticmethod
    def _decision_response():
        return {"response": json.dumps(
            {"action": {"kind": "mouse_click", "schema_version": 1},
             "reasoning": "r", "confidence": 0.9})}

    def test_decide_with_image_sends_images_and_json_mode(self):
        from ai_game_agent.ai import OllamaProvider

        server = _ScriptedOllamaServer(response=self._decision_response())
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            img = _png_bytes()
            out = p.decide("choose", image=img)
            assert out.action.kind.value == "mouse_click"

            req = server.requests[-1]
            assert req["path"] == "/api/generate"
            assert req["payload"]["format"] == "json"
            assert len(req["payload"]["images"]) == 1
            decoded = base64.b64decode(req["payload"]["images"][0])
            assert decoded == img
        finally:
            server.stop()

    def test_decide_with_image_and_format_schema(self):
        from ai_game_agent.ai import OllamaProvider

        server = _ScriptedOllamaServer(response=self._decision_response())
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            schema = {"type": "object", "properties": {}}
            p.decide("choose", image=_png_bytes(), format_schema=schema)
            req = server.requests[-1]
            assert req["payload"]["format"] == schema
            assert len(req["payload"]["images"]) == 1
        finally:
            server.stop()

    def test_decide_without_image_omits_images_key(self):
        from ai_game_agent.ai import OllamaProvider

        server = _ScriptedOllamaServer(response=self._decision_response())
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            p.decide("choose")
            req = server.requests[-1]
            assert "images" not in req["payload"]
        finally:
            server.stop()

    def test_decide_unsupported_image_mime_raises_without_http(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        server = _ScriptedOllamaServer()
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError):
                p.decide("x", image=b"png", image_mime="image/tiff")
            assert server.requests == [], "mime rejection must happen pre-flight"
        finally:
            server.stop()

    def test_decide_image_non_bytes_raises(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        server = _ScriptedOllamaServer()
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError):
                p.decide("x", image="not-bytes")  # type: ignore[arg-type]
            assert server.requests == []
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
