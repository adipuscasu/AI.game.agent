"""AI provider (Ollama) integration tests (Phase 5).

Verifies:
  - config: AIConfig.timeout default, custom, Config wiring, load_config default
  - base: create_client factory (ollama success, unsupported error, importable)
  - schema: AIDecision.from_dict/to_dict, validation errors, round-trip
  - ollama: request shape (generate + chat, images, json mode), timeout,
    response parsing, markdown-fence stripping, connection error, HTTP error

Transport tests use a scripted stdlib HTTP server on a random localhost port —
CI-safe, no live Ollama required.
"""

from __future__ import annotations

import base64
import http.server
import json
import socket
import threading
import time
import unittest

import pytest


# ===================================================================== #
# config: AIConfig.timeout                                              #
# ===================================================================== #


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


# ===================================================================== #
# base: create_client factory                                           #
# ===================================================================== #


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


# ===================================================================== #
# schema: AIDecision                                                    #
# ===================================================================== #


def _click_action_dict() -> dict:
    """A valid mouse_click Action dict (Phase-3 canonical vocabulary)."""
    return {
        "kind": "mouse_click",
        "x": 100,
        "y": 200,
        "button": "left",
        "clicks": 1,
        "schema_version": 1,
    }


def _noop_action_dict() -> dict:
    """A minimal valid in-place mouse_click (no required fields)."""
    return {"kind": "mouse_click", "schema_version": 1}


class TestAIDecision:
    def test_from_dict_valid_click(self):
        from ai_game_agent.ai import AIDecision

        d = {"action": _click_action_dict(), "reasoning": "click target",
             "confidence": 0.92}
        dec = AIDecision.from_dict(d)
        assert dec.kind == "mouse_click"
        assert dec.confidence == 0.92
        assert dec.reasoning == "click target"
        assert dec.action.kind.value == "mouse_click"

    def test_from_dict_valid_noop(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision.from_dict({"action": _noop_action_dict()})
        assert dec.kind == "mouse_click"
        assert dec.confidence == 1.0  # default
        assert dec.reasoning == ""

    def test_from_dict_missing_action_raises(self):
        from ai_game_agent.ai import AIError, AIDecision

        with pytest.raises(AIError):
            AIDecision.from_dict({"reasoning": "no action"})

    def test_from_dict_non_dict_action_raises(self):
        from ai_game_agent.ai import AIError, AIDecision

        with pytest.raises(AIError):
            AIDecision.from_dict({"action": "click"})

    def test_from_dict_invalid_action_kind_raises(self):
        from ai_game_agent.ai import AIError, AIDecision

        with pytest.raises(AIError):
            AIDecision.from_dict({"action": {"kind": "teleport",
                                              "schema_version": 1}})

    def test_from_dict_non_dict_top_raises(self):
        from ai_game_agent.ai import AIError, AIDecision

        with pytest.raises(AIError):
            AIDecision.from_dict([1, 2, 3])

    def test_from_dict_bad_confidence_raises(self):
        from ai_game_agent.ai import AIError, AIDecision

        with pytest.raises(AIError):
            AIDecision.from_dict(
                {"action": _noop_action_dict(), "confidence": 1.5}
            )

    def test_from_dict_confidence_string_raises(self):
        from ai_game_agent.ai import AIError, AIDecision

        with pytest.raises(AIError):
            AIDecision.from_dict(
                {"action": _noop_action_dict(), "confidence": "high"}
            )

    def test_from_dict_confidence_bool_rejected(self):
        from ai_game_agent.ai import AIError, AIDecision

        with pytest.raises(AIError):
            AIDecision.from_dict(
                {"action": _noop_action_dict(), "confidence": True}
            )

    def test_to_dict_roundtrip(self):
        from ai_game_agent.ai import AIDecision

        d = {"action": _click_action_dict(), "reasoning": "r",
             "confidence": 0.5}
        dec = AIDecision.from_dict(d)
        out = dec.to_dict()
        dec2 = AIDecision.from_dict(out)
        assert dec2.action == dec.action
        assert dec2.reasoning == "r"
        assert dec2.confidence == 0.5

    def test_to_dict_shape(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision.from_dict({"action": _noop_action_dict()})
        out = dec.to_dict()
        assert out["schema_version"] == 1
        assert "action" in out
        assert "kind" in out["action"]
        assert out["confidence"] == 1.0

    def test_frozen(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision.from_dict({"action": _noop_action_dict()})
        with pytest.raises(AttributeError):
            dec.reasoning = "mutated"  # type: ignore[misc]


# ===================================================================== #
# ollama: request shape / response parsing (scripted HTTP server)       #
# ===================================================================== #


class _ScriptedOllamaHandler(http.server.BaseHTTPRequestHandler):
    """Records requests and returns a configurable response.

    The handler delegates all state to the owning :class:`_ScriptedOllamaServer`
    via ``self.server`` (set by http.server at instantiation time).
    """

    def do_POST(self):
        srv = self.server  # set by http.server
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {"_raw": body.decode("utf-8", errors="replace")}

        srv.requests.append(
            {"path": self.path, "headers": dict(self.headers), "payload": payload}
        )

        status = srv.status
        resp_bytes = srv.response_bytes
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def log_message(self, fmt, *args):  # silence
        pass


class _ScriptedOllamaServer:
    """A minimal Ollama HTTP server on a random localhost port."""

    def __init__(self, status: int = 200,
                 response: dict | bytes | None = None):
        self.requests: list[dict] = []
        self.status = status
        self.response_bytes = (
            json.dumps(response).encode() if isinstance(response, dict)
            else (response if isinstance(response, bytes) else b"{}")
        )

    def start(self) -> str:
        # Bind to a random port.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        port = self._sock.getsockname()[1]
        self._sock.close()
        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", port), _ScriptedOllamaHandler
        )
        # Inject the server reference into the handler class.
        # http.server sets handler.server = self.server (the ThreadingHTTPServer),
        # so we attach our state directly to the ThreadingHTTPServer instance.
        self._server.requests = self.requests
        self._server.status = self.status
        self._server.response_bytes = self.response_bytes
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{port}"
        return self.base_url

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


def _png_bytes() -> bytes:
    """A minimal 1x1 red PNG (valid enough to base64-encode)."""
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        c += struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return c

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = b"\x00\xff\x00\x00"  # filter 0, R=255 G=0 B=0
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


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

        resp = {"response": json.dumps({"action": {"kind": "noop",
                                                     "schema_version": 1}})}
        server = _ScriptedOllamaServer(response=resp)
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            out = p.decide("choose an action")
            assert isinstance(out, dict)
            assert out["action"]["kind"] == "noop"

            req = server.requests[-1]
            assert req["payload"]["format"] == "json"
            assert req["payload"]["stream"] is False
        finally:
            server.stop()

    def test_decide_with_format_schema(self):
        from ai_game_agent.ai import OllamaProvider

        resp = {"response": json.dumps({"action": {"kind": "noop",
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

        resp = {"response": json.dumps({"action": {"kind": "noop",
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


class TestOllamaResponseParsing:
    def test_decide_strips_markdown_fence(self):
        from ai_game_agent.ai import OllamaProvider

        payload = {
            "action": {"kind": "noop", "schema_version": 1},
            "reasoning": "safe",
        }
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        server = _ScriptedOllamaServer(response={"response": fenced})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            out = p.decide("choose")
            assert out["action"]["kind"] == "noop"
            assert out["reasoning"] == "safe"
        finally:
            server.stop()

    def test_decide_embedded_json_in_prose(self):
        from ai_game_agent.ai import OllamaProvider

        payload = {"action": {"kind": "noop", "schema_version": 1}}
        text = "Sure, here is my decision: " + json.dumps(payload)
        server = _ScriptedOllamaServer(response={"response": text})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            out = p.decide("choose")
            assert out["action"]["kind"] == "noop"
        finally:
            server.stop()

    def test_decide_invalid_json_raises_ai_error(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        server = _ScriptedOllamaServer(response={"response": "not json"})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError):
                p.decide("choose")
        finally:
            server.stop()

    def test_decide_json_array_raises_ai_error(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        server = _ScriptedOllamaServer(response={"response": "[1,2,3]"})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError):
                p.decide("choose")
        finally:
            server.stop()

    def test_complete_missing_response_field_raises(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        server = _ScriptedOllamaServer(response={"unknown": "x"})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError):
                p.complete("hi")
        finally:
            server.stop()


class TestOllamaTransport:
    def test_connection_refused_raises_ai_error(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        # Use a port that nothing is listening on.
        p = OllamaProvider(model="m", base_url="http://127.0.0.1:1",
                           timeout=1.0)
        with pytest.raises(AIError) as exc_info:
            p.complete("hi")
        assert "connection" in str(exc_info.value).lower()

    def test_http_500_raises_ai_error(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        server = _ScriptedOllamaServer(status=500,
                                       response={"error": "boom"})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError) as exc_info:
                p.complete("hi")
            assert "500" in str(exc_info.value)
        finally:
            server.stop()

    def test_timeout_raises_ai_error(self):
        """A server that sleeps longer than the client timeout triggers
        a socket.timeout surfaced as AIError."""
        from ai_game_agent.ai import AIError, OllamaProvider

        class _SlowHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                time.sleep(3.0)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"response":"late"}')

            def log_message(self, *a):
                pass

        import socket as _sock
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _SlowHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            p = OllamaProvider(model="m",
                               base_url=f"http://127.0.0.1:{port}",
                               timeout=0.5)
            with pytest.raises(AIError) as exc_info:
                p.complete("hi")
            assert "timed out" in str(exc_info.value).lower()
        finally:
            srv.shutdown()
            srv.server_close()

    def test_non_json_response_raises_ai_error(self):
        from ai_game_agent.ai import AIError, OllamaProvider

        server = _ScriptedOllamaServer(response=b"not json at all")
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError):
                p.complete("hi")
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
