"""Scripted Ollama HTTP test harness shared by the Ollama test modules.

Split from tests/test_ai.py (2026-11-08). A minimal stdlib HTTP server
on a random 127.0.0.1 port; the handler delegates all state to the
owning :class:`_ScriptedOllamaServer`. Contains no tests and is not
collected by pytest (no ``test_*`` module name); it exists so
``test_ai_ollama_request.py`` and ``test_ai_ollama_response.py`` can
share one harness.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading


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
        # P2 (PR #11 review): an untrusted server may echo *request* content
        # into the HTTP status reason phrase. When ``status_reason`` is set,
        # echo it into the status line — but only if it is genuinely present
        # in the request body we just read (true echo semantics). The test
        # asserts ``marker_echoed`` to prove the reason phrase was really
        # driven by the request, not fabricated by the harness.
        reason = None
        srv.marker_echoed = False
        if getattr(srv, "status_reason", None):
            if srv.status_reason.encode("utf-8") in body:
                reason = srv.status_reason
                srv.marker_echoed = True
        self.send_response(status, message=reason)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def log_message(self, fmt, *args):  # silence
        pass


class _ScriptedOllamaServer:
    """A minimal Ollama HTTP server on a random localhost port."""

    def __init__(self, status: int = 200,
                 response: dict | bytes | None = None,
                 status_reason: str | None = None):
        self.requests: list[dict] = []
        self.status = status
        self.status_reason = status_reason
        self.response_bytes = (
            json.dumps(response).encode() if isinstance(response, dict)
            else (response if isinstance(response, bytes) else b"{}")
        )
        self._server: http.server.ThreadingHTTPServer | None = None

    @property
    def marker_echoed(self) -> bool:
        """True once the handler echoed ``status_reason`` into the reason
        phrase (proof the echo genuinely happened — the regression test
        asserts this so it cannot be vacuous)."""
        return bool(getattr(self._server, "marker_echoed", False))

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
        self._server.status_reason = self.status_reason
        self._server.marker_echoed = False
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
