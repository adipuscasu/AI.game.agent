"""Ollama response parsing and transport-error tests.

Split from tests/test_ai.py (2026-11-08).

Verifies response parsing (markdown-fence stripping, JSON errors) and
transport failure handling: connection errors, HTTP 5xx with
server-controlled reason-phrase sanitization, timeouts, non-JSON
bodies, and HTTP stream release. Transport runs against the scripted
stdlib HTTP harness (tests/_ai_ollama_harness.py) — CI-safe, no live
Ollama required.
"""

from __future__ import annotations

import http.server
import json
import threading
import time

import pytest
from _ai_ollama_harness import _ScriptedOllamaServer


class TestOllamaResponseParsing:
    def test_decide_strips_markdown_fence(self):
        from ai_game_agent.ai import OllamaProvider

        payload = {
            "action": {"kind": "mouse_click", "schema_version": 1},
            "reasoning": "safe",
        }
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        server = _ScriptedOllamaServer(response={"response": fenced})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            out = p.decide("choose")
            assert out.action.kind.value == "mouse_click"
            assert out.reasoning == "safe"
        finally:
            server.stop()

    def test_decide_embedded_json_in_prose(self):
        from ai_game_agent.ai import OllamaProvider

        payload = {"action": {"kind": "mouse_click", "schema_version": 1}}
        text = "Sure, here is my decision: " + json.dumps(payload)
        server = _ScriptedOllamaServer(response={"response": text})
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            out = p.decide("choose")
            assert out.action.kind.value == "mouse_click"
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

    def test_http_error_message_does_not_echo_response_body(self):
        """A non-2xx response body must not be interpolated into the
        exception message: a non-trusted endpoint could echo back the
        request payload (which may contain base64 screen images) and the
        error text flows into logs, event streams, and recovery events.

        The marker must be sent as raw ``bytes`` so the scripted server
        actually echoes it back (a ``str`` would be coerced to ``b"{}"``
        and the echo — and this test — would be vacuous)."""
        from ai_game_agent.ai import AIError, OllamaProvider

        marker_bytes = b"ECHOED-SECRET-MARKER-EH2"
        server = _ScriptedOllamaServer(status=502, response=marker_bytes)
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError) as exc_info:
                p.complete("hi")
            msg = str(exc_info.value)
            assert "502" in msg
            assert marker_bytes.decode("utf-8") not in msg
            assert repr(marker_bytes) not in msg
        finally:
            server.stop()

    def test_http_error_error_stream_is_closed_not_drained(self):
        """A non-2xx response stream must be *closed*, not drained.

        Reading the error body pulls it into the process, and it may echo
        the request payload (prompts can carry base64 screen images) or
        server internals. The contract is best-effort release via
        ``close()``; the discriminator is ``HTTPError.closed`` — a drained
        (``read()``) stream stays ``closed == False``, a closed one is
        ``closed == True``.

        The failure must still surface as an AIError with the status, and
        the HTTPError must remain on the ``__cause__`` chain so the
        release is auditable."""
        import urllib.error

        from ai_game_agent.ai import AIError, OllamaProvider

        marker = "ECHOED-SECRET-MARKER-EH3"
        server = _ScriptedOllamaServer(status=502, response=marker.encode("utf-8"))
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError) as exc_info:
                p.complete("hi")
            msg = str(exc_info.value)
            assert "502" in msg
            assert marker not in msg

            # The HTTPError must remain on the __cause__ chain.
            http_err = None
            node = exc_info.value
            seen = set()
            while node is not None and id(node) not in seen:
                seen.add(id(node))
                if isinstance(node, urllib.error.HTTPError):
                    http_err = node
                    break
                node = node.__cause__
            assert http_err is not None, "the HTTPError must remain on the __cause__ chain"
            assert http_err.closed is True, (
                "the non-2xx error stream must be closed (released), not drained"
            )
        finally:
            server.stop()

    def test_http_error_status_reason_cannot_leak_request_content(self):
        """The HTTP status *reason phrase* is server-controlled and must not
        reach any diagnostic surface (P2, PR #11 review).

        A malicious (or echoing) local server can read the POST body and put
        request content — prompts, base64 screen images — into the status
        reason phrase of its error response. ``HTTPError`` stores that phrase
        in ``.msg`` and exposes it through ``.reason``, ``str()`` and
        ``repr()``; the provider chains the ``HTTPError`` as ``__cause__``,
        so the untrusted phrase would reappear in the ``AIError`` message
        *and* in the agent-loop's exception diagnostics (``exc_repr``,
        ``exc_cause``, and the formatted traceback, which renders the cause
        with its repr).

        The scripted server genuinely echoes the marker from the request
        body into the reason phrase (and we assert that echo happened), so
        this exercises the reason phrase, not the response body.

        Required invariants:
          * the failure is still an ``AIError`` with the status + path;
          * the marker is absent from ``str(AIError)``;
          * the ``HTTPError`` remains on the ``__cause__`` chain with its
            stream closed (observability preserved), but its ``msg``/
            ``reason``/``str()``/``repr()`` no longer carry the marker;
          * the marker is absent from the agent-loop exception diagnostics
            (``exc_repr``/``exc_cause``/``exc_context``/``exc_traceback``)
            produced by ``AgentLoop._exception_detail``.
        """
        import urllib.error

        from ai_game_agent.ai import AIError, OllamaProvider

        marker = "SECRET-STATUS-REASON-MARKER"
        # Send the marker in the request body so the server can echo it back
        # in the reason phrase — exactly the attack shape from the review.
        server = _ScriptedOllamaServer(
            status=502, response={"error": "opaque"}, status_reason=marker
        )
        base_url = server.start()
        try:
            p = OllamaProvider(model="m", base_url=base_url, timeout=5.0)
            with pytest.raises(AIError) as exc_info:
                p.complete(marker)
            err = exc_info.value

            # The failure must still be an AIError carrying status + path.
            msg = str(err)
            assert "502" in msg
            assert "/api/generate" in msg
            # 1) the top-level message must not carry the marker.
            assert marker not in msg

            # The reason phrase must genuinely have been driven by the
            # request body (echo semantics), or this test is vacuous.
            assert server.marker_echoed, (
                "the scripted server did not echo the marker from the "
                "request into the status reason phrase — the test is vacuous"
            )

            # 2) the HTTPError must remain on the __cause__ chain (EH2
            # observability preserved) with its stream closed…
            http_err = None
            node = err
            seen = set()
            while node is not None and id(node) not in seen:
                seen.add(id(node))
                if isinstance(node, urllib.error.HTTPError):
                    http_err = node
                    break
                node = node.__cause__
            assert http_err is not None, "the HTTPError must remain on the __cause__ chain"
            assert http_err.closed is True

            # …but every stringification surface of the chained HTTPError
            # must be free of the marker.
            assert marker not in str(http_err)
            assert marker not in repr(http_err)
            assert marker not in str(http_err.reason)

            # 3) the agent-loop exception diagnostics (exc_repr, exc_cause,
            # exc_context, exc_traceback) must be marker-free. This runs the
            # real _exception_detail() on the real exception chain, exactly
            # as _on_pipeline_error does when an AI failure is recorded.
            from ai_game_agent.actions import Executor, OperationMode, SafetyGuard
            from ai_game_agent.actions.mock import MockBackend
            from ai_game_agent.agent import AgentLoop
            from ai_game_agent.games.wow import DeterministicPolicy
            from ai_game_agent.state import (
                AgentState,
                EventType,
                GameStateMachine,
            )

            class _StubDecisionPolicy:
                def transition_signals(self, observation):
                    return {}

            class _FakeCapture:
                def start(self):
                    return None

                def stop(self):
                    return None

                def grab(self):
                    import datetime

                    from ai_game_agent.capture import Frame

                    return Frame(
                        width=1, height=1, pixels=b"\x00" * 3,
                        captured_at=datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC),
                        source="test",
                    )

                def wait_next_frame(self):
                    return 0.0

            class _PassingPerception:
                """observe() must succeed so tick() reaches the AI stage."""

                def observe(self, frame):
                    import datetime

                    from ai_game_agent.perception import Observation

                    return Observation(
                        frame_width=1,
                        frame_height=1,
                        captured_at=datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC),
                        source="test",
                        templates=(),
                        ui_zones=(),
                        text_regions=(),
                        objects=(),
                        detector_errors=(),
                    )

            class _RaisingAI:
                def decide(self, prompt, **kwargs):
                    raise err

            machine = GameStateMachine(
                policy=_StubDecisionPolicy(),
                initial=AgentState.AI_ANALYSIS,
                recovery_attempts=1,
            )
            executor = Executor(
                backend=MockBackend(),
                safety=SafetyGuard(mode=OperationMode.OBSERVE_ONLY),
            )
            loop = AgentLoop(
                capture=_FakeCapture(),
                perception=_PassingPerception(),
                executor=executor,
                machine=machine,
                policy=DeterministicPolicy(),
                mode=OperationMode.OBSERVE_ONLY,
                ai=_RaisingAI(),
                max_ticks=1,
                now_fn=lambda: 0.0,
            )
            result = loop.tick()
            recovery = [
                e for e in loop._errors
                if e.event is EventType.RECOVERY
            ]
            assert recovery, "the AI failure must route to a RECOVERY event"
            detail = recovery[0].detail
            for field in ("exc_repr", "exc_cause", "exc_context", "exc_traceback"):
                if field in detail:
                    assert marker not in str(detail[field]), (
                        f"{field} leaks the status-reason marker"
                    )
            assert marker not in str(detail.get("error", ""))
            assert marker not in str(recovery[0].reason)
            assert result.error is not None
            assert marker not in str(result.error)
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
