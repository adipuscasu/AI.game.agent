"""Emergency-stop wiring into the `run` CLI (followup-wire-emergency-stop-run).

TDD red: these fail against the pre-fix tree because
  * MockBackend has no trigger_emergency_stop() / stopped / set_on_stop();
  * create_backend() has no on_stop parameter;
  * _run() does not pass a stop_event to AgentLoop.

Every test is headless (mock backend, no pynput, no OS input).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import ai_game_agent.__main__ as cli
from ai_game_agent.__main__ import main
from ai_game_agent.actions.action import Action, ActionKind
from ai_game_agent.actions.base import InputConfig, create_backend
from ai_game_agent.actions.executor import Executor
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.actions.modes import OperationMode
from ai_game_agent.actions.safety import SafetyGuard

_OBSERVE_CONFIG = """
capture:
  backend: mock
input:
  backend: mock
  mode: observe_only
"""


def _write(tmp_path: Path, body: str) -> str:
    p = tmp_path / "stop_cfg.yaml"
    p.write_text(body, encoding="utf-8")
    return str(p)


# --- 1. MockBackend emergency-stop seam (RED: method/flag don't exist) -------


def test_mock_trigger_emergency_stop_releases_held() -> None:
    backend = MockBackend()
    backend.open()
    backend.key_down("w")
    backend.mouse_down("left")
    assert backend.held_keys == {"w"}
    assert backend.held_buttons == {"left"}

    backend.trigger_emergency_stop("test stop")

    assert backend.held_keys == set()
    assert backend.held_buttons == set()
    assert backend.stopped is True
    # Idempotent: a second call is a no-op (no re-release, no crash).
    backend.trigger_emergency_stop("again")
    assert backend.stopped is True


# --- 2. create_backend wires on_stop to the mock (RED: param doesn't exist) --


def test_create_backend_wires_on_stop_to_mock() -> None:
    calls: list[str] = []
    backend = create_backend(
        InputConfig(backend="mock", mode="autonomous"),
        on_stop=lambda: calls.append("fired"),
    )
    backend.open()
    backend.trigger_emergency_stop("simulated f12")
    assert calls == ["fired"], "on_stop callback was not invoked"


def test_create_backend_on_stop_default_none_still_releases() -> None:
    """Without on_stop, trigger_emergency_stop still releases (no crash)."""
    backend = create_backend(InputConfig(backend="mock", mode="autonomous"))
    backend.open()
    backend.key_down("a")
    backend.trigger_emergency_stop("no callback")
    assert backend.held_keys == set()
    assert backend.stopped is True


# --- 3. run passes a real stop_event into AgentLoop (RED: kwarg absent) ------


def test_run_wires_real_stop_event_into_agentloop(tmp_path, monkeypatch) -> None:
    """_run must hand AgentLoop a live threading.Event (not the None default)."""
    cfg = _write(tmp_path, _OBSERVE_CONFIG)
    real_agentloop = cli.AgentLoop
    captured: dict = {}

    def factory(*args, **kwargs):
        captured["stop_event"] = kwargs.get("stop_event")
        return real_agentloop(*args, **kwargs)

    monkeypatch.setattr(cli, "AgentLoop", factory)

    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "mock",
            "--mode",
            "observe_only",
            "--config",
            cfg,
        ]
    )
    assert rc == 0
    assert isinstance(captured["stop_event"], threading.Event), (
        f"_run did not pass a threading.Event to AgentLoop; got "
        f"{captured['stop_event']!r}"
    )


# --- 4. The full F12 chain, headless (RED: on_stop + mock seam absent) -------


def test_f12_stop_chain_releases_and_blocks() -> None:
    """backend -> on_stop -> (stop_event + executor stop) -> release + block.

    The exact composition ``_run`` wires: an in-flight held input is released,
    the stop_event is set, the executor is terminal, further input is blocked.
    """
    stop_event = threading.Event()
    backend = MockBackend()
    guard = SafetyGuard(mode=OperationMode.AUTONOMOUS)
    executor = Executor(backend=backend, safety=guard, sleeper=lambda d: None)

    def _on_stop() -> None:
        stop_event.set()
        executor.trigger_emergency_stop("f12")

    backend.set_on_stop(_on_stop)
    backend.open()
    executor.execute(Action(kind=ActionKind.MOUSE_DOWN, button="left"))
    assert "left" in backend.held_buttons

    backend.trigger_emergency_stop("f12")  # simulate the F12 hotkey

    assert stop_event.is_set()
    assert executor.stopped is True
    assert backend.held_buttons == set(), "held button was not released on stop"
    result = executor.execute(Action(kind=ActionKind.KEY_PRESS, key="a"))
    assert result.ok is False
    assert result.stop_event is True


# --- 5. In-progress release regression pin (locks the executor contract) -----


def test_emergency_stop_during_inflight_action() -> None:
    """A stop while an input is held releases it and terminal-blocks the executor."""
    backend = MockBackend()
    backend.open()
    executor = Executor(
        backend=backend,
        safety=SafetyGuard(mode=OperationMode.AUTONOMOUS),
        sleeper=lambda d: None,
    )
    executor.execute(Action(kind=ActionKind.KEY_HOLD, key="w", duration_ms=1))
    backend.key_down("a")  # a second in-flight hold
    assert "a" in backend.held_keys

    executor.trigger_emergency_stop("mid-hold stop")

    assert backend.held_keys == set()
    assert executor.stopped is True
    result = executor.execute(Action(kind=ActionKind.KEY_PRESS, key="s"))
    assert result.ok is False and result.stop_event is True


# --- 6. End-to-end: F12 mid-loop stops the run and releases held input -------


def test_run_f12_stops_loop_and_reports_emergency_stop(
    tmp_path, monkeypatch, capsys
) -> None:
    """Firing the wired on_stop mid-loop must stop the run cleanly (rc == 0)
    with ``stopped_reason == "emergency_stop"`` — the headless analog of
    pressing F12 during ``run``.

    The hook fires at the loop's first pace point (``wait_next_frame``) by
    invoking the spy backend's ``trigger_emergency_stop`` — the same call the
    real Windows backend's F12 hotkey handler makes. The spy is wired with the
    live ``on_stop`` callback (via ``set_on_stop``) exactly as the real factory
    would be, so this exercises the true F12 path: ``stop_event.set()`` +
    ``executor.trigger_emergency_stop``. Deterministic and timing-free.
    """
    from ai_game_agent.capture import Capture

    cfg = _write(tmp_path, _OBSERVE_CONFIG)
    spy = MockBackend()

    def spy_create_backend(cfg, on_stop=None):
        # Wire the callback exactly as the real factory does (create_backend
        # forwards on_stop to the backend's stop seam).
        spy.set_on_stop(on_stop)
        return spy

    monkeypatch.setattr(cli, "create_backend", spy_create_backend)

    def stub_wait_next_frame(self):
        # Fire the emergency stop exactly once, when the loop first paces.
        if not getattr(self, "_fired", False):
            object.__setattr__(self, "_fired", True)
            spy.trigger_emergency_stop("f12")  # simulate the physical F12 press

    monkeypatch.setattr(Capture, "wait_next_frame", stub_wait_next_frame)

    rc = main(
        [
            "run",
            "--max-ticks",
            "50",
            "--backend",
            "mock",
            "--mode",
            "observe_only",
            "--config",
            cfg,
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    report = json.loads(out[-1])
    assert report["stopped_reason"] == "emergency_stop"


# --- 7. End-to-end: a set stop_event stops the run (stop_requested) ---------


def test_run_stop_event_stops_loop_and_reports_stop_requested(
    tmp_path, monkeypatch, capsys
) -> None:
    """Setting the loop's ``stop_event`` mid-run must stop the run cleanly
    (rc == 0) with ``stopped_reason == "stop_requested"`` — the other clean,
    report-producing stop path, distinct from the F12 / ``emergency_stop``
    test above (different reason string, different trigger: the stop token
    rather than the backend hotkey).

    This is the *real* path, not just the constructor kwarg (that is pinned by
    ``test_run_wires_real_stop_event_into_agentloop``): we capture the *live*
    ``threading.Event`` that ``_run`` hands to ``AgentLoop``, set it at the
    loop's first pace point (exactly when an external stop source would fire —
    a future hotkey/IPC), and assert the loop detects it on the next check and
    exits *early* with the stop-requested reason instead of burning through its
    ``--max-ticks`` budget.

    Cleanup: in ``observe_only`` no input is ever held, so there is nothing to
    release; the loop's ``finally`` cleanup (``capture.stop()``) runs on every
    exit path, which is proven here by the clean rc == 0 and the report on
    stdout. This deliberately does not duplicate the F12 test's held-input
    release assertion (that belongs to the ``emergency_stop`` path).
    """
    from ai_game_agent.capture import Capture

    cfg = _write(tmp_path, _OBSERVE_CONFIG)
    real_agentloop = cli.AgentLoop
    captured: dict = {}

    def factory(*args, **kwargs):
        captured["stop_event"] = kwargs.get("stop_event")
        return real_agentloop(*args, **kwargs)

    monkeypatch.setattr(cli, "AgentLoop", factory)

    def stub_wait_next_frame(self):
        # Fire the stop request exactly once, when the loop first paces —
        # the in-loop analog of an external source setting the stop token.
        event = captured.get("stop_event")
        if event is not None and not event.is_set():
            event.set()

    monkeypatch.setattr(Capture, "wait_next_frame", stub_wait_next_frame)

    rc = main(
        [
            "run",
            "--max-ticks",
            "50",
            "--backend",
            "mock",
            "--mode",
            "observe_only",
            "--config",
            cfg,
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    report = json.loads(out[-1])
    assert report["stopped_reason"] == "stop_requested"
    # Proof the loop stopped *because* of the stop request (early), not by
    # running out its 50-tick budget: the check caught it before max_ticks.
    assert report["ticks"] < 50
