"""Step 7 red — CLI ``run`` subcommand (plan §Step 7).

TDD red: the ``run`` subcommand does not exist yet, so every
``main(["run", ...])`` call fails at argparse with "invalid choice" and
``rc != 0`` — that is the *correct* red reason. These tests are the spec for
the green: they drive the committed executor / safety / state machine / WoW
policy through the (not-yet-built) ``run`` subcommand and assert the §Step-7
contract:

* ``run --once --mode observe_only`` exits 0 and prints an
  ``AgentLoopReport`` JSON with ``ticks == 1`` and ``actions_attempted == 0``
  (the OBSERVE_ONLY hard gate holds end to end).
* ``run --once --mode autonomous`` with a hostile-target observation runs at
  least one combat action (``actions_attempted >= 1`` and ``actions_ok >= 1``).
* ``run --max-ticks 3 --mode autonomous`` exits 0 with ``ticks == 3``.
* The Phase-3 ``--confirm`` policy gate still fires for the ``windows``
  backend in a non-observe mode (``rc != 0``, "requires --confirm").

Every import here resolves against the committed tree; the only missing piece
is the ``run`` subcommand itself.

The in-process ``run`` tests (1-4) are deterministic and headless. The
long-running "runs until Ctrl-C" case from the plan is deferred to the
subprocess / e2e suite (it needs a real ``uv run`` spawn + SIGINT and would
flakily hang this unit suite); it is covered by ``--once``/``--max-ticks``
here, which prove the same start-and-stop contract without a live process.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import pytest

import ai_game_agent.cli as cli
from ai_game_agent.actions import InputBackendError
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.cli import main
from ai_game_agent.config import load_config

# A config that (a) uses the mock capture backend and (b) configures the two
# target UI zones the WoW policy's COMBAT entry depends on (``target`` and
# ``target_hostile``). The zone *names* are the game adapter's literals —
# they belong in a game config, not in the loop (the §Step-7 contract keeps
# the loop game-agnostic).
_OBSERVE_CONFIG = textwrap.dedent(
    """\
    capture:
      backend: mock
    input:
      backend: mock
      mode: observe_only
    perception:
      enabled: true
      ui_zones:
        - name: target
          x: 0
          y: 0
          width: 4
          height: 4
          check: presence
        - name: target_hostile
          x: 4
          y: 0
          width: 4
          height: 4
          check: presence
    """
)


def _write_config(tmp_path: Path, body: str) -> str:
    cfg = tmp_path / "run_cfg.yaml"
    cfg.write_text(body, encoding="utf-8")
    return str(cfg)


def _stdout_report(capsys):
    """Parse the report JSON from stdout (the last non-empty line)."""
    out = capsys.readouterr().out.strip().splitlines()
    assert out, "run printed no stdout (expected the AgentLoopReport JSON)"
    return json.loads(out[-1])


def test_run_once_observe_only(tmp_path, capsys) -> None:
    """§Step-7 contract: observe_only exits 0, ticks == 1, zero actions."""
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
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
    report = _stdout_report(capsys)
    assert report["ticks"] == 1
    assert report["actions_attempted"] == 0
    assert report["actions_ok"] == 0
    assert report["actions_failed"] == 0
    # A valid final_state and a non-null stop reason (the report contract).
    assert report["final_state"]
    assert report["stopped_reason"] is not None


def test_run_once_assisted_default_timeout_smoke(tmp_path, capsys) -> None:
    """Regression: the *default* ASSISTED path must start (not crash).

    ``run --mode assisted --once --backend mock`` with neither ``--approve-all``
    nor ``--approve-timeout`` used to construct ``TerminalApprover(timeout_s=None)``
    and die with ``TypeError: '<' not supported between 'NoneType' and 'int'``
    before the loop ran. The mock capture emits no hostiles, so the machine
    stays IDLE and the approver is never consulted; the contract is simply
    that the default construction succeeds and a single tick completes.
    """
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "mock",
            "--mode",
            "assisted",
            "--config",
            cfg,
        ]
    )
    assert rc == 0
    report = _stdout_report(capsys)
    assert report["ticks"] == 1
    # IDLE (mock frame has no hostiles) and no input attempted.
    assert report["final_state"] == "idle"
    assert report["actions_attempted"] == 0


def test_run_once_autonomous_combat_actuates(tmp_path, capsys, monkeypatch) -> None:
    """§Step-7 contract: autonomous + hostile target acts (>=1 action ok).

    The mock capture backend is a uniform frame (no "hostile" pixels), so the
    real perception pipeline sees no target and stays IDLE. To make this test
    deterministic and game-agnostic, we inject a fake UI detector (via
    ``_build_perception``'s seam) that reports the target + hostile zones
    present, driving the real state machine into COMBAT. This exercises the
    real executor / safety / policy / dispatch path.
    """
    from ai_game_agent.perception import BBox, UiZoneHit

    # A hostile-target observation: target present AND hostile.
    hostile_hits = (
        UiZoneHit(zone="target", check="presence", bbox=BBox(0, 0, 4, 4), value=1.0),
        UiZoneHit(zone="target_hostile", check="presence", bbox=BBox(4, 0, 4, 4), value=1.0),
    )

    class _FakeUiDetector:
        def detect(self, frame, zone):
            for h in hostile_hits:
                if h.zone == zone.name:
                    return h
            return None

    real_build_perception = cli._build_perception

    def _stub_build_perception(args, cfg):
        perception = real_build_perception(args, cfg)
        # Rebuild with the hostile fake detector layered over the real config
        # (the config already names the two zones; the fake answers them).
        from ai_game_agent.perception import Perception

        return Perception(
            perception.config,
            template_matcher=perception._template_matcher,
            ui_detector=_FakeUiDetector(),
            ocr_engine=perception._ocr_engine,
            object_detector=perception._object_detector,
        )

    monkeypatch.setattr(cli, "_build_perception", _stub_build_perception)

    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "mock",
            "--mode",
            "autonomous",
            "--config",
            cfg,
        ]
    )
    assert rc == 0
    report = _stdout_report(capsys)
    assert report["final_state"] == "combat"
    assert report["actions_attempted"] >= 1
    assert report["actions_ok"] >= 1


def test_run_autonomous_opens_backend_and_actuates_all_actions(
    tmp_path, capsys, monkeypatch
) -> None:
    """Regression: ``_run`` must open the input backend before dispatching.

    The loop only manages the *capture* facade (start/stop); the input
    backend lifecycle is the CLI's responsibility, exactly like ``_act``
    (open before dispatch, close in finally). If the backend is left closed,
    every key/mouse action raised by the policy is returned as a backend-not-
    open failure and only the backend-free DELAY actions succeed — so the
    older ``actions_ok >= 1`` assertion goes green while no real input ever
    reaches the backend (false green). This test pins the real contract: the
    backend is opened and *every* combat action is actuated.
    """
    from ai_game_agent.actions.mock import MockBackend
    from ai_game_agent.perception import BBox, UiZoneHit

    # A hostile-target observation: target present AND hostile.
    hostile_hits = (
        UiZoneHit(zone="target", check="presence", bbox=BBox(0, 0, 4, 4), value=1.0),
        UiZoneHit(zone="target_hostile", check="presence", bbox=BBox(4, 0, 4, 4), value=1.0),
    )

    class _FakeUiDetector:
        def detect(self, frame, zone):
            for h in hostile_hits:
                if h.zone == zone.name:
                    return h
            return None

    real_build_perception = cli._build_perception

    def _stub_build_perception(args, cfg):
        perception = real_build_perception(args, cfg)
        from ai_game_agent.perception import Perception

        return Perception(
            perception.config,
            template_matcher=perception._template_matcher,
            ui_detector=_FakeUiDetector(),
            ocr_engine=perception._ocr_engine,
            object_detector=perception._object_detector,
        )

    monkeypatch.setattr(cli, "_build_perception", _stub_build_perception)

    # Spy on the backend the factory would hand the executor: the run path
    # must open() it (like _act) and the key presses must reach it.
    spy = MockBackend()

    def spy_create_backend(input_cfg, on_stop=None):
        spy.set_on_stop(on_stop)
        return spy

    monkeypatch.setattr(cli, "create_backend", spy_create_backend)

    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "mock",
            "--mode",
            "autonomous",
            "--config",
            cfg,
        ]
    )
    assert rc == 0
    report = _stdout_report(capsys)
    assert report["final_state"] == "combat"
    assert report["actions_attempted"] >= 1
    assert report["actions_ok"] >= 1
    # The real contract: no action may fail — with the backend open the
    # keypresses actuate instead of raising backend-not-open (only the
    # backend-free DELAY actions would have succeeded before the fix).
    assert report["actions_failed"] == 0
    # Both combat key presses reached the backend in order (q → w). This is
    # the proof the backend was open *during* dispatch — the mock refuses to
    # record any event while not opened, so a never-opened backend yields
    # zero presses and the actions_failed==0 assertion above fails.
    assert ("press", "q") in spy.events
    assert ("press", "w") in spy.events
    assert spy.events.index(("press", "q")) < spy.events.index(("press", "w"))
    # Cleanup: the backend is closed after the run (mirrors _act's
    # finally: backend.close()), so the run path never leaves a live backend
    # (and, for the real backend, a running hotkey listener) behind.
    assert spy.opened is False


def test_run_max_ticks_3_autonomous(tmp_path, capsys) -> None:
    """§Step-7 contract: --max-ticks 3 runs exactly 3 ticks and exits 0."""
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--max-ticks",
            "3",
            "--backend",
            "mock",
            "--mode",
            "autonomous",
            "--config",
            cfg,
        ]
    )
    assert rc == 0
    report = _stdout_report(capsys)
    assert report["ticks"] == 3
    assert report["stopped_reason"] == "max_ticks"


def test_run_windows_non_observe_requires_confirm(tmp_path) -> None:
    """§Step-7 contract: the Phase-3 --confirm gate still fires for the
    ``windows`` backend in a non-observe mode (unchanged from Phase 3)."""
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "windows",
            "--mode",
            "autonomous",
            "--config",
            cfg,
        ]
    )
    assert rc != 0


# --- --pretty (Phase 4 close-out) --------------------------------------------


def test_run_once_pretty_prints_multiline_json(tmp_path, capsys) -> None:
    """``--pretty`` renders the AgentLoopReport JSON indented (multi-line),
    honoring the same convention as ``analyze --pretty`` / ``act --pretty``.
    """
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
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
            "--pretty",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    lines = out.splitlines()
    assert len(lines) > 1, "pretty output should be multi-line indented JSON"
    # Still valid JSON, and the report contract holds.
    report = json.loads(out)
    assert report["ticks"] == 1


def test_run_once_default_compact_json_single_line(tmp_path, capsys) -> None:
    """Without ``--pretty`` the report stays compact single-line JSON
    (the existing default the e2e/replay tooling parses)."""
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
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
    out = capsys.readouterr().out.strip()
    assert "\n" not in out, "compact JSON should be a single line"
    report = json.loads(out)
    assert report["ticks"] == 1


# --- EH5 (resource lifecycle): the --record stream handle must be closed ---
#
# ``_run`` opens the ``--record`` .jsonl file once and the loop writes one
# line per event through the recorder closure — a long-lived handle. That
# handle must be closed **deterministically by production code** (success,
# exception, and emergency-stop paths), not left to CPython refcount
# teardown: teardown is non-deterministic and keeps the file locked on
# Windows (recordings/*.jsonl undeletable) and leaks fds on POSIX.
#
# The spy below makes "closed by production code" observable: the test holds
# a reference to the underlying file (``spies`` list), so GC can never close
# it for us — in the buggy tree ``close_calls`` stays 0 while the loop and
# backend paths otherwise pass (a naive "is the fd closed?" check after
# ``main()`` returns would false-green via refcount teardown).


class _SpyFile:
    """Wraps a real file: records ``close()`` calls, delegates everything."""

    def __init__(self, real) -> None:
        self._real = real
        self.close_calls = 0

    def write(self, data):  # delegate
        return self._real.write(data)

    def flush(self) -> None:  # delegate
        return self._real.flush()

    def close(self) -> None:
        self.close_calls += 1
        self._real.close()

    @property
    def closed(self) -> bool:
        return self._real.closed

    @property
    def name(self) -> str:
        return self._real.name

    def __enter__(self) -> _SpyFile:
        return self

    def __exit__(self, *exc) -> None:  # ExitStack protocol
        self.close()


def _spy_record_file(monkeypatch, tmp_path: Path) -> list:
    """Return the list of _SpyFile instances the run path captures."""
    spies: list = []

    real_open = Path.open

    def spy_open(self, mode="r", *args, **kwargs):
        if "a" in mode:
            real = real_open(self, mode, *args, **kwargs)
            spy = _SpyFile(real)
            spies.append(spy)
            return spy
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)
    return spies


def _capture_loop(run):
    """Capture the AgentLoop instance from a patched ``AgentLoop.run``."""
    captured: list = []

    def _run(self):
        captured.append(self)
        return run(self)

    return _run, captured


def test_run_record_closes_stream_on_success(tmp_path, capsys, monkeypatch) -> None:
    """EH5: the --record .jsonl handle is closed by production code after a
    successful run (not left to refcount teardown)."""
    spies = _spy_record_file(monkeypatch, tmp_path)
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "mock",
            "--mode",
            "observe_only",
            "--record",
            "--record-dir",
            str(tmp_path / "rec"),
            "--config",
            cfg,
        ]
    )
    assert rc == 0
    _stdout_report(capsys)  # report contract still holds
    assert spies, "the --record stream was never opened (spy not captured)"
    spy = spies[0]
    assert spy._real.name.endswith(".jsonl")
    # The spy *is* the active recorder (the loop wrote ticks through it).
    with open(spy._real.name, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert lines, "the recorder never wrote a single event"
    # Production code closed the handle (close_calls >= 1), and the
    # underlying file is actually released.
    assert spy.close_calls >= 1, "the --record handle was never closed by production code (leak)"
    assert spy.closed, "the --record file is still open after the run (leak)"


def test_run_record_closes_stream_when_run_raises(tmp_path, capsys, monkeypatch) -> None:
    """EH5: an exception from loop.run() still closes the --record stream
    (cleanup ran despite the failure) and the CLI keeps its error contract."""
    spies = _spy_record_file(monkeypatch, tmp_path)
    from ai_game_agent.actions import ActionError

    def _raise(self):
        raise ActionError("boom")

    monkeypatch.setattr(cli.AgentLoop, "run", _raise)
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "mock",
            "--mode",
            "observe_only",
            "--record",
            "--record-dir",
            str(tmp_path / "rec"),
            "--config",
            cfg,
        ]
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "error:" in err and "boom" in err
    assert spies, "the --record stream was never opened (spy not captured)"
    assert spies[0].close_calls >= 1, "the --record handle leaked across the exception path"
    assert spies[0].closed


def test_run_record_closes_stream_on_emergency_stop(tmp_path, capsys, monkeypatch) -> None:
    """EH5: an emergency stop (F12) halts the loop AND closes the --record
    stream — the early-exit path releases every owned resource, not just the
    input backend."""
    from ai_game_agent.agent import AgentLoopReport
    from ai_game_agent.state import AgentState

    spies = _spy_record_file(monkeypatch, tmp_path)
    backend_spy = MockBackend()

    def spy_create_backend(input_cfg, on_stop=None):
        backend_spy.set_on_stop(on_stop)
        return backend_spy

    monkeypatch.setattr(cli, "create_backend", spy_create_backend)

    def _run(self):
        # Fire the stop the same way the F12 hotkey would.
        backend_spy.trigger_emergency_stop("f12")
        return AgentLoopReport(
            ticks=1,
            transitions=(),
            actions_attempted=0,
            actions_ok=0,
            actions_failed=0,
            errors=(),
            final_state=AgentState.PAUSED,
            stopped_reason="stop_requested",
            wall_time_s=0.0,
        )

    monkeypatch.setattr(cli.AgentLoop, "run", _run)

    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "mock",
            "--mode",
            "autonomous",
            "--record",
            "--record-dir",
            str(tmp_path / "rec"),
            "--config",
            cfg,
        ]
    )
    assert rc == 0
    # The stop actually fired through the wired callback and the backend is
    # released (existing contract).
    assert backend_spy.stopped is True
    assert backend_spy.opened is False
    # EH5: the record stream was released too.
    assert spies, "the --record stream was never opened (spy not captured)"
    assert spies[0].close_calls >= 1, "the --record handle leaked across the emergency-stop path"
    assert spies[0].closed


class _RaisingCloseBackend(MockBackend):
    """A backend whose close() fails (e.g. the OS refuses to release a key).

    The rest of the lifecycle is faithful, so the run path's cleanup
    ordering is exactly what production would exercise.
    """

    def close(self) -> None:
        super().close()  # release_all() + opened=False (faithful teardown)
        raise InputBackendError("backend close failed (simulated OS error)")


def test_run_record_closes_even_if_backend_close_raises(tmp_path, capsys, monkeypatch) -> None:
    """EH5: a failing backend.close() must not defeat the record-stream close.

    Nested-finally guarantee: the record stream's release is registered so it
    runs *even when* the backend's primary cleanup raises.
    """
    spies = _spy_record_file(monkeypatch, tmp_path)
    backend_spy = _RaisingCloseBackend()

    def spy_create_backend(input_cfg, on_stop=None):
        backend_spy.set_on_stop(on_stop)
        return backend_spy

    monkeypatch.setattr(cli, "create_backend", spy_create_backend)

    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "mock",
            "--mode",
            "autonomous",
            "--record",
            "--record-dir",
            str(tmp_path / "rec"),
            "--config",
            cfg,
        ]
    )
    err = capsys.readouterr().err
    # The backend-close failure surfaces as the CLI's clean error contract
    # (not a raw traceback) — and the record stream is still closed.
    assert rc == 1
    assert "error:" in err
    assert spies, "the --record stream was never opened (spy not captured)"
    assert spies[0].close_calls >= 1, "a failing backend.close() defeated the record-stream close"
    assert spies[0].closed


class _RaisingLoopCtor:
    """Stand-in for AgentLoop whose constructor raises (F2).

    Mirrors a real construction failure, e.g. an invalid ``max_ticks``
    (``AgentLoop.__init__`` raises ``ValueError`` for ``max_ticks < 1``).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ValueError("simulated AgentLoop construction failure")


def test_run_record_closes_when_agentloop_ctor_raises(tmp_path, capsys, monkeypatch) -> None:
    """F2: a failing AgentLoop constructor must not leak the record stream.

    The ``--record`` .jsonl handle is registered on an ExitStack *before* the
    loop is built. If ``AgentLoop.__init__`` raises, that handle must still be
    released by production code — it must not be left open (a leaked fd/file
    lock). The existing lifecycle tests cover the run/close/exception paths;
    this one covers the *construction-failure* path, where the loop is never
    built at all.
    """
    spies = _spy_record_file(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "AgentLoop", _RaisingLoopCtor)

    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "mock",
            "--mode",
            "autonomous",
            "--record",
            "--record-dir",
            str(tmp_path / "rec"),
            "--config",
            cfg,
        ]
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "error:" in err
    assert spies, "the --record stream was never opened (spy not captured)"
    assert spies[0].close_calls >= 1, (
        "the --record handle leaked when the AgentLoop constructor raised"
    )
    assert spies[0].closed


# ---------------------------------------------------------------------------
# F5 — cleanup-failure semantics
#
# The requirement: the *primary* (operational) exception always remains the
# exception that propagates out of the run path, no matter how many cleanup
# operations (backend.close(), record-stream flush/close) fail in the final
# region. Every cleanup failure must be observable — attached to the primary
# exception via ``BaseException.add_note`` (Python 3.11+), never silently
# swallowed, and never allowed to *replace* the primary. A lone cleanup
# failure (no operational exception) must still propagate as an error.
#
# These tests drive ``cli._run`` directly (not ``main``) so the raw exception
# object is observable — its type, its message, and any notes added by the
# cleanup region. Driving through ``main`` would only show the rendered
# "error:" line, which is what the existing lifecycle tests already check.
# ---------------------------------------------------------------------------


class _RunException(Exception):
    """The primary operational exception (e.g. a crash mid-run)."""


class _FailingCloseBackend(MockBackend):
    """A backend whose close() fails (e.g. the OS refuses to release a key)."""

    def close(self) -> None:
        raise InputBackendError("backend close failed (simulated OS error)")


class _SpyStream:
    """A record stream whose write/flush/close can fail independently.

    Implements the context-manager protocol so ``ExitStack.enter_context``
    and ``stack.close()`` work correctly (``__exit__`` delegates to
    ``close()``).
    """

    def __init__(self, fail_write: bool = False, fail_flush: bool = False,
                 fail_close: bool = False) -> None:
        self.fail_write = fail_write
        self.fail_flush = fail_flush
        self.fail_close = fail_close
        self.writes = 0
        self.flushes = 0
        self.closes = 0

    def __enter__(self) -> _SpyStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write(self, data: str) -> int:
        self.writes += 1
        if self.fail_write:
            raise OSError("record stream write failed (simulated disk full)")
        return len(data)

    def flush(self) -> None:
        self.flushes += 1
        if self.fail_flush:
            raise OSError("record stream flush failed (simulated disk full)")

    def close(self) -> None:
        self.closes += 1
        if self.fail_close:
            raise OSError("record stream close failed (simulated OS error)")


def _f5_args(tmp_path: Path, cfg_path: str) -> argparse.Namespace:
    """A minimal args namespace for ``cli._run`` (autonomous, --once, --record)."""
    return argparse.Namespace(
        record=True,
        record_dir=str(tmp_path / "rec"),
        mode="autonomous",
        backend="mock",
        confirm=False,
        config=cfg_path,
        max_ticks=1,
        once=True,
        fps=None,
        region=None,
        pretty=False,
    )


def _f5_setup(tmp_path, monkeypatch, *,
              backend: MockBackend | None = None,
              run_exc: Exception | None = None,
              stream: _SpyStream | None = None) -> dict:
    """Wire _run's collaborators: backend, loop.run, and the record stream."""
    captured: dict = {"open_calls": 0, "close_calls": 0}

    if backend is None:
        backend = MockBackend()

    def spy_create_backend(input_cfg, on_stop=None):
        return backend

    monkeypatch.setattr(cli, "create_backend", spy_create_backend)

    # Replace AgentLoop with a fake whose run() does what we tell it.
    fake_loop_holder: dict = {"loop": None}

    class _FakeLoop:
        def __init__(self, *a: object, **kw: object) -> None:
            fake_loop_holder["loop"] = self

        def run(self) -> object:
            captured["run_calls"] = captured.get("run_calls", 0) + 1
            if run_exc is not None:
                raise run_exc
            from ai_game_agent.agent.report import AgentLoopReport
            from ai_game_agent.state import AgentState
            return AgentLoopReport(
                ticks=1,
                transitions=(),
                actions_attempted=0,
                actions_ok=0,
                actions_failed=0,
                errors=(),
                final_state=AgentState.IDLE,
                stopped_reason=None,
                wall_time_s=0.0,
            )

    monkeypatch.setattr(cli, "AgentLoop", _FakeLoop)

    if stream is not None:
        # Intercept the record file open: return our spy stream instead.
        real_open = Path.open

        def spy_open(self, mode="r", *args: object, **kwargs: object):
            if "a" in mode:
                return stream
            return real_open(self, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", spy_open)

    return {"backend": backend, "captured": captured, "loop": fake_loop_holder}


def test_f5_scenario1_operational_ok_cleanup_ok(tmp_path, monkeypatch, capsys) -> None:
    """F5-1: everything works, cleanup succeeds → normal return, record closed.

    This is the baseline that must not regress: the existing success-path
    tests already cover the visible surface (report JSON, exit 0). Here we
    assert the internal guarantee: the record stream was closed by production
    code (not left to teardown).
    """
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    stream = _SpyStream()
    env = _f5_setup(tmp_path, monkeypatch, stream=stream)
    args = _f5_args(tmp_path, cfg)

    rc = cli._run(args, load_config(cfg))

    assert rc == 0
    assert env["captured"]["run_calls"] == 1
    assert stream.closes >= 1, "the record stream was not closed by production code"


def test_f5_scenario2_operational_fails_cleanup_ok(tmp_path, monkeypatch) -> None:
    """F5-2: the operational exception propagates, cleanup succeeds.

    The primary exception must be the one that propagates (not a new exception
    from the cleanup region). The record stream must still be closed.
    """
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    stream = _SpyStream()
    primary = _RunException("agent crashed mid-run")
    _ = _f5_setup(tmp_path, monkeypatch, run_exc=primary, stream=stream)
    args = _f5_args(tmp_path, cfg)

    with pytest.raises(_RunException) as exc_info:
        cli._run(args, load_config(cfg))

    assert exc_info.value is primary, "the primary exception was replaced"
    assert stream.closes >= 1, "the record stream leaked on the exception path"


def test_f5_scenario3_operational_ok_cleanup_fails(tmp_path, monkeypatch) -> None:
    """F5-3: run succeeds but backend.close() fails in cleanup.

    The close() failure must propagate (it is the only failure), be a clean
    error (InputBackendError), and the record stream must still be closed
    (nested finally).
    """
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    stream = _SpyStream()
    backend = _FailingCloseBackend()
    _ = _f5_setup(tmp_path, monkeypatch, backend=backend, stream=stream)
    args = _f5_args(tmp_path, cfg)

    with pytest.raises(InputBackendError) as exc_info:
        cli._run(args, load_config(cfg))

    assert "backend close failed" in str(exc_info.value)
    assert stream.closes >= 1, "a failing backend.close() defeated the record-stream close"


def test_f5_scenario4_operational_fails_cleanup_fails(tmp_path, monkeypatch) -> None:
    """F5-4: BOTH the operational exception and cleanup fail.

    The primary operational exception must be the one that propagates (the
    cleanup failure must NOT mask it). The cleanup failure must be observable
    — attached to the primary exception via ``add_note`` (Python 3.11).
    The record stream must still be closed.
    """
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    stream = _SpyStream(fail_close=True)
    backend = _FailingCloseBackend()
    primary = _RunException("agent crashed mid-run")
    _ = _f5_setup(tmp_path, monkeypatch, backend=backend, run_exc=primary,
                    stream=stream)
    args = _f5_args(tmp_path, cfg)

    with pytest.raises(_RunException) as exc_info:
        cli._run(args, load_config(cfg))

    assert exc_info.value is primary, (
        "the primary operational exception was masked by a cleanup failure"
    )
    # The cleanup failures must be observable on the primary exception.
    notes = getattr(exc_info.value, "__notes__", [])
    notes_text = "\n".join(str(n) for n in notes)
    assert "backend close failed" in notes_text, (
        "the backend.close() failure was not made observable on the primary "
        f"exception (notes: {notes!r})"
    )
    assert "record stream close failed" in notes_text or (
        "record" in notes_text
    ), "the record-stream close failure was not made observable"
    assert stream.closes >= 1, "the record stream was never attempted for close"


def test_f5_scenario5_multiple_cleanup_failures(tmp_path, monkeypatch) -> None:
    """F5-5: run succeeds, but EVERY cleanup operation fails.

    Both backend.close() and the record-stream close raise. The first cleanup
    failure propagates (no primary exists), and the *other* failures must be
    observable via ``add_note`` — none silently swallowed.
    """
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    stream = _SpyStream(fail_close=True, fail_flush=True)
    backend = _FailingCloseBackend()
    _ = _f5_setup(tmp_path, monkeypatch, backend=backend, stream=stream)
    args = _f5_args(tmp_path, cfg)

    with pytest.raises(InputBackendError) as exc_info:
        cli._run(args, load_config(cfg))

    assert "backend close failed" in str(exc_info.value)
    # The record-stream close failure must be observable on the primary.
    notes = getattr(exc_info.value, "__notes__", [])
    notes_text = "\n".join(str(n) for n in notes)
    assert "record stream close failed" in notes_text or (
        "record" in notes_text
    ), (
        "the record-stream close failure was silently swallowed; expected it "
        f"on the primary exception's notes (notes: {notes!r})"
    )
    assert stream.closes >= 1


# ---------------------------------------------------------------------------
# F6 — _act: a failed backend.open() must not leave a half-open backend
# ---------------------------------------------------------------------------


class _RaisingOpenBackend(MockBackend):
    """A backend whose open() fails (e.g. a missing or locked OS input hook)."""

    def open(self) -> None:
        self.opened = True
        raise InputBackendError("backend open failed (simulated OS error)")


def test_act_failing_open_still_closes_backend(tmp_path, capsys, monkeypatch) -> None:
    """F6: a failing backend.open() must still be closed by the act path.

    The ``act`` subcommand opens the backend, runs actions, and closes it in
    a ``finally``. If ``open()`` itself fails, the backend is left in a
    half-open state (``opened == True`` but no listener armed). The current
    code calls ``backend.open()`` *outside* the ``try/finally`` that calls
    ``backend.close()`` — so a failing ``open()`` skips the ``close()`` and
    the half-open backend is leaked. This test catches that.
    """
    backend_spy = _RaisingOpenBackend()

    def spy_create_backend(input_cfg, on_stop=None):
        return backend_spy

    monkeypatch.setattr(cli, "create_backend", spy_create_backend)

    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "act",
            "--backend",
            "mock",
            "--mode",
            "autonomous",
            "--do",
            "key:a",
            "--config",
            cfg,
        ]
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "error:" in err
    assert "backend open failed" in err, (
        f"the open() failure must surface as a clean CLI error; got: {err!r}"
    )
    # The core F6 assertion: the half-open backend was closed.
    assert backend_spy.opened is False, (
        "a failing backend.open() left the backend half-open; the act path "
        "must call close() even when open() fails"
    )


def test_act_failing_open_releases_held_input(tmp_path, capsys, monkeypatch) -> None:
    """F6 (safety variant): a failing open() must not leave held input stuck.

    If the backend had held keys/buttons before the failed open() (e.g. a
    retry path), the close-on-failure must drain them. This mirrors the
    AGENTS.md safety rule: a half-open backend must never leave a key stuck.
    """
    backend_spy = _RaisingOpenBackend()
    # Simulate a held key that would be leaked if close() is skipped.
    backend_spy.held_keys.add("a")

    def spy_create_backend(input_cfg, on_stop=None):
        return backend_spy

    monkeypatch.setattr(cli, "create_backend", spy_create_backend)

    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "act",
            "--backend",
            "mock",
            "--mode",
            "autonomous",
            "--do",
            "key:a",
            "--config",
            cfg,
        ]
    )
    assert rc == 1
    # The held key must be drained by close().
    assert backend_spy.held_keys == set(), (
        "a failing backend.open() left a held key stuck; the act path must "
        "release all held input even when open() fails"
    )

# --- F1: harden the --record filename convention ----------------------------
#
# The --record file must be named run-<stamp>.jsonl where <stamp> is
# %Y%m%d-%H%M%S (8 digits, dash, 6 digits). A bare endswith(".jsonl") check
# (used by the close-leak tests) is too weak to catch a naming regression —
# e.g. a "run.txt" or a missing dash. Freeze the clock and assert the full
# shape.


def test_run_record_filename_matches_timestamped_convention(tmp_path, capsys, monkeypatch) -> None:
    """F1: the --record file name is run-YYYYMMDD-HHMMSS.jsonl exactly."""
    import re
    import time as _time

    spies = _spy_record_file(monkeypatch, tmp_path)
    # Freeze the clock so the stamp is deterministic (2026-11-11 12:00:00).
    # Only the single-arg call (the CLI's stamp) is intercepted; any other
    # call shape (e.g. strftime called elsewhere with a struct_time arg) is
    # delegated to the real function so the global patch stays surgical.
    _fixed = _time.struct_time((2026, 11, 11, 12, 0, 0, 2, 315, 0))
    _real_strftime = _time.strftime

    def _frozen_strftime(fmt, *args: object, **kw: object) -> str:
        if args or kw:
            return _real_strftime(fmt, *args, **kw)  # type: ignore[arg-type]
        return _real_strftime(fmt, _fixed)

    monkeypatch.setattr(_time, "strftime", _frozen_strftime)
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--once",
            "--backend",
            "mock",
            "--mode",
            "observe_only",
            "--record",
            "--record-dir",
            str(tmp_path / "rec"),
            "--config",
            cfg,
        ]
    )
    assert rc == 0
    assert spies, "the --record stream was never opened (spy not captured)"
    name = Path(spies[0]._real.name).name
    assert name == "run-20261111-120000.jsonl", (
        f"the --record file must be run-YYYYMMDD-HHMMSS.jsonl; got {name!r}"
    )
    assert re.fullmatch(r"run-\d{8}-\d{6}\.jsonl", name), (
        f"the --record file name does not match run-<8digits>-<6digits>.jsonl; got {name!r}"
    )
