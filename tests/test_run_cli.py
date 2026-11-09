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

import json
import textwrap
from pathlib import Path

import ai_game_agent.cli as cli
from ai_game_agent.actions import InputBackendError
from ai_game_agent.actions.mock import MockBackend
from ai_game_agent.cli import main

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
