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

import ai_game_agent.__main__ as cli
from ai_game_agent.__main__ import main

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
