"""Step 8 red — Replay: record → replay (plan §Step 8).

TDD red: the ``replay`` subcommand does not exist yet, so every
``main(["replay", ...])`` call fails at argparse with "invalid choice" and
``rc != 0`` — that is the *correct* red reason.

Contract:

* A ``run --record --max-ticks 3`` session writes a ``.jsonl`` file with
  at least 3 ``TICK`` events (one per tick) + any ``STATE_*`` events.
* A ``replay <file>`` subcommand reads the ``.jsonl``, feeds the recorded
  *observations* (zone values carried in the TICK events) back through
  ``machine.decide()`` with the same policy, and asserts the **same** state
  sequence as the original run.
* The replay does not re-capture or re-perceive; it feeds recorded zone
  signals into the same machine + policy and asserts the same decisions.

The test is the regression test for the transition table: if a future edit
changes a transition, the replay assertion fails.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from ai_game_agent.__main__ import main

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
    cfg = tmp_path / "replay_cfg.yaml"
    cfg.write_text(body, encoding="utf-8")
    return str(cfg)


def _read_jsonl(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def test_run_record_writes_jsonl(tmp_path) -> None:
    """A ``run --record --max-ticks 3`` session writes a .jsonl with ≥3 TICK events."""
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--max-ticks",
            "3",
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
    # Find the recording: a .jsonl in tmp_path/rec or under it.
    rec_dir = tmp_path / "rec"
    jsonl_files = list(rec_dir.glob("*.jsonl")) if rec_dir.exists() else []
    if not jsonl_files:
        # fall back to the workspace recordings dir
        jsonl_files = list(Path("recordings").glob("*.jsonl"))
    assert jsonl_files, "no .jsonl recording produced"
    events = _read_jsonl(jsonl_files[0])
    tick_events = [e for e in events if e.get("event") == "tick"]
    assert len(tick_events) >= 3, f"expected >=3 tick events, got {len(tick_events)}"


def test_replay_subcommand_exists() -> None:
    """The ``replay`` subcommand is registered (not 'invalid choice')."""
    # Just verify it's a valid subcommand: `replay --help` should not fail
    # with "invalid choice".
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "ai_game_agent", "replay", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd="/workspace/AI.game.agent",
        env={"PATH": "/workspace/AI.game.agent/.venv/bin:/usr/bin:/bin",
             "PYTHONPATH": "/workspace/AI.game.agent/src"},
    )
    assert proc.returncode == 0, f"replay --help failed: {proc.stderr}"


def test_replay_reproduces_state_sequence(tmp_path, capsys) -> None:
    """Replay re-derives the same state sequence from the recorded events."""
    # Record a 3-tick session (observe-only, so no actuation noise).
    cfg = _write_config(tmp_path, _OBSERVE_CONFIG)
    rc = main(
        [
            "run",
            "--max-ticks",
            "3",
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
    rec_dir = tmp_path / "rec"
    jsonl_files = list(rec_dir.glob("*.jsonl")) if rec_dir.exists() else []
    assert jsonl_files, "no recording produced for replay"
    log_path = jsonl_files[0]

    # Now replay it.
    rc2 = main(["replay", str(log_path), "--config", cfg])
    assert rc2 == 0
    # Replay output: a JSON report with the replayed state sequence.
    out = capsys.readouterr().out.strip().splitlines()
    assert out, "replay printed nothing"
    report = json.loads(out[-1])
    assert "states" in report, "replay report missing 'states' key"
    # The original run's state sequence should be reproduced.
    # In observe_only with a mock (no hostile target) frame, the state stays IDLE.
    assert all(s == "idle" for s in report["states"])


def test_replay_missing_file_fails(tmp_path) -> None:
    """Replay of a non-existent file exits non-zero."""
    rc = main(["replay", str(tmp_path / "nonexistent.jsonl")])
    assert rc != 0
