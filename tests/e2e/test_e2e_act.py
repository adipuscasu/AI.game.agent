"""End-to-end CLI contract tests for the Phase 3 ``act`` subcommand.

Phase 3 plan §7.2 verbatim contract (black-box: the real entry point in a
fresh interpreter, asserted on exit codes / stdout / stderr / JSON shape):

* ``act --backend mock --demo --mode autonomous`` → exit 0; stdout is a
  single JSON object with a ``results`` array and ``stopped == false``.
* ``act --backend mock --demo --mode observe_only`` → exit 0;
  ``results == []``; ``stopped == false`` — the OBSERVE_ONLY zero-input
  regression, proven end to end (unit *and* e2e per §7.3 DoD).
* ``act --backend doesnotexist`` → exit 1; stderr starts with ``error:``;
  stdout empty (the Phase 1/2 stream-routing contract).
* ``act --backend mock --config <tmp>`` with a tight
  ``safety.max_action_duration_ms`` bound → exit 0; the bounded demo action
  is rejected → ``stopped == true`` + ``stop_reason`` set (config-driven
  bounds wired end to end).

All tests use the ``mock`` backend so they run headless in CI (no desktop,
no pynput) — the §7.3 bare-venv DoD line is the same four commands minus a
separately-verified absence of the ``input`` extra.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a fresh interpreter; mirrors a user invocation.

    Same harness as ``test_e2e_cli.run_cli`` (a copy, kept self-contained: the
    ``tests`` tree has no ``__init__.py``, so cross-file imports would depend
    on pytest import-mode internals).
    """
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "ai_game_agent", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.e2e
def test_act_prints_actionlog_json() -> None:
    """AUTONOMOUS demo: exit 0, single JSON object, results array, not stopped."""
    result = run_cli("act", "--backend", "mock", "--demo", "--mode", "autonomous")
    assert result.returncode == 0, result.stderr
    log = json.loads(result.stdout)  # stdout must be *only* the JSON object
    assert isinstance(log, dict)
    assert isinstance(log.get("results"), list)
    assert len(log["results"]) > 0
    assert all(isinstance(r, dict) for r in log["results"])
    assert log.get("stopped") is False


@pytest.mark.e2e
def test_act_observe_only_emits_nothing() -> None:
    """The OBSERVE_ONLY zero-input regression, proven through the real CLI."""
    result = run_cli("act", "--backend", "mock", "--demo", "--mode", "observe_only")
    assert result.returncode == 0, result.stderr
    log = json.loads(result.stdout)
    assert log.get("results") == []  # nothing proposed-and-executed
    assert log.get("stopped") is False
    assert result.stderr == ""  # no error routing on a successful propose


@pytest.mark.e2e
def test_act_error_routing() -> None:
    """Unknown backend: exit 1, ``error:`` on stderr, stdout empty (§7.2)."""
    result = run_cli("act", "--backend", "doesnotexist")
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert result.stdout == ""


@pytest.mark.e2e
def test_act_config_file(tmp_path) -> None:
    """Config-driven safety bounds take effect end to end (§7.2).

    A 100 ms bound rejects the demo's 200 ms hold: the run still exits 0
    (a bounded rejection is a *successful, recorded* stop, not a CLI error)
    with ``stopped == true`` and a non-empty ``stop_reason``.
    """
    cfg = tmp_path / "act_bound.yaml"
    cfg.write_text(
        "input:\n"
        "  backend: mock\n"
        "  mode: autonomous\n"
        "safety:\n"
        "  max_action_duration_ms: 100\n",
        encoding="utf-8",
    )
    result = run_cli("act", "--backend", "mock", "--config", str(cfg))
    assert result.returncode == 0, result.stderr
    log = json.loads(result.stdout)
    assert log.get("stopped") is True
    assert log.get("stop_reason")
    assert any(r.get("ok") is False for r in log.get("results", []))


@pytest.mark.e2e
def test_act_do_batch_runs_both_specs() -> None:
    """Two ``--do`` flags are a batch (append-list), not a parsing crash.

    Regression guard: argparse ``action="append"`` yields a *list*; a handler
    written for a single string crashes on it. Both specs must execute and
    produce two ``ok`` results.
    """
    result = run_cli(
        "act", "--backend", "mock", "--mode", "autonomous",
        "--do", "key:a", "--do", "click:left:1",
    )
    assert result.returncode == 0, result.stderr
    log = json.loads(result.stdout)
    assert log.get("stopped") is False
    results = log.get("results", [])
    assert len(results) == 2
    assert all(r.get("ok") is True for r in results)
