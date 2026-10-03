"""End-to-end CLI contract tests (Phase 1).

Black-box tests: each test spawns the real CLI as a separate Python process
(``python -m ai_game_agent``) and asserts on exit codes, stdout/stderr
routing, and files on disk. This validates the observable contract a user or
CI job experiences (entry-point import path, argument parsing, stream
routing, PNG validity), which in-process tests cannot.

All tests use the deterministic ``mock`` backend and ``--fps 0`` (no
pacing), so they run headless in CI without a desktop session.

Run with::

    uv run pytest -m e2e
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a fresh interpreter; mirrors a user invocation."""
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
def test_capture_produces_png(tmp_path: Path) -> None:
    out = tmp_path / "shots"
    result = run_cli("capture", "--backend", "mock", "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert re.search(r"saved: .*\.png", result.stdout)
    pngs = list(out.glob("*.png"))
    assert len(pngs) == 1
    from PIL import Image

    assert Image.open(pngs[0]).size == (128, 72)  # MockBackend default


@pytest.mark.e2e
def test_capture_region_and_scale_pipeline(tmp_path: Path) -> None:
    out = tmp_path / "shots"
    result = run_cli(
        "capture",
        "--backend",
        "mock",
        "--region",
        "10,20,32,16",
        "--scale",
        "0.5",
        "--out",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    from PIL import Image

    pngs = list(out.glob("*.png"))
    assert len(pngs) == 1
    # 128x72 mock frame -> 32x16 region -> 0.5 scale == 16x8
    assert Image.open(pngs[0]).size == (16, 8)


@pytest.mark.e2e
def test_observe_reports_fps(tmp_path: Path) -> None:
    result = run_cli("observe", "--backend", "mock", "--frames", "5", "--fps", "0")
    assert result.returncode == 0, result.stderr
    assert re.search(r"^frames=5 fps=[0-9.]+", result.stdout, re.MULTILINE)


@pytest.mark.e2e
def test_observe_record_rotation(tmp_path: Path) -> None:
    out = tmp_path / "rec"
    result = run_cli(
        "observe",
        "--backend",
        "mock",
        "--frames",
        "10",
        "--fps",
        "0",
        "--record",
        "--record-max-files",
        "3",
        "--out",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    assert len(list(out.glob("*.png"))) == 3  # oldest rotated out at the boundary


@pytest.mark.e2e
def test_capture_logs_to_stderr_not_stdout(tmp_path: Path) -> None:
    """Contract: stdout carries only ``saved: ...``; log lines never reach stdout."""
    out = tmp_path / "shots"
    result = run_cli("capture", "--backend", "mock", "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert "capture: saved frame" not in result.stdout


@pytest.mark.e2e
def test_observe_logs_to_stderr_not_stdout() -> None:
    """Contract: stdout is only ``frames=N fps=...``; log lines never reach stdout."""
    result = run_cli("observe", "--backend", "mock", "--frames", "3", "--fps", "0")
    assert result.returncode == 0, result.stderr
    assert "observe: frames=" not in result.stdout


@pytest.mark.e2e
def test_invalid_region_reports_error(tmp_path: Path) -> None:
    out = tmp_path / "shots"
    result = run_cli("capture", "--backend", "mock", "--region", "1,2,3", "--out", str(out))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert result.stdout == ""
    assert not out.exists()


@pytest.mark.e2e
def test_invalid_scale_reports_error(tmp_path: Path) -> None:
    out = tmp_path / "shots"
    result = run_cli("capture", "--backend", "mock", "--scale", "0", "--out", str(out))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert result.stdout == ""
    assert not out.exists()


@pytest.mark.e2e
def test_no_command_prints_help() -> None:
    result = run_cli()
    assert result.returncode == 2
    assert "usage:" in result.stderr
    # Help goes to stderr (argparse convention); nothing should go to stdout.
    assert result.stdout == ""


@pytest.mark.e2e
@pytest.mark.skipif(shutil.which("ai-game-agent") is None, reason="console script not on PATH")
def test_console_script_smoke(tmp_path: Path) -> None:
    """Validates the installed ``[project.scripts]`` entry point end to end."""
    script = shutil.which("ai-game-agent")
    assert script is not None
    out = tmp_path / "shots"
    try:
        result = subprocess.run(
            [script, "capture", "--backend", "mock", "--out", str(out)],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as exc:
        # The script is on PATH but the environment refuses to execute it
        # (e.g. Windows AppLocker / application-control policy). Not a product
        # bug; skip rather than fail.
        pytest.skip(f"console script is not executable in this environment: {exc}")
    assert result.returncode == 0, result.stderr
    assert len(list(out.glob("*.png"))) == 1


@pytest.mark.e2e
def test_region_exceeds_frame_reports_error(tmp_path: Path) -> None:
    out = tmp_path / "shots"
    result = run_cli(
        "capture", "--backend", "mock", "--region", "0,0,9999,9999", "--out", str(out)
    )
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert result.stdout == ""
    assert not out.exists()


@pytest.mark.e2e
def test_unknown_backend_reports_error(tmp_path: Path) -> None:
    out = tmp_path / "shots"
    result = run_cli("capture", "--backend", "doesnotexist", "--out", str(out))
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert result.stdout == ""
    assert not out.exists()


# ---------------------------------------------------------------------------
# Phase 2 – ``analyze`` subcommand
# ---------------------------------------------------------------------------


import json


@pytest.mark.e2e
def test_analyze_prints_json(tmp_path: Path) -> None:
    """analyze prints a single-line JSON observation on stdout."""
    result = run_cli("analyze", "--backend", "mock")
    assert result.returncode == 0, result.stderr
    obs = json.loads(result.stdout)
    # Core shape: observation is a JSON object with the expected keys.
    assert isinstance(obs, dict)
    assert isinstance(obs.get("templates"), list)
    assert isinstance(obs.get("text_regions"), list)
    assert isinstance(obs.get("objects"), list)
    # MockBackend default frame dimensions.
    assert obs.get("frame_width") == 128
    assert obs.get("frame_height") == 72
    # stderr must be empty on success.
    assert result.stderr == ""


@pytest.mark.e2e
def test_analyze_pretty_prints_indented_json() -> None:
    """--pretty produces indented (multi-line) JSON."""
    result = run_cli("analyze", "--backend", "mock", "--pretty")
    assert result.returncode == 0, result.stderr
    # Indented JSON should span more than one line.
    assert len(result.stdout.strip().splitlines()) > 1
    obs = json.loads(result.stdout)  # still valid JSON
    assert obs.get("frame_width") == 128


@pytest.mark.e2e
def test_analyze_disabled_subsystems() -> None:
    """--no-ocr --no-objects suppress text and object regions."""
    result = run_cli("analyze", "--backend", "mock", "--no-ocr", "--no-objects")
    assert result.returncode == 0, result.stderr
    obs = json.loads(result.stdout)
    assert obs.get("text_regions") == []
    assert obs.get("objects") == []


@pytest.mark.e2e
def test_analyze_config_file(tmp_path: Path) -> None:
    """analyze --config loads a config file and respects its perception block."""
    cfg = tmp_path / "analyze_config.yaml"
    cfg.write_text(
        """\
perception:
  enabled: true
  templates: {}
  ui_zones: []
  ocr:
    enabled: false
  objects:
    enabled: false
"""
    )
    result = run_cli("analyze", "--backend", "mock", "--config", str(cfg))
    assert result.returncode == 0, result.stderr
    obs = json.loads(result.stdout)
    # OCR and objects disabled in config → empty lists.
    assert obs.get("text_regions") == []
    assert obs.get("objects") == []


@pytest.mark.e2e
def test_analyze_bad_config_reports_error(tmp_path: Path) -> None:
    """analyze --config with a malformed file ⇒ exit 1, error: on stderr, empty stdout."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("perception: [not, a, dict]\n")
    result = run_cli("analyze", "--backend", "mock", "--config", str(bad))
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert result.stdout == ""


@pytest.mark.e2e
def test_analyze_unknown_backend_reports_error() -> None:
    result = run_cli("analyze", "--backend", "doesnotexist")
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert result.stdout == ""
