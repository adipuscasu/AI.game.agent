import logging

import ai_game_agent.__main__ as cli
from ai_game_agent.__main__ import main
from ai_game_agent.config import load_config


def test_cli_capture_saves_screenshot(tmp_path):
    out = tmp_path / "shots"
    rc = main(["capture", "--backend", "mock", "--out", str(out)])
    assert rc == 0
    pngs = list(out.glob("*.png"))
    assert len(pngs) == 1
    from PIL import Image

    assert Image.open(pngs[0]).size == (128, 72)  # MockBackend default


def test_cli_observe_runs_and_reports_fps(tmp_path):
    # fps=0 disables real pacing (no sleeps) so the loop runs headlessly.
    rc = main(["observe", "--backend", "mock", "--frames", "5", "--fps", "0"])
    assert rc == 0


def test_cli_observe_records_when_enabled(tmp_path):
    out = tmp_path / "rec"
    cmd = [
        "observe",
        "--backend",
        "mock",
        "--frames",
        "3",
        "--fps",
        "0",
        "--record",
        "--out",
        str(out),
    ]
    rc = main(cmd)
    assert rc == 0
    assert len(list(out.glob("*.png"))) == 3


def test_cli_capture_with_region(tmp_path):
    out = tmp_path / "shots"
    rc = main(["capture", "--backend", "mock", "--region", "10,20,32,16", "--out", str(out)])
    assert rc == 0
    from PIL import Image

    assert Image.open(next(out.glob("*.png"))).size == (32, 16)


def test_cli_capture_with_invalid_region_returns_nonzero(tmp_path):
    out = tmp_path / "shots"
    rc = main(["capture", "--backend", "mock", "--region", "10,20,32", "--out", str(out)])
    assert rc != 0


def test_cli_capture_with_scale(tmp_path):
    # Mock backend defaults to 128x72; scale 0.5 -> 64x36.
    out = tmp_path / "shots"
    rc = main(["capture", "--backend", "mock", "--scale", "0.5", "--out", str(out)])
    assert rc == 0
    from PIL import Image

    assert Image.open(next(out.glob("*.png"))).size == (64, 36)


def test_cli_capture_with_invalid_scale_returns_nonzero(tmp_path):
    out = tmp_path / "shots"
    rc = main(["capture", "--backend", "mock", "--scale", "0", "--out", str(out)])
    assert rc != 0


def test_cli_capture_with_monitor_flag(tmp_path):
    # --monitor only affects the mss backend's monitor selection; with the
    # mock backend it must be accepted and produce a normal capture.
    out = tmp_path / "shots"
    rc = main(["capture", "--backend", "mock", "--monitor", "1", "--out", str(out)])
    assert rc == 0
    assert len(list(out.glob("*.png"))) == 1


def test_cli_capture_with_invalid_monitor_returns_nonzero():
    rc = main(["capture", "--backend", "mock", "--monitor", "-1", "--out", "ignored"])
    assert rc != 0


def test_cli_observe_accepts_monitor_flag(tmp_path):
    out = tmp_path / "rec"
    rc = main([
        "observe", "--backend", "mock", "--frames", "3", "--fps", "0",
        "--monitor", "2", "--out", str(out),
    ])
    assert rc == 0


def test_cli_no_command_returns_nonzero(capsys):
    assert main([]) == 2


def test_cli_capture_oserror_is_clean_error(capsys, monkeypatch):
    # N2: a genuine I/O failure (permission, full disk, read-only target)
    # surfaces as OSError from save_frame; main() must translate it into a
    # clean "error:" line on stderr with exit code 1, not a raw traceback.

    def _boom(frame, directory):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(cli, "save_frame", _boom)
    rc = main(["capture", "--backend", "mock", "--out", "ignored"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err.startswith("error: ")
    assert "Permission denied" in captured.err
    assert "Traceback" not in captured.err


def test_cli_capture_writes_log_file(tmp_path, monkeypatch, capsys):
    # Logging must go to the configured file (default: logs/agent.log), not
    # to stdout or stderr, so the CLI stream/exit-code contract is preserved.
    monkeypatch.chdir(tmp_path)
    rc = main(["capture", "--backend", "mock", "--out", str(tmp_path / "shots")])
    assert rc == 0
    captured = capsys.readouterr()
    assert "capture: saved frame" not in captured.out
    assert "capture: saved frame" not in captured.err
    log_file = tmp_path / "logs" / "agent.log"
    assert log_file.is_file()
    assert "capture: saved frame" in log_file.read_text(encoding="utf-8")


def test_cli_capture_logs_to_stderr_not_stdout(tmp_path, capsys):
    # Contract: stdout carries only program output (the ``saved: ...`` line);
    # log lines must go to the log file, never stdout.
    out = tmp_path / "shots"
    rc = main(["capture", "--backend", "mock", "--out", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "saved: " in captured.out
    assert "capture: saved frame" not in captured.out


def test_cli_observe_logs_to_stderr_not_stdout(capsys):
    # Contract: stdout carries only ``frames=N fps=...``; the observe log
    # line goes to the log file, never stdout.
    rc = main(["observe", "--backend", "mock", "--frames", "3", "--fps", "0"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("frames=3 fps=")
    assert "observe: frames=" not in captured.out


# --- config-driven capture / logging (the config is the source of defaults) ---
#
# Precedence rule (all subcommands): an explicitly-passed CLI flag wins;
# otherwise the value in the loaded config (``--config``, default
# ``config/default.yaml``) is used; otherwise the code default.  Before this
# was honored, ``--backend``/``--region``/``--scale``/``--fps`` were forced to
# the CLI defaults (mss / whole-monitor / 1.0 / no-pacing) no matter what the
# config said, and the log file path and level were hard-coded to
# ``logs/agent.log`` at INFO.

def test_cli_capture_respects_config_backend_region_scale(tmp_path, monkeypatch):
    """capture honors backend/region/scale from config when no flag is passed."""
    cfg = tmp_path / "cap.yaml"
    cfg.write_text(
        "capture:\n"
        "  backend: mock\n"
        "  region: {x: 10, y: 20, width: 32, height: 16}\n"
        "  scale: 0.5\n"
        "  fps: 0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "load_config", lambda path=None: load_config(str(cfg)))
    out = tmp_path / "shots"
    rc = main(["capture", "--config", str(cfg), "--out", str(out)])  # no --backend/--region/--scale
    assert rc == 0
    from PIL import Image

    # mock backend selected by config (not forced to the default mss);
    # region+scale applied: 32x16 -> 16x8.  A pre-fix run would capture the
    # whole 128x72 mock frame (scale default 1.0) or fail on the mss backend.
    assert Image.open(next(out.glob("*.png"))).size == (16, 8)


def test_cli_capture_fps_from_config_is_a_ceiling(tmp_path, monkeypatch):
    """observe paces at config.fps when --fps is not passed (ceiling, not a guarantee).

    Deterministic: inject a recording clock in place of the real ``time.sleep``
    and assert the CLI paced at exactly ``1 / config.fps`` seconds per frame.
    A pre-fix run forces ``args.fps`` to the CLI default 0 (no pacing), so the
    clock is never called.
    """
    import ai_game_agent.capture as capture_mod

    cfg = tmp_path / "cap.yaml"
    cfg.write_text("capture:\n  backend: mock\n  fps: 1000\n", encoding="utf-8")
    monkeypatch.setattr(cli, "load_config", lambda path=None: load_config(str(cfg)))

    calls: list[float] = []

    def _recording_clock(seconds: float) -> float:
        calls.append(seconds)
        return seconds

    monkeypatch.setattr(capture_mod, "_default_clock", _recording_clock)
    out = tmp_path / "rec"
    rc = main([
        "observe", "--config", str(cfg), "--backend", "mock",
        "--frames", "4", "--out", str(out),
    ])
    assert rc == 0
    # 4 frames -> wait_next_frame() called 4x at 1/1000 s each (config.fps=1000).
    assert calls == [1 / 1000] * 4, (
        f"expected pacing at config.fps (1/1000 s) per frame, got clock calls {calls!r}"
    )


def test_cli_logging_respects_config_directory_and_level(tmp_path, monkeypatch):
    """The log file goes to ``logging.directory`` and DEBUG is enabled per config."""

    cfg = tmp_path / "cap.yaml"
    cfg.write_text(
        "capture:\n  backend: mock\n"
        "logging:\n  level: DEBUG\n  directory: logs_custom\n",
        encoding="utf-8",
    )
    real_load = cli.load_config
    monkeypatch.setattr(cli, "load_config", lambda path=None: real_load(str(cfg)))
    monkeypatch.chdir(tmp_path)
    rc = main(["capture", "--backend", "mock", "--out", str(tmp_path / "shots")])
    assert rc == 0

    # Config directory honored (pre-fix: hard-coded to ./logs/agent.log).
    log_file = tmp_path / "logs_custom" / "agent.log"
    assert log_file.is_file(), "log file must be written to logging.directory from config"
    # Config level honored: DEBUG <= INFO so the INFO-level CLI line is present.
    assert "capture: saved frame" in log_file.read_text(encoding="utf-8")
    # And the level was actually applied to the logger (a DEBUG filter would let
    # DEBUG through, an ERROR filter would not — here INFO must pass).
    assert logging.getLogger("ai_game_agent").getEffectiveLevel() <= logging.DEBUG


# --- bare-environment contract ---------------------------------------------
# §7.3 done-criterion: "analyze prints valid JSON in a bare dev environment
# (no vision/ocr extras installed)".  The CLI entry point must therefore
# import cleanly when cv2/numpy are absent.  color_blobs.py imports cv2 at
# module top and raises PerceptionError if it is missing -- by design it is
# "imported only when the subsystem is selected", so __main__.py must import
# ColorBlobsDetector lazily (inside the `if object_colors` branch), not at the
# module top.

def test_cli_imports_cleanly_without_vision_extra():
    """`import ai_game_agent.__main__` must succeed when cv2/numpy are blocked.

    This is the §7.3 bare-env contract.  The current unconditional top-level
    import of color_blobs in __main__.py violates it: color_blobs raises
    PerceptionError at import time when cv2 is absent, so the whole CLI
    (every subcommand) fails to load in a bare venv.
    """
    import subprocess
    import sys

    script = (
        "import sys\n"
        "class _Blocker:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname in ('cv2', 'numpy') or fullname.startswith(('cv2.', 'numpy.')):\n"
        "            raise ImportError('blocked: ' + fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "import ai_game_agent.__main__\n"
        "print('imported-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "importing ai_game_agent.__main__ must succeed without cv2/numpy "
        f"(§7.3 bare-env); got rc={result.returncode}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "imported-ok" in result.stdout
