from ai_game_agent.__main__ import main


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


def test_cli_no_command_returns_nonzero(capsys):
    assert main([]) == 2


def test_cli_capture_oserror_is_clean_error(capsys, monkeypatch):
    # N2: a genuine I/O failure (permission, full disk, read-only target)
    # surfaces as OSError from save_frame; main() must translate it into a
    # clean "error:" line on stderr with exit code 1, not a raw traceback.
    import ai_game_agent.__main__ as cli

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
