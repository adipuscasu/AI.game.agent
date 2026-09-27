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
