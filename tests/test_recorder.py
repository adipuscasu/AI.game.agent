from datetime import datetime

from ai_game_agent.capture import Frame, Recorder


def frame_at(second: int) -> Frame:
    return Frame(2, 1, b"\x00" * 6, datetime(2026, 1, 1, 0, 0, second), "test")


def test_recorder_disabled_writes_nothing(tmp_path):
    rec = Recorder(tmp_path, enabled=False)
    assert rec.record(frame_at(0)) is None
    assert list(tmp_path.glob("*.png")) == []


def test_recorder_enabled_writes_png(tmp_path):
    rec = Recorder(tmp_path, enabled=True)
    path = rec.record(frame_at(0))
    assert path is not None
    assert path.suffix == ".png"
    assert path.exists()


def test_recorder_rotation_keeps_newest(tmp_path):
    rec = Recorder(tmp_path, max_files=2, enabled=True)
    for second in range(4):
        rec.record(frame_at(second))

    files = sorted(tmp_path.glob("*.png"))
    assert len(files) == 2
    stems = {f.stem for f in files}
    # Oldest two (seconds 0 and 1) are dropped, newest two (2 and 3) remain.
    assert "frame_20260101T000000000000" not in stems
    assert "frame_20260101T000003000000" in stems


def test_recorder_unlimited_when_max_zero(tmp_path):
    rec = Recorder(tmp_path, max_files=0, enabled=True)
    for second in range(5):
        rec.record(frame_at(second))
    assert len(list(tmp_path.glob("*.png"))) == 5


def test_recorder_rotation_prefers_oldest_on_suffix_collision(tmp_path):
    # When same-timestamp collision suffixes grow past 9, lexicographic order
    # ("...-10" < "...-2") would delete the wrong files. Rotation must drop
    # the *numerically* oldest suffix first, keeping the newest.
    base = "frame_20260101T000000000000"
    for suffix in ("1", "2", "10"):
        (tmp_path / f"{base}-{suffix}.png").write_bytes(b"\x00")

    rec = Recorder(tmp_path, max_files=2, enabled=True)
    rec._rotate()

    stems = {f.stem for f in tmp_path.glob("*.png")}
    # Suffix "1" (the numeric oldest) is dropped, not "-10".
    assert stems == {f"{base}-2", f"{base}-10"}
