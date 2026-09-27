from datetime import datetime

from ai_game_agent.capture import Frame, save_frame


def test_save_frame_writes_png_roundtrip(tmp_path):
    from PIL import Image

    frame = Frame(3, 2, bytes((200, 100, 50)) * 6, datetime(2026, 1, 1, 12, 30, 45), "test")
    path = save_frame(frame, tmp_path / "a" / "b")

    assert path.exists()
    img = Image.open(path)
    assert img.size == (3, 2)
    assert img.convert("RGB").getpixel((0, 0)) == (200, 100, 50)


def test_save_frame_creates_missing_directory(tmp_path):
    out = tmp_path / "new" / "dir"
    path = save_frame(Frame(1, 1, b"\x01\x02\x03", datetime(2026, 1, 1), "test"), out)
    assert out.is_dir()
    assert path.exists()


def test_save_frame_collision_yields_distinct_files(tmp_path):
    ts = datetime(2026, 1, 1, 0, 0, 0)
    pixels = b"\x00" * 6
    p1 = save_frame(Frame(2, 1, pixels, ts, "test"), tmp_path)
    p2 = save_frame(Frame(2, 1, pixels, ts, "test"), tmp_path)
    assert p1 != p2
    assert p1.exists() and p2.exists()


def test_save_frame_filename_is_timestamp_sorted(tmp_path):
    early = save_frame(Frame(1, 1, b"\x00\x00\x00", datetime(2026, 1, 1, 0, 0, 0), "t"), tmp_path)
    late = save_frame(Frame(1, 1, b"\x00\x00\x00", datetime(2026, 1, 1, 1, 0, 0), "t"), tmp_path)
    # Timestamped names sort chronologically for replay.
    assert early.name < late.name
