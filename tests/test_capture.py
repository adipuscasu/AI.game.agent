import pytest

from ai_game_agent.capture import (
    Capture,
    CaptureBackend,
    CaptureError,
    Frame,
    MockBackend,
    create_backend,
)
from ai_game_agent.config import CaptureConfig, Region


class RecordingClock:
    """Injectable clock: records sleep durations instead of sleeping."""

    def __init__(self):
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> float:
        self.slept.append(seconds)
        return seconds


def make_frame(width: int = 4, height: int = 2, color: tuple[int, int, int] = (1, 2, 3)) -> Frame:
    from datetime import datetime

    return Frame(width, height, bytes(color) * (width * height), datetime(2026, 1, 1), "test")


def test_mock_backend_returns_configured_frame():
    backend = MockBackend(width=32, height=16, rgb=(255, 0, 128))
    backend.open()
    try:
        frame = backend.grab()
    finally:
        backend.close()

    assert frame.width == 32
    assert frame.height == 16
    assert len(frame.pixels) == 32 * 16 * 3
    assert frame.pixels[:3] == bytes((255, 0, 128))
    assert frame.source == "mock"
    assert backend.opened is False


def test_mock_backend_grab_before_open_still_works():
    # open() is idempotent and non-fatal for the mock backend.
    frame = MockBackend(width=2, height=2).grab()
    assert frame.width == 2


def test_create_backend_mock():
    backend = create_backend(CaptureConfig(backend="mock"))
    assert isinstance(backend, MockBackend)
    assert backend.name == "mock"


def test_create_backend_unknown_raises():
    with pytest.raises(CaptureError):
        create_backend(CaptureConfig(backend="does-not-exist"))


def test_capture_requires_start():
    cap = Capture(CaptureConfig(backend="mock"), backend=MockBackend())
    with pytest.raises(CaptureError):
        cap.grab()


def test_capture_context_manager_starts_and_stops():
    backend = MockBackend(width=4, height=4)
    with Capture(CaptureConfig(backend="mock"), backend=backend) as cap:
        assert backend.opened is True
        frame = cap.grab()
    assert backend.opened is False
    assert frame.width == 4


def test_capture_applies_region():
    full = make_frame(width=10, height=5, color=(9, 9, 9))
    backend = StaticBackend(full)
    cfg = CaptureConfig(backend="mock", region=Region(x=2, y=1, width=3, height=2))
    with Capture(cfg, backend=backend) as cap:
        frame = cap.grab()
    assert (frame.width, frame.height) == (3, 2)
    # Every pixel of the extracted region equals the source color.
    assert frame.pixels == bytes((9, 9, 9)) * (3 * 2)


def test_capture_wait_next_frame_uses_config_fps():
    clock = RecordingClock()
    with Capture(CaptureConfig(backend="mock", fps=30), backend=MockBackend(), clock=clock) as cap:
        cap.wait_next_frame()
    assert clock.slept == [pytest.approx(1 / 30)]


def test_frame_region_out_of_bounds_raises():
    frame = make_frame(width=4, height=2)
    with pytest.raises(CaptureError):
        frame.region(3, 0, 3, 2)  # x + width > frame width


def _gradient_frame(width: int, height: int) -> Frame:
    """A 4x2 frame where every pixel is a unique, easily identifiable color.

    Pixel (col, row) is encoded as RGB (col, row, 0) so any downsampled
    output value can be traced back to the exact source pixel it came from.
    """
    from datetime import datetime

    pixels = bytearray()
    for row in range(height):
        for col in range(width):
            pixels += bytes((col % 256, row % 256, 0))
    return Frame(width, height, bytes(pixels), datetime(2026, 1, 1), "test")


def test_frame_resize_half_picks_expected_nearest_neighbors():
    # 4x2 -> scale 0.5 -> 2x1. Nearest-neighbor sampling maps output column i
    # to source column int(i / 0.5): 0 -> 0, 1 -> 2.
    frame = _gradient_frame(4, 2)
    resized = frame.resize(0.5)
    assert (resized.width, resized.height) == (2, 1)
    assert resized.pixels[:3] == bytes((0, 0, 0))  # source pixel (col 0, row 0)
    assert resized.pixels[3:6] == bytes((2, 0, 0))  # source pixel (col 2, row 0)


def test_frame_resize_one_is_identity():
    frame = _gradient_frame(4, 2)
    resized = frame.resize(1.0)
    assert (resized.width, resized.height) == (4, 2)
    assert resized.pixels == frame.pixels


def test_frame_resize_upscale_clamps_source_pixel():
    # 2x2 -> scale 2.0 -> 4x4. Output column i -> source column int(i / 2.0).
    frame = _gradient_frame(2, 2)
    resized = frame.resize(2.0)
    assert (resized.width, resized.height) == (4, 4)
    # All four pixels in output row 0 must sample source row 0.
    for i in range(4):
        assert resized.pixels[i * 3 : i * 3 + 2] == frame.pixels[(i // 2) * 3 : (i // 2) * 3 + 2]


def test_frame_resize_min_size_and_invalid_scale():
    # Even a tiny downscale must produce at least a 1x1 frame.
    frame = _gradient_frame(2, 2)
    assert (frame.resize(0.1).width, frame.resize(0.1).height) == (1, 1)
    with pytest.raises(CaptureError):
        frame.resize(0.0)
    with pytest.raises(CaptureError):
        frame.resize(-1.0)


def test_capture_applies_scale_after_region():
    full = _gradient_frame(4, 2)
    backend = StaticBackend(full)
    cfg = CaptureConfig(backend="mock", region=Region(x=0, y=0, width=4, height=2), scale=0.5)
    with Capture(cfg, backend=backend) as cap:
        frame = cap.grab()
    assert (frame.width, frame.height) == (2, 1)


def test_capture_scale_one_is_noop():
    full = _gradient_frame(4, 2)
    backend = StaticBackend(full)
    cfg = CaptureConfig(backend="mock", scale=1.0)
    with Capture(cfg, backend=backend) as cap:
        frame = cap.grab()
    assert (frame.width, frame.height) == (4, 2)
    assert frame.pixels == full.pixels


class StaticBackend(CaptureBackend):
    """Returns a fixed frame on every grab; records open/close state."""

    name = "static"

    def __init__(self, frame: Frame):
        self._frame = frame
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def grab(self) -> Frame:
        return self._frame


def test_capture_wraps_backend_errors():
    class ExplodingBackend(CaptureBackend):
        name = "exploding"

        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

        def grab(self) -> Frame:
            raise RuntimeError("display detached")

    with Capture(CaptureConfig(backend="exploding"), backend=ExplodingBackend()) as cap:
        with pytest.raises(CaptureError):
            cap.grab()
