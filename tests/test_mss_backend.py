import pytest

from ai_game_agent.capture import CaptureError, MssBackend, create_backend
from ai_game_agent.config import CaptureConfig


class FakeScreenShot:
    """Mimics the parts of an mss ScreenShot the backend uses."""

    def __init__(self, width: int, height: int, rgb: bytes):
        self.size = {"width": width, "height": height}
        self.rgb = rgb


class FakeMss:
    """Mimics mss.mss(): grab() returns a fixed screenshot; close() is tracked."""

    def __init__(self, screenshot: FakeScreenShot):
        self._screenshot = screenshot
        self.closed = False
        self.monitors = [
            {"left": 0, "top": 0, "width": 64, "height": 36},
            {"left": 0, "top": 0, "width": 10, "height": 5},
        ]

    def grab(self, monitor=None):
        return self._screenshot

    def close(self):
        self.closed = True


def make_backend(shot: FakeScreenShot) -> tuple[MssBackend, FakeMss]:
    fake = FakeMss(shot)
    backend = MssBackend(mss_factory=lambda: fake)
    return backend, fake


def test_mss_backend_grab_returns_rgb_frame():
    backend, fake = make_backend(FakeScreenShot(4, 2, bytes((7, 8, 9)) * 8))
    backend.open()
    try:
        frame = backend.grab()
    finally:
        backend.close()

    assert (frame.width, frame.height) == (4, 2)
    assert frame.pixels == bytes((7, 8, 9)) * 8
    assert frame.source == "mss"
    assert fake.closed is True


def test_mss_backend_grab_before_open_raises():
    backend, _ = make_backend(FakeScreenShot(1, 1, b"\x00\x00\x00"))
    with pytest.raises(CaptureError):
        backend.grab()


def test_mss_backend_open_close_idempotent():
    backend, fake = make_backend(FakeScreenShot(1, 1, b"\x00\x00\x00"))
    backend.open()
    backend.open()
    backend.close()
    backend.close()
    assert fake.closed is True


def test_create_backend_mss_returns_mss_backend():
    backend = create_backend(CaptureConfig(backend="mss"))
    assert isinstance(backend, MssBackend)
    assert backend.name == "mss"
