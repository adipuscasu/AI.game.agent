import pytest

from ai_game_agent.capture import CaptureError, MssBackend, create_backend
from ai_game_agent.config import CaptureConfig


class FakeSize:
    """Mimics mss >= 10's Size object (attribute access, not dict)."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height


class FakeScreenShot:
    """Mimics the parts of an mss ScreenShot the backend uses.

    ``size_as_dict=False`` mimics mss >= 10 (Size object); the dict form
    covers mss 9, which pyproject still allows (``mss>=9.0``).
    """

    def __init__(self, width: int, height: int, rgb: bytes, size_as_dict: bool = True):
        self.size = (
            {"width": width, "height": height} if size_as_dict else FakeSize(width, height)
        )
        self.rgb = rgb


class FakeMss:
    """Mimics mss.mss(): grab() returns the screenshot for the requested monitor.

    ``monitors`` follows real mss semantics: index 0 is the whole virtual
    screen, 1 is the primary, and 2+ are secondaries. ``grab`` records which
    monitor the backend asked for, so tests can assert the selection.
    """

    def __init__(self, screenshot: FakeScreenShot):
        self._screenshot = screenshot
        self.closed = False
        self.last_monitor: int | None = None
        self.monitors = [
            {"left": -10, "top": -5, "width": 64, "height": 36},   # 0: whole virtual screen
            {"left": 0, "top": 0, "width": 10, "height": 5},      # 1: primary
            {"left": 10, "top": 0, "width": 10, "height": 5},     # 2: secondary
        ]

    # mss >= 10 requires the monitor argument; this mirrors the real signature
    # and fails loudly if the backend omits it (regression: mss 10.x). The
    # backend passes the ``monitors[i]`` dict (mss's documented API), so
    # normalize it back to its index for assertions.
    def grab(self, monitor):
        if monitor is None:
            raise TypeError("MSS.grab() missing 1 required positional argument: 'monitor'")
        if isinstance(monitor, dict):
            monitor = self.monitors.index(monitor)
        self.last_monitor = monitor
        return self._screenshot

    def close(self):
        self.closed = True


def make_backend(shot: FakeScreenShot, monitor: int = 1) -> tuple[MssBackend, FakeMss]:
    fake = FakeMss(shot)
    backend = MssBackend(mss_factory=lambda: fake, monitor=monitor)
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


def test_mss_backend_grab_handles_mss10_size_object():
    """mss >= 10 returns shot.size as a Size object, not a dict (regression)."""
    backend, fake = make_backend(
        FakeScreenShot(4, 2, bytes((7, 8, 9)) * 8, size_as_dict=False)
    )
    backend.open()
    try:
        frame = backend.grab()
    finally:
        backend.close()

    assert (frame.width, frame.height) == (4, 2)
    assert fake.closed is True


def test_mss_backend_grab_before_open_raises():
    backend, _ = make_backend(FakeScreenShot(1, 1, b"\x00\x00\x00"))
    with pytest.raises(CaptureError):
        backend.grab()


def test_mss_backend_selects_monitor_from_config():
    """``grab()`` must ask mss for the configured monitor, not always the
    whole virtual screen (regression: the old ``monitors[0]`` captured all
    monitors with black bands; see docs/manual-testing.md)."""
    for monitor in (0, 1, 2):
        backend, fake = make_backend(FakeScreenShot(4, 2, bytes((7, 8, 9)) * 8), monitor=monitor)
        backend.open()
        try:
            backend.grab()
        finally:
            backend.close()
        assert fake.last_monitor == monitor


def test_mss_backend_out_of_range_monitor_raises_capture_error():
    """A configured monitor index that exceeds the live ``monitors`` list must
    surface as a ``CaptureError`` (the capture layer's single error type),
    not a raw ``IndexError`` (regression: review P2-2 — ``--monitor 999`` used
    to leak ``IndexError: list index out of range``)."""
    backend, fake = make_backend(FakeScreenShot(4, 2, bytes((7, 8, 9)) * 8), monitor=5)
    backend.open()
    try:
        with pytest.raises(CaptureError, match="out of range"):
            backend.grab()
    finally:
        backend.close()
    # The backend must fail before ever asking mss for a monitor.
    assert fake.last_monitor is None


class _ScreenWithoutMonitors:
    """A screen object that does not expose a ``monitors`` collection."""

    def grab(self, monitor):  # pragma: no cover - never reached in the test
        raise AssertionError("grab() must not be called when monitors is missing")

    def close(self):
        pass


def test_mss_backend_missing_monitors_list_raises_capture_error():
    """If the screen object does not expose ``monitors`` at all, that must
    also be a ``CaptureError`` (attribute access on ``.monitors`` would leak
    ``AttributeError``)."""
    backend = MssBackend(mss_factory=lambda: _ScreenWithoutMonitors(), monitor=1)
    backend.open()
    try:
        with pytest.raises(CaptureError, match="monitor list"):
            backend.grab()
    finally:
        backend.close()


def test_mss_backend_defaults_to_primary():
    """With no monitor configured, the factory must default to monitors[1]
    (primary), matching the documented "full primary monitor" behavior.

    The factory itself needs mss installed, so guard that; the default value
    under test (``CaptureConfig.monitor``) does not.
    """
    assert CaptureConfig(backend="mss").monitor == 1  # primary is the default
    pytest.importorskip("mss")
    backend = create_backend(CaptureConfig(backend="mss"))
    assert isinstance(backend, MssBackend)
    assert backend._monitor == 1


def test_mss_backend_open_close_idempotent():
    backend, fake = make_backend(FakeScreenShot(1, 1, b"\x00\x00\x00"))
    backend.open()
    backend.open()
    backend.close()
    backend.close()
    assert fake.closed is True


def test_create_backend_mss_returns_mss_backend():
    # This test exercises the real factory path, which ``import mss``-checks
    # availability (``create_backend`` -> ``_mss_available``). The other tests
    # inject a fake via ``mss_factory`` and run headless, so only this one
    # needs the dependency present. Skip cleanly on a bare venv.
    pytest.importorskip("mss")
    backend = create_backend(CaptureConfig(backend="mss"))
    assert isinstance(backend, MssBackend)
    assert backend.name == "mss"
