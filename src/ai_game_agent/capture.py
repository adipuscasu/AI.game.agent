"""Screen capture abstraction (Phase 1).

The capture layer is the first perception building block: it turns the screen
into timestamped frames that downstream perception (CV / OCR / VLM) consumes.

Design rules:

* All capture code sits behind :class:`CaptureBackend`, so the backend can be
  swapped (mss, DXGI, replay/mock) without touching the rest of the system.
* No input, game logic, or AI lives in this module.
* The :class:`Capture` facade owns timing (fps), scaling, and optional
  recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ai_game_agent.config import CaptureConfig


class CaptureError(Exception):
    """Raised when screen capture fails (PERCEPTION_ERROR class)."""


@dataclass(frozen=True)
class Frame:
    """A single captured screen frame.

    ``pixels`` is a row-major RGB buffer (bytes, ``height * width * 3``),
    matching the layout produced by OpenCV's ``cv2.cvtColor(..., COLOR_BGR2RGB)``
    and by the default mss conversion.
    """

    width: int
    height: int
    pixels: bytes
    captured_at: datetime
    source: str

    def to_pil(self):
        """Return this frame as a PIL image (lazy import)."""
        from PIL import Image  # deferred: PIL is optional at import time

        return Image.frombytes("RGB", (self.width, self.height), self.pixels)

    def region(self, x: int, y: int, width: int, height: int) -> Frame:
        """Extract a sub-region of this frame as a new frame."""
        if not (0 <= x and 0 <= y and x + width <= self.width and y + height <= self.height):
            raise CaptureError(
                f"region ({x}, {y}, {width}, {height}) exceeds frame "
                f"({self.width}x{self.height})"
            )
        out = bytearray(width * height * 3)
        for row in range(height):
            src = ((y + row) * self.width + x) * 3
            dst = row * width * 3
            out[dst : dst + width * 3] = self.pixels[src : src + width * 3]
        return Frame(width, height, bytes(out), self.captured_at, self.source)


class CaptureBackend:
    """Interface for frame acquisition backends.

    Implementations must be safe to construct in headless environments only if
    they are the :class:`MockBackend`; real backends require a display.
    """

    name: str = "base"

    def open(self) -> None:
        """Prepare the backend (allocate resources). Idempotent."""
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources. Idempotent."""
        raise NotImplementedError

    def grab(self) -> Frame:
        """Capture one full-screen (or preconfigured region) frame."""
        raise NotImplementedError


class MockBackend(CaptureBackend):
    """Deterministic backend returning a fixed synthetic frame.

    Useful for unit tests, CI, and replay mode (no display required).
    """

    name = "mock"

    def __init__(
        self, width: int = 128, height: int = 72, rgb: tuple[int, int, int] = (10, 20, 30)
    ) -> None:
        self._width = width
        self._height = height
        self._pixel = bytes(rgb)  # one RGB triple per pixel
        self._frame_index = 0
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def grab(self) -> Frame:
        from datetime import datetime

        frame = Frame(
            width=self._width,
            height=self._height,
            pixels=self._pixel * (self._width * self._height),
            captured_at=datetime.now(UTC),
            source=self.name,
        )
        self._frame_index += 1
        return frame


def create_backend(config: CaptureConfig):
    """Factory: return the capture backend named in config.

    Raises:
        CaptureError: if the backend name is unknown or its dependency is
            missing from the environment.
    """
    name = (config.backend or "").strip().lower()
    if name == "mock":
        return MockBackend()
    if name == "mss":
        try:
            import mss  # noqa: F401  (import check only)
        except ImportError as exc:
            raise CaptureError(
                "the 'mss' backend is not installed; "
                "install it with 'pip install mss' or set capture.backend to 'mock'"
            ) from exc
        raise CaptureError(
            "the 'mss' backend is not yet implemented in this build; "
            "use 'mock' for headless/CI use"
        )
    raise CaptureError(f"unknown capture backend: {config.backend!r}")


class Capture:
    """Capture facade: fps pacing, region extraction, and optional recording.

    Args:
        config: Capture configuration.
        backend: Backend instance (dependency injection; defaults to the
            backend named in config).
        clock: Monotonic clock callable for timing. Injectable for tests.
    """

    def __init__(self, config: CaptureConfig, backend: CaptureBackend | None = None, clock=None):
        self.config = config
        self.backend = backend if backend is not None else create_backend(config)
        self._clock = clock if clock is not None else _default_clock
        self._opened = False

    def start(self) -> None:
        """Open the backend and begin capturing."""
        if not self._opened:
            self.backend.open()
            self._opened = True

    def stop(self) -> None:
        """Stop capturing and release backend resources."""
        if self._opened:
            self.backend.close()
            self._opened = False

    def __enter__(self) -> Capture:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def grab(self) -> Frame:
        """Capture one frame, applying region and scaling from config.

        Raises:
            CaptureError: if :meth:`start` was not called or the backend fails.
        """
        if not self._opened:
            raise CaptureError("capture not started; call start() first")
        try:
            frame = self.backend.grab()
        except CaptureError:
            raise
        except Exception as exc:  # surface backend failures as a single error class
            raise CaptureError(f"capture failed: {exc}") from exc
        if self.config.region is not None:
            frame = frame.region(
                self.config.region.x,
                self.config.region.y,
                self.config.region.width,
                self.config.region.height,
            )
        return frame

    def wait_next_frame(self) -> float:
        """Sleep until the next frame is due per config.fps. Returns seconds slept."""
        if self.config.fps <= 0:
            return 0.0
        return self._clock(1.0 / self.config.fps)


def _default_clock(seconds: float) -> float:
    import time

    time.sleep(seconds)
    return seconds


def save_frame(frame: Frame, directory: Path) -> Path:
    """Persist a frame as PNG. Returns the written path.

    The directory is created if missing. Filenames embed the timestamp so
    order is stable for replay.
    """
    raise NotImplementedError("to be implemented in the green step")
