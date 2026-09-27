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

from collections.abc import Callable
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
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def grab(self) -> Frame:
        return Frame(
            width=self._width,
            height=self._height,
            pixels=self._pixel * (self._width * self._height),
            captured_at=datetime.now(UTC),
            source=self.name,
        )

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
        if not _mss_available():
            raise CaptureError(
                "the 'mss' backend is not installed; "
                "install it with 'pip install mss' or set capture.backend to 'mock'"
            )
        return MssBackend()
    raise CaptureError(f"unknown capture backend: {config.backend!r}")


def _mss_available() -> bool:
    try:
        import mss  # noqa: F401

        return True
    except ImportError:
        return False


class MssBackend(CaptureBackend):
    """mss-backed capture (Phase 1 real screen capture).

    Requires a display. Not used by the test suite; tests inject a fake screen
    via ``mss_factory`` to keep the pipeline deterministic and headless.
    """

    name = "mss"

    def __init__(self, mss_factory: Callable[[], object] | None = None) -> None:
        self._factory = mss_factory
        self._screen: object | None = None

    def open(self) -> None:
        if self._screen is None:
            if self._factory is not None:
                factory = self._factory
            else:
                import mss

                factory = mss.mss
            self._screen = factory()

    def close(self) -> None:
        if self._screen is not None:
            self._screen.close()
            self._screen = None

    def grab(self) -> Frame:
        if self._screen is None:
            raise CaptureError("mss backend not opened; call open() first")
        shot = self._screen.grab()
        width = shot.size["width"]
        height = shot.size["height"]
        raw = bytes(shot.rgb)
        if len(raw) != width * height * 3:
            raise CaptureError(
                f"unexpected mss frame size: {len(raw)} bytes for {width}x{height}"
            )
        return Frame(width, height, raw, datetime.now(UTC), self.name)


class Capture:
    """Capture facade: fps pacing, region extraction, and optional recording.

    Args:
        config: Capture configuration.
        backend: Backend instance (dependency injection; defaults to the
            backend named in config).
        clock: Monotonic clock callable for timing. Injectable for tests.
    """

    def __init__(
        self,
        config: CaptureConfig,
        backend: CaptureBackend | None = None,
        clock: Callable[[float], float] | None = None,
    ) -> None:
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
    order is stable for replay. A collision-safe suffix is appended if a file
    with the same timestamp already exists.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"frame_{frame.captured_at:%Y%m%dT%H%M%S%f}"
    candidate = directory / f"{base}.png"
    suffix = 1
    while candidate.exists():
        candidate = directory / f"{base}-{suffix}.png"
        suffix += 1
    frame.to_pil().save(candidate, format="PNG")
    return candidate


class FpsMeter:
    """Measures frames-per-second over a sliding window of recorded timestamps.

    FPS is defined as ``frames / elapsed`` using the first and last recorded
    samples, so a single frame (zero elapsed) reports ``0.0``. Non-monotonic
    timestamps are tolerated; the first and last samples bound the window.
    """

    def __init__(self) -> None:
        self._first: float | None = None
        self._last: float | None = None
        self._count = 0

    def record(self, timestamp: float) -> None:
        if self._first is None:
            self._first = timestamp
        self._last = timestamp
        self._count += 1

    @property
    def frame_count(self) -> int:
        return self._count

    @property
    def fps(self) -> float:
        if self._count < 2:
            return 0.0
        elapsed = self._last - self._first
        if elapsed <= 0:
            return 0.0
        return (self._count - 1) / elapsed

    def reset(self) -> None:
        self._first = None
        self._last = None
        self._count = 0


class Recorder:
    """Records frames to disk and rotates out the oldest beyond ``max_files``.

    ``max_files <= 0`` disables rotation (unbounded). Recording is a no-op
    (returns ``None``) while ``enabled`` is False, satisfying OBSERVE-only and
    default-off behavior.
    """

    def __init__(self, directory: Path, enabled: bool = True, max_files: int = 0) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.max_files = max_files

    def record(self, frame: Frame) -> Path | None:
        if not self.enabled:
            return None
        path = save_frame(frame, self.directory)
        self._rotate()
        return path

    def _rotate(self) -> None:
        if self.max_files is None or self.max_files <= 0:
            return
        files = sorted(self.directory.glob("*.png"), key=_rotation_key)
        excess = len(files) - self.max_files
        for stale in files[:excess] if excess > 0 else []:
            stale.unlink()


def _rotation_key(path: Path) -> tuple[str, int]:
    """Ordering key for rotation: timestamp stem, then numeric collision suffix.

    ``save_frame`` appends ``-1``, ``-2``, ... on same-timestamp collisions.
    A plain lexicographic sort would rank ``-10`` before ``-2``, so the key
    parses the suffix numerically to keep oldest-written files first.
    """
    stem = path.stem
    suffix = 0
    if "-" in stem:
        base, _, tail = stem.rpartition("-")
        if tail.isdigit():
            suffix = int(tail)
            stem = base
    return (stem, suffix)
