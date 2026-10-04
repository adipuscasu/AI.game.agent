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

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ai_game_agent.config import CaptureConfig

# M1: the exact filename shape produced by ``save_frame`` (``frame_`` +
# ``YYYYMMDDTHHMMSS`` + microseconds, an optional ``-N`` collision suffix,
# ``.png``). Rotation must only ever consider files matching this pattern so
# it never deletes unrelated PNGs sharing the directory.
_FRAME_FILE_RE = re.compile(r"^frame_\d{8}T\d{12}(?:-\d+)?\.png$")


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
        """Extract a sub-region of this frame as a new frame.

        Raises:
            CaptureError: if the region dimensions are not positive, or the
                region extends beyond the frame bounds.
        """
        if width <= 0 or height <= 0:
            raise CaptureError(
                f"region dimensions must be positive, got width={width}, height={height}"
            )
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

    def resize(self, scale: float) -> Frame:
        """Return a nearest-neighbor scaled copy of this frame.

        ``scale`` multiplies both dimensions (``1.0`` = identity, ``< 1.0``
        downscales, ``> 1.0`` upscales). Each output pixel samples the nearest
        source pixel, so the result is deterministic and needs no image library.

        Raises:
            CaptureError: if ``scale`` is not a positive finite number.
        """
        if not scale > 0:
            raise CaptureError(f"scale must be > 0, got {scale!r}")
        new_w = max(1, int(self.width * scale))
        new_h = max(1, int(self.height * scale))
        out = bytearray(new_w * new_h * 3)
        for dst_row in range(new_h):
            src_row = min(self.height - 1, int(dst_row / scale))
            src_base = src_row * self.width * 3
            for dst_col in range(new_w):
                src_col = min(self.width - 1, int(dst_col / scale))
                src = src_base + src_col * 3
                dst = (dst_row * new_w + dst_col) * 3
                out[dst : dst + 3] = self.pixels[src : src + 3]
        return Frame(new_w, new_h, bytes(out), self.captured_at, self.source)


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
        # L3: precompute the full pixel buffer once instead of reallocating it
        # on every grab() call.
        self._full_pixels = bytes(rgb) * (width * height)
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def grab(self) -> Frame:
        return Frame(
            width=self._width,
            height=self._height,
            pixels=self._full_pixels,
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
        return MssBackend(monitor=config.monitor)
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

    def __init__(
        self,
        mss_factory: Callable[[], object] | None = None,
        monitor: int = 1,
    ) -> None:
        self._factory = mss_factory
        self._monitor = monitor
        self._screen: object | None = None

    def open(self) -> None:
        if self._screen is None:
            if self._factory is not None:
                factory = self._factory
            else:
                import mss

                factory = mss.MSS  # mss.mss is deprecated since mss 10
            self._screen = factory()

    def close(self) -> None:
        if self._screen is not None:
            self._screen.close()
            self._screen = None

    def grab(self) -> Frame:
        if self._screen is None:
            raise CaptureError("mss backend not opened; call open() first")
        # mss >= 10 requires an explicit monitor. mss monitors[0] is the whole
        # virtual screen (all monitors); monitors[1] is the primary, and
        # monitors[n] for n >= 2 is the (n-1)th secondary. Default is 1 so a
        # bare capture is the primary monitor, not the black-banded virtual
        # desktop (see docs/manual-testing.md).
        shot = self._screen.grab(self._screen.monitors[self._monitor])
        size = shot.size
        # mss 9 exposes shot.size as a dict; mss >= 10 as a Size object.
        if isinstance(size, dict):
            width, height = size["width"], size["height"]
        else:
            width, height = size.width, size.height
        raw = bytes(shot.rgb)
        if len(raw) != width * height * 3:
            raise CaptureError(
                f"unexpected mss frame size: {len(raw)} bytes for {width}x{height}"
            )
        return Frame(width, height, raw, datetime.now(UTC), self.name)


class Capture:
    """Capture facade: fps pacing, region extraction, and optional recording.

    When ``config.record_enabled`` is set, the facade owns a :class:`Recorder`
    and writes each grabbed frame to ``config.record_directory`` as a side
    effect of :meth:`grab`. This keeps the ``record_*`` config fields live in
    the capture layer (M2) instead of leaving them to be re-read ad hoc by
    callers.

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
        # M2: the facade owns recording, driven by the config's record_* fields.
        self.recorder = Recorder(
            Path(config.record_directory),
            enabled=config.record_enabled,
            max_files=config.record_max_files,
        )
    

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

        When recording is enabled, the processed frame is also written to the
        configured record directory as a side effect of this call.

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
        if self.config.scale != 1.0:
            frame = frame.resize(self.config.scale)
        if self.recorder.enabled:
            self.recorder.record(frame)
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

    The candidate name is *claimed* with ``O_CREAT | O_EXCL`` so two writers
    can never pick the same name (L4: closes the exists()-then-write TOCTOU
    window).
    """
    import os

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"frame_{frame.captured_at:%Y%m%dT%H%M%S%f}"
    suffix = 0
    while True:
        name = f"{base}.png" if suffix == 0 else f"{base}-{suffix}.png"
        candidate = directory / name
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            suffix += 1
            continue
        break
    with os.fdopen(fd, "wb") as handle:
        frame.to_pil().save(handle, format="PNG")
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
        # M1: only consider files we recorded ourselves (matching the
        # save_frame naming pattern), never unrelated PNGs in the directory.
        files = [p for p in self.directory.glob("*.png") if _FRAME_FILE_RE.match(p.name)]
        files.sort(key=_rotation_key)
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
