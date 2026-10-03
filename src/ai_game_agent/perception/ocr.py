"""OCR engine for the perception pipeline (Phase 2, step 7).

``TesseractEngine`` is the real implementation of the ``OcrEngine`` protocol
(``perception/base.py``). It wraps a pluggable *Tesseract backend* — any
object exposing ``image_to_data(image, *, config) -> str`` — so:

* tests inject a fake backend and never touch a real Tesseract binary;
* production wires the backend through ``pytesseract`` (the ``ocr`` extra).

Design rules:

* The engine never imports heavy dependencies at module load. ``PIL`` (for
  cropping) and ``pytesseract`` (for the default backend) are imported
  lazily, inside the methods that need them.
* ``read(frame, region)`` crops the frame to ``region`` (clamping to frame
  bounds), runs the backend, and parses the TSV into a single ``TextRegion``
  with the bbox in *frame space* (the requested region, not the crop).
* The engine does not swallow backend errors — it lets them propagate.
  The ``Perception`` pipeline is responsible for catching per-region
  exceptions into ``Observation.detector_errors``.

The default backend (``_default_backend()``) imports ``pytesseract`` and
``PIL`` on first call, so the module imports cleanly in a bare venv.
"""

from __future__ import annotations

from typing import Protocol

from ai_game_agent.capture import Frame
from ai_game_agent.perception.observation import BBox, TextRegion

__all__ = ["TesseractEngine", "TesseractBackend"]


class TesseractBackend(Protocol):
    """A minimal Tesseract backend.

    ``pytesseract`` satisfies this structurally: its ``image_to_data``
    method accepts a PIL image and returns a TSV string.
    """

    def image_to_data(
        self, image: object, *, config: str | None = None
    ) -> str: ...


def _to_pil_image(frame: Frame, region: BBox) -> object:
    """Crop ``frame`` to ``region`` and return a PIL image.

    The region is clamped to the frame bounds. If the clamped region has
    zero area (entirely outside the frame), a 1x1 blank image is returned
    so the backend never sees an invalid crop — callers should check
    ``_region_has_area`` before calling this.

    PIL is imported lazily so the module loads in a bare venv.
    """
    from PIL import Image

    fw, fh = frame.width, frame.height
    x0 = max(0, min(region.x, fw))
    y0 = max(0, min(region.y, fh))
    x1 = max(0, min(region.x + region.width, fw))
    y1 = max(0, min(region.y + region.height, fh))

    # Zero-area crop: return a 1x1 blank so the backend doesn't choke.
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGB", (1, 1), color=(0, 0, 0))

    img = Image.frombytes("RGB", (fw, fh), frame.pixels)
    return img.crop((x0, y0, x1, y1))


def _region_has_area(frame: Frame, region: BBox) -> bool:
    """True if ``region`` overlaps the frame with non-zero area."""
    x0 = max(0, min(region.x, frame.width))
    y0 = max(0, min(region.y, frame.height))
    x1 = max(0, min(region.x + region.width, frame.width))
    y1 = max(0, min(region.y + region.height, frame.height))
    return x1 > x0 and y1 > y0


def _parse_tesseract_tsv(data: str) -> tuple[str | None, float]:
    """Parse a Tesseract ``image_to_data`` TSV string.

    Returns ``(text, confidence)``. ``text`` is ``None`` when no word
    (level-5) rows carry non-whitespace text. ``confidence`` is the mean
    of the per-word confidences, clamped to [0.0, 1.0]; ``0.0`` when there
    are no words.
    """
    lines = data.splitlines()
    if not lines:
        return None, 0.0

    header = lines[0].split("\t")
    try:
        level_idx = header.index("level")
        conf_idx = header.index("conf")
        text_idx = header.index("text")
    except ValueError:
        # Malformed TSV — treat as no text.
        return None, 0.0

    words: list[str] = []
    confidences: list[float] = []

    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        level = parts[level_idx]
        if level != "5":
            continue
        text = parts[text_idx].strip()
        if not text:
            continue
        words.append(text)
        try:
            conf = float(parts[conf_idx])
        except (ValueError, IndexError):
            conf = 0.0
        # Tesseract reports 0-100; normalize to [0, 1].
        conf = max(0.0, min(1.0, conf / 100.0))
        confidences.append(conf)

    if not words:
        return None, 0.0

    text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences)
    return text, max(0.0, min(1.0, mean_conf))


class TesseractEngine:
    """Real ``OcrEngine`` implementation backed by a Tesseract backend.

    Parameters
    ----------
    backend:
        An object satisfying the ``TesseractBackend`` protocol
        (``pytesseract`` or a fake for tests). If ``None``, a default
        ``pytesseract``-based backend is constructed lazily on first use —
        this requires the ``ocr`` extra to be installed.
    """

    __slots__ = ("_backend",)

    def __init__(self, backend: TesseractBackend | None = None) -> None:
        self._backend = backend

    def _resolve_backend(self) -> TesseractBackend:
        if self._backend is not None:
            return self._backend
        # Lazy import: requires the ``ocr`` extra (pytesseract + Pillow).
        import pytesseract

        self._backend = pytesseract
        return pytesseract

    def read(self, frame: Frame, region: BBox) -> TextRegion | None:
        """Read text from ``region`` of ``frame``.

        Returns a ``TextRegion`` (bbox in frame space) or ``None`` when no
        text is recognized. Raises if the backend fails — the ``Perception``
        pipeline is responsible for catching that into ``detector_errors``.
        """
        if not _region_has_area(frame, region):
            return None

        backend = self._resolve_backend()
        img = _to_pil_image(frame, region)
        raw = backend.image_to_data(img, config="--psm 6")
        text, confidence = _parse_tesseract_tsv(raw)

        if text is None:
            return None

        return TextRegion(text=text, bbox=region, confidence=confidence)


def _default_backend() -> TesseractBackend:  # pragma: no cover
    """Construct a default pytesseract-based backend.

    Kept as a standalone function so ``TesseractEngine.__init__`` stays
    lightweight and the import is deferred to first use.
    """
    import pytesseract

    return pytesseract
