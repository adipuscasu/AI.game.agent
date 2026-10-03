"""OpenCV template matching (Phase 2, step 4).

``CvTemplateMatcher`` locates a known UI icon/shape (a template PNG
referenced by *logical name* in configuration) inside a captured frame.

Design rules (plan §5.1, §10):

* Templates are loaded **once at construction** (cache keyed by logical
  name); a missing or corrupt template file raises ``PerceptionError``
  **fail-fast**, never per-frame.
* Matching uses ``cv2.matchTemplate`` with ``TM_CCOEFF_NORMED``; the best
  score is the ``confidence``. Scores below the configured threshold
  (default ``0.8``) are reported as "no hit".
* Frame → OpenCV array is zero-copy (``np.frombuffer``) and RGB→BGR is
  converted in exactly one place, here. ``Frame.pixels`` is documented as
  row-major RGB; OpenCV works in BGR.
* Template files that are larger than the frame are a configuration/setup
  error and raise ``PerceptionError`` at match time.
* ``cv2``/``numpy`` are imported lazily so the package imports cleanly in a
  bare venv; construction without the ``vision`` extra fails with a clear
  error naming the missing extra.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import TYPE_CHECKING

from ai_game_agent.capture import Frame
from ai_game_agent.perception.base import PerceptionError
from ai_game_agent.perception.observation import BBox, TemplateHit

if TYPE_CHECKING:
    import numpy as np

__all__ = ["CvTemplateMatcher"]


def _cv2() -> types.ModuleType:
    """Import OpenCV lazily, with a clean error if the extra is missing."""
    try:
        import cv2

        return cv2
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise PerceptionError(
            "opencv-python-headless is not installed; install the "
            "'vision' extra: uv sync --extra vision"
        ) from exc


def _to_cv_image(cv2: types.ModuleType, frame: Frame) -> np.ndarray:
    """Build a zero-copy BGR ``uint8`` array from a ``Frame``'s RGB pixels."""
    import numpy as np

    if len(frame.pixels) != frame.width * frame.height * 3:
        raise PerceptionError(
            f"frame pixel buffer is {len(frame.pixels)} bytes, expected "
            f"{frame.width}x{frame.height}x3"
        )
    rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
        frame.height, frame.width, 3
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class CvTemplateMatcher:
    """Matches named template PNGs against frames via ``cv2.matchTemplate``.

    Parameters
    ----------
    templates:
        Mapping of logical name → PNG path. Files are loaded and validated
        eagerly so misconfiguration fails at construction, not per-frame.
    threshold:
        Minimum normalized match score (``0.0``–``1.0``) for a hit to count.
        ``TM_CCOEFF_NORMED`` scores are already in this range.
    max_dimensions:
        Upper bound on template width/height (default ``2**32 - 1``, i.e. no
        practical limit). A template larger than *any* frame is a setup
        error and is rejected here.
    """

    def __init__(
        self,
        templates: dict[str, str] | dict[str, Path],
        *,
        threshold: float = 0.8,
        max_dimensions: int = (1 << 32) - 1,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        self._threshold = threshold
        self._templates: dict[str, np.ndarray] = {}
        for name, path in templates.items():
            img = self._load_template(str(path))
            th, tw = img.shape[:2]
            if tw > max_dimensions or th > max_dimensions:
                raise PerceptionError(
                    f"template {name!r} is {tw}x{th}, larger than the "
                    f"maximum {max_dimensions}x{max_dimensions}"
                )
            self._templates[name] = img

    @property
    def threshold(self) -> float:
        return self._threshold

    def match(self, frame: Frame, template: str) -> TemplateHit | None:
        """Return a :class:`TemplateHit` if ``template`` scores ≥ threshold."""
        cv2 = _cv2()
        if template not in self._templates:
            raise PerceptionError(
                f"unknown template {template!r}; known: "
                f"{sorted(self._templates) or 'none'}"
            )

        tmpl = self._templates[template]
        th, tw = tmpl.shape[:2]
        if th > frame.height or tw > frame.width:
            raise PerceptionError(
                f"template {template!r} is {tw}x{th}, larger than the "
                f"{frame.width}x{frame.height} frame"
            )

        tmpl = self._templates[template]
        th, tw = tmpl.shape[:2]
        if th > frame.height or tw > frame.width:
            raise PerceptionError(
                f"template {template!r} is {tw}x{th}, larger than the "
                f"{frame.width}x{frame.height} frame"
            )

        frame_bgr = _to_cv_image(cv2, frame)
        result = cv2.matchTemplate(frame_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
        _, best, _, best_loc = cv2.minMaxLoc(result)
        if best < self._threshold:
            return None

        # ``best_loc`` is the top-left of the best match; ``result`` has
        # shape (h - th + 1, w - tw + 1) — clamp defensively.
        x = max(0, min(int(best_loc[0]), frame.width - tw))
        y = max(0, min(int(best_loc[1]), frame.height - th))
        return TemplateHit(
            template=template,
            bbox=BBox(x=x, y=y, width=tw, height=th),
            confidence=float(best),
        )

    @staticmethod
    def _load_template(path: str) -> np.ndarray:
        cv2 = _cv2()
        file_path = Path(path)
        if not file_path.is_file():
            raise PerceptionError(f"template file not found: {path}")
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001 - surface as setup failure
            raise PerceptionError(f"could not load template {path!r}: {exc}") from exc
        if img is None:
            raise PerceptionError(
                f"could not decode template image {path!r} (corrupt PNG)"
            )
        return img
