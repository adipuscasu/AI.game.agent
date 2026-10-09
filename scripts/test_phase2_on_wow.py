"""Run the Phase 2 perception pipeline against a static screenshot (shots/wow.png).

This exercises the *exact* pipeline the ``analyze`` CLI uses, but feeds it a
``Frame`` decoded from a PNG instead of grabbing the live screen — so it is
deterministic, headless, and repeatable. All four subsystems are pointed at
*real* content visible in the shot:

* **templates**  - the glowing green/yellow orb in the center of the screen
* **ui_zones**   - presence + brightness over that same orb region
* **ocr**        - the "The <zone> 22:02" title strip in the top-right corner
* **objects**    - the red dragonflight wings (saturated red blob)

Run from the repo root:

    uv run python scripts/test_phase2_on_wow.py

It prints the Observation JSON to stdout and a short PASS/FAIL verdict per
subsystem to stderr, then exits 0 (all subsystems produced output, no
``detector_errors``).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from ai_game_agent.capture import Frame
from ai_game_agent.config import PerceptionConfig, UiZone
from ai_game_agent.perception import Perception
from ai_game_agent.perception.color_blobs import ColorBlobsDetector
from ai_game_agent.perception.ocr import TesseractEngine
from ai_game_agent.perception.template import CvTemplateMatcher
from ai_game_agent.perception.ui import UiRegionDetector

ROOT = Path(__file__).resolve().parent.parent
SHOT = ROOT / "shots" / "wow.png"
ORB_TEMPLATE = ROOT / "shots" / "wow_orb_template.png"

# Real content coordinates, measured on this specific 3838x2155 screenshot.
# (Regenerate ORB_TEMPLATE by cropping the orb from shots/wow.png if you
# replace the screenshot.)
ORB = (2033, 938, 222, 184)          # x, y, w, h - the glowing orb
TITLE = (3260, 5, 560, 80)           # x, y, w, h - top-right zone title strip
WING_RGB = (220, 25, 10)             # saturated red (dragonflight wings)
WING_TOL = 45


def load_frame(path: Path) -> Frame:
    im = Image.open(path).convert("RGB")  # RGBA -> RGB
    w, h = im.size
    return Frame(w, h, im.tobytes(), datetime.now(UTC), "file")


def build_config() -> PerceptionConfig:
    ox, oy, ow, oh = ORB
    tx, ty, tw, th = TITLE
    return PerceptionConfig(
        enabled=True,
        template_threshold=0.8,
        templates={"orb": str(ORB_TEMPLATE)},
        ui_zones=(
            UiZone("orb_present", ox, oy, ow, oh, check="presence"),
            UiZone("orb_bright", ox, oy, ow, oh, check="brightness", threshold=40),
        ),
        ocr_enabled=True,
        ocr_regions=(
            {"name": "zone_title", "x": tx, "y": ty, "width": tw, "height": th},
        ),
        objects_enabled=True,
        object_colors=(
            {"name": "red_wings", "rgb": list(WING_RGB), "tolerance": WING_TOL},
        ),
    )


def build_perception(cfg: PerceptionConfig) -> Perception:
    return Perception(
        cfg,
        template_matcher=CvTemplateMatcher({"orb": str(ORB_TEMPLATE)}, threshold=0.8),
        ui_detector=UiRegionDetector(cfg),
        ocr_engine=TesseractEngine(),
        object_detector=ColorBlobsDetector(
            [{"name": "red_wings", "rgb": list(WING_RGB), "tolerance": WING_TOL}]
        ),
    )


def main() -> int:
    if not SHOT.is_file():
        print(f"error: screenshot not found: {SHOT}", file=sys.stderr)
        return 1
    if not ORB_TEMPLATE.is_file():
        print(
            f"error: orb template not found: {ORB_TEMPLATE} "
            "(crop it from shots/wow.png first)",
            file=sys.stderr,
        )
        return 1

    frame = load_frame(SHOT)
    print(f"# frame: {frame.width}x{frame.height} source={frame.source}", file=sys.stderr)

    perception = build_perception(build_config())
    obs = perception.observe(frame)
    d = obs.to_dict()
    print(json.dumps(d, indent=2))

    # -- verdict -------------------------------------------------------------
    print("\n# ---- verdict ----", file=sys.stderr)

    def verdict(label: str, ok: bool, note: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f"  ({note})" if note else ""), file=sys.stderr)

    tmpl = d["templates"]
    verdict(
        "template match (orb)",
        len(tmpl) > 0 and tmpl[0]["confidence"] >= 0.8,
        f"n={len(tmpl)} conf={max((t['confidence'] for t in tmpl), default=0):.3f}",
    )
    verdict("ui_zone checks", len(d["ui_zones"]) > 0, f"n={len(d['ui_zones'])}")
    texts = [t.get("text", "").strip() for t in d["text_regions"] if t.get("text")]
    verdict("ocr read (zone title)", len(texts) > 0, f"text={texts!r}")
    objs = d["objects"]
    big = [o for o in objs if o["confidence"] > 0.01]
    verdict(
        "object blobs (red wings)",
        len(big) > 0,
        f"n={len(objs)} (>=1% conf: {len(big)})",
    )
    verdict(
        "no detector errors",
        len(d["detector_errors"]) == 0,
        f"errors={d['detector_errors']!r}",
    )

    all_ok = (
        len(tmpl) > 0
        and len(d["ui_zones"]) > 0
        and len(texts) > 0
        and len(big) > 0
        and not d["detector_errors"]
    )
    print(f"# overall: {'PASS' if all_ok else 'FAIL'}", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
