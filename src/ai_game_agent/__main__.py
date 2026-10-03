"""Command-line interface for the AI game agent (Phase 1: screen observation).

Subcommands
-----------
``capture``
    Grab a single frame from the configured backend and save it as PNG.
    ``--out`` chooses the output directory (default ``screenshots``).

``observe``
    Run a capture loop for ``--frames`` frames, optionally recording each
    frame to disk. Prints the measured FPS at the end. ``--fps`` is a pacing
    *ceiling*, not a guarantee: each iteration does work and then sleeps, so
    the achieved rate approaches ``--fps`` from below.

Both commands are safe to run headless with ``--backend mock --fps 0``:
``fps=0`` disables real pacing (no sleeps), which is what the test suite and
CI rely on.

Usage::

    ai-game-agent capture --backend mss --out shots/
    ai-game-agent observe --backend mock --frames 30 --fps 0 --record --out rec/

``analyze``
    Run one capture and pass the frame through the perception pipeline
    (template matching → UI zones → OCR → object blobs), printing the
    resulting ``Observation`` as JSON. Detector failures are recorded in
    the observation's ``detector_errors`` list rather than aborting. The
    subsystems run based on ``config/default.yaml`` (or ``--config``) and
    can be individually disabled with ``--no-templates`` / ``--no-ocr`` /
    ``--no-objects``. With nothing configured, the observation is empty
    (Phase 1 behavior).

Usage::

    ai-game-agent capture --backend mss --out shots/
    ai-game-agent observe --backend mock --frames 30 --fps 0 --record --out rec/

    # region example (CLI-only):
    ai-game-agent capture --backend mss --region 0,0,1920,1080

    # perception analysis (needs the "vision" / "ocr" extras to actually run)
    ai-game-agent analyze --backend mock --no-ocr --no-objects --pretty
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ai_game_agent.capture import Capture, CaptureError, FpsMeter, save_frame
from ai_game_agent.config import (
    CaptureConfig,
    ConfigError,
    LoggingConfig,
    PerceptionConfig,
    Region,
    load_config,
)
from ai_game_agent.perception import Perception, PerceptionError
from ai_game_agent.perception.color_blobs import ColorBlobsDetector
from ai_game_agent.perception.ocr import TesseractEngine
from ai_game_agent.perception.template import CvTemplateMatcher

log = logging.getLogger("ai_game_agent")


def _setup_logging() -> None:
    """Configure the ``ai_game_agent`` logger to write to a file.

    Log output goes to ``<logging.directory>/agent.log`` (default
    ``logs/agent.log``) at the configured level.  No ``StreamHandler`` is
    attached so that stdout carries only program output (``saved: …``,
    ``frames=… fps=…``) and stderr carries only ``error:`` / usage — the
    stream-routing contract the CLI and e2e tests assert on.
    """
    log_cfg = LoggingConfig()  # level=INFO, directory=logs (defaults)
    log_dir = Path(log_cfg.directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ai_game_agent")
    logger.setLevel(getattr(logging, log_cfg.level, logging.INFO))

    # Remove handlers left by a previous main() call (happens in-process
    # during the test suite) so the file always reflects the current CWD.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    handler = logging.FileHandler(log_dir / "agent.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False


def _parse_region(value: str | None) -> Region | None:
    """Parse a CLI ``x,y,width,height`` region string into a validated Region."""
    if value is None:
        return None
    x, y, width, height = (int(part) for part in value.split(","))
    return Region(x, y, width, height)


def _parse_scale(value: str | None) -> float | None:
    """Parse a CLI scale factor string; ``None`` leaves the config default."""
    if value is None:
        return None
    scale = float(value)
    if not scale > 0:
        raise ValueError("scale must be a positive number")
    return scale


def _build_capture(
    args: argparse.Namespace,
    *,
    record_enabled: bool,
    record_dir: Path,
    record_max_files: int,
) -> Capture:
    scale = _parse_scale(getattr(args, "scale", None))
    cfg = CaptureConfig(
        backend=args.backend,
        region=_parse_region(args.region),
        fps=args.fps,  # fps=0 is a valid "no pacing" value in the config model
        scale=1.0 if scale is None else scale,
        record_enabled=record_enabled,
        record_directory=str(record_dir),
        record_max_files=record_max_files,
    )
    return Capture(cfg)


def _capture(args: argparse.Namespace) -> int:
    cap = _build_capture(
        args,
        record_enabled=False,
        record_dir=Path(args.out),
        record_max_files=getattr(args, "record_max_files", 1000),
    )
    with cap:
        frame = cap.grab()
    path = save_frame(frame, Path(args.out))
    log.info("capture: saved frame %s", path)
    print(f"saved: {path}")
    return 0


def _observe(args: argparse.Namespace) -> int:
    meter = FpsMeter()
    cap = _build_capture(
        args,
        record_enabled=args.record,
        record_dir=Path(args.out),
        record_max_files=args.record_max_files,
    )
    # M2: recording is owned by the Capture facade (cap.recorder) and happens
    # as a side effect of grab(); the CLI only measures FPS and reports.
    with cap:
        for _ in range(max(0, args.frames)):
            cap.grab()
            meter.record(time.perf_counter())
            # N1: pacing sleeps *after* the work, so each iteration costs
            # grab() + record() + 1/fps; the achieved rate is a little under
            # the requested --fps (a ceiling, not a guarantee). This is
            # deliberate and documented on the --fps flag and in the module
            # docstring; FpsMeter reports the honest measured rate.
            cap.wait_next_frame()
    log.info("observe: frames=%d fps=%.2f", meter.frame_count, meter.fps)
    print(f"frames={meter.frame_count} fps={meter.fps:.2f}")
    return 0


def _build_perception(args: argparse.Namespace, cfg) -> Perception:
    """Assemble a :class:`Perception` from the loaded config and CLI flags.

    Each subsystem is included only if the config enables it *and* the user
    has not opted out via the matching ``--no-*`` flag. Detectors are
    constructed lazily so a missing optional extra only surfaces when the
    subsystem is actually selected (and only when it is selected does the
    pipeline fail fast if the extra is absent).
    """
    perception = cfg.perception
    templates_cfg = perception.templates or {}
    if getattr(args, "no_templates", False):
        templates_cfg = {}
    template_matcher = (
        CvTemplateMatcher(templates_cfg, threshold=perception.template_threshold)
        if templates_cfg
        else None
    )
    ui_zones = perception.ui_zones
    if getattr(args, "no_templates", False):
        # UI zones are gated by the same ``--no-templates`` flag; the flag is
        # about "template-matching-ish vision subsystems" (templates + ui
        # zones) versus text (OCR) versus blobs (objects).
        ui_zones = ()
    ocr_regions = list(perception.ocr_regions) if perception.ocr_enabled else []
    if getattr(args, "no_ocr", False):
        ocr_regions = []
    ocr_engine = TesseractEngine() if ocr_regions else None
    object_colors = (
        list(perception.object_colors) if perception.objects_enabled else []
    )
    if getattr(args, "no_objects", False):
        object_colors = []
    object_detector = (
        ColorBlobsDetector(object_colors)
        if object_colors
        else None
    )
    perception_cfg = PerceptionConfig(
        enabled=perception.enabled,
        template_threshold=perception.template_threshold,
        templates=dict(templates_cfg),
        ui_zones=ui_zones,
        ocr_enabled=bool(ocr_regions),
        ocr_regions=tuple(ocr_regions),
        objects_enabled=bool(object_colors),
        object_colors=tuple(object_colors),
    )
    return Perception(
        perception_cfg,
        template_matcher=template_matcher,
        ui_detector=None,  # ui zones: no standalone detector in Phase 2
        ocr_engine=ocr_engine,
        object_detector=object_detector,
    )


def _analyze(args: argparse.Namespace) -> int:
    cfg = load_config(getattr(args, "config", None))
    cap = _build_capture(
        args,
        record_enabled=False,
        record_dir=Path("screenshots"),
        record_max_files=1000,
    )
    with cap:
        frame = cap.grab()
    perception = _build_perception(args, cfg)
    observation = perception.observe(frame)
    if getattr(args, "pretty", False):
        text = json.dumps(observation.to_dict(), indent=2)
    else:
        text = json.dumps(observation.to_dict(), sort_keys=True)
    print(text)
    log.info("analyze: frame=%dx%d detectors=%d", frame.width, frame.height, len(observation.detector_errors))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-game-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_cap = sub.add_parser("capture", help="Grab one screenshot and save it as PNG.")
    p_cap.add_argument("--backend", default="mss", help="capture backend (mss | mock)")
    p_cap.add_argument("--region", default=None, help="x,y,width,height (integers)")
    p_cap.add_argument("--scale", default=None, help="downscale factor (e.g. 0.5)")
    p_cap.add_argument("--out", default="screenshots", help="output directory")
    p_cap.add_argument("--fps", type=int, default=0, help="0 = no pacing (CLI default)")
    p_cap.set_defaults(func=_capture)

    p_obs = sub.add_parser("observe", help="Run a capture loop and print measured FPS.")
    p_obs.add_argument("--backend", default="mss")
    p_obs.add_argument("--region", default=None, help="x,y,width,height (integers)")
    p_obs.add_argument("--scale", default=None, help="downscale factor (e.g. 0.5)")
    p_obs.add_argument("--frames", type=int, default=30)
    p_obs.add_argument(
        "--fps",
        type=int,
        default=0,
        help="0 = no pacing (headless/CI); otherwise a ceiling, not a guarantee",
    )
    p_obs.add_argument("--record", action="store_true", help="write each frame to --out")
    p_obs.add_argument("--record-max-files", type=int, default=1000)
    p_obs.add_argument("--out", default="recordings")
    p_obs.set_defaults(func=_observe)

    p_an = sub.add_parser(
        "analyze", help="Capture one frame and print the perception Observation as JSON."
    )
    p_an.add_argument("--backend", default="mss", help="capture backend (mss | mock)")
    p_an.add_argument("--region", default=None, help="x,y,width,height (integers)")
    p_an.add_argument("--scale", default=None, help="downscale factor (e.g. 0.5)")
    p_an.add_argument("--fps", type=int, default=0, help="0 = no pacing (CLI default)")
    p_an.add_argument("--config", default=None, help="YAML config file (default: config/default.yaml)")
    p_an.add_argument("--no-templates", action="store_true",
                      help="disable template matching and UI-zone checks")
    p_an.add_argument("--no-ocr", action="store_true", help="disable OCR")
    p_an.add_argument("--no-objects", action="store_true", help="disable color-blob object detection")
    p_an.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    p_an.set_defaults(func=_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # No subcommand: usage goes to stderr (argparse convention), exit 2.
        parser.print_help(sys.stderr)
        return 2
    _setup_logging()
    try:
        return int(args.func(args))
    except (ValueError, CaptureError, ConfigError, PerceptionError, OSError) as exc:
        # OSError covers genuine I/O failures (permission, full disk,
        # read-only target) raised by save_frame/Recorder while writing
        # frames; translating it here keeps the CLI's clean "error:" +
        # exit-1 contract instead of a raw traceback (N2).
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
