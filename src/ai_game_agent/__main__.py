"""Command-line interface for the AI game agent.

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

``analyze``
    Run one capture and pass the frame through the perception pipeline
    (template matching → UI zones → OCR → object blobs), printing the
    resulting ``Observation`` as JSON. Detector failures are recorded in
    the observation's ``detector_errors`` list rather than aborting. The
    subsystems run based on ``config/default.yaml`` (or ``--config``) and
    can be individually disabled with ``--no-templates`` / ``--no-ocr`` /
    ``--no-objects``. With nothing configured, the observation is empty
    (Phase 1 behavior).

``capture`` and ``observe`` are safe to run headless with ``--backend mock
--fps 0``: ``fps=0`` disables real pacing (no sleeps), which is what the
test suite and CI rely on.

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
    Config,
    ConfigError,
    PerceptionConfig,
    Region,
    load_config,
)
from ai_game_agent.perception import Perception, PerceptionError

# NB: the detector *implementation* modules are NOT imported at module top.
# __main__.py must stay importable — and `perception.enabled: false` must
# not import, instantiate, or execute any perception implementation — so
# each implementation is imported lazily inside _build_perception, in the
# branch where its subsystem is actually selected (color_blobs: cv2/numpy;
# template/ui: OpenCV helpers; ocr: pytesseract/PIL). A missing optional
# extra then fails fast with a clean error naming the extra, never at
# import time (§7.3: the CLI works with no vision/ocr extras).

log = logging.getLogger("ai_game_agent")


def _setup_logging(config: Config) -> None:
    """Configure the ``ai_game_agent`` logger to write to a file.

    Log output goes to ``<logging.directory>/agent.log`` at the configured
    level, both taken from the loaded ``config.logging`` block (defaulting to
    ``logs/agent.log`` at INFO). No ``StreamHandler`` is attached so that
    stdout carries only program output (``saved: …``, ``frames=… fps=…``) and
    stderr carries only ``error:`` / usage — the stream-routing contract the
    CLI and e2e tests assert on.
    """
    log_dir = Path(config.logging.directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ai_game_agent")
    logger.setLevel(getattr(logging, config.logging.level, logging.INFO))

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


def _parse_monitor(value: str | None) -> int | None:
    """Parse a CLI monitor index; ``None`` leaves the config default."""
    if value is None:
        return None
    monitor = int(value)
    if monitor < 0:
        raise ValueError("monitor must be a non-negative integer")
    return monitor


def _build_capture(
    cfg: Config,
    args: argparse.Namespace,
    *,
    record_enabled: bool | None,
    record_dir: str | Path | None,
    record_max_files: int | None,
) -> Capture:
    """Build a :class:`Capture` honoring CLI flags, then config, then defaults.

    Precedence: an explicitly-passed flag (``--backend``/``--region``/
    ``--scale``/``--fps``/``--monitor``) wins; otherwise the value in
    ``cfg.capture``; otherwise the :class:`CaptureConfig` code default. Before
    this, the flag defaults (mss / whole-monitor / 1.0 / no-pacing) were forced
    no matter what the config said.

    Recording is the same rule: ``record_enabled``/``record_dir``/
    ``record_max_files`` are set by the caller from the subcommand's own flags
    (``None`` = the flag was not passed), and ``None`` falls back to the
    ``capture.record.{enabled,directory,max_files}`` config block. Subcommands
    that have no recording flags pass explicit values (typically disabled).
    """
    scale = _parse_scale(getattr(args, "scale", None))
    monitor = _parse_monitor(getattr(args, "monitor", None))
    region = _parse_region(getattr(args, "region", None))
    ccfg = cfg.capture
    return Capture(
        CaptureConfig(
            backend=args.backend if args.backend is not None else ccfg.backend,
            region=region if region is not None else ccfg.region,
            fps=args.fps if args.fps is not None else ccfg.fps,
            scale=scale if scale is not None else ccfg.scale,
            record_enabled=(
                ccfg.record_enabled if record_enabled is None else record_enabled
            ),
            record_directory=(
                str(record_dir) if record_dir is not None else ccfg.record_directory
            ),
            record_max_files=(
                ccfg.record_max_files if record_max_files is None else record_max_files
            ),
            monitor=monitor if monitor is not None else ccfg.monitor,
        )
    )


def _capture(args: argparse.Namespace, cfg: Config) -> int:
    cap = _build_capture(
        cfg,
        args,
        record_enabled=False,
        record_dir=args.out,
        record_max_files=1000,
    )
    with cap:
        frame = cap.grab()
    path = save_frame(frame, Path(args.out))
    log.info("capture: saved frame %s", path)
    print(f"saved: {path}")
    return 0


def _observe(args: argparse.Namespace, cfg: Config) -> int:
    meter = FpsMeter()
    # ``--record`` and ``--no-record`` both write to the single
    # ``record`` destination (``store_const`` True/False, default ``None``)
    # so argparse applies them in command-line order and the LAST flag
    # wins: ``--record --no-record`` → False, ``--no-record --record`` →
    # True. A ``None`` sentinel (no flag) means the config's
    # capture.record.{enabled,directory,max_files} supplies the default —
    # the same precedence as every other capture knob (flag → config →
    # code default).
    record_enabled = args.record
    cap = _build_capture(
        cfg,
        args,
        record_enabled=record_enabled,
        record_dir=args.out,
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


def _resolve_template_path(path: str | Path) -> Path:
    """Anchor a configured template path to the repository root.

    The project convention (README, docs/manual-testing.md) is to run the CLI
    from the repository root, and the shipped reference config
    (``config/fulltest.yaml``) uses repo-relative paths such as
    ``assets/templates/loot_glow.png``. Resolving those against the process
    CWD made the template matcher fail-fast with ``PerceptionError``
    ("template file not found") whenever ``analyze``/``capture`` ran from
    anywhere else — review P2-3.

    Absolute paths are used as-is. Relative paths are anchored to the repo
    root, which is stable because ``__main__.py`` lives at
    ``src/ai_game_agent/__main__.py`` inside the repo: three parent hops
    (package dir → ``src`` → repo root) give the root regardless of CWD.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / p


def _build_perception(args: argparse.Namespace, cfg) -> Perception:
    """Assemble a :class:`Perception` from the loaded config and CLI flags.

    Each subsystem is included only if the config enables it *and* the user
    has not opted out via the matching ``--no-*`` flag. Detectors are
    constructed lazily so a missing optional extra only surfaces when the
    subsystem is actually selected (and only when it is selected does the
    pipeline fail fast if the extra is absent).

    ``perception.enabled == False`` is honored first: no detector is
    imported, constructed, or executed at all, so a missing optional extra
    (vision/ocr) or a misconfigured template path cannot turn "perception
    disabled" into a setup error. ``Perception.observe()`` then returns the
    empty observation the config contract promises.
    """
    perception = cfg.perception
    if not perception.enabled:
        return Perception(perception)
    templates_cfg = perception.templates or {}
    if getattr(args, "no_templates", False):
        templates_cfg = {}
    template_matcher = None
    if templates_cfg:
        # Lazy import: only when the template subsystem is actually
        # selected, so `perception.enabled=false` (or a template-free
        # config) imports no OpenCV-backed implementation at all.
        # Resolve relative template paths against the repo root so a
        # config with ``templates: assets/templates/x.png`` works from
        # any CWD (review P2-3), matching the documented convention.
        # Absolute paths pass through unchanged.
        resolved_templates = {
            name: str(_resolve_template_path(p)) for name, p in templates_cfg.items()
        }
        from ai_game_agent.perception.template import CvTemplateMatcher

        template_matcher = CvTemplateMatcher(
            resolved_templates, threshold=perception.template_threshold
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
    ocr_engine = None
    if ocr_regions:
        # Lazy import: the OCR engine module is imported only when OCR is
        # actually selected (see module-top note).
        from ai_game_agent.perception.ocr import TesseractEngine

        ocr_engine = TesseractEngine()
    object_colors = (
        list(perception.object_colors) if perception.objects_enabled else []
    )
    if getattr(args, "no_objects", False):
        object_colors = []
    object_detector = None
    if object_colors:
        # Lazy import: color_blobs needs the "vision" extra (cv2/numpy).  It
        # is only imported when the objects subsystem is actually selected,
        # so a bare venv (no vision extra) still imports the CLI cleanly and
        # the extra's absence surfaces here as a fail-fast PerceptionError.
        from ai_game_agent.perception.color_blobs import ColorBlobsDetector

        object_detector = ColorBlobsDetector(object_colors)
    perception_cfg = PerceptionConfig(
        enabled=perception.enabled,
        template_threshold=perception.template_threshold,
        templates=dict(templates_cfg),
        ui_zones=ui_zones,
        ocr_enabled=bool(ocr_regions),
        ocr_regions=tuple(ocr_regions),
        # Carry the configured OCR noise floor through verbatim — omitting it
        # would silently reset it to the 0.5 default (review P1).
        ocr_min_confidence=perception.ocr_min_confidence,
        objects_enabled=bool(object_colors),
        object_colors=tuple(object_colors),
    )
    ui_detector = None
    if ui_zones:
        # Lazy import: only when UI zones are actually selected.
        from ai_game_agent.perception.ui import UiRegionDetector

        ui_detector = UiRegionDetector(perception_cfg)
    return Perception(
        perception_cfg,
        template_matcher=template_matcher,
        ui_detector=ui_detector,
        ocr_engine=ocr_engine,
        object_detector=object_detector,
    )


def _analyze(args: argparse.Namespace, cfg: Config) -> int:
    # Build the perception pipeline FIRST so any setup failure (invalid
    # config, missing optional extra, detector construction error) is
    # reported before we open a capture backend — fail fast with no half-open
    # capture resource to leak. ``Perception.__init__`` is pure (it only holds
    # detector references; it never touches the screen), so this is safe.
    perception = _build_perception(args, cfg)
    cap = _build_capture(
        cfg,
        args,
        record_enabled=False,
        record_dir="screenshots",
        record_max_files=1000,
    )
    with cap:
        frame = cap.grab()
    observation = perception.observe(frame)
    if getattr(args, "pretty", False):
        text = json.dumps(observation.to_dict(), indent=2)
    else:
        text = json.dumps(observation.to_dict(), sort_keys=True)
    print(text)
    log.info(
        "analyze: frame=%dx%d detector_errors=%d",
        frame.width, frame.height, len(observation.detector_errors),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-game-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_cap = sub.add_parser("capture", help="Grab one screenshot and save it as PNG.")
    p_cap.add_argument("--backend", default=None,
                       help="capture backend (mss | mock); default from config")
    p_cap.add_argument("--monitor", default=None,
                       help="mss monitor index: 1=primary (default), 2+=secondary, 0=all")
    p_cap.add_argument("--region", default=None, help="x,y,width,height (integers)")
    p_cap.add_argument("--scale", default=None, help="downscale factor (e.g. 0.5)")
    p_cap.add_argument("--out", default="screenshots", help="output directory")
    p_cap.add_argument("--fps", type=int, default=None,
                       help="0 = no pacing; default from config (30)")
    p_cap.add_argument(
        "--config", default=None,
        help="YAML config file (default: config/default.yaml); "
             "provides defaults for --backend/--region/--scale/--fps/--monitor",
    )
    p_cap.set_defaults(func=_capture)

    p_obs = sub.add_parser("observe", help="Run a capture loop and print measured FPS.")
    p_obs.add_argument("--backend", default=None,
                       help="capture backend (mss | mock); default from config")
    p_obs.add_argument("--monitor", default=None,
                       help="mss monitor index: 1=primary (default), 2+=secondary, 0=all")
    p_obs.add_argument("--region", default=None, help="x,y,width,height (integers)")
    p_obs.add_argument("--scale", default=None, help="downscale factor (e.g. 0.5)")
    p_obs.add_argument("--frames", type=int, default=30)
    p_obs.add_argument(
        "--fps",
        type=int,
        default=None,
        help="0 = no pacing (headless/CI); otherwise a ceiling, not a guarantee. "
             "Default from config (30)",
    )
    # Both flags share the ``record`` destination (store_const True/False,
    # default None) so argparse applies them in command-line order: the
    # last flag wins, and a missing pair falls back to the config.
    p_obs.add_argument("--record", dest="record", action="store_const",
                       const=True, default=None,
                       help="write each frame to --out"
                            " (default: capture.record.enabled from config)")
    p_obs.add_argument("--no-record", dest="record", action="store_const",
                       const=False, default=None,
                       help="do not record for this run, even if the config "
                            "enables capture.record.enabled")
    p_obs.add_argument("--record-max-files", type=int, default=None,
                       help="max frames to keep before rotation"
                            " (default: capture.record.max_files from config, 1000)")
    p_obs.add_argument("--out", default=None,
                       help="output directory"
                            " (default: capture.record.directory from config, else recordings)")
    p_obs.add_argument(
        "--config", default=None,
        help="YAML config file (default: config/default.yaml); "
             "provides defaults for --backend/--region/--scale/--fps/--monitor"
             "/--record/--record-max-files/--no-record",
    )
    p_obs.set_defaults(func=_observe)

    p_an = sub.add_parser(
        "analyze", help="Capture one frame and print the perception Observation as JSON."
    )
    p_an.add_argument("--backend", default=None,
                      help="capture backend (mss | mock); default from config")
    p_an.add_argument("--monitor", default=None,
                      help="mss monitor index: 1=primary (default), 2+=secondary, 0=all")
    p_an.add_argument("--region", default=None, help="x,y,width,height (integers)")
    p_an.add_argument("--scale", default=None, help="downscale factor (e.g. 0.5)")
    p_an.add_argument("--fps", type=int, default=None,
                      help="0 = no pacing; default from config (30)")
    p_an.add_argument(
        "--config", default=None,
        help="YAML config file (default: config/default.yaml)",
    )
    p_an.add_argument("--no-templates", action="store_true",
                      help="disable template matching AND UI-zone checks "
                           "(they share one flag; see docs/phase-2-implementation-plan.md §5.2)")
    p_an.add_argument("--no-ocr", action="store_true", help="disable OCR")
    p_an.add_argument(
        "--no-objects", action="store_true",
        help="disable color-blob object detection",
    )
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
    # The config is the source of defaults for every subcommand: an
    # explicitly-passed CLI flag wins, otherwise the loaded config, otherwise
    # the code default. Load it once here and hand it to the subcommand
    # handler so capture settings and logging are all config-driven.  The
    # load sits inside the try so a malformed file yields the clean "error:"
    # + exit-1 contract instead of a raw traceback (mirrors N2).
    try:
        cfg = load_config(getattr(args, "config", None))
        _setup_logging(cfg)
        return int(args.func(args, cfg))
    except (ValueError, CaptureError, ConfigError, PerceptionError, OSError) as exc:
        # OSError covers genuine I/O failures (permission, full disk,
        # read-only target) raised by save_frame/Recorder while writing
        # frames; translating it here keeps the CLI's clean "error:" +
        # exit-1 contract instead of a raw traceback (N2).
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
