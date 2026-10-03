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

    # region example (CLI-only):
    ai-game-agent capture --backend mss --region 0,0,1920,1080
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from ai_game_agent.capture import Capture, CaptureError, FpsMeter, save_frame
from ai_game_agent.config import CaptureConfig, LoggingConfig, Region

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
    except (ValueError, CaptureError, OSError) as exc:
        # OSError covers genuine I/O failures (permission, full disk,
        # read-only target) raised by save_frame/Recorder while writing
        # frames; translating it here keeps the CLI's clean "error:" +
        # exit-1 contract instead of a raw traceback (N2).
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
