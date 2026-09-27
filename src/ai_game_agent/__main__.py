"""Command-line interface for the AI game agent (Phase 1: screen observation).

Subcommands
-----------
``capture``
    Grab a single frame from the configured backend and save it as PNG.
    ``--out`` chooses the output directory (default ``screenshots``).

``observe``
    Run a capture loop for ``--frames`` frames, optionally recording each
    frame to disk. Prints the measured FPS at the end.

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
import sys
import time
from pathlib import Path

from ai_game_agent.capture import Capture, CaptureError, FpsMeter, Recorder, save_frame
from ai_game_agent.config import CaptureConfig, Region


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


def _build_capture(args: argparse.Namespace, *, record_enabled: bool, record_dir: Path) -> Capture:
    scale = _parse_scale(getattr(args, "scale", None))
    cfg = CaptureConfig(
        backend=args.backend,
        region=_parse_region(args.region),
        fps=args.fps,  # fps=0 is a valid "no pacing" value in the config model
        scale=1.0 if scale is None else scale,
        record_enabled=record_enabled,
        record_directory=str(record_dir),
    )
    return Capture(cfg)


def _capture(args: argparse.Namespace) -> int:
    cap = _build_capture(args, record_enabled=False, record_dir=Path(args.out))
    with cap:
        frame = cap.grab()
    path = save_frame(frame, Path(args.out))
    print(f"saved: {path}")
    return 0


def _observe(args: argparse.Namespace) -> int:
    record = args.record
    out = Path(args.out)
    rec = Recorder(out, enabled=record, max_files=args.record_max_files) if record else None

    meter = FpsMeter()
    cap = _build_capture(args, record_enabled=record, record_dir=out)
    with cap:
        for _ in range(max(0, args.frames)):
            frame = cap.grab()
            if rec is not None:
                rec.record(frame)
            meter.record(time.perf_counter())
            cap.wait_next_frame()
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
    p_obs.add_argument("--fps", type=int, default=0, help="0 = no pacing (headless/CI)")
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
    try:
        return int(args.func(args))
    except (ValueError, CaptureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
