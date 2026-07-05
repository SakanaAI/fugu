#!/usr/bin/env python3
"""Plot an SVG on AxiDraw using pyaxidraw plot mode."""

from __future__ import annotations

import argparse
from pathlib import Path


def plot_svg(path: Path, *, dry_run: bool = False, speed: int = 50, penup_speed: int = 75, accel: int = 100) -> bool:
    """Plot an existing SVG file with pyaxidraw plot mode."""
    if not path.exists():
        raise FileNotFoundError(path)
    if dry_run:
        print(f"[axi:dry-run] would plot SVG file: {path}")
        print(f"[axi:dry-run] speed_pendown={speed}, speed_penup={penup_speed}, accel={accel}")
        return True
    try:
        from pyaxidraw import axidraw

        ad = axidraw.AxiDraw()
        ad.plot_setup(str(path))
        # High acceleration with variable speed reduces visible corner stutter.
        ad.options.accel = int(accel)
        ad.options.const_speed = False
        ad.options.speed_pendown = int(speed)
        ad.options.speed_penup = int(penup_speed)
        ad.plot_run()
        print(f"[axi] plotted {path}")
        return True
    except Exception as exc:  # noqa: BLE001 - hardware errors should surface in the job log
        print(f"[axi] plot_svg failed: {exc}")
        return False


def home(*, dry_run: bool = False) -> bool:
    """Raise pen and return carriage to home."""
    if dry_run:
        print("[axi:dry-run] would home (pen up, return to origin).")
        return True
    try:
        from pyaxidraw import axidraw

        ad = axidraw.AxiDraw()
        ad.interactive()
        if ad.connect():
            ad.moveto(0, 0)
            ad.disconnect()
        print("[axi] home complete")
        return True
    except Exception as exc:  # noqa: BLE001 - hardware errors should surface in the job log
        print(f"[axi] home failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Send an SVG to AxiDraw hardware.")
    parser.add_argument("svg", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print what would plot without touching hardware.")
    parser.add_argument("--speed", type=int, default=50, help="Pen-down speed, 1-100.")
    parser.add_argument("--penup-speed", type=int, default=75, help="Pen-up travel speed, 1-100.")
    parser.add_argument("--accel", type=int, default=100, help="AxiDraw acceleration, 1-100.")
    parser.add_argument("--no-home", action="store_true", help="Do not return to origin after a successful plot.")
    args = parser.parse_args()

    speed = max(1, min(100, args.speed))
    penup_speed = max(1, min(100, args.penup_speed))
    accel = max(1, min(100, args.accel))
    svg = args.svg.expanduser().resolve()

    ok = plot_svg(svg, dry_run=args.dry_run, speed=speed, penup_speed=penup_speed, accel=accel)
    if ok and not args.no_home:
        home(dry_run=args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
