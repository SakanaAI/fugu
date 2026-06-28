#!/usr/bin/env python3
"""
GPU MODE submission helper.

Submit a kernel file to the gpumode.com leaderboard via popcorn-cli.

Usage:
    python submission.py <codefile> [--leaderboard NAME] [--gpu TYPE] [--mode MODE]

Defaults:
    --leaderboard qr_v2
    --gpu         B200
    --mode        leaderboard      (other modes: test, benchmark, profile)

Examples:
    # submit a kernel to the default leaderboard (qr_v2 on B200, ranked)
    python submission.py mykernel.py

    # correctness-only test
    python submission.py mykernel.py --mode test

    # submit to a different leaderboard / GPU
    python submission.py mykernel.py --leaderboard matmul_v2 --gpu H100

    # submit the qr_v2 seed kernel
    python submission.py init.py

Prereqs (already done in this workspace):
    1. popcorn-cli is on PATH (it's at ~/.local/bin/popcorn-cli)
    2. You have registered: `popcorn register discord` (creates ~/.popcorn.yaml)
"""

import argparse
import os
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit a GPU kernel file to gpumode.com.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("codefile", help="Path to the kernel Python file to submit.")
    parser.add_argument("--leaderboard", default="qr_v2",
                        help="Leaderboard name (default: qr_v2)")
    parser.add_argument("--gpu", default="B200",
                        help="GPU type: H100, A100, B200, L4, MI300, etc. (default: B200)")
    parser.add_argument("--mode", default="leaderboard",
                        choices=["test", "benchmark", "leaderboard", "profile"],
                        help="Submission mode (default: leaderboard)")
    args = parser.parse_args()

    if not os.path.isfile(args.codefile):
        print(f"[error] code file not found: {args.codefile}", file=sys.stderr)
        return 2

    os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", "")
    cli = shutil.which("popcorn-cli")
    if not cli:
        print("[error] popcorn-cli not on PATH. Install with:", file=sys.stderr)
        print("  curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash",
              file=sys.stderr)
        return 3

    if not os.path.exists(os.path.expanduser("~/.popcorn.yaml")):
        print("[error] not authenticated. Run: popcorn register discord", file=sys.stderr)
        return 4

    cmd = [cli, "submit", "--no-tui",
           "--gpu", args.gpu,
           "--leaderboard", args.leaderboard,
           "--mode", args.mode,
           args.codefile]
    print(f"[submit] leaderboard={args.leaderboard} gpu={args.gpu} mode={args.mode} file={args.codefile}",
          file=sys.stderr)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
