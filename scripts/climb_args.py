"""Shared argparse for v2 climb ladder + watchdog."""
from __future__ import annotations

import argparse


def add_climb_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=424242,
        help="Seed used for greedy probe/reliability (comparable floors).",
    )
    parser.add_argument(
        "--lock-run-seed",
        action="store_true",
        help="Lock collect/train to --run-seed (default: random maps each episode).",
    )
    parser.add_argument("--target-mean", type=float, default=2000.0)
    parser.add_argument("--bc-steps", type=int, default=800)
    parser.add_argument("--bc-lr", type=float, default=1e-5)
    parser.add_argument("--td-steps", type=int, default=400)
    parser.add_argument("--td-lr", type=float, default=1e-5)
    parser.add_argument(
        "--browser-bc-steps",
        type=int,
        default=800,
        help="After policy collect, frozen BC on browser/elite replay.",
    )
    parser.add_argument("--collect-minutes", type=float, default=12.0)
    parser.add_argument("--collect-eps", type=float, default=0.05)
    parser.add_argument("--probe-episodes", type=int, default=12)
    parser.add_argument(
        "--confirm-episodes",
        type=int,
        default=40,
        help="On probe improve, confirm with N greedy eps before V2_BEST promote (0=off).",
    )
    parser.add_argument(
        "--bootstrap-keep-floor",
        type=float,
        default=350.0,
        help="Initial keep floor when v2_best is unset.",
    )
    parser.add_argument("--reliability-episodes", type=int, default=40)
    parser.add_argument("--reliability-every", type=int, default=4)
    return parser
