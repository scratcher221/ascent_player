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
    lock = parser.add_mutually_exclusive_group()
    lock.add_argument(
        "--lock-run-seed",
        dest="lock_run_seed",
        action="store_true",
        help="Lock collect/train to --run-seed (default for the 2k push).",
    )
    lock.add_argument(
        "--unlock-run-seed",
        dest="lock_run_seed",
        action="store_false",
        help="Random maps each collect episode (eval seed still locked).",
    )
    parser.set_defaults(lock_run_seed=True)
    parser.add_argument("--target-mean", type=float, default=2000.0)
    parser.add_argument("--bc-steps", type=int, default=1000)
    parser.add_argument("--bc-lr", type=float, default=1e-5)
    parser.add_argument(
        "--td-steps",
        type=int,
        default=0,
        help="Offline TD steps after BC (0 until online TD keep holds).",
    )
    parser.add_argument("--td-lr", type=float, default=1e-5)
    parser.add_argument(
        "--browser-bc-steps",
        type=int,
        default=1000,
        help="After policy collect, frozen BC on browser/elite replay.",
    )
    parser.add_argument("--collect-minutes", type=float, default=15.0)
    parser.add_argument("--collect-eps", type=float, default=0.04)
    parser.add_argument("--probe-episodes", type=int, default=20)
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
