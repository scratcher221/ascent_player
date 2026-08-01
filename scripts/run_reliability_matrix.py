#!/usr/bin/env python3
"""Phase-A browser reliability matrix (100 ep per cell).

Runs seed_map_best, thread_bc, and learned checkpoints across watch priors and
skill-exec settings, then writes a summary JSON under logs/.

Example:
  PYTHONPATH=. .venv/bin/python -u scripts/run_reliability_matrix.py --resume
  PYTHONPATH=. .venv/bin/python -u scripts/run_reliability_matrix.py --quick
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
RELIABILITY = ROOT / "scripts/run_browser_reliability_100.py"
CELL_DIR = ROOT / "logs" / "matrix_cells"

_LEGACY_NAME = re.compile(
    r"browser_reliability_(?P<policy>[\w]+)_w(?P<watch>[0-9.]+)_skill(?P<skill>[01])_"
)


@dataclass(slots=True)
class MatrixCell:
    policy: str
    thread_watch_prior: float
    skill_exec: bool
    episodes: int
    mean: float
    min: float
    max: float
    p50: float
    json_path: str
    elapsed_s: float
    rc: int
    skipped: bool = False


def _grid(quick: bool) -> list[tuple[str, float, bool]]:
    if quick:
        return [
            ("seed_map_best", 0.0, False),
            ("thread_bc", 0.58, False),
            ("thread_bc", 0.58, True),
            ("learned", 0.58, False),
        ]
    cells: list[tuple[str, float, bool]] = [
        ("seed_map_best", 0.0, False),
        ("seed_map_best", 0.58, False),
    ]
    for prior in (0.55, 0.58, 0.62):
        for skill in (False, True):
            cells.append(("thread_bc", prior, skill))
    for prior in (0.55, 0.58, 0.62):
        cells.append(("learned", prior, False))
    return cells


def _cell_key(policy: str, watch_prior: float, skill_exec: bool) -> str:
    return f"{policy}_w{watch_prior:.2f}_skill{int(skill_exec)}"


def _cell_out(policy: str, watch_prior: float, skill_exec: bool) -> Path:
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    return CELL_DIR / f"{_cell_key(policy, watch_prior, skill_exec)}.json"


def _cell_complete(path: Path, episodes: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    scores = payload.get("scores") or []
    return (
        int(payload.get("episodes", 0)) >= episodes
        and len(scores) >= episodes
        and float(payload.get("mean", 0)) > 0
    )


def _load_cell(path: Path, episodes: int) -> MatrixCell:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MatrixCell(
        policy=str(payload.get("policy", "")),
        thread_watch_prior=float(payload.get("thread_watch_prior", 0)),
        skill_exec=bool(payload.get("skill_exec_at_watch", False)),
        episodes=int(payload.get("episodes", episodes)),
        mean=float(payload.get("mean", 0)),
        min=float(payload.get("min", 0)),
        max=float(payload.get("max", 0)),
        p50=float(payload.get("p50", 0)),
        json_path=str(path),
        elapsed_s=float(payload.get("elapsed_s", 0)),
        rc=0,
        skipped=True,
    )


def _import_legacy_cells(episodes: int) -> int:
    """Copy finished timestamped runs into logs/matrix_cells/ for --resume."""
    imported = 0
    log_dir = ROOT / "logs"
    for path in sorted(log_dir.glob("browser_reliability_*_w*_skill*.json")):
        m = _LEGACY_NAME.search(path.name)
        if not m:
            continue
        if not _cell_complete(path, episodes):
            continue
        policy = m.group("policy")
        watch = float(m.group("watch"))
        skill = bool(int(m.group("skill")))
        dest = _cell_out(policy, watch, skill)
        if _cell_complete(dest, episodes):
            continue
        dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        imported += 1
        print(f"MATRIX_IMPORT {dest.name} <- {path.name}", flush=True)
    return imported


def _should_print_line(line: str, *, quiet: bool) -> bool:
    if not quiet:
        return True
    if line.startswith("SEED_THREAD "):
        return False
    markers = (
        "BROWSER_EVAL_PROGRESS",
        "BROWSER_RELIABILITY",
        "BROWSER_100:",
        "BROWSER_RESET",
        "BROWSER_PLAYING",
        "BROWSER_LAUNCHED",
        "MATRIX_",
        "Traceback",
        "Error",
        "FAILED",
        "TargetClosedError",
    )
    return any(m in line for m in markers)


def _run_cell(
    policy: str,
    watch_prior: float,
    skill_exec: bool,
    *,
    episodes: int,
    run_seed: int,
    target_mean: float,
    target_min: float,
    retries: int,
    quiet_log: bool,
) -> MatrixCell:
    out = _cell_out(policy, watch_prior, skill_exec)
    if _cell_complete(out, episodes):
        print(f"MATRIX_SKIP complete {out.name}", flush=True)
        cell = _load_cell(out, episodes)
        cell.skipped = True
        return cell

    cmd = [
        str(PY),
        "-u",
        str(RELIABILITY),
        "--policy",
        policy,
        "--episodes",
        str(episodes),
        "--run-seed",
        str(run_seed),
        "--thread-watch-prior",
        str(watch_prior),
        "--target-mean",
        str(target_mean),
        "--target-min",
        str(target_min),
        "--out",
        str(out),
    ]
    if skill_exec:
        cmd.append("--skill-exec")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"

    rc = 1
    elapsed = 0.0
    for attempt in range(1, max(1, retries) + 1):
        print(
            f"MATRIX_CELL_START policy={policy} watch={watch_prior:.2f} "
            f"skill={skill_exec} attempt={attempt}/{retries} out={out.name}",
            flush=True,
        )
        started = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if _should_print_line(line, quiet=quiet_log):
                sys.stdout.write(line)
                sys.stdout.flush()
        rc = int(proc.wait())
        elapsed = time.time() - started
        if rc == 0 and _cell_complete(out, episodes):
            break
        print(
            f"MATRIX_CELL_FAIL rc={rc} complete={_cell_complete(out, episodes)} "
            f"attempt={attempt}",
            flush=True,
        )
        if out.exists() and not _cell_complete(out, episodes):
            out.unlink(missing_ok=True)
        time.sleep(5)

    mean = min_score = max_score = p50 = 0.0
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
        mean = float(payload.get("mean", 0))
        min_score = float(payload.get("min", 0))
        max_score = float(payload.get("max", 0))
        p50 = float(payload.get("p50", 0))

    print(
        f"MATRIX_CELL_DONE policy={policy} watch={watch_prior:.2f} skill={skill_exec} "
        f"mean={mean:.1f} min={min_score:.1f} rc={rc} elapsed_s={elapsed:.0f}",
        flush=True,
    )
    return MatrixCell(
        policy=policy,
        thread_watch_prior=watch_prior,
        skill_exec=skill_exec,
        episodes=episodes,
        mean=mean,
        min=min_score,
        max=max_score,
        p50=p50,
        json_path=str(out),
        elapsed_s=elapsed,
        rc=rc,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--target-mean", type=float, default=2000.0)
    parser.add_argument("--target-min", type=float, default=1000.0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a 4-cell smoke matrix with 10 episodes each",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells already present under logs/matrix_cells/",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per cell when browser subprocess fails",
    )
    parser.add_argument(
        "--verbose-log",
        action="store_true",
        help="Forward full subprocess log (includes SEED_THREAD)",
    )
    parser.add_argument(
        "--seed-baseline",
        type=float,
        default=892.0,
        help="Initial logs/thread_bc_best_mean.txt if missing",
    )
    args = parser.parse_args()
    episodes = 10 if args.quick else int(args.episodes)
    quiet = not bool(args.verbose_log)

    baseline_path = ROOT / "logs" / "thread_bc_best_mean.txt"
    if not baseline_path.exists() and args.seed_baseline > 0:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(f"{float(args.seed_baseline):.4f}\n", encoding="utf-8")
        print(f"MATRIX_SEED thread_bc_best={args.seed_baseline}", flush=True)

    if args.resume:
        n = _import_legacy_cells(episodes)
        print(f"MATRIX_RESUME imported={n} cells_dir={CELL_DIR}", flush=True)

    results: list[MatrixCell] = []
    failed = 0
    for policy, prior, skill in _grid(args.quick):
        cell = _run_cell(
            policy,
            prior,
            skill,
            episodes=episodes,
            run_seed=args.run_seed,
            target_mean=args.target_mean,
            target_min=args.target_min,
            retries=int(args.retries),
            quiet_log=quiet,
        )
        results.append(cell)
        if cell.rc != 0 or cell.mean <= 0:
            failed += 1

    summary_path = ROOT / "logs" / f"reliability_matrix_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "episodes_per_cell": episodes,
        "run_seed": args.run_seed,
        "cells": [asdict(c) for c in results],
        "failed_cells": failed,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"MATRIX_DONE summary={summary_path} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
