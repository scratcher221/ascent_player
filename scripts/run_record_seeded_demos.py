#!/usr/bin/env python3
"""Record seeded human demos and export replay for skill bootstrap."""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.config import AppConfig
from ascent_player.demo.export import export_demos_to_replay
from ascent_player.demo.recorder import DemoRecorder
from ascent_player.demo.storage import new_demo_path, save_demo
from ascent_player.env.browser_backend import BrowserBackend
from ascent_player.env.game_env import ACTION_LABELS, AscentGameEnv


def _build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig()
    config.browser.run_seed = int(args.run_seed)
    config.browser.lock_run_seed = True
    config.browser.raise_on_launch = not args.no_raise
    config.browser.manual_cdp_url = args.manual_cdp_url
    config.demo.save_dir = Path(args.demo_dir)
    return config


async def main_async(args: argparse.Namespace) -> int:
    config = _build_config(args)
    backend = BrowserBackend(config.browser)
    env = AscentGameEnv(config, backend)
    recorder = DemoRecorder(config, backend, env)
    stop_requested = False

    def request_stop(*_args) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    deadline = (
        time.monotonic() + float(args.minutes) * 60.0
        if args.minutes and args.minutes > 0
        else None
    )
    keep_min_score = max(0.0, float(args.keep_min_score or 0.0))
    target_kept_episodes = max(0, int(args.target_kept_episodes or 0))
    gated_mode = keep_min_score > 0 or target_kept_episodes > 0
    kept_paths: list[Path] = []
    kept_episodes = 0
    completed_episodes = 0

    try:
        await recorder.prepare(attach_existing=bool(args.attach_existing or args.manual_cdp_url))
        mode = "attached" if (args.attach_existing or args.manual_cdp_url) else "launched"
        print(
            f"RECORD_READY seed={config.browser.run_seed} mode={mode} "
            f"demo_dir={config.demo.save_dir}",
            flush=True,
        )
        print(
            "Play with A / D / Space in the browser window. "
            "Press Ctrl+C in this terminal when you want to save.",
            flush=True,
        )
        if deadline is not None:
            print(f"RECORD_DEADLINE seconds={int(max(0, deadline - time.monotonic()))}", flush=True)

        last_report = 0.0
        while not stop_requested:
            frame, action, done = await recorder.capture_step()
            del frame
            now = time.monotonic()
            if now - last_report >= 2.0:
                print(
                    f"RECORD_STATUS episodes={recorder.episode_id + 1} "
                    f"transitions={len(recorder.transitions)} "
                    f"action={ACTION_LABELS[action]}",
                    flush=True,
                )
                last_report = now
            await backend.wait_ms(env.step_wait_ms())
            if done:
                completed_episodes += 1
                print(
                    f"RECORD_EPISODE_END episode={recorder.episode_id} "
                    f"transitions={len(recorder.transitions)}",
                    flush=True,
                )
                if gated_mode:
                    episode = recorder.finish_episode()
                    peak_score = max((float(t.score or 0.0) for t in episode), default=0.0)
                    if episode and peak_score >= keep_min_score:
                        path = save_demo(new_demo_path(config), episode)
                        kept_paths.append(path)
                        kept_episodes += 1
                        print(
                            f"DEMO_KEPT path={path} peak_score={peak_score:.0f} "
                            f"kept={kept_episodes}",
                            flush=True,
                        )
                    else:
                        print(
                            f"DEMO_DISCARDED peak_score={peak_score:.0f} "
                            f"threshold={keep_min_score:.0f}",
                            flush=True,
                        )
                    if target_kept_episodes and kept_episodes >= target_kept_episodes:
                        stop_requested = True
                        break
                else:
                    recorder.on_episode_end()
                    if args.max_episodes and recorder.episode_id >= int(args.max_episodes):
                        stop_requested = True
                        break
                if args.max_episodes and completed_episodes >= int(args.max_episodes):
                    stop_requested = True
                    break
                await env.reset()
            if deadline is not None and now >= deadline:
                stop_requested = True

        if gated_mode:
            if kept_paths:
                print(
                    f"DEMO_KEEP_SUMMARY kept={kept_episodes} completed={completed_episodes} "
                    f"threshold={keep_min_score:.0f}",
                    flush=True,
                )
            else:
                print(
                    f"DEMO_KEEP_SUMMARY kept=0 completed={completed_episodes} "
                    f"threshold={keep_min_score:.0f}",
                    flush=True,
                )
        else:
            path = await recorder.stop_and_save()
            kept_paths = [path]
            print(
                f"DEMO_SAVED path={path} transitions={len(recorder.transitions)} "
                f"episodes={recorder.episode_id + 1}",
                flush=True,
            )

        if args.replay_out and kept_paths:
            result = export_demos_to_replay(
                config,
                demo_paths=kept_paths,
                replay_path=Path(args.replay_out),
                append=bool(args.append_replay),
            )
            print(f"REPLAY_EXPORTED {result.status_message}", flush=True)
        elif args.replay_out and gated_mode:
            print("REPLAY_SKIPPED no kept demos matched the score gate", flush=True)
        return 0
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        await env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record seeded human demos and export replay for bootstrap.",
    )
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=ROOT / "demonstrations" / "seeded_human",
        help="Where to save the recorded .npz demos.",
    )
    parser.add_argument(
        "--replay-out",
        type=Path,
        default=ROOT / "checkpoints" / "human_seeded_replay.pkl",
        help="Replay pickle to write for skill/bootstrap runs.",
    )
    parser.add_argument(
        "--append-replay",
        action="store_true",
        help="Append to an existing replay pickle instead of replacing it.",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=0.0,
        help="Optional auto-stop wall clock limit; 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional auto-stop after this many completed episodes.",
    )
    parser.add_argument(
        "--keep-min-score",
        type=float,
        default=0.0,
        help="Only keep completed episodes whose peak score reaches this threshold.",
    )
    parser.add_argument(
        "--target-kept-episodes",
        type=int,
        default=0,
        help="Stop automatically after this many kept episodes satisfy --keep-min-score.",
    )
    parser.add_argument(
        "--attach-existing",
        action="store_true",
        help="Attach to an existing Ascent tab instead of launching an isolated browser.",
    )
    parser.add_argument(
        "--manual-cdp-url",
        type=str,
        default=None,
        help="Explicit CDP endpoint to attach to.",
    )
    parser.add_argument(
        "--no-raise",
        action="store_true",
        help="Do not try to bring the launched browser window to the front.",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
