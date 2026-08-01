#!/usr/bin/env python3
"""Offline diagnostics for skill labels, action agreement, and routing confusion."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.agent.dqn import DQNAgent
from ascent_player.agent.skills import (
    CLIMB_ABOVE,
    INTERCEPT_SURGE,
    LAND_BELOW,
    RIDE_STREAM,
    index_to_skill,
)
from ascent_player.config import AppConfig, DeviceMode

KEY_SKILLS = (LAND_BELOW, CLIMB_ABOVE, INTERCEPT_SURGE, RIDE_STREAM)


def _build_config(checkpoint: Path) -> AppConfig:
    config = AppConfig()
    config.training.device_mode = DeviceMode.AUTO
    config.training.skills_enabled = True
    config.training.skill_exec_at_watch = True
    config.training.checkpoint_path = checkpoint
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect skill-head quality on replay.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/dqn_thread_bc.keras"),
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("checkpoints/browser_replay.pkl"),
    )
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/skill_diagnostics.json"),
    )
    args = parser.parse_args()

    config = _build_config(args.checkpoint)
    agent = DQNAgent(config)
    if not agent.load(args.checkpoint):
        print(f"FAILED_LOAD checkpoint={args.checkpoint}", flush=True)
        return 1
    agent.replay.clear()
    loaded = agent.replay.load_pickle(
        args.replay,
        max_items=max(1, int(args.max_items)),
        vector_dim=config.observation.vector_dim,
    )
    if loaded <= 0:
        print(f"FAILED_REPLAY replay={args.replay}", flush=True)
        return 2

    batch = agent.replay.sample(min(len(agent.replay), max(256, agent.batch_size)))
    accuracy = agent.evaluate_skill_accuracy()
    distribution = {
        index_to_skill(skill_idx): count
        for skill_idx, count in sorted(agent.replay.skill_label_distribution().items())
    }

    model_batch = agent._to_model_batch(batch.states)
    if agent._vector_dim > 0:
        outputs = agent.online(
            [
                agent.tf.convert_to_tensor(model_batch[0], dtype=agent.tf.float32),
                agent.tf.convert_to_tensor(model_batch[1], dtype=agent.tf.float32),
            ],
            training=False,
        )
    else:
        outputs = agent.online(
            agent.tf.convert_to_tensor(model_batch, dtype=agent.tf.float32),
            training=False,
        )
    q_values, _, skill_logits = agent._unpack_outputs(outputs)
    q_np = q_values.numpy() if hasattr(q_values, "numpy") else np.asarray(q_values)
    s_np = skill_logits.numpy() if hasattr(skill_logits, "numpy") else np.asarray(skill_logits)
    pred_actions = np.argmax(q_np, axis=1).astype(np.int32)
    pred_skills = np.argmax(s_np, axis=1).astype(np.int32)
    skills = np.asarray(batch.skills if batch.skills is not None else [], dtype=np.int32)
    valid = skills >= 0

    action_agreement = float(np.mean(pred_actions == batch.actions))
    confusion: dict[str, dict[str, int]] = {}
    if np.any(valid):
        for target_idx, pred_idx in zip(skills[valid], pred_skills[valid], strict=True):
            target_name = index_to_skill(int(target_idx))
            pred_name = index_to_skill(int(pred_idx))
            confusion.setdefault(target_name, {})
            confusion[target_name][pred_name] = confusion[target_name].get(pred_name, 0) + 1

    key_skill_coverage = {
        skill: distribution.get(skill, 0)
        for skill in KEY_SKILLS
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "replay": str(args.replay),
        "loaded": loaded,
        "distribution": distribution,
        "key_skill_coverage": key_skill_coverage,
        "action_agreement": action_agreement,
        "skill_accuracy": accuracy,
        "routing_confusion": confusion,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"SKILL_DIAGNOSTICS loaded={loaded} action_agree={action_agreement:.3f} "
        f"acc={accuracy.get('accuracy', 0.0):.3f} -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
