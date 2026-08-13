"""Frozen-trunk offline behavior cloning (Nature thread-BC recipe)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ascent_player.agent.dqn import DQNAgent


def freeze_vision_and_skill(agent: "DQNAgent") -> list[str]:
    """Freeze conv trunk + skill head for motor BC. Returns frozen layer names."""
    frozen: list[str] = []
    for layer in agent.online.layers:
        name = (layer.name or "").lower()
        if (
            any(k in name for k in ("conv", "separable", "depthwise", "skill_logits"))
            and layer.trainable
        ):
            layer.trainable = False
            frozen.append(layer.name)
    return frozen


def unfreeze_layers(agent: "DQNAgent", names: list[str]) -> None:
    name_set = set(names)
    for layer in agent.online.layers:
        if layer.name in name_set:
            layer.trainable = True


def offline_bc_frozen_trunk(
    agent: "DQNAgent",
    *,
    steps: int,
    lr: float,
    save_path: Path | None = None,
    log_prefix: str = "OFFLINE_BC",
) -> float | None:
    """Behavior-clone with vision trunk + skill head frozen."""
    if steps <= 0:
        print(f"{log_prefix}_SKIP steps=0", flush=True)
        return None
    if len(agent.replay) < max(64, agent.batch_size):
        print(f"{log_prefix}_SKIP replay={len(agent.replay)}", flush=True)
        return None
    agent.rebuild_optimizer(lr)
    frozen = freeze_vision_and_skill(agent)
    if frozen:
        print(f"{log_prefix}_FREEZE layers={frozen}", flush=True)
    last = None
    batch_size = min(agent.batch_size, len(agent.replay))
    try:
        with agent.tf.device(agent.device_info.training_device):
            for i in range(max(1, steps)):
                batch = agent.replay.sample(batch_size)
                last = float(
                    agent.bc_train_step(batch.states, batch.actions).numpy()
                )
                if (i + 1) % max(1, steps // 5) == 0:
                    print(
                        f"{log_prefix} step={i+1}/{steps} loss={last:.4f} "
                        f"replay={len(agent.replay)}",
                        flush=True,
                    )
        agent.sync_target_network(hard=True)
        if save_path is not None:
            agent.save(save_path)
    finally:
        unfreeze_layers(agent, frozen)
    return last
