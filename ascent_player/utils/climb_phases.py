"""Climb ladder phase helpers (collect / TD / promote / reliability)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ascent_player.agent.checkpoint import checkpoint_exists
from ascent_player.agent.checkpoint_guard import save_guarded
from ascent_player.agent.dqn import DQNAgent
from ascent_player.agent.offline_bc import offline_bc_frozen_trunk
from ascent_player.utils.climb_policy import (
    PromoteAction,
    TD_RECOVERY_DEFER_MEAN,
    aux_enabled_for_score,
    decide_promote,
    elite_gate_for_score,
    offline_td_min_score,
    reliability_below_floor,
    should_defer_offline_td,
)
from ascent_player.utils.policy_floors import (
    record_thread_bc_eval,
    write_thread_bc_best,
    write_v2_probe_best,
)

if TYPE_CHECKING:
    from ascent_player.config import AppConfig


def offline_td(
    agent: DQNAgent,
    *,
    steps: int,
    lr: float,
    min_score: float,
    save_path: Path,
) -> float | None:
    if steps <= 0:
        print("CLIMB_TD_SKIP steps=0", flush=True)
        return None
    if len(agent.replay) < max(64, agent.batch_size):
        return None
    kept = agent.replay.filter_min_episode_score(min_score)
    if kept < max(64, agent.batch_size):
        print(f"CLIMB_TD_SKIP kept>={min_score:.0f} -> {kept}", flush=True)
        return None
    agent.rebuild_optimizer(lr)
    prev_min = agent.config.training.min_replay_size
    agent.config.training.min_replay_size = 0
    prev_every = agent.train_every
    agent.train_every = 1
    last = None
    try:
        for i in range(max(1, steps)):
            batch = agent.sample_training_batch()
            with agent.tf.device(agent.device_info.training_device):
                loss = agent.train_batch(batch)
            last = float(loss)
            if (i + 1) % max(1, steps // 5) == 0:
                print(f"CLIMB_TD step={i+1}/{steps} loss={last:.4f}", flush=True)
            if (i + 1) % agent.config.training.target_sync_interval == 0:
                agent.sync_target_network(hard=True)
            else:
                agent.sync_target_network(hard=False)
    except Exception as exc:
        print(f"CLIMB_TD_FAIL {type(exc).__name__}: {exc} — continuing to probe", flush=True)
        return None
    finally:
        agent.config.training.min_replay_size = prev_min
        agent.train_every = prev_every
    agent.save(save_path)
    return last


def run_browser_bc_and_td(
    agent: DQNAgent,
    config: "AppConfig",
    *,
    browser_replay: Path,
    elite_replay: Path,
    latest: Path,
    browser_bc_steps: int,
    bc_lr: float,
    td_steps: int,
    td_lr: float,
    online_td: bool,
    v2_best: float,
    best_mean: float,
    load_offline_replay,
) -> float | None:
    """Post-collect frozen BC (+ optional deferred offline TD). Returns last BC loss."""
    loss = None
    if (
        browser_replay.exists()
        and browser_replay.stat().st_size > 64
        and browser_bc_steps > 0
    ):
        agent.replay.clear()
        n_br = agent.replay.load_pickle(
            browser_replay,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
        if n_br < 64 and elite_replay.exists():
            aux = DQNAgent(config)
            aux.replay.clear()
            aux.replay.load_pickle(
                elite_replay,
                max_items=config.training.browser_replay_max_items,
                vector_dim=config.observation.vector_dim,
            )
            agent.replay.extend_from(aux.replay)
            print(
                f"CLIMB_BROWSER_BC_TOPUP elite -> replay={len(agent.replay)}",
                flush=True,
            )
        print(
            f"CLIMB_BROWSER_BC replay={len(agent.replay)} steps={browser_bc_steps}",
            flush=True,
        )
        loss = offline_bc_frozen_trunk(
            agent,
            steps=browser_bc_steps,
            lr=float(bc_lr),
            save_path=latest,
            log_prefix="CLIMB_BC",
        )
    if browser_replay.exists() and int(td_steps) > 0 and not online_td:
        if should_defer_offline_td(v2_best):
            print(
                f"CLIMB_TD_SKIP until v2_best>={TD_RECOVERY_DEFER_MEAN:.0f} "
                f"(now={v2_best:.1f}; BC-only until FT gate)",
                flush=True,
            )
        else:
            if len(agent.replay) < 64:
                load_offline_replay(agent, config)
            aux = DQNAgent(config)
            aux.replay.clear()
            if browser_replay.stat().st_size > 64:
                aux.replay.load_pickle(
                    browser_replay,
                    max_items=config.training.browser_replay_max_items,
                    vector_dim=config.observation.vector_dim,
                )
                agent.replay.extend_from(aux.replay)
            offline_td(
                agent,
                steps=int(td_steps),
                lr=float(td_lr),
                min_score=offline_td_min_score(
                    best_mean=best_mean, v2_best=v2_best
                ),
                save_path=latest,
            )
    elif int(td_steps) <= 0 and browser_bc_steps <= 0:
        print("CLIMB_TD_SKIP disabled (td_steps=0) — probe next", flush=True)
    return loss


async def run_reliability(
    *,
    reliability_fn,
    config: "AppConfig",
    episodes: int,
    eval_seed: int,
    round_idx: int,
    append_log,
    ckpt: Path | None = None,
    warn_vs_v2: float | None = None,
    raise_nature_floor: bool = False,
    agent: DQNAgent | None = None,
    v2_best_path: Path | None = None,
    best_mean: float | None = None,
) -> float | None:
    """Unified reliability probe. Returns updated best_mean when raised."""
    rel = await reliability_fn(
        config,
        episodes,
        ckpt=ckpt,
        eval_seed=eval_seed,
    )
    print(
        f"CLIMB_RELIABILITY n={episodes} "
        f"mean={rel['mean']:.1f} min={rel['min']:.1f} max={rel['max']:.1f}",
        flush=True,
    )
    append_log({"round": round_idx, "event": "reliability", **rel})
    if warn_vs_v2 is not None and reliability_below_floor(rel["mean"], warn_vs_v2):
        print(
            f"CLIMB_RELIABILITY_WARN mean={rel['mean']:.1f} "
            f"vs v2_best={warn_vs_v2:.1f} (floor unchanged)",
            flush=True,
        )
    if (
        raise_nature_floor
        and agent is not None
        and v2_best_path is not None
        and best_mean is not None
        and rel["mean"] > best_mean
    ):
        rel_save = save_guarded(agent, v2_best_path)
        if rel_save.rejected:
            print(
                "CLIMB_RELIABILITY_HOLD save_rejected V2_BEST "
                f"(mean={rel['mean']:.1f})",
                flush=True,
            )
            return best_mean
        write_thread_bc_best(rel["mean"])
        return float(rel["mean"])
    return best_mean


async def resolve_probe_and_persist(
    *,
    agent: DQNAgent,
    config: "AppConfig",
    probe_mean: float,
    v2_best: float,
    best_mean: float,
    bootstrap_floor: float,
    confirm_eps: int,
    probe_eps: int,
    greedy_probe_fn,
    eval_seed: int,
    pre: Path,
    latest: Path,
    thread_bc: Path,
    v2_best_path: Path,
    round_idx: int,
    bc_loss: float | None,
    append_log,
) -> tuple[float, float, bool]:
    """Apply promote/hold/revert. Returns (v2_best, best_mean, promoted)."""
    decision = decide_promote(
        probe_mean=probe_mean,
        v2_best=v2_best,
        bootstrap_floor=bootstrap_floor,
    )
    improved = decision.improved
    collapsed = decision.collapsed
    promote = decision.action is PromoteAction.PROMOTE

    if not promote:
        restore = (
            v2_best_path
            if (v2_best > 0 and checkpoint_exists(v2_best_path))
            else pre
        )
        assert agent.load(restore)
        agent.save(latest)
        save_guarded(agent, thread_bc)
        event = "revert" if collapsed else "hold"
        print(
            f"CLIMB_{event.upper()} probe={probe_mean:.1f} "
            f"v2_best={v2_best:.1f} — restored {restore.name}",
            flush=True,
        )
        append_log(
            {
                "round": round_idx,
                "event": event,
                "probe": probe_mean,
                "best": best_mean,
                "v2_best": v2_best,
                "bc_loss": bc_loss,
            }
        )
        return v2_best, best_mean, False

    if confirm_eps > probe_eps and v2_best > 0:
        confirm_mean = await greedy_probe_fn(
            config, confirm_eps, eval_seed=eval_seed
        )
        print(
            f"CLIMB_CONFIRM n={confirm_eps} mean={confirm_mean:.1f} "
            f"(probe={probe_mean:.1f})",
            flush=True,
        )
        append_log(
            {
                "round": round_idx,
                "event": "confirm",
                "probe": probe_mean,
                "confirm": confirm_mean,
                "v2_best": v2_best,
            }
        )
        if confirm_mean <= v2_best + 1.0:
            assert agent.load(
                v2_best_path if checkpoint_exists(v2_best_path) else pre
            )
            agent.save(latest)
            save_guarded(agent, thread_bc)
            print(
                f"CLIMB_HOLD confirm={confirm_mean:.1f} "
                f"v2_best={v2_best:.1f} — not promoted",
                flush=True,
            )
            append_log(
                {
                    "round": round_idx,
                    "event": "hold",
                    "probe": probe_mean,
                    "confirm": confirm_mean,
                    "v2_best": v2_best,
                    "bc_loss": bc_loss,
                }
            )
            return v2_best, best_mean, False
        probe_mean = confirm_mean
        improved = probe_mean > v2_best + 1.0

    bc_res = save_guarded(agent, thread_bc)
    agent.save(latest)
    if bc_res.rejected:
        print(
            f"CLIMB_HOLD save_rejected THREAD_BC — probe={probe_mean:.1f}",
            flush=True,
        )
        assert agent.load(
            v2_best_path if checkpoint_exists(v2_best_path) else pre
        )
        agent.save(latest)
        return v2_best, best_mean, False

    if improved or v2_best <= 0 or probe_mean > v2_best:
        best_res = save_guarded(agent, v2_best_path)
        if best_res.rejected:
            print(
                f"CLIMB_HOLD save_rejected V2_BEST — "
                f"probe={probe_mean:.1f} floor unchanged at {v2_best:.1f}",
                flush=True,
            )
        else:
            v2_best = max(v2_best, probe_mean)
            write_v2_probe_best(v2_best)
            print(f"CLIMB_V2_BEST mean={v2_best:.1f}", flush=True)
    else:
        print(
            f"CLIMB_KEEP probe={probe_mean:.1f} (v2_best stays {v2_best:.1f})",
            flush=True,
        )
    if probe_mean > best_mean:
        best_mean = probe_mean
        write_thread_bc_best(best_mean)
        print(f"CLIMB_BEST_UPDATE mean={best_mean:.1f}", flush=True)
    record_thread_bc_eval(probe_mean)
    if aux_enabled_for_score(v2_best):
        config.training.reason_aux_weight = 0.1
        config.training.skill_aux_weight = 0.12
        print("CLIMB_AUX_ON reason/skill CE re-enabled (≥2k)", flush=True)
    raised = elite_gate_for_score(
        v2_best,
        float(getattr(config.training, "post_2k_elite_gate", 3000.0) or 3000.0),
    )
    if raised is not None:
        config.training.elite_replay_min_episode_score = raised
        print(f"CLIMB_ELITE_GATE->{raised:.0f}", flush=True)
    return v2_best, best_mean, True
