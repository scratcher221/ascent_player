from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
import time

import numpy as np

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig
from ascent_player.env.game_env import ACTION_LABELS
from ascent_player.env.state_detector import JUMP_ACTIONS, FrameState


@dataclass(slots=True)
class BrowserStepContext:
    step_ms: float | None = None
    loop_hz: float | None = None
    score_velocity: float | None = None
    episode_reward: float | None = None
    total_steps: int | None = None


@dataclass
class _WindowStats:
    action_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.int64)
    )
    reward_sum: float = 0.0
    reward_min: float = float("inf")
    reward_max: float = float("-inf")
    reward_count: int = 0
    score_delta_sum: float = 0.0
    score_delta_count: int = 0
    jump_count: int = 0
    jump_while_depleted: int = 0
    depleted_steps: int = 0
    boost_level_sum: float = 0.0
    steps: int = 0
    deaths: int = 0
    train_losses: list[float] = field(default_factory=list)
    train_ms: list[float] = field(default_factory=list)
    loop_hz_sum: float = 0.0
    loop_hz_count: int = 0
    step_ms_sum: float = 0.0
    step_ms_count: int = 0
    platform_dx_sum: float = 0.0
    platform_dy_sum: float = 0.0
    platform_samples: int = 0
    in_menu_steps: int = 0

    def clear(self) -> None:
        self.action_counts.fill(0)
        self.reward_sum = 0.0
        self.reward_min = float("inf")
        self.reward_max = float("-inf")
        self.reward_count = 0
        self.score_delta_sum = 0.0
        self.score_delta_count = 0
        self.jump_count = 0
        self.jump_while_depleted = 0
        self.depleted_steps = 0
        self.boost_level_sum = 0.0
        self.steps = 0
        self.deaths = 0
        self.train_losses.clear()
        self.train_ms.clear()
        self.loop_hz_sum = 0.0
        self.loop_hz_count = 0
        self.step_ms_sum = 0.0
        self.step_ms_count = 0
        self.platform_dx_sum = 0.0
        self.platform_dy_sum = 0.0
        self.platform_samples = 0
        self.in_menu_steps = 0


@dataclass
class _EpisodeStats:
    steps: int = 0
    reward: float = 0.0
    max_score: float = 0.0
    action_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.int64)
    )
    jump_while_depleted: int = 0

    def clear(self) -> None:
        self.steps = 0
        self.reward = 0.0
        self.max_score = 0.0
        self.action_counts.fill(0)
        self.jump_while_depleted = 0


class TrainingLogger:
  """Append-only text log for diagnosing learning progress."""

  def __init__(self, config: AppConfig, phase: str) -> None:
    self.config = config
    self.phase = phase
    self.log_dir = config.training.log_dir
    self.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.path = self.log_dir / f"training_{stamp}_{phase}.log"
    latest = self.log_dir / "training_latest.log"
    self._file = self.path.open("a", encoding="utf-8", buffering=1)
    self._lock = threading.Lock()
    self._started = time.perf_counter()
    self._window = _WindowStats()
    self._episode = _EpisodeStats()
    self._last_score: float | None = None
    self._episode_index = 0
    self._recent_episode_scores: deque[float] = deque(maxlen=50)
    self._recent_episode_lengths: deque[int] = deque(maxlen=50)
    self._recent_episode_rewards: deque[float] = deque(maxlen=50)
    self._last_weight_norm: float | None = None
    self._interval = (
      config.training.log_interval_steps_sim
      if phase == "sim"
      else config.training.log_interval_steps
    )
    self._browser_detail_interval = config.training.log_browser_detail_steps
    self._write(f"Log file: {self.path}")
    if phase == "browser":
      latest_browser = self.log_dir / "training_latest_browser.log"
      try:
        latest_browser.write_text(f"{self.path.name}\n", encoding="utf-8")
      except OSError:
        pass
    try:
      latest.write_text(f"{self.path.name}\n", encoding="utf-8")
    except OSError:
      pass

  def log_note(self, message: str) -> None:
    with self._lock:
      self._write(message)

  def close(self, agent: DQNAgent | None = None) -> None:
    with self._lock:
      total_steps = agent.metrics.total_steps if agent is not None else 0
      self._flush_window(agent=agent, total_steps=total_steps, force=True)
      elapsed = time.perf_counter() - self._started
      self._write(f"SESSION END phase={self.phase} elapsed={elapsed:.1f}s")
      self._file.close()

  def log_session_start(
    self,
    agent: DQNAgent,
    *,
    message: str = "",
    extra: dict[str, object] | None = None,
  ) -> None:
    training = self.config.training
    lines = [
      "=" * 72,
      f"SESSION START {datetime.now().isoformat(timespec='seconds')}",
      f"phase={self.phase}",
      f"device={agent.device_message}",
      f"epsilon={agent.epsilon:.4f}",
      f"learning_rate={training.learning_rate}",
      f"gamma={training.gamma}",
      f"batch_size={agent.batch_size}",
      f"train_every={agent.train_every}",
      f"replay_buffer={training.replay_buffer_size}",
      f"min_replay={training.min_replay_size}",
      f"double_dqn={training.use_double_dqn}",
      f"transfer_from_sim={training.transfer_from_sim}",
      f"sim_mode={training.sim_mode}",
      f"episodes_completed={agent.progress.episodes_completed}",
      f"total_steps={agent.metrics.total_steps}",
      f"best_score={agent.progress.best_score:.0f}",
      f"best_reward={agent.progress.best_reward:.3f}",
    ]
    if self.phase == "browser":
      lines.extend(
        [
          f"frame_skip={training.frame_skip}",
          f"watch_mode={training.watch_mode}",
          f"async_training={training.async_training}",
          f"game_fps={training.game_fps}",
          f"log_detail_every={self._browser_detail_interval}",
        ]
      )
    if message:
      lines.append(f"load_message={message}")
    if extra:
      for key, value in extra.items():
        lines.append(f"{key}={value}")
    lines.append("=" * 72)
    self._write("\n".join(lines))

  def record_browser_step(
    self,
    action: int,
    reward: float,
    frame_state: FrameState,
    agent: DQNAgent,
    *,
    can_boost: bool,
    boost_level: float,
    done: bool,
    context: BrowserStepContext | None = None,
    train_loss: float | None = None,
    train_ms: float | None = None,
  ) -> None:
    with self._lock:
      self._accumulate_step(
        action,
        reward,
        frame_state,
        can_boost=can_boost,
        boost_level=boost_level,
        done=done,
        train_loss=train_loss,
        train_ms=train_ms,
        context=context,
      )
      total_steps = (
        context.total_steps
        if context is not None and context.total_steps is not None
        else agent.metrics.total_steps
      )
      if (
        self.phase == "browser"
        and self._browser_detail_interval > 0
        and total_steps > 0
        and total_steps % self._browser_detail_interval == 0
      ):
        self._write_browser_detail(
          total_steps=total_steps,
          action=action,
          reward=reward,
          frame_state=frame_state,
          can_boost=can_boost,
          boost_level=boost_level,
          done=done,
          context=context,
          agent=agent,
        )

  def record_step(
    self,
    action: int,
    reward: float,
    frame_state: FrameState,
    *,
    can_boost: bool,
    boost_level: float,
    done: bool,
    train_loss: float | None = None,
    train_ms: float | None = None,
  ) -> None:
    with self._lock:
      self._accumulate_step(
        action,
        reward,
        frame_state,
        can_boost=can_boost,
        boost_level=boost_level,
        done=done,
        train_loss=train_loss,
        train_ms=train_ms,
        context=None,
      )

  def _write_browser_detail(
    self,
    *,
    total_steps: int,
    action: int,
    reward: float,
    frame_state: FrameState,
    can_boost: bool,
    boost_level: float,
    done: bool,
    context: BrowserStepContext | None,
    agent: DQNAgent,
  ) -> None:
    parts = [
      f"STEP {total_steps}",
      f"action={ACTION_LABELS.get(action, action)}",
      f"reward={reward:.4f}",
      f"score={frame_state.score if frame_state.score is not None else '-'}",
      f"boost={boost_level:.0%}",
      f"can_boost={can_boost}",
      f"done={done}",
      f"eps={agent.epsilon:.4f}",
      f"loss={agent.metrics.loss}",
    ]
    if frame_state.orb_x is not None and frame_state.orb_y is not None:
      parts.append(f"orb=({frame_state.orb_x:.0f},{frame_state.orb_y:.0f})")
    if frame_state.nearest_platform_dx is not None:
      parts.append(f"plat_dx={frame_state.nearest_platform_dx:.0f}")
    if frame_state.nearest_platform_dy is not None:
      parts.append(f"plat_dy={frame_state.nearest_platform_dy:.0f}")
    if frame_state.target_kind is not None:
      parts.append(f"target={frame_state.target_kind}")
    if frame_state.target_dx is not None:
      parts.append(f"target_dx={frame_state.target_dx:.3f}")
    if frame_state.in_menu:
      parts.append("in_menu=True")
    if context is not None:
      if context.score_velocity is not None:
        parts.append(f"score_vel={context.score_velocity:+.1f}")
      if context.episode_reward is not None:
        parts.append(f"ep_reward={context.episode_reward:.3f}")
      if context.loop_hz is not None:
        parts.append(f"loop_hz={context.loop_hz:.1f}")
      if context.step_ms is not None:
        parts.append(f"step_ms={context.step_ms:.1f}")
    self._write(" | ".join(parts))

  def record_batch(
    self,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    can_boost: np.ndarray,
    boost_levels: np.ndarray,
    scores: list[float | None],
    *,
    train_loss: float | None = None,
    train_ms: float | None = None,
  ) -> None:
    with self._lock:
      prev_scores = self._last_score
      for index, action in enumerate(actions):
        score = scores[index]
        frame_state = FrameState(
          score=int(score) if score is not None else None,
          boost_level=float(boost_levels[index]),
          can_boost=bool(can_boost[index]),
          game_over=bool(dones[index]),
        )
        self._accumulate_step(
          int(action),
          float(rewards[index]),
          frame_state,
          can_boost=bool(can_boost[index]),
          boost_level=float(boost_levels[index]),
          done=bool(dones[index]),
          train_loss=train_loss if index == 0 else None,
          train_ms=train_ms if index == 0 else None,
        )
        if score is not None:
          prev_scores = score

  def maybe_flush(self, agent: DQNAgent, total_steps: int) -> None:
    with self._lock:
      if total_steps > 0 and total_steps % self._interval == 0:
        self._flush_window(agent=agent, total_steps=total_steps)

  def log_episode_end(
    self,
    agent: DQNAgent,
    episode: int,
    episode_reward: float,
    episode_max_score: float,
    *,
    env_id: int | None = None,
    episode_steps: int | None = None,
  ) -> None:
    with self._lock:
      self._episode_index = episode
      self._recent_episode_scores.append(episode_max_score)
      steps = episode_steps if episode_steps is not None else self._episode.steps
      self._recent_episode_lengths.append(steps)
      self._recent_episode_rewards.append(episode_reward)
      label = f"env={env_id} " if env_id is not None else ""
      if episode_steps is None:
        action_hist = self._format_action_hist(self._episode.action_counts)
        jump_depleted = self._episode.jump_while_depleted
      else:
        action_hist = "(parallel env — see window summary for actions)"
        jump_depleted = 0
      self._write(
        "\n".join(
          [
            f"EPISODE END {label}ep={episode} "
            f"len={steps} "
            f"reward={episode_reward:.3f} "
            f"max_score={episode_max_score:.0f} "
            f"epsilon={agent.epsilon:.4f}",
            f"  actions: {action_hist}",
            f"  jump_while_depleted={jump_depleted}",
            *self._browser_episode_extras(agent, episode_max_score, episode_reward),
          ]
        )
      )
      self._episode.clear()
      if env_id is None:
        self._last_score = None

  def _browser_episode_extras(
    self,
    agent: DQNAgent,
    episode_max_score: float,
    episode_reward: float,
  ) -> list[str]:
    if self.phase != "browser":
      return []
    progress = agent.progress
    lines: list[str] = []
    if progress.baseline_score is not None:
      lines.append(
        f"  vs_baseline_score={episode_max_score - progress.baseline_score:+.0f}"
      )
    if progress.baseline_reward is not None:
      lines.append(
        f"  vs_baseline_reward={episode_reward - progress.baseline_reward:+.3f}"
      )
    lines.append(f"  best_score={progress.best_score:.0f}")
    return lines

  def _accumulate_step(
    self,
    action: int,
    reward: float,
    frame_state: FrameState,
    *,
    can_boost: bool,
    boost_level: float,
    done: bool,
    train_loss: float | None,
    train_ms: float | None,
    context: BrowserStepContext | None = None,
  ) -> None:
    self._window.steps += 1
    self._episode.steps += 1
    self._window.action_counts[action] += 1
    self._episode.action_counts[action] += 1
    self._window.reward_sum += reward
    self._window.reward_count += 1
    self._window.reward_min = min(self._window.reward_min, reward)
    self._window.reward_max = max(self._window.reward_max, reward)
    self._episode.reward += reward
    self._window.boost_level_sum += boost_level
    if not can_boost:
      self._window.depleted_steps += 1
    if action in JUMP_ACTIONS:
      self._window.jump_count += 1
      if not can_boost:
        self._window.jump_while_depleted += 1
        self._episode.jump_while_depleted += 1
    if frame_state.score is not None:
      score = float(frame_state.score)
      self._episode.max_score = max(self._episode.max_score, score)
      if self._last_score is not None:
        delta = score - self._last_score
        if delta != 0:
          self._window.score_delta_sum += delta
          self._window.score_delta_count += 1
      self._last_score = score
    if done:
      self._window.deaths += 1
    if train_loss is not None:
      self._window.train_losses.append(train_loss)
    if train_ms is not None:
      self._window.train_ms.append(train_ms)
    if context is not None:
      if context.loop_hz is not None:
        self._window.loop_hz_sum += context.loop_hz
        self._window.loop_hz_count += 1
      if context.step_ms is not None:
        self._window.step_ms_sum += context.step_ms
        self._window.step_ms_count += 1
    if frame_state.nearest_platform_dx is not None and frame_state.nearest_platform_dy is not None:
      self._window.platform_dx_sum += abs(frame_state.nearest_platform_dx)
      self._window.platform_dy_sum += frame_state.nearest_platform_dy
      self._window.platform_samples += 1
    if frame_state.in_menu:
      self._window.in_menu_steps += 1

  def _flush_window(self, agent: DQNAgent | None = None, total_steps: int = 0, *, force: bool = False) -> None:
    if self._window.steps == 0 and not force:
      return
    elapsed = time.perf_counter() - self._started
    sps = total_steps / max(elapsed, 1e-6) if total_steps else 0.0
    lines = [
      "-" * 72,
      f"SUMMARY step={total_steps} window_steps={self._window.steps} "
      f"elapsed={elapsed:.1f}s sps={sps:.0f}",
    ]
    if agent is not None:
      progress = agent.progress
      lines.extend(
        [
          f"  epsilon={agent.epsilon:.4f} replay={len(agent.replay)} "
          f"demo_replay={len(agent.demo_replay)} sim_replay={len(agent.sim_replay)}",
          f"  best_score={progress.best_score:.0f} "
          f"recent_avg_score={self._fmt_optional(progress.recent_avg_score)} "
          f"recent_avg_reward={self._fmt_optional(progress.recent_avg_reward)}",
          f"  episodes_completed={progress.episodes_completed}",
        ]
      )
      if total_steps % max(1, self.config.training.log_weight_norm_every) == 0:
        norm = agent.weight_norm()
        delta = (
          norm - self._last_weight_norm
          if self._last_weight_norm is not None
          else 0.0
        )
        lines.append(f"  weight_norm={norm:.4f} delta_since_last={delta:.6f}")
        self._last_weight_norm = norm
    lines.append(f"  reward mean={self._mean_reward():.4f} min={self._fmt_reward_min()} max={self._window.reward_max:.4f}")
    if self._window.score_delta_count:
      lines.append(
        f"  score_delta mean={self._window.score_delta_sum / self._window.score_delta_count:.3f} "
        f"count={self._window.score_delta_count}"
      )
    else:
      lines.append("  score_delta mean=0.000 count=0")
    lines.append(f"  actions: {self._format_action_hist(self._window.action_counts)}")
    if self._window.steps:
      jump_pct = 100.0 * self._window.jump_count / self._window.steps
      depleted_pct = 100.0 * self._window.depleted_steps / self._window.steps
      lines.append(
        f"  jump_rate={jump_pct:.1f}% depleted_steps={depleted_pct:.1f}% "
        f"jump_while_depleted={self._window.jump_while_depleted}"
      )
      lines.append(
        f"  mean_boost={self._window.boost_level_sum / self._window.steps:.2%} "
        f"deaths_in_window={self._window.deaths}"
      )
    if self.phase == "browser" and self._window.steps:
      if self._window.loop_hz_count:
        lines.append(
          f"  loop_hz_mean={self._window.loop_hz_sum / self._window.loop_hz_count:.1f}"
        )
      if self._window.step_ms_count:
        lines.append(
          f"  step_ms_mean={self._window.step_ms_sum / self._window.step_ms_count:.1f}"
        )
      if self._window.platform_samples:
        lines.append(
          f"  platform |dx|_mean={self._window.platform_dx_sum / self._window.platform_samples:.1f} "
          f"dy_mean={self._window.platform_dy_sum / self._window.platform_samples:.1f}"
        )
      if self._window.in_menu_steps:
        lines.append(f"  in_menu_steps={self._window.in_menu_steps}")
    if self._window.train_losses:
      losses = self._window.train_losses
      lines.append(
        f"  train_loss last={losses[-1]:.4f} mean={sum(losses) / len(losses):.4f} "
        f"n={len(losses)}"
      )
      if self._window.train_ms:
        ms = self._window.train_ms
        lines.append(f"  train_ms mean={sum(ms) / len(ms):.1f}")
    if self._recent_episode_scores:
      scores = list(self._recent_episode_scores)
      lengths = list(self._recent_episode_lengths)
      lines.append(
        f"  last_{len(scores)}_episodes score_mean={sum(scores) / len(scores):.1f} "
        f"score_max={max(scores):.0f} score_min={min(scores):.0f} "
        f"len_mean={sum(lengths) / len(lengths):.1f}"
      )
    diagnoses = self._diagnose(agent)
    if diagnoses:
      lines.append("  DIAGNOSIS:")
      for item in diagnoses:
        lines.append(f"    - {item}")
    lines.append("-" * 72)
    self._write("\n".join(lines))
    self._window.clear()

  def _diagnose(self, agent: DQNAgent | None) -> list[str]:
    hints: list[str] = []
    if agent is None:
      return hints
    progress = agent.progress
    if self._window.train_losses and max(self._window.train_losses) > 50.0:
      hints.append(
        "TD loss > 50 — Q-targets may be mis-scaled or learning unstable."
      )
    if len(agent.replay) < self.config.training.min_replay_size:
      hints.append(
        f"Replay still warming up ({len(agent.replay)} < "
        f"{self.config.training.min_replay_size})."
      )
    if self._window.jump_while_depleted > 0:
      hints.append(
        "Jump actions taken while boost depleted — masking may be failing."
      )
    if self._window.steps >= 100:
      noop_rate = self._window.action_counts[0] / self._window.steps
      if noop_rate > 0.6:
        hints.append(
          f"High noop rate ({noop_rate:.0%}) — agent may be stuck idle."
        )
      jump_rate = self._window.jump_count / self._window.steps
      if jump_rate > 0.45:
        hints.append(
          f"High jump rate ({jump_rate:.0%}) — possible boost spam."
        )
    if len(self._recent_episode_scores) >= 10:
      recent = list(self._recent_episode_scores)[-10:]
      if max(recent) - min(recent) < 50 and progress.best_score > 0:
        hints.append(
          f"Score plateau in last 10 episodes (range {max(recent) - min(recent):.0f}) "
          f"— policy may be in a local optimum."
        )
      if sum(recent) / len(recent) < progress.best_score * 0.5:
        hints.append(
          "Recent avg score well below best — regression or high exploration noise."
        )
    if agent.epsilon <= self.config.training.epsilon_end + 0.01:
      hints.append(
        "Epsilon near floor — limited exploration; need demos or epsilon restart."
      )
    if self._window.score_delta_count == 0 and self._window.steps >= 200:
      hints.append(
        "No score increases in window — agent not climbing (survival-only policy?)."
      )
    if (
      self._last_weight_norm is not None
      and abs(self._last_weight_norm) > 0
      and self._window.train_losses
      and len(self._window.train_losses) >= 5
    ):
      if all(loss > 10 for loss in self._window.train_losses[-5:]):
        hints.append(
          "Sustained high loss with training active — check rewards and BC warm-start."
        )
    if self.phase == "browser" and self._window.in_menu_steps > 5:
      hints.append(
        f"Game menu detected {self._window.in_menu_steps} steps — score/reward may be invalid."
      )
    if self.phase == "browser" and self._window.loop_hz_count:
      mean_hz = self._window.loop_hz_sum / self._window.loop_hz_count
      if mean_hz < 15:
        hints.append(
          f"Low browser loop rate ({mean_hz:.1f} Hz) — sample throughput bottleneck."
        )
    if not hints and self._window.steps >= 500:
      hints.append(
        "No automatic flags — inspect action histogram and episode score trend manually."
      )
    return hints

  def _mean_reward(self) -> float:
    if self._window.reward_count == 0:
      return 0.0
    return self._window.reward_sum / self._window.reward_count

  def _fmt_reward_min(self) -> str:
    if self._window.reward_min == float("inf"):
      return "-"
    return f"{self._window.reward_min:.4f}"

  @staticmethod
  def _format_action_hist(counts: np.ndarray) -> str:
    total = int(counts.sum())
    if total == 0:
      return "(none)"
    parts = []
    for action, count in enumerate(counts):
      if count == 0:
        continue
      pct = 100.0 * count / total
      parts.append(f"{ACTION_LABELS.get(action, action)}={pct:.0f}%")
    return " ".join(parts)

  @staticmethod
  def _fmt_optional(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "-"

  def _write(self, text: str) -> None:
    self._file.write(text + "\n")
    self._file.flush()
