from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path


ASCENT_URL = "http://127.0.0.1:8765/index.html"
ASCENT_HOST = "127.0.0.1"

# On KDE Wayland + NVIDIA, Chromium on native Wayland can crash kwin_wayland.
_DEFAULT_WAYLAND_CHROMIUM_ARGS = (
    "--ozone-platform=x11",
    "--disable-features=Vulkan",
)


def default_chromium_args() -> tuple[str, ...]:
    if os.environ.get("WAYLAND_DISPLAY"):
        return _DEFAULT_WAYLAND_CHROMIUM_ARGS
    return ()


class DeviceMode(str, Enum):
    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"


@dataclass(slots=True)
class BrowserConfig:
    ascent_url: str = ASCENT_URL
    host_match: str = ASCENT_HOST
    agent_mode: bool = True
    cdp_ports: tuple[int, ...] = tuple(range(9222, 9230))
    cdp_timeout_seconds: float = 0.25
    manual_cdp_url: str | None = None
    auto_launch_on_miss: bool = True
    chromium_path: str | None = None
    # Content viewport (Playwright / CSS layout). ASCENT scales sprites with
    # canvas height via gs=H/700, so keep this near ~720–900 — not full monitor.
    viewport_width: int = 1440
    viewport_height: int = 850
    # Outer Chromium window is viewport + frame pad (title bar / borders).
    # Without this, --window-size≈viewport makes the client area shorter and
    # clips bottom HUD (fuel bar / Link wallet) while sprites still look large.
    window_frame_pad_x: int = 0
    window_frame_pad_y: int = 80
    device_scale_factor: float = 1.0
    rescan_seconds: int = 5
    canvas_selector: str = "#gameCanvas"
    # Read canvas pixels via JS instead of Playwright element screenshots.
    # Element screenshots scroll into view and cause visible flicker.
    use_js_canvas_capture: bool = True
    dom_poll_interval: int = 24
    hud_poll_interval: int = 1
    # Downscale before JPEG transfer to cut CDP latency (detection uses ratios).
    capture_max_width: int = 640
    capture_max_height: int = 360
    capture_jpeg_quality: float = 0.82
    chromium_args: tuple[str, ...] = field(default_factory=default_chromium_args)
    # Pin launched Chromium onto a named monitor (EDID / kscreen name substring).
    # Empty = no pin. Default keeps the game off the main/dev screens.
    window_monitor_match: str = "ZOWIE"
    # Explicit override (x, y). When set, skips monitor name lookup.
    window_position: tuple[int, int] | None = None
    # Fixed map seed for curriculum. None = game picks a fresh seed each run.
    # Applied via CHART_TRIAL_CONFIG / __ASCENT_SET_RUN_SEED__ before play.
    run_seed: int | None = None
    lock_run_seed: bool = False
    # Raise/focus the game window after launch (helps when pinned to a side monitor).
    raise_on_launch: bool = True


@dataclass(slots=True)
class ObservationConfig:
    width: int = 84
    height: int = 84
    frame_stack: int = 4
    include_boost_channel: bool = True
    include_platform_channel: bool = True
    include_vector_state: bool = True
    vector_dim: int = 59

    @property
    def channel_count(self) -> int:
        extra = int(self.include_boost_channel) + int(self.include_platform_channel)
        return self.frame_stack + extra


@dataclass(slots=True)
class RewardConfig:
    survival: float = 0.004
    score_gain: float = 0.006
    altitude_gain: float = 0.012
    death: float = -1.0
    early_death_penalty: float = -0.5
    early_death_steps: int = 80
    falling_penalty: float = -0.03
    idle_penalty: float = -0.015
    idle_steps: int = 20
    wasted_jump_penalty: float = -0.08
    boost_gain: float = 0.04
    boost_spent: float = -0.03
    low_boost_penalty: float = -0.004
    boost_jump_threshold: float = 0.06
    boost_min_energy: float = 14.0
    empty_boost_jump_penalty: float = -0.25
    survival_step_bonus: float = 0.00005
    # Dominant: steer toward grey platforms, yellow orbs, green boosters.
    target_steer_gain: float = 0.42
    target_approach_gain: float = 0.35
    target_wrong_way_penalty: float = -0.22
    target_aligned_bonus: float = 0.10
    target_idle_penalty: float = -0.14
    direction_flip_penalty: float = -0.10
    direction_persistence_steps: int = 3
    direction_persistence_bonus: float = 0.06
    platform_align: float = 0.06
    platform_fall_weight: float = 2.5
    platform_land: float = 0.30
    combo_gain: float = 0.05
    streak_level_bonus: float = 0.55
    multiplier_gain: float = 0.20
    combo_break_penalty: float = -0.35
    off_platform_penalty: float = -0.08
    score_stagnation_steps: int = 30
    score_stagnation_penalty: float = -0.04
    milestone_scores: tuple[int, ...] = (500, 1000, 1500, 2000, 2500, 3000)
    milestone_bonus: float = 0.2
    reward_clip: float = 2.0


@dataclass(slots=True)
class MechanicsRewardConfig:
    survival: float = 0.004
    # Raw height/score are weak — game score rises from rocket height alone.
    # Skill is platform hits (combo → multiplier); those dominate the shaping.
    height_gain: float = 0.02
    platform_land: float = 1.55
    falling_penalty: float = -0.08
    steer_gain: float = 0.05
    wrong_way_penalty: float = -0.10
    aligned_bonus: float = 0.015
    # Mild anti-jitter: punish L↔R flips; reward holding a steer briefly.
    # Exempt miss_risk / falling recovery / near-hazard / drag escape.
    direction_flip_penalty: float = -0.06
    direction_persistence_steps: int = 3
    direction_persistence_bonus: float = 0.03
    boost_spent: float = -0.02
    wasted_boost_penalty: float = -0.18
    empty_boost_penalty: float = -0.25
    timed_boost_bonus: float = 0.60
    # Mild rocket-spam penalties — strong *global* values killed all jumping.
    # Early dump (pre-first-landing) can be much harsher: that is the fake-height farm.
    boost_spam_penalty: float = -0.12
    early_boost_dump_penalty: float = -0.45
    early_boost_dump_steps: int = 100
    early_boost_dump_min_drop: float = 0.08
    # Wait for energy recharge before jumping — only when a boost is useful soon
    # and the orb is roughly on-line. Set enabled=False or bonuses to 0 to disable.
    wait_for_energy_enabled: bool = True
    wait_for_energy_bonus: float = 0.08
    wait_for_energy_recharge_bonus: float = 0.05
    wait_for_energy_max_abs_dx: float = 0.12
    combo_gain: float = 0.40
    combo_break_penalty: float = -0.50
    booster_collect: float = 0.25
    drag_penalty: float = -0.30
    # Score delta: pay more once combo is active (platform skill).
    score_gain: float = 0.014
    score_no_combo_scale: float = 0.08
    height_no_combo_scale: float = 0.15
    death: float = -1.2
    early_death_penalty: float = -0.6
    early_death_steps: int = 200
    milestone_scores: tuple[int, ...] = (
        500,
        1000,
        1500,
        2000,
        3000,
        5000,
        7500,
        10000,
    )
    milestone_bonus: float = 0.35
    height_milestones: tuple[int, ...] = (
        500,
        1000,
        2500,
        5000,
        10000,
        25000,
        50000,
    )
    height_milestone_bonus: float = 0.10
    reward_clip: float = 3.5
    boost_min_energy: float = 14.0


@dataclass(slots=True)
class MechanicsCurriculumConfig:
    stage_m1_min_survival_steps: float = 400.0
    stage_m2_min_landing_rate: float = 0.40
    stage_m3_min_height: float = 350.0
    stage_m4_min_combo: float = 4.0
    stage_m5_min_score: float = 800.0
    stage_m6_min_score: float = 1500.0
    training_tier_index: int = 0
    teacher_episodes: int = 120
    teacher_warmstart_epsilon: float = 0.20
    # Phase 4: curriculum start-height when best score clears thresholds.
    start_height_gate_b: float = 3000.0
    start_height_gate_c: float = 6000.0
    start_height_max_b: float = 4_000.0
    start_height_max_c: float = 12_000.0
    start_height_prob: float = 0.25
    # Only count score above the start-height baseline for curriculum stats.
    start_height_score_credit: bool = False


@dataclass(slots=True)
class DemoConfig:
    save_dir: Path = Path("demonstrations")
    replay_multiplier: int = 2
    pretrain_steps: int = 800
    use_demos_on_start: bool = True
    min_episode_score: float = 1500.0
    high_score_weight: float = 2.0
    bc_loss_weight: float = 0.15
    hybrid_bc_every: int = 4
    steering_action_weight: float = 1.5
    demo_reingest_every_runs: int = 20
    # Keep at least this much RAM free for the OS while loading demos.
    os_memory_reserve_mb: int = 2048
    # Hard cap on unique demo transitions; None = derive from available memory.
    max_transitions: int | None = None


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 2e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.992
    replay_buffer_size: int = 50_000
    batch_size_cpu: int = 32
    # Default sized for impala_mid + mixed precision on RTX 3070-class GPUs.
    batch_size_gpu: int = 128
    # Network capacity: nature (~1.8M) | impala_mid (~8–15M) | impala_large (~20M+).
    model_variant: str = "impala_mid"
    mixed_precision: bool = True
    min_replay_size: int = 400
    target_sync_interval: int = 800
    soft_target_tau: float = 0.005
    gradient_clip_norm: float = 10.0
    use_double_dqn: bool = True
    frame_skip: int = 1
    game_fps: float = 60.0
    async_training: bool = True
    train_every_cpu: int = 4
    train_every_gpu: int = 1
    checkpoint_path: Path = Path("checkpoints/dqn_latest.keras")
    sim_checkpoint_path: Path = Path("checkpoints/sim_pretrained.keras")
    # Strongest known policy for Watch / resume (promoted on best gated eval).
    playable_checkpoint_path: Path = Path("checkpoints/best_playable.keras")
    # Browser ε=0 eval winner — preferred over sim playable for UI Watch.
    browser_best_checkpoint_path: Path = Path("checkpoints/browser_best.keras")
    auto_load_checkpoint: bool = True
    # Prefer best-eval / playable over a weak dqn_latest when starting UI or sim.
    prefer_best_checkpoint: bool = True
    sim_resume_from_best: bool = True
    sim_resume_epsilon: float = 0.18
    autosave_every_episodes: int = 1
    autosave_every_steps: int = 250
    baseline_episodes: int = 5
    sim_mode: bool = False
    sim_pretrain_steps: int = 0
    transfer_from_sim: bool = False
    transfer_learning_rate: float = 5e-5
    # Keep exploration moderate — 0.5 + frame_skip=2 destroys early landings.
    transfer_epsilon_start: float = 0.28
    transfer_epsilon_restart: float = 0.22
    transfer_demo_delay_episodes: int = 25
    transfer_plateau_episodes: int = 12
    transfer_frame_skip: int = 2
    # After sim, adapt CNN on rendered frames before browser (closes visual gap).
    transfer_visual_bridge_steps: int = 40_000
    transfer_seed_sim_replay: int = 4_000
    browser_epsilon_cap: float = 0.28
    browser_epsilon_floor: float = 0.05
    browser_plateau_epsilon_decay: float = 0.97
    score_sanity_cap: float = 20_000.0
    replay_trim_size: int = 35_000
    gpu_restart_every_runs: int = 25
    curriculum_stage_a_max: float = 800.0
    curriculum_stage_b_max: float = 1500.0
    sim_refresh_plateau_runs: int = 30
    sim_refresh_steps: int = 50_000
    target_platform_only: bool = False
    target_detection_min_samples: int = 80
    target_special_min_rate: float = 0.02
    finetune_max_seconds: int = 600
    mixed_sim_replay_ratio: float = 0.25
    sim_epsilon_end: float = 0.03
    sim_epsilon_decay: float = 0.990
    sim_epsilon_anneal_steps: int = 100_000
    sim_min_best_score: int = 1000
    sim_max_steps_multiplier: float = 2.0
    target_score: int = 10000
    device_mode: DeviceMode = DeviceMode.GPU
    watch_mode: bool = False
    # Collect-only sessions: count steps / write replay, skip TD updates.
    disable_td: bool = False
    # Blend RulePolicy into Watch/ε=0 play (applies even when training=False).
    watch_rule_prior: float = 0.0
    # When True, force RulePolicy on miss_risk / deep fall (Watch safety net).
    watch_safety_override: bool = False
    # Headless pretrain: parallel envs, batched inference.
    # Prefer rendered obs (False) for browser transfer; set True only for speed.
    sim_pretrain_envs: int = 0
    sim_pretrain_train_every: int = 32
    sim_pretrain_batch_size: int = 128
    sim_pretrain_min_replay: int = 256
    sim_fast_observations: bool = False
    # JPEG round-trip on rendered sim frames to match browser capture domain.
    sim_jpeg_augment: bool = True
    sim_jpeg_quality: float = 0.82
    # When True, run_sim_pretrain keeps config.frame_skip (bridge uses FS=2).
    sim_keep_frame_skip: bool = False
    # Promote sim_best_eval only when ε=0 mean improves and min clears this floor
    # (after the first promote, min must also beat the previous best's min).
    sim_eval_promote_min: float = 1000.0
    sim_warmstart_teacher: bool = True
    sim_warmstart_demos: bool = True
    sim_best_eval_checkpoint_path: Path = Path("checkpoints/sim_best_eval.keras")
    # Overnight: continue from dqn_latest (working), elite lives in browser_best.
    overnight_prefer_latest: bool = True
    browser_replay_path: Path = Path("checkpoints/browser_replay.pkl")
    browser_replay_max_items: int = 20_000
    skill_ledger_path: Path = Path("logs/skill_ledger.csv")
    consistency_eval_episodes: int = 20
    consistency_mean_score: float = 10000.0
    consistency_min_score: float = 7000.0
    # Rolling reliability target (ε=0 consecutive evals).
    reliability_eval_episodes: int = 100
    reliability_mean_score: float = 5000.0
    reliability_min_score: float = 1000.0
    log_dir: Path = Path("logs")
    log_interval_steps: int = 500
    log_interval_steps_sim: int = 2500
    log_browser_detail_steps: int = 100
    log_weight_norm_every: int = 5000
    use_prioritized_replay: bool = True
    n_step: int = 5
    # Minimum greedy 100-ep mean before online fine-tune is allowed in thread-BC.
    finetune_min_greedy_mean: float = 1400.0
    # Policy-collect thread prior ceiling after teacher bootstrap (objective split).
    policy_collect_thread_prior: float = 0.20
    teacher_collect_thread_prior: float = 0.75
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 100_000
    rule_prior_start: float = 0.45
    rule_prior_end: float = 0.08
    rule_prior_steps: int = 150_000
    # Seed-thread prior: prefer a stable climb corridor after successful landings.
    # Anneals separately from rule_prior; gated on landing/climb availability.
    seed_thread_enabled: bool = False
    seed_thread_prior_start: float = 0.40
    seed_thread_prior_end: float = 0.10
    seed_thread_prior_steps: int = 80_000
    seed_thread_only_when_landing: bool = True
    seed_thread_corridor: float = 0.18
    seed_thread_watch_prior: float = 0.0
    # Side checkpoint for thread-BC work that survives Watch regress restores.
    thread_bc_checkpoint_path: Path = Path("checkpoints/dqn_thread_bc.keras")
    smart_explore_boost_bias: float = 0.55
    dueling_dqn: bool = True
    # Gated ε=0 eval during sim pretrain / overnight.
    sim_eval_every_steps: int = 20_000
    sim_eval_episodes: int = 8
    sim_eval_max_steps: int = 12_000
    gate_a_score: float = 900.0
    gate_b_score: float = 2000.0
    gate_c_score: float = 5000.0
    gate_d_score: float = 10000.0
    gate_d_min_score: float = 7000.0
    # Browser overnight: keep frame skip low enough for boost timing.
    browser_frame_skip_min: int = 2
    browser_frame_skip_max: int = 3
    browser_epsilon_cap_after_gate_a: float = 0.15
    # Aux CE weight for the action-reason head (0 disables aux gradient).
    reason_aux_weight: float = 0.1
    # Skill library: deterministic routines + skill head for when to invoke them.
    skills_enabled: bool = False
    skill_exec_at_watch: bool = False
    skill_teacher_prior_start: float = 0.50
    skill_teacher_prior_end: float = 0.08
    skill_teacher_prior_steps: int = 120_000
    skill_aux_weight: float = 0.12
    # Minimum softmax margin (best - second) before Watch trusts skill over Q.
    skill_confidence_margin: float = 0.12
    # Log every N browser steps to decisions CSV (1 = every step).
    log_decision_every: int = 5
    # If > 0, only commit browser transitions from episodes reaching this score.
    replay_min_episode_score: float = 0.0
    # Separate elite store gate (can be higher than collect gate).
    elite_replay_min_episode_score: float = 0.0
    elite_max_episodes: int = 64
    elite_max_transitions_per_episode: int = 128
    # Require skill-head validation accuracy before Watch skill execution.
    skill_exec_min_accuracy: float = 0.85
    # Post-2k path: raise elite gate and re-enable skill only after A/B gain.
    post_2k_elite_gate: float = 3000.0
    skill_reenable_min_accuracy: float = 0.70
    skill_accuracy_path: Path = Path("checkpoints/skill_head_accuracy.json")
    # When True, always persist browser_replay.pkl at session end.
    force_save_browser_replay: bool = False
    # When True, do not load checkpoints/browser_replay.pkl at browser session start.
    skip_browser_replay_load: bool = False


@dataclass(slots=True)
class AppConfig:
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    mechanics_reward: MechanicsRewardConfig = field(default_factory=MechanicsRewardConfig)
    mechanics_curriculum: MechanicsCurriculumConfig = field(
        default_factory=MechanicsCurriculumConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)

    @property
    def action_count(self) -> int:
        return 6
