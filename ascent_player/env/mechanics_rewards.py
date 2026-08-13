"""Reward shaping aligned with ASCENT game mechanics (score-rules.js)."""

from __future__ import annotations

from dataclasses import dataclass

from ascent_player.config import MechanicsRewardConfig
from ascent_player.env.score_rules import MAX_COMBO_MULT, combo_multiplier, live_style
from ascent_player.env.state_detector import JUMP_ACTIONS, FrameState

LEFT_ACTIONS = frozenset({1, 4})
RIGHT_ACTIONS = frozenset({2, 5})


@dataclass(slots=True)
class MechanicsRewardTracker:
    config: MechanicsRewardConfig
    curriculum_stage: str = "M0"
    last_state: FrameState | None = None
    episode_steps: int = 0
    milestones_hit: set[int] | None = None
    height_milestones_hit: set[int] | None = None
    last_steer_dir: int = 0
    persist_steer_dir: int = 0
    persist_steer_steps: int = 0
    steps_since_landing: int = 999

    def set_curriculum_stage(self, stage: str) -> None:
        valid = {f"M{i}" for i in range(7)}
        self.curriculum_stage = stage if stage in valid else "M0"

    def reset(self) -> None:
        self.last_state = None
        self.episode_steps = 0
        self.milestones_hit = set()
        self.height_milestones_hit = set()
        self.last_steer_dir = 0
        self.persist_steer_dir = 0
        self.persist_steer_steps = 0
        self.steps_since_landing = 999

    def compute(self, state: FrameState, action: int) -> float:
        self.episode_steps += 1
        reward = self.config.survival
        previous = self.last_state

        if previous is not None:
            landed = state.bounces > previous.bounces
            if landed:
                self.steps_since_landing = 0
            else:
                self.steps_since_landing += 1

            reward += self._height_reward(previous, state)
            reward += self._landing_reward(previous, state)
            reward += self._falling_penalty(previous, state)
            reward += self._steer_reward(previous, state, action)
            reward += self._boost_economy_reward(previous, state, action)
            # Score only pays well when combo (platform skill) is active.
            reward += self._score_reward(previous, state) * self._early_score_scale()
            reward += self._height_milestone_reward(state)
            # Approach from M0 — horizontal closing is core survival, not a later skill.
            reward += self._approach_reward(previous, state)
            # Combo from M0: multiplier growth is the real skill signal.
            reward += self._combo_reward(previous, state)
            if self._stage_at_least("M3"):
                reward += self._booster_reward(previous, state)
            if self._stage_at_least("M4"):
                reward += self._milestone_reward(state)

        if state.game_over:
            reward += self.config.death
            if self.episode_steps < self.config.early_death_steps:
                reward += self.config.early_death_penalty
            # Floor-hardening + credit real score skill so good runs stay positive.
            score = float(state.score or 0.0)
            if score < 600:
                reward += -2.0
            elif score < 1000:
                reward += -1.0
            elif score >= 10000:
                reward += 0.5
            # Terminal score credit: prevent "1197 score / −40 reward" polarity flips.
            if score >= 800:
                reward += min(score / 400.0, 5.0)
            if previous is not None and previous.combo >= 4:
                reward += 0.35 * combo_multiplier(previous.combo)

        self.last_state = state
        return float(
            max(
                -self.config.reward_clip,
                min(self.config.reward_clip, reward),
            )
        )

    def _stage_at_least(self, stage: str) -> bool:
        return int(self.curriculum_stage[1:]) >= int(stage[1:])

    def _early_score_scale(self) -> float:
        """Score shaping ramps with curriculum, but stays combo-gated inside."""
        if self._stage_at_least("M4"):
            return 1.0
        if self._stage_at_least("M2"):
            return 0.85
        return 0.65

    def _combo_skill_scale(self, state: FrameState) -> float:
        """Map combo multiplier → [no_combo_scale, 1.0]."""
        mult = combo_multiplier(state.combo)
        # mult in [1, MAX_COMBO_MULT]
        t = (mult - 1.0) / max(MAX_COMBO_MULT - 1.0, 1e-6)
        floor = float(self.config.score_no_combo_scale)
        return floor + (1.0 - floor) * min(max(t, 0.0), 1.0)

    def _height_reward(self, previous: FrameState, state: FrameState) -> float:
        delta = state.height - previous.height
        if delta <= 0:
            return 0.0
        raw = delta * self.config.height_gain / 5.0
        # Pure rocket height without pads is cheap score in-game — barely reward it.
        scale = float(self.config.height_no_combo_scale)
        if state.combo > 0:
            scale = max(scale, self._combo_skill_scale(state))
        if self.steps_since_landing > 40 and state.bounces <= 0:
            scale *= 0.35
        return raw * scale

    def _landing_reward(self, previous: FrameState, state: FrameState) -> float:
        if state.bounces <= previous.bounces:
            return 0.0
        # Landing is the primary skill event; scale up as combo (multiplier) grows.
        mult = combo_multiplier(max(state.combo, 1))
        return self.config.platform_land * mult

    def _falling_penalty(self, previous: FrameState, state: FrameState) -> float:
        if state.orb_vy is None or previous.orb_vy is None:
            return 0.0
        if state.orb_vy < -0.35 and state.orb_y is not None and state.orb_y < 0.28:
            return self.config.falling_penalty
        return 0.0

    def _steer_dir(self, action: int) -> int:
        if action in LEFT_ACTIONS:
            return -1
        if action in RIGHT_ACTIONS:
            return 1
        return 0

    def _oscillation_exempt(self, state: FrameState) -> bool:
        """True when left/right flips are legitimate recoveries, not jitter."""
        if state.miss_risk or state.falling or state.landing_window:
            return True
        if state.booster_type == "drag":
            return True
        if state.hazard_dx is not None and abs(float(state.hazard_dx)) < 0.28:
            return True
        return False

    def _anti_oscillation_reward(self, state: FrameState, action: int) -> float:
        """Penalize rapid L↔R flips; small bonus for holding a steer."""
        steer_dir = self._steer_dir(action)
        reward = 0.0
        if steer_dir == 0:
            # Brief noop does not wipe last direction (still detects flip after pause).
            self.persist_steer_steps = 0
            return reward

        exempt = self._oscillation_exempt(state)
        if (
            not exempt
            and self.last_steer_dir != 0
            and steer_dir == -self.last_steer_dir
        ):
            reward += float(self.config.direction_flip_penalty)
            self.persist_steer_dir = steer_dir
            self.persist_steer_steps = 1
        elif steer_dir == self.persist_steer_dir:
            self.persist_steer_steps += 1
            need = max(1, int(self.config.direction_persistence_steps))
            if self.persist_steer_steps == need:
                reward += float(self.config.direction_persistence_bonus)
        else:
            self.persist_steer_dir = steer_dir
            self.persist_steer_steps = 1

        self.last_steer_dir = steer_dir
        return reward

    def _steer_reward(self, previous: FrameState, state: FrameState, action: int) -> float:
        # While falling, reward steering toward the pad below — not above/booster targets.
        if state.falling or state.landing_window or state.miss_risk:
            dx = state.nearest_platform_dx
        else:
            dx = state.target_dx if state.target_dx is not None else state.nearest_platform_dx
        reward = 0.0
        if dx is not None:
            gain = self.config.steer_gain
            if state.falling or state.miss_risk:
                gain *= 2.0
            if dx < -0.012:
                if action in LEFT_ACTIONS:
                    reward += gain * min(abs(dx) * 6.0, 2.0)
                elif action in RIGHT_ACTIONS:
                    reward += self.config.wrong_way_penalty
            elif dx > 0.012:
                if action in RIGHT_ACTIONS:
                    reward += gain * min(abs(dx) * 6.0, 2.0)
                elif action in LEFT_ACTIONS:
                    reward += self.config.wrong_way_penalty
            else:
                reward += self.config.aligned_bonus
            # Falling with a clear pad offset but choosing noop is almost as bad as wrong-way.
            if (
                (state.falling or state.landing_window or state.miss_risk)
                and action == 0
                and abs(dx) > 0.05
            ):
                reward += self.config.wrong_way_penalty * 0.5
            elif state.miss_risk and action == 0:
                reward += self.config.wrong_way_penalty * 0.5

        reward += self._anti_oscillation_reward(state, action)
        return reward

    def _approach_reward(self, previous: FrameState, state: FrameState) -> float:
        if state.falling or state.landing_window or previous.falling:
            prev_dx = abs(previous.nearest_platform_dx or 1.0)
            curr_dx = abs(state.nearest_platform_dx or 1.0)
        else:
            prev_dx = abs(previous.target_dx or previous.nearest_platform_dx or 1.0)
            curr_dx = abs(state.target_dx or state.nearest_platform_dx or 1.0)
        if curr_dx < prev_dx:
            return 0.06
        if curr_dx > prev_dx + 0.02:
            return -0.05
        return 0.0

    def _is_rising(self, state: FrameState) -> bool:
        if state.rising:
            return True
        vy = state.orb_vy
        if vy is None:
            return False
        # Hook may export normalized (~1) or pixel vy (~hundreds).
        return vy > 0.15 if abs(vy) <= 2.0 else vy > 40.0

    def _needs_boost_soon(self, state: FrameState) -> bool:
        """True when a jump/boost would be useful shortly (not idle topping-up)."""
        if state.boost_useful is True:
            return True
        gap = float(state.nearest_platform_dy or 0.0)
        if (state.falling or state.landing_window) and gap > 0.15:
            return True
        above = float(state.nearest_platform_above_dy or 0.0)
        if self._is_rising(state) and above > 0.08:
            return True
        return False

    def _wait_for_energy_reward(
        self,
        previous: FrameState,
        state: FrameState,
        action: int,
    ) -> float:
        """Reward patience while energy recharges — only when a boost is useful soon.

        Does not pay for idling far off-line (steer first) or when energy is full.
        Disable via MechanicsRewardConfig.wait_for_energy_enabled / zero bonuses.
        """
        if not bool(getattr(self.config, "wait_for_energy_enabled", True)):
            return 0.0
        if previous.can_boost:
            return 0.0
        jumped = action in JUMP_ACTIONS
        if jumped:
            return 0.0  # empty_boost_penalty already handles futile jumps

        needs = self._needs_boost_soon(previous) or self._needs_boost_soon(state)
        if not needs:
            return 0.0

        if previous.falling or previous.landing_window or previous.miss_risk:
            dx = previous.nearest_platform_dx
        else:
            dx = (
                previous.target_dx
                if previous.target_dx is not None
                else previous.nearest_platform_dx
            )
        max_dx = float(
            getattr(self.config, "wait_for_energy_max_abs_dx", 0.12) or 0.12
        )
        on_line = dx is None or abs(float(dx)) <= max_dx
        reward = 0.0
        if on_line:
            reward += float(getattr(self.config, "wait_for_energy_bonus", 0.0) or 0.0)
        # Credit actual recharge while being patient (even if still closing dx).
        if float(state.boost_level) > float(previous.boost_level) + 1e-4:
            reward += float(
                getattr(self.config, "wait_for_energy_recharge_bonus", 0.0) or 0.0
            )
        return reward

    def _boost_economy_reward(
        self,
        previous: FrameState,
        state: FrameState,
        action: int,
    ) -> float:
        reward = 0.0
        jumped = action in JUMP_ACTIONS
        boost_drop = float(previous.boost_level) - float(state.boost_level)
        if jumped and previous.can_boost and not state.can_boost:
            reward += self.config.boost_spent
            if self._is_rising(previous):
                reward += self.config.wasted_boost_penalty
        if jumped and not previous.can_boost:
            reward += self.config.empty_boost_penalty

        # Rocket spam: jump while rising / not a useful gap-close.
        if jumped and self._is_rising(previous) and not state.boost_useful:
            reward += self.config.boost_spam_penalty

        # Early fuel dump before any platform contact — fake height without skill.
        # Scoped to pre-first-bounce so mid-climb recovery jumps stay allowed.
        if (
            jumped
            and self.episode_steps <= self.config.early_boost_dump_steps
            and previous.bounces <= 0
            and state.bounces <= 0
        ):
            min_drop = float(
                getattr(self.config, "early_boost_dump_min_drop", 0.08) or 0.08
            )
            dumping = boost_drop >= min_drop or (
                self._is_rising(previous) and not state.boost_useful
            )
            if dumping:
                # Scale with fuel spent so a full dump hurts more than a tap.
                scale = max(1.0, float(boost_drop) / max(min_drop, 1e-3))
                reward += self.config.early_boost_dump_penalty * min(scale, 3.0)

        vy = state.orb_vy or 0.0
        gap = state.nearest_platform_dy or 0.0
        falling_vy = vy < -0.2 if abs(vy) <= 2.0 else vy < -40.0
        if jumped and falling_vy and gap > 0.15:
            reward += self.config.timed_boost_bonus
        if jumped and state.boost_useful:
            reward += self.config.timed_boost_bonus * 0.5

        reward += self._wait_for_energy_reward(previous, state, action)
        return reward

    def _combo_reward(self, previous: FrameState, state: FrameState) -> float:
        reward = 0.0
        if state.combo > previous.combo:
            scale = combo_multiplier(state.combo)
            reward += self.config.combo_gain * scale
        elif (
            not state.game_over
            and state.combo < previous.combo
            and previous.combo >= 3
        ):
            lost_style = live_style(bonus=previous.bonus, combo=previous.combo)
            reward += self.config.combo_break_penalty * min(lost_style / 400.0, 1.0)
        return reward

    def _booster_reward(self, previous: FrameState, state: FrameState) -> float:
        if state.bonus <= previous.bonus and state.booster_type is None:
            return 0.0
        if state.booster_type == "drag" or (
            previous.booster_type != "drag" and state.combo < previous.combo
        ):
            return self.config.drag_penalty
        if state.bonus > previous.bonus:
            return self.config.booster_collect
        return 0.0

    def _score_reward(self, previous: FrameState, state: FrameState) -> float:
        if state.score is None or previous.score is None:
            return 0.0
        delta = state.score - previous.score
        if delta <= 0:
            return 0.0
        # Without combo, score rises mostly from height/5 — do not pay full freight.
        return delta * self.config.score_gain * self._combo_skill_scale(state)

    def _milestone_reward(self, state: FrameState) -> float:
        if state.score is None or self.milestones_hit is None:
            return 0.0
        reward = 0.0
        for milestone in self.config.milestone_scores:
            if state.score >= milestone and milestone not in self.milestones_hit:
                self.milestones_hit.add(milestone)
                reward += self.config.milestone_bonus
        return reward

    def _height_milestone_reward(self, state: FrameState) -> float:
        if self.height_milestones_hit is None:
            return 0.0
        # Require at least one pad hit so pure rocket climbs do not farm milestones.
        if state.bounces <= 0:
            return 0.0
        reward = 0.0
        for milestone in self.config.height_milestones:
            if state.height >= milestone and milestone not in self.height_milestones_hit:
                self.height_milestones_hit.add(milestone)
                reward += self.config.height_milestone_bonus
        return reward
