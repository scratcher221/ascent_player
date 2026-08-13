from __future__ import annotations

from pathlib import Path

import numpy as np

from ascent_player.agent.checkpoint import (
    LoadResult,
    checkpoint_exists,
    load_progress,
    prefer_checkpoint,
    save_progress,
)
from ascent_player.agent.progress_sanitize import sanitize_browser_progress
from ascent_player.agent.replay_buffer import ReplayBuffer


class CheckpointIOMixin:
    """Save/load, architecture migrate, and checkpoint promotion."""

    def maybe_autosave(self, *, force: bool = False) -> bool:
        steps = self.metrics.total_steps
        every_steps = max(1, self.config.training.autosave_every_steps)
        if not force and (steps - self._last_autosave_steps) < every_steps:
            return False
        self.save()
        self._last_autosave_steps = steps
        return True

    def resolve_best_checkpoint(self) -> Path | None:
        training = self.config.training
        # Browser-adapted weights beat sim-only playable for Watch / UI.
        if not training.sim_mode and checkpoint_exists(
            training.browser_best_checkpoint_path
        ):
            return training.browser_best_checkpoint_path
        for path in (
            training.playable_checkpoint_path,
            training.sim_best_eval_checkpoint_path,
        ):
            if checkpoint_exists(path):
                return path
        return prefer_checkpoint(
            training.sim_checkpoint_path,
            training.checkpoint_path,
        )

    def promote_browser_best(self) -> Path:
        """Persist current weights as the browser Watch baseline."""
        path = self.save(self.config.training.browser_best_checkpoint_path)
        self.save(self.config.training.playable_checkpoint_path)
        self.save(self.config.training.checkpoint_path)
        return path

    def promote_playable_checkpoint(self, source: Path | None = None) -> Path | None:
        """Copy the strongest (or given) weights into playable + UI checkpoint slots."""
        training = self.config.training
        source = source or self.resolve_best_checkpoint()
        if source is None or not self.load(source):
            return None
        playable = self.save(training.playable_checkpoint_path)
        self.save(training.checkpoint_path)
        return playable

    def try_autoload(self) -> LoadResult:
        target = self.config.training.checkpoint_path
        if self.config.training.transfer_from_sim:
            sim_path = prefer_checkpoint(
                self.config.training.sim_best_eval_checkpoint_path,
                self.config.training.sim_checkpoint_path,
                self.config.training.playable_checkpoint_path,
            ) or self.config.training.sim_checkpoint_path
            if checkpoint_exists(sim_path) and self.load(sim_path):
                self.prepare_transfer_from_sim()
                self._promote_replay_for_mixed_transfer()
                message = (
                    f"Loaded sim pretrain from {sim_path.name} — "
                    f"fine-tuning with ε={self.epsilon:.2f}, "
                    f"sim_replay={len(self.sim_replay)}"
                )
                return LoadResult(True, message, self.progress)
        if not self.config.training.auto_load_checkpoint:
            return LoadResult(False, "Auto-load disabled — starting from scratch.")

        preferred = target
        if self.config.training.overnight_prefer_latest and not self.config.training.sim_mode:
            # Overnight working weights: continue dqn_latest; elite is browser_best.
            if checkpoint_exists(target):
                preferred = target
            elif self.config.training.prefer_best_checkpoint:
                best = self.resolve_best_checkpoint()
                if best is not None:
                    preferred = best
        elif self.config.training.prefer_best_checkpoint:
            best = self.resolve_best_checkpoint()
            if best is not None:
                preferred = best

        if not checkpoint_exists(preferred):
            return LoadResult(False, "No checkpoint found — starting from scratch.")
        if not self.load(preferred):
            detail = self._last_load_error or "incompatible"
            return LoadResult(
                False,
                f"Checkpoint load failed ({detail}) — starting from scratch.",
            )
        # Keep UI default path aligned with the strongest weights we just loaded.
        # Overnight mode: do NOT overwrite dqn_latest with browser_best.
        if (
            preferred != target
            and not self.config.training.overnight_prefer_latest
        ):
            try:
                self.save(target)
                self.save(self.config.training.playable_checkpoint_path)
            except Exception as exc:
                print(f"Could not promote preferred checkpoint: {exc}")
        message = (
            f"Resumed from {preferred.name} — "
            f"{self.progress.episodes_completed} episodes, "
            f"{self.progress.total_steps:,} steps, ε={self.progress.epsilon:.3f}"
        )
        if self.progress.has_baseline:
            message += (
                f" | baseline reward {self.progress.baseline_reward:.1f}, "
                f"score {self.progress.baseline_score:.1f}"
            )
        message += f" | best score {self.progress.best_score:.0f}"
        return LoadResult(True, message, self.progress)

    @staticmethod
    def weights_sidecar_path(keras_path: Path) -> Path:
        return keras_path.with_name(f"{keras_path.stem}.weights.h5")

    def save(self, path: Path | None = None) -> Path:
        target = path or self.config.training.checkpoint_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self.progress.total_steps = self.metrics.total_steps
        self.progress.epsilon = self.epsilon
        weights_path = self.weights_sidecar_path(target)
        with self.tf.device(self.device_info.training_device):
            self.online.save_weights(weights_path)
            try:
                self.online.save(target)
            except Exception as exc:
                print(f"Full-model save skipped ({exc}); weights saved to {weights_path.name}")
        # Meta always keyed off the .keras path so resume finds it next to weights.
        save_progress(target, self.progress)
        return target

    def save_sim_checkpoint(self) -> Path:
        return self.save(self.config.training.sim_checkpoint_path)

    def _apply_loaded_progress(self, target: Path) -> None:
        progress = load_progress(target)
        if progress is None:
            weights_meta = self.weights_sidecar_path(target)
            progress = load_progress(weights_meta)
        if progress is None:
            return
        if progress.best_score <= 0 and progress.recent_scores:
            progress.best_score = max(progress.recent_scores)
        if not self.config.training.sim_mode:
            progress, sanitize_notes = sanitize_browser_progress(
                progress,
                score_cap=self.config.training.score_sanity_cap,
                epsilon_cap=self.config.training.browser_epsilon_cap,
            )
            for note in sanitize_notes:
                print(f"Progress sanitize: {note}")
        elif progress.baseline_score is not None:
            progress.best_score = max(progress.best_score, progress.baseline_score)
        self.progress = progress
        self.epsilon = progress.epsilon
        if not self.config.training.sim_mode:
            self._cap_browser_epsilon()
            # Strong checkpoints: keep exploration modest, but respect floor/cap
            # (hardcoding 0.06 fought fine-tune floors and stalled map learning).
            if progress.best_score >= self.config.training.gate_b_score:
                play_eps = min(
                    self.epsilon,
                    self.config.training.browser_epsilon_cap_after_gate_a,
                )
                play_eps = max(
                    self.config.training.browser_epsilon_floor,
                    min(play_eps, self.config.training.browser_epsilon_cap),
                )
                if abs(play_eps - self.epsilon) > 1e-6:
                    print(
                        f"Progress sanitize: epsilon {self.epsilon:.3f}→{play_eps:.3f} "
                        f"(strong checkpoint)"
                    )
                self.epsilon = play_eps
        self.progress.epsilon = self.epsilon
        self.metrics.epsilon = self.epsilon
        self.metrics.total_steps = progress.total_steps
        self._last_autosave_steps = progress.total_steps
        self._baseline_samples = []
        # Continue / seed-thread sessions: anneal priors from load time, not lifetime steps.
        if getattr(self.config.training, "seed_thread_enabled", False) or (
            float(getattr(self.config.training, "rule_prior_start", 0.0) or 0.0) > 0
            and not self.config.training.sim_mode
        ):
            self.reset_prior_anneal_origin()

    def _merge_weight_arrays(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
        """Copy overlapping slices when shapes differ (channel/vector migrate)."""
        if tuple(src.shape) == tuple(dst.shape):
            return src
        if src.ndim != dst.ndim:
            return None
        merged = np.array(dst, copy=True, dtype=np.float32)
        src = np.asarray(src, dtype=np.float32)
        if src.ndim == 1:
            n = min(src.shape[0], dst.shape[0])
            merged[:n] = src[:n]
            return merged
        if src.ndim == 2:
            r = min(src.shape[0], dst.shape[0])
            c = min(src.shape[1], dst.shape[1])
            merged[:r, :c] = src[:r, :c]
            return merged
        if src.ndim == 4:
            # Conv2D kernel: H, W, in_channels, out_channels
            h = min(src.shape[0], dst.shape[0])
            w = min(src.shape[1], dst.shape[1])
            cin = min(src.shape[2], dst.shape[2])
            cout = min(src.shape[3], dst.shape[3])
            merged[:h, :w, :cin, :cout] = src[:h, :w, :cin, :cout]
            return merged
        return None

    def _try_set_layer_weights(self, dest, src_weights) -> bool:
        dst_weights = dest.get_weights()
        if not src_weights or not dst_weights or len(src_weights) != len(dst_weights):
            return False
        merged: list[np.ndarray] = []
        for src, dst in zip(src_weights, dst_weights, strict=True):
            piece = self._merge_weight_arrays(np.asarray(src), np.asarray(dst))
            if piece is None:
                return False
            merged.append(piece)
        try:
            dest.set_weights(merged)
            return True
        except Exception:
            return False

    def _copy_compatible_weights(self, loaded) -> int:
        """Copy layers by name, then by Conv2D/Dense order (legacy auto-names)."""
        copied = 0
        online_by_name = {layer.name: layer for layer in self.online.layers}
        used_dest: set[int] = set()

        for layer in loaded.layers:
            dest = online_by_name.get(layer.name)
            if dest is None:
                continue
            src_w = layer.get_weights()
            if not src_w:
                continue
            if self._try_set_layer_weights(dest, src_w):
                copied += 1
                used_dest.add(id(dest))

        def _ordered(model, cls_name: str) -> list:
            return [
                layer
                for layer in model.layers
                if layer.__class__.__name__ == cls_name and layer.get_weights()
            ]

        for cls_name in ("Conv2D", "Dense"):
            src_layers = _ordered(loaded, cls_name)
            dst_layers = _ordered(self.online, cls_name)
            for src, dest in zip(src_layers, dst_layers):
                if id(dest) in used_dest:
                    continue
                if self._try_set_layer_weights(dest, src.get_weights()):
                    copied += 1
                    used_dest.add(id(dest))

        self.target.set_weights(self.online.get_weights())
        return copied

    def load(self, path: Path | None = None) -> bool:
        target = Path(path) if path is not None else self.config.training.checkpoint_path
        weights_path = self.weights_sidecar_path(target)
        errors: list[str] = []

        with self.tf.device(self.device_info.training_device):
            # Prefer the full .keras model so architecture migrations (vector dim /
            # reason head) always re-run against the real checkpoint, not a stale
            # same-arch weights sidecar written after a partial migrate.
            if target.exists():
                try:
                    from ascent_player.agent.model import _advantage_center_layer

                    AdvantageCenter = _advantage_center_layer()
                    loaded = self.tf.keras.models.load_model(
                        target,
                        custom_objects={"AdvantageCenter": AdvantageCenter},
                        safe_mode=False,
                    )
                    migrated = False
                    try:
                        if self._vector_dim > 0:
                            if len(loaded.inputs) != 2:
                                raise ValueError("expected hybrid visual+vector inputs")
                            visual_shape = tuple(loaded.inputs[0].shape[1:])
                            vector_shape = tuple(loaded.inputs[1].shape[1:])
                            if visual_shape != tuple(self.online.inputs[0].shape[1:]):
                                raise ValueError(
                                    f"visual shape {visual_shape} != "
                                    f"{tuple(self.online.inputs[0].shape[1:])}"
                                )
                            if vector_shape != (self._vector_dim,):
                                raise ValueError(
                                    f"vector shape {vector_shape} != ({self._vector_dim},)"
                                )
                            if len(loaded.outputs) < 2 and self._reason_count > 0:
                                raise ValueError("missing reason head")
                            if self._skill_count > 0 and len(loaded.outputs) < 3:
                                raise ValueError("missing skill head")
                        self.online.set_weights(loaded.get_weights())
                        self.target.set_weights(loaded.get_weights())
                    except Exception as shape_exc:
                        copied = self._copy_compatible_weights(loaded)
                        migrated = True
                        print(
                            f"VECTOR_DIM_MIGRATE / REASON_HEAD_INIT: "
                            f"copied {copied} layers from {target.name} ({shape_exc})",
                            flush=True,
                        )
                    try:
                        self.online.save_weights(weights_path)
                    except Exception:
                        pass
                    self._apply_loaded_progress(target)
                    self._last_load_error = None
                    if migrated:
                        print(
                            "REASON_HEAD_INIT: reason/skill logits randomly initialized",
                            flush=True,
                        )
                    return True
                except Exception as exc:
                    errors.append(f"model load: {exc}")

            if weights_path.exists():
                try:
                    self.online.load_weights(weights_path)
                    self.target.set_weights(self.online.get_weights())
                    self._apply_loaded_progress(target)
                    self._last_load_error = None
                    return True
                except Exception as exc:
                    errors.append(f"weights load: {exc}")

        self._last_load_error = "; ".join(errors) if errors else "checkpoint missing"
        print(f"Checkpoint load failed: {self._last_load_error}")
        return False

