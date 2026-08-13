"""Scoped config mutation so collect/eval overrides cannot leak across phases."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from ascent_player.config import AppConfig


@contextmanager
def override_attrs(obj, **fields) -> Iterator[object]:
    previous = {name: getattr(obj, name) for name in fields}
    try:
        for name, value in fields.items():
            setattr(obj, name, value)
        yield obj
    finally:
        for name, value in previous.items():
            setattr(obj, name, value)


@contextmanager
def override_training(config: AppConfig, **fields) -> Iterator[object]:
    with override_attrs(config.training, **fields) as training:
        yield training


@contextmanager
def locked_eval_seed(config: AppConfig, seed: int) -> Iterator[None]:
    prev_seed = config.browser.run_seed
    prev_lock = bool(config.browser.lock_run_seed)
    config.browser.run_seed = int(seed)
    config.browser.lock_run_seed = True
    try:
        yield
    finally:
        config.browser.run_seed = prev_seed
        config.browser.lock_run_seed = prev_lock


def collect_only_fields(
    *,
    eps: float,
    skip_replay_load: bool = True,
    thread_prior: float | None = None,
    rule_prior: float | None = None,
) -> dict:
    """Training fields for policy collect without online TD."""
    fields: dict = {
        "watch_mode": False,
        "disable_td": True,
        "transfer_epsilon_start": float(eps),
        "browser_epsilon_cap": float(eps),
        "browser_epsilon_floor": min(0.03, float(eps)),
        "force_save_browser_replay": True,
        "skip_browser_replay_load": bool(skip_replay_load),
        "sim_warmstart_teacher": False,
        "sim_warmstart_demos": False,
    }
    if thread_prior is not None:
        fields.update(
            seed_thread_enabled=True,
            seed_thread_prior_start=float(thread_prior),
            seed_thread_prior_end=float(thread_prior),
        )
    if rule_prior is not None:
        fields.update(
            rule_prior_start=float(rule_prior),
            rule_prior_end=float(rule_prior),
        )
    return fields
