"""Build normalized state vectors aligned with window.__ASCENT_AGENT__."""

from __future__ import annotations

import numpy as np

from ascent_player.config import ObservationConfig
from ascent_player.env.navigation import platform_type_onehot
from ascent_player.env.state_detector import FrameState

VECTOR_DIM = 43


def _norm_velocity(value: float | None, scale: float) -> float:
    if value is None:
        return 0.0
    if abs(value) <= 1.5:
        return float(np.clip(value, -1.0, 1.0))
    return float(np.clip(value / scale, -1.0, 1.0))


def vector_from_frame_state(state: FrameState, *, width: float = 640.0, height: float = 360.0) -> np.ndarray:
    w = max(float(state.canvas_w or width), 1.0)
    h = max(float(state.canvas_h or height), 1.0)
    orb_x = (state.orb_x or w * 0.5) / w if (state.orb_x or 0) > 1.5 else float(state.orb_x or 0.5)
    orb_y = (state.orb_y or h * 0.5) / h if (state.orb_y or 0) > 1.5 else float(state.orb_y or 0.5)
    vx = _norm_velocity(state.orb_vx, 575.0)
    vy = _norm_velocity(state.orb_vy, 1500.0)
    energy = float(np.clip(state.boost_level, 0.0, 1.0))
    combo = float(np.clip(state.combo / 48.0, 0.0, 1.0))
    bonus = float(np.clip(state.bonus / 4000.0, 0.0, 1.0))
    bank = float(np.clip(state.bank_style / 4000.0, 0.0, 1.0))
    climb = float(np.clip(state.height / 5000.0, 0.0, 1.0))
    bounces = float(np.clip(state.bounces / 100.0, 0.0, 1.0))
    storm = float(np.clip(state.storm_level / 10.0, 0.0, 1.0))
    mult = float(np.clip((state.score_multiplier - 1.0) / 9.0, 0.0, 1.0))

    ndx = state.nearest_platform_dx if state.nearest_platform_dx is not None else 0.0
    ndy = state.nearest_platform_dy if state.nearest_platform_dy is not None else 0.0
    wear = float(np.clip(state.platform_wear, 0.0, 1.0))
    pwidth = float(np.clip(state.nearest_platform_width, 0.0, 1.0))

    adx = state.nearest_platform_above_dx if state.nearest_platform_above_dx is not None else 0.0
    ady = state.nearest_platform_above_dy if state.nearest_platform_above_dy is not None else 0.0
    awear = float(np.clip(state.nearest_platform_above_wear, 0.0, 1.0))
    awidth = float(np.clip(state.nearest_platform_above_width, 0.0, 1.0))

    tdx = state.target_dx if state.target_dx is not None else adx or ndx
    tdy = state.target_dy if state.target_dy is not None else ady or ndy
    target_type = platform_type_onehot(state.target_platform_type)

    bdx = state.booster_dx if state.booster_dx is not None else 0.0
    bdy = state.booster_dy if state.booster_dy is not None else 0.0
    booster_onehot = np.zeros(3, dtype=np.float32)
    if state.booster_type == "surge":
        booster_onehot[0] = 1.0
    elif state.booster_type == "stream":
        booster_onehot[1] = 1.0
    elif state.booster_type == "drag":
        booster_onehot[2] = 1.0

    vec = np.array(
        [
            orb_x,
            orb_y,
            vx,
            vy,
            energy,
            combo,
            bonus,
            bank,
            climb,
            bounces,
            float(np.clip(ndx, -1.0, 1.0)),
            float(np.clip(ndy, -1.0, 1.0)),
            wear,
            pwidth,
            float(np.clip(adx, -1.0, 1.0)),
            float(np.clip(ady, -1.0, 1.0)),
            awear,
            awidth,
            float(np.clip(tdx, -1.0, 1.0)),
            float(np.clip(tdy, -1.0, 1.0)),
            *target_type,
            float(np.clip(state.horizontal_error, -1.0, 1.0)),
            1.0 if state.rising else 0.0,
            1.0 if state.falling else 0.0,
            1.0 if state.landing_window else 0.0,
            1.0 if state.airborne else 0.0,
            1.0 if state.boost_useful else 0.0,
            float(np.clip(state.time_to_platform, 0.0, 1.0)),
            1.0 if state.danger_worn else 0.0,
            1.0 if state.danger_sell else 0.0,
            1.0 if state.miss_risk else 0.0,
            float(np.clip(bdx, -1.0, 1.0)),
            float(np.clip(bdy, -1.0, 1.0)),
            *booster_onehot.tolist(),
            1.0 if state.can_boost else 0.0,
            storm,
            mult,
            1.0 if state.agent_hook_ok else 0.0,
            float(np.clip(state.tier_index / 9.0, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    assert vec.shape[0] == VECTOR_DIM
    return vec


def attach_vector(
    visual: np.ndarray,
    state: FrameState,
    config: ObservationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if not config.include_vector_state:
        return visual, np.zeros(VECTOR_DIM, dtype=np.float32)
    return visual, vector_from_frame_state(state)
