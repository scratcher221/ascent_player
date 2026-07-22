"""Stable entity IDs for structured state, reasons, and decision logs."""

from __future__ import annotations

# Platforms
PLATFORM_NEUTRAL = "platform_neutral"
PLATFORM_BUY = "platform_buy"
PLATFORM_SELL = "platform_sell"

# Boosters (surge ≈ yellow orb / stream ≈ boost highway)
BOOSTER_SURGE = "booster_surge"
BOOSTER_STREAM = "booster_stream"
BOOSTER_DRAG = "booster_drag"

# Anomalies
ANOMALY_LIQUIDITY_VOID = "anomaly_liquidityVoid"
ANOMALY_SHORT_SQUEEZE = "anomaly_shortSqueeze"
ANOMALY_DARK_POOL = "anomaly_darkPool"

# Hazards / goals
VOID_PORTAL = "void_portal"
VOID_SHARD = "void_shard"
SQUEEZE_SPIKE = "squeeze_spike"

NONE = "none"

ENTITY_IDS: tuple[str, ...] = (
    PLATFORM_NEUTRAL,
    PLATFORM_BUY,
    PLATFORM_SELL,
    BOOSTER_SURGE,
    BOOSTER_STREAM,
    BOOSTER_DRAG,
    ANOMALY_LIQUIDITY_VOID,
    ANOMALY_SHORT_SQUEEZE,
    ANOMALY_DARK_POOL,
    VOID_PORTAL,
    VOID_SHARD,
    SQUEEZE_SPIKE,
    NONE,
)

# Compact target_kind one-hot slots for the vector (order matters).
TARGET_KIND_SLOTS: tuple[str, ...] = (
    "platform",
    "booster_surge",
    "booster_stream",
    "booster_drag",
    "portal",
    "hazard",
    "none",
)

ANOMALY_TYPE_SLOTS: tuple[str, ...] = (
    "liquidityVoid",
    "shortSqueeze",
    "darkPool",
    "none",
)


def platform_entity_id(platform_type: str | None) -> str:
    if platform_type == "buy":
        return PLATFORM_BUY
    if platform_type == "sell":
        return PLATFORM_SELL
    return PLATFORM_NEUTRAL


def booster_entity_id(booster_type: str | None) -> str:
    if booster_type == "surge":
        return BOOSTER_SURGE
    if booster_type == "stream":
        return BOOSTER_STREAM
    if booster_type == "drag":
        return BOOSTER_DRAG
    return NONE


def anomaly_entity_id(anomaly_type: str | None) -> str:
    if anomaly_type == "liquidityVoid":
        return ANOMALY_LIQUIDITY_VOID
    if anomaly_type == "shortSqueeze":
        return ANOMALY_SHORT_SQUEEZE
    if anomaly_type == "darkPool":
        return ANOMALY_DARK_POOL
    return NONE


def target_kind_slot(target_kind: str | None) -> str:
    """Map FrameState.target_kind to a TARGET_KIND_SLOTS entry."""
    if not target_kind:
        return "none"
    if target_kind == "platform" or target_kind.startswith("platform"):
        return "platform"
    if target_kind in ("booster_surge", "surge") or target_kind.endswith("surge"):
        return "booster_surge"
    if target_kind in ("booster_stream", "stream") or target_kind.endswith("stream"):
        return "booster_stream"
    if target_kind in ("booster_drag", "drag") or target_kind.endswith("drag"):
        return "booster_drag"
    if "portal" in target_kind:
        return "portal"
    if "hazard" in target_kind or "shard" in target_kind or "spike" in target_kind:
        return "hazard"
    if target_kind.startswith("booster_"):
        suffix = target_kind.split("_", 1)[-1]
        return target_kind_slot(suffix)
    return "none"


def onehot(slot: str, slots: tuple[str, ...]) -> list[float]:
    return [1.0 if slot == name else 0.0 for name in slots]
