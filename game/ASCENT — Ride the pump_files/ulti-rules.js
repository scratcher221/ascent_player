export function advanceUltiCharge(charge,dt,cooldownSeconds) {
  if (!Number.isFinite(cooldownSeconds) || cooldownSeconds <= 0) return 100;
  return Math.min(100,Math.max(0,charge) + dt / cooldownSeconds * 100);
}

export function ultiCooldownRemaining(charge,cooldownSeconds) {
  if (!Number.isFinite(cooldownSeconds) || cooldownSeconds <= 0) return 0;
  return Math.max(0,Math.ceil((100-Math.max(0,Math.min(100,charge))) / 100 * cooldownSeconds));
}

export function canActivateUltiSlot(activeState) {
  return !activeState;
}

// Kept as an empty compatibility export: arcade pacing has no post-ulti penalty phase.
export const ULTI_PENALTY_DURATION = {};

// Returns true if AEGIS (tier 2) is active and shield unused in any slot.
export function aegisShieldActive(activeStates) {
  return activeStates.some(s => s?.tier === 2 && s?.phase === 'active' && !s?.counters?.shieldUsed);
}

// Ticks the HELIX impulse timer; returns true if an impulse should fire this tick.
export function tickHelixTimer(counters, dt) {
  counters.helixTimer -= dt;
  if (counters.helixTimer <= 0) {
    counters.helixTimer = 2.5;
    return true;
  }
  return false;
}
