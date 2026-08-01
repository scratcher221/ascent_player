export const DEFAULT_GAMEPAD_BINDINGS = Object.freeze({
  left:Object.freeze({ type:"axis", index:0, direction:-1 }),
  right:Object.freeze({ type:"axis", index:0, direction:1 }),
  boost:Object.freeze({ type:"button", index:0 }),
  pause:Object.freeze({ type:"button", index:9 }),
  ulti1:Object.freeze({ type:"button", index:4 }),
  ulti2:Object.freeze({ type:"button", index:5 }),
  ulti3:Object.freeze({ type:"button", index:3 })
});

export const QUALITY_MODES = Object.freeze(["auto","high","low"]);

export const TRAIL_MODES = Object.freeze(["all","basic","off"]);

export const DEFAULT_OPTIONS = Object.freeze({
  musicVolume:0.8,
  sfxVolume:0.4,
  musicMuted:false,
  sfxMuted:false,
  vibrationEnabled:true,
  qualityMode:"auto",
  trailMode:"all",
  ultiKeys:Object.freeze(["Q","E","Z"]),
  pauseKey:"Escape",
  boostKey:"Space",
  leftKey:"A",
  rightKey:"D",
  gamepadBindings:DEFAULT_GAMEPAD_BINDINGS
});

const RESERVED_KEYS = new Set([
  " ", "SPACE", "SPACEBAR",
  "ARROWLEFT", "ARROWRIGHT", "ARROWUP", "ARROWDOWN",
  "A", "D", "R", "M",
  "ALT", "ALTGRAPH", "CAPSLOCK", "CONTROL", "ESCAPE", "META", "SHIFT", "TAB",
  "BACKSPACE", "DELETE", "END", "ENTER", "HOME", "INSERT", "PAGEDOWN", "PAGEUP"
]);

const CONTROL_FORBIDDEN = new Set([
  "ALT","ALTGRAPH","CAPSLOCK","CONTROL","META","SHIFT","TAB",
  "BACKSPACE","DELETE","END","ENTER","HOME","INSERT","PAGEDOWN","PAGEUP",
  "R","M",
  "0","1","2","3","4","5","6","7","8","9"
]);

const CONTROL_SPECIALS = new Map([
  ["ESCAPE","Escape"],["SPACE","Space"],
  ["ARROWLEFT","ArrowLeft"],["ARROWRIGHT","ArrowRight"],
  ["ARROWUP","ArrowUp"],["ARROWDOWN","ArrowDown"]
]);

export function normalizeControlKey(value) {
  if (typeof value !== "string") return null;
  const upper = value.toUpperCase();
  if (!upper) return null;
  if (CONTROL_SPECIALS.has(upper)) return CONTROL_SPECIALS.get(upper);
  if (/^F([1-9]|1[0-2])$/.test(upper)) return upper;
  if (upper.length === 1 && /^[A-Z]$/.test(upper) && !CONTROL_FORBIDDEN.has(upper)) return upper;
  return null;
}

function volume(value,fallback) {
  return Number.isFinite(value) ? Math.min(1,Math.max(0,value)) : fallback;
}

function cloneGamepadBinding(binding) {
  return { ...binding };
}

export function normalizeGamepadBinding(value, fallback) {
  const source = value && typeof value === "object" ? value : {};
  const fallbackBinding = cloneGamepadBinding(fallback);
  const index = Number(source.index);
  if (!Number.isInteger(index) || index < 0 || index > 31) return fallbackBinding;
  if (source.type === "button") return { type:"button", index };
  if (source.type === "axis") {
    const direction = Number(source.direction);
    if (direction !== -1 && direction !== 1) return fallbackBinding;
    return { type:"axis", index, direction };
  }
  return fallbackBinding;
}

export function normalizeGamepadBindings(value) {
  const source = value && typeof value === "object" ? value : {};
  return Object.fromEntries(Object.entries(DEFAULT_GAMEPAD_BINDINGS).map(([action,fallback]) => [
    action,
    normalizeGamepadBinding(source[action], fallback)
  ]));
}

export function normalizeUltiKey(value) {
  if (typeof value !== "string") return null;
  const key = value.toUpperCase();
  if (!key || RESERVED_KEYS.has(key) || /^[0-9]$/.test(key)) return null;
  if (key.length === 1 || /^F([1-9]|1[0-2])$/.test(key)) return key;
  return null;
}

export function normalizeOptions(value) {
  const source = value && typeof value === "object" ? value : {};
  const keys = Array.isArray(source.ultiKeys) ? source.ultiKeys.map(normalizeUltiKey) : [];
  const validKeys = keys.length === 3 && keys.every(Boolean) && new Set(keys).size === 3
    ? keys
    : [...DEFAULT_OPTIONS.ultiKeys];

  const pKey = normalizeControlKey(source.pauseKey)  ?? DEFAULT_OPTIONS.pauseKey;
  const bKey = normalizeControlKey(source.boostKey)  ?? DEFAULT_OPTIONS.boostKey;
  const lKey = normalizeControlKey(source.leftKey)   ?? DEFAULT_OPTIONS.leftKey;
  const rKey = normalizeControlKey(source.rightKey)  ?? DEFAULT_OPTIONS.rightKey;
  const ctrlSet = new Set([pKey, bKey, lKey, rKey]);
  const ctrlOk  = ctrlSet.size === 4; // no duplicates among control keys

  return {
    musicVolume:volume(source.musicVolume,DEFAULT_OPTIONS.musicVolume),
    sfxVolume:volume(source.sfxVolume,DEFAULT_OPTIONS.sfxVolume),
    musicMuted:typeof source.musicMuted === "boolean" ? source.musicMuted : DEFAULT_OPTIONS.musicMuted,
    sfxMuted:typeof source.sfxMuted === "boolean" ? source.sfxMuted : DEFAULT_OPTIONS.sfxMuted,
    vibrationEnabled:typeof source.vibrationEnabled === "boolean" ? source.vibrationEnabled : DEFAULT_OPTIONS.vibrationEnabled,
    qualityMode:QUALITY_MODES.includes(source.qualityMode) ? source.qualityMode : DEFAULT_OPTIONS.qualityMode,
    trailMode:TRAIL_MODES.includes(source.trailMode) ? source.trailMode : DEFAULT_OPTIONS.trailMode,
    ultiKeys:validKeys,
    pauseKey: ctrlOk ? pKey : DEFAULT_OPTIONS.pauseKey,
    boostKey: ctrlOk ? bKey : DEFAULT_OPTIONS.boostKey,
    leftKey:  ctrlOk ? lKey : DEFAULT_OPTIONS.leftKey,
    rightKey: ctrlOk ? rKey : DEFAULT_OPTIONS.rightKey,
    gamepadBindings:normalizeGamepadBindings(source.gamepadBindings)
  };
}
