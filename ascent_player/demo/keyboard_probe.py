from __future__ import annotations

from typing import Any

_INSTALL_LISTENERS_JS = """
() => {
    if (window.__ascentKeyState) {
        return true;
    }
    window.__ascentKeyState = { left: false, right: false, space: false };
    const apply = (event, down) => {
        const key = (event.key || "").toLowerCase();
        if (key === "a" || key === "arrowleft") {
            window.__ascentKeyState.left = down;
        }
        if (key === "d" || key === "arrowright") {
            window.__ascentKeyState.right = down;
        }
        if (key === " " || key === "space" || key === "spacebar") {
            window.__ascentKeyState.space = down;
            event.preventDefault();
        }
    };
    document.addEventListener("keydown", (event) => apply(event, true), true);
    document.addEventListener("keyup", (event) => apply(event, false), true);
    return true;
}
"""

_READ_STATE_JS = """
() => {
    const state = window.__ascentKeyState || { left: false, right: false, space: false };
    return {
        left: !!state.left,
        right: !!state.right,
        space: !!state.space,
    };
}
"""


async def install_keyboard_probe(page: Any) -> None:
    await page.evaluate(_INSTALL_LISTENERS_JS)


async def read_keyboard_state(page: Any) -> dict[str, bool]:
    payload = await page.evaluate(_READ_STATE_JS)
    if not isinstance(payload, dict):
        return {"left": False, "right": False, "space": False}
    return {
        "left": bool(payload.get("left")),
        "right": bool(payload.get("right")),
        "space": bool(payload.get("space")),
    }


def keys_to_action(key_state: dict[str, bool]) -> int:
    left = key_state.get("left", False)
    right = key_state.get("right", False)
    space = key_state.get("space", False)
    if left and space:
        return 4
    if right and space:
        return 5
    if space:
        return 3
    if left:
        return 1
    if right:
        return 2
    return 0
