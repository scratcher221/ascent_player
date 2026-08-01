from __future__ import annotations

from typing import Any

_INSTALL_LISTENERS_JS = """
() => {
    const install = (target) => {
        if (!target || target.__ascentKeyProbeInstalled) {
            return;
        }
        target.__ascentKeyProbeInstalled = true;
        target.addEventListener("keydown", (event) => apply(event, true), true);
        target.addEventListener("keyup", (event) => apply(event, false), true);
    };
    const apply = (event, down) => {
        const key = (event.key || "").toLowerCase();
        window.__ascentKeyState = window.__ascentKeyState || {
            left: false,
            right: false,
            space: false,
        };
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
    window.__ascentKeyState = window.__ascentKeyState || {
        left: false,
        right: false,
        space: false,
    };
    install(window);
    install(document);
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
        installed: !!window.__ascentKeyState,
    };
}
"""


async def install_keyboard_probe(page: Any) -> None:
    context = getattr(page, "context", None)
    if callable(context):
        ctx = context()
        if hasattr(ctx, "add_init_script"):
            await ctx.add_init_script(_INSTALL_LISTENERS_JS)
    if hasattr(page, "add_init_script"):
        await page.add_init_script(_INSTALL_LISTENERS_JS)
    await page.evaluate(_INSTALL_LISTENERS_JS)


async def read_keyboard_state(page: Any) -> dict[str, bool]:
    payload = await page.evaluate(_READ_STATE_JS)
    if not isinstance(payload, dict):
        return {"left": False, "right": False, "space": False, "installed": False}
    return {
        "left": bool(payload.get("left")),
        "right": bool(payload.get("right")),
        "space": bool(payload.get("space")),
        "installed": bool(payload.get("installed", True)),
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
