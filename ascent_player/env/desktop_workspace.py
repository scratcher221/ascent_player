"""Pin training Chromium + monitor onto a named KDE virtual desktop.

KDE Wayland does not expose virtual desktops to X11 (`wmctrl`). KWin
scripting can assign windows by PID / title / WM class. Desktop name
defaults to ``Ascent`` (see kwinrc ``Name_2``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_WORKSPACE = "Ascent"
KWIN_PLUGIN = "ascentPlayerWorkspacePin"
PLAYWRIGHT_HINTS = (
    "playwright_chromiumdev_profile",
    "ascent-player-game",
    "Google Chrome for Testing",
)
MONITOR_TITLE_PREFIXES = (
    "Ascent — v2 climb",
    "Ascent — thread-BC",
    "Ascent — Reliability",
    "Ascent — Training",
)


def workspace_name(override: str | None = None) -> str:
    if override is not None:
        return str(override).strip()
    return str(os.environ.get("ASCENT_WINDOW_WORKSPACE", DEFAULT_WORKSPACE) or "").strip()


def parse_kwinrc_desktops(text: str) -> list[tuple[str, str]]:
    """Return [(id, name), ...] from a kwinrc [Desktops] section."""
    ids: dict[int, str] = {}
    names: dict[int, str] = {}
    in_desktops = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_desktops = line.lower() == "[desktops]"
            continue
        if not in_desktops or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key.startswith("Id_"):
            try:
                ids[int(key[3:])] = value
            except ValueError:
                continue
        elif key.startswith("Name_"):
            try:
                names[int(key[5:])] = value
            except ValueError:
                continue
    out: list[tuple[str, str]] = []
    for idx in sorted(set(ids) | set(names)):
        desk_id = ids.get(idx, "")
        name = names.get(idx, f"Desktop {idx}")
        if desk_id or name:
            out.append((desk_id, name))
    return out


def build_kwin_pin_script(
    *,
    desktop_name: str,
    pids: list[int] | tuple[int, ...] = (),
    title_prefixes: list[str] | tuple[str, ...] = MONITOR_TITLE_PREFIXES,
    resource_hints: list[str] | tuple[str, ...] = PLAYWRIGHT_HINTS,
) -> str:
    """KWin 6 JS: pin matching windows to ``desktop_name`` and keep doing so."""
    payload = {
        "desktop": desktop_name,
        "pids": [int(p) for p in pids if int(p) > 0],
        "titles": list(title_prefixes),
        "hints": list(resource_hints),
    }
    blob = json.dumps(payload)
    return f"""const CFG = {blob};
function targetDesktop() {{
    const desks = workspace.desktops;
    for (let i = 0; i < desks.length; i++) {{
        if ((desks[i].name || "") === CFG.desktop) return desks[i];
    }}
    return null;
}}
function matches(c) {{
    if (!c) return false;
    if (CFG.pids.indexOf(c.pid) !== -1) return true;
    const cap = c.caption || "";
    const blob = ((c.resourceClass || "") + " " + (c.resourceName || "") + " " + cap).toLowerCase();
    for (let i = 0; i < CFG.hints.length; i++) {{
        if (blob.indexOf(String(CFG.hints[i]).toLowerCase()) !== -1) return true;
    }}
    for (let i = 0; i < CFG.titles.length; i++) {{
        if (cap.indexOf(CFG.titles[i]) !== -1) return true;
    }}
    return false;
}}
function pin(c, desk) {{
    try {{ c.onAllDesktops = false; }} catch (e) {{}}
    try {{ c.desktops = [desk]; }} catch (e) {{}}
}}
function applyAll() {{
    const desk = targetDesktop();
    if (!desk) {{
        print("WORKSPACE_PIN_FAIL no desktop named " + CFG.desktop);
        return;
    }}
    let n = 0;
    workspace.windowList().forEach(function (c) {{
        if (matches(c)) {{
            pin(c, desk);
            n += 1;
            print("WORKSPACE_PIN pid=" + c.pid + " cap=" + c.caption);
        }}
    }});
    print("WORKSPACE_PIN done n=" + n + " desktop=" + CFG.desktop);
}}
try {{ workspace.windowAdded.connect(function (c) {{
    const desk = targetDesktop();
    if (desk && matches(c)) pin(c, desk);
}}); }} catch (e) {{}}
applyAll();
"""


def _qdbus() -> str | None:
    return shutil.which("qdbus6") or shutil.which("qdbus")


def _run_qdbus(args: list[str], *, timeout: float = 5.0) -> str:
    exe = _qdbus()
    if not exe:
        raise FileNotFoundError("qdbus6")
    return subprocess.check_output(
        [exe, *args],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    ).strip()


def pin_windows_to_named_desktop(
    name: str | None = None,
    *,
    pids: list[int] | tuple[int, ...] = (),
    title_prefixes: list[str] | tuple[str, ...] = MONITOR_TITLE_PREFIXES,
    resource_hints: list[str] | tuple[str, ...] = PLAYWRIGHT_HINTS,
) -> bool:
    """Load/run a KWin script that pins matching windows onto ``name``.

    Returns False when KWin is unavailable or ``name`` is empty (pin disabled).
    """
    desktop = workspace_name(name)
    if not desktop:
        return False
    if not _qdbus():
        print("WORKSPACE_PIN_SKIP qdbus6 not found (need KWin)", flush=True)
        return False
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "ascent-player"
    runtime.mkdir(parents=True, exist_ok=True)
    script_path = runtime / "workspace_pin.js"
    script_path.write_text(
        build_kwin_pin_script(
            desktop_name=desktop,
            pids=pids,
            title_prefixes=title_prefixes,
            resource_hints=resource_hints,
        ),
        encoding="utf-8",
    )
    try:
        _run_qdbus(
            [
                "org.kde.KWin",
                "/Scripting",
                "org.kde.kwin.Scripting.unloadScript",
                KWIN_PLUGIN,
            ]
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    try:
        script_id = _run_qdbus(
            [
                "org.kde.KWin",
                "/Scripting",
                "org.kde.kwin.Scripting.loadScript",
                str(script_path),
                KWIN_PLUGIN,
            ]
        )
        try:
            _run_qdbus(
                ["org.kde.KWin", "/Scripting", "org.kde.kwin.Scripting.start"]
            )
        except (subprocess.SubprocessError, OSError):
            pass
        _run_qdbus(
            [
                "org.kde.KWin",
                f"/Scripting/Script{script_id}",
                "org.kde.kwin.Script.run",
            ]
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        print(f"WORKSPACE_PIN_FAIL {type(exc).__name__}: {exc}", flush=True)
        return False
    print(f"WORKSPACE_PIN desktop={desktop!r} script={script_path}", flush=True)
    return True


def pin_current_process_to_workspace(name: str | None = None) -> bool:
    """Pin this process's windows (training monitor) plus the usual game matches."""
    return pin_windows_to_named_desktop(name, pids=(os.getpid(),))
