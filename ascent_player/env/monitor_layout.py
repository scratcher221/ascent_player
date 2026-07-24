"""Resolve desktop monitor geometry for pinning the game Chromium window."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MonitorGeometry:
    name: str
    x: int
    y: int
    width: int
    height: int


def _edid_names() -> dict[str, str]:
    """Map drm connector basename (e.g. HDMI-A-1) → EDID model string."""
    from pathlib import Path

    out: dict[str, str] = {}
    drm = Path("/sys/class/drm")
    if not drm.exists():
        return out
    for status_path in drm.glob("card*-*/status"):
        try:
            if status_path.read_text().strip() != "connected":
                continue
        except OSError:
            continue
        connector = status_path.parent.name  # card1-HDMI-A-1
        short = connector.split("-", 1)[-1] if "-" in connector else connector
        # Prefer full suffix after cardN-
        if connector.startswith("card"):
            parts = connector.split("-", 1)
            short = parts[1] if len(parts) > 1 else connector
        edid_path = status_path.parent / "edid"
        if not edid_path.exists():
            continue
        try:
            data = edid_path.read_bytes()
        except OSError:
            continue
        texts: list[str] = []
        for i in range(54, 126, 18):
            block = data[i : i + 18]
            if len(block) < 18:
                continue
            if block[3] in (0xFC, 0xFF, 0xFE):
                raw = bytes(b for b in block[5:] if 32 <= b < 127)
                s = raw.decode("ascii", "ignore").strip()
                if s:
                    texts.append(s)
        if texts:
            # Prefer the descriptive model token (ZOWIE / Acer / MAG).
            out[short] = " ".join(texts)
            # Also index by trailing connector token variants.
            out[short.replace("HDMI-A-", "HDMI-")] = out[short]
    return out


def list_monitors() -> list[MonitorGeometry]:
    """Parse `kscreen-doctor -o` geometries and attach EDID names when possible."""
    if not shutil.which("kscreen-doctor"):
        return []
    try:
        raw = subprocess.check_output(
            ["kscreen-doctor", "-o"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    # Strip ANSI
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    edid = _edid_names()
    monitors: list[MonitorGeometry] = []
    current_conn: str | None = None
    for line in raw.splitlines():
        out_m = re.match(r"Output:\s+\d+\s+(\S+)", line.strip())
        if out_m:
            current_conn = out_m.group(1)
            continue
        geo_m = re.search(
            r"Geometry:\s*(-?\d+),(-?\d+)\s+(\d+)x(\d+)",
            line,
        )
        if geo_m and current_conn:
            name = edid.get(current_conn, current_conn)
            monitors.append(
                MonitorGeometry(
                    name=name,
                    x=int(geo_m.group(1)),
                    y=int(geo_m.group(2)),
                    width=int(geo_m.group(3)),
                    height=int(geo_m.group(4)),
                )
            )
            current_conn = None
    return monitors


def find_monitor(match: str) -> MonitorGeometry | None:
    """Find first monitor whose EDID/name contains ``match`` (case-insensitive)."""
    needle = (match or "").strip().lower()
    if not needle:
        return None
    for mon in list_monitors():
        if needle in mon.name.lower():
            return mon
    return None


def fit_window_to_monitor(
    mon: MonitorGeometry,
    *,
    width: int,
    height: int,
    margin: int = 12,
) -> tuple[int, int, int, int]:
    """Return (x, y, width, height) clamped to the monitor with a small border.

    Prefer near-full usable area so game chrome (sidebar icons, Link wallet)
    is not clipped by the window edge or OS title bar.
    """
    max_w = max(640, mon.width - 2 * margin)
    max_h = max(480, mon.height - 2 * margin)
    width = min(int(width), max_w)
    height = min(int(height), max_h)
    x = mon.x + max(0, (mon.width - width) // 2)
    y = mon.y + max(0, (mon.height - height) // 2)
    x = min(x, mon.x + max(0, mon.width - width))
    y = min(y, mon.y + max(0, mon.height - height))
    return int(x), int(y), int(width), int(height)


def window_origin_for_monitor(
    mon: MonitorGeometry,
    *,
    width: int,
    height: int,
    margin: int = 12,
) -> tuple[int, int]:
    """Top-left for a window inset on the monitor (keeps full UI on-screen)."""
    x, y, _, _ = fit_window_to_monitor(
        mon, width=width, height=height, margin=margin
    )
    return x, y
