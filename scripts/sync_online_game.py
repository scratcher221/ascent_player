#!/usr/bin/env python3
"""Download the latest ASCENT web build and patch it for local Ascent Player training."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_GAME_DIR = _PROJECT_ROOT / "game"
_DEFAULT_ORIGIN = "https://ascent.xrd.workers.dev"
_SKIP_PREFIXES = ("/admin", "http://", "https://", "//", "#", "data:", "mailto:")
_ASSET_RE = re.compile(
    r"""(?:href|src)=["']([^"']+)["']|url\(\s*['"]?([^'")]+)['"]?\s*\)""",
    re.IGNORECASE,
)
_IMPORT_RE = re.compile(
    r"""from\s*["']\./([^"']+\.[a-z0-9]+)["']|import\s*["']\./([^"']+\.[a-z0-9]+)["']""",
    re.IGNORECASE,
)


def _curl_fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-fsSL",
            "-A",
            "Mozilla/5.0 (AscentPlayer/sync)",
            "-o",
            str(dest),
            url,
        ],
        check=True,
    )


def _collect_paths(text: str) -> set[str]:
    found: set[str] = set()
    for m in _ASSET_RE.finditer(text):
        for g in m.groups():
            if not g:
                continue
            path = g.split("?")[0].strip()
            if not path or path.startswith(_SKIP_PREFIXES):
                continue
            if path.startswith("/") or "${" in path or "`" in path:
                continue
            if not re.match(r"^[A-Za-z0-9_./-]+\\.[A-Za-z0-9]+$", path):
                continue
            found.add(path)
    for m in _IMPORT_RE.finditer(text):
        for g in m.groups():
            if g:
                found.add(g.split("?")[0])
    return found


def _apply_offline_market_chart_patch(text: str) -> str:
    """Skip background candle chart when local training has no live feed."""
    if "CHART_TRIAL_CONFIG?.offlineMode)return" in text:
        return text
    hook_old = "function Pp(){if(I||Bp())return;"
    hook_new = (
        "function Pp(){if(I||Bp()||window.CHART_TRIAL_CONFIG?.hideMarketChart"
        "||window.CHART_TRIAL_CONFIG?.offlineMode)return;"
    )
    if hook_old not in text:
        raise RuntimeError(
            "Could not find market chart draw hook; the upstream build may have changed."
        )
    return text.replace(hook_old, hook_new, 1)


def _patch_offline_market_chart(game_js: Path) -> None:
    text = game_js.read_text(encoding="utf-8")
    updated = _apply_offline_market_chart_patch(text)
    if updated == text:
        print(f"Offline market chart patch already applied: {game_js.name}")
        return
    game_js.write_text(updated, encoding="utf-8")
    print(f"Patched {game_js.name} to hide fallback market candles when offline.")


def _patch_game_js(game_js: Path, inject_src: Path) -> None:
    text = game_js.read_text(encoding="utf-8")
    marker = "/*__ASCENT_PLAYER_INJECTED__*/"
    if marker in text:
        print(f"Already patched: {game_js.name}")
        _patch_offline_market_chart(game_js)
        return

    inject = inject_src.read_text(encoding="utf-8").strip()
    inject = f"{marker}\n{inject}\n"

    # Seed resolver must exist before the bundle reads Kc= on load.
    seed_prelude = """
function __ascentResolveLocalSeed__() {
  const hostOk = /^(localhost|127\\.0\\.0\\.1|\\[::1\\])$/.test(location.hostname);
  if (!hostOk) return null;
  const params = new URLSearchParams(location.search);
  const fromUrl = params.get("devSeed") ?? params.get("runSeed");
  if (fromUrl != null && fromUrl !== "") {
    const n = Number(fromUrl);
    if (Number.isFinite(n)) return n >>> 0;
  }
  const cfg = window.CHART_TRIAL_CONFIG || {};
  const raw = cfg.runSeed;
  if (raw != null && raw !== "") {
    const n = Number(raw);
    if (Number.isFinite(n)) return n >>> 0;
  }
  return null;
}
""".strip()

    seed_old = (
        'Kc=/^(localhost|127\\.0\\.0\\.1|\\[::1\\])$/.test(location.hostname)?'
        'new URLSearchParams(location.search).get("devSeed"):null'
    )
    if seed_old not in text:
        raise RuntimeError(
            "Could not find localhost devSeed hook in game bundle; "
            "the upstream build may have changed."
        )
    text = text.replace(seed_old, "Kc=__ascentResolveLocalSeed__()", 1)

    if seed_prelude not in text:
        text = text.replace('"use strict";', f'"use strict";\n{seed_prelude}', 1)

    hook_old = "V0(r,{goldenTrace:ms}),r._runStep++"
    hook_new = (
        "V0(r,{goldenTrace:ms}),"
        "__ASCENT_AGENT_MODE__&&__ascentExportAgentState__(r),"
        "r._runStep++"
    )
    if hook_old not in text:
        raise RuntimeError("Could not find simulation step hook in game bundle.")
    text = text.replace(hook_old, hook_new, 1)

    text = f'{text.rstrip()}\n{inject}\n'
    text = _apply_offline_market_chart_patch(text)
    game_js.write_text(text, encoding="utf-8")
    print(f"Patched {game_js.name} for agent export + run seed.")


def _patch_index_html(index_path: Path) -> None:
    html = index_path.read_text(encoding="utf-8")
    if "ascent-player-local.js" in html:
        return
    config_match = re.search(r'(<script[^>]+src="config\.[^"]+\.js"[^>]*></script>)', html)
    if not config_match:
        raise RuntimeError("Could not find config script tag in index.html")
    insert = (
        f'{config_match.group(1)}\n'
        '    <script defer src="ascent-player-local.js"></script>'
    )
    html = html.replace(config_match.group(1), insert, 1)
    # Prefer same-origin icons when mirrored locally.
    html = html.replace("https://ascent.xrd.workers.dev/", "")
    index_path.write_text(html, encoding="utf-8")
    print("Patched index.html with ascent-player-local.js")


def sync_game(
    *,
    game_dir: Path,
    origin: str,
    backup_legacy: bool,
) -> None:
    origin = origin.rstrip("/") + "/"
    game_dir.mkdir(parents=True, exist_ok=True)

    legacy_dir = game_dir.parent / "game_legacy"
    old_bundle = game_dir / "ASCENT — Ride the pump.html"
    if backup_legacy and old_bundle.is_file() and not legacy_dir.exists():
        print(f"Backing up previous bundle to {legacy_dir}")
        shutil.move(str(game_dir), str(legacy_dir))
        game_dir.mkdir(parents=True, exist_ok=True)

    queue: deque[str] = deque(["index.html"])
    seen: set[str] = set()
    while queue:
        rel = queue.popleft()
        if rel.startswith("/") or "." not in rel:
            continue
        if rel in seen:
            continue
        seen.add(rel)
        url = urljoin(origin, rel)
        dest = game_dir / rel
        print(f"GET {rel}")
        _curl_fetch(url, dest)
        if dest.suffix.lower() in {".html", ".css", ".js"}:
            try:
                body = dest.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for child in _collect_paths(body):
                if child not in seen:
                    queue.append(child)

    index_path = game_dir / "index.html"
    if not index_path.is_file():
        raise RuntimeError("Download did not produce index.html")

    inject_src = _PROJECT_ROOT / "game_patches" / "ascent-agent-inject.js"
    if not inject_src.is_file():
        raise RuntimeError(f"Missing agent inject source: {inject_src}")
    game_js = next(game_dir.glob("game.*.js"), None)
    if game_js is None:
        raise RuntimeError("No game.*.js bundle found after sync.")
    _patch_game_js(game_js, inject_src)
    _patch_index_html(index_path)

    # Keep local override next to the synced bundle.
    local_js = _PROJECT_ROOT / "game_patches" / "ascent-player-local.js"
    if local_js.is_file():
        shutil.copy2(local_js, game_dir / "ascent-player-local.js")

    print(f"Sync complete: {len(seen)} files -> {game_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, default=_DEFAULT_GAME_DIR)
    parser.add_argument("--origin", default=_DEFAULT_ORIGIN)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not move the old game/ folder to game_legacy/",
    )
    args = parser.parse_args()
    try:
        sync_game(
            game_dir=args.game_dir.resolve(),
            origin=args.origin,
            backup_legacy=not args.no_backup,
        )
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"sync failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
