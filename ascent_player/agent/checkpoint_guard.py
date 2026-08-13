"""Guards for Impala vs Nature checkpoint files."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Impala-mid weights-only ≈ 55–60MB; Nature ≈ 7MB.
MIN_IMPALA_CHECKPOINT_MB = 40.0
NATURE_MAX_MB = 20.0


@dataclass(frozen=True)
class GuardResult:
    path: Path
    rejected: bool
    size_mb: float
    reason: str = ""


def checkpoint_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def is_likely_nature_checkpoint(path: Path) -> bool:
    return 0 < checkpoint_size_mb(path) < NATURE_MAX_MB


def is_impala_sized_checkpoint(path: Path) -> bool:
    return checkpoint_size_mb(path) >= MIN_IMPALA_CHECKPOINT_MB


def reject_undersized_impala_save(
    path: Path,
    *,
    previous_bytes: int | None = None,
) -> bool:
    """True if save looks Nature-sized or catastrophically shrank a full Impala file."""
    mb = checkpoint_size_mb(path)
    if mb < NATURE_MAX_MB:
        return True
    if previous_bytes is not None and previous_bytes > 100 * 1024 * 1024:
        if path.exists() and path.stat().st_size < previous_bytes * 0.5:
            return True
    return False


def copy_checkpoint_bundle(src: Path, dst: Path) -> None:
    """Copy .keras plus optional .weights.h5 / .meta.json sidecars."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    for suffix in (".weights.h5", ".meta.json"):
        side = src.with_name(src.stem + suffix)
        if side.exists():
            shutil.copy2(side, dst.with_name(dst.stem + suffix))


def restore_work_checkpoints_from_best(
    best: Path,
    work_paths: Iterable[Path],
) -> list[str]:
    """Copy Impala best onto work paths that are missing or undersized only."""
    restored: list[str] = []
    if not best.exists() or not is_impala_sized_checkpoint(best):
        return restored
    for dst in work_paths:
        if dst.exists() and is_impala_sized_checkpoint(dst):
            continue
        copy_checkpoint_bundle(best, dst)
        restored.append(dst.name)
    return restored


def save_guarded(agent, path: Path) -> GuardResult:
    """Save agent; restore previous if the new file looks invalid for Impala best paths."""
    previous = path.stat().st_size if path.exists() else None
    bak = path.with_suffix(path.suffix + ".bak_guard")
    side_baks: list[tuple[Path, Path]] = []
    if path.exists():
        shutil.copy2(path, bak)
        for suffix in (".weights.h5", ".meta.json"):
            side = path.with_name(path.stem + suffix)
            if side.exists():
                side_bak = side.with_suffix(side.suffix + ".bak_guard")
                shutil.copy2(side, side_bak)
                side_baks.append((side, side_bak))
    agent.save(path)
    size_mb = checkpoint_size_mb(path)
    rejected = reject_undersized_impala_save(path, previous_bytes=previous)
    if rejected:
        reason = f"undersized_or_shrunk size={size_mb:.1f}MB"
        print(f"CKPT_GUARD reject {path.name} {reason} — restoring previous", flush=True)
        if bak.exists():
            shutil.copy2(bak, path)
        for side, side_bak in side_baks:
            if side_bak.exists():
                shutil.copy2(side_bak, side)
    for victim in [bak, *[b for _, b in side_baks]]:
        if victim.exists():
            try:
                victim.unlink()
            except OSError:
                pass
    return GuardResult(path=path, rejected=rejected, size_mb=size_mb, reason="" if not rejected else "rejected")
