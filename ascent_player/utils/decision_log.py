"""Append-only CSV log of action + reason decisions for transparency."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ascent_player.env.game_env import ACTION_LABELS
from ascent_player.env.state_detector import FrameState


DECISION_FIELDS = (
    "ts",
    "episode",
    "step",
    "action",
    "action_label",
    "reason",
    "reason_pred",
    "source",
    "score",
    "eps",
    "target_kind",
    "target_dx",
    "plat_dx",
    "booster_type",
    "anomaly",
    "hazard_kind",
    "agree",
    "wrong_vs_target",
)


@dataclass
class DecisionLogger:
    path: Path
    every: int = 1
    _count: int = 0
    _file: object | None = None
    _writer: csv.DictWriter | None = None

    @classmethod
    def create(
        cls,
        log_dir: Path,
        *,
        every: int = 1,
        prefix: str = "decisions",
    ) -> DecisionLogger:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"{prefix}_{stamp}.csv"
        latest = log_dir / "decisions_latest.path"
        latest.write_text(str(path), encoding="utf-8")
        logger = cls(path=path, every=max(1, int(every)))
        logger._open()
        return logger

    def _open(self) -> None:
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=DECISION_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def maybe_log(
        self,
        *,
        episode: int,
        step: int,
        action: int,
        reason: str,
        reason_pred: str,
        source: str,
        score: float | int | None,
        eps: float,
        frame_state: FrameState | None,
        wrong_vs_target: bool = False,
    ) -> None:
        self._count += 1
        if self._count % self.every != 0:
            return
        if self._writer is None:
            return
        fs = frame_state
        agree = "1" if reason == reason_pred else "0"
        row = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "episode": episode,
            "step": step,
            "action": action,
            "action_label": ACTION_LABELS.get(action, str(action)),
            "reason": reason,
            "reason_pred": reason_pred,
            "source": source,
            "score": "" if score is None else score,
            "eps": f"{eps:.4f}",
            "target_kind": (fs.target_kind if fs else "") or "",
            "target_dx": (
                f"{fs.target_dx:.4f}" if fs and fs.target_dx is not None else ""
            ),
            "plat_dx": (
                f"{fs.nearest_platform_dx:.4f}"
                if fs and fs.nearest_platform_dx is not None
                else ""
            ),
            "booster_type": (fs.booster_type if fs else "") or "",
            "anomaly": (fs.anomaly_type if fs else "") or "",
            "hazard_kind": (fs.hazard_kind if fs else "") or "",
            "agree": agree,
            "wrong_vs_target": "1" if wrong_vs_target else "0",
        }
        self._writer.writerow(row)
        if self._file is not None and self._count % (self.every * 20) == 0:
            self._file.flush()
