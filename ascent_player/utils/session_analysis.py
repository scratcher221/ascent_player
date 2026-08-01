"""Summarize a thread-BC watchdog log and recommend the next cycle CLI."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

_COLLECT = re.compile(r"COLLECT_DONE recent_avg=([0-9.]+)")
_BC_REVERT = re.compile(r"BC_REVERT")
_BC_ALIGNED = re.compile(r"BC_PROBE_ALIGNED mean=([0-9.]+)")
_BC_GREEDY = re.compile(r"BC_PROBE_GREEDY mean=([0-9.]+)")
_EVAL_GREEDY = re.compile(r"EVAL_GREEDY mean=([0-9.]+)")
_EVAL_ALIGNED = re.compile(r"EVAL_ALIGNED mean=([0-9.]+)")
_PROMOTE = re.compile(r"PROMOTE_VIA|TARGET_MET")
_CYCLE_END = re.compile(r"THREAD_BC_CYCLE_END floor=([0-9.]+)")
_ELITE = re.compile(
    r"ELITE_REPLAY merge before=\d+ browser=\d+ episodes=(\d+) saved=(\d+)"
)


@dataclass(slots=True)
class SessionAnalysis:
    log_path: str
    collect_avgs: list[float] = field(default_factory=list)
    bc_reverts: int = 0
    bc_aligned_probes: list[float] = field(default_factory=list)
    bc_greedy_probes: list[float] = field(default_factory=list)
    eval_greedy: list[float] = field(default_factory=list)
    eval_aligned: list[float] = field(default_factory=list)
    promotions: int = 0
    final_floor: float | None = None
    elite_episodes: int | None = None
    elite_transitions: int | None = None
    cycle_rounds: int = 0
    recommendations: list[str] = field(default_factory=list)
    watchdog_extra_args: list[str] = field(default_factory=list)


def analyze_watchdog_log(path: Path, *, max_bytes: int = 8_000_000) -> SessionAnalysis:
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raw = raw[-max_bytes:]
    text = raw.decode("utf-8", errors="replace")
    out = SessionAnalysis(log_path=str(path))

    for m in _COLLECT.finditer(text):
        out.collect_avgs.append(float(m.group(1)))
    out.bc_reverts = len(_BC_REVERT.findall(text))
    out.bc_aligned_probes = [float(m.group(1)) for m in _BC_ALIGNED.finditer(text)]
    out.bc_greedy_probes = [float(m.group(1)) for m in _BC_GREEDY.finditer(text)]
    out.eval_greedy = [float(m.group(1)) for m in _EVAL_GREEDY.finditer(text)]
    out.eval_aligned = [float(m.group(1)) for m in _EVAL_ALIGNED.finditer(text)]
    out.promotions = len(_PROMOTE.findall(text))
    out.cycle_rounds = text.count("CYCLE_ROUND id=")

    m = list(_CYCLE_END.finditer(text))
    if m:
        out.final_floor = float(m[-1].group(1))

    elite = list(_ELITE.finditer(text))
    if elite:
        out.elite_episodes = int(elite[-1].group(1))
        out.elite_transitions = int(elite[-1].group(2))

    out.recommendations = _recommend(out)
    out.watchdog_extra_args = _watchdog_args(out)
    return out


def _recommend(analysis: SessionAnalysis) -> list[str]:
    notes: list[str] = []
    if analysis.bc_reverts > 0 and not analysis.eval_greedy:
        notes.append(
            "BC kept reverting on greedy probe while collect avg was ~1k+; "
            "use aligned BC probe (code fix) for next session."
        )
    if analysis.collect_avgs:
        tail = mean(analysis.collect_avgs[-5:]) if len(analysis.collect_avgs) >= 5 else mean(
            analysis.collect_avgs
        )
        notes.append(f"Recent collect mean ~{tail:.0f}.")
        if tail >= 1100 and analysis.promotions == 0:
            notes.append(
                "Collect strong but no promotion — run dual eval after BC; "
                "increase bc-steps and skip ceiling/bootstrap on resume."
            )
    if analysis.elite_transitions and analysis.elite_transitions >= 2500:
        notes.append("Elite buffer saturated — raise bc-steps for next session.")
    if analysis.eval_greedy:
        notes.append(
            f"Last greedy eval mean {analysis.eval_greedy[-1]:.0f}."
        )
    return notes


def _watchdog_args(analysis: SessionAnalysis) -> list[str]:
    """Extra args forwarded to run_thread_bc_cycle via watchdog."""
    extra: list[str] = [
        "--skip-ceiling",
        "--skip-skill-bootstrap",
    ]
    bc_steps = 500
    if analysis.elite_transitions and analysis.elite_transitions >= 2000:
        bc_steps = 1000
    if analysis.bc_reverts >= 8 and not analysis.eval_greedy:
        bc_steps = min(900, bc_steps + 150)
    extra.extend(
        [
            "--bc-steps",
            str(bc_steps),
            "--bc-lr",
            "2.5e-6",
            "--thread-watch-prior",
            "0.58",
            "--collect-minutes",
            "22",
            "--eval-episodes",
            "10",
        ]
    )
    collect_gate = 1000.0
    if analysis.collect_avgs:
        recent = mean(analysis.collect_avgs[-8:])
        if recent >= 1150:
            collect_gate = 1100.0
    extra.extend(
        [
            "--elite-max-episodes",
            "48",
            "--bc-secondary-gate",
            "1800",
        ]
    )
    extra.extend(["--collect-gate", f"{collect_gate:.0f}"])
    return extra


def write_report(analysis: SessionAnalysis, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(analysis), indent=2) + "\n", encoding="utf-8")


def analyze_session_logs(paths: list[Path]) -> SessionAnalysis:
    """Merge stats from one or more watchdog logs (e.g. mid-session watchdog restart)."""
    if not paths:
        raise ValueError("paths must not be empty")
    if len(paths) == 1:
        return analyze_watchdog_log(paths[0])
    merged = SessionAnalysis(
        log_path=",".join(str(p.resolve()) for p in paths),
    )
    for path in paths:
        part = analyze_watchdog_log(path)
        merged.collect_avgs.extend(part.collect_avgs)
        merged.bc_reverts += part.bc_reverts
        merged.bc_aligned_probes.extend(part.bc_aligned_probes)
        merged.bc_greedy_probes.extend(part.bc_greedy_probes)
        merged.eval_greedy.extend(part.eval_greedy)
        merged.eval_aligned.extend(part.eval_aligned)
        merged.promotions += part.promotions
        merged.cycle_rounds += part.cycle_rounds
        if part.final_floor is not None:
            merged.final_floor = part.final_floor
        if part.elite_episodes is not None:
            merged.elite_episodes = part.elite_episodes
        if part.elite_transitions is not None:
            merged.elite_transitions = part.elite_transitions
    merged.recommendations = _recommend(merged)
    merged.watchdog_extra_args = _watchdog_args(merged)
    return merged


def watchdog_logs_since(log_dir: Path, since_ts: float) -> list[Path]:
    if not log_dir.is_dir():
        return []
    out: list[Path] = []
    for path in log_dir.glob("skill_training_watchdog_*.log"):
        try:
            if path.stat().st_mtime >= since_ts - 120:
                out.append(path)
        except OSError:
            continue
    return sorted(out, key=lambda p: p.stat().st_mtime)
