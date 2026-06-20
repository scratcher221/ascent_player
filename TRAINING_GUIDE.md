# Ascent Player Training Guide

This guide explains how to train the Ascent agent, measure whether it is improving, and troubleshoot common failure modes.

## Prerequisites

1. Python 3.11 virtual environment with dependencies installed.
2. Playwright Chromium installed.
3. The local game starts automatically when you launch browser training.

```bash
cd "Ascent Player"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## What the agent uses now

The agent is a **hybrid policy**:

- **Structured game state** from `window.__ASCENT_AGENT__` (platform targets, velocity, boost usefulness, phase flags)
- **Screenshots** as supporting visual context
- **Rule-based prior** during early training to teach steering and meaningful boost usage
- **Dueling Double DQN** with prioritized replay and n-step returns

This is much easier to learn from than screenshots alone.

## Recommended training workflow

### 1. Verify the rule baseline works

Run the rule policy in sim first:

```bash
python main.py --rule-baseline --eval-sim --eval-episodes 5 --no-ui
```

Then in the browser:

```bash
python main.py --rule-baseline --eval-episodes 3 --no-ui
```

You should see non-zero survival length, some landings, and steering toward platforms.

### 2. Sim pretrain

```bash
python main.py --pretrain-steps 500000 --no-ui --device gpu
```

Watch for:

- `sps` above 300 on GPU
- `loss` trending down
- `checkpoints/sim_pretrained.keras` saved

### 3. Browser fine-tune from sim

```bash
python main.py --transfer-from-sim --no-ui --device gpu
```

Or with the UI:

```bash
python main.py --transfer-from-sim
```

The app keeps part of the sim replay buffer during transfer and uses a rule-policy prior during early browser steps.

### 4. Evaluate the learned policy

```bash
python main.py --evaluate-policy --eval-episodes 10 --no-ui
```

Compare against the rule baseline:

```bash
python main.py --rule-baseline --eval-episodes 10 --no-ui
```

### 5. Add human demos if progress stalls

See [DEMO_RECORDING.md](DEMO_RECORDING.md). Record runs that reach **1500+** score, then train with demos enabled.

## How to view progress

### Console output

During training you will see:

- `step=... action=... reward=... score=...`
- Episode summaries with reward and max score
- Sim pretrain progress every few thousand steps

### Log files

Every session writes to `logs/`:

| File | Meaning |
|------|---------|
| `logs/training_latest.log` | Pointer to latest session |
| `logs/training_latest_browser.log` | Latest browser session |
| `logs/training_<timestamp>_sim.log` | Sim pretrain |
| `logs/training_<timestamp>_browser.log` | Browser training |

Follow browser logs live:

```bash
tail -f logs/training_latest_browser.log
```

### Skill metrics to watch

Each episode log now includes:

- **`landing_rate`** — platform landings per step. Should rise above ~0.01 for beginner play.
- **`meaningful_boost_rate`** — fraction of jump actions taken when boost was useful.
- **`mean_platform_dx`** — average horizontal distance to target platform while falling.
- **`max_score`** — best score in the episode.
- **`score_velocity`** — score change per step in the UI status bar.

Good beginner progress looks like:

- Landing rate increasing over time
- Mean platform dx decreasing
- Meaningful boost rate above wasted boost rate
- Max score climbing past a few hundred, then past 800+

### Checkpoints

| Checkpoint | Purpose |
|------------|---------|
| `checkpoints/sim_pretrained.keras` | Sim-only weights |
| `checkpoints/dqn_latest.keras` | Browser training resume point |

## Useful commands

```bash
# UI training
python main.py

# Sim calibration
python main.py --calibrate-sim --no-ui

# Sim pretrain
python main.py --pretrain-steps 500000 --no-ui

# Browser fine-tune
python main.py --transfer-from-sim --no-ui

# Rule baseline in sim
python main.py --rule-baseline --eval-sim --eval-episodes 10 --no-ui

# Evaluate learned checkpoint
python main.py --evaluate-policy --eval-episodes 10 --no-ui

# Watch mode (no learning)
python main.py --watch
```

## Troubleshooting

### Agent never lands on platforms

- Run `--rule-baseline --eval-sim` first. If the rule policy also fails, the state hook is broken.
- Check logs for `agent_hook_ok` / structured orb and platform fields.
- Pretrain longer in sim (`--pretrain-steps 1000000`).

### Agent spams jump / boost

- Check `meaningful_boost_rate` vs `wasted_boost` in logs.
- Lower exploration temporarily with `--watch` to inspect behavior.
- Ensure boost masking is active (`can_boost=False` should block jump actions).

### Browser training is very slow

- Look for loop Hz below 15 in logs.
- Close other GPU-heavy apps.
- Use `--device gpu`.
- Prefer sim pretrain first, then shorter browser fine-tune sessions.

### KDE Wayland compositor crashes when Chromium launches

On Plasma Wayland with NVIDIA, native Wayland Chromium can crash `kwin_wayland`.
Ascent Player auto-passes `--ozone-platform=x11` and disables Vulkan when
`WAYLAND_DISPLAY` is set. If problems persist, log into an **X11 Plasma session**
instead of Wayland for browser training.

### Scores stay flat for hours

- Evaluate with `--evaluate-policy` instead of guessing from runtime alone.
- Compare against `--rule-baseline`.
- Record human demos and re-run with demo ingest enabled.
- Delete stale checkpoints and repeat sim pretrain if the model learned the wrong policy.

### Transfer from sim does not help

- Confirm `checkpoints/sim_pretrained.keras` exists and loads.
- Run sim evaluation before browser fine-tune.
- Fine-tune in shorter sessions and evaluate after each session.

## Target: entry-level human play

Before expecting high scores, the agent should reliably:

1. Steer toward the next safe platform.
2. Land on platforms repeatedly.
3. Use Space only when falling with a gap or miss risk.
4. Beat random play and approach the rule baseline in sim.

Once those skills are stable in sim and browser evaluation, longer training sessions are worth running.
