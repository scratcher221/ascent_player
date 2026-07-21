# Ascent Player — Quick Start

Concise guide for using, training, and monitoring the agent in its current form.

The agent is a **hybrid policy**: structured game state from `window.__ASCENT_AGENT__`, screenshots for visual context, a **rule-based prior** during early training, and a **dueling Double DQN** with prioritized replay.

For deeper troubleshooting, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

---

## 1. One-time setup

```bash
cd "Ascent Player"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Confirm GPU (optional but recommended):

```bash
nvidia-smi
```

The local game server starts automatically when browser training launches. You do not need to run `./game/serve.sh` first.

---

## 2. Use the model (no training)

**Desktop UI** — live preview, start/stop training, load/save checkpoints:

```bash
python main.py
```

**Watch mode** — run the loaded checkpoint with no exploration:

```bash
python main.py --watch
```

Checkpoints load automatically from `checkpoints/dqn_latest.keras` if present.

---

## 3. Train the model

Recommended order:

### Step A — Confirm the rule baseline works

```bash
# Fast check in sim
python main.py --rule-baseline --eval-sim --eval-episodes 5 --no-ui

# Real game (starts Chromium + local server)
python main.py --rule-baseline --eval-episodes 3 --no-ui
```

You should see non-zero survival length and some platform landings.

### Step B — Sim pretrain (fast, headless)

```bash
python main.py --pretrain-steps 500000 --no-ui --device gpu
```

Output: `checkpoints/sim_pretrained.keras`

| Budget | Command |
|--------|---------|
| Quick test | `--pretrain-steps 100000` |
| Standard | `--pretrain-steps 500000` |
| Strong prior | `--pretrain-steps 1000000` |

### Step C — Browser fine-tune from sim

**With UI:**

```bash
python main.py --transfer-from-sim
```

**Headless:**

```bash
python main.py --transfer-from-sim --no-ui
```

**Time-limited session:**

```bash
python main.py --transfer-from-sim --no-ui --finetune-seconds 600
```

Output: `checkpoints/dqn_latest.keras` (auto-saved during training)

### Step D — Resume later

```bash
python main.py                          # resumes dqn_latest.keras
python main.py --transfer-from-sim      # reload sim weights and fine-tune again
```

### Other training modes

```bash
# Sim only (debugging)
python main.py --sim --no-ui

# Full pipeline: calibrate → sim pretrain → browser fine-tune
python main.py --run-pipeline --no-ui

# Long unattended browser sessions
python main.py --overnight-train --no-ui
```

---

## 4. Evaluate progress

### Command-line evaluation (no UI)

```bash
# Learned policy
python main.py --evaluate-policy --eval-episodes 10 --no-ui

# Rule baseline for comparison
python main.py --rule-baseline --eval-episodes 10 --no-ui

# In sim (faster)
python main.py --evaluate-policy --eval-sim --eval-episodes 10 --no-ui
```

Example output:

```
Learned policy: episodes=10 mean_score=... landing_rate=... meaningful_boost=...
```

### Training logs

Every session writes to `logs/`:

| File | Contents |
|------|----------|
| `logs/training_latest.log` | Pointer to latest session |
| `logs/training_latest_browser.log` | Latest browser session |
| `logs/training_<timestamp>_sim.log` | Sim pretrain |
| `logs/training_<timestamp>_browser.log` | Browser training |

Follow browser training live:

```bash
tail -f logs/training_latest_browser.log
```

The log path is printed at startup.

### Metrics that matter

Watch these per episode (in logs and evaluation output):

| Metric | What it means | Good sign |
|--------|---------------|-----------|
| `landing_rate` | Landings per step | Rising above ~0.01 |
| `mean_platform_dx` | Horizontal distance to target | Decreasing |
| `meaningful_boost_rate` | Jumps when boost helps | Higher than wasted boost |
| `max_score` | Best score in episode | Climbing past hundreds, then 800+ |
| `loss` | Training loss | Trending down (sim pretrain) |
| `loop_hz` | Browser steps/sec | Above ~15 |

### UI status bar

During UI training you also see: current action, reward, score, boost level, epsilon, loss, and score velocity.

### Checkpoints

| File | Role |
|------|------|
| `checkpoints/sim_pretrained.keras` | Sim-only weights |
| `checkpoints/dqn_latest.keras` | Browser training / resume point |

---

## 5. Command cheat sheet

```bash
python main.py                                    # UI training
python main.py --watch                            # inference only
python main.py --pretrain-steps 500000 --no-ui    # sim pretrain
python main.py --transfer-from-sim                # browser fine-tune (UI)
python main.py --transfer-from-sim --no-ui        # browser fine-tune (headless)
python main.py --rule-baseline --eval-sim --no-ui # sanity check
python main.py --evaluate-policy --no-ui          # test learned policy
python main.py --device cpu                       # force CPU
```

### Seeded maps (curriculum)

Local `game/` supports fixed layouts via `runSeed`:

- URL: `...?runSeed=424242`
- Config: `CHART_TRIAL_CONFIG.runSeed = 424242` in `game/.../config.js`
- Agent API: `__ASCENT_SET_RUN_SEED__(n)` / `__ASCENT_GET_RUN_SEED__()`; seed also on `__ASCENT_AGENT__.runSeed`
- Python: `BrowserConfig.run_seed = 424242` (pushed on connect); `backend.set_run_seed(n)`

Same seed → same initial staircase/boosters. FX stay random. Smoke check: `python scripts/smoke_run_seed.py`

---

## 6. Quick troubleshooting

| Problem | Try |
|---------|-----|
| Rule baseline fails in sim | State/reward pipeline is broken — check tests: `python -m unittest discover -s tests` |
| Browser won't connect | Ensure port 8765 is free; app auto-starts the game server |
| KDE Wayland crash on Chromium launch | Use an X11 Plasma session, or rely on auto X11 Chromium args (see TRAINING_GUIDE) |
| Scores flat for hours | Run `--evaluate-policy` and compare to `--rule-baseline`; pretrain longer or record demos |
| `GpuNotAvailableError` | Run `nvidia-smi`; try `--device cpu` or `./scripts/run_with_gpu.sh` |
