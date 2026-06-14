# Ascent Neural Network Player

A Python reinforcement-learning app that learns to play
[Ascent](https://ascent.xrd.workers.dev/) from browser canvas screenshots.

The app uses:

- Playwright to control Chromium and capture the `#gameCanvas` element.
- TensorFlow/Keras for a DQN CNN policy.
- PyQt6 for a desktop control panel and live preview.
- OpenCV/NumPy for frame preprocessing and reward signals.

The game controls are `A` for left, `D` for right, and `Space` for jump/boost.

---

## Setup

**Python 3.10–3.12 required.** TensorFlow does not publish wheels for Python
3.13+ yet. On this machine, use Python 3.11:

```bash
cd "Ascent Player"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

If you see `No matching distribution found for tensorflow`, your venv was
probably created with a newer Python (for example 3.14). Recreate it with
`python3.11 -m venv .venv`.

### GPU (recommended)

TensorFlow uses your **NVIDIA GPU by default** via `tensorflow[and-cuda]`
(bundled CUDA/cuDNN wheels). The proprietary NVIDIA driver must be installed:

```bash
nvidia-smi
```

If that works, pretraining and training will use the GPU automatically. You should
see `device=GPU: /physical_device:GPU:0` in pretrain output.

For CPU-only debugging:

```bash
python main.py --device cpu
```

Optional wrapper (sets `LD_LIBRARY_PATH` before Python starts):

```bash
./scripts/run_with_gpu.sh --pretrain-steps 100000 --no-ui
```

---

## Simulator pretraining (step by step)

Pretraining runs a **headless physics simulator** at hundreds of steps per second
(no browser, no Playwright). It teaches gravity, platform navigation, and boost
gating cheaply. You then **fine-tune in the real game** with the sim checkpoint.

### Overview

```
1. Pretrain in sim  →  checkpoints/sim_pretrained.keras
2. Fine-tune in browser  →  checkpoints/dqn_latest.keras
3. (Optional) Record high-score demos  →  demonstrations/
```

### Step 1 — Activate the environment

```bash
cd "Ascent Player"
source .venv/bin/activate
```

### Step 2 — (Optional) Calibrate the simulator

Runs a short random-policy report so you can confirm the sim is working:

```bash
python main.py --calibrate-sim
```

Example output:

```
sim calibration: episodes=7 mean_score=133.0 mean_length=651.7
```

### Step 3 — Run pretraining

Pick a step budget. Recommended starting points:

| Goal | Steps | Approx. time (GPU) |
|------|-------|--------------------|
| Quick test | 100,000 | ~3 min |
| Standard | 500,000 | ~15 min |
| Strong prior | 1,000,000 | ~30 min |

Run headless pretrain (no UI):

```bash
python main.py --pretrain-steps 500000 --no-ui
```

What happens:

- 16 parallel sim environments (auto from CPU count; override with `--sim-envs N`)
- Batched GPU inference and training
- Progress every 5,000 steps: `sps`, `loss`, `epsilon`, `replay`
- Checkpoint saved to **`checkpoints/sim_pretrained.keras`**

Example output:

```
Fast sim pretrain: 16 envs, batch=128, train_every=32, device=GPU: /physical_device:GPU:0
sim step=5000/500000 sps=650 loss=0.001 eps=0.961 replay=5000
...
Saved sim checkpoint to checkpoints/sim_pretrained.keras (500000 steps in 768.2s, 651 sps)
```

**Confirm the checkpoint exists:**

```bash
ls -lh checkpoints/sim_pretrained.keras
```

### Step 4 — Fine-tune in the real browser (apply pretrain)

Load the sim weights and continue training on the actual Ascent game. The app
uses a lower learning rate and restarts exploration (ε ≈ 0.3 → 0.05).

#### Option A — PyQt UI (recommended)

1. Start Chromium with remote debugging (if not auto-launching):

   ```bash
   chromium --remote-debugging-port=9222 https://ascent.xrd.workers.dev/
   ```

2. Launch the app with transfer enabled:

   ```bash
   python main.py --transfer-from-sim
   ```

3. Click **Start** in the UI.

4. On startup you should see a dialog like:

   ```
   Loaded sim pretrain from sim_pretrained.keras — fine-tuning with ε=0.30
   ```

5. Watch the progress panel — **score velocity** (Δscore/step) is a leading
   indicator of improvement. Best score updates as episodes complete.

6. Checkpoints auto-save to **`checkpoints/dqn_latest.keras`** during fine-tune.

#### Option B — Headless browser fine-tune

```bash
python main.py --transfer-from-sim --no-ui
```

Requires a reachable Ascent tab or auto-launch Chromium.

### Step 5 — (Optional) Add human demonstrations

After sim pretrain, high-score human demos help with mid/late-game decisions
(gap jumps, boost timing). See **[DEMO_RECORDING.md](DEMO_RECORDING.md)**.

1. Record demos targeting **2000+** score.
2. Keep **Load demonstrations on start** checked in the UI.
3. Start training — demos are ingested with score-weighted sampling and BC warm-up.

### Step 6 — Resume later

| Situation | Command |
|-----------|---------|
| Resume browser training | `python main.py` (loads `dqn_latest.keras` automatically) |
| Re-run sim pretrain only | `python main.py --pretrain-steps 500000 --no-ui` |
| Fine-tune again from sim | `python main.py --transfer-from-sim` |

Sim and browser checkpoints are separate:

- `checkpoints/sim_pretrained.keras` — sim pretrain only
- `checkpoints/dqn_latest.keras` — browser training / resume

### Pretraining CLI reference

```bash
# Standard pretrain
python main.py --pretrain-steps 500000 --no-ui

# More parallel sim envs (if CPU has headroom)
python main.py --pretrain-steps 500000 --no-ui --sim-envs 20

# Force GPU
python main.py --pretrain-steps 500000 --no-ui --device gpu

# Calibrate sim physics
python main.py --calibrate-sim

# Train only in sim (no transfer) — debugging
python main.py --sim --no-ui
```

### Troubleshooting pretrain

| Problem | What to do |
|---------|------------|
| `GpuNotAvailableError` | Run `nvidia-smi`. Reinstall: `pip install -r requirements.txt`. Try `./scripts/run_with_gpu.sh`. |
| Very slow (`sps` &lt; 100) | Confirm output shows `device=GPU`. Close other GPU-heavy apps. |
| Loss stays huge (&gt; 100) | You may be on an old checkpoint; delete `checkpoints/` and pretrain again. |
| Fine-tune does not improve | Pretrain longer (500k–1M steps). Record 2000+ demos. Let ε explore (don't use `--watch`). |
| Transfer dialog says "No checkpoint" | Run Step 3 first; verify `checkpoints/sim_pretrained.keras` exists. |

### Training diagnostics log

Every training session writes a detailed text log to **`logs/`**:

- **`logs/training_<timestamp>_sim.log`** — sim pretrain
- **`logs/training_<timestamp>_browser.log`** — browser fine-tune / UI training
- **`logs/training_latest.log`** — pointer to the most recent session (any phase)
- **`logs/training_latest_browser.log`** — pointer to the most recent browser session

The log path is printed at startup (`Training log: logs/...`). Inspect it when
progress stalls; each summary block includes automatic **DIAGNOSIS** hints.

Logged metrics include:

- Loss, epsilon, replay size, weight norm drift
- Action distribution (noop / jump / turn rates)
- Score deltas, episode lengths, recent score trends
- Boost depletion and jump-while-depleted counts
- Per-episode reward and max score
- **Browser only:** loop Hz, step latency, orb position, platform distance
- **Browser only:** detailed `STEP` lines every 100 steps (score, boost, actions)
- Checkpoint load/save events

```bash
# Browser training (UI) — log path shown in startup dialog and status bar
python main.py --transfer-from-sim

# Browser training (headless)
python main.py --transfer-from-sim --no-ui

# Follow the latest browser log
cat logs/training_latest_browser.log
tail -f logs/training_*_browser.log
```

---

## Run (browser / UI)

```bash
python main.py
python main.py --no-auto-launch
python main.py --cdp http://localhost:9222
python main.py --watch
python main.py --device gpu
python main.py --device cpu
python main.py --chromium-path /usr/bin/chromium
python main.py --transfer-from-sim          # fine-tune after sim pretrain
```

## Existing Chromium Auto-Detect

The app first scans local Chrome DevTools Protocol ports (`9222`-`9229`) for an
existing Chromium tab at the Ascent URL. To make your normal Chromium window
detectable, start it with remote debugging enabled:

```bash
chromium --remote-debugging-port=9222 https://ascent.xrd.workers.dev/
```

On Arch Linux, you can also add this line to Chromium flags:

```bash
--remote-debugging-port=9222
```

If no existing Ascent tab is found, the app launches a visible Chromium window
by default.

## UI

The top browser panel auto-attaches when possible, or lets you rescan, attach to
a CDP URL, pick a Chromium window, or force-launch a new browser. Training
controls stay disabled until `#gameCanvas` is detected.

The right panel controls DQN hyperparameters and mode:

- **Train** — epsilon-greedy learning with replay.
- **Watch** — inference only, no training.
- **Paused** — connected but idle.

The status bar shows browser connection, loop Hz, replay size, loss, compute
device, best score, and score velocity.

## Notes

This is a learning scaffold, not a pre-trained bot. Expect early episodes to be
mostly random until pretrain + fine-tune. The DQN improves by collecting
transitions, training from replay, and saving checkpoints under `checkpoints/`.

**Recommended workflow:** sim pretrain → browser fine-tune (`--transfer-from-sim`)
→ record high-score demos if progress stalls.

## Human demonstrations

See **[DEMO_RECORDING.md](DEMO_RECORDING.md)** for full instructions on recording
your own playthroughs and training the agent from them.

Quick summary:

1. Click **Record demo** → play with `A` / `D` / `Space` → **Stop recording**
2. Demos save to `demonstrations/`
3. Keep **Load demonstrations on start** checked, then click **Start**
