# Ascent Neural Network Player

A Python reinforcement-learning app that learns to play
[Ascent](https://ascent.xrd.workers.dev/) from browser canvas screenshots.

The app uses:

- Playwright to control Chromium and capture the `#gameCanvas` element.
- TensorFlow/Keras for a DQN CNN policy.
- PyQt6 for a desktop control panel and live preview.
- OpenCV/NumPy for frame preprocessing and reward signals.

The game controls are `A` for left, `D` for right, and `Space` for jump/boost.

## Setup

**Python 3.10–3.12 required.** TensorFlow does not publish wheels for Python
3.13+ yet. On this machine, use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

If you see `No matching distribution found for tensorflow`, your venv was
probably created with a newer Python (for example 3.14). Recreate it with
`python3.11 -m venv .venv`.

TensorFlow will use a compatible GPU when one is available via
`tensorflow[and-cuda]`. For CPU-only installs:

```bash
pip install tensorflow
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

## Run

```bash
python main.py
python main.py --no-auto-launch
python main.py --cdp http://localhost:9222
python main.py --watch
python main.py --device gpu
python main.py --device cpu
python main.py --chromium-path /usr/bin/chromium
```

## UI

The top browser panel auto-attaches when possible, or lets you rescan, attach to
a CDP URL, pick a Chromium window, or force-launch a new browser. Training
controls stay disabled until `#gameCanvas` is detected.

The right panel controls DQN hyperparameters and mode:

- Train: epsilon-greedy learning with replay.
- Watch: inference only, no training.
- Paused: connected but idle.

The status bar shows browser connection, FPS, replay size, loss, compute
device, and train-step latency.

## Notes

This is a learning scaffold, not a pre-trained bot. Expect early episodes to be
mostly random. The DQN improves by collecting transitions, training from replay,
and saving checkpoints under `checkpoints/`.

## Human demonstrations

See **[DEMO_RECORDING.md](DEMO_RECORDING.md)** for full instructions on recording
your own playthroughs and training the agent from them.

Quick summary:

1. Click **Record demo** → play with `A` / `D` / `Space` → **Stop recording**
2. Demos save to `demonstrations/`
3. Keep **Load demonstrations on start** checked, then click **Start**
