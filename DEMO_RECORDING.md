# Recording Human Demonstrations

This guide explains how to record your own Ascent playthroughs so the agent can
learn from them before reinforcement learning continues.

## What gets recorded

Each demonstration is saved as a compressed `.npz` file in the `demonstrations/`
folder. For every game step the app stores:

| Field | Description |
|-------|-------------|
| `states` | Preprocessed CNN input (84×84 frames + boost channel) |
| `actions` | Discrete action id (0–5) derived from your keys |
| `rewards` | Shaped reward signal for that transition |
| `next_states` | Observation after the action |
| `dones` | Whether the run ended on that step |
| `scores` | In-game height score at that step |
| `episode_ids` | Which run within the recording session |
| `peak_score` | Highest score reached in the file |

High-score segments (1500+) are weighted more heavily during ingest. Record sessions
that reach **2000+** when possible — timed boosts through mid-game gaps teach skills
the agent rarely discovers on its own.

Files are named like `demonstrations/demo_20260614_153045.npz`.

## Prerequisites

1. Complete the normal setup in [README.md](README.md) (Python 3.11 venv, deps,
   Playwright Chromium).
2. Launch the app:
   ```bash
   source .venv/bin/activate
   python main.py
   ```
3. Have Ascent reachable in Chromium — either:
   - Already open at `https://ascent.xrd.workers.dev/` with CDP enabled, or
   - Let the app auto-launch / attach a browser for you.

## Step-by-step recording

### 1. Start recording

1. Open the Ascent Neural Network Player window.
2. Click **Record demo** in the right panel.
3. Wait for the status bar message:
   ```
   Recording: play in the browser with A / D / Space. Click Stop when done.
   ```
4. **Click inside the game browser window** so it has keyboard focus.

### 2. Play the game

Use the same controls as normal Ascent:

| Key | Action |
|-----|--------|
| `A` or `←` | Move left |
| `D` or `→` | Move right |
| `Space` | Jump / boost |

Tips for useful demos:

- Play for at least **30–60 seconds** per recording.
- Mix horizontal movement with **timed** Space boosts — don't mash Space.
- Try to survive longer and climb higher; the agent learns from your survival
  patterns and boost usage.
- Collect yellow orbs and green boost arrows when you can.
- It is fine to record multiple shorter runs; each becomes a separate file.

The live preview in the app updates while you play. The status bar shows your
current mapped action and frame count, for example:

```
Recording demo | action=left+jump | frames=142
```

### 3. If you die during recording

When the game shows the death screen, the recorder **automatically restarts** a
new run and keeps recording into the **same session**. You can chain several
attempts in one demo file.

### 4. Stop and save

1. Click **Stop recording** in the app.
2. The demo is written to `demonstrations/demo_<timestamp>.npz`.
3. The status bar confirms the path and transition count, for example:
   ```
   Saved demonstration: demonstrations/demo_20260614_153045.npz (512 transitions)
   ```

## Train from your demonstrations

1. Ensure **Load demonstrations on start** is checked in the DQN parameters panel.
2. Click **Start** (not Record demo).
3. On startup the agent will:
   - Load every `.npz` file in `demonstrations/`
   - Insert transitions into replay memory (3× oversampling)
   - Run a short **behavior-cloning warm-up** (~300 gradient steps)
4. Watch the status bar for confirmation:
   ```
   Loaded 1536 demo transitions | BC loss 0.0412
   ```
5. RL training continues after the warm-up.

## Recording multiple demos

You can record as many sessions as you want:

1. **Record demo** → play → **Stop recording**
2. Repeat

All files in `demonstrations/` are loaded the next time you click **Start**.

To remove a bad demo, delete its `.npz` file from `demonstrations/`.

## Action mapping reference

Your held keys each frame are converted to one of six actions:

| Action id | Label | Keys |
|-----------|-------|------|
| 0 | noop | (none) |
| 1 | left | A / ← |
| 2 | right | D / → |
| 3 | jump | Space |
| 4 | left+jump | A + Space |
| 5 | right+jump | D + Space |

## Troubleshooting

### Keys are not detected

- Click the **game browser window** so it has focus.
- Make sure you are recording (status bar says `Recording demo`).
- If you attached via CDP to an existing Chromium tab, keep that tab focused.

### No file after stopping

- You must click **Stop recording** — closing the app without stopping may lose
  the session.
- At least one frame transition is required; play for a few seconds before
  stopping.

### Demos not loaded on training start

- Confirm **Load demonstrations on start** is checked.
- Confirm `.npz` files exist in `demonstrations/`.
- Check the status bar after clicking **Start** for a `Loaded N demo transitions`
  message.

### App crashes on launch

- Use Python 3.11 and an activated `.venv` (see main README).

## File layout

```
Ascent Player/
├── demonstrations/
│   ├── demo_20260614_150012.npz
│   └── demo_20260614_153045.npz
├── DEMO_RECORDING.md          ← this file
└── README.md
```
