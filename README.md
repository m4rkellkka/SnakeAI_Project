<p align="center">
  <img src="assets/snake_ai_banner.png" alt="Snake AI Banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch Badge"/>
  <img src="https://img.shields.io/badge/Pygame-Green.svg?style=for-the-badge&logo=python&logoColor=white" alt="Pygame Badge"/>
  <img src="https://img.shields.io/badge/Tauri-v2-FFC131.svg?style=for-the-badge&logo=tauri&logoColor=white" alt="Tauri Badge"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB.svg?style=for-the-badge&logo=react&logoColor=black" alt="React Badge"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg?style=for-the-badge" alt="License Badge"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg?style=for-the-badge" alt="Status Badge"/>
</p>

<h1 align="center">🐍 Snake AI</h1>

<p align="center">
  <strong>A Snake game environment paired with a CNN classifier trained via behavioral cloning + DAgger-lite to imitate a near-perfect Hamiltonian-cycle teacher algorithm.</strong><br>
  <em>This project demonstrates learning-from-demonstration (LfD) with a clean, hands-on implementation.</em>
</p>

<p align="center">
  <b>Language:</b> <a href="#english">English</a> | <a href="#русский">Русский</a>
</p>

---

## English

### Overview

This project trains a neural network to play Snake by learning from a rule-based expert (the Hamiltonian-cycle teacher). The network learns to:

- Navigate a 16×16 grid via relative actions (straight, right, left)
- Reach food while avoiding walls and self-collision
- Use an egocentric + compass-fix state representation for robust decision-making

**Key results:**

- **Best honest evaluation score: 234.9 / 253** (93% of max possible)
- **Perfect teacher baseline:** 253/253 consistently
- Trained with ~350 games using 2.5M+ steps of DAgger-lite sampling

### Demo

| Trained model (`pretrained.pth`) | Untrained (random init) |
|:---:|:---:|
| <img src="assets/demo_pretrained.gif" width="320"> | <img src="assets/demo_untrained.gif" width="320"> |

### Features

- **Supervised learning, not RL** — uses CrossEntropyLoss with teacher labels, no reward shaping
- **DAgger-lite curriculum** — network sees its own mistakes via probability-scheduled policy disturbance
- **Curriculum starts** — agent encounters late-game states early via longer initial snakes
- **Honest evaluation** — separate evaluation loop with no teacher (true learned skill)
- **Hamiltonian cycle teacher** with corner-cutting shortcuts (provably collision-free)
- **Full reproducibility** — includes pretrained checkpoint + training logs
- **Hyperparameter sweeps & benchmarking** — CLI overrides (`--lr`, `--dagger-prob-max`, `--curriculum-prob`, `--run-name`) for isolated, comparable runs, plus `tools/benchmark.py` for head-to-head checkpoint comparison
- **Tauri desktop app** — self-contained GUI (`app/`) with live training charts, one-click watch/benchmark/sweep panels, and per-process management; no Python install exposed to the user

### UI Gallery (Before & After)

**Start screen**

| Before | After (Premium Dark Theme) |
|:---:|:---:|
| <img src="assets/start_old.png" width="400"> | <img src="assets/start_new.png" width="400"> |

**Game Interface**

| Before | After (Modernized) |
|:---:|:---:|
| <img src="assets/game_old.png" width="400"> | <img src="assets/game_new.png" width="400"> |

**Training Dashboard**
<img src="assets/dashboard_new.png" width="800">

### How It Works

#### State Representation

9-channel state array (all normalized to [0, 1]):

- **Channels 0–4 (egocentric, rotated):** head, body, food, danger map (walls + body), board-fullness
  - Rotated so snake's head always faces "up" (egocentric perspective)
- **Channels 5–8 (absolute, non-rotated):** one-hot absolute direction (compass fix)
  - Lets network distinguish turns by absolute grid position, not just local patterns

#### Action Space

Relative to current heading: `[straight, turn right, turn left]` (not absolute directions)

#### The Teacher

`src/teacher.py`: Hamiltonian-cycle algorithm that:

1. Follows a fixed cycle visiting all 256 grid cells (serpentine + highway back)
2. Takes corner-cutting shortcuts to reach food faster
3. Has two safety nets ensuring collision-free movement
4. Achieves 253/253 (100%) on standard starts, 253/253 on curriculum starts

#### Training Loop

1. **Step:** Teacher computes label; network takes action (probabilistically network's own)
2. **Learn:** Supervised loss (CrossEntropy) on teacher's label
3. **Evaluate:** Every 25 games, honest eval (network only, no teacher)
4. **Checkpoint:** Save best eval model

#### DAgger-lite & Curriculum

- **DAgger:** ~0–30% of steps use network's action (ramped over 100k steps), exposing distribution shift
- **Curriculum:** After 150 games, 20% of episodes start with snakes of length 4–50 (late-game states)

### Results

**Training Progression:**

- Games 1–150: Honest eval score climbs unevenly from ~34 to ~134, with a dip to ~48 around game 125
- Games 150–350: Continued (volatile) improvement, peaking at **234.9 / 253** (93%) at game 350
- Games 350–450: Oscillates in the ~185–216 range — late-game "spinning" on a subset of boards
  caps the honest score short of a perfect 253; see `checkpoint_best.pth` vs. `checkpoint_last.pth`
  in the codebase notes for why the peak checkpoint, not the final one, ships as `pretrained.pth`

**Learning Curve:** [assets/learning_curve.png](assets/learning_curve.png)

**Benchmark (100 games, seed 42, `pretrained.pth`):**

| Mode | Mean | Median | Std | Min | Max | Win rate (253/253) | Stuck rate |
|------|------|--------|-----|-----|-----|---------------------|------------|
| With unstick (default) | 226.0 | 253.0 | 56.9 | 45 | 253 | 81% | 0% |
| Without unstick (`--no-unstick`, raw network) | 206.0 | 253.0 | 70.4 | 36 | 253 | 67% | 13% |

The "unstick" crutch detects when the network repeats a `(head, direction)` loop and hands
that one step to the teacher — it removes nearly all loop-timeouts (13% → 0% stuck) and lifts
the mean score by ~20 points. Run it yourself with `tools/benchmark.py` (see [Usage](#usage)).

### Installation

#### Prerequisites

- Python 3.8+
- For `launcher.py`: Tkinter (included on macOS/Windows; on Linux: `apt install python3-tk`)

#### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SnakeAI_Project.git
cd SnakeAI_Project

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** The project uses minimal, standard dependencies: `torch`, `pygame`, `numpy`, `matplotlib`.

### Usage

#### 1. **Watch the Pretrained Model**

```bash
python src/train_ai.py --watch --pretrained --games 5
```

Load the best trained model and watch it play 5 games (recommended: see it in action first).

#### 2. **Watch the Perfect Teacher**

```bash
python src/teacher.py
```

Runs the Hamiltonian-cycle algorithm in real-time. Shows 100% win rate.

#### 3. **Train Your Own Model**

```bash
python src/train_ai.py
```

Starts training from scratch (or resumes from `model/checkpoint_last.pth` if it exists).

- Ctrl+C to stop gracefully
- In-game controls: `+` to speed up, `-` to slow down, `0` for max speed

#### 4. **Test Current Model (No Pretrained)**

```bash
python src/train_ai.py --watch --games 10
```

Load the best checkpoint from your training run.

#### 5. **Tauri Desktop App (Recommended GUI)**

```bash
cd app
npm install       # first time only
npm run tauri dev # dev mode with hot-reload
# or: npm run tauri build  →  produces a native .app / installer
```

Full-featured dark-theme dashboard with four panels:

- **Train** — start/stop training with live score + eval charts and a colored log
- **Watch AI** — one-click buttons for pretrained model, your checkpoint, teacher, or manual play (each opens its own Pygame window)
- **Benchmark** — run N games with configurable checkpoint/seed/unstick toggle
- **Sweep Run** — launch headless training with custom LR, DAgger prob, curriculum prob

Requires: [Node.js](https://nodejs.org) ≥ 18 and [Rust](https://rustup.rs) (for `cargo`/Tauri CLI).

#### 6. **Tkinter Control Panel (Legacy)**

```bash
python launcher.py
```

Simple alternative GUI — still works, no extra dependencies beyond Python + Tkinter.

#### 7. **Custom Game Count**

```bash
python src/train_ai.py --watch --pretrained --games 25
```

#### 8. **Hyperparameter Sweep Run**

```bash
python src/train_ai.py --headless --run-name sweep_lr0.001 \
    --lr 0.001 --dagger-prob-max 0.5 --curriculum-prob 0.3
```

Override `LR`, `DAGGER_PROB_MAX`, or `CURRICULUM_PROB` for a single run. `--run-name` isolates checkpoints and `learning_curve.png` under `model/<run-name>/`, so sweeps don't clobber each other or the main run — each checkpoint also stores its `run_config` for traceability. Or use the **New Sweep Run** dashboard in `launcher.py`.

#### 9. **Benchmark Checkpoints**

```bash
python tools/benchmark.py --checkpoint checkpoint_best.pth \
    --checkpoint sweep_lr0.001/checkpoint_best.pth \
    --games 100 --seed 42 --plot benchmark.png
```

Runs N games per checkpoint and reports full score distributions (mean/median/std/percentiles/win rate) plus a side-by-side comparison table — sharper signal than avg/max alone. Or use the **Benchmark Models** dashboard in `launcher.py`.

### Project Structure

| File / Directory | Purpose |
| ---------------- | ------- |
| `src/snake_game.py` | Game engine (`SnakeGameAI`), grid constants, Hamiltonian cycle |
| `src/teacher.py` | Perfect Hamiltonian-cycle algorithm with shortcuts; runnable demo |
| `src/train_ai.py` | Network (`SnakeNet`), replay buffer, trainer, agent, main training/eval loop |
| `src/play_manual.py` | Human-playable Snake (WASD / Arrow keys), same Pygame window |
| `app/` | Tauri v2 desktop app (React 19 frontend + Rust backend) — full GUI control panel |
| `launcher.py` | Legacy Tkinter control panel |
| `tools/record_demo.py` | Utility to record gameplay as animated GIFs |
| `tools/benchmark.py` | Benchmark harness — run N games per checkpoint, report score distributions & comparisons |

### Understanding the Code

**Key Classes:**

- `SnakeGameAI` — Core game engine (state, collisions, food, reset)
- `SnakeNet` — CNN classifier (9→16→32→32→8192→128→3 logits)
- `Agent` — Network wrapper with state representation, action selection, checkpointing
- `Trainer` — Adam optimizer + CrossEntropyLoss training loop
- `ReplayBuffer` — Simple ring buffer of (state, action) pairs

**Key Functions:**

- `play_step(action)` — Single game step; returns (game_over, score)
- `safe_moves()` — Computes which of the 3 actions avoid immediate collision
- `get_state(game)` — Builds 9-channel state with rotation and compass encoding
- `get_network_action(state, game)` — Forward pass + safe_moves masking
- `get_best_move(game)` — Teacher's Hamiltonian-cycle logic with shortcuts

### Customization

**Hyperparameters** (edit `src/train_ai.py`):

- `BATCH_SIZE` — Training batch size (default 128)
- `LR` — Learning rate (default 0.0005)
- `DAGGER_PROB_MAX` — Max probability of network-driven steps (default 0.7)
- `CURRICULUM_PROB` — Probability of curriculum start (default 0.2)
- `EVAL_GAMES` — Games per evaluation (default 15)

**Network Architecture** (in `SnakeNet.__init__`):

- Modify conv layer channels, fully-connected layer sizes, etc.
- Input: 9 channels, output: 3 logits (must match action space)

### Troubleshooting

**`model/pretrained.pth` not found?**

- The file is included in the repo; if missing, the network trains untrained (random initialization)

**Tkinter not installed (Linux)?**

- Run: `apt install python3-tk`

**Game window unresponsive during training?**

- Normal — rendering is throttled. Speed keys still work. Training finishes normally.

**Out of memory?**

- Reduce `MAX_MEMORY` (replay buffer capacity) in `src/train_ai.py`

### Roadmap

Our planned next steps are grouped into milestones, leading up to our first stable release and beyond.

#### 🚀 Version 1.0 (Upcoming First Stable Release)
Our immediate goal is to finalize the behavioral cloning baseline and polish the presentation before stamping our first official release on GitHub.

- **Finalize Feature Polish** — Add a few remaining features to the benchmark tooling and Tauri app.
- **Standalone Application** — Tauri v2 desktop app (`app/`) is in progress; goal is a native installer so anyone can run without installing Python.
- **Documentation & Presentation** — Complete the Mermaid diagram of the training pipeline, add a Table of Contents, and potentially a browser-playable demo (`pygbag`).

#### 🧠 Version 2.0 (New Training Paradigms)
After releasing v1.0, the focus shifts to entirely new methods of training beyond imitation learning.
- **Reinforcement Learning (RL)** — Implement DQN/PPO trained from scratch (`src/train_rl.py`), adding a reward wrapper without breaking the existing environment API.
- **Multi-snake Environment** — Generalize the grid for N snakes (snake-vs-snake collisions, shared food, opponent-aware state channels).
- **Self-play / AI vs AI** — Train agents against each other in the new environment.
- **Ensemble Benchmarking** — Compare BC, RL, and self-play models head-to-head via the benchmark harness.
- **Obstacles & Custom Boards** — Since the Hamiltonian teacher can't handle random obstacles, RL agents will step up to learn dynamic obstacle avoidance.

#### 🎮 Future Expansions
- **Human vs AI mode** — Extend `play_manual.py` to the multi-snake environment.
- **Tournament / Leaderboard tab** — Pick checkpoints/modes, run them through the benchmark harness, and show a results table right in `launcher.py`.
- **Architecture Experiments** — Residual blocks, deeper conv stacks, and alternative FC sizes for `SnakeNet`.

### License

This project is released under the **MIT License** — see [LICENSE](LICENSE) for details.

### Acknowledgments

- Inspired by behavioral cloning + DAgger techniques in imitation learning
- Hamiltonian cycle concept for deterministic, collision-free pathfinding
- Built with PyTorch, Pygame, and NumPy

---

<a name="русский"></a>

## Русский

[Перейдите к README.ru.md](README.ru.md) для русской версии документации.

---

**Questions or contributions?** Open an issue or submit a pull request!
