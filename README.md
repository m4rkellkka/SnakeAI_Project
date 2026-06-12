# 🐍 Snake AI

A Snake game environment paired with a CNN classifier trained via **behavioral cloning + DAgger-lite** to imitate a near-perfect **Hamiltonian-cycle teacher** algorithm. This project demonstrates learning-from-demonstration (LfD) with a clean, hands-on implementation.

**Language:** [English](#english) | [Русский](#русский) ([README.ru.md](README.ru.md))

---

## English

### Overview

This project trains a neural network to play Snake by learning from a rule-based expert (the Hamiltonian-cycle teacher). The network learns to:

- Navigate a 16×16 grid via relative actions (straight, right, left)
- Reach food while avoiding walls and self-collision
- Use an egocentric + compass-fix state representation for robust decision-making

**Key results:**

- **Best honest evaluation score: 182.2 / 253** (72% of max possible)
- **Perfect teacher baseline:** 253/253 consistently
- Trained with ~275 games using 300k+ steps of DAgger-lite sampling

### Features

- **Supervised learning, not RL** — uses CrossEntropyLoss with teacher labels, no reward shaping
- **DAgger-lite curriculum** — network sees its own mistakes via probability-scheduled policy disturbance
- **Curriculum starts** — agent encounters late-game states early via longer initial snakes
- **Honest evaluation** — separate evaluation loop with no teacher (true learned skill)
- **Hamiltonian cycle teacher** with corner-cutting shortcuts (provably collision-free)
- **Full reproducibility** — includes pretrained checkpoint + training logs

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

- Games 1–25: Honest eval score climbs from ~30 to ~60
- Games 25–150: Score stabilizes ~110–140, with some variance
- Games 150–275: Score improves to **182.2**, reaching **253/253** in several eval runs

**Learning Curve:** [assets/learning_curve.png](assets/learning_curve.png)

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

#### 5. **Tkinter Control Panel (All-in-One)**

```bash
python launcher.py
```

Simple GUI to launch training, watch games, view stats, and stop processes.

#### 6. **Custom Game Count**

```bash
python src/train_ai.py --watch --pretrained --games 25
```

### Project Structure

| File | Purpose |
|------|---------|
| `src/snake_game.py` | Game engine (`SnakeGameAI`), grid constants, Hamiltonian cycle |
| `src/teacher.py` | Perfect Hamiltonian-cycle algorithm with shortcuts; runnable demo |
| `src/train_ai.py` | Network (`SnakeNet`), replay buffer, trainer, agent, main training/eval loop |
| `launcher.py` | Tkinter control panel for easy access to all features |
| `tools/record_demo.py` | Utility to record gameplay as animated GIFs |

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

Planned next steps, grouped by track. Tracks aren't strictly sequential — some have dependencies, noted inline.

#### Foundation

- **Benchmark harness** — script to run N games across one or more checkpoints and report full score distributions (not just avg/max). Sharper signal for comparing hyperparameters/architectures, and the base for the tournament/leaderboard below.

#### Track A — Model quality (current BC + DAgger pipeline)

- Hyperparameter sweeps: `DAGGER_PROB_MAX` (0.7), `CURRICULUM_PROB`, `LR`, evaluated via the benchmark harness
- `SnakeNet` architecture experiments — residual blocks, deeper conv stack, alternative FC sizes
- Longer training runs beyond the current 275-game checkpoint

#### Track B — New training paradigms

- **Reward layer for RL** — additive wrapper deriving a reward signal from `(game_over, score)`, without changing `play_step`'s existing contract (BC stays reward-free by design)
- **`src/train_rl.py`** — DQN/PPO trained from scratch (no teacher), reusing the 9-channel state representation
- **Multi-snake environment** — generalize `SnakeGameAI` to N snakes: snake-vs-snake collisions, shared food, opponent-aware state channels
- **Self-play / AI vs AI** — agents trained against each other on the multi-snake environment
- **Ensemble / model comparison** — run BC, RL, and self-play checkpoints head-to-head via the benchmark harness

#### Track C — Game modes

- **Human vs AI** — extend `play_manual.py` to the multi-snake environment, one snake human-controlled
- **Other boards / obstacles** — generalize `GRID_SIZE` and add obstacle cells. The Hamiltonian-cycle teacher doesn't generalize to obstacle maps, so this depends on Track B (an RL-trained agent, or a new BFS/A*-based "imperfect" teacher)
- **Tournament / leaderboard tab in `launcher.py`** — pick checkpoints/modes, run them through the benchmark harness, show a results table

#### Presentation polish

- Animated GIF of the trained agent playing, at the very top of the README (use the existing `tools/record_demo.py`)
- Status badges (Python version, license, framework)
- Table of contents for easier navigation
- Mermaid diagram of the training pipeline (env → teacher/DAgger → trainer → checkpoint → eval)
- Optional: browser-playable demo (pygame → wasm via `pygbag`)
- **Standalone Application** — create a fully self-contained executable so anyone can run the program on their OS out-of-the-box like a regular application, without installing Python, Visual Studio, or other dependencies

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
