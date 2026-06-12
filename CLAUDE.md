# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Snake game environment (Pygame, 16x16 grid / 640x640px) paired with a CNN classifier
trained via **behavioral cloning + DAgger-lite** to imitate a near-perfect Hamiltonian-cycle
"teacher" algorithm. Four files:

- [snake_game.py](snake_game.py) - game engine/environment (`SnakeGameAI`), plus the shared
  `GRID_SIZE` and `HAMILTONIAN_CYCLE` constants.
- [teacher.py](teacher.py) - the "perfect" Hamiltonian-cycle algorithm with corner-cutting
  shortcuts (`get_best_move`); also runnable standalone as a visual demo.
- [train_ai.py](train_ai.py) - network (`SnakeNet`), replay buffer, trainer, agent, and the
  training/eval/checkpoint loop.
- [launcher.py](launcher.py) - small Tkinter control panel that launches the above as
  subprocesses.

## Commands

```bash
python3 train_ai.py                          # train from scratch / resume from checkpoint_last.pth (Ctrl+C to stop)
python3 train_ai.py --watch                  # load checkpoint_best.pth, watch 10 greedy games
python3 train_ai.py --watch --pretrained     # load model/pretrained.pth (showcase model) instead
python3 train_ai.py --watch --games 25       # custom game count for either --watch mode
python3 teacher.py                           # watch the Hamiltonian-cycle teacher play (no learning)
python3 launcher.py                          # GUI control panel for all of the above
```

There are no linters or tests. Dependencies are in `requirements.txt`: `torch`, `pygame`,
`numpy`, `matplotlib`.

During a run, the Pygame window accepts live controls: `=`/`+` increases game speed, `-`
decreases it, `0` removes the FPS cap (max speed - used during training).

## Architecture - the non-obvious parts

**This is supervised learning (behavioral cloning), not RL.** `train_ai.py` trains
`SnakeNet` with `CrossEntropyLoss` against the teacher's action label for each state. There
is no reward, no Q-values, no target network, no replay-buffer bootstrapping - the buffer
just stores `(state, teacher_action)` pairs and is sampled uniformly.

**State representation (`Agent.get_state`):** the board is a `GRID_SIZE x GRID_SIZE`
(16x16, `BLOCK_SIZE=40`, board 640x640) grid. State is a 9-channel `(9, 16, 16)` float
array: channels 0–4 are egocentric (rotated so head faces "up"), channels 5–8 encode
absolute direction (one-hot, non-rotated). Channels 0–4: head, body, food, danger map
(border ring + body cells), board-fullness signal `len(snake)/256`. Channels 5–8: absolute
direction one-hot ("compass fix" — lets the net distinguish cycle turns by absolute
position, e.g., even vs. odd columns in the serpentine Hamiltonian cycle, not just local
patterns). The grid channels 0–4 are rotated with `np.rot90` so the head faces "up"
(egocentric view); channels 5–8 remain absolute. `state.copy()` is required because `rot90`
returns a negative-stride view PyTorch can't consume.

**Action space is relative:** `[straight, right, left]` relative to the snake's current
heading (`SnakeGameAI._move`), not absolute directions. `SnakeNet`'s output layer has 3
logits matching this.

**`play_step(action)` returns `(game_over, score)`** - no reward shaping. Game-over
conditions: wall/self collision (`is_collision`), or `frame_iteration > 100 * len(snake)`
(timeout, reset on every food eaten). Win condition: `len(snake) >= grid_cells` (256),
checked *before* `_place_food` so food is never placed on a full board. `_place_food`
enumerates free cells and samples from them - never loops forever on a crowded board.

**`safe_moves()` and `is_collision()` are kept consistent on purpose.** In `play_step`,
`will_eat` is computed *before* `insert(0, head)`, and the tail `pop()` happens immediately
after `insert` (before the collision check) - this matches `safe_moves()`'s assumption that
moving into the current tail cell is safe (the tail vacates on the same step). Getting this
out of sync was the root cause of the teacher occasionally "colliding" with a cell that was
actually free.

### The teacher (`teacher.py`)

`HAMILTONIAN_CYCLE` (defined once in `snake_game.py`, imported by both `reset()` and
`teacher.py`) is a single cycle visiting all 256 cells: a serpentine sweep of columns
`x=1..15` (alternating direction) plus row `y=0` as a "highway" back to the start.

`get_best_move(game)` normally just follows the cycle one step at a time, but takes
**corner-cutting shortcuts**: if a neighboring cell is closer to the apple (in
cycle-distance) than the next cycle cell, isn't currently occupied by the snake's body, and
leaves at least as much free cycle-space ahead of the head as before (with a 75%-fullness
safety margin), it jumps there instead. Two safety nets make this provably collision-free
(verified via headless fuzz testing - 300/300 standard-start games and 500/500
curriculum-start games reach the maximum score 253 with zero collisions):

1. Shortcut candidates are checked directly against `body_cells` (not just cycle-distance
   arithmetic, which becomes an unreliable proxy once shortcuts make the body
   non-contiguous in cycle order).
2. A final fallback: if the chosen action isn't in `game.safe_moves()` but another action
   is, pick the safe action whose target cell has the smallest cycle-distance from the head
   (stays closest to the intended route).

**Curriculum starts and the teacher share the same cycle on purpose.** `reset(start_length=N)`
builds a curriculum-start snake as a contiguous segment of `HAMILTONIAN_CYCLE` - this keeps
the body contiguous in cycle-index space, which the teacher's `avail_space`/shortcut
heuristics rely on. A mismatched cycle (an earlier boustrophedon-vs-Hamiltonian mismatch)
broke this invariant and caused real collisions on curriculum starts.

### Training (`train_ai.py`)

**`SnakeNet`**: `conv(5→16, k3, p1) → conv(16→32) → conv(32→32) → flatten(32*16*16=8192) →
fc(128) → fc(3)`, returns logits over `[straight, right, left]`. Plain classifier, no
dueling V/A heads.

**`ReplayBuffer`**: a simple `(state, action)` ring buffer (`MAX_MEMORY=50_000`), uniform
`random.sample` for minibatches. None of `next_state`/`reward`/`done`/PER/n-step returns
exist here - they're not meaningful for a CE-loss classifier.

**`Trainer.train_step`**: Adam (`LR=0.0005`) optimizing `CrossEntropyLoss(logits,
argmax(actions))` with gradient clipping (`GRAD_CLIP_NORM=10`). Returns the scalar loss for
logging. One batch (`BATCH_SIZE=128`) every `TRAIN_EVERY_N_STEPS=16` environment steps.

**DAgger-lite (`dagger_prob`)**: every step, the teacher's action is *always* computed and
pushed to the buffer as the label. With probability `dagger_prob(total_steps)` (linearly
ramped 0 → `DAGGER_PROB_MAX=0.3` over `DAGGER_RAMP_STEPS=100_000` total steps), the
*environment* is stepped with the network's own action (`get_network_action`) instead of
the teacher's - so the network sees states its own (imperfect) policy produces, not just
the teacher's idealized trajectory. This is the standard fix for the behavioral-cloning
train/play distribution shift. Episodes where this happened are tagged `D` in the console;
curriculum episodes are tagged `C`.

**Curriculum**: after `CURRICULUM_START_GAMES=150` games, with probability
`CURRICULUM_PROB=0.2`, the next *training* episode starts via
`game.reset(start_length=random.randint(CURRICULUM_MIN_LEN, CURRICULUM_MAX_LEN))` (4-50
cells) - exposes the classifier to "late-game" board states early. Eval episodes always use
plain `reset()`.

**`get_network_action(state, game)`**: forward pass through `SnakeNet`, masked by
`game.safe_moves()` (falls back to unmasked argmax only if no move is safe). Used for
DAgger-driven steps, evaluation, and `--watch`.

**Honest evaluation (`evaluate`)**: every `EVAL_EVERY_N_GAMES=25` completed training games,
plays `EVAL_GAMES=5` games using *only* `get_network_action` (no teacher, plain `reset()`)
and reports avg/max score. This - not the training-episode score, which the teacher mostly
drives - is the real measure of what the network has learned, and is the only thing plotted
as "score" in `learning_curve.png`.

**Checkpointing** (`model/`):

- `checkpoint_last.pth` - saved every `SAVE_EVERY_N_GAMES=10` games. Contains
  `model_state_dict`, `optimizer_state_dict`, `n_games`, `total_steps`, `best_eval_score`.
  Used to *resume* training without losing optimizer momentum, step counters (which drive
  `dagger_prob`), or the best-eval bookkeeping. Gitignored.
- `checkpoint_best.pth` - saved *only* when `evaluate()`'s avg score improves on
  `best_eval_score`. Contains `model_state_dict`, `n_games`, `total_steps`, `eval_score`.
  This is "the smartest model so far" - later regressions can never overwrite it.
  Gitignored. Loaded by `--watch` (default).
- `pretrained.pth` - same shape as `checkpoint_best.pth`, produced by a from-scratch
  training run and committed to the repo as the showcase model. Loaded by
  `--watch --pretrained`.
- `load_checkpoint` catches `RuntimeError` on shape mismatch (architecture changed) and
  starts fresh instead of crashing.

**CPU/perf knobs:** `torch.set_num_threads(TORCH_THREADS=2)` right after imports - without
it torch grabs all cores and starves Pygame's event loop. `train()` sets `game.speed = 0`
(uncapped FPS); `snake_game.py`'s `play_step` only calls `_update_ui()` every
`RENDER_EVERY_N_FRAMES=1000` frames in that mode (events are still pumped every frame, so
the speed-control keys keep working).

**Plotting:** `learning_curve.png` (two panels, matplotlib `Agg` backend, `plt.close()`d
each time to avoid leaking memory over long runs):

1. Honest eval avg/max score vs. games played (the real progress curve).
2. Per-game training loss + its running mean vs. games played.
