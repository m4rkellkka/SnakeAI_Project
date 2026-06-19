import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No GUI — compatible with Pygame on macOS
import matplotlib.pyplot as plt
from snake_game import SnakeGameAI, Direction, BLOCK_SIZE, GRID_SIZE
from teacher import get_best_move

# On multi-core Macs, torch defaults to using all cores (8–10) for convolutions,
# which starves Pygame's event loop during training. The network is small, so
# 2 threads have minimal training impact but huge responsiveness gain.
TORCH_THREADS = 2
torch.set_num_threads(TORCH_THREADS)

# Headless training (the default) has no Pygame event loop to starve, so torch can use
# more cores. Applied in train() once `headless` is known; the import-time default above
# stays at 2 for watch/benchmark, which may run alongside a visible Pygame window.
TORCH_THREADS_HEADLESS = max(2, (os.cpu_count() or 4) - 1)

# --- Settings ---
NUM_APPLES = 1
MAX_MEMORY = 50_000
BATCH_SIZE = 128
LR = 0.0005
TRAIN_EVERY_N_STEPS = 16
GRAD_CLIP_NORM = 10

# Channel 5 (visit history) is encoded as a normalized visit COUNT per cell since the
# last apple, not just a binary mask. This lets the network perceive "I've re-entered
# this cell N times" — the signal it needs to notice it's stuck in a loop. Counts are
# capped and scaled to [0, 1] by this value.
VISIT_COUNT_CAP = 4

# Teacher labels are heavily skewed toward "straight"; turns (where loops form) are
# under-represented, so they get under-trained. Mildly up-weight the two turn classes in
# the CrossEntropy loss. Kept conservative on purpose — full inverse-frequency weighting
# over-emphasizes rare turns and risks regressing the well-tuned honest score.
# Pareto-tuned: softened from [1.0, 2.5, 2.5]. At 2.5 the turn bias eroded the dominant
# "straight" precision the Hamiltonian cycle depends on, capping the honest avg via
# mid-length collisions (~227 honest vs ~234 with no weighting). 1.5 is the midpoint
# between neutral [1,1,1] (avg ~234, but ~20% loop-stuck) and [1,2.5,2.5] (avg ~227,
# ~0% stuck) — aiming to recover the avg without bringing looping back.
CE_CLASS_WEIGHTS = [1.0, 1.5, 1.5]  # [straight, right, left]

# DAgger-lite: with this probability, the network (not the teacher) takes an
# environment step — but the teacher's label is always used for training.
# Goal: expose the network to states its own (imperfect) policy produces, not just
# idealized teacher trajectories — the standard fix for behavioral cloning train/play
# distribution shift. Probability ramps linearly from 0 to DAGGER_PROB_MAX over
# DAGGER_RAMP_STEPS total steps: initially the network is too random to be useful,
# so we ramp in the policy gradually.
# Increased from 0.3: by this point, total_steps >> DAGGER_RAMP_STEPS, network is
# not random anymore (honest eval ~30-60), so more self-driven steps = more useful
# data, not risk.
DAGGER_PROB_MAX = 0.7
DAGGER_RAMP_STEPS = 100_000

# In-training loop handling: when a DAgger-driven episode revisits a (head, direction)
# since the last apple, the network is stuck in a loop. Rather than immediately handing
# control to the teacher (which truncated the loop and starved the buffer of the very
# deep-loop states the network fails on), let the network keep driving for up to this
# many looping steps — those states are still labeled with the teacher's action, so the
# buffer learns "deep in a loop, here is the way out" — then force the teacher to break
# out. The budget caps the inflow so loop states don't flood the buffer.
DAGGER_LOOP_BUDGET = 40

# Curriculum: with this probability (only after CURRICULUM_START_GAMES games),
# a new episode starts with a long snake via game.reset(start_length=...) —
# the network sees "late-game" states early rather than waiting to reach them.
CURRICULUM_PROB = 0.2
CURRICULUM_START_GAMES = 150
CURRICULUM_MIN_LEN = 4
CURRICULUM_MAX_LEN = 50

# Honest evaluation: every EVAL_EVERY_N_GAMES games, we run EVAL_GAMES without
# the teacher (greedy network only) — this is the true learned skill, unlike
# training-game scores (almost always driven by the teacher).
# EVAL_GAMES=5 has huge variance (std ~25–28, mean error ~12); 15 games reduces
# this to ~7. Raised to 30 (mean error ~5): at 15 the avg still swung ±35 between
# evals (e.g. the game-125 dip to 23.4 was almost certainly a noise spike), too noisy
# to trust checkpoint_best selection or to compare runs on a ~7-point Pareto difference.
EVAL_EVERY_N_GAMES = 25
EVAL_GAMES = 30
# Fixed seed for honest eval so checkpoint_best.pth is selected on a paired, repeatable
# game set rather than a lucky/unlucky random draw. evaluate() saves and restores the
# global RNG state around this so training's own randomness is unaffected.
EVAL_SEED = 1234

SAVE_EVERY_N_GAMES = 10
MODEL_FOLDER = './model'
PLOT_EVERY_N_GAMES = 10


def dagger_prob(total_steps):
    return min(DAGGER_PROB_MAX, DAGGER_PROB_MAX * total_steps / DAGGER_RAMP_STEPS)


# --- Plots ---
def plot(eval_games, eval_avg, eval_max, eval_stuck, losses, mean_losses, output_path='learning_curve.png'):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11))

    ax1.set_title('Honest Evaluation (no teacher, greedy)')
    ax1.set_xlabel('Games Played')
    ax1.set_ylabel('Score')
    if eval_games:
        ax1.plot(eval_games, eval_avg, label=f'Average Score ({EVAL_GAMES} games)', color='green')
        ax1.plot(eval_games, eval_max, label='Maximum', color='lightgreen', alpha=0.7)
        ax1.legend(loc='upper left')
    ax1.set_ylim(ymin=0)

    # The real progress metric for the looping problem: fraction of honest-eval games that
    # ended in a loop-timeout. Target is to drive this toward 0 (no need for the unstick crutch).
    ax2.set_title('Honest Stuck-Rate (loop-timeouts, no teacher)')
    ax2.set_xlabel('Games Played')
    ax2.set_ylabel('Stuck Rate')
    if eval_games:
        ax2.plot(eval_games, eval_stuck, label='Stuck rate', color='purple')
        ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)

    ax3.set_title('Training Loss (CrossEntropy)')
    ax3.set_xlabel('Games Played')
    ax3.set_ylabel('Loss')
    ax3.plot(losses, label='Loss per game', alpha=0.3, color='red')
    ax3.plot(mean_losses, label='Average Loss', color='darkred', linewidth=2)
    ax3.set_ylim(ymin=0)
    ax3.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# --- Neural Network ---
class SnakeNet(nn.Module):
    """CNN classifier: predicts teacher action [straight, right, left]
    from a 10-channel grid view (channels 0–5 egocentric, channels 6–9 absolute direction).
    Channel 5 is a normalized visit-count map (loop-detection signal).

    Compass fix: direction is encoded separately (absolute, not rotated) so the network
    can distinguish cycle turns that depend on absolute column parity vs. just local
    patterns (e.g., even vs. odd columns in the serpentine Hamiltonian cycle)."""

    def __init__(self, output_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * GRID_SIZE * GRID_SIZE, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits


# --- Replay Buffer ---
class ReplayBuffer:
    """Ring buffer of (state, action) pairs for supervised learning.
    No next_state/reward/done — not needed for CrossEntropyLoss."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action))
        else:
            self.buffer[self.pos] = (state, action)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions = zip(*batch)
        return states, actions


# --- Trainer ---
class Trainer:
    def __init__(self, model, lr):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        # Up-weight turn classes so the (rare but loop-critical) turn decisions aren't
        # drowned out by the dominant "straight" label — see CE_CLASS_WEIGHTS.
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(CE_CLASS_WEIGHTS, dtype=torch.float))

    def train_step(self, states, actions):
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.float)
        targets = torch.argmax(actions, dim=1)

        logits = self.model(states)
        loss = self.criterion(logits, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        return loss.item()


# --- Agent ---
class Agent:
    def __init__(self):
        self.n_games = 0
        self.total_steps = 0
        self.best_eval_score = 0.0
        self.memory = ReplayBuffer(MAX_MEMORY)
        self.model = SnakeNet(3)
        self.trainer = Trainer(self.model, lr=LR)

        # Training history stored here (not as local variables in train()) so it
        # persists/loads with checkpoints and doesn't reset on training resume.
        self.eval_games_history = []
        self.eval_avg_history = []
        self.eval_max_history = []
        self.eval_stuck_history = []
        self.loss_history = []
        self.mean_loss_history = []
        self.score_history = []

    def get_state(self, game):
        state = np.zeros((10, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Channels 0–4: head, body, food, danger, fullness (before rotation, absolute coords)

        # Channel 0: head
        head = game.snake[0]
        hx, hy = head.x // BLOCK_SIZE, head.y // BLOCK_SIZE
        state[0, hy, hx] = 1.0

        # Channel 1: body
        for pt in game.snake[1:]:
            bx, by = pt.x // BLOCK_SIZE, pt.y // BLOCK_SIZE
            state[1, by, bx] = 1.0

        # Channel 2: food
        for f in game.foods:
            fx, fy = f.x // BLOCK_SIZE, f.y // BLOCK_SIZE
            state[2, fy, fx] = 1.0

        # Channel 3: danger map — border ring (walls) + body cells
        state[3, 0, :] = 1.0
        state[3, -1, :] = 1.0
        state[3, :, 0] = 1.0
        state[3, :, -1] = 1.0
        for pt in game.snake[1:]:
            bx, by = pt.x // BLOCK_SIZE, pt.y // BLOCK_SIZE
            state[3, by, bx] = 1.0

        # Channel 4: board fullness
        state[4, :, :] = len(game.snake) / (GRID_SIZE ** 2)

        # Channel 5: visit COUNT per cell since last food (egocentric — rotated with 0–4),
        # normalized to [0, 1] by VISIT_COUNT_CAP. Unlike a binary mask, this lets the
        # network perceive how many times it has re-entered a cell — the signal it needs
        # to detect it's stuck in a loop. Cleared on each apple.
        for (vx, vy), cnt in game.visited_since_food.items():
            state[5, vy, vx] = min(cnt, VISIT_COUNT_CAP) / VISIT_COUNT_CAP

        # Channels 6–9: absolute direction (one-hot, not rotated!)
        # Compass fix: direction is encoded separately (absolute, not egocentric).
        # This lets the network distinguish cycle turns that depend on absolute
        # position (even/odd columns in the serpentine grid), not just local patterns.
        dir_idx = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT].index(game.direction)
        state[6 + dir_idx, :, :] = 1.0

        # Egocentric view for channels 0–5: rotate so head faces "up".
        # Channel 5 (visited) is positional, so it rotates with the egocentric block.
        # Channels 6–9 (direction) remain in absolute coordinates.
        if game.direction == Direction.RIGHT:
            state[:6] = np.rot90(state[:6], k=1, axes=(1, 2))
        elif game.direction == Direction.DOWN:
            state[:6] = np.rot90(state[:6], k=2, axes=(1, 2))
        elif game.direction == Direction.LEFT:
            state[:6] = np.rot90(state[:6], k=3, axes=(1, 2))

        # rot90 returns a view with negative strides — copy it for PyTorch compatibility.
        return state.copy()

    def get_network_action(self, state, game):
        """Forward pass through network + masking by safe_moves().
        If no safe move exists (snake trapped), use unmasked argmax."""
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state0)

        safe = game.safe_moves()
        if any(safe):
            unsafe_indices = [i for i, ok in enumerate(safe) if not ok]
            masked = logits.clone()
            masked[0, unsafe_indices] = float('-inf')
            move = torch.argmax(masked).item()
        else:
            move = torch.argmax(logits).item()

        action = [0, 0, 0]
        action[move] = 1
        return action

    def save_checkpoint(self, filename, **extra):
        if not os.path.exists(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'n_games': self.n_games,
            'total_steps': self.total_steps,
        }
        checkpoint.update(extra)
        torch.save(checkpoint, os.path.join(MODEL_FOLDER, filename))

    def load_checkpoint(self, filename, load_optimizer=False):
        file_path = os.path.join(MODEL_FOLDER, filename)
        if not os.path.exists(file_path):
            return None

        checkpoint = torch.load(file_path, weights_only=True)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            print(f"Architecture changed, {filename} incompatible — starting fresh")
            return None

        self.n_games = checkpoint.get('n_games', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.best_eval_score = checkpoint.get('best_eval_score', 0.0)
        self.eval_games_history = checkpoint.get('eval_games_history', [])
        self.eval_avg_history = checkpoint.get('eval_avg_history', [])
        self.eval_max_history = checkpoint.get('eval_max_history', [])
        self.eval_stuck_history = checkpoint.get('eval_stuck_history', [])
        self.loss_history = checkpoint.get('loss_history', [])
        self.mean_loss_history = checkpoint.get('mean_loss_history', [])
        self.score_history = checkpoint.get('score_history', [])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint


def play_game(agent, game, unstick=False):
    """Run one episode with the network's policy (greedy + safe_moves).

    If unstick, fall back to the teacher's action whenever the snake repeats a
    (head, direction) it has already visited since the last apple — late-game,
    the network sometimes settles into a closed loop over a subset of the board
    that never passes through the apple's cell, and would otherwise just run out
    the clock (frame_iteration > 100 * len(snake)). The teacher's route follows
    HAMILTONIAN_CYCLE, which almost never matches the network's self-formed loop,
    so it diverges — and heads toward the apple — within one lap.
    """
    done = False
    visited = set()
    score = 0
    while not done:
        key = (game.head, game.direction)
        if unstick and key in visited:
            action = get_best_move(game)
        else:
            state = agent.get_state(game)
            action = agent.get_network_action(state, game)
        if unstick:
            visited.add(key)
        done, score = game.play_step(action)
        if game.frame_iteration == 0:
            visited.clear()
    return score


def game_outcome(game):
    """Classify how a finished episode ended, from the final game state:
    'collision' (hit wall/self), 'win' (snake filled the board), or 'stuck'
    (neither — i.e. the frame-timeout fired because the network looped over a
    subset of the board without reaching the apple). 'stuck' is the failure mode
    the unstick crutch targets, and is reported as the stuck-rate metric."""
    if game.is_collision():
        return 'collision'
    if len(game.snake) >= game.grid_cells:
        return 'win'
    return 'stuck'


def evaluate(agent, game, num_games):
    """Honest evaluation: network only (greedy + safe_moves), no teacher.
    Uses plain reset() — the target conditions. Also counts how many games ended
    'stuck' (loop-timeout) — the metric that tracks progress toward not needing the
    unstick crutch. The score itself stays honest (no unstick during eval)."""
    scores = []
    stuck = 0
    # Reseed per game for a paired, repeatable eval set (food placement is random.*-driven),
    # but save/restore the global RNG so training's own stochasticity is left untouched.
    rng_state = random.getstate()
    try:
        for i in range(num_games):
            random.seed(EVAL_SEED + i)
            game.reset()
            scores.append(play_game(agent, game))
            if game_outcome(game) == 'stuck':
                stuck += 1
    finally:
        random.setstate(rng_state)
    return sum(scores) / len(scores), max(scores), stuck


def train(headless=True, load_checkpoint='checkpoint_last.pth'):
    # Headless training has no Pygame window to keep responsive, so let torch use more cores.
    if headless:
        torch.set_num_threads(TORCH_THREADS_HEADLESS)
    agent = Agent()
    game = SnakeGameAI(w=640, h=640, num_apples=NUM_APPLES, headless=headless)
    game.speed = 0  # Uncapped FPS — rendering is throttled in snake_game.py

    loaded = False
    if load_checkpoint:
        loaded = agent.load_checkpoint(load_checkpoint, load_optimizer=True)
    
    if loaded:
        print(f"Resuming: games={agent.n_games}, steps={agent.total_steps}, "
              f"best honest eval={agent.best_eval_score:.1f}", flush=True)

    # Sweep traceability: captured after CLI overrides have been applied to the module
    # globals (in __main__, before train() is called). Round-trips through checkpoint
    # dicts for benchmark.py to display — not a persistent Agent attribute.
    run_config = {'lr': LR, 'dagger_prob_max': DAGGER_PROB_MAX, 'curriculum_prob': CURRICULUM_PROB}
    print(f"Config: LR={LR}, DAGGER_PROB_MAX={DAGGER_PROB_MAX}, "
          f"CURRICULUM_PROB={CURRICULUM_PROB}, MODEL_FOLDER={MODEL_FOLDER}", flush=True)

    # Sweep runs (MODEL_FOLDER != './model') get their own learning_curve.png so
    # parallel --run-name runs from the same cwd don't clobber the root plot.
    learning_curve_path = (
        os.path.join(MODEL_FOLDER, 'learning_curve.png')
        if MODEL_FOLDER != './model'
        else 'learning_curve.png'
    )

    game_loss = 0.0
    game_batches = 0
    episode_dagger = False
    current_is_curriculum = False
    dagger_visited = set()  # (head Point, Direction) since last food — used to detect loops
    dagger_loop_steps = 0   # consecutive looping steps allowed before forcing the teacher (DAGGER_LOOP_BUDGET)

    if headless:
        print("Training started (headless mode — no game window).", flush=True)
    else:
        print("Training started... '+'/'-' adjust speed, '0' for max FPS.", flush=True)

    while True:
        state_old = agent.get_state(game)
        teacher_action = get_best_move(game)

        if random.random() < dagger_prob(agent.total_steps):
            episode_dagger = True
            dagger_key = (game.head, game.direction)
            if dagger_key in dagger_visited:
                # Loop detected. Let the network keep digging for up to DAGGER_LOOP_BUDGET
                # steps so the buffer collects deep-loop states (label stays the teacher's),
                # then force the teacher to break out. Label is unchanged either way.
                dagger_loop_steps += 1
                if dagger_loop_steps <= DAGGER_LOOP_BUDGET:
                    action = agent.get_network_action(state_old, game)
                else:
                    action = teacher_action
            else:
                # New (head, direction) this interval = genuine progress, not a loop.
                # Re-arm the dig budget so it fires afresh on the NEXT distinct loop —
                # otherwise it's spent once per inter-apple interval and under-collects.
                dagger_visited.add(dagger_key)
                dagger_loop_steps = 0
                action = agent.get_network_action(state_old, game)
        else:
            action = teacher_action

        # Training label is always teacher action for current state,
        # even if we took a network action (DAgger).
        agent.memory.push(state_old, teacher_action)

        done, score = game.play_step(action)

        # frame_iteration resets to 0 inside play_step when food is eaten
        if game.frame_iteration == 0 and not done:
            dagger_visited.clear()
            dagger_loop_steps = 0

        agent.total_steps += 1
        if agent.total_steps % TRAIN_EVERY_N_STEPS == 0 and len(agent.memory) >= BATCH_SIZE:
            states, actions = agent.memory.sample(BATCH_SIZE)
            game_loss += agent.trainer.train_step(states, actions)
            game_batches += 1

        if done:
            agent.n_games += 1
            agent.score_history.append(score)

            avg_loss = game_loss / game_batches if game_batches > 0 else 0.0
            agent.loss_history.append(avg_loss)
            agent.mean_loss_history.append(sum(agent.loss_history) / len(agent.loss_history))
            game_loss = 0.0
            game_batches = 0

            tags = []
            if current_is_curriculum:
                tags.append('C')
            if episode_dagger:
                tags.append('D')
            tag = f" [{','.join(tags)}]" if tags else ''

            print(f'Game: {agent.n_games}{tag} | Score: {score} | Loss: {avg_loss:.4f} | '
                  f'Steps: {agent.total_steps} | DAgger: {dagger_prob(agent.total_steps):.2f}',
                  flush=True)

            if agent.n_games % EVAL_EVERY_N_GAMES == 0:
                avg_eval, max_eval, stuck = evaluate(agent, game, EVAL_GAMES)
                agent.eval_games_history.append(agent.n_games)
                agent.eval_avg_history.append(avg_eval)
                agent.eval_max_history.append(max_eval)
                agent.eval_stuck_history.append(stuck / EVAL_GAMES)
                print(f'  >> Honest eval: avg={avg_eval:.1f}, max={max_eval}, '
                      f'stuck={stuck}/{EVAL_GAMES} ({EVAL_GAMES} games)', flush=True)

                if avg_eval > agent.best_eval_score:
                    agent.best_eval_score = avg_eval
                    agent.save_checkpoint('checkpoint_best.pth', eval_score=avg_eval, run_config=run_config)
                    print(f'  >> New best honest eval: {avg_eval:.1f} -> checkpoint_best.pth',
                          flush=True)

            if agent.n_games % SAVE_EVERY_N_GAMES == 0:
                agent.save_checkpoint(
                    'checkpoint_last.pth',
                    optimizer_state_dict=agent.trainer.optimizer.state_dict(),
                    best_eval_score=agent.best_eval_score,
                    eval_games_history=agent.eval_games_history,
                    eval_avg_history=agent.eval_avg_history,
                    eval_max_history=agent.eval_max_history,
                    eval_stuck_history=agent.eval_stuck_history,
                    loss_history=agent.loss_history,
                    mean_loss_history=agent.mean_loss_history,
                    score_history=agent.score_history[-1000:], # keep only last 1000 to save space
                    run_config=run_config,
                )

            if agent.n_games % PLOT_EVERY_N_GAMES == 0:
                plot(agent.eval_games_history, agent.eval_avg_history, agent.eval_max_history,
                     agent.eval_stuck_history, agent.loss_history, agent.mean_loss_history,
                     output_path=learning_curve_path)

            episode_dagger = False
            dagger_visited.clear()
            dagger_loop_steps = 0
            current_is_curriculum = (
                agent.n_games > CURRICULUM_START_GAMES and random.random() < CURRICULUM_PROB
            )
            start_length = (
                random.randint(CURRICULUM_MIN_LEN, CURRICULUM_MAX_LEN) if current_is_curriculum else None
            )
            game.reset(start_length=start_length)


def watch(num_games=10, pretrained=False, unstick=True, checkpoint_name=None):
    agent = Agent()
    if pretrained:
        checkpoint_name = 'pretrained.pth'
    elif not checkpoint_name:
        checkpoint_name = 'checkpoint_best.pth'
    checkpoint = agent.load_checkpoint(checkpoint_name)

    if checkpoint is None:
        print(f"{checkpoint_name} not found — using untrained network.")
    else:
        eval_score = checkpoint.get('eval_score')
        info = f"games trained: {agent.n_games}"
        if eval_score is not None:
            info += f", honest eval at save: {eval_score:.1f}"
        if agent.mean_loss_history:
            info += f", loss at save: {agent.mean_loss_history[-1]:.4f}"
        print(f"Loaded {checkpoint_name} ({info})")

    game = SnakeGameAI(w=640, h=640, num_apples=NUM_APPLES)
    game.speed = 15

    scores = []
    for i in range(num_games):
        game.reset()
        score = play_game(agent, game, unstick=unstick)
        scores.append(score)
        print(f'Game {i + 1}/{num_games} | Score: {score}')

    print(f'Average: {sum(scores) / len(scores):.2f} | Max: {max(scores)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or watch Snake AI game')
    parser.add_argument('--watch', action='store_true', help='Load checkpoint and watch without training')
    parser.add_argument('--games', type=int, default=10, help='Number of games to watch (default 10)')
    parser.add_argument('--pretrained', action='store_true',
                         help='For --watch: load pretrained model (model/pretrained.pth)')
    parser.add_argument('--checkpoint', type=str, default=None,
                         help='For --watch: load a specific checkpoint file (e.g. checkpoint_last.pth)')
    parser.add_argument('--no-unstick', action='store_true',
                         help='For --watch: disable the teacher-assisted loop-breaking '
                              'fallback (pure network policy)')
    parser.add_argument('--headless', action='store_true',
                         help='No-op (training is headless by default, kept for Tauri/launcher compatibility)')
    parser.add_argument('--render', action='store_true',
                         help='Show game window during training (disabled by default for performance)')
    parser.add_argument('--init-only', action='store_true',
                         help='Initialize a new model with specified parameters and exit without training')
    parser.add_argument('--load-checkpoint', type=str, default='checkpoint_last.pth',
                         help='For training: load a specific checkpoint to resume from. Default: checkpoint_last.pth. Use "none" or "" to train from scratch.')
    parser.add_argument('--lr', type=float, default=None,
                         help='Override learning rate (default: module LR=0.0005)')
    parser.add_argument('--dagger-prob-max', type=float, default=None,
                         help='Override DAgger max probability (default: module DAGGER_PROB_MAX=0.7)')
    parser.add_argument('--curriculum-prob', type=float, default=None,
                         help='Override curriculum episode probability (default: module CURRICULUM_PROB=0.2)')
    parser.add_argument('--run-name', type=str, default=None,
                         help='If set, checkpoints/plots go to model/<run-name>/ instead of model/')
    args = parser.parse_args()

    # Hyperparameter sweeps: rebind module globals from CLI overrides before Agent()/
    # train()/watch() are constructed — they're read as bare names at call time, so a
    # late rebind here is honored.
    if args.lr is not None:
        LR = args.lr
    if args.dagger_prob_max is not None:
        DAGGER_PROB_MAX = args.dagger_prob_max
    if args.curriculum_prob is not None:
        CURRICULUM_PROB = args.curriculum_prob
    if args.run_name is not None:
        MODEL_FOLDER = os.path.join('./model', args.run_name)

    if args.init_only:
        agent = Agent()
        run_config = {'lr': LR, 'dagger_prob_max': DAGGER_PROB_MAX, 'curriculum_prob': CURRICULUM_PROB}
        agent.save_checkpoint('checkpoint_last.pth', run_config=run_config)
        agent.save_checkpoint('checkpoint_best.pth', eval_score=0.0, run_config=run_config)
        print(f"Initialized new model at {MODEL_FOLDER} with config: {run_config}", flush=True)
        sys.exit(0)

    if args.watch:
        watch(num_games=args.games, pretrained=args.pretrained, unstick=not args.no_unstick, checkpoint_name=args.checkpoint)
    else:
        # Handle the special case where user wants to force train from scratch
        load_cp = None if args.load_checkpoint.lower() == 'none' or args.load_checkpoint == '' else args.load_checkpoint
        train(headless=not args.render, load_checkpoint=load_cp)
