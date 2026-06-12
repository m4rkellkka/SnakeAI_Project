import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
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

# --- Settings ---
NUM_APPLES = 1
MAX_MEMORY = 50_000
BATCH_SIZE = 128
LR = 0.0005
TRAIN_EVERY_N_STEPS = 16
GRAD_CLIP_NORM = 10

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
# this to ~7, so "new record" in checkpoint_best.pth isn't just luck.
EVAL_EVERY_N_GAMES = 25
EVAL_GAMES = 15

SAVE_EVERY_N_GAMES = 10
MODEL_FOLDER = './model'
PLOT_EVERY_N_GAMES = 10


def dagger_prob(total_steps):
    return min(DAGGER_PROB_MAX, DAGGER_PROB_MAX * total_steps / DAGGER_RAMP_STEPS)


# --- Plots ---
def plot(eval_games, eval_avg, eval_max, losses, mean_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.set_title('Honest Evaluation (no teacher, greedy)')
    ax1.set_xlabel('Games Played')
    ax1.set_ylabel('Score')
    if eval_games:
        ax1.plot(eval_games, eval_avg, label=f'Average Score ({EVAL_GAMES} games)', color='green')
        ax1.plot(eval_games, eval_max, label='Maximum', color='lightgreen', alpha=0.7)
        ax1.legend(loc='upper left')
    ax1.set_ylim(ymin=0)

    ax2.set_title('Training Loss (CrossEntropy)')
    ax2.set_xlabel('Games Played')
    ax2.set_ylabel('Loss')
    ax2.plot(losses, label='Loss per game', alpha=0.3, color='red')
    ax2.plot(mean_losses, label='Average Loss', color='darkred', linewidth=2)
    ax2.set_ylim(ymin=0)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig('learning_curve.png')
    plt.close(fig)


# --- Neural Network ---
class SnakeNet(nn.Module):
    """CNN classifier: predicts teacher action [straight, right, left]
    from a 9-channel grid view (channels 0–4 egocentric, channels 5–8 absolute direction).

    Compass fix: direction is encoded separately (absolute, not rotated) so the network
    can distinguish cycle turns that depend on absolute column parity vs. just local
    patterns (e.g., even vs. odd columns in the serpentine Hamiltonian cycle)."""

    def __init__(self, output_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 16, kernel_size=3, stride=1, padding=1)
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
        self.criterion = nn.CrossEntropyLoss()

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
        self.loss_history = []
        self.mean_loss_history = []

    def get_state(self, game):
        state = np.zeros((9, GRID_SIZE, GRID_SIZE), dtype=np.float32)

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

        # Channels 5–8: absolute direction (one-hot, not rotated!)
        # Compass fix: direction is encoded separately (absolute, not egocentric).
        # This lets the network distinguish cycle turns that depend on absolute
        # position (even/odd columns in the serpentine grid), not just local patterns.
        dir_idx = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT].index(game.direction)
        state[5 + dir_idx, :, :] = 1.0

        # Egocentric view ONLY for channels 0–4: rotate so head faces "up".
        # Channels 5–8 (direction) remain in absolute coordinates.
        if game.direction == Direction.RIGHT:
            state[:5] = np.rot90(state[:5], k=1, axes=(1, 2))
        elif game.direction == Direction.DOWN:
            state[:5] = np.rot90(state[:5], k=2, axes=(1, 2))
        elif game.direction == Direction.LEFT:
            state[:5] = np.rot90(state[:5], k=3, axes=(1, 2))

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
        self.loss_history = checkpoint.get('loss_history', [])
        self.mean_loss_history = checkpoint.get('mean_loss_history', [])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint


def evaluate(agent, game, num_games):
    """Honest evaluation: network only (greedy + safe_moves), no teacher.
    Uses plain reset() — the target conditions."""
    scores = []
    for _ in range(num_games):
        game.reset()
        done = False
        while not done:
            state = agent.get_state(game)
            action = agent.get_network_action(state, game)
            done, score = game.play_step(action)
        scores.append(score)
    return sum(scores) / len(scores), max(scores)


def train():
    agent = Agent()
    game = SnakeGameAI(w=640, h=640, num_apples=NUM_APPLES)
    game.speed = 0  # Uncapped FPS — rendering is throttled in snake_game.py

    loaded = agent.load_checkpoint('checkpoint_last.pth', load_optimizer=True)
    if loaded:
        print(f"Resuming: games={agent.n_games}, steps={agent.total_steps}, "
              f"best honest eval={agent.best_eval_score:.1f}")

    game_loss = 0.0
    game_batches = 0
    episode_dagger = False
    current_is_curriculum = False

    print("Training started... '+'/'-' adjust speed, '0' for max FPS.")

    while True:
        state_old = agent.get_state(game)
        teacher_action = get_best_move(game)

        if random.random() < dagger_prob(agent.total_steps):
            action = agent.get_network_action(state_old, game)
            episode_dagger = True
        else:
            action = teacher_action

        # Training label is always teacher action for current state,
        # even if we took a network action (DAgger).
        agent.memory.push(state_old, teacher_action)

        done, score = game.play_step(action)

        agent.total_steps += 1
        if agent.total_steps % TRAIN_EVERY_N_STEPS == 0 and len(agent.memory) >= BATCH_SIZE:
            states, actions = agent.memory.sample(BATCH_SIZE)
            game_loss += agent.trainer.train_step(states, actions)
            game_batches += 1

        if done:
            agent.n_games += 1

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
                  f'Steps: {agent.total_steps} | DAgger: {dagger_prob(agent.total_steps):.2f}')

            if agent.n_games % EVAL_EVERY_N_GAMES == 0:
                avg_eval, max_eval = evaluate(agent, game, EVAL_GAMES)
                agent.eval_games_history.append(agent.n_games)
                agent.eval_avg_history.append(avg_eval)
                agent.eval_max_history.append(max_eval)
                print(f'  >> Honest eval: avg={avg_eval:.1f}, max={max_eval} '
                      f'({EVAL_GAMES} games)')

                if avg_eval > agent.best_eval_score:
                    agent.best_eval_score = avg_eval
                    agent.save_checkpoint('checkpoint_best.pth', eval_score=avg_eval)
                    print(f'  >> New best honest eval: {avg_eval:.1f} -> checkpoint_best.pth')

            if agent.n_games % SAVE_EVERY_N_GAMES == 0:
                agent.save_checkpoint(
                    'checkpoint_last.pth',
                    optimizer_state_dict=agent.trainer.optimizer.state_dict(),
                    best_eval_score=agent.best_eval_score,
                    eval_games_history=agent.eval_games_history,
                    eval_avg_history=agent.eval_avg_history,
                    eval_max_history=agent.eval_max_history,
                    loss_history=agent.loss_history,
                    mean_loss_history=agent.mean_loss_history,
                )

            if agent.n_games % PLOT_EVERY_N_GAMES == 0:
                plot(agent.eval_games_history, agent.eval_avg_history, agent.eval_max_history,
                     agent.loss_history, agent.mean_loss_history)

            episode_dagger = False
            current_is_curriculum = (
                agent.n_games > CURRICULUM_START_GAMES and random.random() < CURRICULUM_PROB
            )
            start_length = (
                random.randint(CURRICULUM_MIN_LEN, CURRICULUM_MAX_LEN) if current_is_curriculum else None
            )
            game.reset(start_length=start_length)


def watch(num_games=10, pretrained=False):
    agent = Agent()
    checkpoint_name = 'pretrained.pth' if pretrained else 'checkpoint_best.pth'
    checkpoint = agent.load_checkpoint(checkpoint_name)

    if checkpoint is None:
        print(f"{checkpoint_name} not found — using untrained network.")
    else:
        eval_score = checkpoint.get('eval_score')
        info = f"games trained: {agent.n_games}"
        if eval_score is not None:
            info += f", honest eval at save: {eval_score:.1f}"
        print(f"Loaded {checkpoint_name} ({info})")

    game = SnakeGameAI(w=640, h=640, num_apples=NUM_APPLES)
    game.speed = 15

    scores = []
    for i in range(num_games):
        game.reset()
        done = False
        while not done:
            state = agent.get_state(game)
            action = agent.get_network_action(state, game)
            done, score = game.play_step(action)

        scores.append(score)
        print(f'Game {i + 1}/{num_games} | Score: {score}')

    print(f'Average: {sum(scores) / len(scores):.2f} | Max: {max(scores)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or watch Snake AI game')
    parser.add_argument('--watch', action='store_true', help='Load checkpoint and watch without training')
    parser.add_argument('--games', type=int, default=10, help='Number of games to watch (default 10)')
    parser.add_argument('--pretrained', action='store_true',
                         help='For --watch: load pretrained model (model/pretrained.pth)')
    args = parser.parse_args()

    if args.watch:
        watch(num_games=args.games, pretrained=args.pretrained)
    else:
        train()
