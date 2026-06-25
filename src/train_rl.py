"""Reinforcement-learning trainer for Snake — the *self-learning* method.

Unlike train_ai.py (behavioral cloning: imitate the Hamiltonian teacher with
CrossEntropyLoss), this trainer learns from a **reward signal** with no per-step
label: Double DQN with a Dueling head. The snake discovers how to play by trial and
error, earning reward for eating and being punished for dying or looping.

The teacher is allowed only as an optional *warm-start* (pre-fill the replay buffer
and bias early exploration) — it is NEVER part of the reward and the network never
inherits teacher/BC weights. RL and BC are fully separate models: this trainer writes
to model/<run-name>/ (default model/rl/) and only ever loads its own checkpoints.

Run:
    python3 src/train_rl.py --algo dqn --run-name rl_test --warmstart-episodes 50
    python3 src/train_rl.py --watch --run-name rl_test          # watch the RL model
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from snake_game import SnakeGameAI, BLOCK_SIZE, GRID_SIZE
from teacher import get_best_move
from state_encoding import encode_state
# Honest evaluation is identical for both methods (greedy network, no teacher, no
# unstick) — reuse the BC implementation rather than duplicate it.
from train_ai import evaluate, play_game, game_outcome, EVAL_GAMES, EVAL_EVERY_N_GAMES

torch.set_num_threads(2)
TORCH_THREADS_HEADLESS = max(2, (os.cpu_count() or 4) - 1)

MODEL_ARCH = 'snakenet_dueling'

# --- DQN hyperparameters ---
MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 0.0005
GAMMA = 0.95            # discount factor
TRAIN_EVERY_N_STEPS = 4
TARGET_SYNC_STEPS = 2000  # hard-sync target net every N optimizer-eligible steps
GRAD_CLIP_NORM = 10

# Epsilon-greedy exploration: linearly decay from EPS_START to EPS_END over EPS_DECAY_STEPS.
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 150_000

# Teacher-guided exploration: of the *exploratory* steps, this fraction takes the
# teacher's move (vs. a uniformly random safe move). Decays to 0 so the agent weans
# off the teacher and explores on its own. This is the "warm-start" bias in the live
# loop; set --teacher-eps 0 for purely random exploration.
TEACHER_EPS_START = 1.0
TEACHER_EPS_DECAY_STEPS = 80_000

# --- Reward shaping ---
REWARD_FOOD = 10.0       # ate an apple (score increased)
REWARD_DEATH = -10.0     # wall/self collision OR loop-timeout ('stuck') — looping is punished as hard as dying
REWARD_WIN = 10.0        # filled the board
STEP_PENALTY = -0.01     # small per-step cost — discourages endless circling
# Potential-based shaping toward the nearest food (Ng et al. 1999): F = γ·Φ(s') − Φ(s)
# with Φ(s) = -dist_to_food. Provably policy-invariant — it speeds learning without
# creating new optimal policies, so it can't be "hacked" by oscillating toward/away.
SHAPING_WEIGHT = 0.1

SAVE_EVERY_N_GAMES = 5
EVAL_EVERY = EVAL_EVERY_N_GAMES   # reuse BC cadence (25 games)
PLOT_EVERY_N_GAMES = 10
MODEL_FOLDER = './model/rl'


# --- Reward helpers ---
def nearest_food_dist(game):
    """Manhattan distance (in grid cells) from the head to the nearest food.
    Returns 0 if there is no food on the board."""
    if not game.foods:
        return 0
    hx, hy = game.head.x, game.head.y
    return min(abs(f.x - hx) + abs(f.y - hy) for f in game.foods) // BLOCK_SIZE


def compute_reward(ate, done, outcome, prev_dist, new_dist):
    """Reward for a single transition.

    `outcome` is game_outcome(game) when done (else None). Potential-based shaping is
    skipped on terminal steps (Φ(terminal)=0 convention) and on the eat step (the apple
    relocated, so the distance delta is meaningless — the +REWARD_FOOD dominates anyway).
    """
    r = STEP_PENALTY
    if ate:
        r += REWARD_FOOD
    if done:
        if outcome == 'win':
            r += REWARD_WIN
        else:  # 'collision' or 'stuck'
            r += REWARD_DEATH
        return r
    if not ate:
        # F = γ·Φ(s') − Φ(s), Φ = -dist  =>  γ·(-new) - (-prev) = prev - γ·new
        r += SHAPING_WEIGHT * (prev_dist - GAMMA * new_dist)
    return r


# --- Network: Dueling CNN ---
class DuelingSnakeNet(nn.Module):
    """Same conv trunk as SnakeNet (10→16→32→32), but a dueling head: a state-value
    stream V(s) and an advantage stream A(s,a), combined as Q = V + (A - mean(A)).
    In most Snake states the action barely matters (open board), so learning "is this
    state good" separately from "which of 3 turns" converges faster."""

    def __init__(self, output_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * GRID_SIZE * GRID_SIZE, 128)
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))  # Q-values


# --- Replay buffer (5-tuples) ---
class RLReplayBuffer:
    """Ring buffer of (state, action_idx, reward, next_state, done, next_safe) for
    off-policy DQN. `next_safe` is the safe-move mask of next_state, stored so the
    Double-DQN target can mask its action selection the same way the behavior policy does."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action_idx, reward, next_state, done, next_safe):
        item = (state, action_idx, reward, next_state, done, next_safe)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_safes = zip(*batch)
        return states, actions, rewards, next_states, dones, next_safes


# --- DQN agent ---
class DQNAgent:
    """Holds the online + target Dueling nets and the Double-DQN update. Exposes
    get_state/get_network_action with the same signatures as the BC Agent so the shared
    evaluate()/play_game() helpers from train_ai.py work unchanged."""

    def __init__(self, lr=LR):
        self.n_games = 0
        self.total_steps = 0
        self.best_eval_score = 0.0
        self.memory = RLReplayBuffer(MAX_MEMORY)

        self.model = DuelingSnakeNet(3)        # online net
        self.target = DuelingSnakeNet(3)       # target net
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # History (persisted in checkpoints). Eval keys mirror BC so a future
        # export_history.py / UI can render RL runs with the same plumbing.
        self.eval_games_history = []
        self.eval_avg_history = []
        self.eval_max_history = []
        self.eval_stuck_history = []
        self.reward_history = []      # total reward per episode
        self.score_history = []       # final score per training game
        self.loss_history = []

    def get_state(self, game):
        return encode_state(game)

    def get_network_action(self, state, game):
        """Greedy Q action masked by safe_moves() — same masking policy as BC.
        Returns a one-hot [straight, right, left] action."""
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q = self.model(state0)

        safe = game.safe_moves()
        if any(safe):
            unsafe = [i for i, ok in enumerate(safe) if not ok]
            masked = q.clone()
            masked[0, unsafe] = float('-inf')
            move = torch.argmax(masked).item()
        else:
            move = torch.argmax(q).item()

        action = [0, 0, 0]
        action[move] = 1
        return action

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones, next_safes = self.memory.sample(BATCH_SIZE)

        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1)
        next_safe = torch.tensor(np.array(next_safes), dtype=torch.bool)  # (B, 3)

        # Q(s,a) for the actions actually taken
        q = self.model(states).gather(1, actions)

        # Double DQN target: action chosen by ONLINE net, evaluated by TARGET net.
        # The action selection is masked by next_state's safe moves — the behavior
        # policy never takes unsafe moves, so their Q-values are never grounded by real
        # samples; an unmasked argmax could latch onto an inflated unsafe-action Q and
        # blow up the target. Rows with no safe move keep all actions (avoids an
        # all -inf argmax); their target is zeroed by (1 - done) anyway.
        with torch.no_grad():
            next_q_online = self.model(next_states)
            has_safe = next_safe.any(dim=1, keepdim=True)
            next_q_online = next_q_online.masked_fill(~next_safe & has_safe, float('-inf'))
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            target_q = rewards + GAMMA * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q, target_q)  # Huber

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def save_checkpoint(self, filename, run_config, **extra):
        if not os.path.exists(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)
        checkpoint = {
            'algo': 'dqn',
            'model_arch': MODEL_ARCH,
            'model_state_dict': self.model.state_dict(),
            'n_games': self.n_games,
            'total_steps': self.total_steps,
            'run_config': run_config,
        }
        checkpoint.update(extra)
        torch.save(checkpoint, os.path.join(MODEL_FOLDER, filename))

    def load_checkpoint(self, filename, load_optimizer=False):
        file_path = os.path.join(MODEL_FOLDER, filename)
        if not os.path.exists(file_path):
            return None

        checkpoint = torch.load(file_path, weights_only=False)
        # Hard isolation: refuse anything that isn't a matching RL checkpoint, so a BC
        # model can never be loaded here (and vice-versa). Start fresh instead of crashing.
        if checkpoint.get('algo') != 'dqn' or checkpoint.get('model_arch') != MODEL_ARCH:
            print(f"{filename} is not a {MODEL_ARCH} DQN checkpoint — starting fresh", flush=True)
            return None
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            print(f"Architecture changed, {filename} incompatible — starting fresh", flush=True)
            return None
        self.target.load_state_dict(self.model.state_dict())

        self.n_games = checkpoint.get('n_games', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.best_eval_score = checkpoint.get('best_eval_score', 0.0)
        self.eval_games_history = checkpoint.get('eval_games_history', [])
        self.eval_avg_history = checkpoint.get('eval_avg_history', [])
        self.eval_max_history = checkpoint.get('eval_max_history', [])
        self.eval_stuck_history = checkpoint.get('eval_stuck_history', [])
        self.reward_history = checkpoint.get('reward_history', [])
        self.score_history = checkpoint.get('score_history', [])
        self.loss_history = checkpoint.get('loss_history', [])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


# --- Exploration schedules ---
def epsilon(total_steps):
    frac = min(1.0, total_steps / EPS_DECAY_STEPS)
    return EPS_START + (EPS_END - EPS_START) * frac


def teacher_eps(total_steps):
    """Fraction of exploratory moves that follow the teacher (decays to 0)."""
    frac = min(1.0, total_steps / TEACHER_EPS_DECAY_STEPS)
    return TEACHER_EPS_START * (1.0 - frac)


def random_safe_action_idx(game):
    safe = game.safe_moves()
    choices = [i for i, ok in enumerate(safe) if ok]
    return random.choice(choices) if choices else random.randint(0, 2)


def select_action_idx(agent, game, state, use_teacher_bias):
    """ε-greedy action index. With prob ε explore (teacher move with prob teacher_eps,
    else a random safe move); otherwise the network's greedy masked action."""
    if random.random() < epsilon(agent.total_steps):
        if use_teacher_bias and random.random() < teacher_eps(agent.total_steps):
            return get_best_move(game).index(1)
        return random_safe_action_idx(game)
    return agent.get_network_action(state, game).index(1)


def idx_to_action(idx):
    a = [0, 0, 0]
    a[idx] = 1
    return a


# --- Warm-start: pre-fill the buffer with teacher trajectories ---
def warmstart(agent, game, num_episodes):
    """Run num_episodes fully teacher-driven games, computing rewards and pushing the
    transitions into the replay buffer. The teacher only generates *data* here — the
    learning target is still the reward/Q-update, never the teacher's action label."""
    pushed = 0
    for ep in range(num_episodes):
        game.reset()
        state = agent.get_state(game)
        prev_dist = nearest_food_dist(game)
        prev_score = game.score
        done = False
        while not done:
            action_idx = get_best_move(game).index(1)
            done, score = game.play_step(idx_to_action(action_idx))
            ate = score > prev_score
            outcome = game_outcome(game) if done else None
            new_dist = nearest_food_dist(game)
            reward = compute_reward(ate, done, outcome, prev_dist, new_dist)
            if done:
                next_state = np.zeros_like(state)
            else:
                next_state = agent.get_state(game)
            agent.memory.push(state, action_idx, reward, next_state, float(done))
            pushed += 1
            state = next_state
            prev_dist = new_dist
            prev_score = score
        print(f"  warm-start episode {ep + 1}/{num_episodes} | score={score} | buffer={len(agent.memory)}",
              flush=True)
    print(f"Warm-start done: {pushed} transitions, buffer={len(agent.memory)}", flush=True)


# --- Plot ---
def plot(agent, output_path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11))

    ax1.set_title('Honest Evaluation (no teacher, greedy)')
    ax1.set_xlabel('Games Played')
    ax1.set_ylabel('Score')
    if agent.eval_games_history:
        ax1.plot(agent.eval_games_history, agent.eval_avg_history, label='Average', color='green')
        ax1.plot(agent.eval_games_history, agent.eval_max_history, label='Maximum',
                 color='lightgreen', alpha=0.7)
        ax1.legend(loc='upper left')
    ax1.set_ylim(ymin=0)

    ax2.set_title('Total Reward per Episode (training)')
    ax2.set_xlabel('Games Played')
    ax2.set_ylabel('Reward')
    ax2.plot(agent.reward_history, color='blue', alpha=0.5)

    ax3.set_title('Training Loss (Huber / TD error)')
    ax3.set_xlabel('Games Played')
    ax3.set_ylabel('Loss')
    ax3.plot(agent.loss_history, color='red', alpha=0.6)
    ax3.set_ylim(ymin=0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# --- Training loop ---
def train(headless=True, load_cp='checkpoint_last.pth', warmstart_episodes=0,
          use_teacher_bias=True):
    if headless:
        torch.set_num_threads(TORCH_THREADS_HEADLESS)

    agent = DQNAgent(lr=LR)
    game = SnakeGameAI(w=640, h=640, num_apples=1, headless=headless)
    game.speed = 0

    if load_cp and agent.load_checkpoint(load_cp, load_optimizer=True):
        print(f"Resuming: games={agent.n_games}, steps={agent.total_steps}, "
              f"best honest eval={agent.best_eval_score:.1f}", flush=True)

    run_config = {
        'algo': 'dqn', 'model_arch': MODEL_ARCH, 'lr': LR, 'gamma': GAMMA,
        'eps_decay_steps': EPS_DECAY_STEPS, 'warmstart_episodes': warmstart_episodes,
        'teacher_bias': use_teacher_bias, 'reward_food': REWARD_FOOD,
        'reward_death': REWARD_DEATH, 'shaping_weight': SHAPING_WEIGHT,
    }
    print(f"Config: algo=dqn, LR={LR}, GAMMA={GAMMA}, warmstart={warmstart_episodes}, "
          f"teacher_bias={use_teacher_bias}, MODEL_FOLDER={MODEL_FOLDER}", flush=True)

    learning_curve_path = os.path.join(MODEL_FOLDER, 'learning_curve.png')

    if warmstart_episodes > 0 and agent.n_games == 0:
        print(f"Warm-starting buffer with {warmstart_episodes} teacher episodes...", flush=True)
        warmstart(agent, game, warmstart_episodes)

    print("RL training started (Double DQN)." + ("" if headless else
          " '+'/'-' adjust speed, '0' for max FPS."), flush=True)

    game.reset()
    state = agent.get_state(game)
    prev_dist = nearest_food_dist(game)
    prev_score = game.score
    ep_reward = 0.0
    ep_loss = 0.0
    ep_batches = 0

    while True:
        action_idx = select_action_idx(agent, game, state, use_teacher_bias)
        done, score = game.play_step(idx_to_action(action_idx))
        ate = score > prev_score
        outcome = game_outcome(game) if done else None
        new_dist = nearest_food_dist(game)
        reward = compute_reward(ate, done, outcome, prev_dist, new_dist)
        if done:
            next_state = np.zeros_like(state)
            next_safe = (True, True, True)  # unused: target is zeroed by (1 - done)
        else:
            next_state = agent.get_state(game)
            next_safe = tuple(game.safe_moves())  # safe mask for next_state

        agent.memory.push(state, action_idx, reward, next_state, float(done), next_safe)
        agent.total_steps += 1
        ep_reward += reward

        if agent.total_steps % TRAIN_EVERY_N_STEPS == 0:
            loss = agent.train_step()
            if loss is not None:
                ep_loss += loss
                ep_batches += 1
        if agent.total_steps % TARGET_SYNC_STEPS == 0:
            agent.sync_target()

        state = next_state
        prev_dist = new_dist
        prev_score = score

        if done:
            agent.n_games += 1
            avg_loss = ep_loss / ep_batches if ep_batches else 0.0
            # Total episode reward — same metric the log line prints, so the UI chart
            # is continuous across disk-loaded history and live-streamed points.
            agent.reward_history.append(round(ep_reward, 2))
            agent.score_history.append(score)
            agent.loss_history.append(avg_loss)

            print(f'Game: {agent.n_games} | Score: {score} | Loss: {avg_loss:.4f} | '
                  f'Steps: {agent.total_steps} | Reward: {ep_reward:.1f} | '
                  f'Eps: {epsilon(agent.total_steps):.2f}', flush=True)

            if agent.n_games % EVAL_EVERY == 0:
                avg_eval, max_eval, stuck = evaluate(agent, game, EVAL_GAMES)
                agent.eval_games_history.append(agent.n_games)
                agent.eval_avg_history.append(avg_eval)
                agent.eval_max_history.append(max_eval)
                agent.eval_stuck_history.append(stuck / EVAL_GAMES)
                print(f'  >> Honest eval: avg={avg_eval:.1f}, max={max_eval}, '
                      f'stuck={stuck}/{EVAL_GAMES}', flush=True)
                if avg_eval > agent.best_eval_score:
                    agent.best_eval_score = avg_eval
                    agent.save_checkpoint('checkpoint_best.pth', run_config,
                                          eval_score=avg_eval, best_eval_score=avg_eval)
                    print(f'  >> New best honest eval: {avg_eval:.1f} -> checkpoint_best.pth',
                          flush=True)

            if agent.n_games % SAVE_EVERY_N_GAMES == 0:
                agent.save_checkpoint(
                    'checkpoint_last.pth', run_config,
                    optimizer_state_dict=agent.optimizer.state_dict(),
                    best_eval_score=agent.best_eval_score,
                    eval_games_history=agent.eval_games_history,
                    eval_avg_history=agent.eval_avg_history,
                    eval_max_history=agent.eval_max_history,
                    eval_stuck_history=agent.eval_stuck_history,
                    reward_history=agent.reward_history[-2000:],
                    score_history=agent.score_history[-2000:],
                    loss_history=agent.loss_history[-2000:],
                )

            if agent.n_games % PLOT_EVERY_N_GAMES == 0:
                plot(agent, learning_curve_path)

            game.reset()
            state = agent.get_state(game)
            prev_dist = nearest_food_dist(game)
            prev_score = game.score
            ep_reward = 0.0
            ep_loss = 0.0
            ep_batches = 0


def watch(num_games=10, unstick=False, load_cp='checkpoint_best.pth'):
    agent = DQNAgent()
    if agent.load_checkpoint(load_cp) is None:
        print(f"{load_cp} not found in {MODEL_FOLDER} — using untrained network.")
    else:
        print(f"Loaded {load_cp} (games trained: {agent.n_games}, "
              f"best honest eval: {agent.best_eval_score:.1f})")

    game = SnakeGameAI(w=640, h=640, num_apples=1)
    game.speed = 15
    scores = []
    for i in range(num_games):
        game.reset()
        score = play_game(agent, game, unstick=unstick)
        scores.append(score)
        print(f'Game {i + 1}/{num_games} | Score: {score}')
    print(f'Average: {sum(scores) / len(scores):.2f} | Max: {max(scores)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or watch the RL (DQN) Snake AI')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn'],
                        help='RL algorithm (only dqn for now; reserved for future PPO/etc.)')
    parser.add_argument('--watch', action='store_true', help='Load checkpoint and watch without training')
    parser.add_argument('--games', type=int, default=10, help='Number of games to watch (default 10)')
    parser.add_argument('--unstick', action='store_true',
                        help='For --watch: enable the teacher loop-breaker (off by default for RL)')
    parser.add_argument('--render', action='store_true', help='Show the game window during training')
    parser.add_argument('--headless', action='store_true', help='No-op (training is headless by default)')
    parser.add_argument('--init-only', action='store_true',
                        help='Initialize a new RL run (empty checkpoints) and exit without training')
    parser.add_argument('--run-name', type=str, default='rl',
                        help='Checkpoints/plots go to model/<run-name>/ (default: rl). '
                             'RL never writes into the model/ root used by behavioral cloning.')
    parser.add_argument('--load-checkpoint', type=str, default='checkpoint_last.pth',
                        help='Checkpoint to resume from. Use "none" or "" to train from scratch.')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--gamma', type=float, default=None, help='Override discount factor')
    parser.add_argument('--warmstart-episodes', type=int, default=0,
                        help='Pre-fill the replay buffer with N fully teacher-driven episodes')
    parser.add_argument('--no-teacher-bias', action='store_true',
                        help='Disable teacher-guided exploration (pure random exploration)')
    args = parser.parse_args()

    # Rebind module globals from CLI overrides before constructing the agent/loop.
    if args.lr is not None:
        LR = args.lr
    if args.gamma is not None:
        GAMMA = args.gamma
    MODEL_FOLDER = os.path.join('./model', args.run_name)

    if args.init_only:
        agent = DQNAgent(lr=LR)
        run_config = {'algo': 'dqn', 'model_arch': MODEL_ARCH, 'lr': LR, 'gamma': GAMMA,
                      'warmstart_episodes': args.warmstart_episodes,
                      'teacher_bias': not args.no_teacher_bias}
        agent.save_checkpoint('checkpoint_last.pth', run_config)
        agent.save_checkpoint('checkpoint_best.pth', run_config, eval_score=0.0, best_eval_score=0.0)
        print(f"Initialized new RL run at {MODEL_FOLDER} with config: {run_config}", flush=True)
        sys.exit(0)

    if args.watch:
        watch(num_games=args.games, unstick=args.unstick)
    else:
        load_cp = None if args.load_checkpoint.lower() in ('none', '') else args.load_checkpoint
        train(headless=not args.render, load_cp=load_cp,
              warmstart_episodes=args.warmstart_episodes,
              use_teacher_bias=not args.no_teacher_bias)
