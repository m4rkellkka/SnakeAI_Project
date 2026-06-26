"""Shared state encoding for the Snake AI.

Both training methods consume the same observation of the board:

- Behavioral cloning (`train_ai.py`) feeds it to `SnakeNet` to imitate the teacher.
- Reinforcement learning (`train_rl.py`) feeds it to `DuelingSnakeNet` as the DQN input.

Keeping the encoding in one place guarantees a model trained by one method sees the
exact same observation shape/semantics as the other, and the channel documentation
lives in a single spot. See CLAUDE.md ("State representation") for the full rationale.
"""

import numpy as np
from snake_game import Direction, BLOCK_SIZE, GRID_SIZE

# Channel 5 (visit history) is encoded as a normalized visit COUNT per cell since the
# last apple, not just a binary mask. This lets the network perceive "I've re-entered
# this cell N times" — the signal it needs to notice it's stuck in a loop. Counts are
# capped and scaled to [0, 1] by this value.
VISIT_COUNT_CAP = 4


def encode_state(game):
    """Build the 10-channel (10, GRID_SIZE, GRID_SIZE) float observation.

    Channels 0–5 are egocentric (rotated so the head faces "up"); channels 6–9 encode
    the absolute heading (one-hot, not rotated — the "compass fix").

    - 0: head position
    - 1: body cells
    - 2: food positions
    - 3: danger map (border ring + body cells)
    - 4: board-fullness signal len(snake)/cells (constant across the grid)
    - 5: visit-count map since the last apple (loop-detection signal)
    - 6–9: absolute direction one-hot
    """
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
    dir_idx = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT].index(game.direction)
    state[6 + dir_idx, :, :] = 1.0

    # Egocentric view for channels 0–5: rotate so head faces "up".
    # Channels 6–9 (direction) remain in absolute coordinates.
    if game.direction == Direction.RIGHT:
        state[:6] = np.rot90(state[:6], k=1, axes=(1, 2))
    elif game.direction == Direction.DOWN:
        state[:6] = np.rot90(state[:6], k=2, axes=(1, 2))
    elif game.direction == Direction.LEFT:
        state[:6] = np.rot90(state[:6], k=3, axes=(1, 2))

    # rot90 returns a view with negative strides — copy it for PyTorch compatibility.
    return state.copy()
