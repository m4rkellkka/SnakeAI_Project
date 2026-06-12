from snake_game import SnakeGameAI, Direction, BLOCK_SIZE, GRID_SIZE, HAMILTONIAN_CYCLE

# Hamiltonian cycle for a GRID_SIZE x GRID_SIZE (16x16 = 256 cell) grid.
# Same cycle used by reset(start_length=...) for curriculum starts (see snake_game.py) —
# the body always remains contiguous in cycle indices, even on curriculum starts.
GRID_W = GRID_SIZE
GRID_H = GRID_SIZE
NUM_CELLS = GRID_W * GRID_H

cycle = HAMILTONIAN_CYCLE

# Fast lookup: cell → cycle index
cycle_map = {cell: i for i, cell in enumerate(cycle)}

CLOCK_WISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


def cycle_dist(a_idx, b_idx):
    """Distance from a to b along the cycle (forward direction)."""
    d = (b_idx - a_idx) % NUM_CELLS
    return NUM_CELLS if d == 0 else d


def get_action_for_target(game, target_x, target_y):
    head = game.snake[0]
    hx = head.x // BLOCK_SIZE
    hy = head.y // BLOCK_SIZE

    dx = target_x - hx
    dy = target_y - hy

    if dx > 0:
        desired_dir = Direction.RIGHT
    elif dx < 0:
        desired_dir = Direction.LEFT
    elif dy > 0:
        desired_dir = Direction.DOWN
    else:
        desired_dir = Direction.UP

    idx = CLOCK_WISE.index(game.direction)
    desired_idx = CLOCK_WISE.index(desired_dir)

    if desired_idx == idx:
        return [1, 0, 0]  # Straight
    elif desired_idx == (idx + 1) % 4:
        return [0, 1, 0]  # Right
    else:
        return [0, 0, 1]  # Left


def _relative_move_cell(game, offset):
    """Cell (gx, gy) where the head will move after a relative turn offset
    (0=straight, 1=right, -1=left), without changing game state."""
    idx = CLOCK_WISE.index(game.direction)
    new_dir = CLOCK_WISE[(idx + offset) % 4]

    head = game.snake[0]
    hx, hy = head.x // BLOCK_SIZE, head.y // BLOCK_SIZE

    if new_dir == Direction.RIGHT:
        return hx + 1, hy
    elif new_dir == Direction.LEFT:
        return hx - 1, hy
    elif new_dir == Direction.DOWN:
        return hx, hy + 1
    else:
        return hx, hy - 1


def get_best_move(game):
    head = game.snake[0]
    hx = head.x // BLOCK_SIZE
    hy = head.y // BLOCK_SIZE
    h_idx = cycle_map[(hx, hy)]

    # Body cells minus tail (the tail is freed on non-eating moves)
    body_cells = {(p.x // BLOCK_SIZE, p.y // BLOCK_SIZE) for p in game.snake[:-1]}

    if not game.foods:
        # No food (e.g., win condition), just follow the cycle
        next_cell = cycle[(h_idx + 1) % NUM_CELLS]
        action = get_action_for_target(game, next_cell[0], next_cell[1])
    else:
        apple = game.foods[0]
        ax = apple.x // BLOCK_SIZE
        ay = apple.y // BLOCK_SIZE
        a_idx = cycle_map[(ax, ay)]

        tail = game.snake[-1]
        tx = tail.x // BLOCK_SIZE
        ty = tail.y // BLOCK_SIZE
        t_idx = cycle_map[(tx, ty)]

        # Available space ahead of head (until tail) in cycle
        avail_space = cycle_dist(h_idx, t_idx)

        # Default: just follow the cycle one step
        best_n_idx = (h_idx + 1) % NUM_CELLS
        best_dist = 1

        # Check neighboring cells for corner-cutting shortcuts
        neighbors = [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)]
        for nx, ny in neighbors:
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if (nx, ny) in body_cells:
                    continue  # Direct check — don't jump into body

                n_idx = cycle_map[(nx, ny)]
                dist = cycle_dist(h_idx, n_idx)

                # Check if this is a valid shortcut
                if dist > best_dist:
                    # 1. Don't jump over the apple
                    if dist <= cycle_dist(h_idx, a_idx):
                        # 2. Jump strictly into free space before tail.
                        # Safety margin: don't shortcut if snake is >75% full.
                        if dist < avail_space and len(game.snake) < (NUM_CELLS * 0.75):
                            best_dist = dist
                            best_n_idx = n_idx

        next_cell = cycle[best_n_idx]
        action = get_action_for_target(game, next_cell[0], next_cell[1])

    # Final safety net: cycle arithmetic is an approximation, and after a series
    # of shortcuts, both the shortcut and the default next-cycle-cell may point
    # into the body. If the chosen action is unsafe but another is safe, pick the
    # safe action whose target is closest in cycle distance (minimum deviation
    # from the intended route). If no safe moves exist, collision is inevitable.
    safe = game.safe_moves()  # [straight, right, left]
    action_idx = action.index(1)
    if not safe[action_idx] and any(safe):
        best_idx = None
        best_cycle_dist = None
        for i, offset in enumerate((0, 1, -1)):
            if not safe[i]:
                continue
            cell = _relative_move_cell(game, offset)
            d = cycle_dist(h_idx, cycle_map[cell])
            if best_cycle_dist is None or d < best_cycle_dist:
                best_cycle_dist = d
                best_idx = i

        new_action = [0, 0, 0]
        new_action[best_idx] = 1
        action = new_action

    return action


def play():
    game = SnakeGameAI(w=640, h=640, num_apples=1)
    game.speed = 60  # Visible speed for demonstration

    game.reset(start_length=3)

    record = 0
    games = 0
    last_score = 0

    print("Hamiltonian Cycle Algorithm started.")
    print("Speed control: '+' faster, '-' slower, '0' max speed.")

    while True:
        action = get_best_move(game)
        done, score = game.play_step(action)

        if score > last_score:
            print(f'Current score: {score}', end='\r', flush=True)
            last_score = score

        if done:
            game.reset(start_length=3)
            games += 1
            if score > record:
                record = score
            print(f'\nGame: {games} | Score: {score} | Record: {record}')
            last_score = 0


if __name__ == '__main__':
    play()
