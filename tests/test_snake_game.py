"""Tests for the Snake game engine (src/snake_game.py).

Run headless (no Pygame window) with plain stdlib unittest — no extra deps:

    python3 tests/test_snake_game.py
    python3 -m unittest discover -s tests

These guard the non-obvious invariants documented in CLAUDE.md, most importantly
the safe_moves()/is_collision() consistency that was the root cause of phantom
collisions when the two drifted out of sync.
"""

import os
import sys
import random
import unittest

# Tests run with cwd-independent imports of the src/ modules, mirroring how the
# scripts add their own dir to sys.path[0] when run as `python3 src/<file>.py`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snake_game import (  # noqa: E402
    SnakeGameAI,
    Direction,
    Point,
    BLOCK_SIZE,
    GRID_SIZE,
    HAMILTONIAN_CYCLE,
)


def new_game(num_apples=3):
    """A headless game (no Pygame display) for fast, windowless testing."""
    return SnakeGameAI(headless=True, num_apples=num_apples)


class TestHamiltonianCycle(unittest.TestCase):
    """The cycle is the backbone of curriculum starts and the teacher; if it is
    not a true Hamiltonian cycle the avail_space/shortcut math silently breaks."""

    def test_visits_every_cell_exactly_once(self):
        self.assertEqual(len(HAMILTONIAN_CYCLE), GRID_SIZE * GRID_SIZE)
        self.assertEqual(len(set(HAMILTONIAN_CYCLE)), GRID_SIZE * GRID_SIZE)

    def test_all_cells_in_bounds(self):
        for gx, gy in HAMILTONIAN_CYCLE:
            self.assertTrue(0 <= gx < GRID_SIZE)
            self.assertTrue(0 <= gy < GRID_SIZE)

    def test_consecutive_cells_are_adjacent(self):
        # Including the wrap-around from the last cell back to the first — that
        # closure is what makes it a *cycle*, not just a path.
        n = len(HAMILTONIAN_CYCLE)
        for i in range(n):
            (ax, ay) = HAMILTONIAN_CYCLE[i]
            (bx, by) = HAMILTONIAN_CYCLE[(i + 1) % n]
            manhattan = abs(ax - bx) + abs(ay - by)
            self.assertEqual(
                manhattan, 1, f"cells {i} and {(i + 1) % n} are not adjacent"
            )


class TestResetNormal(unittest.TestCase):
    def setUp(self):
        self.g = new_game(num_apples=3)

    def test_initial_state(self):
        self.assertEqual(len(self.g.snake), 3)
        self.assertEqual(self.g.direction, Direction.RIGHT)
        self.assertEqual(self.g.score, 0)
        self.assertEqual(self.g.frame_iteration, 0)
        self.assertEqual(self.g.visited_since_food, {})

    def test_head_is_first_cell(self):
        self.assertEqual(self.g.head, self.g.snake[0])

    def test_food_count_and_placement(self):
        self.assertEqual(len(self.g.foods), 3)
        # No apple is ever placed on the snake.
        self.assertTrue(set(self.g.foods).isdisjoint(set(self.g.snake)))
        # Apples are distinct cells.
        self.assertEqual(len(set(self.g.foods)), len(self.g.foods))

    def test_grid_cells(self):
        self.assertEqual(self.g.grid_cells, GRID_SIZE * GRID_SIZE)  # 256


class TestResetCurriculum(unittest.TestCase):
    """Curriculum starts must be a contiguous segment of HAMILTONIAN_CYCLE so the
    body stays contiguous in cycle indices (the teacher relies on this)."""

    def test_lengths_and_no_self_intersection(self):
        for start_length in (4, 10, 25, 50):
            g = new_game(num_apples=1)
            g.reset(start_length=start_length)
            self.assertEqual(len(g.snake), start_length)
            # No cell repeated — a curriculum snake never overlaps itself.
            self.assertEqual(len(set(g.snake)), start_length)
            self.assertEqual(g.head, g.snake[0])

    def test_body_is_contiguous_cycle_segment(self):
        cycle_index = {cell: i for i, cell in enumerate(HAMILTONIAN_CYCLE)}
        for _ in range(20):
            g = new_game(num_apples=1)
            g.reset(start_length=30)
            # snake[0] is the head (last cell of the forward segment); reversing
            # gives the forward-ordered segment, whose cycle indices must be a run.
            grid_cells = [(p.x // BLOCK_SIZE, p.y // BLOCK_SIZE) for p in reversed(g.snake)]
            indices = [cycle_index[c] for c in grid_cells]
            expected = list(range(indices[0], indices[0] + len(indices)))
            self.assertEqual(indices, expected)


class TestPlaceFood(unittest.TestCase):
    def test_never_on_snake_or_existing_food(self):
        g = new_game(num_apples=3)
        for _ in range(50):
            g._place_food(1)
            occupied_by_snake = set(g.foods) & set(g.snake)
            self.assertEqual(occupied_by_snake, set())
            self.assertEqual(len(set(g.foods)), len(g.foods))

    def test_graceful_degradation_on_full_board(self):
        # Fill every cell with the snake; _place_food must place 0 (never hang).
        g = new_game(num_apples=1)
        g.snake = [Point(x, y)
                   for x in range(0, g.w, BLOCK_SIZE)
                   for y in range(0, g.h, BLOCK_SIZE)]
        g.foods = []
        g._place_food(3)
        self.assertEqual(g.foods, [])


class TestIsCollision(unittest.TestCase):
    def setUp(self):
        self.g = new_game(num_apples=0)

    def test_wall_collisions(self):
        w, h = self.g.w, self.g.h
        self.assertTrue(self.g.is_collision(Point(-BLOCK_SIZE, 200)))
        self.assertTrue(self.g.is_collision(Point(w, 200)))          # x > w - BLOCK_SIZE
        self.assertTrue(self.g.is_collision(Point(200, -BLOCK_SIZE)))
        self.assertTrue(self.g.is_collision(Point(200, h)))

    def test_self_collision(self):
        # snake[1] is a body cell.
        self.assertTrue(self.g.is_collision(self.g.snake[1]))

    def test_free_cell_is_safe(self):
        # A cell directly ahead of the (centered) head is free.
        ahead = Point(self.g.head.x + BLOCK_SIZE, self.g.head.y)
        self.assertFalse(self.g.is_collision(ahead))


class TestPlayStep(unittest.TestCase):
    STRAIGHT = [1, 0, 0]

    def test_non_eating_move(self):
        g = new_game(num_apples=0)
        g.foods = []
        old_len = len(g.snake)
        old_head = g.head
        done, score = g.play_step(self.STRAIGHT)
        self.assertFalse(done)
        self.assertEqual(score, 0)
        self.assertEqual(len(g.snake), old_len)  # tail vacated → length unchanged
        self.assertEqual(g.head, Point(old_head.x + BLOCK_SIZE, old_head.y))
        self.assertEqual(g.frame_iteration, 1)

    def test_eating_grows_and_resets_counters(self):
        g = new_game(num_apples=0)
        # Put exactly one apple directly ahead of the head.
        target = Point(g.head.x + BLOCK_SIZE, g.head.y)
        g.foods = [target]
        g.visited_since_food = {(1, 1): 5}
        old_len = len(g.snake)
        done, score = g.play_step(self.STRAIGHT)
        self.assertFalse(done)
        self.assertEqual(score, 1)
        self.assertEqual(len(g.snake), old_len + 1)  # ate → grew
        self.assertEqual(g.frame_iteration, 0)       # counter reset on eat
        self.assertEqual(g.visited_since_food, {})   # cleared on eat

    def test_timeout_ends_game(self):
        g = new_game(num_apples=0)
        g.foods = []
        g.frame_iteration = 100 * len(g.snake) + 5  # already past the starvation cap
        done, _ = g.play_step(self.STRAIGHT)
        self.assertTrue(done)


class TestSafeMovesConsistency(unittest.TestCase):
    """The central invariant from CLAUDE.md: a move that safe_moves() reports as
    safe must never trigger is_collision() after it is actually taken. This is
    what keeps the teacher from 'colliding' with a tail cell that vacates."""

    def test_turning_into_vacating_tail_is_safe(self):
        # A tight 2x2 loop: head at (4,4) facing UP, tail at (5,4). A right turn
        # targets the tail cell, which vacates on this non-eating step.
        g = new_game(num_apples=0)
        g.snake = [Point(160, 160), Point(160, 200), Point(200, 200), Point(200, 160)]
        g.head = g.snake[0]
        g.direction = Direction.UP
        g.foods = []
        g.frame_iteration = 0

        safe = g.safe_moves()  # [straight, right, left]
        self.assertTrue(safe[1], "right turn into the vacating tail should be safe")

        done, _ = g.play_step([0, 1, 0])  # turn right, into the old tail cell
        self.assertFalse(done)
        self.assertEqual(g.head, Point(200, 160))
        self.assertEqual(len(g.snake), 4)

    def test_safe_move_never_collides_under_random_play(self):
        # Drive many random-but-safe steps across many resets; every move that
        # safe_moves() approved must not be a collision after it is taken.
        random.seed(1234)
        actions = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        for trial in range(40):
            g = new_game(num_apples=3)
            if trial % 2 == 0:
                g.reset(start_length=random.randint(4, 40))
            for _ in range(300):
                safe = g.safe_moves()
                safe_idxs = [i for i, ok in enumerate(safe) if ok]
                if not safe_idxs:
                    break  # genuinely trapped — collision unavoidable, not a bug
                choice = random.choice(safe_idxs)
                done, _ = g.play_step(actions[choice])
                # A safe-approved move must not have caused a body/wall collision.
                self.assertFalse(
                    g.is_collision(),
                    f"safe_moves approved action {choice} but it collided",
                )
                if done:
                    break  # win or timeout (not a collision) — start a fresh trial


if __name__ == "__main__":
    unittest.main(verbosity=2)
