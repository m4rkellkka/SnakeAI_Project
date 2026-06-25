"""Tests for the Hamiltonian-cycle teacher (src/teacher.py).

Run headless with plain stdlib unittest:

    python3 tests/test_teacher.py

The headline guarantee in CLAUDE.md is that the teacher is provably collision-free
and reaches the maximum score from both standard and curriculum starts. These tests
fuzz full games (each ~11k steps, ~0.3s headless) and assert exactly that: every
game ends in a WIN (full board), never a collision.
"""

import os
import sys
import random
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snake_game import SnakeGameAI, GRID_SIZE  # noqa: E402
from teacher import get_best_move  # noqa: E402

GRID_CELLS = GRID_SIZE * GRID_SIZE  # 256 — a WIN means the snake fills the whole board

# A standard game starts at length 3, so its winning score is 253. Curriculum games
# start longer, so their winning score is GRID_CELLS - start_length (fewer apples to
# eat) — which is why "won" is measured by a full board, not a fixed score.
STANDARD_MAX_SCORE = GRID_CELLS - 3  # 253


def play_full_game(game):
    """Drive a game to completion with the teacher. Returns (final_score, collided, won).

    `collided` is True only for a real wall/self collision — both a win (full board)
    and a timeout end with is_collision() == False, so `won` is measured separately
    as the snake having filled every cell.
    """
    while True:
        action = get_best_move(game)
        done, score = game.play_step(action)
        if done:
            won = len(game.snake) >= GRID_CELLS
            return score, game.is_collision(), won


class TestActionValidity(unittest.TestCase):
    def test_returns_one_hot_triplet(self):
        g = SnakeGameAI(headless=True, num_apples=1)
        g.reset(start_length=3)
        for _ in range(200):
            action = get_best_move(g)
            self.assertEqual(len(action), 3)
            self.assertEqual(sum(action), 1)          # exactly one move chosen
            self.assertTrue(all(a in (0, 1) for a in action))
            done, _ = g.play_step(action)
            if done:
                break


class TestTeacherIsCollisionFree(unittest.TestCase):
    def test_standard_starts_always_win(self):
        random.seed(2024)
        for i in range(12):
            g = SnakeGameAI(headless=True, num_apples=1)
            g.reset(start_length=3)
            score, collided, won = play_full_game(g)
            self.assertFalse(collided, f"standard game {i} died by collision")
            self.assertTrue(won, f"standard game {i} did not fill the board")
            self.assertEqual(score, STANDARD_MAX_SCORE, f"standard game {i} wrong score")

    def test_curriculum_starts_always_win(self):
        random.seed(777)
        for i in range(12):
            g = SnakeGameAI(headless=True, num_apples=1)
            start_length = random.randint(4, 50)
            g.reset(start_length=start_length)
            score, collided, won = play_full_game(g)
            self.assertFalse(collided, f"curriculum game {i} died by collision")
            self.assertTrue(won, f"curriculum game {i} (len {start_length}) did not fill the board")
            # Curriculum games start longer, so a win eats GRID_CELLS - start_length apples.
            self.assertEqual(score, GRID_CELLS - start_length, f"curriculum game {i} wrong score")


if __name__ == "__main__":
    unittest.main(verbosity=2)
