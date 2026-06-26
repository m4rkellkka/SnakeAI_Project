"""
Headless game streamer for the Tauri app's in-app Canvas "Watch AI".

Runs the AI (a checkpoint / the pretrained model) or the Hamiltonian-cycle
teacher with NO Pygame window, and prints one JSON line per step to stdout:

    {"snake": [[gx,gy], ...], "foods": [[gx,gy], ...],
     "dir": "RIGHT", "score": N, "over": false, "game": 1}

Coordinates are in GRID CELLS (0..15), not pixels — the React <GameCanvas>
renders at whatever pixel size it wants. The Tauri backend (lib.rs) already
streams a child process's stdout line-by-line to the frontend via the
`proc-out:<id>` event, so no new IPC is needed: the frontend parses each line
into a frame and feeds it to the same canvas the manual game uses.

This is the "Python is the brain" model (variant A): inference stays in
PyTorch; the browser only draws.
"""
import os
import sys
import json
import time
import argparse

# Hide pygame's import-time banner — it writes to stdout, which here carries the
# JSON frame stream. Must be set before snake_game (which imports pygame) loads.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from snake_game import SnakeGameAI, BLOCK_SIZE
from teacher import get_best_move
from train_ai import Agent, NUM_APPLES


def to_cells(points):
    """Convert pixel-space Points (multiples of BLOCK_SIZE) to [gx, gy] cells."""
    return [[p.x // BLOCK_SIZE, p.y // BLOCK_SIZE] for p in points]


def emit(game, game_idx, over=False):
    frame = {
        "snake": to_cells(game.snake),
        "foods": to_cells(game.foods),
        "dir": game.direction.name,
        "score": game.score,
        "over": over,
        "game": game_idx,
    }
    sys.stdout.write(json.dumps(frame, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def run(mode, checkpoint_name, num_games, unstick, delay):
    agent = None
    if mode != "teacher":
        agent = Agent()
        loaded = agent.load_checkpoint(checkpoint_name)
        if loaded is None:
            print(f"[err] {checkpoint_name} not found — using untrained network.",
                  file=sys.stderr, flush=True)
        else:
            print(f"Loaded {checkpoint_name} (games trained: {agent.n_games})",
                  file=sys.stderr, flush=True)

    game = SnakeGameAI(w=640, h=640, num_apples=NUM_APPLES, headless=True)

    for i in range(1, num_games + 1):
        game.reset()
        visited = set()
        emit(game, i)

        done = False
        while not done:
            if mode == "teacher":
                action = get_best_move(game)
            else:
                key = (game.head, game.direction)
                if unstick and key in visited:
                    # Loop-breaker: same crutch as train_ai.play_game — when the
                    # network repeats a (head, direction) since the last apple,
                    # follow the teacher for one step to diverge out of the loop.
                    action = get_best_move(game)
                else:
                    state = agent.get_state(game)
                    action = agent.get_network_action(state, game)
                if unstick:
                    visited.add(key)

            done, _ = game.play_step(action)
            if game.frame_iteration == 0:  # apple eaten — loop history resets
                visited.clear()

            emit(game, i, over=done)
            if delay > 0:
                time.sleep(delay)

    print("Done.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream a headless Snake game as JSON lines")
    parser.add_argument("--mode", choices=["network", "teacher"], default="network",
                        help="network = run a checkpoint; teacher = Hamiltonian-cycle teacher")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth",
                        help="Checkpoint under model/ (e.g. pretrained.pth, foo/checkpoint_best.pth)")
    parser.add_argument("--games", type=int, default=1000,
                        help="Number of games to play before exiting")
    parser.add_argument("--no-unstick", action="store_true",
                        help="Disable the teacher-assisted loop-breaker (raw network)")
    parser.add_argument("--delay", type=float, default=0.07,
                        help="Seconds to sleep between steps (controls playback speed)")
    args = parser.parse_args()

    try:
        run(args.mode, args.checkpoint, args.games, not args.no_unstick, args.delay)
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass  # frontend stopped reading — exit quietly
