#!/usr/bin/env python3
"""
Benchmark harness: run N games across one or more checkpoints and report
full score distributions (not just avg/max).

Usage:
    python tools/benchmark.py --checkpoint checkpoint_best.pth --games 100
    python tools/benchmark.py --checkpoint checkpoint_best.pth \
                               --checkpoint sweep_lr0.001/checkpoint_best.pth \
                               --games 100 --seed 42 \
                               --plot benchmark.png --csv benchmark.csv
"""

import os
import sys
import argparse
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')  # No GUI — safe alongside headless Pygame
import matplotlib.pyplot as plt

# Add src/ directory to path so we can import the game and agent modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import train_ai
from snake_game import SnakeGameAI


def run_benchmark(checkpoint_path, num_games, seed, render, unstick=True):
    """Run num_games games and return (scores, checkpoint, max_score, stuck_count).
    stuck_count = games that ended in a loop-timeout (the failure mode the network
    fix targets) — most visible with --no-unstick."""
    dirname, basename = os.path.split(checkpoint_path)
    train_ai.MODEL_FOLDER = os.path.join('./model', dirname) if dirname else './model'

    agent = train_ai.Agent()
    checkpoint = agent.load_checkpoint(basename)
    if checkpoint is None:
        print(f"  [warning] {checkpoint_path}: checkpoint not found, using untrained network")
    else:
        n_games = checkpoint.get('n_games')
        eval_score = checkpoint.get('eval_score')
        run_config = checkpoint.get('run_config')
        meta_parts = []
        if n_games is not None:
            meta_parts.append(f"n_games={n_games}")
        if eval_score is not None:
            meta_parts.append(f"eval_score={eval_score:.2f}")
        if run_config is not None:
            meta_parts.append(f"run_config={run_config}")
        print(f"  Loaded {checkpoint_path} ({', '.join(meta_parts) if meta_parts else 'no metadata'})")
    agent.model.eval()

    game = SnakeGameAI(w=640, h=640, num_apples=train_ai.NUM_APPLES, headless=not render)
    game.speed = 30 if render else 0

    max_score = game.grid_cells - 3  # initial snake length is 3 (plain reset())

    scores = []
    stuck_count = 0
    for i in range(num_games):
        # Reseeding per game index before reset() gives identical food-placement
        # sequences across checkpoints for the same --seed (paired comparison) —
        # _place_food() consumes random.shuffle().
        if seed is not None:
            random.seed(seed + i)
        game.reset()
        score = train_ai.play_game(agent, game, unstick=unstick)
        scores.append(score)
        if train_ai.game_outcome(game) == 'stuck':
            stuck_count += 1

    return scores, checkpoint, max_score, stuck_count


def compute_stats(scores, max_score, stuck_count=0):
    arr = np.array(scores, dtype=float)
    return {
        'n': len(arr),
        'mean': arr.mean(),
        'median': np.median(arr),
        'std': arr.std(),
        'min': arr.min(),
        'max': arr.max(),
        'p10': np.percentile(arr, 10),
        'p25': np.percentile(arr, 25),
        'p75': np.percentile(arr, 75),
        'p90': np.percentile(arr, 90),
        'win_rate': float(np.mean(arr == max_score)),
        'stuck_rate': stuck_count / len(arr) if len(arr) else 0.0,
    }


def print_summary(checkpoint_path, stats, run_config):
    print(f"\n{checkpoint_path}")
    print(f"  n={stats['n']}  mean={stats['mean']:.2f}  median={stats['median']:.1f}  "
          f"std={stats['std']:.2f}")
    print(f"  min={stats['min']:.0f}  p10={stats['p10']:.1f}  p25={stats['p25']:.1f}  "
          f"p75={stats['p75']:.1f}  p90={stats['p90']:.1f}  max={stats['max']:.0f}")
    print(f"  win_rate={stats['win_rate']:.1%}  stuck_rate={stats['stuck_rate']:.1%}")
    if run_config:
        print(f"  run_config: {run_config}")


def print_comparison_table(results):
    """results: list of (checkpoint_path, stats, run_config_or_None)."""
    name_w = 35
    print("\n" + "=" * (name_w + 68))
    header = (f"{'checkpoint':<{name_w}} {'mean':>7} {'median':>7} {'std':>7} "
              f"{'min':>5} {'max':>5} {'win%':>7} {'stuck%':>7}  run_config")
    print(header)
    print("-" * (name_w + 68))
    for path, stats, run_config in results:
        label = path if len(path) <= name_w else '...' + path[-(name_w - 3):]
        cfg_str = ''
        if run_config:
            cfg_str = ', '.join(f"{k}={v}" for k, v in run_config.items())
        print(f"{label:<{name_w}} {stats['mean']:>7.2f} {stats['median']:>7.1f} "
              f"{stats['std']:>7.2f} {stats['min']:>5.0f} {stats['max']:>5.0f} "
              f"{stats['win_rate']:>6.1%} {stats['stuck_rate']:>6.1%}  {cfg_str}")
    print("=" * (name_w + 68))


def write_csv(path, all_scores):
    """all_scores: list of (checkpoint_path, scores_list)."""
    with open(path, 'w') as f:
        f.write("checkpoint,game_index,score\n")
        for checkpoint_path, scores in all_scores:
            for i, score in enumerate(scores):
                f.write(f"{checkpoint_path},{i},{score}\n")
    print(f"\nWrote per-game CSV to {path}")


def save_plot(path, all_scores, all_stats, max_score):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    bins = np.arange(0, max_score + 2) - 0.5  # integer-aligned bins
    for idx, ((checkpoint_path, scores), stats) in enumerate(zip(all_scores, all_stats)):
        color = colors[idx % len(colors)]
        ax.hist(scores, bins=bins, alpha=0.5, color=color,
                label=f"{checkpoint_path} (mean={stats['mean']:.1f})")
    ax.set_xlabel('Score')
    ax.set_ylabel('Games')
    ax.set_title(f'Score distribution ({all_stats[0]["n"]} games each)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved plot to {path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark one or more checkpoints over N games')
    parser.add_argument('--checkpoint', action='append', default=None,
                         help='Checkpoint path relative to model/ (may include subdir, e.g. '
                              "sweep_lr0.001/checkpoint_best.pth). May be repeated. "
                              "Default: checkpoint_best.pth")
    parser.add_argument('--games', type=int, default=100,
                         help='Number of games per checkpoint (default 100)')
    parser.add_argument('--seed', type=int, default=None,
                         help='If set, seed RNG per-game (seed + game_index) for paired comparison')
    parser.add_argument('--render', action='store_true',
                         help='Show a window (headless=False, speed=30). Default: headless, speed=0')
    parser.add_argument('--plot', type=str, default=None, metavar='PATH',
                         help='Save overlaid score-distribution histogram to PATH')
    parser.add_argument('--csv', type=str, default=None, metavar='PATH',
                         help='Save long-format per-game scores (checkpoint,game_index,score) to PATH')
    parser.add_argument('--no-unstick', action='store_true',
                         help='Disable the teacher-assisted loop-breaking fallback '
                              '(pure honest network policy)')
    args = parser.parse_args()

    checkpoints = args.checkpoint if args.checkpoint else ['checkpoint_best.pth']

    all_scores = []
    all_stats = []
    all_run_configs = []
    max_score = None

    for checkpoint_path in checkpoints:
        print(f"\nRunning {args.games} games for {checkpoint_path}"
              + (" (render, speed=30)" if args.render else " (headless)"))
        scores, checkpoint, max_score, stuck_count = run_benchmark(
            checkpoint_path, args.games, args.seed, args.render, unstick=not args.no_unstick
        )
        stats = compute_stats(scores, max_score, stuck_count)
        run_config = checkpoint.get('run_config') if checkpoint else None

        print_summary(checkpoint_path, stats, run_config)

        all_scores.append((checkpoint_path, scores))
        all_stats.append(stats)
        all_run_configs.append(run_config)

    if len(checkpoints) > 1:
        print_comparison_table(list(zip(checkpoints, all_stats, all_run_configs)))

    if args.csv:
        write_csv(args.csv, all_scores)

    if args.plot:
        save_plot(args.plot, all_scores, all_stats, max_score)


if __name__ == '__main__':
    main()
