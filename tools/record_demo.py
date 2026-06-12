#!/usr/bin/env python3
"""
Headless demo recorder: generates animated GIFs of Snake AI gameplay.
Supports both pretrained (smart) and untrained (random) network modes.

Usage:
    python tools/record_demo.py pretrained   # Generates demo_pretrained.gif
    python tools/record_demo.py untrained    # Generates demo_untrained.gif
"""

import os
import sys

# Headless rendering — must be set BEFORE pygame.init()
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
from PIL import Image
import torch

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snake_game import SnakeGameAI
from train_ai import Agent


def record_demo(pretrained=True, max_frames=400, output_path='demo.gif', target_fps=10):
    """
    Record a game as an animated GIF.

    Args:
        pretrained: If True, load pretrained.pth; if False, use untrained network.
        max_frames: Maximum frames to record per game.
        output_path: Output GIF file path.
        target_fps: Frames per second in output GIF (affects duration per frame).
    """
    print(f"Recording {'pretrained' if pretrained else 'untrained'} demo to {output_path}...")

    # Initialize agent
    agent = Agent()
    if pretrained:
        checkpoint = agent.load_checkpoint('model/pretrained.pth')
        if checkpoint is None:
            print("ERROR: model/pretrained.pth not found!")
            sys.exit(1)
        agent.model.eval()
        print(f"  Loaded pretrained model (eval score: {checkpoint.get('eval_score', 'N/A')})")
    else:
        print("  Using untrained network (random initialization)")

    # Initialize game with headless rendering
    game = SnakeGameAI(w=640, h=640, num_apples=1)
    game.speed = 0  # Uncapped (we control timing via pygame.time.Clock in rendering)
    game.reset()

    frames = []
    frame_count = 0
    done = False

    with torch.no_grad():
        while not done and frame_count < max_frames:
            # Get state and action
            state = agent.get_state(game)
            action = agent.get_network_action(state, game)

            # Step environment
            done, score = game.play_step(action)

            # Capture frame (render once per step)
            game._update_ui()
            surface = game.display
            frame_array = pygame.surfarray.array3d(surface)  # (width, height, 3)
            frame_array = np.transpose(frame_array, (1, 0, 2))  # (height, width, 3)
            frame_array = np.flip(frame_array, axis=2)  # BGR -> RGB

            frames.append(Image.fromarray(frame_array, mode='RGB'))
            frame_count += 1

            if (frame_count + 1) % 50 == 0:
                print(f"  Frame {frame_count + 1} | Score: {score}")

    print(f"  Total frames recorded: {frame_count}")

    if not frames:
        print("ERROR: No frames recorded!")
        return False

    # Downscale frames if too large (for reasonable file size)
    # 640x640 is large; downscale to 320x320
    downscale_size = (320, 320)
    frames = [frame.resize(downscale_size, Image.Resampling.LANCZOS) for frame in frames]

    # Save as animated GIF
    # duration = 1000 / target_fps gives milliseconds per frame
    duration_per_frame = int(1000 / target_fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_per_frame,
        loop=0,  # Infinite loop
        optimize=False
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved to {output_path} ({file_size_mb:.2f} MB)")
    return True


def main():
    # Create assets directory if it doesn't exist
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage: python tools/record_demo.py [pretrained|untrained|both]")
        print("       Generates demo GIFs in assets/ folder")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == 'pretrained' or mode == 'both':
        output = os.path.join(assets_dir, 'demo_pretrained.gif')
        success = record_demo(pretrained=True, max_frames=400, output_path=output)
        if not success:
            sys.exit(1)

    if mode == 'untrained' or mode == 'both':
        output = os.path.join(assets_dir, 'demo_untrained.gif')
        success = record_demo(pretrained=False, max_frames=400, output_path=output)
        if not success:
            sys.exit(1)

    print("\nDemo recording complete!")


if __name__ == '__main__':
    main()
