"""
Human-playable Snake game with premium visuals.
Controls: WASD or Arrow keys. ESC to quit.
"""
import pygame
import sys
from snake_game import SnakeGameAI, Direction, BLOCK_SIZE


def play_manual():
    game = SnakeGameAI(w=640, h=640, num_apples=1)
    game.speed = 10  # Comfortable human speed
    pygame.display.set_caption('Snake AI — Manual Play (WASD / Arrows)')

    game.reset()
    print("Manual play started! Use WASD or Arrow keys. ESC to quit.")
    print("'+'/'-' adjust speed, '0' for max speed.")

    while True:
        # Default action: go straight
        action = [1, 0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    game.speed += 2
                elif event.key == pygame.K_MINUS:
                    game.speed = max(3, game.speed - 2)
                elif event.key == pygame.K_0:
                    game.speed = 0

                # Map absolute direction keys to relative actions
                # [straight, right, left]
                desired_dir = None
                if event.key in (pygame.K_w, pygame.K_UP):
                    desired_dir = Direction.UP
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    desired_dir = Direction.DOWN
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    desired_dir = Direction.LEFT
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    desired_dir = Direction.RIGHT

                if desired_dir is not None:
                    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
                    cur_idx = clock_wise.index(game.direction)
                    des_idx = clock_wise.index(desired_dir)

                    if des_idx == cur_idx:
                        action = [1, 0, 0]  # Straight
                    elif des_idx == (cur_idx + 1) % 4:
                        action = [0, 1, 0]  # Right turn
                    elif des_idx == (cur_idx - 1) % 4:
                        action = [0, 0, 1]  # Left turn
                    else:
                        # Reverse direction — ignore (can't go backwards)
                        action = [1, 0, 0]

        # We need to handle play_step without its internal event handling
        # since we already consumed events above.
        # Instead, use the game's play_step which handles rendering
        game.frame_iteration += 1
        game._frame_tick += 1
        game._move(action)

        will_eat = game.head in game.foods
        game.snake.insert(0, game.head)
        if not will_eat:
            game.snake.pop()

        game_over = False
        if game.is_collision() or game.frame_iteration > 100 * len(game.snake):
            game_over = True

        if not game_over and will_eat:
            game.score += 1
            game.foods.remove(game.head)
            if len(game.snake) >= game.grid_cells:
                game_over = True
            else:
                game._place_food(1)
                game.frame_iteration = 0

        game._update_ui()
        game.clock.tick(game.speed if game.speed > 0 else 60)

        if game_over:
            print(f"Game Over! Score: {game.score} | Length: {len(game.snake)}")
            game.reset()


if __name__ == '__main__':
    play_manual()
