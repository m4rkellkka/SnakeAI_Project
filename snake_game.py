import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(pygame.font.get_default_font(), 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors for rendering
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)

BLOCK_SIZE = 40

# Grid size in cells (640 // 40). Defined here and imported by teacher.py / train_ai.py
# to avoid duplication.
GRID_SIZE = 640 // BLOCK_SIZE

# Hamiltonian cycle over all grid cells — a list of (gx, gy) in grid coordinates.
# Traverses columns x=1..15 in a serpentine (alternating) pattern, with row y=0
# as a "highway" back to the start. Used in two places:
#  - reset(start_length=...) builds curriculum-start snakes as a contiguous segment
#    of this cycle, so the body remains contiguous in cycle indices;
#  - teacher.get_best_move() uses the same cycle and cycle_dist() for corner-cutting
#    shortcuts — the invariant "body contiguous in cycle indices" is needed for
#    correctness of avail_space, including on curriculum starts.
HAMILTONIAN_CYCLE = []
for x in range(GRID_SIZE):
    if x % 2 == 0:
        for y in range(1, GRID_SIZE):
            HAMILTONIAN_CYCLE.append((x, y))
    else:
        for y in range(GRID_SIZE - 1, 0, -1):
            HAMILTONIAN_CYCLE.append((x, y))
HAMILTONIAN_CYCLE.append((GRID_SIZE - 1, 0))
for x in range(GRID_SIZE - 2, -1, -1):
    HAMILTONIAN_CYCLE.append((x, 0))

# At speed == 0 (uncapped FPS, fast training mode), rendering is the most expensive part
# of play_step compared to the game logic itself. We render not every frame but once every
# RENDER_EVERY_N_FRAMES — the window stays responsive (events are pumped every frame, so
# speed-control keys work), but rendering cost drops dramatically.
RENDER_EVERY_N_FRAMES = 1000

class SnakeGameAI:
    def __init__(self, w=640, h=640, num_apples=3):
        self.w = w
        self.h = h
        self.num_apples = num_apples
        self.speed = 80 # Initial game speed
        self._render_counter = 0 # Frame counter for render throttling

        # Screen setup
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    @property
    def grid_cells(self):
        # Total number of grid cells (16x16 = 256) — used for win condition
        # (snake occupies entire grid).
        return (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE)

    def reset(self, start_length=None):
        # Initialize game state
        if start_length is None or start_length <= 3:
            # Normal start: short snake in center of grid.
            self.direction = Direction.RIGHT

            head_x = (self.w // 2 // BLOCK_SIZE) * BLOCK_SIZE
            head_y = (self.h // 2 // BLOCK_SIZE) * BLOCK_SIZE
            self.head = Point(head_x, head_y)
            self.snake = [self.head,
                          Point(self.head.x - BLOCK_SIZE, self.head.y),
                          Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        else:
            # Curriculum start: initialize with a long snake to expose the agent
            # to "late-game" states early. The snake is built as a contiguous segment
            # of HAMILTONIAN_CYCLE — the same cycle the teacher uses.
            # A contiguous segment of the cycle is guaranteed to not self-intersect
            # and remain contiguous in cycle indices, which the teacher's avail_space
            # heuristics depend on.
            path = [Point(gx * BLOCK_SIZE, gy * BLOCK_SIZE) for gx, gy in HAMILTONIAN_CYCLE]

            start_length = min(start_length, len(path))
            i = random.randint(0, len(path) - start_length)
            segment = path[i:i + start_length]

            # snake[0] is the head. Head is the last cell of the segment,
            # body extends backward to the segment start.
            self.snake = list(reversed(segment))
            self.head = self.snake[0]

            # Direction is derived from the vector (head - second cell).
            dx = self.head.x - self.snake[1].x
            dy = self.head.y - self.snake[1].y
            if dx > 0:
                self.direction = Direction.RIGHT
            elif dx < 0:
                self.direction = Direction.LEFT
            elif dy > 0:
                self.direction = Direction.DOWN
            else:
                self.direction = Direction.UP

        self.score = 0
        self.foods = []
        self._place_food(self.num_apples)
        self.frame_iteration = 0

    def _place_food(self, count):
        # Collect all free cells (not occupied by snake or existing food),
        # shuffle, and take the first count. Unlike the old while-True with random
        # retries, this never hangs — if the board is nearly full, it just places
        # fewer apples (graceful degradation).
        occupied = set(self.snake) | set(self.foods)
        free_cells = [
            Point(x, y)
            for x in range(0, self.w, BLOCK_SIZE)
            for y in range(0, self.h, BLOCK_SIZE)
            if Point(x, y) not in occupied
        ]
        random.shuffle(free_cells)
        self.foods.extend(free_cells[:count])

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. Collect user events (including speed control)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.speed += 20 # Increase speed
                elif event.key == pygame.K_MINUS:
                    self.speed = max(10, self.speed - 20) # Decrease speed (minimum 10)
                elif event.key == pygame.K_0:
                    self.speed = 0 # 0 means uncapped FPS (max speed)

        # 2. Move
        self._move(action) # Updates self.head

        # If food is not eaten, the tail leaves its cell on THIS same step —
        # this is why safe_moves() considers moving into the current tail cell safe.
        # Thus we insert the head and pop the tail BEFORE checking is_collision();
        # otherwise the head would temporarily overlap the tail (not yet freed)
        # and falsely register as a collision.
        will_eat = self.head in self.foods

        self.snake.insert(0, self.head)
        if not will_eat:
            self.snake.pop() # Tail leaves the cell on this same step

        game_over = False

        # 3. Check collision (wall, self, or timeout)
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            return game_over, self.score

        # 4. Check food consumption
        if will_eat:
            self.score += 1
            self.foods.remove(self.head)

            # Win condition: snake occupies entire grid. Check BEFORE _place_food(1),
            # because placing a new apple on a full board would have nowhere to go.
            if len(self.snake) >= self.grid_cells:
                game_over = True
                return game_over, self.score

            self._place_food(1)
            self.frame_iteration = 0 # RESET COUNTER! Otherwise snake would starve over time.

        # 5. Render
        # At max speed (speed == 0, uncapped FPS training), rendering is the most
        # expensive part, so we render only every RENDER_EVERY_N_FRAMES frames.
        # pygame.event.get() is still called every frame (above), so the window
        # stays responsive and speed-control keys work.
        self._render_counter += 1
        if self.speed == 0:
            if self._render_counter % RENDER_EVERY_N_FRAMES == 0:
                self._update_ui()
        else:
            self._update_ui()
            self.clock.tick(self.speed)

        # 6. Return results
        return game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hit wall
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hit self
        if pt in self.snake[1:]:
            return True
        return False

    def safe_moves(self):
        # For each of the 3 relative actions [straight, right, left], we compute
        # where the head would move without changing state, then check if that cell
        # collides with wall or body.
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # The tail (last cell) is freed on non-eating moves, so it doesn't count
        # as an obstacle — otherwise the agent would falsely think a move into the
        # current tail cell is unsafe and lose maneuver space at edges.
        body_without_tail = self.snake[:-1]

        result = []
        for next_idx_offset in (0, 1, -1): # straight, right, left
            new_dir = clock_wise[(idx + next_idx_offset) % 4]

            x, y = self.head.x, self.head.y
            if new_dir == Direction.RIGHT:
                x += BLOCK_SIZE
            elif new_dir == Direction.LEFT:
                x -= BLOCK_SIZE
            elif new_dir == Direction.DOWN:
                y += BLOCK_SIZE
            elif new_dir == Direction.UP:
                y -= BLOCK_SIZE

            new_head = Point(x, y)
            in_bounds = 0 <= new_head.x <= self.w - BLOCK_SIZE and 0 <= new_head.y <= self.h - BLOCK_SIZE
            safe = in_bounds and new_head not in body_without_tail
            result.append(safe)

        return result

    def _update_ui(self):
        self.display.fill(BLACK)

        # Render snake
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, 32, 32))

        # Render apples
        for food in self.foods:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Display score and speed
        text = font.render(f"Score: {self.score} | Speed: {'MAX' if self.speed == 0 else self.speed}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, turn right, turn left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # Straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Right
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
