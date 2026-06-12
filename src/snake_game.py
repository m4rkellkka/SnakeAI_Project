import pygame
import random
import math
from enum import Enum
from collections import namedtuple
import numpy as np

# Defer pygame.init() — only initialize when needed (not in headless mode)
_pygame_initialized = False

def _ensure_pygame():
    global _pygame_initialized
    if not _pygame_initialized:
        pygame.init()
        _pygame_initialized = True


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


# ──────────────────────────────────────────────────
# Premium color palette
# ──────────────────────────────────────────────────
_BG_COLOR = (13, 15, 18)
_GRID_LINE_COLOR = (28, 32, 38)
_GRID_DOT_COLOR = (35, 40, 48)

# Snake gradient: bright green head → dark teal tail
_SNAKE_HEAD_COLOR = (0, 230, 120)
_SNAKE_HEAD_GLOW = (0, 255, 140)
_SNAKE_BODY_START = (0, 200, 100)
_SNAKE_BODY_END = (0, 80, 60)
_SNAKE_EYE_WHITE = (240, 240, 240)
_SNAKE_EYE_PUPIL = (20, 20, 20)

# Food
_FOOD_COLOR = (255, 60, 80)
_FOOD_GLOW_COLOR = (255, 40, 60)

# HUD
_HUD_BG = (0, 0, 0, 140)
_HUD_TEXT_COLOR = (220, 230, 220)
_HUD_ACCENT = (0, 220, 110)

# Score popup
_SCORE_POPUP_COLOR = (0, 255, 140)


def _lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


class SnakeGameAI:
    def __init__(self, w=640, h=640, num_apples=3, headless=False):
        self.w = w
        self.h = h
        self.num_apples = num_apples
        self.speed = 80 # Initial game speed
        self._render_counter = 0 # Frame counter for render throttling
        self.headless = headless
        self._frame_tick = 0  # For animations (food pulse, etc.)

        if not headless:
            _ensure_pygame()
            # Screen setup
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
            self.clock = pygame.time.Clock()
            self._font_large = pygame.font.SysFont('Helvetica Neue,Helvetica,Arial', 22, bold=True)
            self._font_small = pygame.font.SysFont('Helvetica Neue,Helvetica,Arial', 14)

            # Pre-render the grid background (static — only done once)
            self._grid_surface = self._build_grid_surface()

            # Glow surface for food (pre-computed for performance)
            self._food_glow_surface = self._build_glow_surface(BLOCK_SIZE // 2, _FOOD_GLOW_COLOR, alpha=40)
            self._head_glow_surface = self._build_glow_surface(BLOCK_SIZE // 2, _SNAKE_HEAD_GLOW, alpha=30)
        else:
            self.display = None
            self.clock = None

        self.reset()

    def _build_grid_surface(self):
        """Pre-render the dark grid background with subtle lines and corner dots."""
        surf = pygame.Surface((self.w, self.h))
        surf.fill(_BG_COLOR)

        # Subtle grid lines
        for x in range(0, self.w + 1, BLOCK_SIZE):
            pygame.draw.line(surf, _GRID_LINE_COLOR, (x, 0), (x, self.h))
        for y in range(0, self.h + 1, BLOCK_SIZE):
            pygame.draw.line(surf, _GRID_LINE_COLOR, (0, y), (self.w, y))

        # Small dots at grid intersections for premium feel
        for x in range(0, self.w + 1, BLOCK_SIZE):
            for y in range(0, self.h + 1, BLOCK_SIZE):
                pygame.draw.circle(surf, _GRID_DOT_COLOR, (x, y), 1)

        return surf

    def _build_glow_surface(self, radius, color, alpha=60):
        """Create a soft radial glow surface."""
        size = radius * 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        center = radius
        for r in range(radius, 0, -1):
            a = int(alpha * (r / radius) ** 2)
            a = max(0, min(255, a))
            pygame.draw.circle(surf, (*color, a), (center, center), r)
        return surf

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
        self._frame_tick += 1

        # 1. Collect user events (including speed control)
        if not self.headless:
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
        if not self.headless:
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
        if self.headless:
            return

        # Blit pre-rendered grid background
        self.display.blit(self._grid_surface, (0, 0))

        # ── Render snake ──
        snake_len = len(self.snake)
        for i, pt in enumerate(self.snake):
            t = i / max(1, snake_len - 1)  # 0 = head, 1 = tail

            # Gradient color from head to tail
            color = _lerp_color(_SNAKE_BODY_START, _SNAKE_BODY_END, t)

            # Segment size shrinks slightly toward tail for a tapered look
            shrink = int(t * 4)
            rect = pygame.Rect(pt.x + shrink, pt.y + shrink,
                               BLOCK_SIZE - shrink * 2, BLOCK_SIZE - shrink * 2)

            # Rounded rectangle for body segments
            border_radius = max(4, BLOCK_SIZE // 4 - int(t * 4))

            if i == 0:
                # HEAD — brighter, with subtle glow
                head_rect = pygame.Rect(pt.x + 1, pt.y + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2)

                pygame.draw.rect(self.display, _SNAKE_HEAD_COLOR, head_rect,
                                 border_radius=BLOCK_SIZE // 3)

                # Eyes
                self._draw_eyes(pt)
            else:
                pygame.draw.rect(self.display, color, rect, border_radius=border_radius)

                # Subtle inner highlight for depth
                inner_rect = pygame.Rect(rect.x + 2, rect.y + 2,
                                         rect.width - 4, rect.height - 4)
                highlight = _lerp_color(color, (255, 255, 255), 0.08)
                pygame.draw.rect(self.display, highlight, inner_rect,
                                 border_radius=max(2, border_radius - 2))

        # ── Render food ──
        pulse = math.sin(self._frame_tick * 0.08) * 0.06 + 0.94  # subtle 0.88-1.0
        for food in self.foods:
            # Small food rectangle with rounded corners
            margin = int(BLOCK_SIZE * 0.2)
            food_size = BLOCK_SIZE - margin * 2
            food_rect = pygame.Rect(food.x + margin, food.y + margin,
                                     food_size, food_size)
            pygame.draw.rect(self.display, _FOOD_COLOR, food_rect, border_radius=6)

            # Small shine highlight
            shine_rect = pygame.Rect(food.x + margin + 3, food.y + margin + 3,
                                      food_size // 3, food_size // 3)
            pygame.draw.rect(self.display, (255, 130, 150), shine_rect, border_radius=3)

        # ── HUD overlay ──
        self._draw_hud()

        pygame.display.flip()

    def _draw_eyes(self, head_pt):
        """Draw two eyes on the head, oriented by direction."""
        cx = head_pt.x + BLOCK_SIZE // 2
        cy = head_pt.y + BLOCK_SIZE // 2
        eye_offset = BLOCK_SIZE // 5
        eye_r = BLOCK_SIZE // 7
        pupil_r = max(2, eye_r // 2)

        if self.direction == Direction.RIGHT:
            e1 = (cx + eye_offset, cy - eye_offset)
            e2 = (cx + eye_offset, cy + eye_offset)
            p_off = (2, 0)
        elif self.direction == Direction.LEFT:
            e1 = (cx - eye_offset, cy - eye_offset)
            e2 = (cx - eye_offset, cy + eye_offset)
            p_off = (-2, 0)
        elif self.direction == Direction.UP:
            e1 = (cx - eye_offset, cy - eye_offset)
            e2 = (cx + eye_offset, cy - eye_offset)
            p_off = (0, -2)
        else:  # DOWN
            e1 = (cx - eye_offset, cy + eye_offset)
            e2 = (cx + eye_offset, cy + eye_offset)
            p_off = (0, 2)

        for ex, ey in (e1, e2):
            pygame.draw.circle(self.display, _SNAKE_EYE_WHITE, (ex, ey), eye_r)
            pygame.draw.circle(self.display, _SNAKE_EYE_PUPIL,
                               (ex + p_off[0], ey + p_off[1]), pupil_r)

    def _draw_hud(self):
        """Draw a semi-transparent HUD bar at the top."""
        hud_height = 36
        hud_surf = pygame.Surface((self.w, hud_height), pygame.SRCALPHA)
        hud_surf.fill(_HUD_BG)
        self.display.blit(hud_surf, (0, 0))

        # Score
        score_text = self._font_large.render(f"SCORE  {self.score}", True, _HUD_ACCENT)
        self.display.blit(score_text, (12, 7))

        # Speed
        speed_str = 'MAX' if self.speed == 0 else str(self.speed)
        speed_text = self._font_small.render(f"SPEED: {speed_str}", True, _HUD_TEXT_COLOR)
        self.display.blit(speed_text, (self.w - speed_text.get_width() - 12, 11))

        # Snake length
        len_text = self._font_small.render(f"LEN: {len(self.snake)}", True, _HUD_TEXT_COLOR)
        self.display.blit(len_text, (self.w - speed_text.get_width() - len_text.get_width() - 30, 11))

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
