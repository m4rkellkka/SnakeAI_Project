// Pure-JS Snake engine for manual play in the browser.
//
// This is a faithful 1:1 port of the invariants in src/snake_game.py — same
// 16x16 grid, same relative action model ([straight, right, left]), same
// collision / food / win / timeout rules — so manual play behaves identically
// to the Python engine the AI was trained against. Coordinates are in GRID
// CELLS (0..15), not pixels: the renderer decides the pixel size.

export const GRID = 16;

// Clockwise order, matching snake_game.py's `clock_wise` list. A "right" turn
// advances one step clockwise, a "left" turn one step counter-clockwise.
const CW = ["RIGHT", "DOWN", "LEFT", "UP"];

const DELTAS = {
  RIGHT: { x: 1, y: 0 },
  LEFT: { x: -1, y: 0 },
  UP: { x: 0, y: -1 },
  DOWN: { x: 0, y: 1 },
};

// Action constants (relative to current heading).
export const STRAIGHT = "straight";
export const RIGHT = "right";
export const LEFT = "left";

const key = (p) => `${p.x},${p.y}`;

function placeFood(state, count) {
  const occupied = new Set([...state.snake, ...state.foods].map(key));
  const free = [];
  for (let x = 0; x < GRID; x++) {
    for (let y = 0; y < GRID; y++) {
      const p = { x, y };
      if (!occupied.has(key(p))) free.push(p);
    }
  }
  // Fisher–Yates shuffle, take first `count`.
  for (let i = free.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [free[i], free[j]] = [free[j], free[i]];
  }
  state.foods.push(...free.slice(0, count));
}

// Create a fresh game. Mirrors reset() for the normal (short) start: a
// 3-cell snake in the center heading RIGHT.
export function createState(numApples = 1) {
  const hx = Math.floor(GRID / 2);
  const hy = Math.floor(GRID / 2);
  const state = {
    snake: [
      { x: hx, y: hy },
      { x: hx - 1, y: hy },
      { x: hx - 2, y: hy },
    ],
    dir: "RIGHT",
    foods: [],
    score: 0,
    over: false,
    frame: 0,
    numApples,
  };
  placeFood(state, numApples);
  return state;
}

// Resolve a relative action into the new absolute direction.
function turn(dir, action) {
  const idx = CW.indexOf(dir);
  if (action === RIGHT) return CW[(idx + 1) % 4];
  if (action === LEFT) return CW[(idx + 3) % 4];
  return dir; // STRAIGHT
}

function isCollision(state, pt) {
  if (pt.x < 0 || pt.x >= GRID || pt.y < 0 || pt.y >= GRID) return true;
  // self: skip head (index 0)
  for (let i = 1; i < state.snake.length; i++) {
    if (state.snake[i].x === pt.x && state.snake[i].y === pt.y) return true;
  }
  return false;
}

// Advance one tick. `action` is STRAIGHT | RIGHT | LEFT. Returns a NEW state
// object (so React re-renders); the input is never mutated.
export function step(prev, action) {
  if (prev.over) return prev;

  const dir = turn(prev.dir, action);
  const d = DELTAS[dir];
  const head = { x: prev.snake[0].x + d.x, y: prev.snake[0].y + d.y };

  const willEat = prev.foods.some((f) => f.x === head.x && f.y === head.y);

  // Insert head; pop tail unless we just ate (tail vacates on the same step —
  // matches snake_game.py so moving into the current tail cell is safe).
  let snake = [head, ...prev.snake];
  if (!willEat) snake = snake.slice(0, -1);

  const state = {
    ...prev,
    snake,
    dir,
    foods: prev.foods,
    frame: prev.frame + 1,
  };

  // Collision (wall/self) or starvation timeout.
  if (isCollision(state, head) || state.frame > 100 * snake.length) {
    state.over = true;
    return state;
  }

  if (willEat) {
    state.score = prev.score + 1;
    state.foods = prev.foods.filter((f) => !(f.x === head.x && f.y === head.y));
    if (snake.length >= GRID * GRID) {
      state.over = true; // board full — win
      return state;
    }
    placeFood(state, 1);
    state.frame = 0;
  }

  return state;
}

// Translate an absolute desired direction (from a key press) into a relative
// action, ignoring reversals (can't turn 180°). Returns null if it's a no-op
// reversal so the caller can keep going straight.
export function actionForDesiredDir(curDir, desiredDir) {
  const cur = CW.indexOf(curDir);
  const des = CW.indexOf(desiredDir);
  if (des === cur) return STRAIGHT;
  if (des === (cur + 1) % 4) return RIGHT;
  if (des === (cur + 3) % 4) return LEFT;
  return null; // reversal
}
