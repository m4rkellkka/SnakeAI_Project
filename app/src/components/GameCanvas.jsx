import { useRef, useEffect } from "react";
import { GRID } from "../game/engine";

// Neon palette — ported from src/snake_game.py so the in-app canvas matches the
// Pygame look (then refined: smooth inter-tick interpolation Pygame never had).
const BG = "#080a0e";
const GRID_LINE = "rgba(20,26,34,1)";
const GRID_DOT = "rgba(28,38,50,1)";
const GLOW_LINE = "0,150,130";
const GLOW_DOT = "0,220,190";
const HEAD = [90, 255, 175];
const BODY_START = [0, 235, 145];
const BODY_END = [0, 95, 115];
const EDGE = "rgba(170,255,215,";
const EYE_WHITE = "#f5fffa";
const EYE_PUPIL = "#08100e";
const FOOD = "#ff375a";
const FOOD_CORE = "#ff8ca5";
const FOOD_SHINE = "#ffd7e1";
const FOOD_GLOW = "255,40,80";
const HUD_ACCENT = "rgba(60,255,185,";
const PARTICLE_COLORS = [
  [255, 90, 120], [255, 165, 90], [120, 255, 195], [255, 235, 130],
];

const lerp = (a, b, t) => a + (b - a) * t;
const lerpColor = (c1, c2, t) => [
  Math.round(lerp(c1[0], c2[0], t)),
  Math.round(lerp(c1[1], c2[1], t)),
  Math.round(lerp(c1[2], c2[2], t)),
];
const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

const DIR_VEC = {
  RIGHT: [1, 0], LEFT: [-1, 0], UP: [0, -1], DOWN: [0, 1],
};

/**
 * Pure renderer. Draws whatever `frame` it's given onto a canvas, interpolating
 * smoothly between successive frames over `tickMs` so the snake slides instead
 * of jumping cell-to-cell. `frame` = { snake:[{x,y}], foods:[{x,y}], dir, score,
 * over }. Works for both manual play and (later) AI-streamed frames.
 */
export default function GameCanvas({ frame, tickMs = 120, size = 560 }) {
  const canvasRef = useRef(null);
  const curRef = useRef(frame);
  const prevRef = useRef(frame);
  const arrivalRef = useRef(performance.now());
  const particlesRef = useRef([]);
  const animTickRef = useRef(0);
  const rafRef = useRef(0);
  const tickMsRef = useRef(tickMs);
  const lastScoreRef = useRef(frame?.score ?? 0);

  tickMsRef.current = tickMs;

  // When a new logical frame arrives, the previously displayed frame becomes the
  // interpolation source. Detect score bumps to fire an eat-burst at the head.
  useEffect(() => {
    if (!frame) return;
    prevRef.current = curRef.current || frame;
    curRef.current = frame;
    arrivalRef.current = performance.now();

    if (frame.score > lastScoreRef.current) {
      spawnBurst(particlesRef.current, frame.snake[0]);
      lastScoreRef.current = frame.score;
    } else if (frame.score < lastScoreRef.current) {
      // reset happened — clear stale particles & snap (no slide across the board)
      lastScoreRef.current = frame.score;
      particlesRef.current = [];
      prevRef.current = frame;
    }
  }, [frame]);

  // Single rAF loop, set up once per canvas size. Reads the latest frame via
  // refs so it never needs to tear down on every tick.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cell = size / GRID;

    const draw = () => {
      animTickRef.current += 1;
      const now = performance.now();
      const cur = curRef.current;
      const prev = prevRef.current || cur;
      const rawT = Math.max(0, Math.min(1, (now - arrivalRef.current) / tickMsRef.current));
      const t = rawT * (2 - rawT);

      drawBackground(ctx, size, cell, animTickRef.current);
      if (cur) {
        drawSnake(ctx, cell, prev, cur, t);
        drawFood(ctx, cell, cur.foods, animTickRef.current);
      }
      drawParticles(ctx, particlesRef.current, cell);

      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [size]);

  return <canvas ref={canvasRef} className="game-canvas" />;
}

function drawBackground(ctx, size, cell, tick) {
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, size, size);

  // dim static grid
  ctx.strokeStyle = GRID_LINE;
  ctx.lineWidth = 1;
  for (let i = 0; i <= GRID; i++) {
    const p = i * cell;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(size, p); ctx.stroke();
  }
  // breathing neon overlay
  const breath = Math.sin(tick * 0.035) * 0.5 + 0.5;
  const a = (14 + 26 * breath) / 255;
  ctx.strokeStyle = `rgba(${GLOW_LINE},${a})`;
  for (let i = 0; i <= GRID; i++) {
    const p = i * cell;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(size, p); ctx.stroke();
  }
  // intersection dots
  ctx.fillStyle = GRID_DOT;
  for (let x = 0; x <= GRID; x++) {
    for (let y = 0; y <= GRID; y++) {
      ctx.beginPath(); ctx.arc(x * cell, y * cell, 1.2, 0, Math.PI * 2); ctx.fill();
    }
  }
  ctx.fillStyle = `rgba(${GLOW_DOT},${a * 1.2})`;
  for (let x = 0; x <= GRID; x++) {
    for (let y = 0; y <= GRID; y++) {
      ctx.beginPath(); ctx.arc(x * cell, y * cell, 2, 0, Math.PI * 2); ctx.fill();
    }
  }
}

function segPos(prev, cur, i, t, cell) {
  const c = cur.snake[i];
  const pSnake = prev.snake || cur.snake;
  const p = i < pSnake.length ? pSnake[i] : pSnake[pSnake.length - 1];
  const x = lerp(p.x, c.x, t) * cell + cell / 2;
  const y = lerp(p.y, c.y, t) * cell + cell / 2;
  return [x, y];
}

const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

function drawSnake(ctx, cell, prev, cur, t) {
  const n = cur.snake.length;
  if (n === 0) return;
  const c = [];
  for (let i = 0; i < n; i++) c.push(segPos(prev, cur, i, t, cell));

  const headR = cell * 0.4;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (n === 1) {
    ctx.beginPath();
    ctx.arc(c[0][0], c[0][1], headR + 2, 0, Math.PI * 2);
    ctx.fillStyle = EDGE + "1)";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(c[0][0], c[0][1], headR, 0, Math.PI * 2);
    ctx.fillStyle = rgb(BODY_START);
    ctx.fill();
  } else {
    const traceBody = (endIdx) => {
      ctx.beginPath();
      ctx.moveTo(c[0][0], c[0][1]);
      for (let i = 0; i < endIdx && i < n - 1; i++) {
        const ep = i === n - 2 ? c[n - 1] : mid(c[i], c[i + 1]);
        ctx.quadraticCurveTo(c[i][0], c[i][1], ep[0], ep[1]);
      }
    };

    // Rim — single continuous path
    traceBody(n);
    ctx.strokeStyle = EDGE + "1)";
    ctx.lineWidth = 2 * headR + 4;
    ctx.stroke();

    // Body — 3 layered passes: tail (dark, narrow) → mid → head (bright, wide).
    // Each shorter pass paints over the previous, creating gradient + taper.
    const midColor = lerpColor(BODY_START, BODY_END, 0.5);
    const layers = [
      { extent: n,                                    color: BODY_END,  w: headR * 0.72 },
      { extent: Math.max(2, Math.floor(n * 0.65)),    color: midColor,  w: headR * 0.88 },
      { extent: Math.max(2, Math.floor(n * 0.3)),     color: BODY_START, w: headR },
    ];
    for (const { extent, color, w } of layers) {
      traceBody(extent);
      ctx.strokeStyle = rgb(color);
      ctx.lineWidth = Math.max(2, 2 * w);
      ctx.stroke();
    }
  }

  // Head cap + eyes
  ctx.beginPath();
  ctx.arc(c[0][0], c[0][1], headR, 0, Math.PI * 2);
  ctx.fillStyle = rgb(HEAD);
  ctx.fill();
  drawEyes(ctx, c[0], cur.dir, cell);
}

function drawEyes(ctx, center, dir, cell) {
  const [cx, cy] = center;
  const off = cell / 5;
  const eyeR = cell / 7;
  const pupilR = Math.max(2, eyeR / 2);
  const [dx, dy] = DIR_VEC[dir] || [1, 0];
  // two eyes perpendicular to heading
  const perp = [-dy, dx];
  const eyes = [
    [cx + dx * off + perp[0] * off, cy + dy * off + perp[1] * off],
    [cx + dx * off - perp[0] * off, cy + dy * off - perp[1] * off],
  ];
  for (const [ex, ey] of eyes) {
    ctx.beginPath(); ctx.arc(ex, ey, eyeR, 0, Math.PI * 2);
    ctx.fillStyle = EYE_WHITE; ctx.fill();
    ctx.beginPath(); ctx.arc(ex + dx * 2, ey + dy * 2, pupilR, 0, Math.PI * 2);
    ctx.fillStyle = EYE_PUPIL; ctx.fill();
  }
}

function drawFood(ctx, cell, foods, tick) {
  if (!foods) return;
  const pulse = Math.sin(tick * 0.09) * 0.12 + 0.88;
  for (const f of foods) {
    const cx = f.x * cell + cell / 2;
    const cy = f.y * cell + cell / 2;
    const r = Math.max(3, cell * 0.3 * pulse);
    // soft glow
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, cell * 0.7);
    grad.addColorStop(0, `rgba(${FOOD_GLOW},0.45)`);
    grad.addColorStop(1, `rgba(${FOOD_GLOW},0)`);
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(cx, cy, cell * 0.7, 0, Math.PI * 2); ctx.fill();
    // gem
    ctx.fillStyle = FOOD;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = FOOD_CORE;
    ctx.beginPath(); ctx.arc(cx, cy, Math.max(1, r - 5), 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = FOOD_SHINE;
    ctx.beginPath(); ctx.arc(cx - r / 3, cy - r / 3, Math.max(2, r / 4), 0, Math.PI * 2); ctx.fill();
  }
}

function spawnBurst(particles, head) {
  for (let i = 0; i < 14; i++) {
    const ang = Math.random() * Math.PI * 2;
    const spd = 1.5 + Math.random() * 3;
    particles.push({
      x: head.x, y: head.y, // grid coords; converted at draw time
      vx: Math.cos(ang) * spd, vy: Math.sin(ang) * spd,
      life: 1, r: 2 + Math.random() * 2,
      color: PARTICLE_COLORS[Math.floor(Math.random() * PARTICLE_COLORS.length)],
    });
  }
  particles.push({ ring: true, x: head.x, y: head.y, life: 1 });
  if (particles.length > 200) particles.splice(0, particles.length - 200);
}

function drawParticles(ctx, particles, cell) {
  const toPx = (gx) => gx * cell + cell / 2;
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.life -= 0.045;
    if (p.life <= 0) { particles.splice(i, 1); continue; }
    if (p.ring) {
      const rr = cell * 0.25 + (1 - p.life) * cell * 1.1;
      ctx.strokeStyle = HUD_ACCENT + (0.67 * p.life) + ")";
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(toPx(p.x), toPx(p.y), rr, 0, Math.PI * 2); ctx.stroke();
    } else {
      p.x += p.vx / cell; p.y += p.vy / cell;
      p.vy += 0.12; p.vx *= 0.95; p.vy *= 0.95;
      ctx.fillStyle = `rgba(${p.color[0]},${p.color[1]},${p.color[2]},${p.life})`;
      ctx.beginPath(); ctx.arc(toPx(p.x), toPx(p.y), Math.max(1, p.r), 0, Math.PI * 2); ctx.fill();
    }
  }
}
