# Product

## Register

product

## Users

A single technical user — the developer/researcher who owns this repo — running a local
desktop control panel for a Snake-playing neural network. Context: a focused, often
long-running session at a desk, watching training metrics tick by, launching eval/watch
windows, and comparing model runs. They are fluent with terminals and CLIs; the GUI exists
to make the repeated commands faster, more legible, and more pleasant than retyping
`python3 src/train_ai.py …` by hand.

## Product Purpose

A Tauri desktop GUI that wraps the project's Python training/eval scripts: start and stop
training, watch live charts (honest eval, loss, score, reward), launch Pygame watch/teacher/
manual windows, run seed-locked benchmarks, and manage named model runs with their
hyperparameters. Success = the user can run the whole train→watch→benchmark→compare loop
without touching the terminal, and can tell at a glance what is running and how it's going.

## Brand Personality

Confident, precise, alive. Three words: **neon, instrument, focused**. It should feel like a
high-end telemetry console — dark, glassy, with a single electric accent — that reacts to
live data (numbers that count up, status that pulses) without ever getting in the way of the
task. Voice in UI copy is direct and technical, not cute.

## Anti-references

- Generic SaaS-cream dashboards with pastel cards and a hero-metric template.
- Bootstrap/Material default-component look. No mismatched form controls.
- Toy/gamified ML UIs that bury the real numbers under decoration.
- Anything that animates for its own sake and makes the user wait for choreography.

## Design Principles

1. **The numbers are the product.** Charts and metrics get the space and the contrast; chrome recedes.
2. **Always answer "what's running?"** Process state is visible and controllable from anywhere.
3. **Earned familiarity.** Standard affordances (nav, tabs, selects, command patterns) behave exactly as expected.
4. **Alive, not busy.** Motion conveys state change (new best, count-up, running pulse) — 150–250ms, never decorative loops.
5. **One electric accent.** A single committed accent color carries primary actions, selection, and live state; the user may recolor it.

## Accessibility & Inclusion

Dark theme is the product default (long telemetry sessions, low ambient light). Body text and
metrics must clear WCAG AA (≥4.5:1) against the dark surfaces. Honor `prefers-reduced-motion`
and offer a manual reduced-motion toggle (count-ups, pulses, and entrances all degrade to
instant/crossfade). Live state must never be conveyed by color alone — pair it with text/icon.
