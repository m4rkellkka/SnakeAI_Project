# Changelog

## [1.0.0] — 2026-06-16

### Added
- **Tauri v2 desktop app** — dark-theme GUI with Train, Watch AI, Benchmark, and Models panels
- **Live training charts** — score per game and honest eval avg/max via Recharts
- **First-run setup screen** — auto-detects Python and missing packages, installs deps with one click
- **Honest evaluation loop** — separate eval (no teacher, no unstick), stuck-rate tracking, best checkpoint preserved
- **Hamiltonian-cycle teacher** with corner-cutting shortcuts — achieves 253/253 (100%) provably collision-free
- **DAgger-lite training** — network sees its own distribution (0 → 70% of steps) over 100k ramp steps
- **Curriculum starts** — 20% of episodes start with snakes of length 4–50 after game 150
- **Named hyperparameter runs** — `--run-name`, `--lr`, `--dagger-prob-max`, `--curriculum-prob` CLI flags; each run isolated under `model/<name>/`
- **Benchmark harness** (`tools/benchmark.py`) — full score distributions, seed-locked head-to-head checkpoint comparison
- **Pretrained model** (`model/pretrained.pth`) — best honest eval 234.9 / 253 (93%), included in repo
- **Demo GIFs** — trained vs untrained side-by-side (`assets/demo_pretrained.gif`, `assets/demo_untrained.gif`)
- **Visit-history state channel** — binary map of head positions since last apple; helps the network detect its own loops
- **In-training loop guard** — prevents replay buffer from flooding with identical loop states during DAgger steps
- **Inference-time unstick crutch** — hands loop steps to the teacher during watch/benchmark; honest eval always runs without it
- **10-channel egocentric + compass state** — channels 0–5 rotated so head faces "up"; channels 6–9 absolute direction one-hot
- **Dual-language README** — English + Russian, Mermaid pipeline diagram, full usage guide
- **Custom app icon** — snake-themed icon for macOS, Windows, and Linux builds
- **Legacy Tkinter launcher** (`launcher.py`) — still works, no extra dependencies
