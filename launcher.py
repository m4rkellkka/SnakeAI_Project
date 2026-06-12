import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys
import threading
import time
import glob
from PIL import Image, ImageTk
DIR = os.path.dirname(os.path.abspath(__file__))

# Core scripts live in src/
TRAIN_AI = os.path.join("src", "train_ai.py")
TEACHER = os.path.join("src", "teacher.py")
PLAY_MANUAL = os.path.join("src", "play_manual.py")

# ──────────────────────────────────────────────────
# Color palette (premium dark theme)
# ──────────────────────────────────────────────────
BG_DARK = "#0d0d0d"
BG_CARD = "#161618"
BG_HOVER = "#1e1e22"
BG_BUTTON = "#1c1c20"
BG_BUTTON_HOVER = "#28282e"
BORDER_COLOR = "#2a2a30"
TEXT_PRIMARY = "#e8e8e8"
TEXT_SECONDARY = "#888890"
TEXT_MUTED = "#555560"
ACCENT_GREEN = "#00e878"
ACCENT_GREEN_DIM = "#00b85c"
ACCENT_CYAN = "#00d4aa"
ACCENT_BLUE = "#4488ff"
ACCENT_PURPLE = "#aa66ff"
ACCENT_RED = "#ff4466"
ACCENT_ORANGE = "#ff8844"
LOG_BG = "#111114"
LOG_FG = "#c8c8d0"


# ──────────────────────────────────────────────────
# Training Dashboard (Toplevel window)
# ──────────────────────────────────────────────────
class TrainingDashboard(tk.Toplevel):
    # ── Stat card configuration ──
    STAT_CONFIG = {
        "Game":      {"icon": "🎮", "color": ACCENT_BLUE,   "desc": "Episodes played"},
        "Score":     {"icon": "🍎", "color": ACCENT_GREEN,  "desc": "Last game score"},
        "Loss":      {"icon": "📉", "color": ACCENT_PURPLE, "desc": "Cross-entropy loss"},
        "Steps":     {"icon": "👣", "color": ACCENT_CYAN,   "desc": "Total env steps"},
        "Best Eval": {"icon": "🏆", "color": ACCENT_ORANGE, "desc": "Best honest avg"},
    }

    def __init__(self, master):
        super().__init__(master)
        self.title("Snake AI — Training Dashboard")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.geometry("1100x1020")
        self.minsize(850, 750)

        self._process = None
        self._running = False
        self._log_lines = []
        self._log_line_count = 0
        self._start_time = None
        self._elapsed_id = None
        self._pulse_id = None
        self._pulse_on = True
        self._progress_id = None
        self._progress_pos = 0

        self._build_ui()

    # ──────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────
    def _build_ui(self):
        # ── Header bar ──
        header = tk.Frame(self, bg="#101014")
        header.pack(fill=tk.X)

        header_inner = tk.Frame(header, bg="#101014")
        header_inner.pack(fill=tk.X, padx=24, pady=(14, 10))

        # Left: icon + title
        left_hdr = tk.Frame(header_inner, bg="#101014")
        left_hdr.pack(side=tk.LEFT)

        tk.Label(left_hdr, text="⚡", font=("", 20), bg="#101014", fg=ACCENT_GREEN
                 ).pack(side=tk.LEFT, padx=(0, 10))

        title_block = tk.Frame(left_hdr, bg="#101014")
        title_block.pack(side=tk.LEFT)
        tk.Label(title_block, text="TRAINING DASHBOARD",
                 font=("Helvetica Neue", 15, "bold"),
                 bg="#101014", fg=TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Label(title_block, text="Snake AI · Reinforcement Learning",
                 font=("Helvetica Neue", 10),
                 bg="#101014", fg=TEXT_MUTED).pack(anchor=tk.W)

        # Right: elapsed + status
        right_hdr = tk.Frame(header_inner, bg="#101014")
        right_hdr.pack(side=tk.RIGHT)

        self._elapsed_label = tk.Label(right_hdr, text="",
                                        font=("SF Mono", 11), bg="#101014", fg=TEXT_MUTED)
        self._elapsed_label.pack(side=tk.RIGHT, padx=(12, 0))

        self._status_dot = tk.Canvas(right_hdr, width=12, height=12,
                                      bg="#101014", highlightthickness=0)
        self._status_dot.pack(side=tk.RIGHT, padx=(0, 6))
        self._status_dot.create_oval(2, 2, 10, 10, fill=TEXT_MUTED,
                                      outline="", tags="dot")

        self._status_label = tk.Label(right_hdr, text="Ready",
                                       font=("Helvetica Neue", 11), bg="#101014", fg=TEXT_SECONDARY)
        self._status_label.pack(side=tk.RIGHT)

        # ── Animated progress bar ──
        self._progress_canvas = tk.Canvas(self, height=2, bg=BG_DARK,
                                           highlightthickness=0)
        self._progress_canvas.pack(fill=tk.X)

        # ── Stats cards ──
        stats_outer = tk.Frame(self, bg=BG_DARK)
        stats_outer.pack(fill=tk.X, padx=24, pady=(12, 4))

        self._stats = {}
        self._stat_frames = {}
        for i, (label_text, cfg) in enumerate(self.STAT_CONFIG.items()):
            card = tk.Frame(stats_outer, bg=BG_CARD,
                            highlightbackground="#222228", highlightthickness=1)
            card.pack(side=tk.LEFT, expand=True, fill=tk.X,
                      padx=(0 if i == 0 else 4, 0 if i == len(self.STAT_CONFIG) - 1 else 4),
                      ipady=6)

            inner = tk.Frame(card, bg=BG_CARD)
            inner.pack(padx=14, pady=(10, 8))

            # Header row: icon + label
            hdr_row = tk.Frame(inner, bg=BG_CARD)
            hdr_row.pack(anchor=tk.W)
            tk.Label(hdr_row, text=cfg["icon"], font=("", 11),
                     bg=BG_CARD, fg=cfg["color"]).pack(side=tk.LEFT, padx=(0, 5))
            tk.Label(hdr_row, text=label_text.upper(),
                     font=("Helvetica Neue", 9, "bold"),
                     bg=BG_CARD, fg=cfg["color"]).pack(side=tk.LEFT)

            # Value
            val = tk.Label(inner, text="—",
                           font=("Helvetica Neue", 22, "bold"),
                           bg=BG_CARD, fg=TEXT_PRIMARY)
            val.pack(anchor=tk.W, pady=(2, 0))

            # Description
            tk.Label(inner, text=cfg["desc"],
                     font=("Helvetica Neue", 9), bg=BG_CARD, fg=TEXT_MUTED
                     ).pack(anchor=tk.W)

            # Colored bottom accent line
            accent_line = tk.Frame(card, bg=cfg["color"], height=2)
            accent_line.pack(fill=tk.X, side=tk.BOTTOM)

            self._stats[label_text] = val
            self._stat_frames[label_text] = card

        # ── Bottom action bar (packed first to bottom so it's always visible) ──
        bottom = tk.Frame(self, bg=BG_DARK)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=(10, 18))

        self._action_btn = tk.Label(
            bottom, text="▶  START TRAINING",
            font=("Helvetica Neue", 13, "bold"),
            bg=ACCENT_GREEN, fg="#000000", cursor="hand2",
            padx=24, pady=12
        )
        self._action_btn.pack(fill=tk.X)
        self._action_btn.bind("<Button-1>", lambda e: self._on_action_click())
        self._action_btn.bind("<Enter>", lambda e: self._action_btn.configure(
            bg=ACCENT_GREEN_DIM if not self._running else "#ff2244"))
        self._action_btn.bind("<Leave>", lambda e: self._action_btn.configure(
            bg=ACCENT_GREEN if not self._running else ACCENT_RED))

        # ── Log area (packed to bottom, above action bar) ──
        log_section = tk.Frame(self, bg=BG_DARK, height=180)
        log_section.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=(8, 4))
        log_section.pack_propagate(False)

        log_header = tk.Frame(log_section, bg=BG_DARK)
        log_header.pack(fill=tk.X, pady=(0, 4))

        tk.Label(log_header, text="📋  TRAINING LOG",
                 font=("Helvetica Neue", 10, "bold"),
                 bg=BG_DARK, fg=TEXT_MUTED).pack(side=tk.LEFT)

        self._log_count_label = tk.Label(log_header, text="0 lines",
                                          font=("Helvetica Neue", 9),
                                          bg=BG_DARK, fg=TEXT_MUTED)
        self._log_count_label.pack(side=tk.RIGHT)

        self._clear_log_btn = tk.Label(log_header, text="✕ Clear",
                                        font=("Helvetica Neue", 9),
                                        bg=BG_DARK, fg=TEXT_MUTED, cursor="hand2")
        self._clear_log_btn.pack(side=tk.RIGHT, padx=(0, 10))
        self._clear_log_btn.bind("<Button-1>", lambda e: self._clear_log())
        self._clear_log_btn.bind("<Enter>",
                                  lambda e: self._clear_log_btn.configure(fg=ACCENT_RED))
        self._clear_log_btn.bind("<Leave>",
                                  lambda e: self._clear_log_btn.configure(fg=TEXT_MUTED))

        log_container = tk.Frame(log_section, bg=LOG_BG,
                                  highlightbackground=BORDER_COLOR, highlightthickness=1)
        log_container.pack(fill=tk.BOTH, expand=True)

        self._log_scrollbar = tk.Scrollbar(log_container, orient=tk.VERTICAL,
                                            troughcolor=LOG_BG, bg="#222228")
        self._log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_text = tk.Text(log_container, bg=LOG_BG, fg=LOG_FG,
                                  font=("SF Mono", 11),
                                  insertbackground=ACCENT_GREEN,
                                  selectbackground="#334455",
                                  relief=tk.FLAT, padx=12, pady=8,
                                  wrap=tk.WORD,
                                  yscrollcommand=self._log_scrollbar.set,
                                  state=tk.DISABLED)
        self._log_text.pack(fill=tk.BOTH, expand=True)
        self._log_scrollbar.config(command=self._log_text.yview)

        # Log tag colors (expanded set)
        self._log_text.tag_configure("eval", foreground=ACCENT_CYAN)
        self._log_text.tag_configure("best", foreground=ACCENT_GREEN,
                                      font=("SF Mono", 11, "bold"))
        self._log_text.tag_configure("game", foreground=LOG_FG)
        self._log_text.tag_configure("info", foreground=TEXT_SECONDARY)
        self._log_text.tag_configure("dagger", foreground=ACCENT_PURPLE)
        self._log_text.tag_configure("curriculum", foreground=ACCENT_ORANGE)
        self._log_text.tag_configure("save", foreground=ACCENT_BLUE)

        # ── Chart area (takes up remaining space in the middle) ──
        chart_section = tk.Frame(self, bg=BG_DARK)
        chart_section.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=24, pady=(8, 4))
        
        chart_header = tk.Frame(chart_section, bg=BG_DARK)
        chart_header.pack(fill=tk.X, pady=(0, 4))

        tk.Label(chart_header, text="📊  LEARNING CURVE",
                 font=("Helvetica Neue", 10, "bold"),
                 bg=BG_DARK, fg=TEXT_MUTED).pack(side=tk.LEFT)

        self._chart_time_label = tk.Label(chart_header, text="",
                                           font=("Helvetica Neue", 9),
                                           bg=BG_DARK, fg=TEXT_MUTED)
        self._chart_time_label.pack(side=tk.RIGHT)

        self._chart_refresh_btn = tk.Label(chart_header, text="⟳ Refresh",
                                            font=("Helvetica Neue", 9),
                                            bg=BG_DARK, fg=ACCENT_CYAN, cursor="hand2")
        self._chart_refresh_btn.pack(side=tk.RIGHT, padx=(0, 10))
        self._chart_refresh_btn.bind("<Button-1>", lambda e: self._refresh_chart())
        self._chart_refresh_btn.bind("<Enter>",
                                      lambda e: self._chart_refresh_btn.configure(fg=ACCENT_GREEN))
        self._chart_refresh_btn.bind("<Leave>",
                                      lambda e: self._chart_refresh_btn.configure(fg=ACCENT_CYAN))

        self._chart_container = tk.Frame(chart_section, bg=BG_CARD,
                                          highlightbackground=BORDER_COLOR,
                                          highlightthickness=1)
        self._chart_container.pack(fill=tk.BOTH, expand=True)

        self._chart_label = tk.Label(self._chart_container,
                                      text="No data yet — start training to see the learning curve",
                                      font=("Helvetica Neue", 11), bg=BG_CARD, fg=TEXT_MUTED)
        self._chart_label.pack(fill=tk.BOTH, expand=True)
        self._chart_image = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────
    # Animations
    # ──────────────────────────────────────────────
    def _start_pulse(self):
        """Pulsing green dot while training."""
        if not self._running:
            return
        self._pulse_on = not self._pulse_on
        color = ACCENT_GREEN if self._pulse_on else "#005530"
        self._status_dot.itemconfig("dot", fill=color)
        self._pulse_id = self.after(600, self._start_pulse)

    def _stop_pulse(self, color=TEXT_MUTED):
        if self._pulse_id:
            self.after_cancel(self._pulse_id)
            self._pulse_id = None
        self._status_dot.itemconfig("dot", fill=color)

    def _start_progress_anim(self):
        """Sliding cyan bar at the top during training."""
        if not self._running:
            return
        self._progress_canvas.delete("bar")
        w = self._progress_canvas.winfo_width()
        if w < 10:
            w = 1100
        bar_len = w // 4
        x = self._progress_pos % (w + bar_len) - bar_len
        self._progress_canvas.create_rectangle(x, 0, x + bar_len, 2,
                                                fill=ACCENT_CYAN, outline="",
                                                tags="bar")
        self._progress_pos += 4
        self._progress_id = self.after(30, self._start_progress_anim)

    def _stop_progress_anim(self):
        if self._progress_id:
            self.after_cancel(self._progress_id)
            self._progress_id = None
        try:
            if self._progress_canvas.winfo_exists():
                self._progress_canvas.delete("bar")
        except tk.TclError:
            pass

    def _update_elapsed(self):
        if not self._running or not self._start_time:
            return
        elapsed = int(time.time() - self._start_time)
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        if hrs > 0:
            txt = f"⏱ {hrs}h {mins:02d}m {secs:02d}s"
        else:
            txt = f"⏱ {mins}m {secs:02d}s"
        self._elapsed_label.configure(text=txt)
        self._elapsed_id = self.after(1000, self._update_elapsed)

    def _stop_elapsed(self):
        if self._elapsed_id:
            self.after_cancel(self._elapsed_id)
            self._elapsed_id = None

    # ──────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────
    def _on_action_click(self):
        if not self._running:
            self._start_training()
        else:
            self._stop_training()

    def _start_training(self):
        self._running = True
        self._start_time = time.time()
        self._status_label.configure(text="Training", fg=ACCENT_GREEN)
        self._action_btn.configure(text="■  STOP TRAINING", bg=ACCENT_RED, fg="white")

        # Start animations
        self._start_pulse()
        self._start_progress_anim()
        self._update_elapsed()

        self._process = subprocess.Popen(
            [sys.executable, "-u", TRAIN_AI, "--headless"],
            cwd=DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        # Reader thread
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        # Chart refresh timer
        self._schedule_chart_refresh()

    def _read_output(self):
        try:
            for line in self._process.stdout:
                line = line.rstrip('\n')
                if line:
                    self.after(0, self._append_log, line)
                    self.after(0, self._parse_stats, line)
        except Exception:
            pass
        finally:
            self.after(0, self._on_process_ended)

    def _append_log(self, line):
        self._log_text.configure(state=tk.NORMAL)

        # Determine tag with expanded categories
        tag = "game"
        if "Honest eval" in line or ">> " in line:
            tag = "eval"
        if "New best" in line:
            tag = "best"
        elif "checkpoint" in line.lower() or "Saving" in line:
            tag = "save"
        if "Training started" in line or "Resuming" in line:
            tag = "info"
        if "[D]" in line or "[C,D]" in line:
            tag = "dagger"
        if "[C]" in line or "[C," in line:
            tag = "curriculum"

        self._log_text.insert(tk.END, line + "\n", tag)
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

        self._log_line_count += 1
        self._log_count_label.configure(text=f"{self._log_line_count} lines")

    def _clear_log(self):
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.configure(state=tk.DISABLED)
        self._log_line_count = 0
        self._log_count_label.configure(text="0 lines")

    def _parse_stats(self, line):
        """Parse training log lines to update the stats display."""
        try:
            if line.startswith("Game:"):
                parts = line.split("|")
                for part in parts:
                    part = part.strip()
                    if part.startswith("Game:"):
                        # Extract game number (e.g. "Game: 42 [D]" -> "42")
                        game_str = part.replace("Game:", "").strip()
                        game_num = game_str.split()[0]
                        self._stats["Game"].configure(text=game_num)
                    elif part.startswith("Score:"):
                        self._stats["Score"].configure(
                            text=part.replace("Score:", "").strip())
                    elif part.startswith("Loss:"):
                        self._stats["Loss"].configure(
                            text=part.replace("Loss:", "").strip())
                    elif part.startswith("Steps:"):
                        steps = int(part.replace("Steps:", "").strip())
                        if steps >= 1_000_000:
                            self._stats["Steps"].configure(
                                text=f"{steps/1_000_000:.1f}M")
                        elif steps >= 1000:
                            self._stats["Steps"].configure(
                                text=f"{steps/1000:.0f}K")
                        else:
                            self._stats["Steps"].configure(text=str(steps))

            elif "Honest eval:" in line:
                # "  >> Honest eval: avg=8.5, max=42 (15 games)"
                if "avg=" in line:
                    avg_part = line.split("avg=")[1].split(",")[0]
                    self._stats["Best Eval"].configure(text=avg_part)

            elif "New best honest eval:" in line:
                # "  >> New best honest eval: 8.5 -> checkpoint_best.pth"
                val = line.split("New best honest eval:")[1].split("->")[0].strip()
                self._stats["Best Eval"].configure(text=f"★ {val}", fg=ACCENT_GREEN)

        except (IndexError, ValueError):
            pass

    def _schedule_chart_refresh(self):
        if not self._running:
            return
        self._refresh_chart()
        self.after(15000, self._schedule_chart_refresh)  # Every 15 seconds

    def _refresh_chart(self):
        plot_path = os.path.join(DIR, "learning_curve.png")
        if not os.path.exists(plot_path):
            return
        try:
            # Use PIL for precise high-quality resizing to avoid cutoff
            pil_img = Image.open(plot_path)
            
            # Container dimensions (with a small margin to be safe)
            container_w = max(600, self._chart_container.winfo_width() - 8)
            container_h = max(300, self._chart_container.winfo_height() - 8)
            
            # Calculate aspect ratio preserving size
            img_ratio = pil_img.width / pil_img.height
            container_ratio = container_w / container_h
            
            if img_ratio > container_ratio:
                # Constrain by width
                new_w = container_w
                new_h = int(container_w / img_ratio)
            else:
                # Constrain by height
                new_h = container_h
                new_w = int(container_h * img_ratio)
                
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(pil_img)

            self._chart_image = img  # Keep reference
            self._chart_label.configure(image=img, text="")

            # Update timestamp
            mod_time = os.path.getmtime(plot_path)
            t = time.strftime("%H:%M:%S", time.localtime(mod_time))
            self._chart_time_label.configure(text=f"Last update: {t}")
        except Exception:
            pass

    def _stop_training(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._running = False
            self._status_label.configure(text="Stopped", fg=ACCENT_RED)
            self._action_btn.configure(text="▶  START TRAINING", bg=ACCENT_GREEN, fg="#000000")
            self._stop_pulse(ACCENT_RED)
            self._stop_progress_anim()
            self._stop_elapsed()

    def _on_process_ended(self):
        self._running = False
        self._stop_progress_anim()
        self._stop_elapsed()
        try:
            if self.winfo_exists() and self._status_label.winfo_exists():
                if self._status_label.cget("text") != "Stopped":
                    self._status_label.configure(text="Finished", fg=ACCENT_ORANGE)
                    self._stop_pulse(ACCENT_ORANGE)
        except tk.TclError:
            pass

    def _on_close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._running = False
        self._stop_pulse()
        self._stop_progress_anim()
        self._stop_elapsed()
        self.destroy()


# ──────────────────────────────────────────────────
# Sweep Run Dashboard (Toplevel window)
# ──────────────────────────────────────────────────
class SweepRunDashboard(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Snake AI — New Sweep Run")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.geometry("1100x1020")
        self.minsize(850, 750)

        self._process = None
        self._running = False
        self._log_lines = []
        self._log_line_count = 0
        self._start_time = None
        self._elapsed_id = None
        self._pulse_id = None
        self._pulse_on = True
        self._progress_id = None
        self._progress_pos = 0

        self._build_ui()

    def _build_ui(self):
        # ── Header bar ──
        header = tk.Frame(self, bg="#101014")
        header.pack(fill=tk.X)

        header_inner = tk.Frame(header, bg="#101014")
        header_inner.pack(fill=tk.X, padx=24, pady=(14, 10))

        left_hdr = tk.Frame(header_inner, bg="#101014")
        left_hdr.pack(side=tk.LEFT)

        tk.Label(left_hdr, text="⚡", font=("", 20), bg="#101014", fg=ACCENT_CYAN).pack(side=tk.LEFT, padx=(0, 10))

        title_block = tk.Frame(left_hdr, bg="#101014")
        title_block.pack(side=tk.LEFT)
        tk.Label(title_block, text="SWEEP RUN DASHBOARD", font=("Helvetica Neue", 15, "bold"), bg="#101014", fg=TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Label(title_block, text="Run custom configurations", font=("Helvetica Neue", 10), bg="#101014", fg=TEXT_MUTED).pack(anchor=tk.W)

        right_hdr = tk.Frame(header_inner, bg="#101014")
        right_hdr.pack(side=tk.RIGHT)

        self._elapsed_label = tk.Label(right_hdr, text="", font=("SF Mono", 11), bg="#101014", fg=TEXT_MUTED)
        self._elapsed_label.pack(side=tk.RIGHT, padx=(12, 0))

        self._status_dot = tk.Canvas(right_hdr, width=12, height=12, bg="#101014", highlightthickness=0)
        self._status_dot.pack(side=tk.RIGHT, padx=(0, 6))
        self._status_dot.create_oval(2, 2, 10, 10, fill=TEXT_MUTED, outline="", tags="dot")

        self._status_label = tk.Label(right_hdr, text="Ready", font=("Helvetica Neue", 11), bg="#101014", fg=TEXT_SECONDARY)
        self._status_label.pack(side=tk.RIGHT)

        # ── Animated progress bar ──
        self._progress_canvas = tk.Canvas(self, height=2, bg=BG_DARK, highlightthickness=0)
        self._progress_canvas.pack(fill=tk.X)

        # ── Inputs ──
        inputs_outer = tk.Frame(self, bg=BG_DARK)
        inputs_outer.pack(fill=tk.X, padx=24, pady=(12, 4))

        inputs_outer.columnconfigure(0, weight=1)
        inputs_outer.columnconfigure(1, weight=1)

        self.entry_lr = self._make_input(inputs_outer, "Learning Rate (default 0.0005):", 0, 0)
        self.entry_dagger = self._make_input(inputs_outer, "DAgger Prob Max (default 0.7):", 0, 1)
        self.entry_curr = self._make_input(inputs_outer, "Curriculum Prob (default 0.2):", 1, 0)
        self.entry_name = self._make_input(inputs_outer, "Run Name (optional):", 1, 1)

        self._config_label = tk.Label(self, text="Config: —", font=("SF Mono", 10), bg=BG_DARK, fg=ACCENT_CYAN)
        self._config_label.pack(fill=tk.X, padx=24, pady=(4, 8), anchor=tk.W)

        # ── Bottom action bar ──
        bottom = tk.Frame(self, bg=BG_DARK)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=(10, 18))

        self._action_btn = tk.Label(
            bottom, text="▶  START SWEEP",
            font=("Helvetica Neue", 13, "bold"),
            bg=ACCENT_GREEN, fg="#000000", cursor="hand2",
            padx=24, pady=12
        )
        self._action_btn.pack(fill=tk.X)
        self._action_btn.bind("<Button-1>", lambda e: self._on_action_click())
        self._action_btn.bind("<Enter>", lambda e: self._action_btn.configure(
            bg=ACCENT_GREEN_DIM if not self._running else "#ff2244"))
        self._action_btn.bind("<Leave>", lambda e: self._action_btn.configure(
            bg=ACCENT_GREEN if not self._running else ACCENT_RED))

        # ── Log area ──
        log_section = tk.Frame(self, bg=BG_DARK, height=180)
        log_section.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=(8, 4))
        log_section.pack_propagate(False)

        log_header = tk.Frame(log_section, bg=BG_DARK)
        log_header.pack(fill=tk.X, pady=(0, 4))

        tk.Label(log_header, text="📋  TRAINING LOG", font=("Helvetica Neue", 10, "bold"), bg=BG_DARK, fg=TEXT_MUTED).pack(side=tk.LEFT)

        self._log_count_label = tk.Label(log_header, text="0 lines", font=("Helvetica Neue", 9), bg=BG_DARK, fg=TEXT_MUTED)
        self._log_count_label.pack(side=tk.RIGHT)

        self._clear_log_btn = tk.Label(log_header, text="✕ Clear", font=("Helvetica Neue", 9), bg=BG_DARK, fg=TEXT_MUTED, cursor="hand2")
        self._clear_log_btn.pack(side=tk.RIGHT, padx=(0, 10))
        self._clear_log_btn.bind("<Button-1>", lambda e: self._clear_log())
        self._clear_log_btn.bind("<Enter>", lambda e: self._clear_log_btn.configure(fg=ACCENT_RED))
        self._clear_log_btn.bind("<Leave>", lambda e: self._clear_log_btn.configure(fg=TEXT_MUTED))

        log_container = tk.Frame(log_section, bg=LOG_BG, highlightbackground=BORDER_COLOR, highlightthickness=1)
        log_container.pack(fill=tk.BOTH, expand=True)

        self._log_scrollbar = tk.Scrollbar(log_container, orient=tk.VERTICAL, troughcolor=LOG_BG, bg="#222228")
        self._log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_text = tk.Text(log_container, bg=LOG_BG, fg=LOG_FG, font=("SF Mono", 11),
                                  insertbackground=ACCENT_GREEN, selectbackground="#334455",
                                  relief=tk.FLAT, padx=12, pady=8, wrap=tk.WORD,
                                  yscrollcommand=self._log_scrollbar.set, state=tk.DISABLED)
        self._log_text.pack(fill=tk.BOTH, expand=True)
        self._log_scrollbar.config(command=self._log_text.yview)

        # Log tag colors
        self._log_text.tag_configure("eval", foreground=ACCENT_CYAN)
        self._log_text.tag_configure("best", foreground=ACCENT_GREEN, font=("SF Mono", 11, "bold"))
        self._log_text.tag_configure("game", foreground=LOG_FG)
        self._log_text.tag_configure("info", foreground=TEXT_SECONDARY)
        self._log_text.tag_configure("dagger", foreground=ACCENT_PURPLE)
        self._log_text.tag_configure("curriculum", foreground=ACCENT_ORANGE)
        self._log_text.tag_configure("save", foreground=ACCENT_BLUE)

        # ── Chart area ──
        chart_section = tk.Frame(self, bg=BG_DARK)
        chart_section.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=24, pady=(8, 4))
        
        chart_header = tk.Frame(chart_section, bg=BG_DARK)
        chart_header.pack(fill=tk.X, pady=(0, 4))

        tk.Label(chart_header, text="📊  LEARNING CURVE", font=("Helvetica Neue", 10, "bold"), bg=BG_DARK, fg=TEXT_MUTED).pack(side=tk.LEFT)

        self._chart_time_label = tk.Label(chart_header, text="", font=("Helvetica Neue", 9), bg=BG_DARK, fg=TEXT_MUTED)
        self._chart_time_label.pack(side=tk.RIGHT)

        self._chart_refresh_btn = tk.Label(chart_header, text="⟳ Refresh", font=("Helvetica Neue", 9), bg=BG_DARK, fg=ACCENT_CYAN, cursor="hand2")
        self._chart_refresh_btn.pack(side=tk.RIGHT, padx=(0, 10))
        self._chart_refresh_btn.bind("<Button-1>", lambda e: self._refresh_chart())
        self._chart_refresh_btn.bind("<Enter>", lambda e: self._chart_refresh_btn.configure(fg=ACCENT_GREEN))
        self._chart_refresh_btn.bind("<Leave>", lambda e: self._chart_refresh_btn.configure(fg=ACCENT_CYAN))

        self._chart_container = tk.Frame(chart_section, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        self._chart_container.pack(fill=tk.BOTH, expand=True)

        self._chart_label = tk.Label(self._chart_container, text="No data yet — start sweep to see the learning curve",
                                      font=("Helvetica Neue", 11), bg=BG_CARD, fg=TEXT_MUTED)
        self._chart_label.pack(fill=tk.BOTH, expand=True)
        self._chart_image = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _make_input(self, parent, label_text, row, col):
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.grid(row=row, column=col, sticky="ew", padx=10, pady=5)
        tk.Label(frame, text=label_text, bg=BG_DARK, fg=TEXT_MUTED, font=("Helvetica Neue", 10)).pack(anchor=tk.W)
        entry = tk.Entry(frame, bg=BG_CARD, fg=TEXT_PRIMARY, insertbackground=ACCENT_GREEN,
                         font=("SF Mono", 11), relief=tk.FLAT, highlightbackground=BORDER_COLOR, highlightthickness=1)
        entry.pack(fill=tk.X, pady=(2, 0), ipady=4)
        return entry

    def _start_pulse(self):
        if not self._running: return
        self._pulse_on = not self._pulse_on
        color = ACCENT_GREEN if self._pulse_on else "#005530"
        self._status_dot.itemconfig("dot", fill=color)
        self._pulse_id = self.after(600, self._start_pulse)

    def _stop_pulse(self, color=TEXT_MUTED):
        if self._pulse_id:
            self.after_cancel(self._pulse_id)
            self._pulse_id = None
        self._status_dot.itemconfig("dot", fill=color)

    def _start_progress_anim(self):
        if not self._running: return
        self._progress_canvas.delete("bar")
        w = self._progress_canvas.winfo_width()
        if w < 10: w = 1100
        bar_len = w // 4
        x = self._progress_pos % (w + bar_len) - bar_len
        self._progress_canvas.create_rectangle(x, 0, x + bar_len, 2, fill=ACCENT_CYAN, outline="", tags="bar")
        self._progress_pos += 4
        self._progress_id = self.after(30, self._start_progress_anim)

    def _stop_progress_anim(self):
        if self._progress_id:
            self.after_cancel(self._progress_id)
            self._progress_id = None
        try:
            if self._progress_canvas.winfo_exists():
                self._progress_canvas.delete("bar")
        except tk.TclError:
            pass

    def _update_elapsed(self):
        if not self._running or not self._start_time: return
        elapsed = int(time.time() - self._start_time)
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        txt = f"⏱ {hrs}h {mins:02d}m {secs:02d}s" if hrs > 0 else f"⏱ {mins}m {secs:02d}s"
        self._elapsed_label.configure(text=txt)
        self._elapsed_id = self.after(1000, self._update_elapsed)

    def _stop_elapsed(self):
        if self._elapsed_id:
            self.after_cancel(self._elapsed_id)
            self._elapsed_id = None

    def _on_action_click(self):
        if not self._running:
            self._start_training()
        else:
            self._stop_training()

    def _start_training(self):
        self._running = True
        self._start_time = time.time()
        self._status_label.configure(text="Running Sweep", fg=ACCENT_GREEN)
        self._action_btn.configure(text="■  STOP SWEEP", bg=ACCENT_RED, fg="white")

        self._start_pulse()
        self._start_progress_anim()
        self._update_elapsed()

        cmd = [sys.executable, "-u", TRAIN_AI, "--headless"]
        
        lr = self.entry_lr.get().strip()
        if lr: cmd.extend(["--lr", lr])
        
        dagger = self.entry_dagger.get().strip()
        if dagger: cmd.extend(["--dagger-prob-max", dagger])
        
        curr = self.entry_curr.get().strip()
        if curr: cmd.extend(["--curriculum-prob", curr])
        
        rname = self.entry_name.get().strip()
        if rname: cmd.extend(["--run-name", rname])

        self._process = subprocess.Popen(
            cmd, cwd=DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        self._schedule_chart_refresh()

    def _read_output(self):
        try:
            for line in self._process.stdout:
                line = line.rstrip('\n')
                if line:
                    self.after(0, self._append_log, line)
                    self.after(0, self._parse_stats, line)
        except Exception:
            pass
        finally:
            self.after(0, self._on_process_ended)

    def _append_log(self, line):
        self._log_text.configure(state=tk.NORMAL)
        tag = "game"
        if "Honest eval" in line or ">> " in line: tag = "eval"
        if "New best" in line: tag = "best"
        elif "checkpoint" in line.lower() or "Saving" in line: tag = "save"
        if "Training started" in line or "Resuming" in line: tag = "info"
        if "[D]" in line or "[C,D]" in line: tag = "dagger"
        if "[C]" in line or "[C," in line: tag = "curriculum"

        self._log_text.insert(tk.END, line + "\n", tag)
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

        self._log_line_count += 1
        self._log_count_label.configure(text=f"{self._log_line_count} lines")

    def _clear_log(self):
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.configure(state=tk.DISABLED)
        self._log_line_count = 0
        self._log_count_label.configure(text="0 lines")

    def _parse_stats(self, line):
        if line.startswith("Config:"):
            self._config_label.configure(text=line)

    def _schedule_chart_refresh(self):
        if not self._running: return
        self._refresh_chart()
        self.after(15000, self._schedule_chart_refresh)

    def _refresh_chart(self):
        rname = self.entry_name.get().strip()
        if rname:
            plot_path = os.path.join(DIR, "model", rname, "learning_curve.png")
        else:
            plot_path = os.path.join(DIR, "learning_curve.png")
            
        if not os.path.exists(plot_path): return
        try:
            pil_img = Image.open(plot_path)
            container_w = max(600, self._chart_container.winfo_width() - 8)
            container_h = max(300, self._chart_container.winfo_height() - 8)
            img_ratio = pil_img.width / pil_img.height
            container_ratio = container_w / container_h
            if img_ratio > container_ratio:
                new_w = container_w
                new_h = int(container_w / img_ratio)
            else:
                new_h = container_h
                new_w = int(container_h * img_ratio)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(pil_img)
            self._chart_image = img
            self._chart_label.configure(image=img, text="")
            mod_time = os.path.getmtime(plot_path)
            t = time.strftime("%H:%M:%S", time.localtime(mod_time))
            self._chart_time_label.configure(text=f"Last update: {t}")
        except Exception:
            pass

    def _stop_training(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._running = False
            self._status_label.configure(text="Stopped", fg=ACCENT_RED)
            self._action_btn.configure(text="▶  START SWEEP", bg=ACCENT_GREEN, fg="#000000")
            self._stop_pulse(ACCENT_RED)
            self._stop_progress_anim()
            self._stop_elapsed()

    def _on_process_ended(self):
        self._running = False
        self._stop_progress_anim()
        self._stop_elapsed()
        try:
            if self.winfo_exists() and self._status_label.winfo_exists():
                if self._status_label.cget("text") != "Stopped":
                    self._status_label.configure(text="Finished", fg=ACCENT_ORANGE)
                    self._stop_pulse(ACCENT_ORANGE)
        except tk.TclError:
            pass

    def _on_close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._running = False
        self._stop_pulse()
        self._stop_progress_anim()
        self._stop_elapsed()
        self.destroy()

# ──────────────────────────────────────────────────
# Benchmark Dashboard (Toplevel window)
# ──────────────────────────────────────────────────
class BenchmarkDashboard(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Snake AI — Benchmark Models")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.geometry("1100x1020")
        self.minsize(850, 750)

        self._process = None
        self._running = False
        self._log_lines = []
        self._log_line_count = 0
        self._start_time = None
        self._elapsed_id = None
        self._pulse_id = None
        self._pulse_on = True
        self._progress_id = None
        self._progress_pos = 0
        self._temp_plot_path = None
        self._available_cps = []

        self._build_ui()

    def _find_checkpoints(self):
        model_dir = os.path.join(DIR, "model")
        if not os.path.exists(model_dir):
            return []
        cps = []
        for root_d, dirs, files in os.walk(model_dir):
            for f in files:
                if f in ("checkpoint_best.pth", "checkpoint_last.pth"):
                    path = os.path.join(root_d, f)
                    rel = os.path.relpath(path, model_dir)
                    cps.append(rel)
        return sorted(cps)

    def _build_ui(self):
        # ── Header bar ──
        header = tk.Frame(self, bg="#101014")
        header.pack(fill=tk.X)

        header_inner = tk.Frame(header, bg="#101014")
        header_inner.pack(fill=tk.X, padx=24, pady=(14, 10))

        left_hdr = tk.Frame(header_inner, bg="#101014")
        left_hdr.pack(side=tk.LEFT)

        tk.Label(left_hdr, text="📊", font=("", 20), bg="#101014", fg=ACCENT_ORANGE).pack(side=tk.LEFT, padx=(0, 10))

        title_block = tk.Frame(left_hdr, bg="#101014")
        title_block.pack(side=tk.LEFT)
        tk.Label(title_block, text="BENCHMARK MODELS", font=("Helvetica Neue", 15, "bold"), bg="#101014", fg=TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Label(title_block, text="Evaluate and compare checkpoints", font=("Helvetica Neue", 10), bg="#101014", fg=TEXT_MUTED).pack(anchor=tk.W)

        right_hdr = tk.Frame(header_inner, bg="#101014")
        right_hdr.pack(side=tk.RIGHT)

        self._elapsed_label = tk.Label(right_hdr, text="", font=("SF Mono", 11), bg="#101014", fg=TEXT_MUTED)
        self._elapsed_label.pack(side=tk.RIGHT, padx=(12, 0))

        self._status_dot = tk.Canvas(right_hdr, width=12, height=12, bg="#101014", highlightthickness=0)
        self._status_dot.pack(side=tk.RIGHT, padx=(0, 6))
        self._status_dot.create_oval(2, 2, 10, 10, fill=TEXT_MUTED, outline="", tags="dot")

        self._status_label = tk.Label(right_hdr, text="Ready", font=("Helvetica Neue", 11), bg="#101014", fg=TEXT_SECONDARY)
        self._status_label.pack(side=tk.RIGHT)

        # ── Animated progress bar ──
        self._progress_canvas = tk.Canvas(self, height=2, bg=BG_DARK, highlightthickness=0)
        self._progress_canvas.pack(fill=tk.X)

        # ── Inputs ──
        inputs_outer = tk.Frame(self, bg=BG_DARK)
        inputs_outer.pack(fill=tk.X, padx=24, pady=(12, 4))

        # Checkpoints selection
        cp_frame = tk.Frame(inputs_outer, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        cp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(cp_frame, text="Select Checkpoints (Checkboxes):", bg=BG_CARD, fg=TEXT_MUTED).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        list_frame = tk.Frame(cp_frame, bg=BG_CARD)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Use a canvas for actual checkboxes to fulfill "как чекбоксы" request exactly
        self.canvas_cb = tk.Canvas(list_frame, bg=BG_CARD, highlightthickness=0)
        scroll_cb = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.canvas_cb.yview, bg=BG_DARK)
        self.inner_cb_frame = tk.Frame(self.canvas_cb, bg=BG_CARD)
        
        self.inner_cb_frame.bind("<Configure>", lambda e: self.canvas_cb.configure(scrollregion=self.canvas_cb.bbox("all")))
        self.canvas_cb.create_window((0, 0), window=self.inner_cb_frame, anchor="nw")
        
        self.canvas_cb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_cb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_cb.configure(yscrollcommand=scroll_cb.set)

        self._available_cps = self._find_checkpoints()
        self._cp_vars = {}
        for cp in self._available_cps:
            var = tk.BooleanVar(value=False)
            self._cp_vars[cp] = var
            disp = f"(root) {cp}" if "/" not in cp and "\\" not in cp else cp
            cb = tk.Checkbutton(self.inner_cb_frame, text=disp, variable=var,
                                bg=BG_CARD, fg=TEXT_PRIMARY, selectcolor=BG_DARK,
                                activebackground=BG_CARD, activeforeground=ACCENT_GREEN,
                                font=("SF Mono", 11))
            cb.pack(anchor=tk.W, pady=2)

        # Right side params
        params_frame = tk.Frame(inputs_outer, bg=BG_DARK)
        params_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.entry_games = self._make_input(params_frame, "Games (default 100):")
        self.entry_seed = self._make_input(params_frame, "Seed (optional):")

        # ── Bottom action bar ──
        bottom = tk.Frame(self, bg=BG_DARK)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=(10, 18))

        self._action_btn = tk.Label(
            bottom, text="▶  RUN BENCHMARK",
            font=("Helvetica Neue", 13, "bold"),
            bg=ACCENT_ORANGE, fg="#000000", cursor="hand2",
            padx=24, pady=12
        )
        self._action_btn.pack(fill=tk.X)
        self._action_btn.bind("<Button-1>", lambda e: self._on_action_click())
        self._action_btn.bind("<Enter>", lambda e: self._action_btn.configure(
            bg="#ffaa66" if not self._running else "#ff2244"))
        self._action_btn.bind("<Leave>", lambda e: self._action_btn.configure(
            bg=ACCENT_ORANGE if not self._running else ACCENT_RED))

        # ── Log area ──
        log_section = tk.Frame(self, bg=BG_DARK, height=180)
        log_section.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=(8, 4))
        log_section.pack_propagate(False)

        log_header = tk.Frame(log_section, bg=BG_DARK)
        log_header.pack(fill=tk.X, pady=(0, 4))

        tk.Label(log_header, text="📋  BENCHMARK LOG", font=("Helvetica Neue", 10, "bold"), bg=BG_DARK, fg=TEXT_MUTED).pack(side=tk.LEFT)

        self._log_count_label = tk.Label(log_header, text="0 lines", font=("Helvetica Neue", 9), bg=BG_DARK, fg=TEXT_MUTED)
        self._log_count_label.pack(side=tk.RIGHT)

        log_container = tk.Frame(log_section, bg=LOG_BG, highlightbackground=BORDER_COLOR, highlightthickness=1)
        log_container.pack(fill=tk.BOTH, expand=True)

        self._log_scrollbar = tk.Scrollbar(log_container, orient=tk.VERTICAL, troughcolor=LOG_BG, bg="#222228")
        self._log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_text = tk.Text(log_container, bg=LOG_BG, fg=LOG_FG, font=("SF Mono", 11),
                                  insertbackground=ACCENT_GREEN, selectbackground="#334455",
                                  relief=tk.FLAT, padx=12, pady=8, wrap=tk.WORD,
                                  yscrollcommand=self._log_scrollbar.set, state=tk.DISABLED)
        self._log_text.pack(fill=tk.BOTH, expand=True)
        self._log_scrollbar.config(command=self._log_text.yview)

        self._log_text.tag_configure("eval", foreground=ACCENT_CYAN)
        self._log_text.tag_configure("best", foreground=ACCENT_GREEN, font=("SF Mono", 11, "bold"))
        self._log_text.tag_configure("game", foreground=LOG_FG)
        self._log_text.tag_configure("info", foreground=TEXT_SECONDARY)

        # ── Chart area ──
        chart_section = tk.Frame(self, bg=BG_DARK)
        chart_section.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=24, pady=(8, 4))
        
        chart_header = tk.Frame(chart_section, bg=BG_DARK)
        chart_header.pack(fill=tk.X, pady=(0, 4))

        tk.Label(chart_header, text="📊  BENCHMARK RESULTS", font=("Helvetica Neue", 10, "bold"), bg=BG_DARK, fg=TEXT_MUTED).pack(side=tk.LEFT)

        self._chart_container = tk.Frame(chart_section, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        self._chart_container.pack(fill=tk.BOTH, expand=True)

        self._chart_label = tk.Label(self._chart_container, text="Select checkpoints and run benchmark to see plot",
                                      font=("Helvetica Neue", 11), bg=BG_CARD, fg=TEXT_MUTED)
        self._chart_label.pack(fill=tk.BOTH, expand=True)
        self._chart_image = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _make_input(self, parent, label_text):
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        tk.Label(frame, text=label_text, bg=BG_DARK, fg=TEXT_MUTED, font=("Helvetica Neue", 10)).pack(anchor=tk.W)
        entry = tk.Entry(frame, bg=BG_CARD, fg=TEXT_PRIMARY, insertbackground=ACCENT_GREEN,
                         font=("SF Mono", 11), relief=tk.FLAT, highlightbackground=BORDER_COLOR, highlightthickness=1)
        entry.pack(fill=tk.X, pady=(2, 0), ipady=4)
        return entry

    def _start_pulse(self):
        if not self._running: return
        self._pulse_on = not self._pulse_on
        color = ACCENT_ORANGE if self._pulse_on else "#884400"
        self._status_dot.itemconfig("dot", fill=color)
        self._pulse_id = self.after(600, self._start_pulse)

    def _stop_pulse(self, color=TEXT_MUTED):
        if self._pulse_id:
            self.after_cancel(self._pulse_id)
            self._pulse_id = None
        self._status_dot.itemconfig("dot", fill=color)

    def _start_progress_anim(self):
        if not self._running: return
        self._progress_canvas.delete("bar")
        w = self._progress_canvas.winfo_width()
        if w < 10: w = 1100
        bar_len = w // 4
        x = self._progress_pos % (w + bar_len) - bar_len
        self._progress_canvas.create_rectangle(x, 0, x + bar_len, 2, fill=ACCENT_ORANGE, outline="", tags="bar")
        self._progress_pos += 4
        self._progress_id = self.after(30, self._start_progress_anim)

    def _stop_progress_anim(self):
        if self._progress_id:
            self.after_cancel(self._progress_id)
            self._progress_id = None
        try:
            if self._progress_canvas.winfo_exists():
                self._progress_canvas.delete("bar")
        except tk.TclError:
            pass

    def _update_elapsed(self):
        if not self._running or not self._start_time: return
        elapsed = int(time.time() - self._start_time)
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        txt = f"⏱ {hrs}h {mins:02d}m {secs:02d}s" if hrs > 0 else f"⏱ {mins}m {secs:02d}s"
        self._elapsed_label.configure(text=txt)
        self._elapsed_id = self.after(1000, self._update_elapsed)

    def _stop_elapsed(self):
        if self._elapsed_id:
            self.after_cancel(self._elapsed_id)
            self._elapsed_id = None

    def _on_action_click(self):
        if not self._running:
            self._start_benchmark()
        else:
            self._stop_benchmark()

    def _start_benchmark(self):
        selected = [cp for cp, var in self._cp_vars.items() if var.get()]
        if not selected:
            messagebox.showerror("Error", "No checkpoints selected!")
            return

        self._running = True
        self._start_time = time.time()
        self._status_label.configure(text="Benchmarking", fg=ACCENT_ORANGE)
        self._action_btn.configure(text="■  STOP BENCHMARK", bg=ACCENT_RED, fg="white")

        self._start_pulse()
        self._start_progress_anim()
        self._update_elapsed()

        # clear log
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.configure(state=tk.DISABLED)
        self._log_line_count = 0

        cmd = [sys.executable, "-u", os.path.join("tools", "benchmark.py")]
        
        for cp in selected:
            cmd.extend(["--checkpoint", cp])

        games = self.entry_games.get().strip()
        if not games: games = "100"
        cmd.extend(["--games", games])
        
        seed = self.entry_seed.get().strip()
        if seed: cmd.extend(["--seed", seed])

        self._temp_plot_path = os.path.join(DIR, f"temp_benchmark_{int(time.time())}.png")
        cmd.extend(["--plot", self._temp_plot_path])

        self._process = subprocess.Popen(
            cmd, cwd=DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self):
        try:
            for line in self._process.stdout:
                line = line.rstrip('\n')
                if line:
                    self.after(0, self._append_log, line)
        except Exception:
            pass
        finally:
            self.after(0, self._on_process_ended)

    def _append_log(self, line):
        self._log_text.configure(state=tk.NORMAL)
        tag = "game"
        if "Running" in line: tag = "eval"
        if "mean=" in line: tag = "best"

        self._log_text.insert(tk.END, line + "\n", tag)
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

        self._log_line_count += 1
        self._log_count_label.configure(text=f"{self._log_line_count} lines")

    def _refresh_chart(self):
        if not self._temp_plot_path or not os.path.exists(self._temp_plot_path): return
        try:
            pil_img = Image.open(self._temp_plot_path)
            container_w = max(600, self._chart_container.winfo_width() - 8)
            container_h = max(300, self._chart_container.winfo_height() - 8)
            img_ratio = pil_img.width / pil_img.height
            container_ratio = container_w / container_h
            if img_ratio > container_ratio:
                new_w = container_w
                new_h = int(container_w / img_ratio)
            else:
                new_h = container_h
                new_w = int(container_h * img_ratio)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(pil_img)
            self._chart_image = img
            self._chart_label.configure(image=img, text="")
        except Exception:
            pass

    def _stop_benchmark(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._running = False
            self._status_label.configure(text="Stopped", fg=ACCENT_RED)
            self._action_btn.configure(text="▶  RUN BENCHMARK", bg=ACCENT_ORANGE, fg="#000000")
            self._stop_pulse(ACCENT_RED)
            self._stop_progress_anim()
            self._stop_elapsed()

    def _on_process_ended(self):
        self._running = False
        self._stop_progress_anim()
        self._stop_elapsed()
        try:
            if self.winfo_exists() and self._status_label.winfo_exists():
                if self._status_label.cget("text") != "Stopped":
                    self._status_label.configure(text="Finished", fg=ACCENT_ORANGE)
                    self._stop_pulse(ACCENT_ORANGE)
                    self.after(500, self._refresh_chart)
        except tk.TclError:
            pass

    def _on_close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._running = False
        self._stop_pulse()
        self._stop_progress_anim()
        self._stop_elapsed()
        self.destroy()

# ──────────────────────────────────────────────────
# Launcher actions
# ──────────────────────────────────────────────────
def run_training(root):
    TrainingDashboard(root)


def run_exam():
    print("Running exam (current model)...")
    subprocess.Popen([sys.executable, TRAIN_AI, "--watch", "--games", "10"], cwd=DIR)


def run_algo():
    print("Running perfect algorithm...")
    subprocess.Popen([sys.executable, TEACHER], cwd=DIR)


def run_pretrained():
    print("Running pretrained model...")
    subprocess.Popen([sys.executable, TRAIN_AI, "--watch", "--pretrained", "--games", "10"], cwd=DIR)


def run_manual():
    print("Starting manual play...")
    subprocess.Popen([sys.executable, PLAY_MANUAL], cwd=DIR)


def view_stats():
    plot_path = os.path.join(DIR, "learning_curve.png")
    if not os.path.exists(plot_path):
        messagebox.showinfo("Statistics", "learning_curve.png not created yet. Run training!")
        return

    print("Opening statistics...")
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", plot_path])
        elif sys.platform == "win32":  # Windows
            os.startfile(plot_path)
        else:  # Linux
            subprocess.run(["xdg-open", plot_path])
    except Exception as e:
        messagebox.showerror("Error", f"Could not open file: {e}")


def stop_all_and_exit(root):
    print("Stopping processes and exiting...")
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"])
        else:
            subprocess.run(["pkill", "-f", "train_ai.py"], stderr=subprocess.DEVNULL)
            subprocess.run(["pkill", "-f", "teacher.py"], stderr=subprocess.DEVNULL)
            subprocess.run(["pkill", "-f", "play_manual.py"], stderr=subprocess.DEVNULL)
    except Exception:
        pass
    root.destroy()
    sys.exit()


# ──────────────────────────────────────────────────
# Premium UI Widgets
# ──────────────────────────────────────────────────
class PremiumButton(tk.Canvas):
    """A premium dark button with hover effects, optional colored dot, and chevron."""

    def __init__(self, parent, text, command, dot_color=None, icon=None,
                 show_chevron=True, accent=False, **kwargs):
        super().__init__(parent, highlightthickness=0, bg=BG_DARK, height=48, **kwargs)
        self.command = command
        self._text = text
        self._dot_color = dot_color
        self._icon = icon
        self._show_chevron = show_chevron
        self._accent = accent
        self._hovered = False

        self.bind("<Configure>", self._draw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.configure(cursor="hand2")

    def _draw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()

        if self._accent:
            bg = ACCENT_GREEN if not self._hovered else ACCENT_GREEN_DIM
            fg = "#000000"
            border = ACCENT_GREEN
        else:
            bg = BG_BUTTON_HOVER if self._hovered else BG_BUTTON
            fg = TEXT_PRIMARY
            border = BORDER_COLOR if not self._hovered else TEXT_MUTED

        # Rounded rectangle background
        r = 8
        self._round_rect(2, 2, w - 2, h - 2, r, fill=bg, outline=border)

        x_offset = 18

        # Icon (emoji)
        if self._icon:
            self.create_text(x_offset, h // 2, text=self._icon,
                             font=("", 14), fill=fg, anchor=tk.W)
            x_offset += 28

        # Colored dot
        if self._dot_color:
            dot_r = 5
            self.create_oval(x_offset, h // 2 - dot_r, x_offset + dot_r * 2,
                             h // 2 + dot_r, fill=self._dot_color, outline="")
            x_offset += 20

        # Text
        self.create_text(x_offset, h // 2, text=self._text,
                         font=("Helvetica Neue", 13), fill=fg, anchor=tk.W)

        # Chevron
        if self._show_chevron and not self._accent:
            chevron_x = w - 24
            chevron_color = TEXT_MUTED if not self._hovered else TEXT_SECONDARY
            self.create_text(chevron_x, h // 2, text="›",
                             font=("Helvetica Neue", 18), fill=chevron_color)

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
            x1 + r, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_enter(self, e):
        self._hovered = True
        self._draw()

    def _on_leave(self, e):
        self._hovered = False
        self._draw()

    def _on_click(self, e):
        if self.command:
            self.command()


class BottomButton(tk.Canvas):
    """Outlined bottom action button (View Statistics / Stop All)."""

    def __init__(self, parent, text, command, color=TEXT_PRIMARY, **kwargs):
        super().__init__(parent, highlightthickness=0, bg=BG_DARK, height=44, **kwargs)
        self.command = command
        self._text = text
        self._color = color
        self._hovered = False

        self.bind("<Configure>", self._draw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.configure(cursor="hand2")

    def _draw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()

        bg = BG_BUTTON_HOVER if self._hovered else BG_DARK
        border = self._color if self._hovered else BORDER_COLOR

        r = 8
        points = [
            2 + r, 2, w - 2 - r, 2, w - 2, 2, w - 2, 2 + r,
            w - 2, h - 2 - r, w - 2, h - 2, w - 2 - r, h - 2,
            2 + r, h - 2, 2, h - 2, 2, h - 2 - r,
            2, 2 + r, 2, 2, 2 + r, 2
        ]
        self.create_polygon(points, smooth=True, fill=bg, outline=border)
        self.create_text(w // 2, h // 2, text=self._text,
                         font=("Helvetica Neue", 12), fill=self._color)

    def _on_enter(self, e):
        self._hovered = True
        self._draw()

    def _on_leave(self, e):
        self._hovered = False
        self._draw()

    def _on_click(self, e):
        if self.command:
            self.command()


# ──────────────────────────────────────────────────
# Main Launcher Window
# ──────────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.title("Snake AI")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    # ── Main container ──
    container = tk.Frame(root, bg=BG_DARK, padx=24, pady=20)
    container.pack(fill=tk.BOTH, expand=True)

    # ── Title area ──
    title_frame = tk.Frame(container, bg=BG_DARK)
    title_frame.pack(fill=tk.X, pady=(10, 0))

    tk.Label(title_frame, text="N E U R A L   N E T W O R K",
             font=("Helvetica Neue", 9), bg=BG_DARK, fg=ACCENT_GREEN,
             ).pack()

    # SNAKE·AI title
    title_label = tk.Label(title_frame, text="SNAKE·AI",
                            font=("Helvetica Neue", 42, "bold"), bg=BG_DARK, fg=TEXT_PRIMARY)
    title_label.pack(pady=(4, 0))

    tk.Label(title_frame, text="Reinforcement Learning Controller",
             font=("Helvetica Neue", 11), bg=BG_DARK, fg=TEXT_MUTED
             ).pack(pady=(0, 16))

    # Separator line
    sep = tk.Frame(container, bg=BORDER_COLOR, height=1)
    sep.pack(fill=tk.X, pady=(0, 16))

    # ── TRAINING section ──
    tk.Label(container, text="TRAINING", font=("Helvetica Neue", 10, "bold"),
             bg=BG_DARK, fg=TEXT_MUTED).pack(anchor=tk.W, pady=(0, 6))

    PremiumButton(container, "Start / Continue Training",
                  lambda: run_training(root),
                  icon="▶", accent=True, show_chevron=False
                  ).pack(fill=tk.X, pady=(0, 6))

    PremiumButton(container, "New Sweep Run",
                  lambda: SweepRunDashboard(root),
                  icon="⚡", accent=False, show_chevron=False
                  ).pack(fill=tk.X, pady=(0, 12))

    # ── WATCH section ──
    tk.Label(container, text="WATCH", font=("Helvetica Neue", 10, "bold"),
             bg=BG_DARK, fg=TEXT_MUTED).pack(anchor=tk.W, pady=(4, 6))

    PremiumButton(container, "Perfect Algorithm",
                  run_algo, dot_color=ACCENT_GREEN,
                  ).pack(fill=tk.X, pady=2)

    # "Teacher" subtitle
    teacher_hint = tk.Label(container, text="", bg=BG_DARK)
    teacher_hint.pack()

    PremiumButton(container, "Test Current Model",
                  run_exam, dot_color=ACCENT_BLUE,
                  ).pack(fill=tk.X, pady=2)

    PremiumButton(container, "Pretrained Model",
                  run_pretrained, dot_color=ACCENT_PURPLE,
                  ).pack(fill=tk.X, pady=2)

    PremiumButton(container, "Play Manually",
                  run_manual, dot_color=ACCENT_ORANGE,
                  ).pack(fill=tk.X, pady=(2, 12))

    # Hint text for manual play
    tk.Label(container, text="WASD / Arrows", font=("Helvetica Neue", 9),
             bg=BG_DARK, fg=TEXT_MUTED).pack(anchor=tk.E, pady=(0, 4))

    # ── BENCHMARK section ──
    tk.Label(container, text="BENCHMARK", font=("Helvetica Neue", 10, "bold"),
             bg=BG_DARK, fg=TEXT_MUTED).pack(anchor=tk.W, pady=(4, 6))

    PremiumButton(container, "Benchmark Models",
                  lambda: BenchmarkDashboard(root),
                  icon="📊", accent=False, show_chevron=False, dot_color=ACCENT_ORANGE
                  ).pack(fill=tk.X, pady=(0, 12))

    # ── OTHER section ──
    tk.Label(container, text="OTHER", font=("Helvetica Neue", 10, "bold"),
             bg=BG_DARK, fg=TEXT_MUTED).pack(anchor=tk.W, pady=(4, 6))

    # Bottom buttons row
    bottom_row = tk.Frame(container, bg=BG_DARK)
    bottom_row.pack(fill=tk.X, pady=(0, 8))
    bottom_row.columnconfigure(0, weight=1)
    bottom_row.columnconfigure(1, weight=1)

    stats_btn = BottomButton(bottom_row, "View Statistics", view_stats, color=ACCENT_CYAN)
    stats_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

    exit_btn = BottomButton(bottom_row, "Exit", lambda: stop_all_and_exit(root), color=ACCENT_RED)
    exit_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    # ── Footer ──
    tk.Label(container,
             text="Windows open in separate processes.\nClosing this panel will not stop games.",
             font=("Helvetica Neue", 10), bg=BG_DARK, fg=TEXT_MUTED,
             justify=tk.CENTER).pack(pady=(12, 0))

    # ── Size and center window ──
    root.update_idletasks()
    width = 420
    height = root.winfo_reqheight()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
