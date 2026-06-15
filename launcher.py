

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
RECORD_DEMO = os.path.join("tools", "record_demo.py")


def find_model_names():
    """Available model run-names: '(default)' (model/) plus subfolders of model/."""
    model_dir = os.path.join(DIR, "model")
    names = ["(default)"]
    if os.path.isdir(model_dir):
        for entry in sorted(os.listdir(model_dir)):
            if os.path.isdir(os.path.join(model_dir, entry)):
                names.append(entry)
    return names


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

TOOLTIP_BG = "#1e1e24"
TOOLTIP_FG = "#ccccdd"
TOAST_SUCCESS_BG = "#0a2a18"
TOAST_SUCCESS_BORDER = "#00e878"
TOAST_ERROR_BG = "#2a0a14"
TOAST_ERROR_BORDER = "#ff4466"


# ──────────────────────────────────────────────────
# Tooltip widget
# ──────────────────────────────────────────────────
class Tooltip:
    """Show a tooltip on hover for any widget."""

    def __init__(self, widget, text, delay=400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tip_window = None
        self._after_id = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<Button-1>", self._hide, add="+")

    def _schedule(self, event=None):
        self._hide()
        self._after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self._tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4

        self._tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        frame = tk.Frame(tw, bg=TOOLTIP_BG, highlightbackground=BORDER_COLOR,
                         highlightthickness=1)
        frame.pack()

        label = tk.Label(frame, text=self.text, bg=TOOLTIP_BG, fg=TOOLTIP_FG,
                         font=("Helvetica Neue", 11), padx=10, pady=6,
                         wraplength=280, justify=tk.LEFT)
        label.pack()

    def _hide(self, event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


# ──────────────────────────────────────────────────
# Toast notification
# ──────────────────────────────────────────────────
class ToastNotification:
    """Slide-in notification at the top of a window."""

    def __init__(self, parent):
        self.parent = parent
        self._toast_frame = None
        self._hide_id = None

    def show(self, message, kind="success", duration=2500):
        """Show a toast. kind = 'success' | 'error'."""
        self._dismiss()

        bg = TOAST_SUCCESS_BG if kind == "success" else TOAST_ERROR_BG
        border = TOAST_SUCCESS_BORDER if kind == "success" else TOAST_ERROR_BORDER
        icon = "✓" if kind == "success" else "✕"
        icon_color = ACCENT_GREEN if kind == "success" else ACCENT_RED

        self._toast_frame = tk.Frame(self.parent, bg=bg,
                                      highlightbackground=border,
                                      highlightthickness=1)
        self._toast_frame.place(relx=0.5, y=8, anchor=tk.N)

        inner = tk.Frame(self._toast_frame, bg=bg)
        inner.pack(padx=16, pady=8)

        tk.Label(inner, text=icon, font=("", 13), bg=bg, fg=icon_color
                 ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(inner, text=message, font=("Helvetica Neue", 11),
                 bg=bg, fg=TEXT_PRIMARY).pack(side=tk.LEFT)

        self._hide_id = self.parent.after(duration, self._dismiss)

    def _dismiss(self):
        if self._hide_id:
            try:
                self.parent.after_cancel(self._hide_id)
            except Exception:
                pass
            self._hide_id = None
        if self._toast_frame:
            try:
                self._toast_frame.destroy()
            except Exception:
                pass
            self._toast_frame = None


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
        self._on_model_selected()

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

        # ── Model selection ──
        model_row = tk.Frame(self, bg=BG_DARK)
        model_row.pack(fill=tk.X, padx=24, pady=(12, 0))

        tk.Label(model_row, text="MODEL", font=("Helvetica Neue", 10, "bold"),
                 bg=BG_DARK, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(0, 10))

        self.option_add('*TCombobox*Listbox.background', BG_CARD)
        self.option_add('*TCombobox*Listbox.foreground', TEXT_PRIMARY)
        self.option_add('*TCombobox*Listbox.selectBackground', BG_BUTTON_HOVER)
        self.option_add('*TCombobox*Listbox.selectForeground', ACCENT_GREEN)

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("Dark.TCombobox",
                        fieldbackground=BG_CARD, background=BG_CARD,
                        foreground=TEXT_PRIMARY, arrowcolor=TEXT_PRIMARY,
                        bordercolor=BORDER_COLOR, borderwidth=1)
        style.map("Dark.TCombobox",
                  fieldbackground=[('readonly', BG_CARD)],
                  background=[('active', BG_BUTTON_HOVER)])

        self._model_var = tk.StringVar(value="(default)")
        self._model_combo = ttk.Combobox(model_row, textvariable=self._model_var,
                                          values=self._find_models(),
                                          style="Dark.TCombobox",
                                          font=("SF Mono", 11), width=26)
        self._model_combo.pack(side=tk.LEFT)
        self._model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        self._model_combo.bind("<Return>", self._on_model_selected)
        self._model_combo.bind("<FocusOut>", self._on_model_selected)

        self._model_info_label = tk.Label(model_row, text="",
                                           font=("Helvetica Neue", 10),
                                           bg=BG_DARK, fg=TEXT_SECONDARY)
        self._model_info_label.pack(side=tk.LEFT, padx=(14, 0))

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
        self._model_combo.configure(state="disabled")

        # Start animations
        self._start_pulse()
        self._start_progress_anim()
        self._update_elapsed()

        cmd = [sys.executable, "-u", TRAIN_AI, "--headless"]
        model_name = self._model_var.get().strip()
        if model_name and model_name != "(default)":
            cmd.extend(["--run-name", model_name])

        self._process = subprocess.Popen(
            cmd,
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
                        self._stats["Steps"].configure(text=self._format_steps(steps))

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

    # ──────────────────────────────────────────────
    # Model selection
    # ──────────────────────────────────────────────
    def _find_models(self):
        """Available model run-names: '(default)' (model/) plus subfolders of model/."""
        return find_model_names()

    def _model_folder(self, name=None):
        name = (name if name is not None else self._model_var.get()).strip()
        if not name or name == "(default)":
            return os.path.join(DIR, "model")
        return os.path.join(DIR, "model", name)

    def _learning_curve_path(self, name=None):
        name = (name if name is not None else self._model_var.get()).strip()
        if not name or name == "(default)":
            return os.path.join(DIR, "learning_curve.png")
        return os.path.join(DIR, "model", name, "learning_curve.png")

    @staticmethod
    def _format_steps(steps):
        if steps >= 1_000_000:
            return f"{steps/1_000_000:.1f}M"
        elif steps >= 1000:
            return f"{steps/1000:.0f}K"
        return str(steps)

    def _load_checkpoint_info(self, folder):
        """Read n_games/total_steps/best eval from this model's last checkpoint, if any."""
        import torch
        for fname in ("checkpoint_last.pth", "checkpoint_best.pth"):
            path = os.path.join(folder, fname)
            if os.path.exists(path):
                try:
                    cp = torch.load(path, map_location="cpu", weights_only=True)
                except Exception:
                    continue
                return {
                    "n_games": cp.get("n_games", 0),
                    "total_steps": cp.get("total_steps", 0),
                    "best_eval_score": cp.get("best_eval_score", cp.get("eval_score", 0.0)),
                }
        return None

    def _on_model_selected(self, event=None):
        """Refresh the 'latest results' preview (stats + chart) for the selected model."""
        if self._running:
            return
        name = self._model_var.get().strip() or "(default)"
        if name not in self._model_combo["values"]:
            self._model_combo["values"] = (*self._model_combo["values"], name)

        info = self._load_checkpoint_info(self._model_folder(name))
        if info:
            games = info["n_games"]
            steps = self._format_steps(info["total_steps"])
            best = info["best_eval_score"]
            self._model_info_label.configure(
                text=f"Last: {games} games · {steps} steps · best eval {best:.1f}",
                fg=TEXT_SECONDARY)
            self._stats["Game"].configure(text=str(games))
            self._stats["Steps"].configure(text=steps)
            self._stats["Best Eval"].configure(text=f"{best:.1f}", fg=TEXT_PRIMARY)
            self._stats["Score"].configure(text="—")
            self._stats["Loss"].configure(text="—")
        else:
            self._model_info_label.configure(text="No checkpoint yet — will start fresh",
                                              fg=TEXT_MUTED)
            for stat in self._stats.values():
                stat.configure(text="—", fg=TEXT_PRIMARY)

        self._refresh_chart()

    def _schedule_chart_refresh(self):
        if not self._running:
            return
        self._refresh_chart()
        self.after(15000, self._schedule_chart_refresh)  # Every 15 seconds

    def _refresh_chart(self):
        plot_path = self._learning_curve_path()
        if not os.path.exists(plot_path):
            self._chart_image = None
            self._chart_label.configure(
                image="", text="No data yet — start training to see the learning curve")
            self._chart_time_label.configure(text="")
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
            self._model_combo.configure(state="normal")
            self._stop_pulse(ACCENT_RED)
            self._stop_progress_anim()
            self._stop_elapsed()

    def _on_process_ended(self):
        self._running = False
        self._stop_progress_anim()
        self._stop_elapsed()
        try:
            if self.winfo_exists() and self._status_label.winfo_exists():
                self._model_combo.configure(state="normal")
                if self._status_label.cget("text") != "Stopped":
                    self._status_label.configure(text="Finished", fg=ACCENT_ORANGE)
                    self._stop_pulse(ACCENT_ORANGE)
                    self._action_btn.configure(text="▶  START TRAINING", bg=ACCENT_GREEN, fg="#000000")
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
                    self._action_btn.configure(text="▶  START SWEEP", bg=ACCENT_GREEN, fg="#000000")
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
        self._last_csv_data = None  # Store last benchmark output for CSV export

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

        # Checkpoint header with Select All / Deselect All
        cp_header = tk.Frame(cp_frame, bg=BG_CARD)
        cp_header.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(cp_header, text="Select Checkpoints:", bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Helvetica Neue", 10)).pack(side=tk.LEFT)

        deselect_btn = tk.Label(cp_header, text="Deselect All", bg=BG_CARD, fg=TEXT_MUTED,
                                font=("Helvetica Neue", 9), cursor="hand2")
        deselect_btn.pack(side=tk.RIGHT, padx=(6, 0))
        deselect_btn.bind("<Button-1>", lambda e: self._toggle_all_checkpoints(False))
        deselect_btn.bind("<Enter>", lambda e: deselect_btn.configure(fg=ACCENT_RED))
        deselect_btn.bind("<Leave>", lambda e: deselect_btn.configure(fg=TEXT_MUTED))

        select_btn = tk.Label(cp_header, text="Select All", bg=BG_CARD, fg=ACCENT_CYAN,
                              font=("Helvetica Neue", 9), cursor="hand2")
        select_btn.pack(side=tk.RIGHT, padx=(0, 6))
        select_btn.bind("<Button-1>", lambda e: self._toggle_all_checkpoints(True))
        select_btn.bind("<Enter>", lambda e: select_btn.configure(fg=ACCENT_GREEN))
        select_btn.bind("<Leave>", lambda e: select_btn.configure(fg=ACCENT_CYAN))

        list_frame = tk.Frame(cp_frame, bg=BG_CARD)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Use a canvas for actual checkboxes
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

        # ── Options row ──
        options_frame = tk.Frame(params_frame, bg=BG_DARK)
        options_frame.pack(fill=tk.X, padx=10, pady=(10, 5))


        self._render_var = tk.BooleanVar(value=False)
        render_cb = tk.Checkbutton(
            options_frame, text="Show game window",
            variable=self._render_var, bg=BG_DARK, fg=TEXT_PRIMARY,
            selectcolor=BG_CARD, activebackground=BG_DARK,
            activeforeground=ACCENT_GREEN, font=("Helvetica Neue", 10))
        render_cb.pack(anchor=tk.W, pady=2)
        Tooltip(render_cb, "Render the game during benchmark.\nSlower but lets you watch the AI play.")

        # ── Bottom action bar ──
        bottom = tk.Frame(self, bg=BG_DARK)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=(10, 18))

        bottom_row = tk.Frame(bottom, bg=BG_DARK)
        bottom_row.pack(fill=tk.X)

        self._action_btn = tk.Label(
            bottom_row, text="▶  RUN BENCHMARK",
            font=("Helvetica Neue", 13, "bold"),
            bg=ACCENT_ORANGE, fg="#000000", cursor="hand2",
            padx=24, pady=12
        )
        self._action_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._action_btn.bind("<Button-1>", lambda e: self._on_action_click())
        self._action_btn.bind("<Enter>", lambda e: self._action_btn.configure(
            bg="#ffaa66" if not self._running else "#ff2244"))
        self._action_btn.bind("<Leave>", lambda e: self._action_btn.configure(
            bg=ACCENT_ORANGE if not self._running else ACCENT_RED))

        self._csv_btn = tk.Label(
            bottom_row, text="💾 CSV",
            font=("Helvetica Neue", 12, "bold"),
            bg=BG_BUTTON, fg=TEXT_MUTED, cursor="hand2",
            padx=16, pady=12
        )
        self._csv_btn.pack(side=tk.RIGHT, padx=(8, 0))
        self._csv_btn.bind("<Button-1>", lambda e: self._export_csv())
        self._csv_btn.bind("<Enter>", lambda e: self._csv_btn.configure(
            bg=BG_BUTTON_HOVER, fg=ACCENT_CYAN))
        self._csv_btn.bind("<Leave>", lambda e: self._csv_btn.configure(
            bg=BG_BUTTON, fg=TEXT_MUTED if not self._last_csv_data else ACCENT_CYAN))
        Tooltip(self._csv_btn, "Export benchmark results to CSV file.\nAvailable after a benchmark run completes.")

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

    def _toggle_all_checkpoints(self, state):
        """Select or deselect all checkpoints."""
        for var in self._cp_vars.values():
            var.set(state)

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


        # Render option
        if self._render_var.get():
            cmd.append("--render")

        self._temp_plot_path = os.path.join(DIR, f"temp_benchmark_{int(time.time())}.png")
        cmd.extend(["--plot", self._temp_plot_path])

        # CSV temp path for export
        self._temp_csv_path = os.path.join(DIR, f"temp_benchmark_{int(time.time())}.csv")
        cmd.extend(["--csv", self._temp_csv_path])

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

    def _export_csv(self):
        """Export benchmark CSV to user-chosen location."""
        if hasattr(self, '_temp_csv_path') and os.path.exists(self._temp_csv_path):
            dest = filedialog.asksaveasfilename(
                parent=self,
                title="Export Benchmark CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile="benchmark_results.csv"
            )
            if dest:
                import shutil
                shutil.copy2(self._temp_csv_path, dest)
                messagebox.showinfo("Export", f"CSV saved to:\n{dest}", parent=self)
        else:
            messagebox.showinfo("Export", "No benchmark data available yet.\nRun a benchmark first.",
                                parent=self)

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
                    self._action_btn.configure(text="▶  RUN BENCHMARK", bg=ACCENT_ORANGE, fg="#000000")
                    self.after(500, self._refresh_chart)
                    # Enable CSV button highlight
                    if hasattr(self, '_temp_csv_path') and os.path.exists(self._temp_csv_path):
                        self._last_csv_data = True
                        self._csv_btn.configure(fg=ACCENT_CYAN)
        except tk.TclError:
            pass

    def _cleanup_temp_files(self):
        """Remove temporary benchmark plot/csv files."""
        for pattern in ["temp_benchmark_*.png", "temp_benchmark_*.csv"]:
            for f in glob.glob(os.path.join(DIR, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def _on_close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._running = False
        self._stop_pulse()
        self._stop_progress_anim()
        self._stop_elapsed()
        self._cleanup_temp_files()
        self.destroy()


# ──────────────────────────────────────────────────
# Record Demo Dialog (Toplevel window)
# ──────────────────────────────────────────────────
class RecordDemoDialog(tk.Toplevel):
    """Dialog for recording GIF demos of the AI."""

    def __init__(self, master):
        super().__init__(master)
        self.title("Snake AI — Record GIF Demo")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self.geometry("460x380")

        self._process = None
        self._running = False
        self._toast = ToastNotification(self)

        self._build_ui()

        # Center on parent
        self.transient(master)
        self.update_idletasks()
        x = master.winfo_rootx() + (master.winfo_width() - 460) // 2
        y = master.winfo_rooty() + (master.winfo_height() - 380) // 2
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg="#101014")
        header.pack(fill=tk.X)

        header_inner = tk.Frame(header, bg="#101014")
        header_inner.pack(fill=tk.X, padx=20, pady=(12, 10))

        tk.Label(header_inner, text="📹", font=("", 18), bg="#101014", fg=ACCENT_PURPLE
                 ).pack(side=tk.LEFT, padx=(0, 10))

        title_block = tk.Frame(header_inner, bg="#101014")
        title_block.pack(side=tk.LEFT)
        tk.Label(title_block, text="RECORD GIF DEMO",
                 font=("Helvetica Neue", 14, "bold"),
                 bg="#101014", fg=TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Label(title_block, text="Generate animated GIF of AI gameplay",
                 font=("Helvetica Neue", 10),
                 bg="#101014", fg=TEXT_MUTED).pack(anchor=tk.W)

        # Options
        body = tk.Frame(self, bg=BG_DARK, padx=20, pady=14)
        body.pack(fill=tk.BOTH, expand=True)

        tk.Label(body, text="MODE", font=("Helvetica Neue", 10, "bold"),
                 bg=BG_DARK, fg=TEXT_MUTED).pack(anchor=tk.W, pady=(0, 6))

        self._mode_var = tk.StringVar(value="pretrained")

        modes = [
            ("pretrained", "Pretrained Model", "Use pretrained.pth — shows trained AI skill"),
            ("untrained", "Untrained Network", "Random network — shows baseline before training"),
            ("both", "Both", "Record pretrained AND untrained side by side"),
        ]

        for val, label, desc in modes:
            row = tk.Frame(body, bg=BG_DARK)
            row.pack(fill=tk.X, pady=2)
            rb = tk.Radiobutton(row, text=label, variable=self._mode_var, value=val,
                                bg=BG_DARK, fg=TEXT_PRIMARY, selectcolor=BG_CARD,
                                activebackground=BG_DARK, activeforeground=ACCENT_GREEN,
                                font=("Helvetica Neue", 11))
            rb.pack(side=tk.LEFT)
            tk.Label(row, text=f"  {desc}", bg=BG_DARK, fg=TEXT_MUTED,
                     font=("Helvetica Neue", 9)).pack(side=tk.LEFT)

        # Status
        self._status_label = tk.Label(body, text="Ready to record",
                                       font=("Helvetica Neue", 10),
                                       bg=BG_DARK, fg=TEXT_MUTED)
        self._status_label.pack(anchor=tk.W, pady=(16, 0))

        # Progress dots
        self._progress_label = tk.Label(body, text="",
                                         font=("SF Mono", 10),
                                         bg=BG_DARK, fg=ACCENT_PURPLE)
        self._progress_label.pack(anchor=tk.W, pady=(4, 0))

        # Action button
        self._action_btn = tk.Label(
            body, text="⏺  START RECORDING",
            font=("Helvetica Neue", 12, "bold"),
            bg=ACCENT_PURPLE, fg="white", cursor="hand2",
            padx=20, pady=10
        )
        self._action_btn.pack(fill=tk.X, pady=(16, 0))
        self._action_btn.bind("<Button-1>", lambda e: self._on_action())
        self._action_btn.bind("<Enter>", lambda e: self._action_btn.configure(
            bg="#9955ee" if not self._running else "#ff2244"))
        self._action_btn.bind("<Leave>", lambda e: self._action_btn.configure(
            bg=ACCENT_PURPLE if not self._running else ACCENT_RED))

        # Open folder button (hidden until done)
        self._open_btn = tk.Label(
            body, text="📂  Open assets folder",
            font=("Helvetica Neue", 10),
            bg=BG_BUTTON, fg=TEXT_SECONDARY, cursor="hand2",
            padx=12, pady=6
        )
        self._open_btn.bind("<Button-1>", lambda e: self._open_assets())
        self._open_btn.bind("<Enter>", lambda e: self._open_btn.configure(fg=ACCENT_CYAN))
        self._open_btn.bind("<Leave>", lambda e: self._open_btn.configure(fg=TEXT_SECONDARY))
        # Not packed yet — shown after recording

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_action(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        mode = self._mode_var.get()
        self._running = True
        self._status_label.configure(text=f"Recording {mode}...", fg=ACCENT_PURPLE)
        self._action_btn.configure(text="■  STOP RECORDING", bg=ACCENT_RED)

        cmd = [sys.executable, "-u", RECORD_DEMO, mode]
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
                    self.after(0, self._update_progress, line)
        except Exception:
            pass
        finally:
            self.after(0, self._on_done)

    def _update_progress(self, line):
        # Show last meaningful line
        if "Frame" in line or "Saved" in line or "Recording" in line or "ERROR" in line:
            self._progress_label.configure(text=line[:60])

    def _on_done(self):
        self._running = False
        self._action_btn.configure(text="⏺  START RECORDING", bg=ACCENT_PURPLE)

        if self._process and self._process.returncode == 0:
            self._status_label.configure(text="✓ Recording complete!", fg=ACCENT_GREEN)
            self._progress_label.configure(text="GIF saved to assets/ folder")
            self._open_btn.pack(fill=tk.X, pady=(8, 0))
        else:
            self._status_label.configure(text="Recording failed", fg=ACCENT_RED)

    def _stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._running = False
        self._status_label.configure(text="Stopped", fg=ACCENT_RED)
        self._action_btn.configure(text="⏺  START RECORDING", bg=ACCENT_PURPLE)

    def _open_assets(self):
        assets_dir = os.path.join(DIR, "assets")
        if sys.platform == "darwin":
            subprocess.run(["open", assets_dir])
        elif sys.platform == "win32":
            os.startfile(assets_dir)
        else:
            subprocess.run(["xdg-open", assets_dir])

    def _on_close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self.destroy()


# ──────────────────────────────────────────────────
# Watch Results Dialog (Toplevel window)
# ──────────────────────────────────────────────────
class WatchResultsDialog(tk.Toplevel):
    """Streams a `--watch` run's stdout (checkpoint info, per-game scores, final
    average/max) into a log panel alongside the separate pygame game window."""

    def __init__(self, master, cmd, subtitle):
        super().__init__(master)
        self.title("Snake AI — Test Results")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self.geometry("420x360")

        self._process = None
        self._running = True

        self._build_ui(subtitle)

        self.transient(master)
        self.update_idletasks()
        x = master.winfo_rootx() + (master.winfo_width() - 420) // 2
        y = master.winfo_rooty() + (master.winfo_height() - 360) // 2
        self.geometry(f"+{x}+{y}")

        self._start(cmd)

    def _build_ui(self, subtitle):
        # Header
        header = tk.Frame(self, bg="#101014")
        header.pack(fill=tk.X)

        header_inner = tk.Frame(header, bg="#101014")
        header_inner.pack(fill=tk.X, padx=20, pady=(12, 10))

        tk.Label(header_inner, text="🎮", font=("", 18), bg="#101014", fg=ACCENT_BLUE
                 ).pack(side=tk.LEFT, padx=(0, 10))

        title_block = tk.Frame(header_inner, bg="#101014")
        title_block.pack(side=tk.LEFT)
        tk.Label(title_block, text="TEST RESULTS",
                 font=("Helvetica Neue", 14, "bold"),
                 bg="#101014", fg=TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Label(title_block, text=subtitle,
                 font=("Helvetica Neue", 10),
                 bg="#101014", fg=TEXT_MUTED).pack(anchor=tk.W)

        self._status_label = tk.Label(header_inner, text="Running",
                                       font=("Helvetica Neue", 11),
                                       bg="#101014", fg=ACCENT_ORANGE)
        self._status_label.pack(side=tk.RIGHT)

        # Log area
        log_container = tk.Frame(self, bg=LOG_BG, highlightbackground=BORDER_COLOR,
                                  highlightthickness=1)
        log_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 6))

        log_scrollbar = tk.Scrollbar(log_container, orient=tk.VERTICAL, troughcolor=LOG_BG, bg="#222228")
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_text = tk.Text(log_container, bg=LOG_BG, fg=LOG_FG, font=("SF Mono", 11),
                                  insertbackground=ACCENT_GREEN, selectbackground="#334455",
                                  relief=tk.FLAT, padx=12, pady=8, wrap=tk.WORD,
                                  yscrollcommand=log_scrollbar.set, state=tk.DISABLED)
        self._log_text.pack(fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=self._log_text.yview)

        self._log_text.tag_configure("info", foreground=ACCENT_CYAN)
        self._log_text.tag_configure("game", foreground=LOG_FG)
        self._log_text.tag_configure("summary", foreground=ACCENT_GREEN, font=("SF Mono", 11, "bold"))
        self._log_text.tag_configure("error", foreground=ACCENT_RED)

        # Stop / Close button
        self._action_btn = tk.Label(
            self, text="■  STOP",
            font=("Helvetica Neue", 12, "bold"),
            bg=ACCENT_RED, fg="white", cursor="hand2",
            padx=20, pady=8
        )
        self._action_btn.pack(fill=tk.X, padx=20, pady=(0, 14))
        self._action_btn.bind("<Button-1>", lambda e: self._on_action())
        self._action_btn.bind("<Enter>", lambda e: self._action_btn.configure(
            bg="#ff2244" if self._running else BG_BUTTON_HOVER))
        self._action_btn.bind("<Leave>", lambda e: self._action_btn.configure(
            bg=ACCENT_RED if self._running else BG_BUTTON))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start(self, cmd):
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
        tag = "game"
        if line.startswith("Loaded") or line.startswith("Resuming"):
            tag = "info"
        elif line.startswith("Average:"):
            tag = "summary"
        elif "not found" in line or "ERROR" in line:
            tag = "error"

        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, line + "\n", tag)
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _on_process_ended(self):
        self._running = False
        try:
            if self.winfo_exists():
                if self._status_label.cget("text") == "Running":
                    self._status_label.configure(text="Done", fg=ACCENT_GREEN)
                self._action_btn.configure(text="✕  CLOSE", bg=BG_BUTTON, fg=TEXT_PRIMARY)
        except tk.TclError:
            pass

    def _on_action(self):
        if self._running and self._process and self._process.poll() is None:
            self._process.terminate()
            self._running = False
            self._status_label.configure(text="Stopped", fg=ACCENT_RED)
            self._action_btn.configure(text="✕  CLOSE", bg=BG_BUTTON, fg=TEXT_PRIMARY)
        else:
            self.destroy()

    def _on_close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self.destroy()


# ──────────────────────────────────────────────────
# Launcher actions
# ──────────────────────────────────────────────────
def run_training(root):
    TrainingDashboard(root)


def run_exam(root, num_games=10, model_name=None):
    label = model_name if model_name and model_name != "(default)" else "current"
    print(f"Running exam ({label} model, {num_games} games)...")
    cmd = [sys.executable, "-u", TRAIN_AI, "--watch", "--games", str(num_games)]
    if model_name and model_name != "(default)":
        cmd.extend(["--run-name", model_name])
    WatchResultsDialog(root, cmd, f"{label} model — {num_games} games")


def run_algo():
    print("Running perfect algorithm...")
    subprocess.Popen([sys.executable, TEACHER], cwd=DIR)


def run_pretrained(root, num_games=10):
    print(f"Running pretrained model ({num_games} games)...")
    cmd = [sys.executable, "-u", TRAIN_AI, "--watch", "--pretrained", "--games", str(num_games)]
    WatchResultsDialog(root, cmd, f"pretrained.pth — {num_games} games")


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


def open_model_folder():
    """Open the model/ directory in the file manager."""
    model_dir = os.path.join(DIR, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", model_dir])
        elif sys.platform == "win32":
            os.startfile(model_dir)
        else:
            subprocess.run(["xdg-open", model_dir])
    except Exception as e:
        messagebox.showerror("Error", f"Could not open folder: {e}")


def stop_all_and_exit(root):
    if not messagebox.askyesno("Exit", "Stop all running Snake AI processes and exit?",
                                parent=root):
        return
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
    """A premium dark button with hover effects, optional colored dot, subtitle, and chevron."""

    def __init__(self, parent, text, command, dot_color=None, icon=None,
                 show_chevron=True, accent=False, subtitle=None, **kwargs):
        height = 48 if not subtitle else 58
        super().__init__(parent, highlightthickness=0, bg=BG_DARK, height=height, **kwargs)
        self.command = command
        self._text = text
        self._dot_color = dot_color
        self._icon = icon
        self._show_chevron = show_chevron
        self._accent = accent
        self._subtitle = subtitle
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
        text_y = h // 2 if not self._subtitle else h // 2 - 6

        # Icon (emoji)
        if self._icon:
            self.create_text(x_offset, text_y, text=self._icon,
                             font=("", 14), fill=fg, anchor=tk.W)
            x_offset += 28

        # Colored dot
        if self._dot_color:
            dot_r = 5
            self.create_oval(x_offset, text_y - dot_r, x_offset + dot_r * 2,
                             text_y + dot_r, fill=self._dot_color, outline="")
            x_offset += 20

        # Text
        self.create_text(x_offset, text_y, text=self._text,
                         font=("Helvetica Neue", 13), fill=fg, anchor=tk.W)

        # Subtitle
        if self._subtitle:
            sub_fg = TEXT_MUTED if not self._accent else "#005530"
            self.create_text(x_offset, text_y + 18, text=self._subtitle,
                             font=("Helvetica Neue", 9), fill=sub_fg, anchor=tk.W)

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

    toast = ToastNotification(root)

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

    btn_train = PremiumButton(container, "Start / Continue Training",
                  lambda: run_training(root),
                  icon="▶", accent=True, show_chevron=False,
                  subtitle="Open dashboard with real-time stats and learning curve"
                  )
    btn_train.pack(fill=tk.X, pady=(0, 6))
    Tooltip(btn_train, "Open training dashboard with live stats,\ntraining log, and learning curve chart.")

    btn_sweep = PremiumButton(container, "New Sweep Run",
                  lambda: SweepRunDashboard(root),
                  icon="⚡", accent=False, show_chevron=False,
                  subtitle="Train with custom LR, DAgger, Curriculum parameters"
                  )
    btn_sweep.pack(fill=tk.X, pady=(0, 12))
    Tooltip(btn_sweep, "Start training with custom hyperparameters.\nResults saved to a separate model folder.")

    # ── WATCH section ──
    tk.Label(container, text="WATCH", font=("Helvetica Neue", 10, "bold"),
             bg=BG_DARK, fg=TEXT_MUTED).pack(anchor=tk.W, pady=(4, 6))

    # Model selector row
    model_row = tk.Frame(container, bg=BG_DARK)
    model_row.pack(fill=tk.X, pady=(0, 8))

    tk.Label(model_row, text="Model:", font=("Helvetica Neue", 10),
             bg=BG_DARK, fg=TEXT_SECONDARY).pack(side=tk.LEFT, padx=(0, 6))

    root.option_add('*TCombobox*Listbox.background', BG_CARD)
    root.option_add('*TCombobox*Listbox.foreground', TEXT_PRIMARY)
    root.option_add('*TCombobox*Listbox.selectBackground', BG_BUTTON_HOVER)
    root.option_add('*TCombobox*Listbox.selectForeground', ACCENT_GREEN)

    combo_style = ttk.Style(root)
    combo_style.theme_use('clam')
    combo_style.configure("Dark.TCombobox",
                           fieldbackground=BG_CARD, background=BG_CARD,
                           foreground=TEXT_PRIMARY, arrowcolor=TEXT_PRIMARY,
                           bordercolor=BORDER_COLOR, borderwidth=1)
    combo_style.map("Dark.TCombobox",
                     fieldbackground=[('readonly', BG_CARD)],
                     background=[('active', BG_BUTTON_HOVER)])

    exam_model_var = tk.StringVar(value="(default)")
    exam_model_combo = ttk.Combobox(model_row, textvariable=exam_model_var,
                                     values=find_model_names(),
                                     style="Dark.TCombobox", state="readonly",
                                     font=("SF Mono", 11), width=22)
    exam_model_combo.pack(side=tk.LEFT)
    Tooltip(exam_model_combo, "Model folder for 'Test Current Model'.\n(default) = model/checkpoint_best.pth")

    def _get_exam_model():
        return exam_model_var.get().strip() or "(default)"

    # Watch controls row
    watch_controls = tk.Frame(container, bg=BG_DARK)
    watch_controls.pack(fill=tk.X, pady=(0, 8))

    # Games spinner
    games_frame = tk.Frame(watch_controls, bg=BG_DARK)
    games_frame.pack(side=tk.LEFT)

    tk.Label(games_frame, text="Games:", font=("Helvetica Neue", 10),
             bg=BG_DARK, fg=TEXT_SECONDARY).pack(side=tk.LEFT, padx=(0, 6))

    games_var = tk.StringVar(value="10")
    games_spinner = tk.Spinbox(games_frame, from_=1, to=100, textvariable=games_var,
                                width=4, bg=BG_CARD, fg=TEXT_PRIMARY, buttonbackground=BG_BUTTON,
                                insertbackground=ACCENT_GREEN, font=("SF Mono", 11),
                                relief=tk.FLAT, highlightbackground=BORDER_COLOR,
                                highlightthickness=1)
    games_spinner.pack(side=tk.LEFT)
    Tooltip(games_spinner, "Number of games to play in Watch mode.\nApplies to Test Current and Pretrained.")



    def _get_watch_games():
        try:
            return int(games_var.get())
        except ValueError:
            return 10

    btn_algo = PremiumButton(container, "Perfect Algorithm",
                  lambda: [run_algo(), toast.show("Perfect Algorithm started")],
                  dot_color=ACCENT_GREEN,
                  subtitle="Hamiltonian cycle teacher — always reaches max score"
                  )
    btn_algo.pack(fill=tk.X, pady=2)
    Tooltip(btn_algo, "Watch the Hamiltonian cycle algorithm play.\nIt always wins (fills the entire board).")

    btn_test = PremiumButton(container, "Test Current Model",
                  lambda: [run_exam(root, _get_watch_games(), _get_exam_model()),
                           toast.show(f"Testing {_get_exam_model()} — {_get_watch_games()} games")],
                  dot_color=ACCENT_BLUE,
                  subtitle="Watch the selected model's checkpoint_best.pth"
                  )
    btn_test.pack(fill=tk.X, pady=2)
    Tooltip(btn_test, "Watch the selected model's checkpoint_best.pth\nplay the selected number of games.")

    btn_pre = PremiumButton(container, "Pretrained Model",
                  lambda: [run_pretrained(root, _get_watch_games()),
                           toast.show(f"Pretrained model — {_get_watch_games()} games")],
                  dot_color=ACCENT_PURPLE,
                  subtitle="Watch the included pretrained.pth demo model"
                  )
    btn_pre.pack(fill=tk.X, pady=2)
    Tooltip(btn_pre, "Watch the included pretrained model play.\nUseful as a reference for your own training.")

    btn_manual = PremiumButton(container, "Play Manually",
                  lambda: [run_manual(), toast.show("Manual play started — WASD / Arrows")],
                  dot_color=ACCENT_ORANGE,
                  subtitle="Play Snake yourself — WASD or Arrow keys, ESC to quit"
                  )
    btn_manual.pack(fill=tk.X, pady=(2, 12))
    Tooltip(btn_manual, "Play Snake yourself using WASD or Arrow keys.\nESC to quit, +/- to adjust speed.")

    # ── TOOLS section ──
    tk.Label(container, text="TOOLS", font=("Helvetica Neue", 10, "bold"),
             bg=BG_DARK, fg=TEXT_MUTED).pack(anchor=tk.W, pady=(4, 6))

    btn_benchmark = PremiumButton(container, "Benchmark Models",
                  lambda: BenchmarkDashboard(root),
                  icon="📊", accent=False, show_chevron=False, dot_color=ACCENT_ORANGE,
                  subtitle="Run N games on checkpoints, compare stats, export CSV"
                  )
    btn_benchmark.pack(fill=tk.X, pady=2)
    Tooltip(btn_benchmark, "Benchmark one or more checkpoints.\nCompare stats, view distributions, export CSV.")

    btn_record = PremiumButton(container, "Record GIF Demo",
                  lambda: RecordDemoDialog(root),
                  icon="📹", accent=False, show_chevron=False, dot_color=ACCENT_PURPLE,
                  subtitle="Generate animated GIF of AI gameplay for sharing"
                  )
    btn_record.pack(fill=tk.X, pady=(2, 12))
    Tooltip(btn_record, "Record an animated GIF of the AI playing.\nChoose pretrained, untrained, or both.")

    # ── OTHER section ──
    tk.Label(container, text="OTHER", font=("Helvetica Neue", 10, "bold"),
             bg=BG_DARK, fg=TEXT_MUTED).pack(anchor=tk.W, pady=(4, 6))

    # Bottom buttons row — 3 columns
    bottom_row = tk.Frame(container, bg=BG_DARK)
    bottom_row.pack(fill=tk.X, pady=(0, 8))
    bottom_row.columnconfigure(0, weight=1)
    bottom_row.columnconfigure(1, weight=1)
    bottom_row.columnconfigure(2, weight=1)

    stats_btn = BottomButton(bottom_row, "📈 Statistics", view_stats, color=ACCENT_CYAN)
    stats_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    Tooltip(stats_btn, "Open learning_curve.png in your\ndefault image viewer.")

    folder_btn = BottomButton(bottom_row, "📂 Models", open_model_folder, color=ACCENT_BLUE)
    folder_btn.grid(row=0, column=1, sticky="ew", padx=4)
    Tooltip(folder_btn, "Open the model/ directory\nin your file manager.")

    exit_btn = BottomButton(bottom_row, "⏻ Exit", lambda: stop_all_and_exit(root), color=ACCENT_RED)
    exit_btn.grid(row=0, column=2, sticky="ew", padx=(4, 0))
    Tooltip(exit_btn, "Stop all running Snake AI processes\nand close the launcher.")

    # ── Footer ──
    tk.Label(container,
             text="Snake AI · Neural Network Controller\n"
                  "Game windows open separately · Press ESC in game to quit",
             font=("Helvetica Neue", 10), bg=BG_DARK, fg=TEXT_MUTED,
             justify=tk.CENTER).pack(pady=(12, 0))

    # ── Size and center window ──
    root.update_idletasks()
    width = 460
    height = root.winfo_reqheight()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
