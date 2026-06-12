import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys

# Project working directory
DIR = os.path.dirname(os.path.abspath(__file__))


def run_training():
    print("Starting training...")
    subprocess.Popen([sys.executable, "train_ai.py"], cwd=DIR)


def run_exam():
    print("Running exam (current model)...")
    subprocess.Popen([sys.executable, "train_ai.py", "--watch", "--games", "10"], cwd=DIR)


def run_algo():
    print("Running perfect algorithm...")
    subprocess.Popen([sys.executable, "teacher.py"], cwd=DIR)


def run_pretrained():
    print("Running pretrained model...")
    subprocess.Popen([sys.executable, "train_ai.py", "--watch", "--pretrained", "--games", "10"], cwd=DIR)


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


def stop_all():
    print("Stopping processes...")
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"])
        else:
            subprocess.run(["pkill", "-f", "train_ai.py"])
            subprocess.run(["pkill", "-f", "teacher.py"])
        messagebox.showinfo("Stopped", "All game and training processes stopped.")
    except Exception as e:
        messagebox.showerror("Error", f"Could not stop processes: {e}")


def exit_app(root):
    root.destroy()
    sys.exit()


def add_section(parent, title):
    lbl = tk.Label(parent, text=title, font=("Helvetica", 11, "bold"), fg="gray40")
    lbl.pack(fill=tk.X, padx=30, pady=(16, 2))


def add_button(parent, text, command):
    btn = ttk.Button(parent, text=text, command=command)
    btn.pack(fill=tk.X, padx=30, pady=4)
    return btn


def main():
    root = tk.Tk()
    root.title("Snake AI - Control Panel")
    root.resizable(False, False)

    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TButton", font=("Helvetica", 13), padding=8)

    # Title
    lbl = tk.Label(root, text="🐍 Snake AI Control Panel", font=("Helvetica", 18, "bold"))
    lbl.pack(pady=(20, 5))

    # Training
    add_section(root, "TRAINING")
    add_button(root, "Start / Continue Training", run_training)

    # Watch gameplay
    add_section(root, "WATCH")
    add_button(root, "Perfect Algorithm (Teacher)", run_algo)
    add_button(root, "Test Current Model", run_exam)
    add_button(root, "Pretrained Model", run_pretrained)

    # Other
    add_section(root, "OTHER")
    add_button(root, "View Statistics", view_stats)
    add_button(root, "🛑 Stop All", stop_all)
    add_button(root, "Exit", lambda: exit_app(root))

    # Info
    lbl_info = tk.Label(
        root,
        text="Windows open in separate processes.\nClosing this panel will not stop games.",
        font=("Helvetica", 10), fg="gray", justify=tk.CENTER,
    )
    lbl_info.pack(pady=(16, 16))

    # Size and center window
    root.update_idletasks()
    width = max(400, root.winfo_reqwidth())
    height = root.winfo_reqheight()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
