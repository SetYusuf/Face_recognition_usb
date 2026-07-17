"""
gui.py — Attendance System GUI  (School Edition)
=================================================
Connects:
  - capture_images.py   (capture 30 photos)
  - faces-train.py      (train model)
  - Face_Recognition.py (live face detection)

Run:  python gui.py
"""

import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import sys
import os
from openpyxl import load_workbook

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────────────────────
BG        = "#f4f6fb"
SURFACE   = "#ffffff"
SURFACE2  = "#eef1f8"
BORDER    = "#d5daea"
BORDER_LT = "#e6eaf4"

NAVY      = "#1a3a6b"
NAVY_LT   = "#2554a0"
NAVY_DIM  = "#cdd8ee"
SKY       = "#2e8fcf"
GREEN     = "#27935a"
AMBER     = "#d4820a"
RED       = "#c0392b"
GOLD      = "#c9a227"

T_HI      = "#0f1d36"
T_MED     = "#4a5675"
T_LOW     = "#8b95b0"
T_WHITE   = "#ffffff"
GREEN_LOG = "#1d7a48"

FONT_TITLE = ("Segoe UI", 11, "bold")
FONT_LABEL = ("Segoe UI", 11, "bold")
FONT_BODY  = ("Segoe UI", 11)
FONT_MONO  = ("Consolas", 11)
FONT_SMALL = ("Segoe UI", 10)
FONT_TINY  = ("Segoe UI", 9)

EXCEL_FILE = r"D:\Data\Excel\attendance.xlsx"


# ─────────────────────────────────────────────────────────────
#  HELPERS  (logic unchanged)
# ─────────────────────────────────────────────────────────────
def run_script(script_name, log_widget, env_input=None):
    script_path = os.path.join(BASE_DIR, script_name)
    _log(log_widget, f"▶  {script_name}\n")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            input=env_input,
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )
        _log(log_widget, result.stdout + result.stderr)
        _log(log_widget, "─" * 44 + "\n")
        return result.returncode == 0
    except Exception as e:
        _log(log_widget, f"ERROR: {e}\n")
        return False


def run_script_live(script_name, log_widget, env_input=None):
    import threading
    script_path = os.path.join(BASE_DIR, script_name)
    _log(log_widget, f"▶  {script_name}\n")

    def _run():
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=BASE_DIR
        )
        if env_input:
            proc.stdin.write(env_input)
            proc.stdin.flush()
            proc.stdin.close()
        for line in proc.stdout:
            log_widget.after(0, lambda l=line: _log(log_widget, l))
        proc.wait()
        log_widget.after(0, lambda: _log(log_widget, "─" * 44 + "\n"))
        if proc.returncode == 0:
            log_widget.after(0, lambda: _log(log_widget, "✓  Training complete!\n"))
            log_widget.after(0, lambda: messagebox.showinfo(
                "Success", "✓  Model trained successfully!"))
        else:
            log_widget.after(0, lambda: _log(log_widget, "✗  Training failed.\n"))

    threading.Thread(target=_run, daemon=True).start()


def _log(widget, text):
    widget.config(state="normal")
    widget.insert("end", text)
    widget.see("end")
    widget.config(state="disabled")


# ─────────────────────────────────────────────────────────────
#  ATTENDANCE WINDOW
# ─────────────────────────────────────────────────────────────
class AttendanceWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Attendance Records")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.grab_set()
        self._build()
        self._center()
        self._load()

    def _build(self):
        # banner
        banner = tk.Frame(self, bg=NAVY, pady=16, padx=24)
        banner.pack(fill="x")
        left = tk.Frame(banner, bg=NAVY)
        left.pack(side="left")
        tk.Label(left, text="Attendance Records",
                 bg=NAVY, fg=T_WHITE,
                 font=("Georgia", 18, "bold")).pack(anchor="w")
        tk.Label(left, text="Student presence log",
                 bg=NAVY, fg=NAVY_DIM, font=FONT_SMALL).pack(anchor="w")
        tk.Button(banner, text="↻  Refresh", command=self._load,
                  bg=NAVY_LT, fg=T_WHITE, font=FONT_BODY,
                  relief="flat", cursor="hand2", padx=12, pady=5,
                  activebackground=SKY, activeforeground=T_WHITE
                  ).pack(side="right")

        tk.Frame(self, bg=GOLD, height=3).pack(fill="x")

        # path bar
        path_bar = tk.Frame(self, bg=SURFACE2, pady=6, padx=24)
        path_bar.pack(fill="x")
        tk.Label(path_bar, text="File:  " + EXCEL_FILE,
                 bg=SURFACE2, fg=T_LOW, font=FONT_SMALL).pack(side="left")
        self.summary_var = tk.StringVar(value="Loading…")
        tk.Label(path_bar, textvariable=self.summary_var,
                 bg=SURFACE2, fg=GREEN,
                 font=("Segoe UI", 10, "bold")).pack(side="right")

        # table
        tbl = tk.Frame(self, bg=BG)
        tbl.pack(fill="both", expand=True)
        vsb = ttk.Scrollbar(tbl, orient="vertical")
        hsb = ttk.Scrollbar(tbl, orient="horizontal")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Att.Treeview",
                        background=SURFACE, foreground=T_MED,
                        fieldbackground=SURFACE, rowheight=30,
                        font=FONT_MONO, borderwidth=0)
        style.configure("Att.Treeview.Heading",
                        background=NAVY_DIM, foreground=NAVY,
                        font=("Segoe UI", 11, "bold"),
                        relief="flat", borderwidth=0)
        style.map("Att.Treeview",
                  background=[("selected", NAVY)],
                  foreground=[("selected", T_WHITE)])

        self.tree = ttk.Treeview(tbl, style="Att.Treeview",
                                 yscrollcommand=vsb.set,
                                 xscrollcommand=hsb.set,
                                 show="headings")
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        # footer
        foot = tk.Frame(self, bg=SURFACE, pady=12, padx=24,
                        highlightthickness=1, highlightbackground=BORDER)
        foot.pack(fill="x")
        tk.Button(foot, text="Open in Excel", command=self._open_excel,
                  bg=GREEN, fg=T_WHITE, font=FONT_LABEL,
                  relief="flat", cursor="hand2", padx=16, pady=6,
                  activebackground=NAVY, activeforeground=T_WHITE
                  ).pack(side="left")
        tk.Button(foot, text="Close", command=self.destroy,
                  bg=BORDER, fg=T_MED, font=FONT_LABEL,
                  relief="flat", cursor="hand2", padx=16, pady=6
                  ).pack(side="right")

    def _load(self):
        self.tree.delete(*self.tree.get_children())
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
        self.tree["columns"] = ()

        if not os.path.exists(EXCEL_FILE):
            self.summary_var.set("File not found")
            return
        try:
            wb  = load_workbook(EXCEL_FILE)
            ws  = wb.active
            headers = [str(c.value) if c.value is not None else "" for c in ws[1]]
            self.tree["columns"] = headers
            for col in headers:
                self.tree.heading(col, text=col, anchor="w")
                w = 130 if col.lower() in ("name", "date", "time") else 70
                self.tree.column(col, width=w, minwidth=50,
                                 anchor="w" if w > 70 else "center")
            row_count = 0
            for row in ws.iter_rows(min_row=2, values_only=True):
                if all(v is None for v in row):
                    continue
                values = [str(v) if v is not None else "" for v in row]
                present = sum(1 for wv in values[3:] if wv == "1")
                tag = "present" if present > 0 else "absent"
                self.tree.insert("", "end", values=values, tags=(tag,))
                row_count += 1
            self.tree.tag_configure("present", foreground=GREEN)
            self.tree.tag_configure("absent",  foreground=T_LOW)
            self.summary_var.set(
                f"{row_count} student(s)   "
                f"{len(headers)-3} weeks   "
                f"updated {self._mtime()}"
            )
        except Exception as ex:
            self.summary_var.set(f"Error: {ex}")

    def _mtime(self):
        try:
            import datetime
            return datetime.datetime.fromtimestamp(
                os.path.getmtime(EXCEL_FILE)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "—"

    def _open_excel(self):
        if not os.path.exists(EXCEL_FILE):
            messagebox.showerror("Not Found",
                                 f"File not found:\n{EXCEL_FILE}", parent=self)
            return
        os.startfile(EXCEL_FILE)

    def _center(self):
        self.update_idletasks()
        w, h = 920, 540
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


# ─────────────────────────────────────────────────────────────
#  REGISTER WINDOW
# ─────────────────────────────────────────────────────────────
class RegisterWindow(tk.Toplevel):
    def __init__(self, master, log_widget):
        super().__init__(master)
        self.log_widget = log_widget
        self.title("Register Student")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self._build()
        self._center()

    def _build(self):
        # banner
        banner = tk.Frame(self, bg=NAVY, pady=18, padx=30)
        banner.pack(fill="x")
        tk.Label(banner, text="Register New Student",
                 bg=NAVY, fg=T_WHITE,
                 font=("Georgia", 16, "bold")).pack(anchor="w")
        tk.Label(banner, text="Fill in details below, then capture face photos.",
                 bg=NAVY, fg=NAVY_DIM, font=FONT_SMALL).pack(anchor="w")

        tk.Frame(self, bg=GOLD, height=3).pack(fill="x")

        # form
        body = tk.Frame(self, bg=BG, padx=30, pady=24)
        body.pack(fill="both", expand=False)
        self._field(body, "Student ID", 0)
        self._field(body, "Full Name",  1)

        # info box
        tip = tk.Frame(body, bg=NAVY_DIM, padx=14, pady=10)
        tip.grid(row=2, column=0, columnspan=2, sticky="ew", pady=18)
        tk.Label(tip,
                 text="Camera will open. Press SPACE to begin capturing 30 photos.",
                 bg=NAVY_DIM, fg=NAVY, font=FONT_SMALL).pack(anchor="w")
        body.columnconfigure(1, weight=1)

        # footer
        foot = tk.Frame(self, bg=SURFACE, padx=30, pady=14,
                        highlightthickness=1, highlightbackground=BORDER)
        foot.pack(fill="x")
        tk.Button(foot, text="Capture 30 Photos",
                  command=self._capture,
                  bg=NAVY, fg=T_WHITE, font=("Segoe UI", 11, "bold"),
                  relief="flat", cursor="hand2", padx=20, pady=9,
                  activebackground=NAVY_LT, activeforeground=T_WHITE
                  ).pack(side="left")
        tk.Button(foot, text="Cancel",
                  command=self.destroy,
                  bg=BORDER, fg=T_MED, font=FONT_LABEL,
                  relief="flat", cursor="hand2", padx=18, pady=9
                  ).pack(side="right")

    def _field(self, parent, label, row):
        tk.Label(parent, text=label, bg=BG, fg=T_MED,
                 font=("Segoe UI", 10, "bold"), anchor="w").grid(
            row=row, column=0, sticky="w", pady=10)
        var = tk.StringVar()
        entry = tk.Entry(parent, textvariable=var,
                         bg=SURFACE, fg=T_HI,
                         font=("Segoe UI", 12),
                         relief="flat", insertbackground=NAVY,
                         width=28,
                         highlightthickness=1,
                         highlightbackground=BORDER,
                         highlightcolor=NAVY)
        entry.grid(row=row, column=1, padx=16, ipady=8,
                   sticky="ew", pady=10)
        if row == 0:
            self.var_id   = var
        else:
            self.var_name = var

    def _capture(self):
        sid = self.var_id.get().strip()
        name = self.var_name.get().strip()

        if not sid:
            messagebox.showerror("Missing Field", "Please enter ID", parent=self)
            return

        if not name:
            messagebox.showerror("Missing Field", "Please enter Name", parent=self)
            return

        # START CAMERA CAPTURE
        log_widget = self.log_widget  # Store reference before destroying
        _log(log_widget, f"Registered: {sid} - {name}\n")

        self.destroy()

        # Run capture script as separate process (for interactive camera)
        script_path = os.path.join(BASE_DIR, "capture_images.py")
        # Pass ID and name as command-line arguments
        subprocess.Popen(
            [sys.executable, script_path, sid, name],
            cwd=BASE_DIR
        )

    def _center(self):
        self.update_idletasks()
        w, h = 480, 420
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


# ─────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Attendance System")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._build()
        self._center()

    def _build(self):
        # ── navy header ──────────────────────────────────────────
        header = tk.Frame(self, bg=NAVY, pady=0)
        header.pack(fill="x")

        crest_area = tk.Frame(header, bg=NAVY, padx=28, pady=18)
        crest_area.pack(side="left")

        crest_canvas = tk.Canvas(crest_area, width=52, height=52,
                                 bg=NAVY, highlightthickness=0)
        crest_canvas.pack(side="left", padx=(0, 16))
        crest_canvas.create_oval(2, 2, 50, 50, fill=GOLD, outline="")
        crest_canvas.create_text(26, 26, text="SA", fill=NAVY,
                                 font=("Georgia", 16, "bold"))

        title_block = tk.Frame(crest_area, bg=NAVY)
        title_block.pack(side="left")
        tk.Label(title_block, text="Smart Attendance System",
                 bg=NAVY, fg=T_WHITE,
                 font=("Georgia", 20, "bold")).pack(anchor="w")
        tk.Label(title_block,
                 text="Face Recognition  ·  Automated Student Tracking",
                 bg=NAVY, fg=NAVY_DIM,
                 font=("Segoe UI", 10)).pack(anchor="w")

        tag = tk.Frame(header, bg=NAVY, padx=24)
        tag.pack(side="right")
        tk.Label(tag, text="Academic Year 2024 / 2025",
                 bg=NAVY, fg=NAVY_DIM, font=FONT_TINY).pack(anchor="e")
        tk.Label(tag, text="v2.0",
                 bg=NAVY, fg=BORDER_LT, font=FONT_TINY).pack(anchor="e")

        # ── gold stripe ──────────────────────────────────────────
        tk.Frame(self, bg=GOLD, height=4).pack(fill="x")

        # ── workflow breadcrumb ──────────────────────────────────
        wf_bar = tk.Frame(self, bg=NAVY_DIM, pady=7, padx=28)
        wf_bar.pack(fill="x")
        steps = [("01", "Register Student"),
                 ("02", "Train Model"),
                 ("03", "Start Detection"),
                 ("04", "View Records"),
                 ("05", "AI Assistant")]
        for i, (num, label) in enumerate(steps):
            s = tk.Frame(wf_bar, bg=NAVY_DIM)
            s.pack(side="left")
            tk.Label(s, text=num, bg=NAVY_DIM, fg=NAVY_LT,
                     font=("Consolas", 9, "bold")).pack(anchor="w")
            tk.Label(s, text=label, bg=NAVY_DIM, fg=NAVY,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w")
            if i < len(steps) - 1:
                tk.Label(wf_bar, text="   →   ",
                         bg=NAVY_DIM, fg=NAVY_LT,
                         font=FONT_SMALL).pack(side="left")

        # ── section label ────────────────────────────────────────
        sec = tk.Frame(self, bg=BG, padx=28, pady=10)
        sec.pack(fill="x")
        tk.Label(sec, text="QUICK ACTIONS",
                 bg=BG, fg=T_LOW,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Frame(sec, bg=BORDER, height=1).pack(fill="x", pady=4)

        # ── horizontal action rows (matching the sketch) ─────────
        rows_frame = tk.Frame(self, bg=BG, padx=28)
        rows_frame.pack(fill="x")

        row_data = [
            ("01", "Register Student",
             "Capture 30 face photos to enrol a new student",
             NAVY,  "#e8ecf7", self._open_register),
            ("02", "Train Model",
             "Build the face recognition model from captured data",
             AMBER, "#fdf3e0", self._train),
            ("03", "Start Detection",
             "Open live camera and mark attendance automatically",
             RED,   "#faeaea", self._detect),
            ("04", "View Attendance",
             "Browse records and open the Excel attendance file",
             GREEN, "#e6f5ed", self._attendance),
            ("05", "AI Assistant",
             "Open AI Vision Assistant — blink detection, object ID, recipe suggestions",
             SKY,  "#e3f0fc", self._ai_assistant),
        ]

        for step, title, desc, color, row_bg, cmd in row_data:
            self._action_row(rows_frame, step, title, desc,
                             color, row_bg, cmd).pack(
                fill="x", pady=5)

        # ── log section label ────────────────────────────────────
        div = tk.Frame(self, bg=BG, padx=28, pady=8)
        div.pack(fill="x")
        tk.Label(div, text="SYSTEM LOG",
                 bg=BG, fg=T_LOW,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Frame(div, bg=BORDER, height=1).pack(fill="x", pady=4)

        # ── log panel ────────────────────────────────────────────
        log_outer = tk.Frame(self, bg=BG, padx=28, pady=4)
        log_outer.pack(fill="both", expand=True)

        log_top = tk.Frame(log_outer, bg=BG)
        log_top.pack(fill="x", pady=4)
        tk.Button(log_top, text="Clear log",
                  command=self._clear_log,
                  bg=BORDER_LT, fg=T_LOW, font=FONT_SMALL,
                  relief="flat", cursor="hand2",
                  padx=10, pady=3).pack(side="right")

        log_box = tk.Frame(log_outer, bg=SURFACE,
                           highlightthickness=1,
                           highlightbackground=BORDER)
        log_box.pack(fill="both", expand=True)

        self.log = tk.Text(
            log_box, height=8,
            bg=SURFACE, fg=GREEN_LOG,
            font=FONT_MONO, relief="flat",
            state="disabled", wrap="word",
            padx=14, pady=10,
            selectbackground=NAVY_DIM,
            insertbackground=NAVY
        )
        sb = tk.Scrollbar(log_box, command=self.log.yview,
                          bg=BORDER_LT, troughcolor=SURFACE)
        self.log.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

        # ── footer ───────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        foot = tk.Frame(self, bg=NAVY, pady=8, padx=28)
        foot.pack(fill="x")
        tk.Label(foot,
                 text="Register  →  Train  →  Detect  →  Attendance",
                 bg=NAVY, fg=NAVY_DIM,
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Label(foot, text="Smart Attendance System  ·  v2.0",
                 bg=NAVY, fg=NAVY_DIM,
                 font=("Segoe UI", 10)).pack(side="right")

    # ── HORIZONTAL ROW BUTTON ─────────────────────────────────
    def _action_row(self, parent, step, title, desc, color, row_bg, cmd):
        """Full-width clickable row — matches the sketch layout."""
        row = tk.Frame(parent, bg=row_bg,
                       highlightthickness=1,
                       highlightbackground=BORDER,
                       cursor="hand2")

        # left colour accent bar (4 px wide)
        accent = tk.Frame(row, bg=color, width=6)
        accent.pack(side="left", fill="y")

        # step badge
        badge = tk.Label(row, text=step,
                         bg=color, fg=T_WHITE,
                         font=("Consolas", 11, "bold"),
                         width=3, pady=0, padx=8)
        badge.pack(side="left", fill="y", padx=(0, 14))

        # text block (title + description)
        text_block = tk.Frame(row, bg=row_bg, pady=14)
        text_block.pack(side="left", fill="both", expand=True)

        title_lbl = tk.Label(text_block, text=title,
                             bg=row_bg, fg=T_HI,
                             font=("Segoe UI", 13, "bold"),
                             anchor="w")
        title_lbl.pack(anchor="w")

        desc_lbl = tk.Label(text_block, text=desc,
                            bg=row_bg, fg=T_MED,
                            font=("Segoe UI", 10),
                            anchor="w")
        desc_lbl.pack(anchor="w")

        # right arrow
        arrow = tk.Label(row, text="→",
                         bg=row_bg, fg=color,
                         font=("Segoe UI", 18, "bold"),
                         padx=20)
        arrow.pack(side="right")

        # hover
        all_widgets = [row, text_block, title_lbl, desc_lbl, arrow]

        def on_enter(e):
            for w in all_widgets:
                w.config(bg=color)
            title_lbl.config(fg=T_WHITE)
            desc_lbl.config(fg=T_WHITE)
            arrow.config(fg=T_WHITE, bg=color)
            badge.config(bg=T_WHITE, fg=color)
            accent.config(bg=T_WHITE)

        def on_leave(e):
            for w in all_widgets:
                w.config(bg=row_bg)
            title_lbl.config(fg=T_HI)
            desc_lbl.config(fg=T_MED)
            arrow.config(fg=color, bg=row_bg)
            badge.config(bg=color, fg=T_WHITE)
            accent.config(bg=color)

        for w in all_widgets + [badge, accent]:
            w.bind("<Button-1>", lambda e, c=cmd: c())
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)

        return row

    # ── ACTIONS  (logic unchanged) ────────────────────────────
    def _open_register(self):
        RegisterWindow(self, self.log)

    def _train(self):
        _log(self.log, "\n" + "─" * 44 + "\n")
        _log(self.log, "Starting model training…\n")
        run_script_live("faces-train.py", self.log)

    def _detect(self):
        _log(self.log, "\n" + "─" * 44 + "\n")
        _log(self.log, "Launching face detection…\n")
        script = os.path.join(BASE_DIR, "Face_Recognition.py")
        subprocess.Popen([sys.executable, script], cwd=BASE_DIR)

    def _attendance(self):
        _log(self.log, "\n" + "─" * 44 + "\n")
        _log(self.log, "Opening attendance records…\n")
        AttendanceWindow(self)

    def _ai_assistant(self):
        _log(self.log, "\n" + "─" * 44 + "\n")
        _log(self.log, "Launching AI Vision Assistant…\n")
        script = os.path.join(BASE_DIR, "ai_vision_assistant.py")
        subprocess.Popen([sys.executable, script], cwd=BASE_DIR)

    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _center(self):
        self.update_idletasks()
        w, h = 660, 680
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


# ── Entry point ──
if __name__ == "__main__":
    App().mainloop()