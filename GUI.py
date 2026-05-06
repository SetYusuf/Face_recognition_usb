"""
gui.py — Attendance System GUI  (Redesigned)
=============================================
Connects:
  - capture_images.py  (capture 30 photos)
  - faces-train.py     (train model)
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
#  DESIGN TOKENS  — warm dark "biometric lab" palette
# ─────────────────────────────────────────────────────────────
BG          = "#0d0e14"      # deep space background
SURFACE     = "#13151f"      # card surface
SURFACE2    = "#1a1d2b"      # elevated surface
BORDER      = "#252836"      # subtle border
BORDER_LT   = "#2e3347"      # lighter border / separator

# Accent
TEAL        = "#00c9a7"      # primary accent (alive green-teal)
TEAL_DIM    = "#00725e"      # dimmed teal for subtle elements
AMBER       = "#f5a623"      # warm highlight / warning
ROSE        = "#e05c7a"      # danger / detection action
VIOLET      = "#7b6ef6"      # attendance / secondary
BLUE        = "#3d9bf2"      # train action

# Text
T_HI        = "#eef0f8"      # high-emphasis text
T_MED       = "#9498b4"      # medium text
T_LOW       = "#525874"      # muted / captions
T_TEAL      = "#00c9a7"      # teal text
GREEN_LOG   = "#43d98c"      # log output

FONT_HERO   = ("Georgia", 22, "bold")
FONT_TITLE  = ("Georgia", 12, "bold")
FONT_LABEL  = ("Consolas", 10, "bold")
FONT_BODY   = ("Consolas", 9)
FONT_MONO   = ("Consolas", 9)
FONT_SMALL  = ("Consolas", 8)

EXCEL_FILE  = r"D:\Data\Excel\attendance.xlsx"


# ─────────────────────────────────────────────────────────────
#  HELPERS
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
        output = result.stdout + result.stderr
        _log(log_widget, output)
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
        # ── accent stripe ──
        tk.Frame(self, bg=VIOLET, height=3).pack(fill="x")

        # ── header ──
        hdr = tk.Frame(self, bg=SURFACE, pady=14, padx=24)
        hdr.pack(fill="x")

        left = tk.Frame(hdr, bg=SURFACE)
        left.pack(side="left")
        tk.Label(left, text="ATTENDANCE", bg=SURFACE, fg=VIOLET,
                 font=("Georgia", 16, "bold")).pack(anchor="w")
        tk.Label(left, text="Student presence records",
                 bg=SURFACE, fg=T_LOW, font=FONT_SMALL).pack(anchor="w")

        # pill button
        tk.Button(hdr, text="↻  Refresh", command=self._load,
                  bg=BORDER_LT, fg=T_MED, font=FONT_BODY,
                  relief="flat", cursor="hand2",
                  padx=12, pady=5,
                  activebackground=VIOLET, activeforeground=T_HI
                  ).pack(side="right", padx=(0, 4))

        # ── path bar ──
        path_bar = tk.Frame(self, bg=SURFACE2, pady=6, padx=24)
        path_bar.pack(fill="x")
        tk.Label(path_bar, text="📁  " + EXCEL_FILE,
                 bg=SURFACE2, fg=T_LOW, font=FONT_SMALL).pack(side="left")
        self.summary_var = tk.StringVar(value="Loading…")
        tk.Label(path_bar, textvariable=self.summary_var,
                 bg=SURFACE2, fg=GREEN_LOG, font=FONT_SMALL).pack(side="right")

        # ── table ──
        tbl = tk.Frame(self, bg=BG)
        tbl.pack(fill="both", expand=True, padx=0, pady=0)

        vsb = ttk.Scrollbar(tbl, orient="vertical")
        hsb = ttk.Scrollbar(tbl, orient="horizontal")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Att.Treeview",
                        background=SURFACE,
                        foreground=T_MED,
                        fieldbackground=SURFACE,
                        rowheight=30,
                        font=FONT_MONO,
                        borderwidth=0)
        style.configure("Att.Treeview.Heading",
                        background=SURFACE2,
                        foreground=VIOLET,
                        font=("Consolas", 9, "bold"),
                        relief="flat",
                        borderwidth=0)
        style.map("Att.Treeview",
                  background=[("selected", VIOLET)],
                  foreground=[("selected", T_HI)])

        self.tree = ttk.Treeview(
            tbl, style="Att.Treeview",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            show="headings"
        )
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        # ── footer ──
        foot = tk.Frame(self, bg=SURFACE, pady=12, padx=24)
        foot.pack(fill="x")

        self._pill_btn(foot, "📂  Open in Excel", self._open_excel, GREEN_LOG).pack(side="left")
        self._pill_btn(foot, "✕  Close", self.destroy, BORDER_LT, fg=T_MED).pack(side="right")

    def _pill_btn(self, parent, text, cmd, bg, fg=BG):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg, font=FONT_LABEL,
                         relief="flat", cursor="hand2",
                         padx=16, pady=6,
                         activebackground=T_HI, activeforeground=BG)

    def _load(self):
        self.tree.delete(*self.tree.get_children())
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
        self.tree["columns"] = ()

        if not os.path.exists(EXCEL_FILE):
            self.summary_var.set(f"✗  File not found")
            return
        try:
            wb = load_workbook(EXCEL_FILE)
            ws = wb.active
            headers = [str(c.value) if c.value is not None else "" for c in ws[1]]
            self.tree["columns"] = headers
            for col in headers:
                self.tree.heading(col, text=col, anchor="w")
                w = 130 if col.lower() in ("name", "date", "time") else 70
                self.tree.column(col, width=w, minwidth=50, anchor="w" if w > 70 else "center")

            row_count = 0
            for row in ws.iter_rows(min_row=2, values_only=True):
                if all(v is None for v in row):
                    continue
                values = [str(v) if v is not None else "" for v in row]
                present = sum(1 for w in values[3:] if w == "1")
                tag = "present" if present > 0 else "absent"
                self.tree.insert("", "end", values=values, tags=(tag,))
                row_count += 1

            self.tree.tag_configure("present", foreground=GREEN_LOG)
            self.tree.tag_configure("absent",  foreground=T_LOW)

            self.summary_var.set(
                f"✓  {row_count} student(s)   "
                f"{len(headers)-3} weeks   "
                f"updated {self._mtime()}"
            )
        except Exception as ex:
            self.summary_var.set(f"✗  {ex}")

    def _mtime(self):
        try:
            import datetime
            return datetime.datetime.fromtimestamp(
                os.path.getmtime(EXCEL_FILE)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "—"

    def _open_excel(self):
        if not os.path.exists(EXCEL_FILE):
            messagebox.showerror("Not Found", f"File not found:\n{EXCEL_FILE}", parent=self)
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
        tk.Frame(self, bg=TEAL, height=3).pack(fill="x")

        # ── header ──
        hdr = tk.Frame(self, bg=SURFACE, pady=18, padx=30)
        hdr.pack(fill="x")

        tk.Label(hdr, text="REGISTER STUDENT",
                 bg=SURFACE, fg=TEAL, font=("Georgia", 16, "bold")).pack(anchor="w")
        tk.Label(hdr, text="Fill in student details, then capture 30 face photos.",
                 bg=SURFACE, fg=T_LOW, font=FONT_SMALL).pack(anchor="w", pady=(3, 0))

        # ── divider ──
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── form body ──
        body = tk.Frame(self, bg=BG, padx=30, pady=24)
        body.pack(fill="both", expand=True)

        self._field(body, "Student ID", 0)
        self._field(body, "Student Name", 1)

        # tip
        tip = tk.Frame(body, bg=SURFACE2, padx=12, pady=8)
        tip.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        tk.Label(tip, text="📷  Camera will open.  Press SPACE to begin capturing.",
                 bg=SURFACE2, fg=T_MED, font=FONT_SMALL).pack(anchor="w")

        body.columnconfigure(1, weight=1)

        # ── footer ──
        foot = tk.Frame(self, bg=SURFACE, padx=30, pady=14)
        foot.pack(fill="x")
        tk.Button(foot, text="📷  Capture 30 Photos",
                  command=self._capture,
                  bg=TEAL, fg=BG, font=FONT_LABEL,
                  relief="flat", cursor="hand2",
                  padx=18, pady=8,
                  activebackground=T_HI, activeforeground=BG
                  ).pack(side="left")
        tk.Button(foot, text="Cancel",
                  command=self.destroy,
                  bg=BORDER, fg=T_MED, font=FONT_LABEL,
                  relief="flat", cursor="hand2",
                  padx=18, pady=8
                  ).pack(side="right")

    def _field(self, parent, label, row):
        tk.Label(parent, text=label, bg=BG, fg=T_MED,
                 font=FONT_LABEL, anchor="w").grid(
            row=row, column=0, sticky="w", pady=(0, 14))

        var = tk.StringVar()
        entry = tk.Entry(
            parent, textvariable=var,
            bg=SURFACE2, fg=T_HI,
            font=("Consolas", 11),
            relief="flat",
            insertbackground=TEAL,
            width=26,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=TEAL
        )
        entry.grid(row=row, column=1, padx=(16, 0), ipady=7, sticky="ew", pady=(0, 14))

        if row == 0:
            self.var_id = var
        else:
            self.var_name = var

    def _capture(self):
        sid  = self.var_id.get().strip()
        name = self.var_name.get().strip()
        if not sid:
            messagebox.showerror("Missing Field", "Please enter a Student ID.", parent=self)
            return
        if not name:
            messagebox.showerror("Missing Field", "Please enter a Student Name.", parent=self)
            return
        person_label = f"{name}_{sid}"
        self.destroy()
        run_script("capture_images.py", self.log_widget, env_input=person_label + "\n")

    def _center(self):
        self.update_idletasks()
        w, h = 460, 340
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


# ─────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Attendance System — Face Recognition")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._build()
        self._center()

    # ── BUILD ──────────────────────────────────────────────────
    def _build(self):
        # top accent stripe
        tk.Frame(self, bg=TEAL, height=3).pack(fill="x")

        # ── hero header ──
        hero = tk.Frame(self, bg=SURFACE, pady=22, padx=32)
        hero.pack(fill="x")

        left = tk.Frame(hero, bg=SURFACE)
        left.pack(side="left")
        tk.Label(left, text="ATTENDANCE SYSTEM",
                 bg=SURFACE, fg=T_HI, font=("Georgia", 20, "bold")).pack(anchor="w")
        tk.Label(left, text="Face Recognition · Automated Tracking",
                 bg=SURFACE, fg=T_LOW, font=FONT_SMALL).pack(anchor="w", pady=(3, 0))

        # status dot (decorative)
        dot_frame = tk.Frame(hero, bg=SURFACE)
        dot_frame.pack(side="right", padx=(0, 4))
        tk.Label(dot_frame, text="⬤  SYSTEM READY",
                 bg=SURFACE, fg=TEAL_DIM, font=FONT_SMALL).pack()

        # ── thin separator ──
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── workflow label ──
        wf = tk.Frame(self, bg=BG, pady=10, padx=32)
        wf.pack(fill="x")

        steps = [("01", "Register"), ("02", "Train"), ("03", "Detect"), ("04", "Review")]
        for i, (num, label) in enumerate(steps):
            s = tk.Frame(wf, bg=BG)
            s.pack(side="left")
            tk.Label(s, text=num, bg=BG, fg=TEAL_DIM,
                     font=("Consolas", 7, "bold")).pack(anchor="w")
            tk.Label(s, text=label, bg=BG, fg=T_MED,
                     font=("Consolas", 8, "bold")).pack(anchor="w")
            if i < len(steps) - 1:
                tk.Label(wf, text="  —  ", bg=BG, fg=BORDER_LT,
                         font=FONT_SMALL).pack(side="left")

        # ── action cards ──
        cards_frame = tk.Frame(self, bg=BG, padx=24, pady=6)
        cards_frame.pack(fill="x")

        card_data = [
            ("⊕", "Register",      "Capture student\nface samples",     TEAL,   self._open_register),
            ("⚙", "Train Model",   "Build recognition\nmodel from data", AMBER,  self._train),
            ("◉", "Face\nDetect",  "Live camera\nattendance marking",    ROSE,   self._detect),
            ("▦", "Attendance",    "View & export\nattendance records",  VIOLET, self._attendance),
        ]

        for icon, title, desc, color, cmd in card_data:
            self._action_card(cards_frame, icon, title, desc, color, cmd).pack(
                side="left", padx=6, pady=10)

        # ── separator ──
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24)

        # ── log panel ──
        log_outer = tk.Frame(self, bg=BG, padx=24, pady=16)
        log_outer.pack(fill="both", expand=True)

        log_hdr = tk.Frame(log_outer, bg=BG)
        log_hdr.pack(fill="x", pady=(0, 6))
        tk.Label(log_hdr, text="CONSOLE OUTPUT", bg=BG, fg=T_LOW,
                 font=("Consolas", 8, "bold")).pack(side="left")
        tk.Button(log_hdr, text="Clear", command=self._clear_log,
                  bg=BORDER, fg=T_LOW, font=FONT_SMALL,
                  relief="flat", cursor="hand2", padx=8, pady=2
                  ).pack(side="right")

        log_container = tk.Frame(log_outer, bg=SURFACE,
                                 highlightthickness=1, highlightbackground=BORDER)
        log_container.pack(fill="both", expand=True)

        self.log = tk.Text(
            log_container, height=9,
            bg=SURFACE, fg=GREEN_LOG,
            font=FONT_MONO, relief="flat",
            state="disabled", wrap="word",
            padx=14, pady=10,
            selectbackground=BORDER_LT,
            insertbackground=TEAL
        )
        sb = tk.Scrollbar(log_container, command=self.log.yview,
                          bg=BORDER, troughcolor=SURFACE)
        self.log.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

        # ── footer strip ──
        tk.Frame(self, bg=SURFACE2, height=1).pack(fill="x")
        foot = tk.Frame(self, bg=SURFACE2, pady=7, padx=24)
        foot.pack(fill="x")
        tk.Label(foot, text="Register  →  Train  →  Face Detection  →  Attendance",
                 bg=SURFACE2, fg=T_LOW, font=FONT_SMALL).pack(side="left")
        tk.Label(foot, text="v2.0",
                 bg=SURFACE2, fg=BORDER_LT, font=FONT_SMALL).pack(side="right")

    # ── ACTION CARD ────────────────────────────────────────────
    def _action_card(self, parent, icon, title, desc, color, cmd):
        card = tk.Frame(parent, bg=SURFACE2,
                        width=158, height=170,
                        highlightthickness=1,
                        highlightbackground=BORDER,
                        cursor="hand2")
        card.pack_propagate(False)

        # colored top strip on each card
        tk.Frame(card, bg=color, height=3).pack(fill="x")

        icon_lbl = tk.Label(card, text=icon, bg=SURFACE2, fg=color,
                            font=("Consolas", 28, "bold"))
        icon_lbl.pack(pady=(14, 4))

        title_lbl = tk.Label(card, text=title, bg=SURFACE2, fg=T_HI,
                             font=FONT_TITLE, justify="center")
        title_lbl.pack()

        desc_lbl = tk.Label(card, text=desc, bg=SURFACE2, fg=T_LOW,
                            font=FONT_SMALL, justify="center")
        desc_lbl.pack(pady=(4, 0))

        # hover effect
        def on_enter(e):
            card.config(bg=color, highlightbackground=color)
            icon_lbl.config(bg=color, fg=BG)
            title_lbl.config(bg=color, fg=BG)
            desc_lbl.config(bg=color, fg=BG)
            for child in card.winfo_children():
                if isinstance(child, tk.Frame) and child.winfo_height() <= 3:
                    child.config(bg=BG)

        def on_leave(e):
            card.config(bg=SURFACE2, highlightbackground=BORDER)
            icon_lbl.config(bg=SURFACE2, fg=color)
            title_lbl.config(bg=SURFACE2, fg=T_HI)
            desc_lbl.config(bg=SURFACE2, fg=T_LOW)
            for child in card.winfo_children():
                if isinstance(child, tk.Frame) and child.winfo_height() <= 3:
                    child.config(bg=color)

        for w in [card, icon_lbl, title_lbl, desc_lbl]:
            w.bind("<Button-1>", lambda e, c=cmd: c())
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)

        return card

    # ── ACTIONS ────────────────────────────────────────────────
    def _open_register(self):
        RegisterWindow(self, self.log)

    def _train(self):
        _log(self.log, "\n" + "─" * 44 + "\n")
        _log(self.log, "⚙  Starting model training…\n")
        run_script_live("faces-train.py", self.log)

    def _detect(self):
        _log(self.log, "\n" + "─" * 44 + "\n")
        _log(self.log, "◉  Launching face detection…\n")
        script = os.path.join(BASE_DIR, "Face_Recognition.py")
        subprocess.Popen([sys.executable, script], cwd=BASE_DIR)

    def _attendance(self):
        _log(self.log, "\n" + "─" * 44 + "\n")
        _log(self.log, f"▦  Opening attendance records…\n")
        AttendanceWindow(self)

    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _center(self):
        self.update_idletasks()
        w, h = 720, 610
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


# ── Entry point ──
if __name__ == "__main__":
    App().mainloop()