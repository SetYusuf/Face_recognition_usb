from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from datetime import datetime
import os
import pickle

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = r"D:\Data\Excel"
EXCEL_FILE  = os.path.join(FOLDER_PATH, "attendance.xlsx")
LABELS_PATH = os.path.join(BASE_DIR, "labels.pickle")

# ─────────────────────────────────────────────
#  LOAD REGISTERED LABELS  →  {student_id (int): name (str)}
#  labels.pickle now stores {student_id: name} directly
#  e.g., {1: "john doe", 2: "jane smith"}
# ─────────────────────────────────────────────
def _load_labels() -> dict:
    if not os.path.exists(LABELS_PATH):
        return {}
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)  # {student_id: name}
    return labels

# ─────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────
_HEADER_FILL = PatternFill("solid", start_color="1F4E79", end_color="1F4E79")
_HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=11)
_DATA_FONT   = Font(name="Arial", size=10)
_CENTER      = Alignment(horizontal="center", vertical="center")
_LEFT        = Alignment(horizontal="left",   vertical="center")
_THIN        = Side(style="thin", color="BFBFBF")
_BORDER      = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_ALT_FILL    = PatternFill("solid", start_color="EBF3FB", end_color="EBF3FB")

HEADERS    = ["ID", "Name", "Date", "Time", "Status"]
COL_WIDTHS = [8, 22, 14, 12, 10]

def _apply_header(ws):
    for col, (header, width) in enumerate(zip(HEADERS, COL_WIDTHS), start=1):
        cell            = ws.cell(row=1, column=col)
        cell.value      = header
        cell.font       = _HEADER_FONT
        cell.fill       = _HEADER_FILL
        cell.alignment  = _CENTER
        cell.border     = _BORDER
        ws.column_dimensions[cell.column_letter].width = width
    ws.row_dimensions[1].height = 20

def _apply_row_style(ws, row_idx: int):
    fill = _ALT_FILL if row_idx % 2 == 0 else None
    for col in range(1, len(HEADERS) + 1):
        cell           = ws.cell(row=row_idx, column=col)
        cell.font      = _DATA_FONT
        cell.border    = _BORDER
        cell.alignment = _LEFT if col == 2 else _CENTER
        if fill:
            cell.fill  = fill

# ─────────────────────────────────────────────
#  CREATE FOLDER + EXCEL ON FIRST RUN
# ─────────────────────────────────────────────
if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)

if not os.path.exists(EXCEL_FILE):
    wb = Workbook()
    ws = wb.active
    ws.title        = "Attendance"
    ws.freeze_panes = "A2"
    _apply_header(ws)
    wb.save(EXCEL_FILE)

# ─────────────────────────────────────────────
#  MAIN FUNCTION
# ─────────────────────────────────────────────
def mark_attendance(face_id: int, display_name: str):
    """
    Mark attendance using the EXACT registered ID and name from labels.pickle.

    face_id      → the numeric ID assigned during face registration
    display_name → fallback only if ID is not found in labels.pickle

    Excel output:
        ID   | Name  | Date       | Time     | Status
        1    | Yusuf | 2025-07-10 | 09:03:21 | Present
    """

    # ── 1. Resolve exact registered identity ─────────────────────
    labels          = _load_labels()
    registered_name = labels.get(face_id, display_name)  # exact name from pickle
    registered_id   = face_id                             # numeric ID from pickle

    now      = datetime.now()
    date_now = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    # ── 2. Load workbook ──────────────────────────────────────────
    wb = load_workbook(EXCEL_FILE)
    ws = wb.active

    # ── 3. Check if this ID is already recorded TODAY ─────────────
    for row in ws.iter_rows(min_row=2):
        existing_id   = row[0].value
        existing_date = row[2].value

        if existing_id == registered_id and str(existing_date) == date_now:
            # Already marked today — update time only, never change ID or Name
            row[1].value = registered_name
            row[3].value = time_now
            row[4].value = "Present"
            wb.save(EXCEL_FILE)
            print(f"[UPDATE]  ID={registered_id}  Name={registered_name}  already marked today — time updated.")
            return

    # ── 4. First time today — append a new row ────────────────────
    next_row = ws.max_row + 1
    ws.cell(row=next_row, column=1).value = registered_id     # exact numeric ID
    ws.cell(row=next_row, column=2).value = registered_name   # exact registered name
    ws.cell(row=next_row, column=3).value = date_now
    ws.cell(row=next_row, column=4).value = time_now
    ws.cell(row=next_row, column=5).value = "Present"

    _apply_row_style(ws, next_row)

    wb.save(EXCEL_FILE)
    print(f"[MARKED]  ID={registered_id}  Name={registered_name}  {date_now}  {time_now}")