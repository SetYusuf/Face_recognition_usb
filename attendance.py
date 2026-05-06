from openpyxl import Workbook, load_workbook
from datetime import datetime
import os

# ---------------- PATH ----------------
folder_path = r"D:\Data\Excel"
excel_file = os.path.join(folder_path, "attendance.xlsx")

# ---------------- CREATE FOLDER ----------------
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# ---------------- CREATE EXCEL ----------------
if not os.path.exists(excel_file):
    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"

    ws.append(["ID", "Name", "Date", "Time", "Status"])
    wb.save(excel_file)


def mark_attendance(student_id, name):
    wb = load_workbook(excel_file)
    ws = wb.active

    now = datetime.now()
    date_now = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    # ---------------- FIND USER ----------------
    for row in ws.iter_rows(min_row=2):
        if row[0].value == student_id:

            # update existing record
            row[1].value = name
            row[2].value = date_now
            row[3].value = time_now
            row[4].value = "Present"

            wb.save(excel_file)
            print(f"{name} marked present")
            return

    # ---------------- NEW PERSON ----------------
    ws.append([student_id, name, date_now, time_now, "Present"])
    wb.save(excel_file)

    print(f"{name} added and marked present")