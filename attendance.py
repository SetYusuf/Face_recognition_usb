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

    ws.append(["Name", "Date", "Time"] + [f"Week{i}" for i in range(1, 11)])
    wb.save(excel_file)


def mark_attendance(name):
    wb = load_workbook(excel_file)
    ws = wb.active

    now = datetime.now()
    date_now = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    # ---------------- FIND USER ----------------
    for row in ws.iter_rows(min_row=2):
        if row[0].value == name:

            # update date/time
            row[1].value = date_now
            row[2].value = time_now

            # fill next empty week with 1
            for i in range(3, 13):  # Week1 -> Week10 (columns 4-13)
                if row[i].value is None or row[i].value == "":
                    row[i].value = 1
                    break

            wb.save(excel_file)
            print(f"{name} marked present (week updated)")
            return

    # ---------------- NEW PERSON ----------------
    new_row = [name, date_now, time_now] + [0]*10
    new_row[3] = 1  # first week = 1

    ws.append(new_row)
    wb.save(excel_file)

    print(f"{name} added and marked present")