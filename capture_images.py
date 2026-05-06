"""
capture_images.py
-----------------
Run this script BEFORE faces-train.py to collect training photos
directly from your webcam.

Usage:
    python capture_images.py

It will ask for the person's name, then capture 30 photos automatically.
Repeat for each person you want to recognise.
"""

import cv2
import os
import time

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

CASCADE_PATH = os.path.join(BASE_DIR, "cascades", "data", "haarcascade_frontalface_alt2.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ---- How many photos per person ----
NUM_PHOTOS   = 30
DELAY_SECS   = 0.3      # seconds between captures

name = input("Enter person's name (no spaces): ").strip().replace(" ", "-").lower()
if not name:
    print("No name entered. Exiting.")
    exit()

save_dir = os.path.join(IMAGE_DIR, name)
os.makedirs(save_dir, exist_ok=True)

# ---- Find how many photos already exist ----
existing = len([f for f in os.listdir(save_dir) if f.endswith((".jpg", ".png"))])
count    = existing

# ---- Camera ----
cap = None
for i in range(5):
    t = cv2.VideoCapture(i)
    if t.isOpened():
        cap = t
        break
    t.release()

if cap is None:
    print("ERROR: No camera found.")
    exit()

print(f"\nCapturing {NUM_PHOTOS} photos for '{name}'.")
print("Look at the camera. Press SPACE to start, Q to quit early.\n")

capturing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    status = f"Captured: {count - existing}/{NUM_PHOTOS}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    if not capturing:
        cv2.putText(frame, "Press SPACE to start", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.imshow(f"Capturing: {name}", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord(' '):
        capturing = True

    if capturing and len(faces) > 0:
        fname = os.path.join(save_dir, f"{name}_{count:04d}.jpg")
        cv2.imwrite(fname, frame)
        count += 1
        print(f"  Saved {fname}")
        time.sleep(DELAY_SECS)

        if count - existing >= NUM_PHOTOS:
            print(f"\nDone! {NUM_PHOTOS} photos saved to {save_dir}")
            break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal photos for '{name}': {count}")
print("Now run:  python faces-train.py")