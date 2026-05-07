import cv2
import os
import time
import sys

# ─────────────────────────────
# PATHS
# ─────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

CASCADE_PATH = os.path.join(
    BASE_DIR, "cascades", "data", "haarcascade_frontalface_alt2.xml"
)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("ERROR: Cascade not loaded properly.")
    exit()

# ─────────────────────────────
# SETTINGS
# ─────────────────────────────
NUM_PHOTOS = 30
DELAY_SECS = 0.25

# ─────────────────────────────
# INPUT (GUI OR MANUAL)
# ─────────────────────────────
if len(sys.argv) >= 3:
    person_id = sys.argv[1].strip()
    person_name = sys.argv[2].strip()
else:
    person_id = input("Enter Student ID: ").strip()
    person_name = input("Enter Full Name: ").strip()

if not person_id or not person_name:
    print("ERROR: Student ID or Name cannot be empty")
    exit()

# Keep the name exactly as entered (no lowercase conversion)

# ─────────────────────────────
# FIXED FOLDER STRUCTURE
# ─────────────────────────────
# IMPORTANT: ID + NAME TOGETHER (CONSISTENT WITH TRAINING)
folder_name = f"{person_id}_{person_name}"
save_dir = os.path.join(IMAGE_DIR, folder_name)
os.makedirs(save_dir, exist_ok=True)

# existing images count
existing_images = len([
    f for f in os.listdir(save_dir)
    if f.endswith((".jpg", ".png"))
])

count = existing_images

# ─────────────────────────────
# CAMERA SETUP
# ─────────────────────────────
cap = None

for i in range(3):
    temp = cv2.VideoCapture(i)
    if temp.isOpened():
        cap = temp
        print(f"[INFO] Camera opened at index {i}")
        break
    temp.release()

if cap is None:
    print("ERROR: No camera found")
    exit()

# ─────────────────────────────
# INFO
# ─────────────────────────────
print("\n==============================")
print(f"Capturing {NUM_PHOTOS} photos")
print(f"ID   : {person_id}")
print(f"NAME : {person_name}")
print("==============================")
print("Press SPACE to start | Q to quit\n")

capturing = False

# ─────────────────────────────
# MAIN LOOP
# ─────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # draw faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # UI text
    cv2.putText(
        frame,
        f"Captured: {count - existing_images}/{NUM_PHOTOS}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if not capturing:
        cv2.putText(
            frame,
            "Press SPACE to start capturing",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2
        )

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord(' '):
        capturing = True
        print("[INFO] Capturing started...")

    # ─────────────────────────────
    # SAVE IMAGE ONLY IF FACE EXISTS
    # ─────────────────────────────
    if capturing and len(faces) > 0:
        filename = os.path.join(
            save_dir,
            f"{person_name}_{count:04d}.jpg"
        )

        cv2.imwrite(filename, frame)
        print(f"[SAVED] {filename}")

        count += 1
        time.sleep(DELAY_SECS)

        if (count - existing_images) >= NUM_PHOTOS:
            print("[DONE] Required photos captured.")
            break

# ─────────────────────────────
# CLEANUP
# ─────────────────────────────
cap.release()
cv2.destroyAllWindows()

print("\nDONE!")
print(f"Saved under: {save_dir}")
print(f"ID={person_id}, Name={person_name}")
print("Next step: python faces-train.py")