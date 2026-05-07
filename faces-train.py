import cv2
import os
import numpy as np
from PIL import Image
import pickle

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

CASCADE_PATH = os.path.join(BASE_DIR, "cascades", "data", "haarcascade_frontalface_alt2.xml")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer.yml")
LABELS_PATH  = os.path.join(BASE_DIR, "labels.pickle")

# ---------- LOAD CASCADE ----------
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("ERROR: Cascade not loaded")
    exit()

# ---------- RECOGNIZER ----------
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Store mapping: student_id (int) -> name (str)
student_labels = {}

x_train = []
y_labels = []

# ---------- SCAN ALL IMAGES ----------
for root, dirs, files in os.walk(IMAGE_DIR):
    # Skip top-level directory
    if root == IMAGE_DIR:
        continue

    for file in files:
        if not file.lower().endswith(("png", "jpg", "jpeg")):
            continue

        path = os.path.join(root, file)

        # Parse folder name: "ID_name" format (e.g., "1_john doe")
        folder_name = os.path.basename(root)
        
        # Split on first underscore to get ID and name
        parts = folder_name.split("_", 1)
        if len(parts) == 2:
            student_id_str, student_name = parts
            try:
                student_id = int(student_id_str)
            except ValueError:
                print(f"WARNING: Invalid ID in folder '{folder_name}', skipping")
                continue
        else:
            # Fallback: use folder name as-is (no ID prefix)
            print(f"WARNING: Folder '{folder_name}' doesn't follow ID_name format, using folder name")
            student_id = hash(folder_name) % 10000  # Generate a numeric ID
            student_name = folder_name

        # Store the student ID -> name mapping
        if student_id not in student_labels:
            student_labels[student_id] = student_name

        id_ = student_id

        # ---------- LOAD IMAGE ----------
        try:
            pil_img = Image.open(path).convert("L")
            img = np.array(pil_img, "uint8")
        except:
            print("Skipping corrupted:", path)
            continue

        # ---------- DETECT FACE ----------
        faces = face_cascade.detectMultiScale(img, 1.2, 5)

        if len(faces) == 0:
            print("No face:", path)
            continue

        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            roi = cv2.resize(roi, (200, 200))

            # ---------- ADD DATA ----------
            x_train.append(roi)
            y_labels.append(id_)

            # simple augmentation (safe)
            x_train.append(cv2.flip(roi, 1))
            y_labels.append(id_)

            print(f"Loaded: {folder_name} -> {file}")

# ---------- CHECK ----------
if len(x_train) == 0:
    print("ERROR: No training data found!")
    exit()

# ---------- SAVE LABELS ----------
# Save student_id -> name mapping (e.g., {1: "john doe", 2: "jane smith"})
with open(LABELS_PATH, "wb") as f:
    pickle.dump(student_labels, f)

print("Labels saved:", student_labels)

# ---------- TRAIN ----------
recognizer.train(x_train, np.array(y_labels))
recognizer.save(TRAINER_PATH)

print("\n==========================")
print("TRAINING COMPLETE")
print("Samples:", len(x_train))
print("People :", len(student_labels))
print("==========================")
