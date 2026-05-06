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
    print(f"ERROR: Could not load cascade from {CASCADE_PATH}")
    exit()

# ---------- LBPH RECOGNIZER ----------
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

current_id = 0
label_ids  = {}
x_train    = []
y_labels   = []

# ---------- PROCESS DATASET ----------
for root, dirs, files in os.walk(IMAGE_DIR):

    # Skip images sitting directly in images/ (no person subfolder)
    if os.path.abspath(root) == os.path.abspath(IMAGE_DIR):
        continue

    for file in files:
        if not file.lower().endswith(("png", "jpg", "jpeg")):
            continue

        path  = os.path.join(root, file)
        label = os.path.basename(root).replace(" ", "-").lower()

        if label not in label_ids:
            label_ids[label] = current_id
            current_id += 1

        id_ = label_ids[label]

        # ---------- LOAD IMAGE AS GRAYSCALE ----------
        pil_image   = Image.open(path).convert("L")
        image_array = np.array(pil_image, "uint8")

        # ---------- FACE DETECTION ----------
        faces = face_cascade.detectMultiScale(
            image_array,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            print(f"  WARNING: No face found in {path} — skipping")
            continue

        for (x, y, w, h) in faces:
            roi = image_array[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            # Resize THEN normalize
            roi = cv2.resize(roi, (200, 200))
            roi = cv2.equalizeHist(roi)

            # ---------- AUGMENTATION ----------
            # Original
            x_train.append(roi)
            y_labels.append(id_)

            # Horizontal flip
            x_train.append(cv2.flip(roi, 1))
            y_labels.append(id_)

            # Slight brightness boost
            bright = cv2.convertScaleAbs(roi, alpha=1.1, beta=15)
            x_train.append(bright)
            y_labels.append(id_)

            # Slight brightness reduction
            dark = cv2.convertScaleAbs(roi, alpha=0.9, beta=-15)
            x_train.append(dark)
            y_labels.append(id_)

            # Flipped + bright
            x_train.append(cv2.flip(bright, 1))
            y_labels.append(id_)

            # Flipped + dark
            x_train.append(cv2.flip(dark, 1))
            y_labels.append(id_)

            print(f"  + {label} [{id_}]  ->  {file}  (6 samples added)")

# ---------- GUARD ----------
if len(x_train) == 0:
    print("\nERROR: No faces found. Check your dataset folder structure:")
    print("  images/")
    print("    PersonName/")
    print("      photo1.jpg")
    print("      photo2.jpg  ...")
    exit()

# ---------- SAVE LABELS ----------
with open(LABELS_PATH, 'wb') as f:
    pickle.dump(label_ids, f)

print(f"\nLabels saved  ->  {LABELS_PATH}")

# ---------- TRAIN ----------
recognizer.train(x_train, np.array(y_labels))
recognizer.save(TRAINER_PATH)

print(f"Model saved   ->  {TRAINER_PATH}")
print(f"\n{'='*40}")
print(f"  Training complete!")
print(f"  Total samples : {len(x_train)}")
print(f"  Total people  : {len(label_ids)}")
print(f"  Per person    : {len(x_train) // max(len(label_ids),1)} avg")
print(f"{'='*40}")
print("\nTIP: For best accuracy, add 15-30 REAL photos per person")
print("     (different angles, lighting, expressions)")