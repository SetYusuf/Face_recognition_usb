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
LABELS_PATH = os.path.join(BASE_DIR, "labels.pickle")

# ---------- LOAD ----------
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# LBPH parameters tuned slightly better than default
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

current_id = 0
label_ids = {}
x_train = []
y_labels = []

# ---------- PROCESS DATASET ----------
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:

        if file.lower().endswith(("png", "jpg", "jpeg")):

            path = os.path.join(root, file)

            # Folder name = person name
            label = os.path.basename(root).replace(" ", "-").lower()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            # ---------- LOAD IMAGE ----------
            pil_image = Image.open(path).convert("L")  # grayscale
            image_array = np.array(pil_image, "uint8")

            # ---------- FACE DETECTION ----------
            faces = face_cascade.detectMultiScale(
                image_array,
                scaleFactor=1.3,      # better than 1.5 (more precise)
                minNeighbors=5,
                minSize=(80, 80)      # ignore tiny/noisy faces
            )

            # ---------- EXTRACT FACE ----------
            for (x, y, w, h) in faces:

                roi = image_array[y:y+h, x:x+w]

                # ---------- RESIZE AFTER DETECTION ----------
                roi = cv2.resize(roi, (200, 200))  # consistent size

                # ---------- NORMALIZE LIGHTING ----------
                roi = cv2.equalizeHist(roi)

                # ---------- SAVE TRAINING DATA ----------
                x_train.append(roi)
                y_labels.append(id_)

# ---------- SAVE LABELS ----------
with open(LABELS_PATH, 'wb') as f:
    pickle.dump(label_ids, f)

# ---------- TRAIN MODEL ----------
if len(x_train) == 0:
    print("No faces found. Check your dataset.")
    exit()

recognizer.train(x_train, np.array(y_labels))

# ---------- SAVE MODEL ----------
recognizer.save(TRAINER_PATH)

print("Training complete!")
print(f"Total faces trained: {len(x_train)}")
print(f"Total people: {len(label_ids)}")