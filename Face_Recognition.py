import cv2
import os
import pickle
from attendance import mark_attendance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------- CASCADES -----------------
face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "cascades", "data", "haarcascade_frontalface_alt2.xml")
)
eye_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "cascades", "data", "haarcascade_eye.xml")
)

# ----------------- RECOGNIZER -----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(BASE_DIR, "trainer.yml"))

with open(os.path.join(BASE_DIR, "labels.pickle"), 'rb') as f:
    labels = {v: k for k, v in pickle.load(f).items()}

# ----------------- AUTO CAMERA DETECTION -----------------
cap = None
for i in range(5):
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        cap = test_cap
        print(f"Connected to webcam at index {i}")
        break
    test_cap.release()

if cap is None:
    print("ERROR: No camera found.")
    exit()

# ----------------- ONE-TIME ATTENDANCE CONTROL -----------------
marked_today = set()

# ----------------- MAIN LOOP -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # ----------------- MUST MATCH TRAINING -----------------
        roi_gray = cv2.resize(roi_gray, (200, 200))
        roi_gray = cv2.equalizeHist(roi_gray)

        id_, conf = recognizer.predict(roi_gray)

        if conf < 85:
            name = labels[id_]

            # ----------------- ONLY ONCE PER RUN -----------------
            if name not in marked_today:
                mark_attendance(name)
                marked_today.add(name)

            text = f"{name} ({int(conf)})"
            color = (255, 255, 255)
        else:
            text = f"Unknown ({int(conf)})"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Eye detection (unchanged)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

    cv2.imshow("Face Recognition (DroidCam USB)", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()