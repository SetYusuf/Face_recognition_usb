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

if face_cascade.empty():
    print("ERROR: Face cascade failed to load.")
    exit()

# ----------------- RECOGNIZER -----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(BASE_DIR, "trainer.yml"))

with open(os.path.join(BASE_DIR, "labels.pickle"), 'rb') as f:
    labels = {v: k for k, v in pickle.load(f).items()}

print("Loaded people:", list(labels.values()))

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

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ----------------- ONE-TIME ATTENDANCE CONTROL -----------------
marked_today = set()

# ----------------- MAIN LOOP -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame")
        break

    frame = cv2.resize(frame, (640, 480))
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        roi_gray_raw  = gray[y:y+h, x:x+w]
        roi_color     = frame[y:y+h, x:x+w]

        roi_for_recog = cv2.resize(roi_gray_raw, (200, 200))
        roi_for_recog = cv2.equalizeHist(roi_for_recog)

        id_, conf = recognizer.predict(roi_for_recog)

        CONFIDENCE_THRESHOLD = 60

        if conf < CONFIDENCE_THRESHOLD:
            name = labels.get(id_, "Unknown").split("_")[0].capitalize()

            if name not in marked_today:
                try:
                    mark_attendance(id_, name)
                    marked_today.add(name)
                    print(f"Attendance marked: {name}")
                except Exception as e:
                    print(f"ERROR marking attendance: {e}")

            label_text = f"{name} ({int(conf)})"
            box_color  = (0, 200, 0)

        elif conf < 85:
            name       = labels.get(id_, "Unknown").split("_")[0].capitalize()
            label_text = f"Maybe {name}? ({int(conf)})"
            box_color  = (0, 200, 255)

        else:
            label_text = f"Unknown ({int(conf)})"
            box_color  = (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - 12), (x + tw + 6, y), box_color, -1)
        cv2.putText(
            frame, label_text, (x + 3, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Eye detection
        eyes = eye_cascade.detectMultiScale(
            roi_gray_raw, scaleFactor=1.1,
            minNeighbors=5, minSize=(20, 20)
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)

    # HUD
    cv2.putText(frame, f"Marked today: {len(marked_today)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "Press Q to quit", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("Face Recognition — Attendance", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nSession ended. Marked {len(marked_today)} people: {marked_today}")