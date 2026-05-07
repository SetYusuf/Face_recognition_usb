import cv2
import os
import pickle
import numpy as np
from collections import defaultdict, deque
from attendance import mark_attendance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
#  CASCADES
# ─────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "cascades", "data", "haarcascade_frontalface_alt2.xml")
)
eye_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "cascades", "data", "haarcascade_eye.xml")
)

if face_cascade.empty():
    print("ERROR: Face cascade failed to load.")
    exit()

# ─────────────────────────────────────────────
#  RECOGNIZER + LABELS
# ─────────────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(BASE_DIR, "trainer.yml"))

# labels.pickle now stores {student_id: name} directly
# e.g., {1: "john doe", 2: "jane smith"}
with open(os.path.join(BASE_DIR, "labels.pickle"), "rb") as f:
    labels = pickle.load(f)

print("Loaded people:", list(labels.values()))

# ─────────────────────────────────────────────
#  TEMPORAL SMOOTHING  (sliding window per face region)
#  Key  : grid cell (cx//GRID, cy//GRID)
#  Value: deque of (id_, conf) predictions
# ─────────────────────────────────────────────
SMOOTH_FRAMES  = 7        # frames to accumulate before finalising a label
GRID           = 80       # pixels per grid cell for identity tracking
face_history: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=SMOOTH_FRAMES))

def grid_key(x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    return (cx // GRID, cy // GRID)

def smooth_prediction(key, id_, conf):
    """Push latest prediction; return (best_id, avg_conf, is_stable)."""
    face_history[key].append((id_, conf))
    buf = face_history[key]

    # Vote on identity
    id_votes = defaultdict(list)
    for pid, pc in buf:
        id_votes[pid].append(pc)

    best_id   = min(id_votes, key=lambda k: np.mean(id_votes[k]))   # lowest avg conf = best
    avg_conf  = np.mean(id_votes[best_id])
    is_stable = len(buf) >= SMOOTH_FRAMES // 2

    return best_id, avg_conf, is_stable

# ─────────────────────────────────────────────
#  NMS – remove overlapping duplicate detections
# ─────────────────────────────────────────────
def nms_faces(faces, overlap_thresh=0.35):
    if len(faces) == 0:
        return faces
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in faces], dtype=float)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]          # largest first

    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < overlap_thresh]

    return [faces[k] for k in keep]

# ─────────────────────────────────────────────
#  PREPROCESSING  – improved ROI quality
# ─────────────────────────────────────────────
def preprocess_roi(roi_gray):
    roi = cv2.resize(roi_gray, (200, 200))
    roi = cv2.GaussianBlur(roi, (3, 3), 0)           # mild denoise
    roi = cv2.equalizeHist(roi)                       # contrast normalise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply(roi)                            # local contrast
    return roi

# ─────────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────────
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

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)   # higher res → more face pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
cap.set(cv2.CAP_PROP_FPS,           30)

# ─────────────────────────────────────────────
#  THRESHOLDS
# ─────────────────────────────────────────────
CONF_SURE    = 55    # confident match
CONF_MAYBE   = 80    # uncertain match
MARK_FRAMES  = 4     # stable frames before marking attendance

# Track by face_id (int) — matches exactly what attendance.py uses as the key
confirmed_streak: dict[int, int] = defaultdict(int)
marked_today: set[int]           = set()   # stores face_id, not name

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame")
        break

    frame_idx += 1

    # Work at reduced size for detection speed; draw on full frame
    scale       = 0.5
    small       = cv2.resize(frame, None, fx=scale, fy=scale)
    gray_small  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_full   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Detect faces on the small frame ──────────────────────────
    raw_faces = face_cascade.detectMultiScale(
        gray_small,
        scaleFactor  = 1.05,    # finer scale steps  → catch more faces
        minNeighbors = 4,       # slightly relaxed   → detect partial faces
        minSize      = (50, 50),
        flags        = cv2.CASCADE_SCALE_IMAGE,
    )

    # Scale coords back to full resolution
    if len(raw_faces):
        raw_faces = [(int(x/scale), int(y/scale),
                      int(w/scale), int(h/scale)) for (x,y,w,h) in raw_faces]
    else:
        raw_faces = []

    # NMS to drop overlapping duplicates
    faces = nms_faces(raw_faces)

    active_keys = set()

    for (x, y, w, h) in faces:
        # Clamp to frame bounds
        x, y = max(0, x), max(0, y)
        w  = min(w, frame.shape[1] - x)
        h  = min(h, frame.shape[0] - y)

        roi_gray  = gray_full[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        roi_proc  = preprocess_roi(roi_gray)
        id_, conf = recognizer.predict(roi_proc)

        key = grid_key(x, y, w, h)
        active_keys.add(key)

        best_id, avg_conf, is_stable = smooth_prediction(key, id_, conf)

        # ── Classify ─────────────────────────────────────────────
        if avg_conf < CONF_SURE and is_stable:
            # Exact registered name — no split/capitalize mangling
            exact_name = labels.get(best_id, "Unknown")
            label_text = f"{exact_name}  {int(avg_conf)}"
            box_color  = (0, 210, 0)

            # Streak + duplicate guard keyed on face_id (int)
            confirmed_streak[best_id] += 1
            if confirmed_streak[best_id] >= MARK_FRAMES and best_id not in marked_today:
                try:
                    mark_attendance(best_id, exact_name)
                    marked_today.add(best_id)
                    print(f"✔ Attendance marked: ID={best_id}  Name={exact_name}")
                except Exception as e:
                    print(f"ERROR marking attendance: {e}")

        elif avg_conf < CONF_MAYBE:
            exact_name = labels.get(best_id, "Unknown")
            label_text = f"Maybe {exact_name}?  {int(avg_conf)}"
            box_color  = (0, 180, 255)
            confirmed_streak[best_id] = 0   # reset – not confident enough

        else:
            label_text = f"Unknown  {int(avg_conf)}"
            box_color  = (0, 0, 220)

        # ── Bounding box ─────────────────────────────────────────
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 8, y), box_color, -1)
        cv2.putText(frame, label_text, (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # ── Eye detection (optional cosmetic) ────────────────────
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(20, 20)
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (180, 255, 0), 1)

    # ── Expire stale history cells ────────────────────────────────
    stale = [k for k in face_history if k not in active_keys]
    for k in stale:
        del face_history[k]

    # ── HUD ──────────────────────────────────────────────────────
    hud_lines = [
        f"Faces detected : {len(faces)}",
        f"Marked today   : {len(marked_today)}",
        "Q  quit",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(frame, line, (12, 28 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Face Recognition — Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
marked_names = [labels.get(fid, str(fid)) for fid in marked_today]
print(f"\nSession ended. Marked {len(marked_today)} people: {marked_names}")