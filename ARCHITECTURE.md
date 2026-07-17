# Architecture — Face Recognition & AI Vision Assistant

## Overview

This project is a **dual-purpose computer vision system** running on Windows:

1. **Face Recognition Attendance System** — Detects known faces via webcam, identifies them using LBPH (Local Binary Patterns Histograms), and logs attendance to an Excel spreadsheet.
2. **AI Vision Assistant** — An automated "point-and-describe" assistant that watches the webcam feed, detects motion + stillness, and sends captured frames to an Ollama vision language model (VLM) for description. Also monitors blink rate via MediaPipe FaceMesh to detect tiredness.

Both systems share the same camera infrastructure but serve different use cases. A Tkinter GUI (`GUI.py`) provides a unified launcher.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER (via Webcam)                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Camera Layer                                  │
│  cv2.VideoCapture (indices 0-4, CAP_DSHOW, 1280×720 @ 30fps)        │
└──────────┬───────────────────────────────────┬───────────────────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────────┐    ┌──────────────────────────────────────────┐
│  Face Recognition    │    │  AI Vision Assistant                      │
│  (Face_Recognition.py)│   │  (ai_vision_assistant.py)                 │
│                      │    │                                          │
│  Haar Cascade        │    │  ┌──────────────┐  ┌──────────────────┐  │
│  → detect faces      │    │  │ Motion       │  │ BlinkDetector    │  │
│  LBPH Recognizer     │    │  │ Detection    │  │ (blink_detector  │  │
│  → identify person   │    │  │ (frame diff) │  │  .py)            │  │
│  attendance.py       │    │  └──────┬───────┘  └────────┬─────────┘  │
│  → log to Excel      │    │         │                   │            │
└──────────────────────┘    │         ▼                   ▼            │
                            │  ┌──────────────────────────────────┐    │
                            │  │  Ollama Vision Client            │    │
                            │  │  (ollama_vision_client.py)       │    │
                            │  │  → POST image + prompt to Ollama │    │
                            │  │  → Receive text description      │    │
                            │  └──────────────┬───────────────────┘    │
                            │                 │                        │
                            │                 ▼                        │
                            │  ┌──────────────────────────────────┐    │
                            │  │  Terminal Output                 │    │
                            │  │  [AI] <description>              │    │
                            │  │  [BLINK] tiredness alert         │    │
                            │  └──────────────────────────────────┘    │
                            └──────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Configuration Layer                           │
│  config_ai.py — all thresholds, URLs, model names                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### 1. `config_ai.py` — Central Configuration

All tunable constants in one place. No hardcoded magic numbers in implementation files.

| Constant | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5vl:7b` | Vision language model |
| `EAR_THRESHOLD` | `0.21` | Eye Aspect Ratio below which eye is "closed" |
| `EAR_CONSEC_FRAMES` | `2` | Consecutive low-EAR frames to confirm a blink |
| `BLINK_RATE_WINDOW_SEC` | `60` | Rolling window (seconds) for blink rate |
| `TIRED_BLINK_RATE` | `25` | Blinks/minute threshold for tiredness alert |
| `TIRED_COOLDOWN_SEC` | `20` | Min seconds between tiredness alerts |
| `MOTION_THRESHOLD` | `3000` | Pixel-diff count that counts as "something moved" |
| `HOLD_STILL_FRAMES` | `15` | Consecutive still frames before triggering AI (~0.5s) |
| `TRIGGER_COOLDOWN_SEC` | `6` | Min seconds between AI calls (prevents spam) |

---

### 2. `blink_detector.py` — BlinkDetector Class

**Purpose:** Detect eye blinks in real-time using MediaPipe FaceMesh landmarks and the Eye Aspect Ratio (EAR) formula.

**Key Algorithm — Eye Aspect Ratio:**

```
EAR = (|p2 - p6| + |p3 - p5|) / (2 × |p1 - p4|)

Where p1-p6 are 6 specific FaceMesh landmarks per eye:
  LEFT_EYE  = [33, 160, 158, 133, 153, 144]
  RIGHT_EYE = [362, 385, 387, 263, 373, 380]
```

- EAR drops when the eye closes (vertical distance shrinks).
- A blink is registered when EAR stays below `EAR_THRESHOLD` for `EAR_CONSEC_FRAMES` consecutive frames, then rises again.
- Blink timestamps are stored in a rolling deque with a `BLINK_RATE_WINDOW_SEC` window.
- Tiredness is flagged when the blink count in the window exceeds `TIRED_BLINK_RATE`.

**State Machine:**

```
                    EAR < threshold
    ┌──────────────────────────────┐
    │                              ▼
  IDLE ──→ consec_low++ ──→ consec_low >= EAR_CONSEC_FRAMES
    ▲                              │
    │                              ▼
    └────── EAR rises ────── BLINK DETECTED (timestamp logged)
```

**Public API:**
- `process(frame_bgr) → (blinked: bool, tired: bool)` — Process a single frame.
- `close()` — Release FaceMesh resources.

**Dependencies:** `mediapipe`, `cv2`, `config_ai`

---

### 3. `ollama_vision_client.py` — Ollama VLM Client

**Purpose:** Send an image frame to a locally running Ollama vision model and receive a text description.

**Flow:**
1. Encode BGR frame as JPEG → base64 string.
2. POST to `OLLAMA_URL` with model name, prompt, and base64 image.
3. Parse JSON response and return the `"response"` field.

**Functions:**
- `ask_about_frame(frame_bgr, prompt=None) → str` — Describe the scene/object.
- `ask_for_recipe(frame_bgr) → str` — Identify food/ingredients and suggest a recipe.

**Error Handling:** Wraps network errors and exceptions, returning `"(AI unavailable: ...)"` or `"(AI error: ...)"` strings.

**Dependencies:** `requests`, `PIL`, `cv2`, `config_ai`

---

### 4. `ai_vision_assistant.py` — Main AI Vision Loop

**Purpose:** The primary entry point for the AI Vision Assistant. Opens a camera, runs motion detection + blink detection in a loop, and automatically triggers AI descriptions.

**Architecture — Main Loop:**

```
while True:
    frame = cap.read()
    
    # 1. Blink detection (runs every frame)
    blinked, tired = detector.process(frame)
    if tired and cooldown_ok:
        print("[BLINK] ...")
    
    # 2. Motion detection (frame differencing)
    gray = cv2.cvtColor(frame, GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)
    diff = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(diff, 25, 255, BINARY)
    motion_pixels = cv2.countNonZero(thresh)
    
    if motion_pixels > MOTION_THRESHOLD:
        still_counter = 0          # something moved → reset
    else:
        still_counter += 1         # holding still → increment
        if still_counter >= HOLD_STILL_FRAMES and not ai_busy:
            if cooldown_ok:
                # Trigger AI in background thread
                threading.Thread(target=_ai_worker, args=(frame_copy,)).start()
                still_counter = 0
    
    # 3. Status overlay ("watching..." / "thinking...")
    # 4. cv2.imshow() + waitKey(Q to quit)
```

**Key Design Decisions:**
- **Background threading:** AI calls run in a `daemon=True` thread so the video loop never blocks.
- **Cooldown:** `TRIGGER_COOLDOWN_SEC` prevents repeated triggers while an object sits in frame.
- **No TTS:** All output goes to terminal via `print()`.
- **No keypress triggers:** Detection is fully automatic based on motion + stillness.

**State Variables:**
- `ai_busy: bool` — Prevents concurrent AI calls.
- `last_trigger_time: float` — Cooldown timer.
- `still_counter: int` — Consecutive frames with no motion.
- `prev_gray: ndarray` — Previous grayscale frame for differencing.
- `last_tired_print: float` — Cooldown for tiredness terminal prints.

**Dependencies:** `cv2`, `numpy`, `threading`, `time`, `blink_detector`, `ollama_vision_client`, `config_ai`

---

### 5. `Face_Recognition.py` — Live Face Recognition & Attendance

**Purpose:** Real-time face detection + LBPH-based recognition + automatic attendance marking.

**Pipeline per frame:**

```
1. Resize frame to 50% for faster detection
2. Haar Cascade detectMultiScale → raw face bounding boxes
3. NMS (Non-Maximum Suppression) → remove overlapping duplicates
4. For each face:
   a. Extract ROI, preprocess (resize 200×200, blur, equalizeHist, CLAHE)
   b. LBPH recognizer.predict(roi) → (id, confidence)
   c. Temporal smoothing via grid-based sliding window (7 frames)
   d. Vote on identity → best_id, avg_conf
   e. If confident (conf < 55) and stable:
      - Draw green box + name
      - Increment confirmation streak
      - If streak >= 4 frames and not marked today:
        → mark_attendance(id, name) → Excel
   f. If uncertain (conf < 80): amber "Maybe {name}?"
   g. Else: red "Unknown"
5. Draw HUD (face count, marked count)
6. cv2.imshow() + waitKey(Q to quit)
```

**Key Algorithms:**
- **NMS (Non-Maximum Suppression):** Sorts faces by area (largest first), iteratively removes boxes with IoU > 0.35.
- **Temporal Smoothing:** Tracks face identity per grid cell (80×80 px). Accumulates predictions over 7 frames, picks the ID with the lowest average confidence.
- **CLAHE Preprocessing:** Contrast Limited Adaptive Histogram Equalization for robust recognition under varying lighting.

**Attendance Flow:**
- `confirmed_streak[face_id]` increments each frame the face is confidently identified.
- When streak ≥ `MARK_FRAMES` (4) and `face_id not in marked_today`:
  - Calls `attendance.mark_attendance(face_id, name)`
  - Adds to `marked_today` set (prevents duplicate marks per session).

**Dependencies:** `cv2`, `numpy`, `pickle`, `collections`, `attendance`

---

### 6. `faces-train.py` — LBPH Model Training

**Purpose:** Train the LBPH face recognizer from labelled images in the `images/` directory.

**Data Format:**
```
images/
├── 1_john doe/
│   ├── john doe_0000.jpg
│   ├── john doe_0001.jpg
│   └── ...
├── 2_jane smith/
│   ├── jane smith_0000.jpg
│   └── ...
```

**Training Pipeline:**
1. Walk `images/` directory.
2. Parse folder name as `{student_id}_{student_name}`.
3. For each image:
   - Load as grayscale via PIL.
   - Detect face with Haar Cascade.
   - Resize ROI to 200×200.
   - Add original + horizontally flipped version (data augmentation).
4. Train LBPH recognizer with all samples.
5. Save:
   - `trainer.yml` — LBPH model parameters.
   - `labels.pickle` — `{student_id: student_name}` mapping.

**Dependencies:** `cv2`, `numpy`, `PIL`, `pickle`

---

### 7. `capture_images.py` — Image Capture Utility

**Purpose:** Capture 30 face photos for a new student registration.

**Flow:**
1. Accept student ID + name (CLI args or interactive input).
2. Create folder `images/{id}_{name}/`.
3. Open camera, show live feed with face detection overlay.
4. Press SPACE to start capturing.
5. For each frame with a detected face:
   - Save full frame as JPEG.
   - Wait 0.25s between captures.
   - Stop after 30 new photos.
6. Press Q to quit early.

**Dependencies:** `cv2`, `os`, `time`

---

### 8. `attendance.py` — Excel Attendance Logger

**Purpose:** Log student attendance to an Excel spreadsheet with formatted headers and styling.

**Excel Output:**
```
| ID | Name      | Date       | Time     | Status  |
|----|-----------|------------|----------|---------|
| 1  | john doe  | 2025-07-10 | 09:03:21 | Present |
```

**Key Logic:**
- Checks if the student ID is already recorded today → updates time only (no duplicate rows).
- If not recorded today → appends a new row.
- Uses `openpyxl` for Excel I/O with custom styling (navy header, alternating row colours, borders).
- Resolves exact registered name from `labels.pickle` (fallback to `display_name` parameter).

**Dependencies:** `openpyxl`, `pickle`, `datetime`, `os`

---

### 9. `GUI.py` — Tkinter Launcher

**Purpose:** Unified graphical interface for the entire system.

**Layout:**
- Navy header with crest + title.
- Gold accent stripe.
- Workflow breadcrumb: Register → Train → Detect → Records → AI Assistant.
- Five clickable action rows with hover effects.
- System log panel (monospace, green text).
- Footer with workflow summary.

**Actions:**
| Button | Action |
|--------|--------|
| Register Student | Opens `RegisterWindow` → runs `capture_images.py` |
| Train Model | Runs `faces-train.py` with live log output |
| Start Detection | Launches `Face_Recognition.py` in subprocess |
| View Attendance | Opens `AttendanceWindow` (reads Excel file) |
| AI Assistant | Launches `ai_vision_assistant.py` in subprocess |

**Dependencies:** `tkinter`, `subprocess`, `openpyxl`, `os`

---

## Data Flow Diagrams

### Face Recognition Attendance Flow

```
Camera Frame
    │
    ▼
Haar Cascade (face detection)
    │
    ▼
NMS (remove overlapping boxes)
    │
    ▼
For each face:
    ├── ROI extraction + CLAHE preprocessing
    ├── LBPH Recognizer → (id, confidence)
    ├── Temporal smoothing (7-frame sliding window)
    ├── Confidence classification:
    │   ├── conf < 55  → "Known" → increment streak
    │   ├── conf < 80  → "Maybe" → reset streak
    │   └── conf ≥ 80  → "Unknown"
    └── If streak ≥ 4 and not marked today:
        └── attendance.mark_attendance(id, name)
            └── openpyxl → Excel file
```

### AI Vision Assistant Flow

```
Camera Frame
    │
    ├──► BlinkDetector.process(frame)
    │       ├── MediaPipe FaceMesh → 468 landmarks
    │       ├── Compute EAR for both eyes
    │       ├── Detect blink transition
    │       └── If tired → print [BLINK] alert
    │
    └──► Motion Detection
            ├── Convert to grayscale + GaussianBlur
            ├── cv2.absdiff(prev, current)
            ├── Threshold + countNonZero
            ├── If motion > MOTION_THRESHOLD:
            │   └── still_counter = 0
            ├── If motion ≤ MOTION_THRESHOLD:
            │   └── still_counter++
            │       └── If still_counter ≥ HOLD_STILL_FRAMES
            │           AND not ai_busy AND cooldown OK:
            │           ├── Capture frame
            │           ├── ai_busy = True
            │           └── Thread: ask_about_frame(capture)
            │               └── print [AI] response
            │               └── ai_busy = False
            │
            └── Update status overlay ("watching..." / "thinking...")
```

---

## Threading Model

```
┌─────────────────────────────────────────────────────────┐
│                    Main Thread                           │
│  - Camera loop (cap.read, process, imshow, waitKey)     │
│  - Blink detection (synchronous, fast)                  │
│  - Motion detection (synchronous, fast)                 │
│  - Status overlay drawing                               │
│  - Key handling (Q to quit)                             │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │  (triggered by stillness) │
        ▼                           ▼
┌──────────────────┐    ┌──────────────────────────────┐
│  AI Worker Thread │    │  (daemon=True, auto-cleaned) │
│  - ask_about_frame│    │                              │
│  - blocks 1-5 sec │    │  Only one active at a time  │
│  - print [AI]     │    │  (guarded by ai_busy flag)  │
│  - set ai_busy=F  │    │                              │
└──────────────────┘    └──────────────────────────────┘
```

**Why background threads?** Ollama API calls take 1-5 seconds. Running them synchronously in the camera loop would freeze the video feed. The `ai_busy` boolean prevents concurrent calls, and the cooldown timer prevents rapid re-triggering.

---

## Configuration & Tuning

### Motion Detection Sensitivity

- `MOTION_THRESHOLD = 3000` — Lower = more sensitive (detects smaller movements). Increase if the system triggers on noise/lighting changes.
- `HOLD_STILL_FRAMES = 15` — At ~30fps, this is ~0.5 seconds. Increase for longer stillness requirement.

### Blink Detection

- `EAR_THRESHOLD = 0.21` — Typical values: 0.18-0.25. Lower = harder to trigger (only counts fully closed eyes).
- `EAR_CONSEC_FRAMES = 2` — At 30fps, a blink lasts ~3-5 frames. 2 frames prevents noise from counting as a blink.

### Tiredness

- `TIRED_BLINK_RATE = 25` — Normal blink rate is 15-20 blinks/min. 25+ indicates possible fatigue.
- `TIRED_COOLDOWN_SEC = 20` — Prevents repeated alerts.

---

## Dependencies

| Package | Version | Used By |
|---------|---------|---------|
| `opencv-python` | ≥4.5 | All vision modules |
| `opencv-contrib-python` | ≥4.5 | LBPH face recognizer |
| `mediapipe` | ≥0.10 | BlinkDetector (FaceMesh) |
| `numpy` | ≥1.21 | Array operations everywhere |
| `Pillow` | ≥8.0 | Image loading in training |
| `requests` | ≥2.25 | Ollama API calls |
| `openpyxl` | ≥3.0 | Excel attendance logging |
| `matplotlib` | ≥3.5 | MediaPipe dependency |
| `pyttsx3` | (removed) | Was used for TTS, no longer needed |

---

## File Structure

```
Face_recognition_usb/
│
├── ai_vision_assistant.py    # Main AI Vision Assistant loop
├── blink_detector.py         # BlinkDetector class (MediaPipe + EAR)
├── config_ai.py              # All thresholds and settings
├── ollama_vision_client.py   # Ollama VLM query helper
├── Face_Recognition.py       # Live face recognition + attendance
├── faces-train.py            # LBPH model training
├── GUI.py                    # Tkinter launcher
├── capture_images.py         # Face photo capture utility
├── attendance.py             # Excel attendance logger
├── test.py                   # Test / debug script
├── requirements.txt          # Python dependencies
├── README.md                 # User documentation
├── ARCHITECTURE.md           # This file
│
├── images/                   # Training images
│   ├── 1_john doe/
│   ├── 2_jane smith/
│   └── ...
│
├── cascades/                 # Haar Cascade XML classifiers
│   ├── data/
│   │   ├── haarcascade_frontalface_alt2.xml
│   │   ├── haarcascade_eye.xml
│   │   └── haarcascade_smile.xml
│   └── third-party/
│
├── ExcelData/                # Attendance Excel output
│   └── attendance.xlsx
│
├── trainer.yml               # Trained LBPH model
├── labels.pickle             # {student_id: name} mapping
└── images/Thumbs.db          # Windows thumbnail cache
```

---

## Error Handling Strategy

| Scenario | Handling |
|----------|----------|
| Camera not found | Scans indices 0-4, prints error and exits if none found |
| Frame read failure | Prints warning and breaks the loop |
| Ollama unavailable | Returns `"(AI unavailable: ...)"` string, no crash |
| AI call exception | Returns `"(AI error: ...)"` string, resets `ai_busy` |
| Excel file missing | Creates new workbook with headers on first run |
| Corrupted training image | Skips with warning, continues training |
| No faces in training data | Prints error and exits before training |

---

## Performance Considerations

- **Face detection** runs at 50% scale (640×360) for speed, then coordinates are scaled back to full resolution.
- **Motion detection** uses a simple `absdiff` + threshold — extremely fast (<1ms per frame).
- **Blink detection** uses MediaPipe which runs at ~30fps on a modern CPU with GPU acceleration.
- **AI calls** are the bottleneck (1-5 seconds). Background threading prevents video stuttering.
- **CLAHE preprocessing** adds ~2ms per face but significantly improves recognition accuracy under varied lighting.

---

## Future Considerations

- Replace Haar Cascade with MediaPipe Face Detection for better accuracy and angle tolerance.
- Add face tracking (e.g., CentroidTracker) to maintain identity across frames without re-detecting.
- Use a dedicated blink detection model instead of EAR heuristic for better accuracy.
- Add a web dashboard (Flask/FastAPI) for remote attendance monitoring.
- Support multiple cameras / IP cameras (RTSP streams).