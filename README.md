# Face Recognition

Recognises known faces in a live webcam feed using OpenCV's LBPH (Local Binary Patterns Histograms) face recogniser. A training script processes labelled face images to build a recognition model, and a separate script performs real-time face, eye, and smile detection with identity labelling via the webcam.

## Dataset

Custom face images organised in the `images/` folder with one subfolder per person. Haar Cascade XML classifiers for face, eye, and smile detection are stored in the `cascades/` folder.

## Tech Stack

- OpenCV (with `cv2.face` contrib module)
- NumPy
- Pillow
- pickle
- MediaPipe FaceMesh
- Ollama (vision language model)

## Installation

```bash
# Core dependencies for face recognition
pip install opencv-python opencv-contrib-python numpy Pillow openpyxl

# Dependencies for AI Vision Assistant
pip install mediapipe requests matplotlib

# Install Ollama (separate — see https://ollama.com)
# Then pull a vision model:
ollama pull qwen2.5vl:7b
```

## Face Recognition (Original)

1. Place training images in `images/`, with a subfolder for each person's name.
2. Update file paths in `faces-train.py` and `Face Recognition.py` to match your local setup.
3. Run `python faces-train.py` to train the LBPH recogniser and generate `trainner.yml` and `labels.pickle`.
4. Run `python "Face Recognition.py"` to start real-time face recognition via webcam. Press **q** to quit.

### GUI Alternative

```bash
python GUI.py
```

Launches a Tkinter interface for capturing training images, training the model, and running live face recognition.

## AI Vision Assistant

An automated "point-and-describe" assistant that watches your webcam feed and uses Ollama's vision model to describe objects held still in front of the camera.

### How It Works

1. The camera feed opens in an OpenCV window.
2. **Motion detection** tracks movement between frames using frame differencing.
3. When something enters the frame and **stays still** for ~0.5 seconds (15 frames at ~30 fps), an AI description is triggered automatically.
4. The AI analysis runs in a **background thread** — the video feed never freezes.
5. Descriptions print to the **terminal** as `[AI] <description>`.
6. **Blink/tiredness detection** (via MediaPipe FaceMesh + Eye Aspect Ratio) runs simultaneously and prints `[BLINK] ...` alerts to the terminal.
7. A small "watching..." / "thinking..." indicator appears in the corner of the video feed.

### Controls

| Key | Action |
|-----|--------|
| **Q** | Quit cleanly |

No manual triggers needed — detection is fully automatic based on motion and stillness.

### Run

```bash
# Make sure Ollama is running first (separate terminal):
ollama serve

# Then launch the assistant:
python ai_vision_assistant.py
```

### Configuration

All thresholds live in `config_ai.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `MOTION_THRESHOLD` | `3000` | Pixel-diff count that counts as "something moved" |
| `HOLD_STILL_FRAMES` | `15` | Consecutive still frames before triggering AI (~0.5 s) |
| `TRIGGER_COOLDOWN_SEC` | `6` | Min seconds between AI calls (prevents spam) |
| `EAR_THRESHOLD` | `0.21` | Eye Aspect Ratio threshold for blink detection |
| `EAR_CONSEC_FRAMES` | `2` | Consecutive low-EAR frames to confirm a blink |
| `BLINK_RATE_WINDOW_SEC` | `60` | Rolling window (seconds) for blink rate calculation |
| `TIRED_BLINK_RATE` | `25` | Blinks per minute threshold for tiredness alert |
| `TIRED_COOLDOWN_SEC` | `20` | Cooldown between tiredness alerts |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5vl:7b` | Vision model to use |

## Project Structure

```
Face_recognition_usb/
├── ai_vision_assistant.py   # AI vision assistant (motion + AI + blink)
├── blink_detector.py        # BlinkDetector class (MediaPipe FaceMesh)
├── config_ai.py             # All thresholds and settings
├── ollama_vision_client.py  # Helper to query Ollama vision models
├── Face_Recognition.py      # Original face recognition
├── faces-train.py           # Train LBPH recogniser
├── GUI.py                   # Tkinter GUI for attendance system
├── capture_images.py        # Capture training images
├── attendance.py            # Attendance tracking
├── tts_speaker.py           # (removed — no longer used)
├── test.py                  # Test script
├── images/                  # Training images (one subfolder per person)
├── cascades/                # Haar Cascade XML files
├── ExcelData/               # Attendance Excel output
├── trainer.yml              # Trained LBPH model
├── labels.pickle            # Label mappings
└── requirements.txt         # Python dependencies