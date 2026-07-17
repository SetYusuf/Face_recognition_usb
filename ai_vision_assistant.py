"""
ai_vision_assistant.py — AI Vision Assistant (standalone entry point)
======================================================================
Opens its own camera + OpenCV window.
Automatically triggers AI description when motion is detected and
an object is held still in frame.
Blink / tiredness monitoring prints to terminal.
    Q → quit
"""

import time
import threading

import cv2
import numpy as np

from blink_detector import BlinkDetector
from ollama_vision_client import ask_about_frame
from config_ai import (
    MOTION_THRESHOLD,
    HOLD_STILL_FRAMES,
    TRIGGER_COOLDOWN_SEC,
    TIRED_COOLDOWN_SEC,
)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    # ── Initialise camera ──────────────────────────────────────────
    cap = None
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            break
        cap.release()
        cap = None

    if cap is None:
        print("ERROR: No camera found (scanned indices 0-4).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Initialise modules ─────────────────────────────────────────
    detector = BlinkDetector()

    # ── State ──────────────────────────────────────────────────────
    ai_busy = False
    last_trigger_time = 0.0

    # Motion detection state
    prev_gray = None
    still_counter = 0

    # Tiredness cooldown (for terminal prints)
    last_tired_print = 0.0

    # ── Camera loop ────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame; exiting.")
            break

        now = time.time()

        # 1. Blink detection & tiredness (unchanged logic, just print)
        blinked, tired = detector.process(frame)
        if tired and (now - last_tired_print >= TIRED_COOLDOWN_SEC):
            last_tired_print = now
            print("[BLINK] You've been blinking a lot — maybe you're tired?")

        # 2. Motion detection using frame differencing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            status_text = "watching..."
        else:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_pixels = cv2.countNonZero(thresh)

            if motion_pixels > MOTION_THRESHOLD:
                still_counter = 0
                status_text = "watching..."
            else:
                still_counter += 1
                if still_counter >= HOLD_STILL_FRAMES and not ai_busy:
                    if now - last_trigger_time >= TRIGGER_COOLDOWN_SEC:
                        status_text = "thinking..."
                        capture = frame.copy()
                        ai_busy = True
                        last_trigger_time = now

                        def _ai_worker(frm):
                            nonlocal ai_busy
                            try:
                                result = ask_about_frame(frm)
                            except Exception as e:
                                result = f"(AI error: {e})"

                            print(f"[AI] {result}")
                            ai_busy = False

                        threading.Thread(
                            target=_ai_worker,
                            args=(capture,),
                            daemon=True,
                        ).start()

                        still_counter = 0
                else:
                    status_text = "thinking..." if ai_busy else "watching..."

        prev_gray = gray

        # 3. Simple status overlay (small text in corner)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 28), (140, h), (36, 36, 36), -1)
        cv2.putText(frame, status_text, (8, h - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # 4. Show frame
        cv2.imshow("AI Vision Assistant", frame)

        # 5. Key handling — only Q to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    # ── Cleanup ────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("AI Vision Assistant exited cleanly.")


if __name__ == "__main__":
    main()