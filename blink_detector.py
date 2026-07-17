"""
blink_detector.py — BlinkDetector class using mediapipe FaceMesh + EAR
========================================================================
Tracks eye blinks via Eye Aspect Ratio and computes blink rate over a
rolling window to detect tiredness.
"""

from collections import deque
import time

import cv2
import mediapipe as mp

from config_ai import (
    EAR_THRESHOLD,
    EAR_CONSEC_FRAMES,
    BLINK_RATE_WINDOW_SEC,
    TIRED_BLINK_RATE,
    TIRED_COOLDOWN_SEC,
)


# Left and right eye landmark indices (mediapipe FaceMesh convention)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _ear(landmarks, face_landmarks, image_width, image_height):
    """Compute the Eye Aspect Ratio for a single eye given its 6 landmarks."""
    pts = []
    for idx in landmarks:
        lm = face_landmarks.landmark[idx]
        pts.append((lm.x * image_width, lm.y * image_height))

    # p2-p6 and p3-p5 are vertical; p1-p4 is horizontal
    p1, p2, p3, p4, p5, p6 = pts
    vert_a = ((p2[0] - p6[0]) ** 2 + (p2[1] - p6[1]) ** 2) ** 0.5
    vert_b = ((p3[0] - p5[0]) ** 2 + (p3[1] - p5[1]) ** 2) ** 0.5
    horiz = ((p1[0] - p4[0]) ** 2 + (p1[1] - p4[1]) ** 2) ** 0.5

    if horiz == 0:
        return 0.0

    return (vert_a + vert_b) / (2.0 * horiz)


class BlinkDetector:
    """Detects blinks and calculates blink-rate tiredness."""

    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # State machine for blink detection
        self._consec_low = 0
        self._blink_occurred = False

        # Rolling blink timestamps
        self._blink_timestamps: deque[float] = deque()

        # Tiredness cooldown
        self._last_tired_alert: float = 0.0

    def process(self, frame_bgr):
        """
        Detect blinks and compute tiredness for the given BGR frame.

        Returns
        -------
        (blinked: bool, tired: bool)
            blinked — True if a blink was detected on *this* frame.
            tired   — True if the blink rate exceeds the threshold
                      (respecting cooldown).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        blinked = False
        tired = False

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame_bgr.shape

            ear_left = _ear(LEFT_EYE, face_landmarks, w, h)
            ear_right = _ear(RIGHT_EYE, face_landmarks, w, h)
            avg_ear = (ear_left + ear_right) / 2.0

            # Blink transition: EAR drops below threshold for enough frames
            if avg_ear < EAR_THRESHOLD:
                self._consec_low += 1
            else:
                if self._consec_low >= EAR_CONSEC_FRAMES and not self._blink_occurred:
                    blinked = True
                    now = time.time()
                    self._blink_timestamps.append(now)
                    # Prune old timestamps outside the window
                    cutoff = now - BLINK_RATE_WINDOW_SEC
                    while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
                        self._blink_timestamps.popleft()

                self._consec_low = 0
                self._blink_occurred = False

            # Mark the blink cycle so we don't double-count
            if blinked:
                self._blink_occurred = True

            # --- Tiredness check ---
            now = time.time()
            if now - self._last_tired_alert >= TIRED_COOLDOWN_SEC:
                if len(self._blink_timestamps) >= TIRED_BLINK_RATE:
                    tired = True
                    self._last_tired_alert = now
        else:
            # No face detected — reset counters
            self._consec_low = 0
            self._blink_occurred = False

        return blinked, tired

    def close(self):
        """Release the FaceMesh resources."""
        self._face_mesh.close()