"""
config_ai.py — Constants for the AI Vision Assistant feature
============================================================
All thresholds, URLs, and flags live here so no hardcoded
numbers appear in the implementation modules.
"""

# ── Ollama ──────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:7b"

# ── Blink detection thresholds ──────────────────────────────
EAR_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 2

# ── Tiredness detection ─────────────────────────────────────
BLINK_RATE_WINDOW_SEC = 60
TIRED_BLINK_RATE = 25
TIRED_COOLDOWN_SEC = 20

# ── Motion detection ────────────────────────────────────────
MOTION_THRESHOLD = 3000    # pixel-diff count that counts as "something moved"
HOLD_STILL_FRAMES = 15     # consecutive still frames before triggering AI

# ── Trigger / banner cooldowns ──────────────────────────────
TRIGGER_COOLDOWN_SEC = 6
