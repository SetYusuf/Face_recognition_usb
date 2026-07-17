"""
ollama_vision_client.py — Async‑safe helper for querying Ollama VLMs
=====================================================================
Provides two functions:
  - ask_about_frame(frame_bgr, prompt=None) -> str
  - ask_for_recipe(frame_bgr) -> str

Both are blocking network calls — run them in a background thread.
"""

import base64
import io

import cv2
import requests
from PIL import Image

from config_ai import OLLAMA_URL, OLLAMA_MODEL


def _encode_frame_bgr(frame_bgr):
    """Convert a BGR OpenCV frame to a base64 JPEG string."""
    # Encode as JPEG into memory
    success, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _call_ollama(prompt, image_b64):
    """Internal helper — POST to Ollama and return the response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def ask_about_frame(frame_bgr, prompt=None):
    """
    Identify the object / scene in *frame_bgr* using the vision model.

    Parameters
    ----------
    frame_bgr : numpy.ndarray
        OpenCV BGR frame.
    prompt : str, optional
        Override the default prompt.  Default: short object‑ID request.

    Returns
    -------
    str
        The model's response, or an error message wrapped in parentheses.
    """
    if prompt is None:
        prompt = (
            "Describe what's visible in this image in 1-2 casual sentences "
            "identifying the main object or scene. Keep it under 40 words."
        )

    try:
        b64 = _encode_frame_bgr(frame_bgr)
        return _call_ollama(prompt, b64)
    except requests.RequestException as e:
        return f"(AI unavailable: {e})"
    except Exception as e:
        return f"(AI error: {e})"


def ask_for_recipe(frame_bgr):
    """
    Identify any food / ingredients in the frame and suggest a short recipe.

    Returns
    -------
    str
        Cooking suggestion or an error message wrapped in parentheses.
    """
    prompt = (
        "Identify any food or ingredients visible in this image. "
        "Then give a short 3-4 step cooking suggestion. "
        "Keep it under 60 words."
    )

    try:
        b64 = _encode_frame_bgr(frame_bgr)
        return _call_ollama(prompt, b64)
    except requests.RequestException as e:
        return f"(AI unavailable: {e})"
    except Exception as e:
        return f"(AI error: {e})"