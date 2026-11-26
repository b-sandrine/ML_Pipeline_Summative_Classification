"""Inference helpers: single-image prediction with class probabilities."""

from __future__ import annotations

import threading

import numpy as np

from .model import MODEL_PATH, load_model
from .preprocessing import NOT_FLOWER_LABEL, image_features

_cache: dict = {"mtime": None, "artifact": None}
_cache_lock = threading.Lock()
_infer_lock = threading.Lock()  # Keras predict() is not guaranteed re-entrant


def get_model() -> dict:
    """Load the model, transparently reloading it when the file changes
    (so a retrain in one worker is picked up by the others)."""
    mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    with _cache_lock:
        if mtime is None or mtime != _cache["mtime"]:
            _cache["artifact"] = load_model()
            _cache["mtime"] = mtime
        return _cache["artifact"]


def predict_image(image_bytes: bytes, model: dict | None = None, top_k: int = 3) -> dict:
    """Classify one image.

    Returns ``{"label", "confidence", "is_flower", "top": [(label, p), ...]}``.
    Raises ``ValueError`` for files that are not valid images.
    """
    model = model or get_model()
    estimator = model["estimator"]
    with _infer_lock:
        features = image_features(image_bytes)
    proba = estimator.predict_proba(features)[0]
    order = np.argsort(proba)[::-1]
    classes = estimator.classes_
    top = [(str(classes[i]), float(proba[i])) for i in order[:top_k]]
    label, confidence = top[0]
    return {
        "label": label,
        "confidence": confidence,
        "is_flower": label != NOT_FLOWER_LABEL,
        "top": top,
    }
