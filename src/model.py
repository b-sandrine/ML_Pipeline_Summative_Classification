"""Model creation, evaluation, persistence and retraining.

The classifier is a transfer-learning model: frozen MobileNetV2 (ImageNet)
embeddings feed a standardised multinomial logistic regression. Retraining
therefore takes seconds-to-minutes on CPU instead of hours.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import (
    MODELS_DIR,
    NOT_FLOWER_LABEL,
    FeatureStore,
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "flower_classifier.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
HISTORY_PATH = MODELS_DIR / "training_history.json"
FEATURE_VERSION = 3
MIN_IMAGES_PER_CLASS = 5
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))


# --------------------------------------------------------------------------- #
# Model creation
# --------------------------------------------------------------------------- #
def build_estimator(C: float = 1.0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=42),
    )


# --------------------------------------------------------------------------- #
# Model testing
# --------------------------------------------------------------------------- #
def evaluate(estimator, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """All metrics required by the assignment, on held-out data."""
    classes = list(estimator.classes_)
    proba = estimator.predict_proba(x_test)
    pred = estimator.predict(x_test)
    try:
        auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes))
    except ValueError:  # a class missing from the test set
        auc = None
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "roc_auc_ovr": auc,
        "log_loss": float(log_loss(y_test, proba, labels=classes)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
        "per_class": classification_report(
            y_test, pred, labels=classes, output_dict=True, zero_division=0
        ),
        "classes": classes,
    }


# --------------------------------------------------------------------------- #
# Model training
# --------------------------------------------------------------------------- #
def train_model(store: FeatureStore | None = None) -> tuple[dict, dict]:
    """Fit on the train split, evaluate on the test split, return (artifact, metrics)."""
    store = store or FeatureStore()
    new_images = store.sync()
    x_train, y_train = store.split("train")
    x_test, y_test = store.split("test")
    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError(
            "No training/test data found. Run `python train.py --prepare` (or upload data) first."
        )
    counts = dict(zip(*np.unique(y_train, return_counts=True)))
    too_small = [c for c, n in counts.items() if n < MIN_IMAGES_PER_CLASS]
    if len(counts) < 2 or too_small:
        raise ValueError(
            f"Need at least 2 classes with >= {MIN_IMAGES_PER_CLASS} images each. "
            f"Too small: {', '.join(too_small) or 'n/a'}"
        )

    started = time.perf_counter()
    estimator = build_estimator()
    estimator.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    metrics = evaluate(estimator, x_test, y_test)
    # The training-set score exposes over-fitting when compared to the test score.
    metrics["train_accuracy"] = float(accuracy_score(y_train, estimator.predict(x_train)))
    metrics.update(
        {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_image_count": int(len(y_train)),
            "test_image_count": int(len(y_test)),
            "image_count": int(len(y_train) + len(y_test)),
            "train_class_counts": {k: int(v) for k, v in counts.items()},
            "test_class_counts": {k: int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
            "class_counts": _merge_counts(y_train, y_test),
            "new_images_embedded": int(new_images),
            "fit_seconds": round(fit_seconds, 2),
        }
    )
    artifact = {
        "estimator": estimator,
        "classes": list(estimator.classes_),
        "feature_version": FEATURE_VERSION,
        "backbone": "MobileNetV2-ImageNet",
        "trained_at": metrics["trained_at"],
    }
    return artifact, metrics


def _merge_counts(*arrays: np.ndarray) -> dict[str, int]:
    merged: dict[str, int] = {}
    for array in arrays:
        for label, n in zip(*np.unique(array, return_counts=True)):
            merged[str(label)] = merged.get(str(label), 0) + int(n)
    return dict(sorted(merged.items()))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_model(artifact: dict, metrics: dict | None = None, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    if metrics is not None:
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        history = load_history()
        history.append({k: metrics[k] for k in _HISTORY_KEYS if k in metrics})
        HISTORY_PATH.write_text(json.dumps(history[-50:], indent=2))


_HISTORY_KEYS = (
    "trained_at", "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "train_image_count", "test_image_count", "fit_seconds", "trigger",
)


def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError("No model has been trained yet. Train one first.")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("feature_version") != FEATURE_VERSION:
        raise ValueError("The saved model is incompatible with this code; retrain it.")
    return artifact


def load_metrics() -> dict | None:
    try:
        return json.loads(METRICS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_history() -> list[dict]:
    try:
        return json.loads(HISTORY_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


# --------------------------------------------------------------------------- #
# Retraining (manual button, or automatic trigger on enough new data)
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_status: dict = {"state": "idle", "message": "No retraining has run yet.", "trigger": None}


def retraining_status() -> dict:
    state = load_state()
    return {
        **_status,
        "pending_images": state.get("pending_images", 0),
        "threshold": RETRAIN_THRESHOLD,
        "last_trained_at": state.get("last_trained_at"),
    }


def should_retrain() -> bool:
    return load_state().get("pending_images", 0) >= RETRAIN_THRESHOLD and not _lock.locked()


def retrain(trigger: str = "manual") -> dict:
    """Retrain on everything in ``data/train`` and activate the new model.

    Thread-safe: only one run at a time; a concurrent call raises RuntimeError.
    """
    if not _lock.acquire(blocking=False):
        raise RuntimeError("A training run is already in progress.")
    try:
        _status.update(state="running", message="Embedding new images and fitting...", trigger=trigger)
        artifact, metrics = train_model()
        metrics["trigger"] = trigger
        save_model(artifact, metrics)
        state = load_state()
        state.update(pending_images=0, last_trained_at=metrics["trained_at"])
        save_state(state)
        _status.update(state="done", message="Training complete; new model is active.")
        return metrics
    except Exception as error:
        _status.update(state="failed", message=str(error))
        raise
    finally:
        _lock.release()
