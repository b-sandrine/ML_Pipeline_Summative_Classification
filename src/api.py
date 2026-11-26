"""REST API: prediction, data upload, retraining, monitoring.

Run:  uvicorn src.api:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import socket
import time
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

from . import model as mdl
from . import preprocessing as pp
from .database import get_monitoring_summary, initialise_database, log_prediction, log_training
from .prediction import get_model, predict_image

logger = logging.getLogger("flower-api")
STARTED_AT = datetime.now(timezone.utc)
STARTED_MONO = time.monotonic()
CONTAINER = os.getenv("HOSTNAME", socket.gethostname())

app = FastAPI(title="Flower Image Classifier API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    initialise_database()
    try:  # warm up: load the model and the TensorFlow backbone once
        get_model()
        pp.feature_extractor()
    except (FileNotFoundError, ValueError) as error:
        logger.warning("Model not ready: %s", error)


# --------------------------------------------------------------------------- #
@app.get("/health", tags=["monitoring"])
def health() -> dict:
    """Liveness + uptime. Used by the UI up-time panel and Docker health checks."""
    try:
        model = get_model()
        loaded, trained_at = True, model.get("trained_at")
    except (FileNotFoundError, ValueError):
        loaded, trained_at = False, None
    return {
        "status": "ok" if loaded else "degraded",
        "uptime_seconds": round(time.monotonic() - STARTED_MONO, 1),
        "started_at": STARTED_AT.isoformat(),
        "model_loaded": loaded,
        "model_trained_at": trained_at,
        "container": CONTAINER,
    }


@app.post("/predict", tags=["model"])
async def predict(file: UploadFile = File(...)) -> dict:
    """Classify one uploaded image."""
    started = time.perf_counter()
    data = await file.read()
    try:
        result = predict_image(data)
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        log_prediction(None, None, (time.perf_counter() - started) * 1000, error=str(error))
        raise HTTPException(422, str(error)) from error
    latency = (time.perf_counter() - started) * 1000
    log_prediction(result["label"], result["confidence"], latency)
    return {**result, "latency_ms": round(latency, 1), "container": CONTAINER}


@app.post("/upload", tags=["retraining"])
def upload(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    label: str | None = Form(None),
) -> dict:
    """Bulk-upload training data: images (with a ``label``) and/or ZIPs of
    ``<class>/<image>`` folders. Triggers retraining automatically once the
    number of un-trained uploads reaches ``RETRAIN_THRESHOLD``."""
    try:
        saved = pp.save_uploaded_images([(f.filename or "", f.file.read()) for f in files], label)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    auto = mdl.should_retrain()
    if auto:
        background.add_task(_run_retrain, "auto-threshold")
    status = mdl.retraining_status()
    return {"saved": saved, "total_saved": sum(saved.values()), "auto_retrain_started": auto,
            "pending_images": status["pending_images"], "threshold": status["threshold"]}


@app.post("/retrain", status_code=202, tags=["retraining"])
def retrain(background: BackgroundTasks) -> dict:
    """Manual trigger: retrain in the background; poll ``/retrain/status``."""
    if mdl.retraining_status()["state"] == "running":
        raise HTTPException(409, "A training run is already in progress.")
    background.add_task(_run_retrain, "manual")
    return {"message": "Retraining started", "status_url": "/retrain/status"}


def _run_retrain(trigger: str) -> None:
    try:
        metrics = mdl.retrain(trigger)
        log_training(metrics)
        get_model()  # refresh the in-memory copy
        _dataset_features.cache_clear()
    except Exception:  # status already recorded by mdl.retrain
        logger.exception("Retraining failed")


@app.get("/retrain/status", tags=["retraining"])
def retrain_status() -> dict:
    return mdl.retraining_status()


# --------------------------------------------------------------------------- #
@app.get("/metrics", tags=["monitoring"])
def metrics() -> dict:
    """Latest evaluation metrics (held-out test set) and training history."""
    latest = mdl.load_metrics()
    if latest is None:
        raise HTTPException(404, "No metrics yet; train the model first.")
    return {"latest": latest, "history": mdl.load_history()}


@app.get("/stats", tags=["monitoring"])
def stats() -> dict:
    """Dataset counts and live prediction usage."""
    return {"dataset": pp.dataset_stats(), "usage": get_monitoring_summary()}


@lru_cache(maxsize=1)
def _dataset_features() -> list[dict]:
    return pp.image_statistics(per_class=120)


@app.get("/features", tags=["visualisation"])
def features() -> dict:
    """Per-image feature statistics for the UI's data visualisations."""
    rows = _dataset_features()
    return {"rows": rows, "count": len(rows)}
