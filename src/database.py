"""SQLite-backed application usage and model-training monitoring."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "database.db"


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialise_database(path: Path = DB_PATH) -> None:
    with connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS prediction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                predicted_class TEXT,
                confidence REAL,
                duration_ms REAL NOT NULL,
                succeeded INTEGER NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                completed_at TEXT NOT NULL,
                image_count INTEGER NOT NULL,
                class_counts_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                model_version TEXT NOT NULL
            );
            """
        )


def log_prediction(
    predicted_class: str | None,
    confidence: float | None,
    duration_ms: float,
    error: str | None = None,
    path: Path = DB_PATH,
) -> None:
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_events
                (occurred_at, predicted_class, confidence, duration_ms, succeeded, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                predicted_class,
                confidence,
                duration_ms,
                int(error is None),
                error,
            ),
        )


def log_training(metrics: dict[str, object], path: Path = DB_PATH) -> None:
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO training_runs
                (completed_at, image_count, class_counts_json, metrics_json, model_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                int(metrics["image_count"]),
                json.dumps(metrics["class_counts"]),
                json.dumps(
                    {
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "macro_precision": metrics.get("macro_precision"),
                        "macro_recall": metrics.get("macro_recall"),
                        "trigger": metrics.get("trigger"),
                        "confusion_matrix": metrics["confusion_matrix"],
                        "classes": metrics["classes"],
                        "train_image_count": metrics["train_image_count"],
                        "test_image_count": metrics["test_image_count"],
                        "train_class_counts": metrics["train_class_counts"],
                        "test_class_counts": metrics["test_class_counts"],
                    }
                ),
                "mobilenetv2-transfer-logreg-v3",
            ),
        )


def get_monitoring_summary(path: Path = DB_PATH) -> dict[str, object]:
    initialise_database(path)
    with connect(path) as connection:
        predictions = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(succeeded), 0) AS succeeded,
                   AVG(CASE WHEN succeeded = 1 THEN confidence END) AS mean_confidence,
                   AVG(duration_ms) AS mean_duration_ms
            FROM prediction_events
            """
        ).fetchone()
        classes = connection.execute(
            """
            SELECT predicted_class AS name, COUNT(*) AS count
            FROM prediction_events
            WHERE succeeded = 1
            GROUP BY predicted_class
            ORDER BY count DESC
            """
        ).fetchall()
        runs = connection.execute(
            """
            SELECT completed_at, image_count, class_counts_json, metrics_json, model_version
            FROM training_runs ORDER BY id DESC LIMIT 10
            """
        ).fetchall()
    return {
        "predictions": dict(predictions),
        "class_counts": [dict(row) for row in classes],
        "training_runs": [
            {
                **dict(row),
                "class_counts": json.loads(row["class_counts_json"]),
                "metrics": json.loads(row["metrics_json"]),
            }
            for row in runs
        ],
    }
