"""Command-line entry point: acquire data, split, train, evaluate.

    python train.py --prepare      # download (if needed) + build data/train & data/test
    python train.py                # (re)train on the current data/train, test on data/test
"""

from __future__ import annotations

import argparse
import json

from src.database import initialise_database, log_training
from src.model import MODEL_PATH, retrain
from src.preprocessing import RAW_DIR, dataset_stats, download_dataset, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--prepare", action="store_true", help="download data and rebuild the 80/20 split")
    args = parser.parse_args()

    if args.prepare:
        if not RAW_DIR.exists():
            download_dataset()
        print("Split created:", json.dumps(prepare_dataset(), indent=2))
    else:
        print("Dataset:", json.dumps(dataset_stats()))

    metrics = retrain(trigger="cli")
    initialise_database()
    log_training(metrics)
    print(f"Saved model to {MODEL_PATH}")
    print(f"train/test images : {metrics['train_image_count']} / {metrics['test_image_count']}")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr", "log_loss"):
        print(f"{key:16s}: {metrics[key]:.4f}")
    print("classes:", metrics["classes"])
    print("confusion matrix:", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
