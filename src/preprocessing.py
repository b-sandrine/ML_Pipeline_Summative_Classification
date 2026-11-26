"""Data acquisition and preprocessing for the flower image pipeline.

Responsibilities
----------------
* acquire data: download the public flowers dataset, accept ZIP uploads
* validate / clean images (format, size, decompression-bomb guard)
* split a folder-per-class dataset into ``data/train`` and ``data/test``
* turn images into numeric features (MobileNetV2 embeddings) and cache them
  so retraining only embeds images that have not been seen before
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import tarfile
import urllib.request
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "flower_photos"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = ROOT / "models"
FEATURE_STORE_PATH = MODELS_DIR / "feature_store.npz"
STATE_PATH = MODELS_DIR / "state.json"

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
IMAGE_SIZE = 160
EMBEDDING_DIM = 1280
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
NOT_FLOWER_LABEL = "not_flower"

MAX_ARCHIVE_BYTES = 350 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGES = 10_000


# --------------------------------------------------------------------------- #
# Data acquisition
# --------------------------------------------------------------------------- #
def download_dataset(destination: Path = DATA_DIR) -> Path:
    """Download and extract the TensorFlow flowers dataset (~220 MB)."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "flower_photos.tgz"
    if not RAW_DIR.exists():
        logger.info("Downloading %s", DATASET_URL)
        urllib.request.urlretrieve(DATASET_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(destination, filter="data")
        archive.unlink(missing_ok=True)
    return RAW_DIR


def normalise_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def iter_class_dirs(directory: Path):
    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")):
        yield normalise_label(class_dir.name), class_dir


def list_images(directory: Path) -> list[tuple[Path, str]]:
    """Return ``(path, label)`` for every supported image under ``directory/<class>/``."""
    if not directory.is_dir():
        return []
    items: list[tuple[Path, str]] = []
    for label, class_dir in iter_class_dirs(directory):
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                items.append((path, label))
    return items


def prepare_dataset(
    source: Path = RAW_DIR,
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Create a reproducible stratified train/test split on disk.

    ``source`` must hold one sub-folder per class. Images are *copied* so the
    raw download stays untouched. Re-running wipes and rebuilds both folders.
    """
    items = list_images(source)
    if not items:
        raise FileNotFoundError(f"No images found under {source}. Run download_dataset() first.")
    labels = [label for _, label in items]
    train_idx, test_idx = train_test_split(
        np.arange(len(items)), test_size=test_size, random_state=seed, stratify=labels
    )
    for target in (train_dir, test_dir):
        if target.exists():
            shutil.rmtree(target)
    for target, indices in ((train_dir, train_idx), (test_dir, test_idx)):
        for index in indices:
            path, label = items[index]
            (target / label).mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / label / path.name)
    reset_state()
    FEATURE_STORE_PATH.unlink(missing_ok=True)
    return dataset_stats()


def dataset_stats() -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {"train": {}, "test": {}}
    for split, directory in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        counts = Counter(label for _, label in list_images(directory))
        stats[split] = dict(sorted(counts.items()))
    return stats


# --------------------------------------------------------------------------- #
# Image validation / decoding
# --------------------------------------------------------------------------- #
def read_image(source: bytes | BinaryIO | Path) -> np.ndarray:
    """Decode, validate, convert to RGB and resize to ``IMAGE_SIZE`` square."""
    try:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        image = Image.open(source)
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Images must contain no more than 16 million pixels.")
        image.load()
        rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The file is not a valid, supported image.") from error
    return np.asarray(rgb, dtype=np.uint8)


def image_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# --------------------------------------------------------------------------- #
# Feature extraction (transfer learning backbone)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def feature_extractor():
    from tensorflow.keras.applications import MobileNetV2

    logger.info("Loading pretrained MobileNetV2 feature extractor")
    return MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )


def embed_arrays(arrays: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    extractor = feature_extractor()
    out = []
    for start in range(0, len(arrays), batch_size):
        batch = np.stack(arrays[start : start + batch_size]).astype(np.float32)
        out.append(extractor.predict(preprocess_input(batch), verbose=0))
    return np.vstack(out).astype(np.float32) if out else np.empty((0, EMBEDDING_DIM), np.float32)


def image_features(source: bytes | BinaryIO | Path) -> np.ndarray:
    """Embedding for one image, shape ``(1, 1280)``."""
    return embed_arrays([read_image(source)])


# --------------------------------------------------------------------------- #
# Feature store: persistent cache of (hash -> label, split, embedding)
# --------------------------------------------------------------------------- #
class FeatureStore:
    """Caches embeddings so that retraining only processes *new* images.

    The store also lets a deployed container retrain without the original
    raw images: the embeddings of the base dataset ship inside the store.
    """

    def __init__(self, path: Path = FEATURE_STORE_PATH):
        self.path = path
        self.hashes: list[str] = []
        self.labels: list[str] = []
        self.splits: list[str] = []
        self.vectors = np.empty((0, EMBEDDING_DIM), np.float16)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                self.hashes = data["hashes"].tolist()
                self.labels = data["labels"].tolist()
                self.splits = data["splits"].tolist()
                self.vectors = data["vectors"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            hashes=np.asarray(self.hashes, dtype="U40"),
            labels=np.asarray(self.labels, dtype="U64"),
            splits=np.asarray(self.splits, dtype="U8"),
            vectors=self.vectors,
        )

    def sync(self) -> int:
        """Embed every image on disk that is not cached yet. Returns #new."""
        known = set(zip(self.hashes, self.splits))
        pending: list[tuple[str, str, str, np.ndarray]] = []
        for split, directory in (("train", TRAIN_DIR), ("test", TEST_DIR)):
            for path, label in list_images(directory):
                data = path.read_bytes()
                digest = image_hash(data)
                if (digest, split) in known:
                    continue
                try:
                    pending.append((digest, label, split, read_image(data)))
                except ValueError:
                    logger.warning("Skipping unreadable image %s", path)
                    continue
                known.add((digest, split))
        if pending:
            vectors = embed_arrays([item[3] for item in pending])
            self.hashes += [p[0] for p in pending]
            self.labels += [p[1] for p in pending]
            self.splits += [p[2] for p in pending]
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float16)])
            self.save()
        return len(pending)

    def split(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(self.splits) == name
        return self.vectors[mask].astype(np.float32), np.asarray(self.labels)[mask]


# --------------------------------------------------------------------------- #
# Uploaded training data
# --------------------------------------------------------------------------- #
def save_uploaded_images(files: list[tuple[str, bytes]], label: str | None = None) -> dict[str, int]:
    """Validate uploaded images and store them in ``data/train/<label>/``.

    ``files`` is a list of ``(filename, bytes)``. A ZIP file is expanded and its
    parent folder name is used as the label; plain images need ``label``.
    Returns ``{label: n_saved}``. Duplicates (same bytes) are skipped.
    """
    saved: Counter[str] = Counter()
    total = 0

    def store(name: str, data: bytes, lbl: str) -> None:
        nonlocal total
        lbl = normalise_label(lbl)
        if not lbl or lbl.startswith("."):
            return
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"{name} exceeds the 20 MB image limit.")
        read_image(data)  # raises ValueError when invalid
        digest = image_hash(data)
        suffix = Path(name).suffix.lower()
        target = TRAIN_DIR / lbl / f"upload_{digest[:16]}{suffix}"
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        saved[lbl] += 1
        total += 1
        if total > MAX_IMAGES:
            raise ValueError("Too many images in a single upload (limit 10,000).")

    for filename, data in files:
        suffix = Path(filename).suffix.lower()
        if suffix == ".zip":
            if len(data) > MAX_ARCHIVE_BYTES:
                raise ValueError("ZIP archives must be smaller than 350 MB.")
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    uncompressed = 0
                    for member in archive.infolist():
                        path = PurePosixPath(member.filename)
                        if member.is_dir() or path.is_absolute() or ".." in path.parts:
                            continue
                        if path.suffix.lower() not in ALLOWED_EXTENSIONS or len(path.parts) < 2:
                            continue
                        uncompressed += member.file_size
                        if uncompressed > MAX_ARCHIVE_BYTES:
                            raise ValueError("Uncompressed images exceed 350 MB.")
                        store(path.name, archive.read(member), path.parts[-2])
            except zipfile.BadZipFile as error:
                raise ValueError("Upload a valid ZIP archive.") from error
        elif suffix in ALLOWED_EXTENSIONS:
            if not label:
                raise ValueError("Choose a class label for individual image uploads.")
            store(filename, data, label)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    if saved:
        state = load_state()
        state["pending_images"] = state.get("pending_images", 0) + sum(saved.values())
        save_state(state)
    return dict(saved)


# --------------------------------------------------------------------------- #
# Small JSON state (pending uploads, last training time)
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"pending_images": 0}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def reset_state() -> None:
    save_state({"pending_images": 0})


# --------------------------------------------------------------------------- #
# Descriptive statistics used by the visualisations
# --------------------------------------------------------------------------- #
def image_statistics(directory: Path = TRAIN_DIR, per_class: int = 150, seed: int = 0) -> list[dict]:
    """Per-image width, height, brightness, colour channels, sharpness.

    A random sample (``per_class``) keeps this fast on the full dataset.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for label, class_dir in iter_class_dirs(directory):
        files = [p for p in class_dir.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS]
        if len(files) > per_class:
            files = list(rng.choice(np.asarray(files, dtype=object), per_class, replace=False))
        for path in files:
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    arr = np.asarray(img.convert("RGB").resize((96, 96)), dtype=np.float32)
            except (UnidentifiedImageError, OSError):
                continue
            gray = arr.mean(axis=2)
            sharpness = float(np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1)))
            rows.append(
                {
                    "label": label,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height,
                    "brightness": float(gray.mean()),
                    "red": float(arr[..., 0].mean()),
                    "green": float(arr[..., 1].mean()),
                    "blue": float(arr[..., 2].mean()),
                    "sharpness": sharpness,
                }
            )
    return rows
