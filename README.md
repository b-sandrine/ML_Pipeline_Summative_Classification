# Flower Image MLOps Pipeline

An end-to-end machine-learning pipeline for **image classification**: data acquisition, preprocessing, model training and evaluation, a REST API, a web UI with retraining, Docker deployment and load testing.

| | |
|---|---|
| **Video demo** | _TODO: add YouTube link_ |
| **Live URL** | _TODO: add deployed URL (UI on port 8501, API docs at `/docs`)_ |
| **GitHub repo** | _TODO: add repository URL_ |

## Project description

The model looks at a photo and answers: **daisy, dandelion, roses, sunflowers, tulips, or `not_flower`**.

* **Data:** the public TensorFlow flowers dataset (3,670 photos) plus a small hand-collected `not_flower` class (24 images) so non-flowers can be rejected.
* **Model:** transfer learning. A frozen MobileNetV2 (ImageNet) converts each 160x160 photo into a 1280-number embedding; a standardised, class-balanced logistic regression is trained on top. Retraining takes seconds because embeddings are cached in `models/feature_store.npz` and only *new* images are embedded.
* **Retraining:** users upload images or a ZIP of `<class>/<image>` folders. Retraining starts when someone presses **Retrain**, or automatically once `RETRAIN_THRESHOLD` (default 50) new images are waiting.
* **Evaluation:** an 80/20 stratified split (`data/train`, `data/test`, seed 42). The test set is never used for fitting.

### Results on the held-out test set (739 images)

| Metric | Value |
|---|---|
| Accuracy | 0.888 |
| Macro precision / recall / F1 | 0.78 / 0.77 / 0.78 |
| ROC-AUC (one-vs-rest, macro) | 0.982 |

The five flower species score F1 0.84-0.93 (roses vs tulips is the main confusion). **`not_flower` is weak (F1 about 0.22) because it has only 19 training and 5 test images.** Its test metrics are noisy and it drags the macro averages down. Upload more varied non-flower photos and retrain to improve it; see the notebook for the full analysis.

## Repository layout

```
├── README.md
├── notebook/project_name.ipynb   EDA, 3+ feature interpretations, preprocessing, training, evaluation, prediction
├── src/
│   ├── preprocessing.py          data download, validation, split, embeddings + feature store, uploads
│   ├── model.py                  model creation, metrics, save/load, retrain() + trigger logic
│   ├── prediction.py             single-image prediction
│   ├── api.py                    FastAPI service
│   └── database.py               SQLite usage + training logs
├── app.py                        Streamlit UI
├── train.py                      CLI: prepare data / train / evaluate
├── data/train, data/test         created by `python train.py --prepare` (git-ignored)
├── models/flower_classifier.pkl  trained model (+ metrics.json, feature_store.npz)
├── Dockerfile, Dockerfile.ui, docker-compose.yml, deploy/   containers + nginx load balancer
└── locust/                       load-test script and sample images
```

## Setup (local)

Python 3.11 is recommended (TensorFlow 2.18 requires NumPy 1.x).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python train.py --prepare        # downloads ~220 MB, builds data/train + data/test, trains, evaluates
# (optional) add your own photos to data/flower_photos/not_flower/ BEFORE running --prepare

uvicorn src.api:app --port 8000                        # API,  docs: http://localhost:8000/docs
API_URL=http://localhost:8000 streamlit run app.py     # UI,   http://localhost:8501
```

The repo already contains a trained model, so the API and UI start without `--prepare`; `--prepare` is only needed to retrain from raw images or to run the notebook from scratch. Open `notebook/project_name.ipynb` with Jupyter to see every step.

## Using the app

| UI page | What it does |
|---|---|
| **Status & up-time** | live up-time, model loaded?, which container answered, prediction count / latency |
| **Predict** | upload one image, get label, confidence and top-3 probabilities |
| **Data visualisations** | class balance, colour, brightness, sharpness, resolution (with interpretations), confusion matrix and metrics |
| **Upload data & retrain** | upload many images (one label) or a ZIP of class folders, see pending count, press **Retrain model now** |

API endpoints: `GET /health`, `POST /predict`, `POST /upload`, `POST /retrain`, `GET /retrain/status`, `GET /metrics`, `GET /stats`, `GET /features`.

```bash
curl -F file=@photo.jpg http://localhost:8000/predict
curl -F files=@flowers.zip http://localhost:8000/upload
curl -X POST http://localhost:8000/retrain
```

## Docker and cloud deployment

```bash
docker compose up -d --build --scale api=3     # UI: :8501, API (through nginx): :8000
```

`nginx` load-balances requests round-robin across the `api` replicas; `models/` and `data/` are shared volumes, so a retrain in one container is picked up by all of them (the model file's modification time is checked on each request).

To deploy on a cloud VM (AWS EC2, GCP Compute Engine, Azure VM, DigitalOcean...): install Docker, clone the repo, open ports 8501 and 8000, and run the command above. Record the public URL at the top of this README.

## Flood-request simulation (Locust)

```bash
deploy/run_load_test.sh 1  50 60     # containers, users, seconds
deploy/run_load_test.sh 2  50 60
deploy/run_load_test.sh 4  50 60
```

Each run writes `locust/results/containers_<N>_stats.csv`. Interactive mode: `locust -f locust/locustfile.py --host http://localhost:8000` and open http://localhost:8089.

### Results

**Local dry run (already measured).** Docker was not available on the development laptop, so the Locust script was validated by running 1, 2 and 4 *uvicorn worker processes* on one machine (20 users, 40 s each, `POST /predict`, 0 failures). This is a stand-in for container scaling, not a Docker measurement; results come from `locust/results/local_workers_*_stats.csv`.

| API workers | Requests (40 s) | Approx. req/s | Median latency | Mean latency | 99th pct |
|---|---|---|---|---|---|
| 1 | 639 | 16 | 440 ms | 487 ms | 1800 ms |
| 2 | 1242 | 31 | 200 ms | 215 ms | 650 ms |
| 4 | 1267 | 32 | 200 ms | 230 ms | 800 ms |

Going from 1 to 2 workers roughly doubled throughput and halved latency; 4 workers gave no further gain on this laptop, which is CPU-bound (inference is serialised per process by a lock).

**Docker containers (fill in after running on the cloud VM):**

| API containers | Users | Req/s | Median (ms) | Mean (ms) | 99th pct (ms) | Failures |
|---|---|---|---|---|---|---|
| 1 | 50 | | | | | |
| 2 | 50 | | | | | |
| 4 | 50 | | | | | |

## Limitations

* `not_flower` has very few examples; the model is not reliable at rejecting arbitrary non-flower images until more are uploaded.
* Dataset photos are small, uniform web images; very different photos (phone shots, unusual angles) may score lower.
* The SQLite usage log is per container; model, metrics and data are shared through volumes. There is no authentication on the API.
