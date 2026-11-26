"""Flood the prediction API.

Headless example (60 s, 50 users, ramp 10/s):
    locust -f locust/locustfile.py --headless -u 50 -r 10 -t 60s \
           --host http://localhost:8000 --csv locust/results/containers_1
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

SAMPLES = [(p.name, p.read_bytes()) for p in sorted((Path(__file__).parent / "samples").glob("*")) if p.is_file()]


class ModelUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        name, data = random.choice(SAMPLES)
        self.client.post("/predict", files={"file": (name, data)}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
