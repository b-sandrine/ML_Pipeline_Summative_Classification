#!/usr/bin/env bash
# Usage: deploy/run_load_test.sh <containers> [users] [seconds]
# Scales the API to N containers, floods /predict with Locust, saves CSVs.
set -euo pipefail
N=${1:?containers}; USERS=${2:-50}; SECS=${3:-60}
docker compose up -d --build --scale api="$N"
sleep 30   # let replicas load TensorFlow
locust -f locust/locustfile.py --headless -u "$USERS" -r 10 -t "${SECS}s" \
  --host http://localhost:8000 --csv "locust/results/containers_${N}"
