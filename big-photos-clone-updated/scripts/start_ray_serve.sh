#!/usr/bin/env bash
set -euo pipefail
cd /app
ray start --head --dashboard-host=0.0.0.0 --port=6379 || true
serve run serve.ray_serve_app:app --host 0.0.0.0 --port 8000
