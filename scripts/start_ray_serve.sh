#!/usr/bin/env bash
# Start Ray Serve inside the ray-head container.
# Uses `serve run` for a proper Ray Serve deployment (shows in Ray Dashboard).
# Falls back to uvicorn if Ray Serve CLI is not available or cluster has issues.
set -e

cd /app

echo "=== Starting Ray Serve ==="

# Install PyTorch CPU if missing (needed for MobileNet inference)
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true

# Try Ray Serve deployment first (proper deployment, visible in Ray Dashboard)
if command -v serve &>/dev/null && python -c "from ray import serve" 2>/dev/null; then
    echo "Trying Ray Serve deployment..."
    serve run serve.ray_serve_app:deployment --host 0.0.0.0 --port 8000 2>/tmp/serve_err.log &
    SERVE_PID=$!
    sleep 8
    if kill -0 $SERVE_PID 2>/dev/null && curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo "Ray Serve deployment running (PID $SERVE_PID)"
        wait $SERVE_PID
    else
        echo "Ray Serve CLI failed (cluster IP issue), falling back to uvicorn..."
        kill $SERVE_PID 2>/dev/null || true
        exec uvicorn serve.ray_serve_app:app --host 0.0.0.0 --port 8000 --workers 2
    fi
else
    echo "Ray Serve CLI not available — falling back to uvicorn..."
    exec uvicorn serve.ray_serve_app:app --host 0.0.0.0 --port 8000 --workers 2
fi
