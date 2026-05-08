#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_MODEL:-llava:7b}"
BASE="${OLLAMA_BASE_URL:-http://ollama:11434}"

echo "Checking Ollama at ${BASE}"
if ! curl -fsS "${BASE}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is unavailable. The pipeline will use deterministic fallback labels."
  exit 0
fi

echo "Requesting model ${MODEL}"
curl -fsS "${BASE}/api/pull" \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"${MODEL}\"}" || true
