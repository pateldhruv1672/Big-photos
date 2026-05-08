#!/usr/bin/env bash
# Wait for all required services before running pipeline steps.
# Part 5 spec: "Wait for HDFS, Spark, Kafka, Ray, Ollama, backend"
set -e

MAX_WAIT=180   # seconds per service
INTERVAL=3

_wait_for() {
  local name="$1"
  local check_cmd="$2"
  local elapsed=0
  echo -n "Waiting for $name..."
  until eval "$check_cmd" >/dev/null 2>&1; do
    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
    if [ $elapsed -ge $MAX_WAIT ]; then
      echo " TIMEOUT after ${MAX_WAIT}s"
      exit 1
    fi
    echo -n "."
  done
  echo " ready."
}

# 1. HDFS NameNode
_wait_for "HDFS NameNode" "curl -sf http://localhost:9870"

# 2. Spark Master
_wait_for "Spark Master" "curl -sf http://localhost:8081"

# 3. Kafka
_wait_for "Kafka" "docker compose exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list"

# 4. Ray dashboard
_wait_for "Ray" "curl -sf http://localhost:8265"

# 5. Ollama (best-effort — GPU may not be available)
echo -n "Waiting for Ollama (best-effort)..."
elapsed=0
until curl -sf http://localhost:11435 >/dev/null 2>&1; do
  sleep $INTERVAL
  elapsed=$((elapsed + INTERVAL))
  if [ $elapsed -ge 30 ]; then
    echo " not available (skipping — fallback labels will be used)."
    break
  fi
  echo -n "."
done
if curl -sf http://localhost:11435 >/dev/null 2>&1; then
  echo " ready."
fi

# 6. Backend API (only if container is running — it may not be up yet during init)
if docker compose ps backend 2>/dev/null | grep -q "running"; then
  _wait_for "Backend API" "curl -sf http://localhost:8080/health"
else
  echo "Backend not yet started — skipping health check."
fi

echo "All core services are up."
