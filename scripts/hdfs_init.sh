#!/usr/bin/env bash
set -e

echo "Initializing HDFS directory layout..."

CMD="docker compose exec -T namenode hdfs dfs"

# Raw image storage
$CMD -mkdir -p /photos/raw/images            || true

# Upload landing zone (date-partitioned per user)
$CMD -mkdir -p /photos/uploads               || true

# Metadata layers
$CMD -mkdir -p /photos/metadata/basic        || true
$CMD -mkdir -p /photos/metadata/enriched     || true
$CMD -mkdir -p /photos/metadata/uploads      || true

# Spark aggregates (matches architecture diagram)
$CMD -mkdir -p /photos/aggregates/user_gallery    || true
$CMD -mkdir -p /photos/aggregates/search_metadata || true
$CMD -mkdir -p /photos/aggregates/final_stories   || true
$CMD -mkdir -p /photos/aggregates/dashboard_metrics || true

# Model artifacts
$CMD -mkdir -p /photos/models/image_classifier || true

# HNSW vector index
$CMD -mkdir -p /photos/vector_index          || true

echo "HDFS folders initialized."
$CMD -ls /photos/
