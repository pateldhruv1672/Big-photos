#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${HDFS_NAMENODE_HTTP:-http://namenode:9870}"
USER_NAME="${HDFS_USER:-root}"
mkdir_hdfs() {
  local path="$1"
  curl -fsS -X PUT "${BASE_URL}/webhdfs/v1${path}?op=MKDIRS&user.name=${USER_NAME}" >/dev/null || true
}
for p in \
  /photos/raw/images \
  /photos/raw/team_gallery/images \
  /photos/raw/mirflickr25k/images \
  /photos/raw/uploads \
  /photos/thumbnails/team_gallery \
  /photos/thumbnails/mirflickr25k \
  /photos/thumbnails/uploads \
  /photos/metadata/basic \
  /photos/metadata/imported \
  /photos/metadata/ui_active \
  /photos/metadata/mirflickr25k/basic \
  /photos/metadata/mirflickr25k/enriched \
  /photos/metadata/uploads \
  /photos/metadata/deletes \
  /photos/aggregates/user_gallery \
  /photos/aggregates/final_stories \
  /photos/aggregates/dashboard_metrics \
  /photos/models/image_classifier \
  /photos/vector_index/current \
  /photos/events/uploads \
  /photos/events/deletes; do
  mkdir_hdfs "$p"
done
echo "Initialized Big Photos HDFS layout"
