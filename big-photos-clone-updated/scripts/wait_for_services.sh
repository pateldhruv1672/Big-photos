#!/usr/bin/env bash
set -euo pipefail
HDFS_URL="${HDFS_NAMENODE_HTTP:-http://namenode:9870}"
for i in $(seq 1 90); do
  if curl -fsS "${HDFS_URL}/webhdfs/v1/?op=GETHOMEDIRECTORY&user.name=${HDFS_USER:-root}" >/dev/null 2>&1; then
    echo "HDFS is ready"
    exit 0
  fi
  echo "Waiting for HDFS (${i}/90)..."
  sleep 2
done
echo "HDFS did not become ready" >&2
exit 1
