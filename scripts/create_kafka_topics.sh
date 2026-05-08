#!/usr/bin/env bash
set -e

for topic in image_uploaded image_labeled processing_failed; do
  docker compose exec -T kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server localhost:9092 \
    --create --if-not-exists \
    --topic "$topic" \
    --partitions 3 \
    --replication-factor 1
done

echo "Kafka topics ready."
