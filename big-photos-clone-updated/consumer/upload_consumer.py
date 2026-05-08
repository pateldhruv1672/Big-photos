from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import requests

from src.common import hdfs
from src.common.embeddings import record_embedding
from src.common.image_utils import image_info_and_thumbnail, infer_fallback_category, infer_fallback_labels, make_caption
from src.search.hnsw_index import HNSWManager

try:
    from kafka import KafkaConsumer, KafkaProducer
except Exception:  # pragma: no cover
    KafkaConsumer = None
    KafkaProducer = None

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC_UPLOADED = os.getenv("KAFKA_TOPIC_UPLOADED", "image_uploaded")
KAFKA_TOPIC_LABELED = os.getenv("KAFKA_TOPIC_LABELED", "image_labeled")
KAFKA_TOPIC_FAILED = os.getenv("KAFKA_TOPIC_FAILED", "processing_failed")
UPLOAD_METADATA_ROOT = os.getenv("UPLOAD_METADATA_ROOT", "/photos/metadata/uploads")
THUMBNAIL_ROOT = os.getenv("THUMBNAIL_ROOT", "/photos/thumbnails/uploads")
RAY_SERVE_URL = os.getenv("RAY_SERVE_URL", "http://ray-head:8000/predict")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))
POLL_TIMEOUT_MS = int(os.getenv("KAFKA_POLL_TIMEOUT_MS", "1000"))


def _json_default(value: Any) -> str:
    return str(value)


def make_producer():
    if KafkaProducer is None:
        return None
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, default=_json_default).encode("utf-8"),
        retries=3,
    )


def call_ray_serve(image_id: str, image_uri: str) -> Dict[str, Any]:
    try:
        resp = requests.post(RAY_SERVE_URL, json={"image_id": image_id, "image_uri": image_uri}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        print(f"Ray Serve returned {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        print(f"Ray Serve unavailable, using fallback: {exc}")
    labels = infer_fallback_labels(image_id + image_uri)
    category = infer_fallback_category(image_id + image_uri)
    return {"predicted_category": category, "confidence": 0.50, "labels": labels}


def process_event(event: Dict[str, Any]) -> Dict[str, Any]:
    image_id = str(event["image_id"])
    user_id = str(event.get("user_id") or "team_gallery")
    image_uri = str(event["image_uri"])
    file_name = str(event.get("file_name") or f"{image_id}.jpg")

    data = hdfs.read_bytes(image_uri)
    info, thumb = image_info_and_thumbnail(data, max_size=256)
    thumb_path = f"{THUMBNAIL_ROOT.rstrip('/')}/{user_id}/{image_id}.jpg"
    hdfs.write_bytes(thumb_path, thumb, overwrite=True)

    prediction = call_ray_serve(image_id=image_id, image_uri=image_uri)
    category = str(prediction.get("predicted_category") or prediction.get("category") or infer_fallback_category(file_name))
    labels = prediction.get("labels") or infer_fallback_labels(file_name)
    if not isinstance(labels, list):
        labels = [str(labels)]
    caption = make_caption(category, labels, file_name)
    embedding = record_embedding(caption=caption, labels=labels, category=category, dim=EMBEDDING_DIM)
    row = {
        "image_id": image_id,
        "user_id": user_id,
        "image_uri": image_uri,
        "thumbnail_uri": hdfs.to_hdfs_uri(thumb_path),
        "file_name": file_name,
        "dataset": "uploads",
        "caption": caption,
        "labels": labels,
        "vlm_labels": labels,
        "objects": labels[:3],
        "category": category,
        "embedding": embedding,
        "embedding_dim": EMBEDDING_DIM,
        "quality_score": float(prediction.get("confidence") or 0.5),
        "width": int(info.get("width", 0)),
        "height": int(info.get("height", 0)),
        "file_size": len(data),
        "taken_at": datetime.now(timezone.utc),
        "location": "Uploaded",
        "deleted": False,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "processed_at": datetime.now(timezone.utc),
    }
    date_part = datetime.now(timezone.utc).strftime("date=%Y-%m-%d")
    hdfs.write_dataframe_parquet(pd.DataFrame([row]), f"{UPLOAD_METADATA_ROOT.rstrip('/')}/{date_part}")
    mgr = HNSWManager(dim=EMBEDDING_DIM)
    try:
        mgr.load()
    except Exception:
        pass
    mgr.add_or_update(image_id, embedding, row)
    mgr.save(metadata_path=UPLOAD_METADATA_ROOT)
    return row


def main() -> None:
    if KafkaConsumer is None:
        raise SystemExit("kafka-python not installed")
    producer = make_producer()
    consumer = KafkaConsumer(
        KAFKA_TOPIC_UPLOADED,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="big-photos-upload-consumer",
    )
    print(f"Listening on topic {KAFKA_TOPIC_UPLOADED}")
    for msg in consumer:
        event = msg.value
        try:
            row = process_event(event)
            if producer:
                producer.send(KAFKA_TOPIC_LABELED, {"event_type": "LABELED", "image_id": row["image_id"], "event_time": datetime.now(timezone.utc).isoformat()})
                producer.flush(timeout=5)
            print(f"Processed upload {row['image_id']}")
        except Exception as exc:
            print(f"Processing failed for event {event}: {exc}")
            if producer:
                producer.send(KAFKA_TOPIC_FAILED, {"event": event, "error": str(exc), "event_time": datetime.now(timezone.utc).isoformat()})
                producer.flush(timeout=5)
        time.sleep(0.01)


if __name__ == "__main__":
    main()
