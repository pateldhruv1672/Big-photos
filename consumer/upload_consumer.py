"""
Kafka Upload Consumer

Listens on the `image_uploaded` topic. For each event:
  1. Uploads the image file to HDFS (/photos/uploads/<user_id>/)
  2. Calls Ray Serve /predict to classify the image
  3. Builds a deterministic embedding for vector search
  4. Updates the HNSW index with the embedding
  5. Appends a Parquet metadata row to HDFS uploads partition
  6. Publishes result to `image_labeled` topic (or `processing_failed` on error)

Designed to run as a long-lived Docker service alongside Kafka, HDFS, and Ray.
"""

import os
import sys
import json
import time
import logging
import datetime
from pathlib import Path

import requests
from kafka import KafkaConsumer, KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("consumer")

sys.path.insert(0, "/app")

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
RAY_SERVE_URL   = os.environ.get("RAY_SERVE_URL", "http://ray-head:8000")
HNSW_INDEX_PATH = "/app/outputs/hnsw_index.bin"
HNSW_MAP_PATH   = "/app/outputs/id_map.json"
UPLOADS_LOCAL   = "/app/outputs/uploads"
METADATA_UPLOADS_LOCAL = "/app/outputs/metadata/uploads"
EMBEDDING_DIM   = int(os.environ.get("EMBEDDING_DIM", "128"))


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _call_ray_serve(event: dict) -> dict:
    """Call Ray Serve /predict with image data for MobileNet inference."""
    import base64
    try:
        payload = {
            "image_id": event.get("image_id", ""),
            "image_uri": event.get("image_uri", ""),
            "caption": "",
            "labels": [],
        }
        # Try to include base64-encoded image for MobileNet inference
        local_path = Path(UPLOADS_LOCAL)
        matches = list(local_path.glob(f"{event.get('image_id', 'xxx')}_*"))
        if matches:
            img_bytes = matches[0].read_bytes()
            payload["image_base64"] = base64.b64encode(img_bytes).decode()

        r = requests.post(f"{RAY_SERVE_URL}/predict", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Ray Serve call failed: {e}")
    return {
        "image_id": event.get("image_id", ""),
        "predicted_category": "general",
        "confidence": 0.5,
        "labels": ["general"],
    }


def _get_embedding(event: dict, prediction: dict) -> list:
    """Build embedding from prediction labels + tags."""
    from src.common.embeddings import get_text_embedding
    parts = prediction.get("labels", []) or []
    parts += [prediction.get("predicted_category", "general")]
    tags = event.get("tags", []) or []
    if isinstance(tags, list):
        parts.extend(tags)
    text = " ".join(str(p) for p in parts) or event.get("image_id", "unknown")
    return get_text_embedding(text, dim=EMBEDDING_DIM)


def _update_hnsw(image_id: str, embedding: list) -> None:
    """Add or update the HNSW index with the new embedding."""
    try:
        from src.search.hnsw_index import HNSWIndex
        Path(HNSW_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        index = HNSWIndex(
            dim=EMBEDDING_DIM,
            path=HNSW_INDEX_PATH,
            map_path=HNSW_MAP_PATH,
        )
        index.add(image_id, embedding)
        print(f"HNSW index updated for {image_id}")
    except Exception as e:
        print(f"HNSW update failed: {e}")


def _upload_to_hdfs(event: dict) -> None:
    """Copy the locally-saved upload file to HDFS under /photos/uploads/{user_id}/date=YYYY-MM-DD/."""
    try:
        from src.common.hdfs import HDFSClient
        local_path = Path(UPLOADS_LOCAL)
        # Find the file matching image_id
        matches = list(local_path.glob(f"{event['image_id']}_*"))
        if not matches:
            return
        file_path = matches[0]
        user_id = event.get("user_id", "demo_user")
        date_str = datetime.date.today().isoformat()
        # Date-partitioned path as shown in architecture diagram
        hdfs_dir = f"/photos/uploads/{user_id}/date={date_str}"
        client = HDFSClient()
        client.mkdirs(hdfs_dir)
        uri = client.put_local(file_path, hdfs_dir)
        # Update the event's image_uri to reflect the real HDFS path
        event["image_uri"] = uri
        print(f"Uploaded to HDFS: {uri}")
    except Exception as e:
        print(f"HDFS upload failed: {e}")


def _infer_category(labels: list) -> str:
    """Map predicted labels to a display category."""
    tag_set = {str(t).lower() for t in labels}
    scores = {
        "food":   len(tag_set & {"food", "meal", "restaurant", "cuisine", "dish", "coffee"}) * 3,
        "event":  len(tag_set & {"event", "concert", "festival", "party", "wedding", "sport", "baby"}) * 3,
        "art":    len(tag_set & {"art", "painting", "sculpture", "museum", "gallery", "flower"}) * 3,
        "travel": len(tag_set & {"travel", "tourism", "vacation", "trip", "holiday", "sea", "river", "lake", "sunset"}) * 2,
        "city":   len(tag_set & {"city", "urban", "street", "building", "skyline", "bridge", "architecture", "structures", "indoor", "car", "transport"}) * 2,
        "people": len(tag_set & {"people", "portrait", "face", "crowd", "person", "selfie", "female", "male"}) * 2,
        "nature": len(tag_set & {"nature", "landscape", "mountain", "beach", "forest", "ocean", "lake", "sky", "tree", "plant_life", "clouds", "sunset", "water", "night", "animals", "bird", "dog"}) * 1,
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _append_metadata(event: dict, prediction: dict, embedding: list) -> None:
    """Append a Parquet metadata row to the uploads partition."""
    try:
        import pandas as pd
        labels = prediction.get("labels", [])
        category = _infer_category(labels)
        row = {
            "image_id": event.get("image_id"),
            "user_id": event.get("user_id", "demo_user"),
            "image_uri": event.get("image_uri", ""),
            "file_name": event.get("file_name", ""),
            "dataset": "UPLOAD",
            "category": category,
            "caption": "",
            "vlm_labels": prediction.get("labels", []),
            "objects": prediction.get("labels", []),
            "confidence": prediction.get("confidence", 0.5),
            "location": event.get("location", "Unknown"),
            "timestamp": event.get("taken_at", datetime.datetime.utcnow().isoformat()),
            "embedding": embedding,
            "embedding_dim": len(embedding),
            "quality_score": prediction.get("confidence", 0.5),
            "processed_at": datetime.datetime.utcnow().isoformat(),
            "date": datetime.date.today().isoformat(),
        }
        date_str = datetime.date.today().isoformat()
        local_dir = Path(METADATA_UPLOADS_LOCAL) / f"date={date_str}"
        local_dir.mkdir(parents=True, exist_ok=True)
        # Append-style: each event gets its own file (simple, avoids locks)
        out_path = local_dir / f"{row['image_id']}.parquet"
        pd.DataFrame([row]).to_parquet(out_path, index=False)
        print(f"Metadata row saved: {out_path}")

        # Also push to HDFS
        try:
            from src.common.hdfs import HDFSClient
            client = HDFSClient()
            hdfs_dir = f"/photos/metadata/uploads/date={date_str}"
            client.mkdirs(hdfs_dir)
            client.put(f"{hdfs_dir}/{row['image_id']}.parquet", out_path)
        except Exception as e:
            print(f"HDFS metadata push failed: {e}")
    except Exception as e:
        print(f"Metadata append failed: {e}")


# --------------------------------------------------------------------------- #
# Main loop                                                                     #
# --------------------------------------------------------------------------- #

def process_event(event: dict, producer: KafkaProducer) -> None:
    image_id = event.get("image_id", "unknown")
    print(f"\n--- Processing image: {image_id} ---")

    try:
        # 1. Upload image to HDFS
        _upload_to_hdfs(event)

        # 2. Classify via Ray Serve
        prediction = _call_ray_serve(event)
        print(f"Prediction: {prediction.get('predicted_category')} ({prediction.get('confidence', 0):.2f})")

        # 3. Build embedding
        embedding = _get_embedding(event, prediction)

        # 4. Update HNSW index
        _update_hnsw(image_id, embedding)

        # 5. Append metadata
        _append_metadata(event, prediction, embedding)

        # 6. Publish success event
        result = {
            **event,
            "status": "LABELED",
            "predicted_category": prediction.get("predicted_category", "general"),
            "confidence": prediction.get("confidence", 0.5),
            "labels": prediction.get("labels", ["general"]),
            "processed_at": datetime.datetime.utcnow().isoformat(),
        }
        producer.send("image_labeled", result)
        producer.flush()
        print(f"Published image_labeled for {image_id}")

    except Exception as e:
        print(f"Error processing {image_id}: {e}")
        error_event = {**event, "status": "FAILED", "error": str(e)}
        try:
            producer.send("processing_failed", error_event)
            producer.flush()
        except Exception:
            pass


def run():
    print("Starting upload consumer...")
    while True:
        try:
            consumer = KafkaConsumer(
                "image_uploaded",
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_deserializer=lambda v: json.loads(v.decode()),
                auto_offset_reset="earliest",
                group_id="upload-consumer-group",
            )
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode(),
            )
            print(f"Connected to Kafka at {KAFKA_BOOTSTRAP}. Waiting for messages...")
            for msg in consumer:
                process_event(msg.value, producer)
        except Exception as e:
            print(f"Consumer error (will retry in 5s): {e}")
            time.sleep(5)


if __name__ == "__main__":
    run()
