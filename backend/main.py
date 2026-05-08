"""
Backend API — Distributed Photo Intelligence Platform

Endpoints:
  GET  /health                   — liveness probe
  GET  /image/{image_id}       — demo thumbnails from local MIRFLICKR images
  POST /upload                   — bulk image upload → Kafka
  GET  /gallery?user_id=&limit=  — paginated image gallery
  GET  /search?q=&k=             — HNSW semantic image search
  GET  /stories?user_id=&limit=  — Spark-generated story cards
  GET  /metrics                  — dashboard summary stats
  POST /predict                  — direct Ray Serve proxy
"""

import os
import sys
import json
import uuid
import glob
import logging
import datetime
import re
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from kafka import KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("backend")

sys.path.insert(0, "/app")

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
RAY_SERVE_URL   = os.environ.get("RAY_SERVE_URL", "http://ray-head:8000")
HNSW_INDEX_PATH = "/app/outputs/hnsw_index.bin"
HNSW_MAP_PATH   = "/app/outputs/id_map.json"
UPLOADS_LOCAL   = "/app/outputs/uploads"
EMBEDDING_DIM   = int(os.environ.get("EMBEDDING_DIM", "128"))

ENRICHED_LOCAL  = "/app/outputs/metadata/enriched"
STORIES_LOCAL   = "/app/outputs/metadata/stories"
GALLERY_LOCAL   = "/app/outputs/metadata"

# Local demo images (mirrors generate_sample_dataset.py layout)
MIRFLICKR_IMAGES_DIR = Path("/app/data/mirflickr/images")
_IMAGE_ID_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,200}$")

app = FastAPI(title="Distributed Photo Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_producer: Optional[KafkaProducer] = None


# --------------------------------------------------------------------------- #
# Startup                                                                        #
# --------------------------------------------------------------------------- #

@app.on_event("startup")
def startup():
    global _producer
    try:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        print("Kafka producer connected.")
    except Exception as e:
        print(f"Kafka unavailable (uploads will still work locally): {e}")
        _producer = None


# --------------------------------------------------------------------------- #
# Helpers                                                                        #
# --------------------------------------------------------------------------- #


def _safe_list(value) -> list:
    """Normalize Parquet list/array cells for Python (no ndarray truthiness)."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return [str(parsed)] if parsed is not None else []
        except Exception:
            return [value] if value.strip() else []
    try:
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return []
    except (ValueError, TypeError):
        pass
    return []


def _confidence_value(r: dict) -> float:
    """Prefer explicit confidence; fall back to quality_score."""
    v = r.get("confidence")
    if v is not None:
        try:
            f = float(v)
            if not (f != f):  # not NaN
                return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            pass
    try:
        return float(r.get("quality_score", 0.5))
    except (TypeError, ValueError):
        return 0.5


def _resolve_demo_image(image_id: str) -> Optional[Path]:
    """
    Map image_id to a file under MIRFLICKR_IMAGES_DIR or UPLOADS_LOCAL.
    Returns None if invalid id or file missing. Prevents path traversal.
    """
    if not image_id or not _IMAGE_ID_SAFE.match(image_id):
        return None
    # 1. Check MIRFLICKR images
    base = MIRFLICKR_IMAGES_DIR.resolve()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = (base / f"{image_id}{ext}").resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    # 2. Check uploads (files are stored as {uuid}_{filename})
    uploads = Path(UPLOADS_LOCAL).resolve()
    if uploads.is_dir():
        for f in uploads.iterdir():
            if f.name.startswith(image_id) and f.is_file():
                return f
    return None


def _media_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")


def _load_all_parquet(base_dir: str) -> pd.DataFrame:
    """Recursively load all parquet files under base_dir."""
    files = glob.glob(f"{base_dir}/**/*.parquet", recursive=True)
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _row_to_photo(r: dict) -> dict:
    """Normalize a metadata row to a frontend-friendly photo dict."""
    vlm_labels_list = _safe_list(r.get("vlm_labels"))
    tags_list = _safe_list(r.get("tags"))
    labels = vlm_labels_list if vlm_labels_list else tags_list
    scene_labels = _safe_list(r.get("scene_labels"))
    vlm_out = vlm_labels_list if vlm_labels_list else labels
    annotation_labels = _safe_list(r.get("annotation_labels"))
    return {
        "image_id": r.get("image_id", ""),
        "image_uri": r.get("image_uri", ""),
        "file_name": r.get("file_name", ""),
        "category": r.get("category", "general"),
        "caption": r.get("caption", ""),
        "labels": [str(x) for x in labels[:5]],
        "vlm_labels": [str(x) for x in vlm_out[:6]],
        "scene_labels": [str(x) for x in scene_labels[:4]],
        "annotation_labels": [str(x) for x in annotation_labels[:10]],
        "location": r.get("location", ""),
        "taken_at": str(r.get("taken_at", "")),
        "quality_score": float(r.get("quality_score", 0.5)),
        "confidence": _confidence_value(r),
        "dataset": r.get("dataset", ""),
    }


# --------------------------------------------------------------------------- #
# Endpoints                                                                     #
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    return {
        "status": "ok",
        "kafka": _producer is not None,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.get("/image/{image_id}")
def serve_image(image_id: str):
    """Serve a local demo image file (path-safe)."""
    path = _resolve_demo_image(image_id)
    if path is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Image not found", "image_id": image_id},
        )
    return FileResponse(path, media_type=_media_type(path))


@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    user_id: str = Query("demo_user"),
):
    """Bulk-upload images. Saves locally and publishes Kafka events."""
    events = []
    local_dir = Path(UPLOADS_LOCAL)
    local_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        image_id = str(uuid.uuid4())
        local_path = local_dir / f"{image_id}_{f.filename}"
        local_path.write_bytes(await f.read())
        date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        # Date-partitioned HDFS path — matches architecture diagram
        image_uri = f"hdfs://namenode:9000/photos/uploads/{user_id}/date={date_str}/{local_path.name}"
        event = {
            "event_type": "image_uploaded",
            "image_id": image_id,
            "user_id": user_id,
            "image_uri": image_uri,
            "file_name": f.filename,
            "uploaded_at": datetime.datetime.utcnow().isoformat(),
        }
        events.append(event)
        if _producer:
            _producer.send("image_uploaded", event)

    if _producer:
        _producer.flush()

    return {
        "uploaded": len(events),
        "kafka_queued": _producer is not None,
        "events": events,
    }


@app.get("/gallery")
def gallery(
    user_id: Optional[str] = Query(None, description="Filter by user_id"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return paginated gallery of processed photos."""
    dfs = []
    enriched = _load_all_parquet(ENRICHED_LOCAL)
    if not enriched.empty:
        dfs.append(enriched)
    uploads = _load_all_parquet(f"{GALLERY_LOCAL}/uploads")
    if not uploads.empty:
        dfs.append(uploads)
    if not dfs:
        dfs.append(_load_all_parquet(f"{GALLERY_LOCAL}/basic"))
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not df.empty and "image_id" in df.columns:
        df = df.drop_duplicates(subset=["image_id"], keep="last")
    if df.empty:
        return {"photos": [], "total": 0}

    if user_id:
        df = df[df["user_id"] == user_id]
    if category:
        df = df[df["category"] == category]

    total = len(df)
    page = df.iloc[offset: offset + limit]
    photos = [_row_to_photo(r) for r in page.to_dict("records")]
    return {"photos": photos, "total": total, "offset": offset, "limit": limit}


@app.get("/search")
def search(
    q: str = Query(..., description="Search query text"),
    k: int = Query(10, ge=1, le=50),
):
    """
    HNSW semantic image search.

    Embeds the query, searches the index, and returns the k nearest images.
    Falls back to keyword search in enriched metadata if index is unavailable.
    """
    from src.common.embeddings import get_text_embedding

    query_vec = get_text_embedding(q, dim=EMBEDDING_DIM)

    # Try HNSW index
    try:
        from src.search.hnsw_index import HNSWIndex
        if Path(HNSW_INDEX_PATH).exists():
            index = HNSWIndex(dim=EMBEDDING_DIM, path=HNSW_INDEX_PATH, map_path=HNSW_MAP_PATH)
            hits = index.search(query_vec, k=k)
            # Enrich hits with metadata
            df = _load_all_parquet(ENRICHED_LOCAL)
            results = []
            for hit in hits:
                iid = hit["image_id"]
                if not df.empty and iid:
                    row = df[df["image_id"] == iid]
                    if not row.empty:
                        photo = _row_to_photo(row.iloc[0].to_dict())
                        photo["distance"] = hit["distance"]
                        results.append(photo)
                    else:
                        results.append({"image_id": iid, "distance": hit["distance"]})
                else:
                    results.append(hit)
            return {"query": q, "results": results, "method": "hnsw"}
    except Exception as e:
        print(f"HNSW search failed: {e}")

    # Fallback: keyword match in enriched metadata
    df = _load_all_parquet(ENRICHED_LOCAL)
    if df.empty:
        return {"query": q, "results": [], "method": "none"}

    q_lower = q.lower()

    def _matches(r):
        vlm = _safe_list(r.get("vlm_labels"))
        tags = _safe_list(r.get("tags"))
        fields = [
            str(r.get("caption", "")),
            str(r.get("category", "")),
            str(r.get("location", "")),
            " ".join(str(x) for x in vlm),
            " ".join(str(x) for x in tags),
        ]
        return any(q_lower in f.lower() for f in fields)

    matches = df[df.apply(_matches, axis=1)].head(k)
    return {
        "query": q,
        "results": [_row_to_photo(r) for r in matches.to_dict("records")],
        "method": "keyword_fallback",
    }


@app.get("/stories")
def stories(
    user_id: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    """Return Spark-generated story cards."""
    # Try local stories parquet first
    df = _load_all_parquet(STORIES_LOCAL)

    if df.empty:
        # Try HDFS aggregates via WebHDFS
        try:
            from src.common.hdfs import HDFSClient
            import pyarrow.parquet as pq
            import pyarrow as pa

            client = HDFSClient()
            file_list = client.listdir("/photos/aggregates/final_stories")
            parquet_files = [f["pathSuffix"] for f in file_list if f["pathSuffix"].endswith(".parquet")]
            if parquet_files:
                dfs = []
                for fname in parquet_files:
                    data = client.get(f"/photos/aggregates/final_stories/{fname}")
                    buf = pa.BufferReader(pa.py_buffer(data))
                    dfs.append(pq.read_table(buf).to_pandas())
                df = pd.concat(dfs, ignore_index=True)
        except Exception:
            pass

    if df.empty:
        # Generate on-the-fly from enriched metadata
        enriched = _load_all_parquet(ENRICHED_LOCAL)
        if not enriched.empty and "location" in enriched.columns:
            grouped = (
                enriched.groupby(["user_id", "location"])
                .agg(
                    photo_count=("image_id", "count"),
                    image_ids=("image_id", list),
                    cover_image_uri=("image_uri", "first"),
                )
                .reset_index()
            )
            df = grouped[grouped["photo_count"] >= 1].head(limit)
            if not df.empty:
                df["story_id"] = "story_" + df["user_id"] + "_" + df["location"].str.replace(" ", "_")
                df["title"] = "Remember your trip to " + df["location"]
                df["summary"] = "A collection of memories from " + df["location"]

    if df.empty:
        return {"stories": [], "note": "Run `make part4` to generate stories."}

    if user_id and "user_id" in df.columns:
        df = df[df["user_id"] == user_id]

    story_list = []
    for r in df.head(limit).to_dict("records"):
        img_ids = _safe_list(r.get("image_ids"))
        story_list.append({
            "story_id": r.get("story_id", ""),
            "title": r.get("title", ""),
            "summary": r.get("summary", ""),
            "location": r.get("location", ""),
            "photo_count": int(r.get("photo_count", 0)),
            "image_ids": img_ids[:10],
            "cover_image_uri": r.get("cover_image_uri", ""),
        })

    return {"stories": story_list, "total": len(story_list)}


@app.get("/metrics")
def metrics():
    """Dashboard summary statistics."""
    enriched = _load_all_parquet(ENRICHED_LOCAL)
    basic = _load_all_parquet(f"{GALLERY_LOCAL}/basic")

    total_enriched = len(enriched)
    total_basic = len(basic)
    categories = {}
    if not enriched.empty and "category" in enriched.columns:
        categories = enriched["category"].value_counts().to_dict()

    hnsw_size = 0
    if Path(HNSW_MAP_PATH).exists():
        try:
            hnsw_size = len(json.loads(Path(HNSW_MAP_PATH).read_text()))
        except Exception:
            pass

    return {
        "total_images_basic": total_basic,
        "total_images_enriched": total_enriched,
        "hnsw_index_size": hnsw_size,
        "category_distribution": categories,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.post("/predict")
def predict_proxy(payload: dict):
    """Proxy to Ray Serve /predict."""
    try:
        r = requests.post(f"{RAY_SERVE_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ray Serve unavailable: {e}")
