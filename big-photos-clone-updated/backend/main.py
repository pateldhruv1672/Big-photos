from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from src.common import hdfs
from src.common.embeddings import cosine_similarity, embedding_to_list, record_embedding, text_embedding
from src.common.image_utils import image_info_and_thumbnail, infer_fallback_category, infer_fallback_labels, make_caption
from src.search.hnsw_index import HNSWManager, build_from_hdfs, load_active_metadata

try:
    from kafka import KafkaProducer
except Exception:  # pragma: no cover
    KafkaProducer = None

UI_ACTIVE_METADATA_ROOT = os.getenv("UI_ACTIVE_METADATA_ROOT", "/photos/metadata/ui_active")
UPLOAD_METADATA_ROOT = os.getenv("UPLOAD_METADATA_ROOT", "/photos/metadata/uploads")
DELETE_METADATA_ROOT = os.getenv("DELETE_METADATA_ROOT", "/photos/metadata/deletes")
UPLOAD_RAW_ROOT = os.getenv("UPLOAD_RAW_ROOT", "/photos/raw/uploads")
THUMBNAIL_ROOT = os.getenv("THUMBNAIL_ROOT", "/photos/thumbnails/uploads")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "team_gallery")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC_UPLOADED = os.getenv("KAFKA_TOPIC_UPLOADED", "image_uploaded")
KAFKA_TOPIC_DELETED = os.getenv("KAFKA_TOPIC_DELETED", "image_deleted")
SYNC_UPLOAD_FALLBACK = os.getenv("SYNC_UPLOAD_FALLBACK", "false").lower() == "true"
STREAM_IMAGES = os.getenv("STREAM_IMAGES", "true").lower() == "true"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))

app = FastAPI(title="Big Photos Backend", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metadata_df = pd.DataFrame()
metadata_by_id: Dict[str, Dict[str, Any]] = {}
hnsw = HNSWManager(dim=EMBEDDING_DIM)
producer = None


def _json_default(value: Any) -> str:
    return str(value)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    try:
        if pd.isna(value) and not isinstance(value, (list, tuple, dict)):
            return []
    except Exception:
        pass
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if isinstance(value, str):
        if value.startswith("["):
            try:
                return [str(x) for x in json.loads(value)]
            except Exception:
                pass
        return [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]
    return [str(value)]


def _load_metadata() -> pd.DataFrame:
    df = load_active_metadata()
    if df.empty:
        return pd.DataFrame()
    df = df.drop_duplicates(subset=["image_id"], keep="last")
    return df.reset_index(drop=True)


def _image_path_candidates(image_id: str) -> List[str]:
    candidates = []
    numeric = "".join(ch for ch in image_id if ch.isdigit())
    if numeric:
        candidates.append(f"/photos/raw/team_gallery/images/{numeric[-1]}/{numeric}.jpg")
        candidates.append(f"/photos/raw/team_gallery/images/{numeric}.jpg")
        candidates.append(f"/photos/raw/team_gallery/images/{numeric[-1]}/im{numeric}.jpg")
    return candidates


def _resolve_image_uri(row: Dict[str, Any]) -> str | None:
    uri = row.get("image_uri")
    if uri and isinstance(uri, str):
        try:
            if hdfs.exists(uri):
                return uri
        except Exception:
            pass
    image_id = str(row.get("image_id") or "")
    for path in _image_path_candidates(image_id):
        try:
            if hdfs.exists(path):
                return hdfs.to_hdfs_uri(path)
        except Exception:
            continue
    return None


def reload_state() -> Dict[str, int]:
    global metadata_df, metadata_by_id, hnsw
    metadata_df = _load_metadata()
    metadata_by_id = {}
    if not metadata_df.empty:
        for row in metadata_df.to_dict("records"):
            image_id = str(row.get("image_id"))
            if image_id and image_id != "None":
                metadata_by_id[image_id] = row
    try:
        hnsw.load()
    except Exception as exc:
        print(f"HNSW load failed: {exc}")
    return {"metadata_rows": len(metadata_by_id), "hnsw_vectors": len(hnsw.label_to_image)}


def get_producer():
    global producer
    if producer is not None:
        return producer
    if KafkaProducer is None:
        return None
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v, default=_json_default).encode("utf-8"),
            retries=3,
        )
    except Exception as exc:
        print(f"Kafka unavailable: {exc}")
        producer = None
    return producer


def publish(topic: str, event: Dict[str, Any]) -> bool:
    prod = get_producer()
    if not prod:
        return False
    prod.send(topic, event)
    prod.flush(timeout=5)
    return True


@app.on_event("startup")
def startup_event() -> None:
    def _warm() -> None:
        try:
            reload_state()
        except Exception as exc:
            print(f"Startup metadata warmup failed: {exc}")

    threading.Thread(target=_warm, daemon=True).start()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "metadata_rows": len(metadata_by_id), "hnsw_vectors": len(hnsw.label_to_image)}


@app.post("/api/refresh")
def refresh() -> Dict[str, int]:
    return reload_state()


@app.post("/api/reload")
def reload_api() -> Dict[str, int]:
    return reload_state()


@app.get("/api/gallery")
def gallery(user_id: str = DEFAULT_USER_ID, limit: int = Query(100, le=1000), offset: int = 0) -> Dict[str, Any]:
    if metadata_df.empty:
        return {"items": [], "count": 0, "offset": offset, "total_known": 0}
    df = metadata_df.copy()
    if "user_id" in df.columns:
        df = df[df["user_id"].fillna(DEFAULT_USER_ID).astype(str) == user_id]
    if "updated_at" in df.columns:
        try:
            df = df.sort_values("updated_at", ascending=False)
        except Exception:
            pass
    rows = df.iloc[offset : offset + limit].to_dict("records")
    items = [_row_to_card(row) for row in rows]
    return {"items": items, "count": len(items), "offset": offset, "total_known": int(len(df))}


def _row_to_card(row: Dict[str, Any], score: float | None = None) -> Dict[str, Any]:
    image_id = str(row.get("image_id"))
    labels = _as_list(row.get("vlm_labels")) or _as_list(row.get("labels")) or _as_list(row.get("tags"))
    image_uri = row.get("image_uri")
    thumbnail_uri = row.get("thumbnail_uri")
    out = {
        "image_id": image_id,
        "caption": row.get("caption") or row.get("file_name") or image_id,
        "category": row.get("category") or (labels[0] if labels else "photo"),
        "labels": labels,
        "thumbnail_url": f"/api/thumb/{image_id}",
        "image_url": f"/api/image/{image_id}",
        "hdfs_image_uri": str(image_uri) if image_uri is not None else None,
        "hdfs_thumbnail_uri": str(thumbnail_uri) if thumbnail_uri is not None else None,
        "predicted_label": row.get("pred_label") or row.get("predicted_label"),
        "predicted_score": row.get("pred_score") if row.get("pred_score") is not None else row.get("predicted_score"),
        "model_version": row.get("model_version"),
        "location": row.get("location"),
        "taken_at": str(row.get("taken_at")) if row.get("taken_at") is not None else None,
    }
    if score is not None:
        out["score"] = float(score)
    return out


def _row_or_404(image_id: str) -> Dict[str, Any]:
    row = metadata_by_id.get(image_id)
    if not row:
        raise HTTPException(status_code=404, detail="image_id not found")
    return row


@app.get("/api/thumb/{image_id}")
def get_thumbnail(image_id: str):
    row = _row_or_404(image_id)
    thumb_uri = row.get("thumbnail_uri")
    try:
        if thumb_uri and isinstance(thumb_uri, str) and hdfs.exists(thumb_uri):
            if STREAM_IMAGES:
                return StreamingResponse(hdfs.iter_bytes(thumb_uri), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
            return Response(content=hdfs.read_bytes(thumb_uri), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
        resolved = _resolve_image_uri(row)
        if not resolved:
            raise HTTPException(status_code=404, detail="thumbnail source image not found in HDFS")
        image_data = hdfs.read_bytes(resolved)
        _, data = image_info_and_thumbnail(image_data, max_size=256)
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"thumbnail read failed: {exc}")


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    row = _row_or_404(image_id)
    try:
        image_uri = _resolve_image_uri(row)
        if not image_uri:
            raise HTTPException(status_code=404, detail="image not found in HDFS")
        if STREAM_IMAGES:
            return StreamingResponse(hdfs.iter_bytes(image_uri), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
        return Response(content=hdfs.read_bytes(image_uri), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"image read failed: {exc}")


@app.post("/api/search")
def search(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or 20)
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    if not hnsw.label_to_image:
        try:
            hnsw.load()
        except Exception:
            pass
    results = hnsw.search_text(query, top_k=top_k) if hnsw.label_to_image else []
    if not results and not metadata_df.empty and "embedding" in metadata_df.columns:
        q = text_embedding(query, dim=EMBEDDING_DIM)
        tmp = []
        for row in metadata_df.to_dict("records"):
            emb = embedding_to_list(row.get("embedding"), dim=EMBEDDING_DIM)
            tmp.append((str(row.get("image_id")), cosine_similarity(q, emb)))
        tmp.sort(key=lambda x: x[1], reverse=True)
        results = tmp[:top_k]
    out = []
    for image_id, score in results:
        row = metadata_by_id.get(image_id) or hnsw.item_meta.get(image_id, {"image_id": image_id})
        out.append(_row_to_card(row, score=score))
    return {"results": out, "query": query}


def _sync_process_upload(row: Dict[str, Any], file_name: str) -> Dict[str, Any]:
    data = hdfs.read_bytes(row["image_uri"])
    info, thumb = image_info_and_thumbnail(data, max_size=256)
    thumb_path = f"{THUMBNAIL_ROOT.rstrip('/')}/{row['user_id']}/{row['image_id']}.jpg"
    hdfs.write_bytes(thumb_path, thumb, overwrite=True)
    labels = infer_fallback_labels(file_name)
    category = infer_fallback_category(file_name)
    caption = make_caption(category, labels, file_name)
    embedding = record_embedding(caption=caption, labels=labels, category=category, dim=EMBEDDING_DIM)
    enriched = {
        **row,
        "thumbnail_uri": hdfs.to_hdfs_uri(thumb_path),
        "caption": caption,
        "vlm_labels": labels,
        "labels": labels,
        "objects": labels[:3],
        "category": category,
        "embedding": embedding,
        "embedding_dim": EMBEDDING_DIM,
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "quality_score": 0.5,
        "processed_at": datetime.now(timezone.utc),
        "deleted": False,
    }
    date_part = datetime.now(timezone.utc).strftime("date=%Y-%m-%d")
    hdfs.write_dataframe_parquet(pd.DataFrame([enriched]), f"{UPLOAD_METADATA_ROOT.rstrip('/')}/{date_part}")
    hnsw.add_or_update(str(row["image_id"]), embedding, enriched)
    hnsw.save(metadata_path=UPLOAD_METADATA_ROOT)
    return enriched


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...), user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    uploaded = []
    date_part = datetime.now(timezone.utc).strftime("date=%Y-%m-%d")
    for file in files:
        data = await file.read()
        image_id = uuid.uuid4().hex
        ext = os.path.splitext(file.filename or "image.jpg")[1].lower() or ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        raw_path = f"{UPLOAD_RAW_ROOT.rstrip('/')}/{user_id}/{date_part}/{image_id}{ext}"
        hdfs.write_bytes(raw_path, data, overwrite=True)
        event = {
            "event_type": "UPLOAD",
            "image_id": image_id,
            "user_id": user_id,
            "image_uri": hdfs.to_hdfs_uri(raw_path),
            "file_name": file.filename or f"{image_id}{ext}",
            "event_time": datetime.now(timezone.utc).isoformat(),
        }
        sent = publish(KAFKA_TOPIC_UPLOADED, event)
        if (not sent) and SYNC_UPLOAD_FALLBACK:
            _sync_process_upload(event, file.filename or image_id)
        uploaded.append({"image_id": image_id, "file_name": file.filename, "queued": sent, "sync_fallback": (not sent) and SYNC_UPLOAD_FALLBACK})
    reload_state()
    return {"uploaded": uploaded}


@app.post("/api/delete")
def delete_images(payload: Dict[str, Any]) -> Dict[str, Any]:
    ids = payload.get("image_ids") or []
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="image_ids must be a list")
    date_part = datetime.now(timezone.utc).strftime("date=%Y-%m-%d")
    rows = [
        {
            "image_id": str(x),
            "user_id": payload.get("user_id", DEFAULT_USER_ID),
            "deleted_at": datetime.now(timezone.utc),
            "reason": payload.get("reason", "user_delete"),
        }
        for x in ids
    ]
    if rows:
        hdfs.write_dataframe_parquet(pd.DataFrame(rows), f"{DELETE_METADATA_ROOT.rstrip('/')}/{date_part}")
        publish(KAFKA_TOPIC_DELETED, {"event_type": "DELETE", "image_ids": [r["image_id"] for r in rows], "event_time": datetime.now(timezone.utc).isoformat()})
        # For consistency with a file-backed HNSW index, rebuild after logical deletes.
        try:
            build_from_hdfs()
        except Exception as exc:
            print(f"HNSW rebuild after delete failed: {exc}")
    reload_state()
    return {"deleted": [r["image_id"] for r in rows]}


@app.get("/api/stories")
def stories(limit: int = 20) -> Dict[str, Any]:
    try:
        df = hdfs.read_parquet_dataset("/photos/aggregates/final_stories", limit=limit)
    except Exception:
        df = pd.DataFrame()
    out = []
    for row in df.to_dict("records") if not df.empty else []:
        image_ids = _as_list(row.get("image_ids"))
        cover_id = image_ids[0] if image_ids else None
        out.append(
            {
                "story_id": row.get("story_id"),
                "title": row.get("title"),
                "summary": row.get("summary"),
                "image_ids": image_ids,
                "cover_image_url": f"/api/thumb/{cover_id}" if cover_id else None,
                "photo_count": int(row.get("photo_count", 0) or 0),
                "location": row.get("location"),
            }
        )
    return {"stories": out}
