"""Normalize user-provided metadata into the single UI/search schema.

The project may receive metadata from CSV, JSONL, or Parquet files with different
column names.  This module maps them into a consistent Parquet schema stored in
HDFS under /photos/metadata/ui_active.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.common import hdfs
from src.common.embeddings import DEFAULT_DIM, embedding_to_list, record_embedding
from src.common.image_utils import infer_fallback_category, infer_fallback_labels, make_caption

DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "team_gallery")
RAW_UI_IMAGE_ROOT = os.getenv("RAW_UI_IMAGE_ROOT", "/photos/raw/team_gallery/images")
THUMBNAIL_UI_ROOT = os.getenv("THUMBNAIL_UI_ROOT", "/photos/thumbnails/team_gallery")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", str(DEFAULT_DIM)))

COLUMN_ALIASES = {
    "image_id": ["image_id", "id", "photo_id", "imageId", "filename_no_ext"],
    "user_id": ["user_id", "user", "owner", "account_id"],
    "image_uri": ["image_uri", "hdfs_path", "path", "image_path", "uri", "file_path"],
    "thumbnail_uri": ["thumbnail_uri", "thumb_uri", "thumbnail_path", "thumb_path"],
    "file_name": ["file_name", "filename", "name", "image_name"],
    "caption": ["caption", "description", "title"],
    "labels": ["labels", "vlm_labels", "tags", "tag", "objects"],
    "objects": ["objects", "detected_objects"],
    "category": ["category", "class", "label", "predicted_category"],
    "embedding": ["embedding", "vector", "embeddings", "features"],
    "width": ["width", "image_width"],
    "height": ["height", "image_height"],
    "file_size": ["file_size", "size", "file_size_bytes"],
    "taken_at": ["taken_at", "datetime", "timestamp", "created_time"],
    "location": ["location", "place", "city"],
    "deleted": ["deleted", "is_deleted"],
}

UI_COLUMNS = [
    "image_id",
    "user_id",
    "image_uri",
    "thumbnail_uri",
    "file_name",
    "dataset",
    "caption",
    "labels",
    "vlm_labels",
    "objects",
    "category",
    "embedding",
    "embedding_dim",
    "width",
    "height",
    "file_size",
    "taken_at",
    "location",
    "deleted",
    "created_at",
    "updated_at",
]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(value: str) -> str:
    base = value or "photo"
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", base.split("/")[-1])
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_")
    if stem:
        return stem
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def _first(row: Dict[str, Any], logical: str) -> Any:
    for col in COLUMN_ALIASES.get(logical, [logical]):
        if col in row and not _is_null(row[col]):
            return row[col]
    return None


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value)) and not isinstance(value, (list, tuple, dict, np.ndarray))
    except Exception:
        return False


def parse_list(value: Any) -> List[str]:
    if _is_null(value):
        return []
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if not _is_null(x)]
    if isinstance(value, dict):
        return [str(k) for k, v in value.items() if v]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            return parse_list(parsed)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
                return parse_list(parsed)
            except Exception:
                pass
    parts = re.split(r"[,;|\s]+", text)
    return [p.strip() for p in parts if p.strip()]


def parse_bool(value: Any) -> bool:
    if _is_null(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "deleted"}


def parse_timestamp(value: Any, fallback: Optional[datetime] = None) -> datetime:
    if _is_null(value):
        return fallback or now_utc()
    try:
        ts = pd.to_datetime(value, utc=True)
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
    except Exception:
        pass
    return fallback or now_utc()


def derive_image_uri(image_id: str, file_name: str, row_uri: Optional[str], raw_root: str = RAW_UI_IMAGE_ROOT) -> str:
    if row_uri:
        return hdfs.to_hdfs_uri(str(row_uri)) if not str(row_uri).startswith("hdfs://") else str(row_uri)
    fname = file_name or f"{image_id}.jpg"
    # If MIRFLICKR is sharded by first digit/last known folder, local metadata can
    # provide shard. Otherwise place directly under raw_root.
    shard = ""
    m = re.search(r"(\d+)", fname)
    if m:
        shard = str(int(m.group(1)) % 10)
    candidate = f"{raw_root.rstrip('/')}/{shard}/{fname}" if shard else f"{raw_root.rstrip('/')}/{fname}"
    return hdfs.to_hdfs_uri(candidate)


def derive_thumb_uri(image_id: str, file_name: str, row_uri: Optional[str], thumb_root: str = THUMBNAIL_UI_ROOT) -> str:
    if row_uri:
        return hdfs.to_hdfs_uri(str(row_uri)) if not str(row_uri).startswith("hdfs://") else str(row_uri)
    fname = f"{image_id}.jpg"
    shard = str(abs(hash(image_id)) % 10)
    if file_name:
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", file_name.split("/")[-1])
        fname = f"{stable_id(stem)}.jpg"
    return hdfs.to_hdfs_uri(f"{thumb_root.rstrip('/')}/{shard}/{fname}")


def normalize_record(row: Dict[str, Any], dataset: str = "team_gallery", raw_root: str = RAW_UI_IMAGE_ROOT) -> Dict[str, Any]:
    file_name = str(_first(row, "file_name") or "").strip()
    row_uri = _first(row, "image_uri")
    image_id = str(_first(row, "image_id") or stable_id(file_name or str(row_uri or "photo"))).strip()
    if not file_name:
        if row_uri:
            file_name = str(row_uri).rstrip("/").split("/")[-1]
        else:
            file_name = f"{image_id}.jpg"

    labels = parse_list(_first(row, "labels"))
    objects = parse_list(_first(row, "objects")) or labels[:3]
    category = str(_first(row, "category") or "").strip()
    if not category:
        category = infer_fallback_category(" ".join(labels) or file_name or image_id)
    if not labels:
        labels = infer_fallback_labels(" ".join([file_name, category, image_id]))
    caption = str(_first(row, "caption") or "").strip()
    if not caption:
        caption = make_caption(category, labels, file_name or image_id)

    emb_value = _first(row, "embedding")
    embedding = embedding_to_list(emb_value, dim=EMBEDDING_DIM) if emb_value is not None else []
    if len(embedding) != EMBEDDING_DIM or not any(abs(float(x)) > 1e-12 for x in embedding):
        embedding = record_embedding(caption=caption, labels=labels, category=category, tags=labels, dim=EMBEDDING_DIM)

    image_uri = derive_image_uri(image_id, file_name, str(row_uri) if row_uri else None, raw_root=raw_root)
    thumbnail_uri = derive_thumb_uri(image_id, file_name, str(_first(row, "thumbnail_uri")) if _first(row, "thumbnail_uri") else None)
    created = parse_timestamp(row.get("created_at") or _first(row, "taken_at"), fallback=now_utc())
    updated = parse_timestamp(row.get("updated_at"), fallback=now_utc())

    return {
        "image_id": image_id,
        "user_id": str(_first(row, "user_id") or DEFAULT_USER_ID),
        "image_uri": image_uri,
        "thumbnail_uri": thumbnail_uri,
        "file_name": file_name,
        "dataset": str(row.get("dataset") or dataset),
        "caption": caption,
        "labels": labels,
        "vlm_labels": labels,
        "objects": objects,
        "category": category,
        "embedding": embedding,
        "embedding_dim": len(embedding),
        "width": int(_first(row, "width") or 0),
        "height": int(_first(row, "height") or 0),
        "file_size": int(_first(row, "file_size") or 0),
        "taken_at": parse_timestamp(_first(row, "taken_at"), fallback=created),
        "location": str(_first(row, "location") or "Unknown"),
        "deleted": parse_bool(_first(row, "deleted")),
        "created_at": created,
        "updated_at": updated,
    }


def normalize_dataframe(df: pd.DataFrame, dataset: str = "team_gallery", raw_root: str = RAW_UI_IMAGE_ROOT) -> pd.DataFrame:
    records = [normalize_record(row, dataset=dataset, raw_root=raw_root) for row in df.to_dict("records")]
    out = pd.DataFrame(records)
    if out.empty:
        return pd.DataFrame(columns=UI_COLUMNS)
    out = out.drop_duplicates(subset=["image_id"], keep="last")
    for col in UI_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[UI_COLUMNS]
