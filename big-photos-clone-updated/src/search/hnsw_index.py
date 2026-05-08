"""HNSW vector index manager backed by HDFS artifacts.

No Postgres/Qdrant is used.  The hnswlib index and id map are persisted in HDFS:
  /photos/vector_index/current/hnsw_index.bin
  /photos/vector_index/current/id_map.json       # integer HNSW label -> image_id
  /photos/vector_index/current/manifest.json
  /photos/vector_index/current/item_meta.json    # convenience metadata cache
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import hnswlib
except Exception:  # pragma: no cover
    hnswlib = None

from src.common import hdfs
from src.common.embeddings import DEFAULT_DIM, cosine_similarity, embedding_to_list, text_embedding

UI_ACTIVE_METADATA_ROOT = os.getenv("UI_ACTIVE_METADATA_ROOT", "/photos/metadata/ui_active")
ENRICHED_METADATA_ROOT = os.getenv("ENRICHED_METADATA_ROOT", "/photos/metadata/enriched")
UPLOAD_METADATA_ROOT = os.getenv("UPLOAD_METADATA_ROOT", "/photos/metadata/uploads")
DELETE_METADATA_ROOT = os.getenv("DELETE_METADATA_ROOT", "/photos/metadata/deletes")
VECTOR_INDEX_ROOT = os.getenv("VECTOR_INDEX_ROOT", "/photos/vector_index/current")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", str(DEFAULT_DIM)))
LOCAL_STATE_DIR = os.getenv("LOCAL_STATE_DIR", "/app/state")


def _list_value(value: Any) -> List[str]:
    if value is None:
        return []
    try:
        if pd.isna(value) and not isinstance(value, (list, tuple, dict, np.ndarray)):
            return []
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return [str(value)]


def load_active_metadata() -> pd.DataFrame:
    """Load UI metadata + uploads and apply delete tombstones."""
    frames = []
    # UI/search should be built from gallery + uploads only.
    # Enriched training datasets can contain overlapping ids/captions that make
    # gallery cards point to unexpected photos.
    for root in [UI_ACTIVE_METADATA_ROOT, UPLOAD_METADATA_ROOT]:
        try:
            df = hdfs.read_parquet_dataset(root)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"Skipping metadata root {root}: {exc}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "image_id" not in df.columns:
        return pd.DataFrame()
    df["image_id"] = df["image_id"].astype(str)
    if "dataset" in df.columns:
        df = df[df["dataset"].fillna("").isin(["team_gallery", "uploads"])]
    df = df.drop_duplicates(subset=["image_id"], keep="last")
    try:
        deletes = hdfs.read_parquet_dataset(DELETE_METADATA_ROOT)
        if not deletes.empty and "image_id" in deletes.columns:
            deleted_ids = set(deletes["image_id"].astype(str).tolist())
            df = df[~df["image_id"].astype(str).isin(deleted_ids)]
    except Exception as exc:
        print(f"Could not apply deletes: {exc}")
    if "deleted" in df.columns:
        df = df[df["deleted"].fillna(False) != True]
    return df.reset_index(drop=True)


class HNSWManager:
    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        local_dir: str = LOCAL_STATE_DIR,
        hdfs_dir: str = VECTOR_INDEX_ROOT,
        space: str = "cosine",
    ):
        self.dim = int(dim)
        self.space = space
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.hdfs_dir = hdfs_dir.rstrip("/")
        self.index_path = self.local_dir / "hnsw_index.bin"
        self.map_path = self.local_dir / "id_map.json"
        self.manifest_path = self.local_dir / "manifest.json"
        self.meta_path = self.local_dir / "item_meta.json"
        self.label_to_image: Dict[int, str] = {}
        self.image_to_label: Dict[str, int] = {}
        self.item_meta: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.index = None

    @property
    def items(self) -> Dict[str, Dict[str, Any]]:
        """Compatibility view used by older backend code."""
        return self.item_meta

    def _new_index(self, max_elements: int) -> None:
        if hnswlib is None:
            self.index = None
            return
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=max(max_elements, 16), ef_construction=200, M=16)
        self.index.set_ef(64)

    def _download_artifacts(self) -> None:
        for name, local in [
            ("id_map.json", self.map_path),
            ("manifest.json", self.manifest_path),
            ("item_meta.json", self.meta_path),
            ("hnsw_index.bin", self.index_path),
        ]:
            try:
                hdfs.download_to_local(f"{self.hdfs_dir}/{name}", str(local))
            except Exception:
                pass

    def load(self) -> None:
        self._download_artifacts()
        self.label_to_image = {}
        self.image_to_label = {}
        self.item_meta = {}
        self.embeddings = {}

        if self.map_path.exists():
            data = json.loads(self.map_path.read_text())
            # New format: {"0": "image_id"}; legacy format accepted too.
            if "items" in data:
                for image_id, item in data.get("items", {}).items():
                    label = int(item.get("label"))
                    self.label_to_image[label] = image_id
                    self.image_to_label[image_id] = label
                    self.item_meta[image_id] = item
                    if "embedding" in item:
                        self.embeddings[image_id] = embedding_to_list(item["embedding"], dim=self.dim)
                self.dim = int(data.get("dim", self.dim))
            else:
                for label, image_id in data.items():
                    self.label_to_image[int(label)] = str(image_id)
                    self.image_to_label[str(image_id)] = int(label)
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            self.dim = int(manifest.get("embedding_dim", self.dim))
            self.space = manifest.get("space", self.space)
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            self.item_meta = {str(k): v for k, v in meta.items()}
            for image_id, item in self.item_meta.items():
                if "embedding" in item:
                    self.embeddings[image_id] = embedding_to_list(item["embedding"], dim=self.dim)

        if hnswlib is not None and self.index_path.exists() and self.label_to_image:
            self.index = hnswlib.Index(space=self.space, dim=self.dim)
            self.index.load_index(str(self.index_path), max_elements=max(len(self.label_to_image) + 1000, 1000))
            self.index.set_ef(64)
        elif self.embeddings:
            self.rebuild_from_items(save=False)

    def save(self, metadata_path: str = UI_ACTIVE_METADATA_ROOT) -> None:
        hdfs.mkdirs(self.hdfs_dir)
        id_map = {str(label): image_id for label, image_id in sorted(self.label_to_image.items())}
        manifest = {
            "embedding_dim": self.dim,
            "space": self.space,
            "total_vectors": len(self.label_to_image),
            "metadata_path": metadata_path,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "format": "hnswlib+label_to_image_map_v1",
        }
        self.map_path.write_text(json.dumps(id_map, indent=2, sort_keys=True))
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        # item_meta includes embeddings for brute-force fallback and demo inspection.
        self.meta_path.write_text(json.dumps(self.item_meta, default=str))
        hdfs.upload_local_file(str(self.map_path), f"{self.hdfs_dir}/id_map.json", overwrite=True)
        hdfs.upload_local_file(str(self.manifest_path), f"{self.hdfs_dir}/manifest.json", overwrite=True)
        hdfs.upload_local_file(str(self.meta_path), f"{self.hdfs_dir}/item_meta.json", overwrite=True)
        if self.index is not None:
            self.index.save_index(str(self.index_path))
            hdfs.upload_local_file(str(self.index_path), f"{self.hdfs_dir}/hnsw_index.bin", overwrite=True)

    def rebuild_from_records(self, records: Iterable[Dict[str, Any]], save: bool = True) -> None:
        self.label_to_image = {}
        self.image_to_label = {}
        self.item_meta = {}
        self.embeddings = {}
        label = 0
        seen: set[str] = set()
        for row in records:
            image_id = str(row.get("image_id") or "").strip()
            if not image_id or image_id in seen or image_id == "None":
                continue
            if bool(row.get("deleted", False)):
                continue
            emb = embedding_to_list(row.get("embedding"), dim=self.dim)
            if len(emb) != self.dim or not any(abs(float(x)) > 1e-12 for x in emb):
                continue
            seen.add(image_id)
            self.label_to_image[label] = image_id
            self.image_to_label[image_id] = label
            meta = {
                "label": label,
                "caption": row.get("caption") or row.get("file_name") or image_id,
                "category": row.get("category") or "photo",
                "labels": _list_value(row.get("vlm_labels")) or _list_value(row.get("labels")) or _list_value(row.get("tags")),
                "thumbnail_uri": row.get("thumbnail_uri"),
                "image_uri": row.get("image_uri"),
                "user_id": row.get("user_id", "team_gallery"),
                "embedding": emb,
            }
            self.item_meta[image_id] = meta
            self.embeddings[image_id] = emb
            label += 1
        self.rebuild_from_items(save=save)

    def rebuild_from_items(self, save: bool = True) -> None:
        if not self.embeddings:
            self.index = None
            if save:
                self.save()
            return
        if hnswlib is None:
            self.index = None
            if save:
                self.save()
            return
        self._new_index(max_elements=max(len(self.embeddings) + 1000, 1000))
        labels = []
        vectors = []
        for image_id, emb in self.embeddings.items():
            labels.append(self.image_to_label[image_id])
            vectors.append(emb)
        self.index.add_items(np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=np.int64))
        if save:
            self.save()

    def add_or_update(self, image_id: str, embedding: Sequence[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        metadata = dict(metadata or {})
        image_id = str(image_id)
        emb = embedding_to_list(embedding, dim=self.dim)
        if image_id in self.image_to_label:
            label = self.image_to_label[image_id]
        else:
            label = max(self.label_to_image.keys(), default=-1) + 1
            self.label_to_image[label] = image_id
            self.image_to_label[image_id] = label
        metadata.update({"label": label, "embedding": emb})
        self.item_meta[image_id] = metadata
        self.embeddings[image_id] = emb
        # hnswlib cannot truly replace an item without delete/re-add complexity; rebuild
        # for correctness in the demo.
        self.rebuild_from_items(save=False)

    def remove(self, image_ids: Iterable[str]) -> None:
        for image_id in [str(x) for x in image_ids]:
            label = self.image_to_label.pop(image_id, None)
            if label is not None:
                self.label_to_image.pop(label, None)
            self.item_meta.pop(image_id, None)
            self.embeddings.pop(image_id, None)
        # Compact labels by rebuilding from remaining metadata.
        records = []
        for img_id, meta in list(self.item_meta.items()):
            row = dict(meta)
            row["image_id"] = img_id
            records.append(row)
        self.rebuild_from_records(records, save=False)

    def search(self, query_embedding: Sequence[float], top_k: int = 20) -> List[Tuple[str, float]]:
        q = embedding_to_list(query_embedding, dim=self.dim)
        if self.index is not None and self.label_to_image:
            k = min(max(int(top_k), 1), len(self.label_to_image))
            labels, distances = self.index.knn_query(np.asarray([q], dtype=np.float32), k=k)
            out: List[Tuple[str, float]] = []
            for label, dist in zip(labels[0], distances[0]):
                image_id = self.label_to_image.get(int(label))
                if image_id:
                    out.append((image_id, float(1.0 - dist)))
            return out
        scored = []
        for image_id, emb in self.embeddings.items():
            scored.append((image_id, cosine_similarity(q, emb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_text(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        return self.search(text_embedding(query, dim=self.dim), top_k=top_k)


def build_from_hdfs(metadata_root: str = UI_ACTIVE_METADATA_ROOT) -> HNSWManager:
    df = load_active_metadata()
    if df.empty:
        raise RuntimeError("No active metadata found for HNSW build")
    manager = HNSWManager()
    manager.rebuild_from_records(df.to_dict("records"), save=True)
    print(f"Built HNSW index with {len(manager.label_to_image)} vectors at {manager.hdfs_dir}")
    return manager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build from HDFS active metadata")
    parser.add_argument("--query", default=None, help="Search text query")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.build:
        build_from_hdfs()
    if args.query:
        mgr = HNSWManager()
        mgr.load()
        print(mgr.search_text(args.query, args.top_k))


if __name__ == "__main__":
    main()
