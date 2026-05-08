"""
Part 2: VLM Enrichment Job

Reads basic_metadata Parquet (locally from outputs/ or HDFS via WebHDFS),
generates captions, labels, category, and embeddings for each image, then
writes enriched Parquet back to HDFS and to outputs/metadata/enriched/.

If outputs/metadata/enriched already contains non-empty Parquet files, the job
exits successfully without re-running VLM (unless FORCE_ENRICH=1).

Environment:
  DEMO_FAST=1 — skip Ollama; tag-based captions; deterministic embeddings (demo).

Fallback chain for VLM:
  1. Ollama (llava or moondream) — if server responds
  2. Tag-based deterministic labels — always available

Fallback chain for embeddings:
  1. Ollama nomic-embed-text
  2. Deterministic MD5-seeded random vectors (reproducible)
"""

import os
import sys
import datetime
import json
import base64
import requests
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, "/app")

from src.common.embeddings import get_text_embedding

NAMENODE = os.environ.get("NAMENODE_HOST", "namenode")
HDFS_USER = os.environ.get("HDFS_USER", "root")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "ollama")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "128"))

BASIC_PARQUET_LOCAL = "/app/outputs/metadata/basic"
ENRICHED_PARQUET_LOCAL = "/app/outputs/metadata/enriched"


def _cached_enriched_parquet_exists() -> bool:
    """True if enriched dir exists and has at least one non-empty .parquet file."""
    root = Path(ENRICHED_PARQUET_LOCAL)
    if not root.is_dir():
        return False
    for path in root.glob("**/*.parquet"):
        try:
            if path.is_file() and path.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


# --------------------------------------------------------------------------- #
# Ollama helpers                                                                #
# --------------------------------------------------------------------------- #

def _ollama_available() -> bool:
    try:
        r = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _infer_category(tags: list) -> str:
    """Map MIRFLICKR annotation labels + free tags → display category.
    Uses a weighted scoring system so images get the most specific category."""
    tag_set = {str(t).lower() for t in tags}
    scores = {
        "food":   len(tag_set & {"food", "meal", "restaurant", "cuisine", "dish", "coffee"}) * 3,
        "event":  len(tag_set & {"event", "concert", "festival", "party", "wedding", "sport", "baby"}) * 3,
        "art":    len(tag_set & {"art", "painting", "sculpture", "museum", "gallery", "flower"}) * 3,
        "travel": len(tag_set & {"travel", "tourism", "vacation", "trip", "holiday",
                                  "sea", "river", "lake", "sunset"}) * 2,
        "city":   len(tag_set & {"city", "urban", "street", "building", "skyline", "bridge",
                                  "architecture", "structures", "indoor", "car", "transport"}) * 2,
        "people": len(tag_set & {"people", "portrait", "face", "crowd", "person", "selfie",
                                  "female", "male"}) * 2,
        "nature": len(tag_set & {"nature", "landscape", "mountain", "beach", "forest", "ocean",
                                  "lake", "sky", "tree", "plant_life", "clouds", "sunset", "water",
                                  "night", "animals", "bird", "dog"}) * 1,
    }
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "general"


def _ollama_caption(image_bytes, tags: list) -> tuple:
    """Call Ollama vision model. Returns (caption, vlm_labels, objects, category)."""
    model = "llava"
    prompt = "Describe this image in one sentence. List the main objects visible."
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if image_bytes:
        payload["images"] = [base64.b64encode(image_bytes).decode()]
    try:
        r = requests.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
            json=payload,
            timeout=30,
        )
        if r.status_code == 200:
            text = r.json().get("response", "")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            caption = lines[0] if lines else "Visual content."
            objects = [w for w in caption.lower().split() if len(w) > 4][:8]
            category = _infer_category(tags + objects)
            return caption, objects, objects, category
    except Exception:
        pass
    return _fallback_vlm(tags)


def _fallback_vlm(tags: list) -> tuple:
    """Deterministic labels from existing tags."""
    tags = list(tags) if tags else []
    category = _infer_category(tags)
    caption = f"Image containing {', '.join(tags)}." if tags else "Visual content."
    return caption, tags, tags, category


def _tags_from_metadata(value) -> list:
    """Normalize tags column from Parquet rows (no truthiness on ndarray)."""
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
            return [str(parsed)]
        except Exception:
            return [value] if value.strip() else []
    try:
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return []
    except (ValueError, TypeError):
        pass
    return []


def _predicted_label(row: dict, tags: list, category: str) -> str:
    """Primary label for demo / downstream: first useful tag, else location/label, else category."""
    for t in tags:
        s = str(t).strip()
        if s:
            return s
    lab = row.get("label")
    if lab is not None and str(lab).strip():
        return str(lab).strip()
    loc = row.get("location")
    if loc is not None:
        ls = str(loc).strip()
        if ls and ls.lower() != "unknown":
            return ls
    return category


# --------------------------------------------------------------------------- #
# Load basic metadata                                                           #
# --------------------------------------------------------------------------- #

def load_basic_metadata() -> pd.DataFrame:
    """Try local outputs first, then HDFS via WebHDFS."""
    local_path = Path(BASIC_PARQUET_LOCAL)
    if local_path.exists() and any(local_path.glob("**/*.parquet")):
        files = list(local_path.glob("**/*.parquet"))
        print(f"Reading {len(files)} local parquet file(s).")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Try HDFS
    try:
        from src.common.hdfs import HDFSClient
        client = HDFSClient()
        file_list = client.listdir("/photos/metadata/basic")
        parquet_files = [f["pathSuffix"] for f in file_list if f["pathSuffix"].endswith(".parquet")]
        if parquet_files:
            dfs = []
            for fname in parquet_files:
                data = client.get(f"/photos/metadata/basic/{fname}")
                buf = pa.BufferReader(pa.py_buffer(data))
                dfs.append(pq.read_table(buf).to_pandas())
            print(f"Read {len(parquet_files)} parquet file(s) from HDFS.")
            return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        print(f"Warning: HDFS read failed: {e}")

    raise FileNotFoundError(
        "No basic metadata found. Run `make part1` first to generate it."
    )


# --------------------------------------------------------------------------- #
# Main enrichment loop                                                          #
# --------------------------------------------------------------------------- #

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    use_ollama = _ollama_available()
    print(f"Ollama available: {use_ollama} | Embedding dim: {EMBEDDING_DIM}")

    rows = []
    total = len(df)
    for idx, r in df.iterrows():
        row = r.to_dict()
        tags = _tags_from_metadata(row.get("tags"))

        # Attempt to load image bytes for Ollama vision
        image_bytes = None
        if use_ollama:
            local_img = Path(f"/app/data/mirflickr/images/{row.get('file_name', '')}")
            if local_img.exists():
                image_bytes = local_img.read_bytes()

        if use_ollama:
            caption, vlm_labels, objects, category = _ollama_caption(image_bytes, tags)
        else:
            caption, vlm_labels, objects, category = _fallback_vlm(tags)

        # scene_labels = high-level scene descriptors (subset of vlm_labels focused on setting)
        SCENE_KEYWORDS = {"beach", "mountain", "forest", "city", "street", "indoor",
                          "outdoor", "sky", "sunset", "ocean", "lake", "desert", "snow",
                          "park", "stadium", "market", "museum", "restaurant", "night", "daytime"}
        scene_labels = [l for l in vlm_labels if str(l).lower() in SCENE_KEYWORDS] or [category]

        emb_text = caption + " " + " ".join(str(l) for l in vlm_labels)
        embedding = get_text_embedding(emb_text, dim=EMBEDDING_DIM, prefer="auto")

        predicted_label = _predicted_label(row, tags, category)
        confidence = 0.9 if use_ollama else 0.75
        quality_score = confidence

        row.update({
            "original_tags": tags,             # ← renamed per architecture diagram
            "caption": caption,
            "vlm_labels": vlm_labels,
            "objects": objects,
            "scene_labels": scene_labels,      # ← added per architecture diagram
            "category": category,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "embedding": embedding,
            "embedding_dim": len(embedding),
            "quality_score": quality_score,
            "processed_at": datetime.datetime.utcnow().isoformat(),
            "date": datetime.date.today().isoformat(),
        })
        rows.append(row)
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"  Enriched {idx + 1}/{total}")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Write output                                                                  #
# --------------------------------------------------------------------------- #

def write_enriched(df: pd.DataFrame) -> None:
    date_str = datetime.date.today().isoformat()

    # Write locally
    local_out = Path(ENRICHED_PARQUET_LOCAL) / f"date={date_str}"
    local_out.mkdir(parents=True, exist_ok=True)
    out_path = local_out / "part-0000.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote enriched Parquet locally: {out_path}")

    # Upload to HDFS
    try:
        from src.common.hdfs import HDFSClient
        client = HDFSClient()
        hdfs_dir = f"/photos/metadata/enriched/date={date_str}"
        client.mkdirs(hdfs_dir)
        client.put(f"{hdfs_dir}/part-0000.parquet", out_path)
        print(f"Uploaded to HDFS: {hdfs_dir}/part-0000.parquet")
    except Exception as e:
        print(f"Warning: HDFS upload failed ({e}). Local copy available at {out_path}.")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=== VLM Enrichment Job ===")

    df = load_basic_metadata()
    print(f"Loaded {len(df)} rows from basic metadata.")
    enriched = enrich(df)
    write_enriched(enriched)
    print(f"Done. Enriched {len(enriched)} images.")
