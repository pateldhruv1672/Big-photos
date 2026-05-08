"""
Build (or rebuild) the HNSW index from enriched Parquet metadata.

Run after vlm_enrich.py has produced embeddings:
  python scripts/build_hnsw_index.py

Reads enriched Parquet from outputs/metadata/enriched/**/*.parquet
(or HDFS fallback) and inserts every row that has a valid embedding
into the HNSW index saved at outputs/hnsw_index.bin + outputs/id_map.json.
"""

import sys
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/app")

from src.search.hnsw_index import HNSWIndex
from src.common.embeddings import EMBEDDING_DIM

ENRICHED_LOCAL = "/app/outputs/metadata/enriched"
INDEX_PATH     = "/app/outputs/hnsw_index.bin"
MAP_PATH       = "/app/outputs/id_map.json"


def load_enriched() -> pd.DataFrame:
    files = glob.glob(f"{ENRICHED_LOCAL}/**/*.parquet", recursive=True)
    if not files:
        try:
            from src.common.hdfs import HDFSClient
            import pyarrow as pa, pyarrow.parquet as pq
            client = HDFSClient()

            def _list_recursive(path):
                result = []
                for item in client.listdir(path):
                    full = f"{path}/{item['pathSuffix']}"
                    if item["type"] == "DIRECTORY":
                        result.extend(_list_recursive(full))
                    elif item["pathSuffix"].endswith(".parquet"):
                        result.append(full)
                return result

            hdfs_files = _list_recursive("/photos/metadata/enriched")
            dfs = []
            for p in hdfs_files:
                data = client.get(p)
                buf = pa.BufferReader(pa.py_buffer(data))
                dfs.append(pq.read_table(buf).to_pandas())
            if dfs:
                return pd.concat(dfs, ignore_index=True)
        except Exception as e:
            print(f"HDFS fallback failed: {e}")
        raise FileNotFoundError("No enriched metadata found. Run `make part2` first.")

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    print("=== Building HNSW Index ===")
    df = load_enriched()
    print(f"Loaded {len(df)} enriched rows.")

    if df.empty:
        print("No data to index.")
        return

    # Determine embedding dimension from data
    sample_emb = df.iloc[0].get("embedding")
    if sample_emb is None:
        print("No embeddings found in metadata. Run vlm_enrich.py first.")
        return

    if isinstance(sample_emb, (list, np.ndarray)):
        dim = len(sample_emb)
    else:
        print(f"Unexpected embedding type: {type(sample_emb)}")
        return

    print(f"Embedding dimension: {dim}")
    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Start fresh
    if Path(INDEX_PATH).exists():
        Path(INDEX_PATH).unlink()
    if Path(MAP_PATH).exists():
        Path(MAP_PATH).unlink()

    index = HNSWIndex(dim=dim, path=INDEX_PATH, map_path=MAP_PATH)
    added = 0

    for _, row in df.iterrows():
        image_id = row.get("image_id")
        emb = row.get("embedding")
        if not image_id or emb is None:
            continue
        if isinstance(emb, list):
            emb_arr = np.array(emb, dtype="float32")
        elif isinstance(emb, np.ndarray):
            emb_arr = emb.astype("float32")
        else:
            continue

        if len(emb_arr) != dim:
            continue

        index.add(image_id, emb_arr.tolist(), auto_save=False)
        added += 1

    index.save()
    print(f"Indexed {added} vectors → {INDEX_PATH}")
    print(f"id_map saved → {MAP_PATH}")

    # Upload index to HDFS
    try:
        from src.common.hdfs import HDFSClient
        client = HDFSClient()
        client.mkdirs("/photos/vector_index")
        client.put("/photos/vector_index/hnsw_index.bin", Path(INDEX_PATH))
        client.put("/photos/vector_index/id_map.json", Path(MAP_PATH))
        print("HNSW index uploaded to HDFS /photos/vector_index/")
    except Exception as e:
        print(f"Warning: HDFS upload failed ({e}). Local index is still available.")


if __name__ == "__main__":
    main()
