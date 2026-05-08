from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    import ray
except Exception:  # pragma: no cover
    ray = None

from src.common import hdfs
from src.common.embeddings import embedding_to_list

ENRICHED_MIR_ROOT = os.getenv("ENRICHED_METADATA_ROOT", "/photos/metadata/enriched")
MODEL_HDFS_PATH = os.getenv("MODEL_HDFS_PATH", "/photos/models/image_classifier/model.pkl")
METRICS_HDFS_PATH = os.getenv("MODEL_METRICS_HDFS_PATH", "/photos/models/image_classifier/metrics.json")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))
RAY_ADDRESS = os.getenv("RAY_ADDRESS", "auto")


def row_to_feature(row: Dict[str, Any]) -> Dict[str, Any]:
    emb = embedding_to_list(row.get("embedding"), dim=EMBEDDING_DIM)
    width = float(row.get("width") or 0) / 5000.0
    height = float(row.get("height") or 0) / 5000.0
    size = float(row.get("file_size") or 0) / 5_000_000.0
    return {"x": emb + [width, height, size], "y": str(row.get("category") or "photo")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=ENRICHED_MIR_ROOT)
    parser.add_argument("--model-output", default=MODEL_HDFS_PATH)
    args = parser.parse_args()

    df = hdfs.read_parquet_dataset(args.input)
    if df.empty:
        raise SystemExit(f"No enriched metadata found at {args.input}. Run ray_jobs/vlm_enrich.py first.")
    records = df.to_dict("records")

    if ray is not None:
        try:
            ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
        except Exception:
            ray.init(ignore_reinit_error=True)

        @ray.remote
        def convert_remote(r):
            return row_to_feature(r)

        converted = ray.get([convert_remote.remote(r) for r in records])
    else:
        converted = [row_to_feature(r) for r in records]

    X = np.asarray([r["x"] for r in converted], dtype=np.float32)
    y = np.asarray([r["y"] for r in converted])
    if len(set(y.tolist())) < 2:
        raise SystemExit("Need at least two categories to train classifier")
    counts = pd.Series(y).value_counts()
    print("Class distribution:", counts.to_dict())
    stratify = y if min(pd.Series(y).value_counts()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(y)),
        "classes": sorted(set(y.tolist())),
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "feature_dim": int(X.shape[1]),
    }
    payload = {"model": model, "embedding_dim": EMBEDDING_DIM, "feature_dim": int(X.shape[1]), "classes": metrics["classes"]}
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pickle.dump(payload, tmp)
        local_model = tmp.name
    hdfs.upload_local_file(local_model, args.model_output, overwrite=True)
    hdfs.write_json(METRICS_HDFS_PATH, metrics, overwrite=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
