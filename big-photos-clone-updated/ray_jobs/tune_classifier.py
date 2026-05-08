from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import ray
from ray import tune
from ray.air import session

from src.common import hdfs
from src.common.embeddings import embedding_to_list

ENRICHED_MIR_ROOT = os.getenv("ENRICHED_METADATA_ROOT", "/photos/metadata/enriched")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))
RAY_ADDRESS = os.getenv("RAY_ADDRESS", "local")
MODEL_OUTPUT = os.getenv("MODEL_HDFS_PATH", "/photos/models/image_classifier/model.pkl")
METRICS_OUTPUT = os.getenv("MODEL_METRICS_HDFS_PATH", "/photos/models/image_classifier/metrics.json")
TUNE_RESULTS_HDFS_DIR = os.getenv("TUNE_RESULTS_HDFS_DIR", "/photos/models/image_classifier/tuning")
TUNE_RESULTS_LOCAL_DIR = os.getenv("TUNE_RESULTS_LOCAL_DIR", "/app/outputs/training")


def row_to_feature(row: Dict[str, Any]) -> Dict[str, Any]:
    emb = embedding_to_list(row.get("embedding"), dim=EMBEDDING_DIM)
    width = float(row.get("width") or 0) / 5000.0
    height = float(row.get("height") or 0) / 5000.0
    size = float(row.get("file_size") or 0) / 5_000_000.0
    return {"x": emb + [width, height, size], "y": str(row.get("category") or "photo")}


def trainable(config: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    model = RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=None if int(config["max_depth"]) <= 0 else int(config["max_depth"]),
        min_samples_split=int(config["min_samples_split"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    acc = float(accuracy_score(y_val, pred))
    f1 = float(f1_score(y_val, pred, average="weighted", zero_division=0))
    session.report({"accuracy": acc, "f1_weighted": f1})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=ENRICHED_MIR_ROOT)
    parser.add_argument("--model-output", default=MODEL_OUTPUT)
    parser.add_argument("--metrics-output", default=METRICS_OUTPUT)
    parser.add_argument("--trials", type=int, default=18)
    args = parser.parse_args()

    if RAY_ADDRESS == "local":
        ray.init(ignore_reinit_error=True)
    else:
        try:
            ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
        except Exception:
            ray.init(ignore_reinit_error=True)

    df = hdfs.read_parquet_dataset(args.input)
    if df.empty:
        raise SystemExit(f"No enriched metadata found at {args.input}")
    converted = [row_to_feature(r) for r in df.to_dict("records")]
    X = np.asarray([r["x"] for r in converted], dtype=np.float32)
    y = np.asarray([r["y"] for r in converted])
    if len(set(y.tolist())) < 2:
        raise SystemExit("Need at least two classes to tune/train classifier")

    stratify = y if min(pd.Series(y).value_counts()) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    search_space = {
        "n_estimators": tune.choice([80, 120, 160, 220, 300]),
        "max_depth": tune.choice([0, 8, 12, 20]),
        "min_samples_split": tune.choice([2, 4, 8]),
        "min_samples_leaf": tune.choice([1, 2, 4]),
    }
    tuner = tune.Tuner(
        tune.with_parameters(trainable, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
        tune_config=tune.TuneConfig(metric="f1_weighted", mode="max", num_samples=int(args.trials)),
        param_space=search_space,
    )
    result_grid = tuner.fit()
    best = result_grid.get_best_result(metric="f1_weighted", mode="max")
    best_cfg = best.config

    final_model = RandomForestClassifier(
        n_estimators=int(best_cfg["n_estimators"]),
        max_depth=None if int(best_cfg["max_depth"]) <= 0 else int(best_cfg["max_depth"]),
        min_samples_split=int(best_cfg["min_samples_split"]),
        min_samples_leaf=int(best_cfg["min_samples_leaf"]),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)
    pred = final_model.predict(X_val)
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(y)),
        "classes": sorted(set(y.tolist())),
        "accuracy": float(accuracy_score(y_val, pred)),
        "f1_weighted": float(f1_score(y_val, pred, average="weighted", zero_division=0)),
        "feature_dim": int(X.shape[1]),
        "best_config": best_cfg,
        "num_trials": int(args.trials),
    }

    payload = {"model": final_model, "embedding_dim": EMBEDDING_DIM, "feature_dim": int(X.shape[1]), "classes": metrics["classes"]}
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pickle.dump(payload, tmp)
        local_model = tmp.name
    hdfs.upload_local_file(local_model, args.model_output, overwrite=True)
    hdfs.write_json(args.metrics_output, metrics, overwrite=True)

    trial_rows: List[Dict[str, Any]] = []
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for idx, r in enumerate(result_grid, start=1):
        trial_rows.append(
            {
                "run_tag": run_tag,
                "trial_id": idx,
                "accuracy": float(r.metrics.get("accuracy", 0.0)),
                "f1_weighted": float(r.metrics.get("f1_weighted", 0.0)),
                **r.config,
            }
        )
    trials_df = pd.DataFrame(trial_rows).sort_values("f1_weighted", ascending=False).reset_index(drop=True)

    local_dir = Path(TUNE_RESULTS_LOCAL_DIR) / run_tag
    local_dir.mkdir(parents=True, exist_ok=True)
    csv_path = local_dir / "trial_results.csv"
    json_path = local_dir / "summary.json"
    trials_df.to_csv(csv_path, index=False)
    summary = {"metrics": metrics, "top_trials": trials_df.head(5).to_dict("records"), "run_tag": run_tag}
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Append cumulative local histories across runs.
    all_trials_path = Path(TUNE_RESULTS_LOCAL_DIR) / "all_trial_results.csv"
    if all_trials_path.exists():
        old_trials = pd.read_csv(all_trials_path)
        all_trials = pd.concat([old_trials, trials_df], ignore_index=True, sort=False)
    else:
        all_trials = trials_df.copy()
    all_trials.to_csv(all_trials_path, index=False)

    run_summary_row = {
        "run_tag": run_tag,
        "trained_at": metrics["trained_at"],
        "rows": metrics["rows"],
        "accuracy": metrics["accuracy"],
        "f1_weighted": metrics["f1_weighted"],
        "num_trials": metrics["num_trials"],
        **{f"best_{k}": v for k, v in metrics["best_config"].items()},
    }
    run_history_path = Path(TUNE_RESULTS_LOCAL_DIR) / "run_history.csv"
    if run_history_path.exists():
        old_hist = pd.read_csv(run_history_path)
        run_hist = pd.concat([old_hist, pd.DataFrame([run_summary_row])], ignore_index=True, sort=False)
    else:
        run_hist = pd.DataFrame([run_summary_row])
    run_hist.to_csv(run_history_path, index=False)

    hdfs.write_dataframe_parquet(trials_df, f"{TUNE_RESULTS_HDFS_DIR.rstrip('/')}/{run_tag}", filename="trial_results.parquet")
    hdfs.write_json(f"{TUNE_RESULTS_HDFS_DIR.rstrip('/')}/{run_tag}/summary.json", summary, overwrite=True)
    hdfs.write_dataframe_parquet(all_trials, TUNE_RESULTS_HDFS_DIR.rstrip("/"), filename="all_trial_results.parquet")
    hdfs.write_dataframe_parquet(run_hist, TUNE_RESULTS_HDFS_DIR.rstrip("/"), filename="run_history.parquet")
    print(json.dumps({"run_tag": run_tag, "local_dir": str(local_dir), "metrics_output": args.metrics_output}, indent=2))


if __name__ == "__main__":
    main()
