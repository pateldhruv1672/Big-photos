from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.common import hdfs

BASIC_ROOT = os.getenv("MIRFLICKR25K_BASIC_ROOT", "/photos/metadata/mirflickr25k/basic")
MODEL_OUTPUT = os.getenv("MODEL_HDFS_PATH", "/photos/models/image_classifier/mobilenet_tuned.pt")
METRICS_OUTPUT = os.getenv("MODEL_METRICS_HDFS_PATH", "/photos/models/image_classifier/mobilenet_tuned_metrics.json")
TUNE_RESULTS_HDFS_DIR = os.getenv("TUNE_RESULTS_HDFS_DIR", "/photos/models/image_classifier/tuning_transfer")
TUNE_RESULTS_LOCAL_DIR = os.getenv("TUNE_RESULTS_LOCAL_DIR", "/app/outputs/training")
RAY_ADDRESS = os.getenv("RAY_ADDRESS", "local")
LOCAL_CACHE_DIR = os.getenv("LOCAL_IMAGE_CACHE_DIR", "/tmp/mirflickr_cache")
LOCAL_MIRFLICKR_DIR = os.getenv("LOCAL_MIRFLICKR_DIR", "/app/shared_data/mirflickr")
LOCAL_ANNOTATION_DIR = os.getenv("LOCAL_ANNOTATION_DIR", "/app/shared_data/mirflickr25k_annotations_v080")
logger = logging.getLogger("tune_transfer_mobilenet")


def extract_label(row: Dict[str, Any]) -> str:
    labels = row.get("labels")
    if isinstance(labels, list) and labels:
        return str(labels[0]).strip().lower()
    if hasattr(labels, "tolist"):
        arr = labels.tolist()
        if isinstance(arr, list) and arr:
            return str(arr[0]).strip().lower()
    if isinstance(labels, str) and labels.strip():
        return labels.strip().lower()
    return "other"


def prepare_samples(input_root: str, max_rows: int = 0) -> Tuple[List[Tuple[str, str]], List[str]]:
    df = hdfs.read_parquet_dataset(input_root)
    if df.empty:
        raise SystemExit(f"No metadata rows found at {input_root}")
    records = df.to_dict("records")
    samples: List[Tuple[str, str]] = []
    for r in records:
        image_uri = str(r.get("image_uri") or "").strip()
        if not image_uri:
            continue
        label = extract_label(r)
        samples.append((image_uri, label))
        if max_rows and len(samples) >= max_rows:
            break
    classes = sorted({y for _, y in samples})
    return samples, classes


def prepare_samples_from_local(annotation_root: str, local_image_dir: str, max_rows: int = 0) -> Tuple[List[Tuple[str, str]], List[str]]:
    df = hdfs.read_parquet_dataset(annotation_root)
    if df.empty:
        raise SystemExit(f"No metadata rows found at {annotation_root}")
    local_dir = Path(local_image_dir)
    if not local_dir.exists():
        raise SystemExit(f"Local MIRFLICKR dir not found: {local_image_dir}")
    samples: List[Tuple[str, str]] = []
    for r in df.to_dict("records"):
        image_id = str(r.get("image_id") or "").strip()
        if not image_id:
            continue
        local_path = local_dir / f"im{int(image_id)}.jpg"
        if not local_path.exists():
            continue
        label = extract_label(r)
        samples.append((str(local_path), label))
        if max_rows and len(samples) >= max_rows:
            break
    classes = sorted({y for _, y in samples})
    return samples, classes


def prepare_samples_from_local_annotations(
    annotation_dir: str,
    local_image_dir: str,
    allowed_labels: List[str],
    add_other: bool = True,
    max_rows: int = 0,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    ann_dir = Path(annotation_dir)
    img_dir = Path(local_image_dir)
    if not ann_dir.exists():
        raise SystemExit(f"Annotation dir not found: {annotation_dir}")
    if not img_dir.exists():
        raise SystemExit(f"Image dir not found: {local_image_dir}")
    files = sorted([p for p in ann_dir.glob("*.txt") if p.is_file() and p.name.lower() != "readme.txt"])
    allowed = {x.strip().lower() for x in allowed_labels if x.strip()}
    labels_by_id: Dict[str, set] = {}
    for f in files:
        label = re.sub(r"_r\d+$", "", f.stem.lower())
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if not s.isdigit():
                    continue
                iid = str(int(s))
                labels_by_id.setdefault(iid, set()).add(label)
    samples: List[Tuple[str, str]] = []
    for iid, labs in labels_by_id.items():
        picked = [x for x in labs if x in allowed]
        label = picked[0] if picked else ("other" if add_other else None)
        if not label:
            continue
        p = img_dir / f"im{int(iid)}.jpg"
        if not p.exists():
            continue
        samples.append((str(p), label))
        if max_rows and len(samples) >= max_rows:
            break
    classes = sorted({y for _, y in samples})
    return samples, classes


def cache_images(samples: List[Tuple[str, str]], cache_dir: Path) -> List[Tuple[str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: List[Tuple[str, str]] = []
    for uri, label in samples:
        hdfs_path = uri
        # deterministic local filename by stripping path separators
        local_name = hdfs.normalize_path(hdfs_path).strip("/").replace("/", "__")
        local_path = cache_dir / local_name
        if not local_path.exists():
            try:
                hdfs.download_to_local(hdfs_path, str(local_path))
            except Exception:
                continue
        out.append((str(local_path), label))
    return out


class LocalImageDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, int]], tfm: transforms.Compose):
        self.rows = rows
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, y = self.rows[idx]
        with Image.open(path) as im:
            x = self.tfm(im.convert("RGB"))
        return x, y


def run_single_trial(config: Dict[str, Any], train_rows: List[Tuple[str, int]], val_rows: List[Tuple[str, int]], num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = LocalImageDataset(train_rows, tfm)
    val_ds = LocalImageDataset(val_rows, tfm)
    loader_train = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0)
    loader_val = DataLoader(val_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(float(config["dropout"])),
        nn.Linear(in_features, num_classes),
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if config["optimizer"] == "sgd":
        opt = torch.optim.SGD(params, lr=float(config["lr"]), momentum=0.9, weight_decay=float(config["weight_decay"]))
    else:
        opt = torch.optim.AdamW(params, lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()

    epochs = int(config["epochs"])
    epoch_metrics: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        ys, ps = [], []
        val_losses = []
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_losses.append(float(loss_fn(logits, yb).item()))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                ys.extend(yb.numpy().tolist())
                ps.extend(pred.tolist())
        acc = float(accuracy_score(ys, ps))
        f1 = float(f1_score(ys, ps, average="weighted", zero_division=0))
        epoch_metrics.append(
            {
                "epoch": epoch,
                "accuracy": acc,
                "f1_weighted": f1,
                "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            }
        )
    final = epoch_metrics[-1] if epoch_metrics else {"accuracy": 0.0, "f1_weighted": 0.0}
    return {"accuracy": float(final["accuracy"]), "f1_weighted": float(final["f1_weighted"]), "epoch_metrics": epoch_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=BASIC_ROOT)
    parser.add_argument("--model-output", default=MODEL_OUTPUT)
    parser.add_argument("--metrics-output", default=METRICS_OUTPUT)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--local-annotation-dir", default=LOCAL_ANNOTATION_DIR)
    parser.add_argument("--use-local-annotation", action="store_true")
    args = parser.parse_args()

    log_level = os.getenv("TRAIN_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if RAY_ADDRESS == "local":
        # local_mode keeps execution in-process to avoid worker overhead/OOM on small machines.
        ray.init(ignore_reinit_error=True, local_mode=True, include_dashboard=False, num_cpus=1)
    else:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)

    if args.use_local_annotation:
        samples, classes = prepare_samples_from_local_annotations(
            args.local_annotation_dir,
            LOCAL_MIRFLICKR_DIR,
            allowed_labels=["structures", "indoor", "plant_life", "people", "sky", "animals", "water", "food", "transport", "female"],
            add_other=True,
            max_rows=args.max_rows,
        )
        cached = samples
    elif Path(LOCAL_MIRFLICKR_DIR).exists():
        samples, classes = prepare_samples_from_local(args.input, LOCAL_MIRFLICKR_DIR, max_rows=args.max_rows)
        cached = samples
    else:
        samples, classes = prepare_samples(args.input, max_rows=args.max_rows)
        cache_dir = Path(LOCAL_CACHE_DIR)
        cached = cache_images(samples, cache_dir)
    if not cached:
        raise SystemExit("No local images available for training")
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
    rows_idx = [(p, class_to_idx[y]) for p, y in cached if y in class_to_idx]
    y = np.asarray([r[1] for r in rows_idx])
    idxs = np.arange(len(rows_idx))
    stratify = y if min(pd.Series(y).value_counts()) >= 2 else None
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42, stratify=stratify)
    train_rows = [rows_idx[i] for i in train_idx]
    val_rows = [rows_idx[i] for i in val_idx]

    search_space = {
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "batch_size": [8, 12, 16],
        "dropout": [0.2, 0.3, 0.4],
        "optimizer": ["adamw", "sgd"],
        "epochs": [3, 4],
    }
    candidates: List[Dict[str, Any]] = []
    for lr in search_space["lr"]:
        for wd in search_space["weight_decay"]:
            for bs in search_space["batch_size"]:
                for d in search_space["dropout"]:
                    for opt in search_space["optimizer"]:
                        for e in search_space["epochs"]:
                            candidates.append(
                                {
                                    "lr": lr,
                                    "weight_decay": wd,
                                    "batch_size": bs,
                                    "dropout": d,
                                    "optimizer": opt,
                                    "epochs": e,
                                }
                            )
    random.seed(42)
    random.shuffle(candidates)
    selected = candidates[: int(args.trials)]

    trial_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(selected, start=1):
        out = run_single_trial(cfg, train_rows, val_rows, len(classes))
        trial_rows.append(
            {
                "trial_id": idx,
                "accuracy": float(out.get("accuracy", 0.0)),
                "f1_weighted": float(out.get("f1_weighted", 0.0)),
                **cfg,
            }
        )
        for row in out.get("epoch_metrics", []):
            epoch_rows.append({"trial_id": idx, **row, **cfg})
        logger.info(
            "Trial %d/%d cfg=%s -> acc=%.4f f1=%.4f",
            idx,
            len(selected),
            cfg,
            float(out.get("accuracy", 0.0)),
            float(out.get("f1_weighted", 0.0)),
        )

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_dir = Path(TUNE_RESULTS_LOCAL_DIR) / run_tag
    local_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(local_dir / "training.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.info("Run started: %s", run_tag)
    logger.info("Samples: %d, classes: %s", len(rows_idx), sorted(classes))

    for row in trial_rows:
        row["run_tag"] = run_tag
    for row in epoch_rows:
        row["run_tag"] = run_tag

    trials_df = pd.DataFrame(trial_rows).sort_values("f1_weighted", ascending=False).reset_index(drop=True)
    epoch_df = pd.DataFrame(epoch_rows)
    best_row = trials_df.iloc[0].to_dict()
    trials_df.to_csv(local_dir / "trial_results.csv", index=False)
    if not epoch_df.empty:
        epoch_df.to_csv(local_dir / "trial_epoch_logs.csv", index=False)

    summary = {
        "run_tag": run_tag,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows_idx),
        "classes": sorted(classes),
        "num_trials": int(args.trials),
        "best_config": {
            "lr": best_row["lr"],
            "weight_decay": best_row["weight_decay"],
            "batch_size": int(best_row["batch_size"]),
            "dropout": float(best_row["dropout"]),
            "optimizer": best_row["optimizer"],
            "epochs": int(best_row["epochs"]),
        },
        "best_accuracy": float(best_row["accuracy"]),
        "best_f1_weighted": float(best_row["f1_weighted"]),
    }
    (local_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Best config: %s", json.dumps(summary["best_config"]))
    logger.info("Best metrics: accuracy=%.4f f1=%.4f", summary["best_accuracy"], summary["best_f1_weighted"])

    # Store run histories
    all_trials_path = Path(TUNE_RESULTS_LOCAL_DIR) / "all_trial_results_transfer.csv"
    all_trials = pd.concat([pd.read_csv(all_trials_path), trials_df], ignore_index=True, sort=False) if all_trials_path.exists() else trials_df
    all_trials.to_csv(all_trials_path, index=False)
    hist_row = {"run_tag": run_tag, "trained_at": summary["trained_at"], "rows": summary["rows"], "best_accuracy": summary["best_accuracy"], "best_f1_weighted": summary["best_f1_weighted"], "num_trials": summary["num_trials"], **{f"best_{k}": v for k, v in summary["best_config"].items()}}
    hist_path = Path(TUNE_RESULTS_LOCAL_DIR) / "run_history_transfer.csv"
    run_hist = pd.concat([pd.read_csv(hist_path), pd.DataFrame([hist_row])], ignore_index=True, sort=False) if hist_path.exists() else pd.DataFrame([hist_row])
    run_hist.to_csv(hist_path, index=False)

    # Save HDFS artifacts
    try:
        hdfs.write_dataframe_parquet(trials_df, f"{TUNE_RESULTS_HDFS_DIR.rstrip('/')}/{run_tag}", filename="trial_results.parquet")
        if not epoch_df.empty:
            hdfs.write_dataframe_parquet(epoch_df, f"{TUNE_RESULTS_HDFS_DIR.rstrip('/')}/{run_tag}", filename="trial_epoch_logs.parquet")
        hdfs.write_json(f"{TUNE_RESULTS_HDFS_DIR.rstrip('/')}/{run_tag}/summary.json", summary, overwrite=True)
        hdfs.write_dataframe_parquet(all_trials, TUNE_RESULTS_HDFS_DIR.rstrip("/"), filename="all_trial_results.parquet")
        hdfs.write_dataframe_parquet(run_hist, TUNE_RESULTS_HDFS_DIR.rstrip("/"), filename="run_history.parquet")
    except Exception as exc:
        logger.warning("HDFS artifact write skipped due to connectivity issue: %s", exc)

    # Save lightweight model card for compatibility output path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(
            json.dumps({"type": "mobilenet_transfer", "classes": sorted(classes), "best_config": summary["best_config"]}, indent=2).encode("utf-8")
        )
        model_card = tmp.name
    try:
        hdfs.upload_local_file(model_card, args.model_output, overwrite=True)
        hdfs.write_json(args.metrics_output, summary, overwrite=True)
        hdfs.upload_local_file(str(local_dir / "training.log"), f"{TUNE_RESULTS_HDFS_DIR.rstrip('/')}/{run_tag}/training.log", overwrite=True)
    except Exception as exc:
        logger.warning("HDFS model/metrics write skipped due to connectivity issue: %s", exc)
    logger.info("Artifacts written to local_dir=%s and hdfs=%s/%s", str(local_dir), TUNE_RESULTS_HDFS_DIR.rstrip("/"), run_tag)
    print(json.dumps({"run_tag": run_tag, "local_dir": str(local_dir), "best_f1_weighted": summary["best_f1_weighted"]}, indent=2))


if __name__ == "__main__":
    main()
