from __future__ import annotations

import argparse
import json
import logging
import os
import random
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
from ray_jobs.tune_transfer_mobilenet import prepare_samples_from_local_annotations

logger = logging.getLogger("compare_backbones")

LOCAL_MIRFLICKR_DIR = os.getenv("LOCAL_MIRFLICKR_DIR", "/app/shared_data/mirflickr")
LOCAL_ANNOTATION_DIR = os.getenv("LOCAL_ANNOTATION_DIR", "/app/shared_data/mirflickr25k_annotations_v080")
TUNE_RESULTS_LOCAL_DIR = os.getenv("TUNE_RESULTS_LOCAL_DIR", "/app/outputs/training")
COMPARE_RESULTS_HDFS_DIR = os.getenv("COMPARE_RESULTS_HDFS_DIR", "/photos/models/image_classifier/backbone_compare")
RAY_ADDRESS = os.getenv("RAY_ADDRESS", "local")

DEFAULT_LABELS = [
    "structures",
    "indoor",
    "plant_life",
    "people",
    "sky",
    "animals",
    "water",
    "food",
    "transport",
    "female",
]


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(backbone: str, num_classes: int, dropout: float) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for p in model.features.parameters():
            p.requires_grad = False
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    if backbone == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        for p in model.fc.parameters():
            p.requires_grad = True
        # Keep training simple and deterministic by not optimizing aux branch.
        model.AuxLogits = None
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")


def _transforms_for(backbone: str) -> transforms.Compose:
    size = 299 if backbone == "inception_v3" else 224
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _train_eval(
    backbone: str,
    cfg: Dict[str, Any],
    train_rows: List[Tuple[str, int]],
    val_rows: List[Tuple[str, int]],
    num_classes: int,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = _transforms_for(backbone)
    train_dl = DataLoader(LocalImageDataset(train_rows, tfm), batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    val_dl = DataLoader(LocalImageDataset(val_rows, tfm), batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    model = _build_model(backbone, num_classes, float(cfg["dropout"])).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "sgd":
        opt = torch.optim.SGD(params, lr=float(cfg["lr"]), momentum=0.9, weight_decay=float(cfg["weight_decay"]))
    else:
        opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(int(cfg["epochs"])):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            logits = model(xb.to(device))
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            ys.extend(yb.numpy().tolist())
            ps.extend(pred.tolist())
    return {
        "accuracy": float(accuracy_score(ys, ps)),
        "f1_weighted": float(f1_score(ys, ps, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(ys, ps, average="macro", zero_division=0)),
    }


def _configs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lr in [3e-4, 1e-3]:
        for wd in [1e-5, 1e-4]:
            for bs in [8, 12]:
                for dr in [0.2, 0.3]:
                    for opt in ["adamw", "sgd"]:
                        for ep in [3, 4]:
                            out.append(
                                {
                                    "lr": lr,
                                    "weight_decay": wd,
                                    "batch_size": bs,
                                    "dropout": dr,
                                    "optimizer": opt,
                                    "epochs": ep,
                                }
                            )
    random.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-per-backbone", type=int, default=6)
    parser.add_argument("--max-rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed-labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--backbones", default="mobilenet_v3_small")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    _seed_everything(args.seed)
    if RAY_ADDRESS == "local":
        ray.init(ignore_reinit_error=True, local_mode=True, include_dashboard=False, num_cpus=1)
    else:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)

    allowed = [x.strip() for x in str(args.allowed_labels).split(",") if x.strip()]
    samples, classes = prepare_samples_from_local_annotations(
        LOCAL_ANNOTATION_DIR,
        LOCAL_MIRFLICKR_DIR,
        allowed_labels=allowed,
        add_other=True,
        max_rows=args.max_rows,
    )
    if not samples:
        raise SystemExit("No local samples found for comparison")

    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
    rows_idx = [(p, class_to_idx[y]) for p, y in samples if y in class_to_idx]
    y = np.asarray([r[1] for r in rows_idx])
    idxs = np.arange(len(rows_idx))
    stratify = y if min(pd.Series(y).value_counts()) >= 2 else None
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=args.seed, stratify=stratify)
    train_rows = [rows_idx[i] for i in train_idx]
    val_rows = [rows_idx[i] for i in val_idx]

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_dir = Path(TUNE_RESULTS_LOCAL_DIR) / f"compare_{run_tag}"
    local_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    selected_backbones = [x.strip() for x in str(args.backbones).split(",") if x.strip()]
    if not selected_backbones:
        raise SystemExit("No backbones selected")
    unsupported = [x for x in selected_backbones if x not in {"mobilenet_v3_small", "inception_v3"}]
    if unsupported:
        raise SystemExit(f"Unsupported backbones requested: {unsupported}")
    cfgs = _configs()[: max(1, int(args.trials_per_backbone))]
    for backbone in selected_backbones:
        for trial_id, cfg in enumerate(cfgs, start=1):
            metrics = _train_eval(backbone, cfg, train_rows, val_rows, len(classes))
            row = {"backbone": backbone, "trial_id": trial_id, **cfg, **metrics, "run_tag": run_tag}
            results.append(row)
            logger.info(
                "%s trial=%d acc=%.4f macro_f1=%.4f weighted_f1=%.4f cfg=%s",
                backbone,
                trial_id,
                metrics["accuracy"],
                metrics["f1_macro"],
                metrics["f1_weighted"],
                cfg,
            )

    df = pd.DataFrame(results).sort_values(["backbone", "f1_macro"], ascending=[True, False]).reset_index(drop=True)
    by_backbone = (
        df.groupby("backbone")[["accuracy", "f1_macro", "f1_weighted"]]
        .max()
        .reset_index()
        .sort_values("f1_macro", ascending=False)
        .to_dict("records")
    )
    summary = {
        "run_tag": run_tag,
        "rows": len(rows_idx),
        "classes": sorted(classes),
        "trials_per_backbone": int(args.trials_per_backbone),
        "best_by_backbone": by_backbone,
        "winner_by_macro_f1": by_backbone[0]["backbone"] if by_backbone else None,
    }

    df.to_csv(local_dir / "backbone_trial_results.csv", index=False)
    (local_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        hdfs.write_dataframe_parquet(df, f"{COMPARE_RESULTS_HDFS_DIR.rstrip('/')}/{run_tag}", filename="backbone_trial_results.parquet")
        hdfs.write_json(f"{COMPARE_RESULTS_HDFS_DIR.rstrip('/')}/{run_tag}/summary.json", summary, overwrite=True)
    except Exception as exc:
        logger.warning("Could not write comparison artifacts to HDFS: %s", exc)

    print(json.dumps({"run_tag": run_tag, "winner": summary["winner_by_macro_f1"], "local_dir": str(local_dir)}, indent=2))


if __name__ == "__main__":
    main()
